/*
 * BaSpaCho dense LU solver for CUDA via XLA FFI.
 *
 * Phase 2a: Non-graph-compatible. Uses BaSpaCho's supernodal LU with CUDA
 * backend, with workspace pre-allocation to eliminate per-call cudaMalloc.
 *
 * For small dense matrices (e.g., ring's 47x47), BaSpaCho creates a single
 * supernode and delegates to cuSOLVER getrf. The benefit over raw
 * jax.scipy.linalg.solve is:
 *   1. Symbolic analysis done once (cached in state)
 *   2. DevMirror grow-only GPU memory (no per-call alloc after warmup)
 *   3. Foundation for Phase 2b graph-compatibility modifications
 *
 * Phase 2b (future): Replace cuSOLVER getrf in BaSpaCho's dense loop with
 * custom LU kernel + eliminate all D2H syncs to enable CUDA graph capture.
 */

#include <cstdint>
#include <memory>
#include <vector>
#include <cstring>
#include <algorithm>
#include <set>

#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"

// BaSpaCho headers
#include "baspacho/baspacho/Solver.h"
#include "baspacho/baspacho/SparseStructure.h"
#include "baspacho/baspacho/CudaDefs.h"

namespace ffi = xla::ffi;
namespace nb = nanobind;

// Helper: map FFI data type tag to native C++ type
template <ffi::DataType T>
struct get_native_data_type;
template <>
struct get_native_data_type<ffi::F32> {
  using type = float;
};
template <>
struct get_native_data_type<ffi::F64> {
  using type = double;
};

// ============================================================================
// State: persists across calls via XLA FFI state mechanism
// ============================================================================

template <ffi::DataType T>
struct BaspachoGpuState {
  static xla::ffi::TypeId id;
  using scalar_t = typename get_native_data_type<T>::type;

  BaSpaCho::SolverPtr solver;
  int64_t n = 0;

  // Persistent GPU buffers (DevMirror grow-only: allocates once, reuses)
  BaSpaCho::DevMirror<scalar_t> devData;  // factor data on GPU
  BaSpaCho::DevMirror<scalar_t> devVec;   // RHS/solution vector on GPU

  // Host staging buffers (allocated once at init)
  std::vector<scalar_t> hostData;  // loadFromCsr target (totalDataSize)
  std::vector<scalar_t> hostVec;   // permuted RHS / unpermuted solution
  std::vector<int64_t> pivots;     // LU pivots (host, uploaded by BaSpaCho)

  // Static dense CSR indices (built once, reused every call)
  std::vector<int64_t> csrRowPtr;   // [0, n, 2n, ..., n*n]
  std::vector<int64_t> csrColInds;  // [0..n-1] repeated n times
  std::vector<int64_t> blockSizes;  // [1, 1, ..., 1]

  ~BaspachoGpuState() = default;
};

template <>
ffi::TypeId BaspachoGpuState<ffi::F32>::id = {};
template <>
ffi::TypeId BaspachoGpuState<ffi::F64>::id = {};

// ============================================================================
// Instantiate: called once by XLA, creates solver + pre-allocates everything
// ============================================================================

template <ffi::DataType T>
static ffi::ErrorOr<std::unique_ptr<BaspachoGpuState<T>>> BaspachoGpuInstantiate(
    const int64_t n) {
  using scalar_t = typename BaspachoGpuState<T>::scalar_t;
  auto state = std::make_unique<BaspachoGpuState<T>>();
  state->n = n;

  // ---- 1. Build dense lower-triangular sparsity pattern ----
  // BaSpaCho expects lower-triangular SparseStructure (CSR with columns <= row).
  std::vector<int64_t> ptrs(n + 1);
  std::vector<int64_t> inds;
  ptrs[0] = 0;
  for (int64_t i = 0; i < n; ++i) {
    for (int64_t j = 0; j <= i; ++j) {
      inds.push_back(j);
    }
    ptrs[i + 1] = static_cast<int64_t>(inds.size());
  }
  BaSpaCho::SparseStructure ss(std::move(ptrs), std::move(inds));

  // ---- 2. Create solver ----
  BaSpaCho::Settings settings;
  settings.backend = BaSpaCho::BackendCuda;
  settings.matrixType = BaSpaCho::MTYPE_GENERAL;
  settings.findSparseEliminationRanges = false;  // fully dense, no sparse elim
  settings.addFillPolicy = BaSpaCho::AddFillComplete;
  settings.numThreads = 1;              // GPU factorization
  settings.staticPivotThreshold = 0.0;  // auto threshold

  std::vector<int64_t> paramSizes(n, 1);  // all 1x1 blocks
  state->solver = BaSpaCho::createSolver(settings, paramSizes, ss);

  // ---- 3. Build dense CSR indices for loadFromCsr ----
  // Dense n×n matrix in CSR: each row has n entries (columns 0..n-1).
  // Row-major J layout maps directly to these CSR values.
  state->csrRowPtr.resize(n + 1);
  state->csrColInds.resize(n * n);
  state->blockSizes.resize(n, 1);
  for (int64_t i = 0; i <= n; ++i) {
    state->csrRowPtr[i] = i * n;
  }
  for (int64_t i = 0; i < n; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      state->csrColInds[i * n + j] = j;
    }
  }

  // ---- 4. Allocate host buffers ----
  int64_t totalData = state->solver->totalDataSize();
  state->hostData.resize(totalData);
  state->hostVec.resize(n);
  state->pivots.resize(n);  // numSpans() == n for 1x1 blocks

  // ---- 5. Pre-allocate GPU buffers (grow-only, no realloc after this) ----
  state->devData.resizeToAtLeast(totalData);
  state->devVec.resizeToAtLeast(n);

  return state;
}

// ============================================================================
// Execute: hot path — factorize J and solve J*x = f
// ============================================================================

template <ffi::DataType T>
static ffi::Error BaspachoGpuExecute(
    ffi::PlatformStream<cudaStream_t> stream,
    BaspachoGpuState<T>* state,
    ffi::Buffer<T> J_buf,                // n×n Jacobian (device, row-major)
    ffi::Buffer<T> f_buf,                // n×1 RHS (device)
    ffi::ResultBuffer<T> x_out           // n×1 solution (device)
) {
  using scalar_t = typename BaspachoGpuState<T>::scalar_t;
  auto& s = *state;
  int64_t n = s.n;

  const scalar_t* J_dev = reinterpret_cast<const scalar_t*>(J_buf.typed_data());
  const scalar_t* f_dev = reinterpret_cast<const scalar_t*>(f_buf.typed_data());
  scalar_t* x_dev = reinterpret_cast<scalar_t*>(x_out->typed_data());

  // ---- Phase 2a: host-side format conversion ----
  // For small dense matrices (47×47 = 17KB), D2H + format + H2D is negligible.
  // Phase 2b will add a GPU kernel for direct device-to-device format conversion.

  // 1. Download J from device to host staging buffer
  //    (reuse hostData temporarily — it's big enough: totalDataSize >= n*n for dense)
  std::vector<scalar_t> hostJ(n * n);
  cuCHECK(cudaMemcpyAsync(hostJ.data(), J_dev, n * n * sizeof(scalar_t),
                           cudaMemcpyDeviceToHost, stream.value()));
  // Must sync before CPU-side loadFromCsr
  cuCHECK(cudaStreamSynchronize(stream.value()));

  // 2. Zero factor data and load J into BaSpaCho's internal coalesced format
  std::fill(s.hostData.begin(), s.hostData.end(), scalar_t(0));
  s.solver->loadFromCsr(s.csrRowPtr.data(), s.csrColInds.data(),
                         s.blockSizes.data(), hostJ.data(), s.hostData.data());

  // 3. Upload formatted data to GPU (DevMirror reuses existing allocation)
  s.devData.load(s.hostData);

  // ---- GPU factorization ----

  // 4. Set stream so BaSpaCho's cuBLAS/cuSOLVER ops use XLA's stream
  s.solver->setStream(static_cast<void*>(stream.value()));

  // 5. LU factorization on GPU
  s.solver->factorLU(s.devData.ptr, s.pivots.data());

  // ---- GPU solve ----
  const auto& perm = s.solver->paramToSpan();

  // 6. Download f from device, permute for BaSpaCho's internal ordering
  cuCHECK(cudaMemcpy(s.hostVec.data(), f_dev, n * sizeof(scalar_t),
                      cudaMemcpyDeviceToHost));
  std::vector<scalar_t> permVec(n);
  for (int64_t i = 0; i < n; ++i) {
    permVec[perm[i]] = s.hostVec[i];
  }

  // 7. Upload permuted RHS to GPU and solve in-place
  s.devVec.load(permVec);
  s.solver->solveLU(s.devData.ptr, s.pivots.data(), s.devVec.ptr, n, 1);

  // 8. Download solution, apply inverse permutation, upload to output
  std::vector<scalar_t> solVec(n);
  cuCHECK(cudaMemcpy(solVec.data(), s.devVec.ptr, n * sizeof(scalar_t),
                      cudaMemcpyDeviceToHost));
  std::vector<scalar_t> result(n);
  for (int64_t i = 0; i < n; ++i) {
    result[i] = solVec[perm[i]];
  }
  cuCHECK(cudaMemcpyAsync(x_dev, result.data(), n * sizeof(scalar_t),
                           cudaMemcpyHostToDevice, stream.value()));

  return ffi::Error::Success();
}

// ============================================================================
// FFI handler definitions
// ============================================================================

#define DEFINE_BASPACHO_GPU_FFI_HANDLERS(TypeName, DataType)                     \
  XLA_FFI_DEFINE_HANDLER(kBaspachoGpuInstantiate##TypeName,                     \
                         BaspachoGpuInstantiate<DataType>,                      \
                         ffi::Ffi::BindInstantiate().Attr<int64_t>("n"));       \
                                                                                \
  XLA_FFI_DEFINE_HANDLER(kBaspachoGpuExecute##TypeName,                         \
                         BaspachoGpuExecute<DataType>,                          \
                         ffi::Ffi::Bind()                                       \
                             .Ctx<ffi::PlatformStream<cudaStream_t>>()          \
                             .Ctx<ffi::State<BaspachoGpuState<DataType>>>()     \
                             .Arg<ffi::Buffer<DataType>>()                      \
                             .Arg<ffi::Buffer<DataType>>()                      \
                             .Ret<ffi::Buffer<DataType>>());

// Generate handlers for f32 and f64 (no complex for LU dense path)
DEFINE_BASPACHO_GPU_FFI_HANDLERS(f32, ffi::F32);
DEFINE_BASPACHO_GPU_FFI_HANDLERS(f64, ffi::F64);

// ============================================================================
// nanobind module: exports handlers + type info for Python FFI registration
// ============================================================================

#if defined(XLA_FFI_API_MINOR) && (XLA_FFI_API_MINOR >= 2)
#define ADD_GPU_TYPE(d, DTYPE)                                                  \
  do {                                                                          \
    using StateT = BaspachoGpuState<DTYPE>;                                     \
    static auto kStateTypeInfo = xla::ffi::MakeTypeInfo<StateT>();              \
    (d)["type_info"] = nb::capsule(reinterpret_cast<void*>(&kStateTypeInfo));   \
    (d)["type_id"] = nb::capsule(reinterpret_cast<void*>(&StateT::id));         \
  } while (0)
#else
#define ADD_GPU_TYPE(d, DTYPE)                                                  \
  do {                                                                          \
    (d)["state_type"] = nb::dict();                                             \
  } while (0)
#endif

#define EXPORT_BASPACHO_GPU_HANDLERS(m, TypeName, DataType)                     \
  m.def("state_dict_" #TypeName, []() {                                        \
    nb::dict d;                                                                 \
    ADD_GPU_TYPE(d, DataType);                                                  \
    return d;                                                                   \
  });                                                                           \
  m.def("type_id_" #TypeName, []() {                                           \
    return nb::capsule(                                                         \
        reinterpret_cast<void*>(&BaspachoGpuState<DataType>::id));              \
  });                                                                           \
  m.def("handler_" #TypeName, []() {                                           \
    nb::dict d;                                                                 \
    d["instantiate"] = nb::capsule(                                             \
        reinterpret_cast<void*>(kBaspachoGpuInstantiate##TypeName));            \
    d["execute"] = nb::capsule(                                                 \
        reinterpret_cast<void*>(kBaspachoGpuExecute##TypeName));                \
    return d;                                                                   \
  });

NB_MODULE(baspacho_dense_solve, m) {
  EXPORT_BASPACHO_GPU_HANDLERS(m, f32, ffi::F32);
  EXPORT_BASPACHO_GPU_HANDLERS(m, f64, ffi::F64);
}
