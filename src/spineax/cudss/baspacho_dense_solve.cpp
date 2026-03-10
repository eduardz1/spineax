/*
 * BaSpaCho dense LU solver for CUDA via XLA FFI.
 *
 * Phase 2b.6: GPU-resident FFI handler. All format conversion, permutation,
 * and solve operations happen on-device via CUDA kernels. No D2H/H2D transfers
 * in the Execute hot path, enabling XLA command buffer (CUDA graph) capture.
 *
 * This file is compiled by the host C++ compiler (not nvcc). CUDA kernels
 * are in baspacho_dense_kernels.cu (compiled by nvcc) and called via the
 * launcher wrappers declared in baspacho_dense_kernels.h.
 */

#include <cstdint>
#include <memory>
#include <vector>
#include <cstring>
#include <algorithm>
#include <set>

// CUDA runtime MUST come before XLA FFI so PlatformStream<cudaStream_t>
// specialization is available when ffi.h is parsed.
#include "cuda_runtime_api.h"

#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"

// BaSpaCho headers (require Eigen — only the host compiler can handle these)
#include "baspacho/baspacho/Solver.h"
#include "baspacho/baspacho/SparseStructure.h"
#include "baspacho/baspacho/CudaDefs.h"
#include "baspacho/baspacho/Utils.h"

// CUDA kernel launcher wrappers (compiled by nvcc in baspacho_dense_kernels.cu)
#include "baspacho_dense_kernels.h"

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
  ::DevMirror<scalar_t> devData;  // factor data on GPU
  ::DevMirror<scalar_t> devVec;   // RHS/solution vector on GPU

  // LU pivots (host-side, uploaded/downloaded by BaSpaCho internally)
  std::vector<int64_t> pivots;

  // Device-side accessor metadata (from deviceAccessor(), set once at init)
  // These are device pointers into CudaSymbolicCtx's DevMirror buffers.
  const int64_t* devSpanStart = nullptr;
  const int64_t* devSpanToLump = nullptr;
  const int64_t* devLumpStart = nullptr;
  const int64_t* devSpanOffsetInLump = nullptr;
  const int64_t* devChainColPtr = nullptr;
  const int64_t* devChainRowSpan = nullptr;
  const int64_t* devChainData = nullptr;
  const int64_t* devPermutation = nullptr;
  // Upper triangle (for MTYPE_GENERAL):
  const int64_t* devUpperChainRowPtr = nullptr;
  const int64_t* devUpperChainColSpan = nullptr;
  const int64_t* devUpperChainData = nullptr;
  int64_t upperDataBase = 0;  // offset to upper triangle in data buffer
  int64_t totalDataSize_ = 0;

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

  // ---- 3. Allocate host buffers ----
  state->totalDataSize_ = state->solver->totalDataSize();
  state->pivots.resize(n);  // numSpans() == n for 1x1 blocks

  // ---- 4. Pre-allocate GPU buffers (grow-only, no realloc after this) ----
  state->devData.resizeToAtLeast(state->totalDataSize_);
  state->devVec.resizeToAtLeast(n);

  // ---- 5. Extract device-side accessor metadata ----
  // deviceAccessor() returns a PermutedCoalescedAccessor with device pointers
  // into CudaSymbolicCtx's DevMirror buffers (uploaded once at solver creation).
  auto devAcc = state->solver->deviceAccessor();
  state->devSpanStart = devAcc.plainAcc.spanStart;
  state->devSpanToLump = devAcc.plainAcc.spanToLump;
  state->devLumpStart = devAcc.plainAcc.lumpStart;
  state->devSpanOffsetInLump = devAcc.plainAcc.spanOffsetInLump;
  state->devChainColPtr = devAcc.plainAcc.chainColPtr;
  state->devChainRowSpan = devAcc.plainAcc.chainRowSpan;
  state->devChainData = devAcc.plainAcc.chainData;
  state->devPermutation = devAcc.permutation;

  // Upper triangle metadata (for MTYPE_GENERAL)
  state->devUpperChainRowPtr = devAcc.plainAcc.upperChainRowPtr;
  state->devUpperChainColSpan = devAcc.plainAcc.upperChainColSpan;
  state->devUpperChainData = devAcc.plainAcc.upperChainData;
  state->upperDataBase = state->solver->dataSize();

  return state;
}

// ============================================================================
// Execute: GPU-resident hot path — factorize J and solve J*x = f
//
// All operations are kernel launches or cuBLAS calls on the XLA stream.
// No D2H copies, no cudaStreamSynchronize, no host memory access.
// XLA can capture this entire sequence in a CUDA command buffer.
//
// Note: factorLU/solveLU may still have internal pivot D2H/H2D roundtrips
// inside BaSpaCho. These are small (~n * 8 bytes) and a follow-up optimization.
// ============================================================================

template <ffi::DataType T>
static ffi::Error BaspachoGpuExecute(
    cudaStream_t stream,
    BaspachoGpuState<T>* state,
    ffi::Buffer<T> J_buf,                // n×n Jacobian (device, row-major)
    ffi::Buffer<T> f_buf,                // n×1 RHS (device)
    ffi::ResultBuffer<T> x_out           // n×1 solution (device)
) {
  using scalar_t = typename BaspachoGpuState<T>::scalar_t;
  auto& s = *state;
  int64_t n = s.n;
  cudaStream_t str = stream;

  const scalar_t* J_dev = reinterpret_cast<const scalar_t*>(J_buf.typed_data());
  const scalar_t* f_dev = reinterpret_cast<const scalar_t*>(f_buf.typed_data());
  scalar_t* x_dev = reinterpret_cast<scalar_t*>(x_out->typed_data());

  // 1. Zero devData, then scatter J into BaSpaCho's coalesced format (GPU kernel)
  cuCHECK(cudaMemsetAsync(s.devData.ptr, 0,
                           s.totalDataSize_ * sizeof(scalar_t), str));
  launchDenseToCoalesced<scalar_t>(
      str, J_dev, s.devData.ptr, n,
      s.devSpanStart, s.devSpanToLump, s.devLumpStart,
      s.devSpanOffsetInLump, s.devChainColPtr,
      s.devChainRowSpan, s.devChainData, s.devPermutation,
      s.devUpperChainRowPtr, s.devUpperChainColSpan,
      s.devUpperChainData, s.upperDataBase);

  // 2. LU factorization on GPU
  s.solver->setStream(static_cast<void*>(str));
  s.solver->factorLU(s.devData.ptr, s.pivots.data());

  // 3. Permute RHS on GPU: devVec[perm[i]] = f[i]
  launchPermuteForward<scalar_t>(str, f_dev, s.devVec.ptr, s.devPermutation, n);

  // 4. Solve in-place on GPU: L*U*x = P*f
  s.solver->solveLU(s.devData.ptr, s.pivots.data(), s.devVec.ptr, n, 1);

  // 5. Unpermute solution directly to output: x_out[i] = devVec[perm[i]]
  launchPermuteInverse<scalar_t>(str, s.devVec.ptr, x_dev, s.devPermutation, n);

  return ffi::Error::Success();
}

// ============================================================================
// FFI handler definitions
// ============================================================================

// NOTE: kCmdBufferCompatible is NOT set yet. The Execute handler currently
// calls cudaMalloc (via DevMirror::resizeToAtLeast) which is illegal during
// CUDA graph capture. Phase 2b.6 will pre-allocate all buffers at Instantiate
// time and add GPU kernels to replace host-side operations, enabling the trait.
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
