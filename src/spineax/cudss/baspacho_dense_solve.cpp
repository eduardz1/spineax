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

  // Persistent contexts (created once at Instantiate, reused across Execute calls).
  // Eliminates per-call cudaMalloc/cudaFreeHost overhead (~43s for ring/500ts).
  BaSpaCho::NumericCtxPtr<scalar_t> numCtx;
  BaSpaCho::SolveCtxPtr<scalar_t> solveCtx;

  // Device-resident pivots (no D2H→H2D roundtrip)
  ::DevMirror<int64_t> devPivots;

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

  // ---- 3. Allocate buffers ----
  state->totalDataSize_ = state->solver->totalDataSize();

  // ---- 4. Pre-allocate GPU buffers (grow-only, no realloc after this) ----
  state->devData.resizeToAtLeast(state->totalDataSize_);
  state->devVec.resizeToAtLeast(n);
  state->devPivots.resizeToAtLeast(n);

  // ---- 5. Create persistent contexts (one-time allocation) ----
  // NumericCtx and SolveCtx are reused across Execute calls via reset().
  // This eliminates the ~43s cudaFreeHost + ~1s cudaMalloc overhead from
  // creating/destroying contexts on every factorLU/solveLU call.
  {
    auto& symCtx = state->solver->internalSymbolicContext();
    // maxElimTempSize for the solver (accessed via skel)
    // For fully dense (no sparse elim), this is 0.
    state->numCtx = symCtx.createNumericCtx<scalar_t>(0, static_cast<scalar_t*>(nullptr));

    // For 1×1 blocks, maxDenseBlockSize = 1, totalDensePivots = n
    state->numCtx->preAllocateForLU(1, n);

    state->solveCtx = symCtx.createSolveCtx<scalar_t>(1, static_cast<scalar_t*>(nullptr));
  }

  // ---- 6. Extract device-side accessor metadata ----
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
// Uses persistent NumericCtx/SolveCtx (created once at Instantiate, reset per call)
// and device-resident pivots (D→D instead of D→H→D roundtrip).
// This eliminates ~55s of CUDA API overhead per session (ring/500ts):
//   - ~43s from cudaFreeHost/cudaHostAlloc (persistent NumericCtx)
//   - ~5s from pivot D→H→D roundtrip (device-resident pivots)
//   - ~5s from readValue bulk D→H (GPU maxAbsDiag kernel)
//   - ~1s from SolveCtx buffer alloc/free (persistent SolveCtx)
//
// Remaining per-lump H→D: prepareAssemble (~376 bytes via pinned memory).
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

  // 2. LU factorization on GPU (persistent context + device-resident pivots)
  s.solver->setStream(static_cast<void*>(str));
  s.solver->factorLU(s.devData.ptr, s.devPivots.ptr, *s.numCtx,
                     BaSpaCho::PivotLocation::Device);

  // 3. Permute RHS on GPU: devVec[perm[i]] = f[i]
  launchPermuteForward<scalar_t>(str, f_dev, s.devVec.ptr, s.devPermutation, n);

  // 4. Solve in-place on GPU: L*U*x = P*f (persistent context + device pivots)
  s.solver->solveLU(s.devData.ptr, s.devPivots.ptr, s.devVec.ptr, n, 1,
                    *s.solveCtx, BaSpaCho::PivotLocation::Device);

  // 5. Unpermute solution directly to output: x_out[i] = devVec[perm[i]]
  launchPermuteInverse<scalar_t>(str, s.devVec.ptr, x_dev, s.devPermutation, n);

  return ffi::Error::Success();
}

// ============================================================================
// FFI handler definitions
// ============================================================================

// NOTE: kCmdBufferCompatible is NOT set yet. While persistent contexts and
// device-resident pivots eliminate most CUDA API overhead, the Execute handler
// still has per-lump H→D copies (prepareAssemble, flushGemmBatch) that use
// pinned memory staging. These are small but involve cudaMemcpyAsync which
// may not be graph-capture compatible. Also, maxAbsDiag does a single D→H copy.
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
