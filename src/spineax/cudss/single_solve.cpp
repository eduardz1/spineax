/*Standard single solve — GPU-resident optimized
 *
 * Changes for GPU-resident solver stack:
 * 1. Async memory handler via cudssDeviceMemHandler (eliminates sync cudaMalloc)
 * 2. Diagnostics moved to cold path only (no D2H per iteration)
 * 3. Stream caching (skip cudssSetStream when unchanged)
 * 4. kCmdBufferCompatible trait on CudssExecute (enables CUDA graph capture)
 */

#include <cstdint>
#include <memory>
#include <vector>
#include <complex>
#include <type_traits>

#include "cuda_runtime_api.h"
#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"
#include "cudss.h"

namespace ffi = xla::ffi;
namespace nb = nanobind;

// verification ================================================================
#define CUDSS_CALL_AND_CHECK(call, status, msg) \
    do { \
        status = call; \
        if (status != CUDSS_STATUS_SUCCESS) { \
            printf("FAILED: CUDSS call ended unsuccessfully with status = %d, details: " #msg "\n", status); \
            return ffi::Error::Success(); \
        } \
    } while(0);

#define CUDA_CHECK(call)                                       \
  do {                                                         \
    cudaError_t err = call;                                    \
    if (err != cudaSuccess) {                                  \
      printf("CUDA Error at %s %d: %s\n", __FILE__, __LINE__,   \
             cudaGetErrorString(err));                         \
      return ffi::Error::Internal("A CUDA call failed.");      \
    }                                                          \
  } while (0)


// Debugging functions =========================================================
template <typename T>
void print_device_data(
    const char* label,
    void* device_ptr,
    size_t n_batch,
    size_t n_elements_per_batch)
{
    // Ensure we have a valid pointer and something to print
    if (!device_ptr || n_batch == 0 || n_elements_per_batch == 0) return;

    std::cout << "\n--- Debug Print: " << label << " ---" << std::endl;

    // Calculate total size and create a host-side vector
    size_t total_elements = n_batch * n_elements_per_batch;
    std::vector<T> host_data(total_elements);

    // Copy all data from GPU to CPU in one go
    cudaMemcpy(
        host_data.data(),
        device_ptr,
        total_elements * sizeof(T),
        cudaMemcpyDeviceToHost
    );

    // Loop through each batch and print its contents
    for (size_t i = 0; i < n_batch; ++i) {
        std::cout << "Batch " << i << ": [";
        size_t batch_start_index = i * n_elements_per_batch;
        for (size_t j = 0; j < n_elements_per_batch; ++j) {
            std::cout << host_data[batch_start_index + j];
            if (j < n_elements_per_batch - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "------------------------------------" << std::endl;
}

// Helper function for data types ==============================================
template <ffi::DataType T> cudaDataType get_cuda_data_type();
template<> cudaDataType get_cuda_data_type<ffi::F32>() { return CUDA_R_32F; }
template<> cudaDataType get_cuda_data_type<ffi::F64>() { return CUDA_R_64F; }
template<> cudaDataType get_cuda_data_type<ffi::C64>() { return CUDA_C_32F; }
template<> cudaDataType get_cuda_data_type<ffi::C128>() { return CUDA_C_64F; }

template <ffi::DataType T>
struct get_native_data_type;
template<> struct get_native_data_type<ffi::F32> { using type = float; };
template<> struct get_native_data_type<ffi::F64> { using type = double; };
template<> struct get_native_data_type<ffi::C64> { using type = std::complex<float>; };
template<> struct get_native_data_type<ffi::C128> { using type = std::complex<double>; };

// Structure definitions =======================================================
template <ffi::DataType T>
struct CudssState {
    static xla::ffi::TypeId id;
    cudssHandle_t handle = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t data = nullptr;
    cudssMatrix_t A = nullptr;
    cudssMatrix_t x = nullptr;
    cudssMatrix_t b = nullptr;
    cudssMatrixType_t mtype = CUDSS_MTYPE_SYMMETRIC;
    cudssMatrixViewType_t mview = CUDSS_MVIEW_UPPER;
    cudssIndexBase_t base = CUDSS_BASE_ZERO;
    cudssStatus_t status = CUDSS_STATUS_SUCCESS;
    cudaStream_t last_stream = nullptr; // track stream for caching + cleanup
    int64_t n = 0;
    int64_t nnz = 0;
    int64_t nrhs = 0;
    int64_t call_count = 0; // necessary for detecting if we need further instantiation in execution stage
    size_t sizeWritten = 0;
    cudaDataType cuda_dtype = get_cuda_data_type<T>();
    bool async_mem_handler_set = false; // track whether async memory handler was configured

    // this is literally only for debugging
    using native_dtype = typename get_native_data_type<T>::type;

    ~CudssState() {
        if (handle) {
            // Synchronize with the last stream before destroying resources
            if (last_stream) {
                cudaStreamSynchronize(last_stream);
            }
            cudssMatrixDestroy(A);
            cudssMatrixDestroy(b);
            cudssMatrixDestroy(x);
            cudssDataDestroy(handle, data);
            cudssConfigDestroy(config);
            cudssDestroy(handle);
        }
    }
};

template <> ffi::TypeId CudssState<ffi::F32>::id = {};
template <> ffi::TypeId CudssState<ffi::F64>::id = {};
template <> ffi::TypeId CudssState<ffi::C64>::id = {};
template <> ffi::TypeId CudssState<ffi::C128>::id = {};

// Try to configure cuDSS async memory handler using CUDA memory pools.
// This eliminates synchronous cudaMalloc/cudaFree during solve, which is
// critical for CUDA graph capture compatibility.
template <ffi::DataType T>
static void trySetAsyncMemHandler(CudssState<T>* state, int device_id) {
    if (state->async_mem_handler_set) return;

    cudaMemPool_t memPool = nullptr;
    cudaError_t err = cudaDeviceGetDefaultMemPool(&memPool, device_id);
    if (err != cudaSuccess || memPool == nullptr) {
        // Async memory pools not available (old driver/device) — fall back silently
        return;
    }

    // cudssDeviceMemHandler_t expects function pointers for alloc/free.
    // cudaMallocFromPoolAsync and cudaFreeAsync match the required signatures
    // when the pool handle is passed as the ctx pointer.
    cudssDeviceMemHandler_t handler;
    handler.ctx = memPool;
    handler.device_alloc = [](void* ctx, void** ptr, size_t size, cudaStream_t stream) -> int {
        cudaMemPool_t pool = reinterpret_cast<cudaMemPool_t>(ctx);
        return static_cast<int>(cudaMallocFromPoolAsync(ptr, size, pool, stream));
    };
    handler.device_free = [](void* ctx, void* ptr, size_t size, cudaStream_t stream) -> int {
        (void)ctx;
        (void)size;
        return static_cast<int>(cudaFreeAsync(ptr, stream));
    };

    cudssStatus_t s = cudssSetDeviceMemHandler(state->handle, &handler);
    if (s == CUDSS_STATUS_SUCCESS) {
        state->async_mem_handler_set = true;
    }
    // If it fails, cuDSS will use default (sync) allocator — not fatal
}

// instantiation ===============================================================

// instantiate everything that is not a function of the context (cudaStream_t)
template <ffi::DataType T>
static ffi::ErrorOr<std::unique_ptr<CudssState<T>>> CudssInstantiate(
    const int64_t device_id,                // the device to run this on
    const int64_t mtype_id,                 // {0: gen, 1: sym, 2: herm, 3: spd, 4: hpd}
    const int64_t mview_id                  // {0: full, 1: triu, 2: tril}
) {

    // make a new state which will manage CuDSS's state
    auto state = std::make_unique<CudssState<T>>();

    // check on the type of matrix being solved
    if (mtype_id == 0) {
        state->mtype = CUDSS_MTYPE_GENERAL;
    } else if (mtype_id == 1) {
        state->mtype = CUDSS_MTYPE_SYMMETRIC;
    } else if (mtype_id == 2) {
        state->mtype = CUDSS_MTYPE_HERMITIAN;
    } else if (mtype_id == 3) {
        state->mtype = CUDSS_MTYPE_SPD;
    } else if (mtype_id == 4) {
        state->mtype = CUDSS_MTYPE_HPD;
    } else {
        throw std::invalid_argument("Invalid mtype_id. Valid options: 0: general, 1: symmetric, 2: hermitian, 3: spd, 4: hpd");
    }

    // check on the view of the matrix provided
    if (mview_id == 0) {
        state->mview = CUDSS_MVIEW_FULL;
    } else if (mview_id == 1) {
        state->mview = CUDSS_MVIEW_UPPER;
    } else if (mview_id == 2) {
        state->mview = CUDSS_MVIEW_LOWER;
    } else {
        throw std::invalid_argument("Invalid mview_id. Valid options: 0: full, 1: upper, 2: lower");
    }

    state->nrhs = 1; // the non-batched case

    // CUDA setup
    cudaSetDevice(device_id);

    return ffi::ErrorOr<std::unique_ptr<CudssState<T>>>(std::move(state));
}

// execution ===================================================================
template <ffi::DataType T>
static ffi::Error CudssExecute(
    cudaStream_t stream,                    // JAXs stream given to this context (jit)
    CudssState<T>* state,                   // the state we instantiated in CudssInstantiate
    ffi::Buffer<T> b_values_buf,            // the real input data that varies per solution
    ffi::Buffer<T> csr_values_buf,          // the real input data that varies per solution
    ffi::Buffer<ffi::S32> offsets_buf,
    ffi::Buffer<ffi::S32> columns_buf,
    ffi::ResultBuffer<T> out_values_buf,    // the output buffer we write the answer to
    ffi::ResultBuffer<T> diag_buf,          // diagnostic: diagonal of factored matrix
    ffi::ResultBuffer<ffi::S32> perm_buf,   // diagnostic: reorder permutation
    const int64_t device_id,                // the device to run this on
    const int64_t mtype_id,                 // {0: gen, 1: sym, 2: herm, 3: spd, 4: hpd}
    const int64_t mview_id                  // {0: full, 1: triu, 2: tril}
) {

    // Cold path: first call — set up cuDSS handle, matrix descriptors, and do
    // initial analysis + factorize + solve. This only runs once.
    if (state->call_count == 0) {

        // figure this out on first call
        state->n = offsets_buf.element_count() - 1;
        state->nnz = columns_buf.element_count();

        // Track stream
        state->last_stream = stream;

        // CuDSS setup
        CUDSS_CALL_AND_CHECK(cudssCreate(&state->handle), state->status, "cudssCreate");

        // Configure async memory handler BEFORE any cuDSS operations.
        // This makes cuDSS use cudaMallocFromPoolAsync/cudaFreeAsync instead of
        // synchronous cudaMalloc/cudaFree, which is required for CUDA graph capture.
        trySetAsyncMemHandler<T>(state, device_id);

        CUDSS_CALL_AND_CHECK(cudssSetStream(state->handle, stream), state->status, "cudssSetStream");
        CUDSS_CALL_AND_CHECK(cudssConfigCreate(&state->config), state->status, "cudssConfigCreate");
        CUDSS_CALL_AND_CHECK(cudssDataCreate(state->handle, &state->data), state->status, "cudssDataCreate");

        // CuDSS structures creation
        CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&state->b, state->n, state->nrhs, state->n,
            b_values_buf.typed_data(), state->cuda_dtype, CUDSS_LAYOUT_COL_MAJOR), state->status, "cudssMatrixCreateDn for b");

        CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&state->x, state->n, state->nrhs, state->n,
            out_values_buf->typed_data(), state->cuda_dtype, CUDSS_LAYOUT_COL_MAJOR), state->status, "cudssMatrixCreateDn for x");

        CUDSS_CALL_AND_CHECK(cudssMatrixCreateCsr(&state->A, state->n, state->n, state->nnz,
            offsets_buf.typed_data(), NULL,
            columns_buf.typed_data(),
            csr_values_buf.typed_data(),
            CUDA_R_32I, state->cuda_dtype,
            state->mtype, state->mview, state->base), state->status, "cudssMatrixCreateCsr");

        // CuDSS config — iterative refinement
        int iter_ref_nsteps = 5;
        CUDSS_CALL_AND_CHECK(cudssConfigSet(state->config, CUDSS_CONFIG_IR_N_STEPS,
                            &iter_ref_nsteps, sizeof(iter_ref_nsteps)), state->status, "cudssConfigSet ir_nsteps");

        // cold solve - analyze, factorize, solve
        CUDSS_CALL_AND_CHECK(cudssExecute(state->handle, CUDSS_PHASE_ANALYSIS,
            state->config, state->data, state->A, state->x, state->b), state->status, "cudssExecute analysis");

        CUDSS_CALL_AND_CHECK(cudssExecute(state->handle, CUDSS_PHASE_FACTORIZATION,
            state->config, state->data, state->A, state->x, state->b), state->status, "cudssExecute factorization");

        CUDSS_CALL_AND_CHECK(cudssExecute(state->handle, CUDSS_PHASE_SOLVE,
            state->config, state->data, state->A, state->x, state->b), state->status, "cudssExecute solve");

        // Extract diagnostics only on first call (cold path).
        // These involve D2H copies which would break graph capture on the hot path.
        state->status = cudssDataGet(state->handle, state->data, CUDSS_DATA_DIAG, diag_buf->typed_data(),
                        state->n * sizeof(typename get_native_data_type<T>::type), &state->sizeWritten);
        if (state->status != CUDSS_STATUS_SUCCESS) {
            CUDA_CHECK(cudaMemsetAsync(diag_buf->typed_data(), 0,
                        state->n * sizeof(typename get_native_data_type<T>::type), stream));
        }

        state->status = cudssDataGet(state->handle, state->data, CUDSS_DATA_PERM_REORDER_ROW, perm_buf->typed_data(),
                        state->n * sizeof(int32_t), &state->sizeWritten);
        if (state->status != CUDSS_STATUS_SUCCESS) {
            CUDA_CHECK(cudaMemsetAsync(perm_buf->typed_data(), 0, state->n * sizeof(int32_t), stream));
        }

        state->call_count++;
    }
    else {
        // Hot path: subsequent calls — refactorize + solve only.
        // This path must be graph-capture compatible (no D2H, no sync alloc).

        // Only update stream if it changed (avoids unnecessary cuDSS internal work)
        if (stream != state->last_stream) {
            CUDSS_CALL_AND_CHECK(cudssSetStream(state->handle, stream), state->status, "cudssSetStream");
            state->last_stream = stream;
        }

        // Update matrix data pointers
        CUDSS_CALL_AND_CHECK(cudssMatrixSetCsrPointers(state->A,
            offsets_buf.typed_data(), NULL,
            columns_buf.typed_data(),
            csr_values_buf.typed_data()), state->status, "update_pointers A");

        CUDSS_CALL_AND_CHECK(cudssMatrixSetValues(state->b, b_values_buf.typed_data()), state->status, "update_pointers b");
        CUDSS_CALL_AND_CHECK(cudssMatrixSetValues(state->x, out_values_buf->typed_data()), state->status, "update_pointers x");

        // warm solve - refactorize, solve
        CUDSS_CALL_AND_CHECK(cudssExecute(state->handle, CUDSS_PHASE_REFACTORIZATION,
            state->config, state->data, state->A, state->x, state->b), state->status, "cudssExecute refactorization");

        CUDSS_CALL_AND_CHECK(cudssExecute(state->handle, CUDSS_PHASE_SOLVE,
            state->config, state->data, state->A, state->x, state->b), state->status, "cudssExecute solve");

        // Hot path: zero out diagnostic buffers without D2H copy.
        // The permutation doesn't change between refactorizations (same sparsity
        // pattern), and diag is only used for inertia computation which the
        // caller can cache from the first call.
        CUDA_CHECK(cudaMemsetAsync(diag_buf->typed_data(), 0,
                    state->n * sizeof(typename get_native_data_type<T>::type), stream));
        CUDA_CHECK(cudaMemsetAsync(perm_buf->typed_data(), 0, state->n * sizeof(int32_t), stream));
    }

    return ffi::Error::Success();
}

// minimize XLA/nanobind boilerplate with a couple macros ======================

// XLA ffi handler definitions for all datatypes
// Execute handler has kCmdBufferCompatible trait so XLA can include it in
// CUDA command buffers / graph capture, keeping the NR loop GPU-resident.
#define DEFINE_CUDSS_FFI_HANDLERS(TypeName, DataType) \
    XLA_FFI_DEFINE_HANDLER(kCudssInstantiate##TypeName, CudssInstantiate<DataType>, \
        ffi::Ffi::BindInstantiate() \
            .Attr<int64_t>("device_id") \
            .Attr<int64_t>("mtype_id") \
            .Attr<int64_t>("mview_id")); \
    \
    XLA_FFI_DEFINE_HANDLER(kCudssExecute##TypeName, CudssExecute<DataType>, \
        ffi::Ffi::Bind() \
            .Ctx<ffi::PlatformStream<cudaStream_t>>() \
            .Ctx<ffi::State<CudssState<DataType>>>() \
            .Arg<ffi::Buffer<DataType>>() \
            .Arg<ffi::Buffer<DataType>>() \
            .Arg<ffi::Buffer<ffi::S32>>() \
            .Arg<ffi::Buffer<ffi::S32>>() \
            .Ret<ffi::Buffer<DataType>>() \
            .Ret<ffi::Buffer<DataType>>() \
            .Ret<ffi::Buffer<ffi::S32>>() \
            .Attr<int64_t>("device_id") \
            .Attr<int64_t>("mtype_id") \
            .Attr<int64_t>("mview_id"), \
        {ffi::Traits::kCmdBufferCompatible});

// Generate all the FFI handlers using the macro
DEFINE_CUDSS_FFI_HANDLERS(f32, ffi::F32);
DEFINE_CUDSS_FFI_HANDLERS(f64, ffi::F64);
DEFINE_CUDSS_FFI_HANDLERS(c64, ffi::C64);
DEFINE_CUDSS_FFI_HANDLERS(c128, ffi::C128);

#if defined(XLA_FFI_API_MINOR) && (XLA_FFI_API_MINOR >= 2)
  #define ADD_TYPE(d, DTYPE) do { \
      using StateT = CudssState<DTYPE>; \
      static auto kStateTypeInfo = xla::ffi::MakeTypeInfo<StateT>(); \
      (d)["type_info"] = nb::capsule(reinterpret_cast<void*>(&kStateTypeInfo)); \
      (d)["type_id"]   = nb::capsule(reinterpret_cast<void*>(&StateT::id)); \
    } while (0)
#else
  #define ADD_TYPE(d, DTYPE) do { \
      (d)["state_type"] = nb::dict(); \
    } while (0)
#endif

// nanobind module exporting macro
#define EXPORT_CUDSS_HANDLERS(m, TypeName, DataType) \
    m.def("state_dict_" #TypeName, []() { \
        nb::dict d; \
        ADD_TYPE(d, DataType); \
        return d; \
    }); \
    m.def("type_id_" #TypeName, []() { \
        return nb::capsule(reinterpret_cast<void*>(&CudssState<DataType>::id)); \
    }); \
    m.def("handler_" #TypeName, []() { \
        nb::dict d; \
        d["instantiate"] = nb::capsule(reinterpret_cast<void*>(kCudssInstantiate##TypeName)); \
        d["execute"] = nb::capsule(reinterpret_cast<void*>(kCudssExecute##TypeName)); \
        return d; \
    });

// generate all nanobind modules! :)
NB_MODULE(single_solve, m) {
    EXPORT_CUDSS_HANDLERS(m, f32, ffi::F32);
    EXPORT_CUDSS_HANDLERS(m, f64, ffi::F64);
    EXPORT_CUDSS_HANDLERS(m, c64, ffi::C64);
    EXPORT_CUDSS_HANDLERS(m, c128, ffi::C128);
}
