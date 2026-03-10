/*
 * Header for BaSpaCho dense solver CUDA kernels.
 *
 * Declares launcher wrapper functions so the FFI handler (.cpp, compiled by
 * the host compiler) can invoke GPU kernels without needing nvcc's <<<>>>
 * syntax.
 */
#pragma once

#include <cstdint>

// Forward-declare cudaStream_t to avoid pulling <cuda_runtime.h> into
// host-compiler translation units (which triggers nv/target issues).
// The .cu file includes the full header.
struct CUstream_st;
typedef CUstream_st* cudaStream_t;

// Launch dense-to-coalesced scatter kernel on the given CUDA stream.
// Scatters a row-major n×n Jacobian into BaSpaCho's coalesced block format.
template <typename T>
void launchDenseToCoalesced(
    cudaStream_t stream,
    const T* J,             // n×n row-major input (device)
    T* data,                // BaSpaCho coalesced output (device)
    int64_t n,
    const int64_t* spanStart,
    const int64_t* spanToLump,
    const int64_t* lumpStart,
    const int64_t* spanOffsetInLump,
    const int64_t* chainColPtr,
    const int64_t* chainRowSpan,
    const int64_t* chainData,
    const int64_t* permutation,
    const int64_t* upperChainRowPtr,
    const int64_t* upperChainColSpan,
    const int64_t* upperChainData,
    int64_t upperDataBase
);

// Launch forward permutation kernel: perm_f[perm[i]] = f[i]
template <typename T>
void launchPermuteForward(
    cudaStream_t stream,
    const T* f,
    T* perm_f,
    const int64_t* perm,
    int64_t n
);

// Launch inverse permutation kernel: out[i] = sol[perm[i]]
template <typename T>
void launchPermuteInverse(
    cudaStream_t stream,
    const T* sol,
    T* out,
    const int64_t* perm,
    int64_t n
);
