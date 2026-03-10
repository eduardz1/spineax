/*
 * BaSpaCho dense solver CUDA kernels.
 *
 * Compiled by nvcc. Contains only device code and launcher wrappers —
 * no BaSpaCho or Eigen headers (which nvcc cannot parse).
 *
 * GPU kernels:
 *   1. denseToCoalescedKernel — scatter row-major J into BaSpaCho's coalesced format
 *   2. permuteForwardKernel  — permute RHS vector for BaSpaCho's internal ordering
 *   3. permuteInverseKernel  — unpermute solution vector back to original ordering
 */

#include <cuda_runtime.h>
#include "baspacho_dense_kernels.h"

// ============================================================================
// Device helpers
// ============================================================================

// Device-side binary search matching BaSpaCho's bisect() from Utils.h.
// Finds position where needle >= array[pos] (lower bound).
__device__ inline int64_t device_bisect(const int64_t* array, int64_t size,
                                        int64_t needle) {
  int64_t a = 0, b = size;
  while (b - a > 1) {
    int64_t m = (a + b) / 2;
    if (needle >= array[m]) {
      a = m;
    } else {
      b = m;
    }
  }
  return a;
}

// ============================================================================
// GPU Kernels
// ============================================================================

/**
 * Scatter a dense n×n row-major Jacobian into BaSpaCho's coalesced block format.
 *
 * Each thread handles one (origRow, origCol) element. The kernel applies the
 * BaSpaCho permutation and writes to either the lower triangle data region
 * or the upper triangle data region (for MTYPE_GENERAL matrices).
 *
 * For 1×1 blocks (our use case), each "block" is a single scalar, so the
 * inner block loops collapse to single assignments.
 */
template <typename T>
__global__ void denseToCoalescedKernel(
    const T* __restrict__ J,       // n×n row-major input (device)
    T* __restrict__ data,          // BaSpaCho coalesced output (device)
    int64_t n,
    // Accessor metadata (all device pointers):
    const int64_t* spanStart,
    const int64_t* spanToLump,
    const int64_t* lumpStart,
    const int64_t* spanOffsetInLump,
    const int64_t* chainColPtr,
    const int64_t* chainRowSpan,
    const int64_t* chainData,
    const int64_t* permutation,
    // Upper triangle metadata (device pointers, for MTYPE_GENERAL):
    const int64_t* upperChainRowPtr,
    const int64_t* upperChainColSpan,
    const int64_t* upperChainData,
    int64_t upperDataBase
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n * n) return;

  int64_t origRow = idx / n;
  int64_t origCol = idx % n;
  T value = J[origRow * n + origCol];

  // Apply permutation
  int64_t permRow = permutation[origRow];
  int64_t permCol = permutation[origCol];

  if (permRow >= permCol) {
    // Lower triangle (including diagonal):
    // Replicate CoalescedAccessor::blockOffset(permRow, permCol) logic
    int64_t lump = spanToLump[permCol];
    int64_t lumpSz = lumpStart[lump + 1] - lumpStart[lump];
    int64_t offInLump = spanOffsetInLump[permCol];
    int64_t start = chainColPtr[lump];
    int64_t end = chainColPtr[lump + 1];
    int64_t pos = device_bisect(chainRowSpan + start, end - start, permRow);
    int64_t offset = chainData[start + pos] + offInLump;
    // Row offset within the chain entry's sub-matrix.
    // For 1×1 blocks this is always 0 (each span = one row).
    int64_t rowOffInSpan = spanStart[permRow] - spanStart[chainRowSpan[start + pos]];
    data[offset + rowOffInSpan * lumpSz] = value;
  } else {
    // Upper triangle (permRow < permCol) — MTYPE_GENERAL only
    int64_t rowLump = spanToLump[permRow];
    int64_t colLump = spanToLump[permCol];

    if (rowLump == colLump) {
      // Intra-lump: both spans in same lump — store in diagonal block.
      // The diagonal block is a full lumpSize×lumpSize matrix (row-major).
      int64_t lumpSz = lumpStart[rowLump + 1] - lumpStart[rowLump];
      int64_t diagStart = chainData[chainColPtr[rowLump]];
      int64_t rowOff = spanOffsetInLump[permRow];
      int64_t colOff = spanOffsetInLump[permCol];
      data[diagStart + rowOff * lumpSz + colOff] = value;
    } else {
      // Inter-lump: use upper triangle storage.
      // Replicate CoalescedAccessor::upperBlockOffset(permRow, permCol) logic
      int64_t lump = spanToLump[permRow];
      int64_t lumpSz = lumpStart[lump + 1] - lumpStart[lump];
      int64_t offInLump = spanOffsetInLump[permRow];
      int64_t start = upperChainRowPtr[lump];
      int64_t end = upperChainRowPtr[lump + 1];
      int64_t pos = device_bisect(upperChainColSpan + start, end - start, permCol);
      int64_t uOff = upperChainData[start + pos] + offInLump;
      data[upperDataBase + uOff] = value;
    }
  }
}

/**
 * Permute RHS vector: perm_f[perm[i]] = f[i]
 * Maps from original ordering to BaSpaCho's internal span ordering.
 */
template <typename T>
__global__ void permuteForwardKernel(
    const T* __restrict__ f,
    T* __restrict__ perm_f,
    const int64_t* __restrict__ perm,
    int64_t n
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) perm_f[perm[i]] = f[i];
}

/**
 * Unpermute solution vector: out[i] = sol[perm[i]]
 * Maps from BaSpaCho's internal span ordering back to original ordering.
 */
template <typename T>
__global__ void permuteInverseKernel(
    const T* __restrict__ sol,
    T* __restrict__ out,
    const int64_t* __restrict__ perm,
    int64_t n
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = sol[perm[i]];
}

// ============================================================================
// Launcher wrappers (callable from host C++ code without <<<>>> syntax)
// ============================================================================

template <typename T>
void launchDenseToCoalesced(
    cudaStream_t stream,
    const T* J,
    T* data,
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
) {
  int threads = 256;
  int blocks = (static_cast<int>(n * n) + threads - 1) / threads;
  denseToCoalescedKernel<T><<<blocks, threads, 0, stream>>>(
      J, data, n,
      spanStart, spanToLump, lumpStart,
      spanOffsetInLump, chainColPtr,
      chainRowSpan, chainData, permutation,
      upperChainRowPtr, upperChainColSpan,
      upperChainData, upperDataBase);
}

template <typename T>
void launchPermuteForward(
    cudaStream_t stream,
    const T* f,
    T* perm_f,
    const int64_t* perm,
    int64_t n
) {
  int threads = 256;
  int blocks = (static_cast<int>(n) + threads - 1) / threads;
  permuteForwardKernel<T><<<blocks, threads, 0, stream>>>(f, perm_f, perm, n);
}

template <typename T>
void launchPermuteInverse(
    cudaStream_t stream,
    const T* sol,
    T* out,
    const int64_t* perm,
    int64_t n
) {
  int threads = 256;
  int blocks = (static_cast<int>(n) + threads - 1) / threads;
  permuteInverseKernel<T><<<blocks, threads, 0, stream>>>(sol, out, perm, n);
}

// Explicit instantiations for float and double
template void launchDenseToCoalesced<float>(
    cudaStream_t, const float*, float*, int64_t,
    const int64_t*, const int64_t*, const int64_t*, const int64_t*,
    const int64_t*, const int64_t*, const int64_t*, const int64_t*,
    const int64_t*, const int64_t*, const int64_t*, int64_t);
template void launchDenseToCoalesced<double>(
    cudaStream_t, const double*, double*, int64_t,
    const int64_t*, const int64_t*, const int64_t*, const int64_t*,
    const int64_t*, const int64_t*, const int64_t*, const int64_t*,
    const int64_t*, const int64_t*, const int64_t*, int64_t);

template void launchPermuteForward<float>(
    cudaStream_t, const float*, float*, const int64_t*, int64_t);
template void launchPermuteForward<double>(
    cudaStream_t, const double*, double*, const int64_t*, int64_t);

template void launchPermuteInverse<float>(
    cudaStream_t, const float*, float*, const int64_t*, int64_t);
template void launchPermuteInverse<double>(
    cudaStream_t, const double*, double*, const int64_t*, int64_t);
