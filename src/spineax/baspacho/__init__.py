"""BaSpaCho backend for Spineax.

This module provides a sparse LU solver using BaSpaCho, which supports
Metal, CUDA, OpenCL, and CPU backends. It's designed to be a drop-in
replacement for the cuDSS backend when running on Apple Silicon.

Usage:
    from spineax.baspacho import BaspachoSolver, is_available

    if is_available():
        solver = BaspachoSolver(indptr, indices, n, backend="metal")
        x, inertia = solver(b, csr_values)
"""

from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Try to import the baspacho Python bindings
_BASPACHO_AVAILABLE = False
_BaspachoSolverImpl = None
_is_metal_available = lambda: False
_is_cuda_available = lambda: False
_is_opencl_available = lambda: False

try:
    from baspacho_py import (
        BaspachoSolver as _BaspachoSolverImpl,
        is_metal_available as _is_metal_available,
        is_cuda_available as _is_cuda_available,
        is_opencl_available as _is_opencl_available,
    )
    _BASPACHO_AVAILABLE = True
    logger.info("BaSpaCho solver available")
except ImportError as e:
    logger.debug(f"BaSpaCho not available: {e}")


def is_available() -> bool:
    """Check if BaSpaCho solver is available."""
    return _BASPACHO_AVAILABLE


def is_metal_available() -> bool:
    """Check if Metal backend is available."""
    return _BASPACHO_AVAILABLE and _is_metal_available()


def is_cuda_available() -> bool:
    """Check if CUDA backend is available (via BaSpaCho, not cuDSS)."""
    return _BASPACHO_AVAILABLE and _is_cuda_available()


def is_opencl_available() -> bool:
    """Check if OpenCL backend is available."""
    return _BASPACHO_AVAILABLE and _is_opencl_available()


def get_best_backend() -> str:
    """Determine the best available backend.

    Returns:
        Backend name: "metal", "cuda", "opencl", or "cpu"
    """
    if is_metal_available():
        return "metal"
    elif is_cuda_available():
        return "cuda"
    elif is_opencl_available():
        return "opencl"
    else:
        return "cpu"


class BaspachoSolver:
    """Sparse LU solver using BaSpaCho.

    This provides the same interface as CuDSSSolver but uses BaSpaCho
    as the backend, which supports Metal for Apple Silicon.

    Usage:
        solver = BaspachoSolver(indptr, indices, n, backend="metal")
        x, inertia = solver(b, csr_values)
    """

    def __init__(
        self,
        indptr,  # CSR row pointers (n+1,)
        indices,  # CSR column indices (nnz,)
        n: int,
        backend: str = "auto",
        matrix_type: str = "general",
    ):
        """Initialize solver with CSR sparsity pattern.

        Args:
            indptr: CSR row pointers array (n+1 elements)
            indices: CSR column indices array (nnz elements)
            n: Matrix dimension
            backend: "metal", "cuda", "opencl", "cpu", or "auto"
            matrix_type: "general", "symmetric", "spd", "hermitian", "hpd"
        """
        if not _BASPACHO_AVAILABLE:
            raise RuntimeError(
                "BaSpaCho is not available. Build with: "
                "cmake -DBASPACHO_BUILD_PYTHON=ON && make baspacho_py"
            )

        import numpy as np
        import jax.numpy as jnp

        # Convert JAX arrays to numpy for the C++ bindings
        if hasattr(indptr, 'numpy'):
            indptr = np.asarray(indptr)
        if hasattr(indices, 'numpy'):
            indices = np.asarray(indices)

        # Ensure int64 for C++ bindings
        indptr = indptr.astype(np.int64)
        indices = indices.astype(np.int64)

        if backend == "auto":
            backend = get_best_backend()

        self._solver = _BaspachoSolverImpl(
            indptr, indices, n, backend, matrix_type
        )
        self.n = n
        self.backend = backend
        logger.debug(f"BaspachoSolver created: {n}x{n}, backend={backend}")

    def __call__(
        self,
        b,  # RHS vector (n,)
        csr_values,  # CSR non-zero values (nnz,)
    ) -> Tuple:
        """Solve Ax = b.

        Args:
            b: Right-hand side vector
            csr_values: CSR non-zero values (same sparsity as constructor)

        Returns:
            Tuple of (x, inertia) where:
                x: Solution vector
                inertia: Array [positive, negative] eigenvalue counts
        """
        import numpy as np
        import jax.numpy as jnp

        # Convert to numpy for C++ bindings
        if hasattr(b, 'numpy'):
            b_np = np.asarray(b)
        else:
            b_np = b

        if hasattr(csr_values, 'numpy'):
            values_np = np.asarray(csr_values)
        else:
            values_np = csr_values

        # Ensure float64 for C++ bindings
        b_np = b_np.astype(np.float64)
        values_np = values_np.astype(np.float64)

        # Call the C++ solver
        x_np, inertia_np = self._solver.solve(b_np, values_np)

        # Convert back to JAX arrays
        x = jnp.array(x_np)
        inertia = jnp.array(inertia_np)

        return x, inertia
