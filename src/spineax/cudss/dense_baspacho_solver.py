"""BaSpaCho dense LU solver for CUDA via XLA FFI.

Provides a JAX-callable function `baspacho_dense_solve(J, f)` that solves
J @ x = f on GPU using BaSpaCho's supernodal LU factorization.

Phase 2a: Host-side format conversion with GPU factorization/solve.
Phase 2b (future): Full GPU-resident with graph capture compatibility.
"""

import logging
from functools import partial

import jax
import jax.core
import jax.extend.core
from jax.interpreters import mlir
import jax.numpy as jnp
from jaxtyping import Array

logger = logging.getLogger(__name__)

# Force JAX to initialize context BEFORE importing C++ modules
jax.devices()

# Try to import the C++ FFI module
_AVAILABLE = False
_module = None
try:
    from spineax import baspacho_dense_solve as _module
    _AVAILABLE = True
    logger.info("BaSpaCho dense CUDA solver available")
except ImportError as e:
    logger.debug(f"BaSpaCho dense CUDA solver not available: {e}")


def is_available() -> bool:
    """Check if BaSpaCho dense CUDA solver is available."""
    return _AVAILABLE


# ---- FFI registration ----

def _register_ffi(name: str, *, dtype_key: str, platform: str = "CUDA"):
    """Register FFI target and type for the dense solver."""
    handler = getattr(_module, f"handler_{dtype_key}")()
    state_dict = getattr(_module, f"state_dict_{dtype_key}")()
    jax.ffi.register_ffi_type(name, state_dict, platform=platform)
    jax.ffi.register_ffi_target(name, handler, platform=platform)


if _AVAILABLE:
    _register_ffi("baspacho_dense_f32", dtype_key="f32")
    _register_ffi("baspacho_dense_f64", dtype_key="f64")


# ---- JAX primitives ----

_dense_solve_f32_p = jax.extend.core.Primitive("baspacho_dense_solve_f32")
_dense_solve_f32_p.multiple_results = False

_dense_solve_f64_p = jax.extend.core.Primitive("baspacho_dense_solve_f64")
_dense_solve_f64_p.multiple_results = False


def _dense_solve_impl(name: str, J: Array, f: Array, *, n: int):
    """Implementation: call the FFI target."""
    call = jax.ffi.ffi_call(
        name,
        jax.ShapeDtypeStruct((n,), f.dtype),  # single output: x
        has_side_effect=True,
    )
    x = call(J, f, n=n)
    return x


@_dense_solve_f32_p.def_impl
def _dense_solve_f32_impl(J, f, *, n):
    return _dense_solve_impl("baspacho_dense_f32", J, f, n=n)


@_dense_solve_f64_p.def_impl
def _dense_solve_f64_impl(J, f, *, n):
    return _dense_solve_impl("baspacho_dense_f64", J, f, n=n)


# ---- Abstract evaluation (shape/dtype inference for JIT) ----

@_dense_solve_f32_p.def_abstract_eval
def _dense_solve_f32_abstract(J, f, *, n):
    assert J.shape == (n, n), f"J must be ({n}, {n}), got {J.shape}"
    assert f.shape == (n,), f"f must be ({n},), got {f.shape}"
    return jax.core.ShapedArray((n,), J.dtype)


@_dense_solve_f64_p.def_abstract_eval
def _dense_solve_f64_abstract(J, f, *, n):
    assert J.shape == (n, n), f"J must be ({n}, {n}), got {J.shape}"
    assert f.shape == (n,), f"f must be ({n},), got {f.shape}"
    return jax.core.ShapedArray((n,), J.dtype)


# ---- MLIR lowering ----

_dense_solve_f32_low = mlir.lower_fun(
    _dense_solve_f32_impl, multiple_results=False
)
mlir.register_lowering(_dense_solve_f32_p, _dense_solve_f32_low)

_dense_solve_f64_low = mlir.lower_fun(
    _dense_solve_f64_impl, multiple_results=False
)
mlir.register_lowering(_dense_solve_f64_p, _dense_solve_f64_low)


# ---- Public API ----

def baspacho_dense_solve(J: Array, f: Array) -> Array:
    """Solve J @ x = f using BaSpaCho LU on CUDA.

    Args:
        J: n×n Jacobian matrix (JAX array, device or host)
        f: n×1 right-hand side vector (JAX array)

    Returns:
        x: n×1 solution vector such that J @ x = f
    """
    assert J.ndim == 2 and J.shape[0] == J.shape[1], (
        f"J must be square, got shape {J.shape}"
    )
    n = J.shape[0]
    assert f.shape == (n,), f"f must have shape ({n},), got {f.shape}"

    if J.dtype == jnp.float32:
        return _dense_solve_f32_p.bind(J, f, n=n)
    elif J.dtype == jnp.float64:
        return _dense_solve_f64_p.bind(J, f, n=n)
    else:
        raise ValueError(
            f"Unsupported dtype {J.dtype}. "
            "BaSpaCho dense solver supports float32 and float64."
        )
