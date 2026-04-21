"""JAX kernel cache - only used when MLIR is disabled."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock

from torch_neuronx.python_ops.jax.kernel import JaxKernel
from torch_neuronx.python_ops.processors.argument_processor import ArgumentProcessor


@dataclass(frozen=True)
class JaxKernelKey:
    """Normalized identity for a JAX kernel configuration."""

    fn_id: str
    op_name: str
    static_argnums: tuple[int, ...]
    static_argnames: tuple[str, ...]
    output_params: tuple[str, ...]
    argproc_id: str


_JAX_CACHE: dict[JaxKernelKey, JaxKernel] = {}
_JAX_LOCK = RLock()


def _fn_identity(fn: Callable) -> str:
    mod = getattr(fn, "__module__", None)
    name = getattr(fn, "__name__", None)
    if mod and name:
        return f"{mod}:{name}"
    return repr(fn)


def get_or_create_jax_kernel(
    *,
    jax_fn: Callable,
    op_name: str,
    static_argnums: tuple[int, ...] | None = None,
    static_argnames: tuple[str, ...] | None = None,
    output_params: tuple[str, ...] | None = None,
    argument_processor_cls: type[ArgumentProcessor] | None = None,
) -> JaxKernel:
    """Return a cached JaxKernel for the given configuration."""

    key = JaxKernelKey(
        fn_id=_fn_identity(jax_fn),
        op_name=op_name,
        static_argnums=tuple(static_argnums or ()),
        static_argnames=tuple(static_argnames or ()),
        output_params=tuple(output_params or ()),
        argproc_id=(argument_processor_cls.__name__ if argument_processor_cls else "default"),
    )
    with _JAX_LOCK:
        k = _JAX_CACHE.get(key)
        if k is not None:
            return k

        kernel = JaxKernel(
            jax_fn,
            op_name=op_name,
            static_argnums=static_argnums,
            static_argnames=static_argnames,
            output_params=output_params,
        )
        if argument_processor_cls is not None:
            kernel.arg_processor = argument_processor_cls(static_argnames)  # type: ignore[assignment]
        _JAX_CACHE[key] = kernel
        return kernel


def kernel_cache_size() -> int:
    """Return the number of cached kernel instances (for tests/diagnostics)."""
    with _JAX_LOCK:
        return len(_JAX_CACHE)
