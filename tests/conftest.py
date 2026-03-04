import inspect

import numpy as np
from jax import config as jax_config


jax_config.update("jax_enable_x64", True)


def _patch_numpy_asarray_copy_kwarg() -> None:
    """Backport `copy=` support for NumPy versions that do not expose it."""
    try:
        has_copy_kwarg = "copy" in inspect.signature(np.asarray).parameters
    except (TypeError, ValueError):
        has_copy_kwarg = False

    if has_copy_kwarg:
        return

    _np_asarray = np.asarray

    def _asarray_compat(a, dtype=None, order=None, *, like=None, copy=None):
        if like is not None:
            out = _np_asarray(a, dtype=dtype, order=order, like=like)
        else:
            out = _np_asarray(a, dtype=dtype, order=order)
        if copy is True:
            return np.array(out, dtype=dtype, order=order, copy=True)
        return out

    np.asarray = _asarray_compat


def _patch_jax_shapedarray_named_shape() -> None:
    """Allow loading pickle payloads created with newer JAX metadata keys."""
    try:
        from jax._src import core as jax_core
    except Exception:
        return

    original_update = jax_core.ShapedArray.update

    def _compat_update(self, shape=None, dtype=None, weak_type=None, **kwargs):
        kwargs.pop("named_shape", None)
        return original_update(self, shape=shape, dtype=dtype, weak_type=weak_type, **kwargs)

    jax_core.ShapedArray.update = _compat_update


_patch_numpy_asarray_copy_kwarg()
_patch_jax_shapedarray_named_shape()
