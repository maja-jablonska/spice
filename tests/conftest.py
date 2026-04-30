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


_patch_numpy_asarray_copy_kwarg()

from spice._jax_compat import apply as _apply_jax_compat
_apply_jax_compat()
