"""JAX cross-version compatibility patches.

JAX 0.4.32 removed the `named_shape` field from `ShapedArray` (xmap removal).
Pickles, kwargs forwarded by older JAX/spice code, and stale on-disk caches
can still try to pass `named_shape=` through `ShapedArray.__init__`,
`ShapedArray.__new__`, or `ShapedArray.update`. Strip it silently so we can
load older artefacts on newer JAX without bespoke patches in every entry
point (notebook, CLI script, pytest conftest).
"""
from __future__ import annotations

_PATCHED = False


def apply() -> None:
    global _PATCHED
    if _PATCHED:
        return
    try:
        from jax._src import core as jax_core
    except Exception:
        return

    cls = jax_core.ShapedArray

    if hasattr(cls, "update"):
        original_update = cls.update

        def _compat_update(self, shape=None, dtype=None, weak_type=None, **kwargs):
            kwargs.pop("named_shape", None)
            return original_update(
                self, shape=shape, dtype=dtype, weak_type=weak_type, **kwargs,
            )

        cls.update = _compat_update

    original_init = cls.__init__

    def _compat_init(self, *args, **kwargs):
        kwargs.pop("named_shape", None)
        return original_init(self, *args, **kwargs)

    cls.__init__ = _compat_init

    original_new = cls.__new__

    def _compat_new(klass, *args, **kwargs):
        kwargs.pop("named_shape", None)
        try:
            return original_new(klass, *args, **kwargs)
        except TypeError:
            return original_new(klass)

    cls.__new__ = staticmethod(_compat_new)

    _PATCHED = True
