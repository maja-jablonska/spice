"""JAX cross-version compatibility patches.

JAX 0.4.32 removed the `named_shape` field from `ShapedArray` (xmap removal).
Pickles and kwargs forwarded by older JAX/spice code can still pass
`named_shape=` into `ShapedArray.__new__`, `ShapedArray.__init__`, or
`ShapedArray.update`. Strip it silently so we can load older artefacts on
newer JAX without bespoke patches in every entry point (notebook, CLI
script, pytest conftest).

Important: only wrap a method if `ShapedArray` actually defines it. On
recent JAX, state is set inside `__init__` and `__new__` is just
`object.__new__`; on older JAX the reverse holds. Wrapping the inherited
`object.<method>` would forward extra args into a no-arg constructor and
break every aval construction.
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

    if cls.__new__ is not object.__new__:
        original_new = cls.__new__

        def _compat_new(klass, *args, **kwargs):
            kwargs.pop("named_shape", None)
            return original_new(klass, *args, **kwargs)

        cls.__new__ = staticmethod(_compat_new)

    if cls.__init__ is not object.__init__:
        original_init = cls.__init__

        def _compat_init(self, *args, **kwargs):
            kwargs.pop("named_shape", None)
            return original_init(self, *args, **kwargs)

        cls.__init__ = _compat_init

    update_method = getattr(cls, "update", None)
    if update_method is not None:
        original_update = update_method

        def _compat_update(self, *args, **kwargs):
            kwargs.pop("named_shape", None)
            return original_update(self, *args, **kwargs)

        cls.update = _compat_update

    _PATCHED = True
