"""Shared helper for tests of ``spice.spectrum.aemu_spectrum_emulator``.

That module imports ``astro_emulators_toolkit`` (the optional ``[aemu]`` extra) at
import time. Its schema/metadata helpers never touch the toolkit runtime, so
tests of them should run without the extra installed.
"""
import importlib
import sys
import types

_TOOLKIT_NAMES = ("astro_emulators_toolkit", "astro_emulators_toolkit.emulator_runtime")


def import_aemu_wrapper():
    """Import ``spice.spectrum.aemu_spectrum_emulator``, with or without the toolkit.

    If the real toolkit is importable it is used. Otherwise minimal stand-ins are
    placed in ``sys.modules`` only for the duration of the import and removed
    again afterwards, so later tests cannot mistake the stand-in for an installed
    toolkit.
    """
    name = "spice.spectrum.aemu_spectrum_emulator"
    if name in sys.modules:
        return sys.modules[name]
    try:
        importlib.import_module("astro_emulators_toolkit.emulator_runtime")
        return importlib.import_module(name)
    except ImportError:
        pass

    saved = {n: sys.modules.get(n) for n in _TOOLKIT_NAMES}
    fake_pkg = types.ModuleType("astro_emulators_toolkit")
    fake_pkg.Emulator = type("Emulator", (), {})
    fake_pkg.normalize_tree = lambda tree, *a, **kw: tree
    fake_pkg.denormalize_tree = lambda tree, *a, **kw: tree
    fake_runtime = types.ModuleType("astro_emulators_toolkit.emulator_runtime")
    fake_runtime.apply_jax_runtime = lambda *a, **kw: None
    fake_runtime.make_frozen_apply_runtime = lambda *a, **kw: (lambda x: x)
    fake_pkg.emulator_runtime = fake_runtime
    sys.modules["astro_emulators_toolkit"] = fake_pkg
    sys.modules["astro_emulators_toolkit.emulator_runtime"] = fake_runtime
    try:
        return importlib.import_module(name)
    finally:
        for n, mod in saved.items():
            if mod is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = mod
