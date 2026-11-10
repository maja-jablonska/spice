"""Tests for the MARCS [Fe/H] node helper and the ``marcs_fe_h`` filling in the aemu
parameter helper (HARPS cno-v2 bundle, 12 inputs)."""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from spice.spectrum.marcs_nodes import MARCS_FEH_NODES, resolve_marcs_fe_h, snap_marcs_feh


def test_nodes_are_the_fifteen_marcs_metallicities():
    assert MARCS_FEH_NODES.tolist() == [-5, -4, -3, -2.5, -2, -1.5, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]


@pytest.mark.parametrize("feh, node", [(0.0, 0.0), (-0.18, -0.25), (-0.1, 0.0), (0.37, 0.25), (0.38, 0.5), (-3.6, -4.0),
                                       (-2.3, -2.5), (-6.0, -5.0), (1.4, 1.0), (-0.125, -0.25)])
def test_snap_scalar(feh, node):
    assert snap_marcs_feh(feh) == pytest.approx(node)


def test_snap_numpy_and_jax_agree():
    import jax
    import jax.numpy as jnp
    x = np.linspace(-5.5, 1.5, 701)
    a = snap_marcs_feh(x)
    b = np.asarray(jax.jit(snap_marcs_feh)(jnp.asarray(x)))
    assert isinstance(a, np.ndarray) and a.shape == x.shape
    np.testing.assert_allclose(a, b)
    assert set(np.unique(a)).issubset(set(MARCS_FEH_NODES.tolist()))


def test_resolve_modes():
    assert resolve_marcs_fe_h(-0.18, "feh") == -0.18
    assert resolve_marcs_fe_h(-0.18, "snap") == pytest.approx(-0.25)
    with pytest.raises(ValueError):
        resolve_marcs_fe_h(0.0, "nearest")


def _install_fake_aemu_modules() -> None:
    if "astro_emulators_toolkit" not in sys.modules:
        fake_pkg = types.ModuleType("astro_emulators_toolkit")
        fake_pkg.Emulator = type("Emulator", (), {})
        fake_pkg.normalize_tree = lambda tree, *a, **kw: tree
        fake_pkg.denormalize_tree = lambda tree, *a, **kw: tree
        sys.modules["astro_emulators_toolkit"] = fake_pkg
    if "astro_emulators_toolkit.emulator_runtime" not in sys.modules:
        fake_runtime = types.ModuleType("astro_emulators_toolkit.emulator_runtime")
        fake_runtime.apply_jax_runtime = lambda *a, **kw: None
        fake_runtime.make_frozen_apply_runtime = lambda *a, **kw: (lambda x: x)
        sys.modules["astro_emulators_toolkit.emulator_runtime"] = fake_runtime


_install_fake_aemu_modules()
_helper = importlib.import_module("spice.spectrum.aemu_spectrum_emulator")._np_parameter_helper

NAMES_12 = ["marcs_teff", "marcs_logg", "feh", "marcs_fe_h", "vmicro", "a", "c", "n", "o", "r", "s"]


def _fake(names, mode=None):
    lo = np.full(len(names), -1e9); hi = np.full(len(names), 1e9)
    ns = SimpleNamespace(stellar_parameter_names=names, min_stellar_parameters=lo, max_stellar_parameters=hi)
    if mode is not None:
        ns.marcs_fe_h_mode = mode
    return ns


def test_marcs_fe_h_follows_feh_by_default():
    p = _helper(_fake(NAMES_12), {"marcs_teff": 4896.0, "marcs_logg": 2.91, "feh": -0.18})
    assert p[NAMES_12.index("marcs_fe_h")] == pytest.approx(-0.18)
    assert p[NAMES_12.index("feh")] == pytest.approx(-0.18)


def test_marcs_fe_h_snap_mode():
    p = _helper(_fake(NAMES_12, "snap"), {"feh": -0.18})
    assert p[NAMES_12.index("marcs_fe_h")] == pytest.approx(-0.25)


def test_explicit_marcs_fe_h_is_kept():
    p = _helper(_fake(NAMES_12, "snap"), {"feh": -0.18, "marcs_fe_h": 0.0})
    assert p[NAMES_12.index("marcs_fe_h")] == 0.0


def test_bundles_without_marcs_fe_h_unchanged():
    names = [n for n in NAMES_12 if n != "marcs_fe_h"]
    d = {"feh": -0.18}
    p = _helper(_fake(names), d)
    assert p.shape == (len(names),) and p[names.index("feh")] == pytest.approx(-0.18)
    assert d == {"feh": -0.18}  # caller's dict is not mutated
