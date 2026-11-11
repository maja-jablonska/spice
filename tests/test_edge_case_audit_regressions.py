"""Regression tests for the 2026-11 audit's edge cases.

* Shoelace areas were evaluated on absolute sky-plane coordinates: in float32
  a face 300 R_sun from the origin lost 23% of its projected area.
* Float32 epochs (x64 off) silently lose orbital phase; SPICE now warns.
* sin_as_fourier_coords had the wrong sign; sin2_as_fourier_coords the wrong
  sign of its oscillating part (and cannot represent its constant term).
"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from spice.geometry.utils import polygon_area
from spice.models.utils import (cos_as_fourier_coords, evaluate_fourier_for_value,
                                sin2_as_fourier_coords, sin_as_fourier_coords)

SRC = str(Path(__file__).resolve().parents[1] / "src")


def _run_without_x64(code: str) -> str:
    out = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True,
                         env={**os.environ, "JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "0",
                              "PYTHONPATH": SRC})
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout + out.stderr


@pytest.mark.parametrize("offset", [0.0, 1e3, 1e6])
def test_polygon_area_is_translation_invariant(offset):
    x = jnp.array([0.0, 0.1, 0.0]) + offset
    y = jnp.array([0.0, 0.0, 0.1]) + offset
    assert float(polygon_area(x, y)) == pytest.approx(0.005, rel=1e-9)


def test_float32_projected_areas_survive_a_large_offset():
    out = _run_without_x64("""
        import jax, numpy as np, jax.numpy as jnp
        assert not jax.config.jax_enable_x64
        from spice.models import IcosphereModel
        m = IcosphereModel.construct(1000, 3.94, 1.96, jnp.array([5000.]), ["teff"])
        far = m._replace(center=jnp.array([300., 0., 0.], dtype=jnp.float32))
        front = np.asarray(m.mus) > 0
        a0, a1 = np.asarray(m.cast_areas)[front], np.asarray(far.cast_areas)[front]
        print("RELERR", np.max(np.abs(a1 - a0) / a0))
    """)
    relerr = float(out.split("RELERR")[1].split()[0])
    assert relerr < 5e-3  # was 0.23


def test_float32_jd_epoch_warns():
    out = _run_without_x64("""
        import warnings, jax
        from spice.models.orbit_utils import warn_if_epoch_precision_lost
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_if_epoch_precision_lost(75.67 / 365.25, 2452599.29 / 365.25)
            warn_if_epoch_precision_lost(75.67 / 365.25, 0.3)
        print("NWARN", len(w))
    """)
    assert "NWARN 1" in out


def test_x64_epoch_does_not_warn():
    from spice.models.orbit_utils import warn_if_epoch_precision_lost
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_epoch_precision_lost(75.67 / 365.25, 2452599.29 / 365.25)


@pytest.mark.parametrize("t", [0.0, 0.1, 0.25, 0.6])
def test_cos_and_sin_coordinates_reproduce_the_functions(t):
    P, A = 1.0, 0.3
    for helper, expected in [(cos_as_fourier_coords, A * np.cos(2 * np.pi * t / P)),
                             (sin_as_fourier_coords, A * np.sin(2 * np.pi * t / P))]:
        period, coords = helper(P, A)
        got = float(evaluate_fourier_for_value(period, coords[:, 0], coords[:, 1], t))
        assert got == pytest.approx(expected, abs=1e-12)


def test_sin2_coordinates_are_the_oscillating_part_and_deprecated():
    P, A, t = 1.0, 0.3, 0.17
    with pytest.warns(DeprecationWarning):
        period, coords = sin2_as_fourier_coords(P, A)
    got = float(evaluate_fourier_for_value(period, coords[:, 0], coords[:, 1], t))
    assert got == pytest.approx(A * np.sin(2 * np.pi * t / P) ** 2 - A / 2, abs=1e-12)
