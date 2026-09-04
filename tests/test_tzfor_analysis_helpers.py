"""Tests for the TZ For analysis helpers in tutorial/paper_results/tz_fornacis.

These are not library code, but every bug that cost real time in this analysis
lived here rather than in ``spice`` itself: a phase-offset window that could not
reach the true alignment, a cross-correlation run across a 1340 A gap, and the
ever-present risk of averaging two normalised spectra instead of blending fluxes
and continua. Each test below pins one of those.

The helpers are imported directly; none of them needs PHOEBE at module scope, so
these run in any environment.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

TZFOR = Path(__file__).resolve().parents[1] / "tutorial" / "paper_results" / "tz_fornacis"
sys.path.insert(0, str(TZFOR))

compare = pytest.importorskip("tzfor_compare_spectra")
fit_spec = pytest.importorskip("tzfor_fit_spectra")


# --------------------------------------------------------------------------
# Blending: (F1+F2)/(C1+C2), never the mean of two normalised spectra
# --------------------------------------------------------------------------

def _two_component_spectra(n=400, depth1=0.5, depth2=0.2, ratio=2.0):
    """Component 1 has a deep line, component 2 is ``ratio`` times brighter."""
    wl = np.linspace(5160.0, 5200.0, n)
    prof = np.exp(-0.5 * ((wl - 5180.0) / 0.4) ** 2)
    c1 = np.ones(n)
    c2 = np.full(n, ratio)
    f1 = c1 * (1 - depth1 * prof)
    f2 = c2 * (1 - depth2 * prof)
    return wl, {"primary": np.array([f1]), "primary_continuum": np.array([c1]),
                "secondary": np.array([f2]), "secondary_continuum": np.array([c2])}


def test_blend_is_flux_weighted_not_a_plain_mean():
    """The brighter star must dominate the blend in proportion to its flux."""
    wl, sp = _two_component_spectra(ratio=2.0)
    blend = compare.composite_normalized(sp, 0, wl, slice(None), broaden=False)
    naive = 0.5 * (sp["primary"][0] / sp["primary_continuum"][0]
                   + sp["secondary"][0] / sp["secondary_continuum"][0])
    # depth 0.5 and 0.2 with a 1:2 continuum ratio -> (0.5 + 2*0.2)/3 = 0.30
    assert 1 - blend.min() == pytest.approx(0.30, abs=0.01)
    assert 1 - naive.min() == pytest.approx(0.35, abs=0.01)
    assert not np.allclose(blend, naive), "blend collapsed to the naive mean"


def test_dilution_follows_the_light_ratio():
    """A fainter companion dilutes less; that dependence IS the light ratio."""
    depths = []
    for ratio in (0.25, 1.0, 4.0):
        wl, sp = _two_component_spectra(depth1=0.6, depth2=0.0, ratio=ratio)
        b = compare.composite_normalized(sp, 0, wl, slice(None), broaden=False)
        depths.append(1 - b.min())
    # a line-free companion only dilutes: more companion light -> shallower
    assert depths[0] > depths[1] > depths[2]
    assert depths[0] == pytest.approx(0.6 / 1.25, abs=0.01)
    assert depths[2] == pytest.approx(0.6 / 5.0, abs=0.01)


# --------------------------------------------------------------------------
# Broadening must redistribute a line, never change its strength
# --------------------------------------------------------------------------

def test_broadening_conserves_equivalent_width():
    wl = np.linspace(5160.0, 5200.0, 2000)
    flux = 1 - 0.5 * np.exp(-0.5 * ((wl - 5180.0) / 0.05) ** 2)
    out = compare._broaden(flux, 5180.0, float(wl[1] - wl[0]), vmacro_kms=6.0)
    ew_in = np.trapezoid(1 - flux, wl)
    ew_out = np.trapezoid(1 - out, wl)
    assert ew_out == pytest.approx(ew_in, rel=1e-3)
    assert out.min() > flux.min(), "broadening must make the core shallower"


def test_broadening_uses_per_component_macroturbulence():
    """Each star carries its own vmacro; a shared value is wrong."""
    wl, sp = _two_component_spectra()
    a = compare.composite_normalized(sp, 0, wl, slice(None), vmacro=(5.0, 6.0))
    b = compare.composite_normalized(sp, 0, wl, slice(None), vmacro=(6.0, 6.0))
    assert not np.allclose(a, b)


# --------------------------------------------------------------------------
# Doppler shift direction — a sign error here is invisible in a single epoch
# --------------------------------------------------------------------------

def test_positive_velocity_redshifts():
    wl = np.linspace(5160.0, 5200.0, 4000)
    flux = 1 - 0.5 * np.exp(-0.5 * ((wl - 5180.0) / 0.1) ** 2)
    red = fit_spec.doppler_shift(wl, flux, +30.0)
    blue = fit_spec.doppler_shift(wl, flux, -30.0)
    assert wl[np.argmin(red)] > 5180.0, "positive dv must move the line redward"
    assert wl[np.argmin(blue)] < 5180.0
    expected = 5180.0 * 30.0 / fit_spec.C_KMS
    assert wl[np.argmin(red)] - 5180.0 == pytest.approx(expected, abs=0.02)


def test_zero_shift_is_identity():
    wl = np.linspace(5160.0, 5200.0, 500)
    flux = np.random.default_rng(0).normal(1.0, 0.01, wl.size)
    assert np.array_equal(fit_spec.doppler_shift(wl, flux, 0.0), flux)


# --------------------------------------------------------------------------
# Cross-correlation: must find a real peak, and must say when it did not
# --------------------------------------------------------------------------

def test_velocity_offset_recovers_an_injected_shift():
    wl = np.linspace(6540.0, 6580.0, 3000)
    rng = np.random.default_rng(1)
    obs = 1 - 0.4 * np.exp(-0.5 * ((wl - 6562.8) / 0.3) ** 2)
    obs = obs + rng.normal(0, 0.002, wl.size)
    model = fit_spec.doppler_shift(wl, obs, -12.0)   # model sits blueward
    dv, cc, rail = compare.velocity_offset(wl, model, obs, max_shift_kms=60.0, n=481)
    assert dv == pytest.approx(12.0, abs=0.5), "did not recover the injected shift"
    assert cc > 0.9 and not rail


def test_velocity_offset_reports_railing():
    """An offset outside the search window must be flagged, not returned quietly.

    This is the failure that wasted a 245-model grid: the correct alignment lay
    outside the searched range, so the correlation slid to the bound and the
    number looked like a measurement.
    """
    wl = np.linspace(6540.0, 6580.0, 3000)
    obs = 1 - 0.4 * np.exp(-0.5 * ((wl - 6562.8) / 0.3) ** 2)
    model = fit_spec.doppler_shift(wl, obs, -80.0)
    dv, cc, rail = compare.velocity_offset(wl, model, obs, max_shift_kms=20.0, n=161)
    assert rail, "an unreachable optimum must set the rail flag"


# --------------------------------------------------------------------------
# Area weighting for the component radial velocity
# --------------------------------------------------------------------------

def test_component_rv_weights_by_visible_area():
    class FakeModel:
        los_velocities = np.array([-50.0, 0.0, +50.0])
        d_cast_areas = np.array([1.0, 0.0, 3.0])
    # (-50*1 + 50*3)/4 = +25
    assert fit_spec.component_rv(FakeModel()) == pytest.approx(25.0)


def test_component_rv_ignores_negative_areas():
    """Back-hemisphere elements arrive with non-positive projected area."""
    class FakeModel:
        los_velocities = np.array([-100.0, 10.0])
        d_cast_areas = np.array([-5.0, 2.0])
    assert fit_spec.component_rv(FakeModel()) == pytest.approx(10.0)
