"""Regression tests for the 2026-11 photometry audit.

* AB_passband_luminosity must return AB magnitudes for every filter: a flat
  f_nu = 3631 Jy source is 0 in every band. Johnson-Cousins, Hipparcos/Tycho and
  Stromgren used to fold Bessell & Murphy (2012) standard-system offsets into
  the AB zero point (U came out -0.761).
* Those offsets are exposed as Vega_passband_luminosity = AB + offset.
* PanSTARRS PS1 curves are photon-counting capture cross-sections; they were
  integrated as energy responses (+18 mmag in g at 5772 K).
* 2MASS Vega zero points are the Cohen et al. (2003) zero-magnitude fluxes.
* The public filter_responses helper is zero outside the tabulated curve.
"""
import inspect

import jax.numpy as jnp
import numpy as np
import pytest

import spice.spectrum.filter as filters
from spice.spectrum.filter import Filter
from spice.spectrum.spectrum import (AB_passband_luminosity, Vega_passband_luminosity,
                                     filter_responses)

C_A_S = 2.99792458e18  # speed of light in Angstrom/s
H_CGS = 6.62607015e-27
K_CGS = 1.380649e-16

BUILTIN = [cls for _, cls in inspect.getmembers(filters, inspect.isclass)
           if issubclass(cls, Filter) and cls is not Filter and cls.__module__ == filters.__name__]
BESSELL_MURPHY = {
    "JohnsonCousinsU": -0.761, "JohnsonCousinsB": 0.130, "JohnsonCousinsV": 0.020,
    "JohnsonCousinsR": -0.163, "JohnsonCousinsI": -0.404, "HipparcosHp": 0.030,
    "TychoBT": 0.100, "TychoVT": 0.037, "Stromgrenu": 0.308, "Stromgrenv": 0.327,
    "Stromgrenb": 0.187, "Stromgreny": 0.027,
}


def _grid_for(f: Filter, n: int = 20000):
    wl = np.asarray(f.transmission_curve_wavelengths[0], float)
    lo, hi = wl.min(), wl.max()
    pad = 0.01 * (hi - lo)
    return jnp.linspace(max(lo - pad, 1.0), hi + pad, n)


def _planck_flambda(wl_A, teff):
    wl = wl_A * 1e-8
    b = 2 * H_CGS * (C_A_S * 1e-8) ** 2 / wl ** 5 / np.expm1(H_CGS * C_A_S * 1e-8 / (wl * K_CGS * teff))
    return b * 1e-8  # per Angstrom


def _instances():
    out = []
    for cls in BUILTIN:
        try:
            out.append(cls())
        except Exception:  # optional data files missing
            pass
    return out


@pytest.mark.parametrize("f", _instances(), ids=lambda f: f.name)
def test_flat_3631_jy_source_is_zero_ab(f):
    wl = _grid_for(f)
    flat_fnu = 3631e-23 * C_A_S / wl ** 2  # erg/s/cm^2/A
    assert abs(float(AB_passband_luminosity(f, wl, flat_fnu))) < 2e-3


@pytest.mark.parametrize("name,offset", sorted(BESSELL_MURPHY.items()))
def test_standard_system_magnitude_is_ab_plus_bessell_murphy_offset(name, offset):
    f = getattr(filters, name)()
    assert f.AB_zeropoint == pytest.approx(-22.40788262039795)
    wl = _grid_for(f)
    flux = jnp.asarray(_planck_flambda(np.asarray(wl), 5772.0)) * 1e-20
    ab = float(AB_passband_luminosity(f, wl, flux))
    vega = float(Vega_passband_luminosity(f, wl, flux))
    assert vega - ab == pytest.approx(offset, abs=1e-9)


@pytest.mark.parametrize("band", ["g", "r", "i", "z", "y", "w"])
def test_panstarrs_uses_photon_counting(band):
    f = getattr(filters, f"PANSTARRS_PS1_{band}")()
    assert not f.non_photonic
    wl = np.asarray(_grid_for(f))
    flux = _planck_flambda(wl, 5772.0) * 1e-20
    t = np.asarray(f.filter_responses_for_wavelengths(jnp.asarray(wl)))
    # Photon-counting AB: <f_nu> = int f_lambda T lambda / int T c / lambda
    fnu = np.trapezoid(flux * t * wl, wl) / np.trapezoid(t * C_A_S / wl, wl)
    expected = -2.5 * np.log10(fnu) - 48.60
    got = float(AB_passband_luminosity(f, jnp.asarray(wl), jnp.asarray(flux)))
    assert got == pytest.approx(expected, abs=2e-3)  # 48.60 vs 3631 Jy rounding


@pytest.mark.parametrize("name,f0", [("TWOMASSJ", 3.129e-10), ("TWOMASSH", 1.133e-10),
                                     ("TWOMASSK", 4.283e-11)])
def test_2mass_zero_magnitude_flux_is_cohen_2003(name, f0):
    f = getattr(filters, name)()
    wl = _grid_for(f)
    assert float(Vega_passband_luminosity(f, wl, jnp.full_like(wl, f0))) == pytest.approx(0.0, abs=1e-9)


def test_public_filter_responses_are_zero_outside_the_curve():
    resp = filter_responses(jnp.array([0.5, 5e4, 1e6]), jnp.array([1., 30000.]), jnp.array([1., 1.]))
    np.testing.assert_array_equal(np.asarray(resp), [0., 0., 0.])
