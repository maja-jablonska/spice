"""Warnings for passbands the wavelength grid does not cover and for wavelengths outside an emulator's range."""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spice.spectrum import AB_passband_luminosity, Blackbody, WavelengthCoverageWarning, simulate_observed_flux
from spice.spectrum.filter import Bolometric, HipparcosHp, JohnsonCousinsB, JohnsonCousinsV, TychoVT
from spice.spectrum.wavelength_coverage import warn_if_outside_emulator_range, warn_if_passband_uncovered
from spice.models import IcosphereModel

HARPS = (3780.0, 6910.0)


def _coverage_warnings(fn, *args):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn(*args)
    return [w for w in caught if issubclass(w.category, WavelengthCoverageWarning)]


@pytest.mark.parametrize("flt, expected", [
    (TychoVT(), 1.0),
    (JohnsonCousinsV(), 0.998),
    (JohnsonCousinsB(), 0.992),
    (HipparcosHp(), 0.906),
])
def test_coverage_of_the_harps_range(flt, expected):
    assert flt.wavelength_coverage(np.linspace(*HARPS, 2000)) == pytest.approx(expected, abs=2e-3)


def test_full_grid_covers_every_passband():
    wl = np.linspace(1.0, 30000.0, 20000)
    for flt in (JohnsonCousinsB(), HipparcosHp(), Bolometric()):
        assert flt.wavelength_coverage(wl) == pytest.approx(1.0, abs=1e-9)


def test_partial_coverage_warns_and_full_coverage_does_not():
    wl_harps = jnp.linspace(*HARPS, 2000)
    flux = jnp.ones_like(wl_harps)
    assert _coverage_warnings(AB_passband_luminosity, HipparcosHp(), wl_harps, flux)
    assert _coverage_warnings(AB_passband_luminosity, JohnsonCousinsB(), wl_harps, flux)      # 0.8 % > tolerance
    assert not _coverage_warnings(AB_passband_luminosity, JohnsonCousinsV(), wl_harps, flux)  # 0.2 % < tolerance
    wl_wide = jnp.linspace(3000.0, 10000.0, 4000)
    assert not _coverage_warnings(AB_passband_luminosity, HipparcosHp(), wl_wide, jnp.ones_like(wl_wide))


def test_traced_wavelengths_are_skipped():
    flt = HipparcosHp()

    @jax.jit
    def f(wl):
        warn_if_passband_uncovered(flt, wl)
        return wl.sum()

    assert not _coverage_warnings(f, jnp.linspace(*HARPS, 100))


class _RangedBlackbody(Blackbody):
    @property
    def wavelength_range(self):
        return HARPS


def test_emulator_range_warning():
    emu = _RangedBlackbody()
    assert _coverage_warnings(warn_if_outside_emulator_range, emu.intensity, jnp.log10(jnp.linspace(3500.0, 6000.0, 50)))
    assert not _coverage_warnings(warn_if_outside_emulator_range, emu.intensity, jnp.log10(jnp.linspace(*HARPS, 50)))
    # an unrestricted emulator never warns
    assert not _coverage_warnings(warn_if_outside_emulator_range, Blackbody().intensity,
                                  jnp.log10(jnp.linspace(100.0, 1e5, 50)))


def test_simulate_observed_flux_warns_outside_the_emulator_range():
    emu = _RangedBlackbody()
    m = IcosphereModel.construct(100, 1.0, 1.0, emu.to_parameters(), emu.stellar_parameter_names)
    lw = jnp.log10(jnp.linspace(6800.0, 7200.0, 20))
    assert _coverage_warnings(simulate_observed_flux, emu.intensity, m, lw)


def test_bundle_wavelength_range_from_extras():
    from types import SimpleNamespace
    from spice.spectrum.aemu_spectrum_emulator import _bundle_wavelength_range
    log_grid = np.log10(np.linspace(*HARPS, 50)).tolist()
    assert _bundle_wavelength_range(SimpleNamespace(bundle_extras={"source_log10_wavelength": log_grid})) == \
        pytest.approx(HARPS)
    assert _bundle_wavelength_range(SimpleNamespace(bundle_extras={"wavelength_angstrom": [4000.0, 5000.0]})) == (4000.0, 5000.0)
    assert _bundle_wavelength_range(SimpleNamespace(bundle_extras={})) is None


def test_simulate_monochromatic_luminosity_warns_outside_the_emulator_range():
    # It was jitted as a whole, so the check saw tracers and never warned.
    from spice.spectrum import simulate_monochromatic_luminosity
    emu = _RangedBlackbody()
    m = IcosphereModel.construct(100, 1.0, 1.0, emu.to_parameters(), emu.stellar_parameter_names)
    lw = jnp.log10(jnp.linspace(6800.0, 7200.0, 20))
    assert _coverage_warnings(simulate_monochromatic_luminosity, emu.flux, m, lw)
    assert not _coverage_warnings(simulate_monochromatic_luminosity, emu.flux, m,
                                  jnp.log10(jnp.linspace(*HARPS, 20)))
