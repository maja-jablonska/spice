"""Absolute calibrations catch spectral-density errors hidden by flux ratios."""
import jax.numpy as jnp
import numpy as np
import pytest

from spice.models import IcosphereModel
from spice.spectrum.filter import Filter, GaiaG, GaiaBP, GaiaRP, PANSTARRS_PS1_g, TWOMASSJ
from spice.spectrum.spectrum import (
    AB_passband_luminosity, ST_passband_luminosity,
    Vega_passband_luminosity, luminosity, simulate_monochromatic_luminosity,
)


@pytest.fixture
def wavelengths():
    return jnp.linspace(3000.0, 15000.0, 4001)


@pytest.fixture(params=[False, True, 'panstarrs'])
def response_filter(request):
    # A nonconstant response exercises both photon and energy weighting.
    return Filter(
        jnp.array([[4000., 4500., 6000., 7000.], [0., 0.8, 1., 0.]]),
        name='PANSTARRS test' if request.param == 'panstarrs' else 'test',
        non_photonic=bool(request.param),
    )


def test_ab_reference_spectrum_has_zero_magnitude(wavelengths, response_filter):
    # F_nu = 3631 Jy, converted to erg/s/cm^2/Angstrom.
    flux = 3.631e-20 * 2.99792458e18 / wavelengths**2
    assert float(AB_passband_luminosity(response_filter, wavelengths, flux)) == pytest.approx(0., abs=1e-10)
    assert float(AB_passband_luminosity(response_filter, wavelengths, flux * 0.01)) == pytest.approx(5., abs=1e-10)


def test_st_reference_spectrum_has_zero_magnitude(wavelengths, response_filter):
    flux = jnp.full_like(wavelengths, 10.**(-21.10 / 2.5))
    assert float(ST_passband_luminosity(response_filter, wavelengths, flux)) == pytest.approx(0., abs=1e-10)


@pytest.mark.parametrize('non_photonic', [False, True])
def test_vega_zero_point_uses_flux_per_angstrom(wavelengths, non_photonic):
    flux0 = 3.5e-9
    filt = Filter(jnp.array([[4000., 7000.], [1., 1.]]), name='test',
                  non_photonic=non_photonic, Vega_zeropoint=2.5 * np.log10(flux0))
    flux = jnp.full_like(wavelengths, flux0)
    assert float(Vega_passband_luminosity(filt, wavelengths, flux)) == pytest.approx(0., abs=1e-10)


@pytest.mark.parametrize('factory', [GaiaG, GaiaBP, GaiaRP])
@pytest.mark.parametrize('system', ['AB', 'Vega'])
def test_gaia_matches_photon_count_rate(wavelengths, factory, system):
    filt = factory()
    wl = np.asarray(wavelengths)
    flux = 3.5e-9 * (5500. / wl)**2
    response = np.asarray(filt.filter_responses_for_wavelengths(wavelengths))
    # Telescope area in cm^2; photon energy = h*c/lambda in erg.
    photon_energy = 6.62607015e-27 * 2.99792458e18 / wl
    count_rate = 7278. * np.trapezoid(flux * response / photon_energy, wl)
    zero_point = getattr(filt, system + '_zeropoint')
    expected = -2.5 * np.log10(count_rate) + zero_point
    fn = AB_passband_luminosity if system == 'AB' else Vega_passband_luminosity
    # The legacy Gaia count-rate coefficient differs from exact SI constants
    # by ~2e-8 mag; that rounding is unrelated to spectral-density conversion.
    assert float(fn(filt, wavelengths, jnp.asarray(flux))) == pytest.approx(expected, abs=1e-7)


@pytest.mark.parametrize('factory', [PANSTARRS_PS1_g, TWOMASSJ])
def test_energy_filters_use_ab_reference(wavelengths, factory):
    filt = factory()
    flux = 3.631e-20 * 2.99792458e18 / wavelengths**2
    assert float(AB_passband_luminosity(filt, wavelengths, flux)) == pytest.approx(0., abs=1e-10)


def test_luminosity_integrates_per_angstrom_density():
    mesh = IcosphereModel.construct(100, 2., 1., [5777.], ['teff'])
    wl = jnp.linspace(4000., 7000., 21)

    def unit_surface_flux(log_wl, parameters):
        return jnp.ones((len(log_wl), 2))

    # Unit surface flux integrated over a 3000-A band and the physical area.
    expected = float(mesh.areas.sum()) * (2. * 6.957e10)**2 * 3000.
    actual = luminosity(unit_surface_flux, mesh, wl)
    mono = simulate_monochromatic_luminosity(unit_surface_flux, mesh, jnp.log10(wl))
    assert float(actual) == pytest.approx(expected, rel=1e-12)
    assert float(actual) == pytest.approx(float(jnp.trapezoid(mono[:, 0], wl)), rel=1e-12)
