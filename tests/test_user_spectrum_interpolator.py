"""Exercise public parameter and limb-darkening inputs with a tiny local grid."""
import pickle
import warnings
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from spice.spectrum.user_spectrum_interpolator import UserSpectrumInterpolator


@pytest.fixture
def emulator(tmp_path):
    path = tmp_path / 'grid.pkl'
    with path.open('wb') as f:
        pickle.dump({'parameters': np.array([[5000., 4., 0.], [6000., 4., 0.]]),
                     'intensities': np.ones((2, 3)),
                     'log10_wavelengths': np.array([3.6, 3.7, 3.8])}, f)
    return UserSpectrumInterpolator(str(path), has_mu=False, has_continuum=False)


@pytest.mark.parametrize('convert', [list, tuple, np.asarray, jnp.asarray])
def test_parameter_arrays_and_sequences_are_accepted(emulator, convert):
    params = [5500., 4., 0.]
    np.testing.assert_allclose(emulator.to_parameters(convert(params)), params)
    rows = np.array([params, [5600., 4., 0.]])
    np.testing.assert_allclose(emulator.to_parameters(convert(rows)), rows)


def test_parameter_conversion_accepts_jit_tracers(emulator):
    params = jnp.array([5500., 4., 0.])
    np.testing.assert_array_equal(jax.jit(emulator.to_parameters)(params), params)


@pytest.mark.parametrize('values', [1., [1., 2.], [1., 2., 3., 4.], np.ones((1, 2)), np.ones((1, 1, 3))])
def test_invalid_parameter_shapes_are_rejected(emulator, values):
    with pytest.raises(ValueError, match='Parameters must'):
        emulator.to_parameters(values)


@pytest.mark.parametrize('alias', ['limb_darkening', 'limb_dark', 'limbdark', 'ld'])
@pytest.mark.parametrize('coefficient', [0., 1.])
def test_explicit_limb_darkening_is_preserved(emulator, alias, coefficient):
    params = {'teff': 5500., 'logg': 4., 'metallicity': 0., alias: coefficient}
    with warnings.catch_warnings(record=True) as caught:
        result = emulator.intensity(jnp.array([3.65, 3.75]), 0.5, params)
    assert not any('No limb darkening coefficients' in str(w.message) for w in caught)
    np.testing.assert_allclose(result, 1. - 0.5 * coefficient, rtol=1e-12)


def test_default_limb_darkening_still_applies(emulator):
    with pytest.warns(UserWarning, match='No limb darkening coefficients'):
        result = emulator.intensity(jnp.array([3.65]), 0.5, [5500., 4., 0.])
    np.testing.assert_allclose(result, 0.7)


@pytest.mark.parametrize('key', ['m/h', 'metallicity', 'metals/h', 'default_metals_h'])
def test_metallicity_dictionary_keys_retain_non_solar_values(emulator, key):
    actual = emulator.to_parameters({'teff': 5100., 'logg': 3.7, key: -1.5})
    np.testing.assert_array_equal(actual, [5100., 3.7, -1.5])


@pytest.mark.parametrize('has_mu', [False, True])
def test_wavelength_dependent_spectrum_and_continuum(tmp_path, has_mu):
    path = tmp_path/'sloped.pkl'
    parameters = np.array([[5000., 4., -1., 1.]]) if has_mu else np.array([[5000., 4., -1.]])
    with path.open('wb') as f:
        pickle.dump({'parameters': parameters, 'intensities': np.array([[10., 20., 30.]]),
                     'log10_wavelengths': np.array([3.6, 3.7, 3.8]),
                     'continuum_intensities': np.array([[100., 200., 300., 400.]]),
                     'continuum_wavelengths': np.array([3.55, 3.65, 3.75, 3.85])}, f)
    interp = UserSpectrumInterpolator(str(path), has_mu=has_mu)
    params = {'teff': 5000., 'logg': 4., 'm/h': -1., 'ld': 0.}
    query = jnp.array([3.6, 3.61, 3.65, 3.69, 3.7, 3.8])
    expected = np.column_stack([10.+100.*(np.asarray(query)-3.6),
                                100.+1000.*(np.asarray(query)-3.55)])
    result = jax.jit(lambda wavelengths: interp.intensity(wavelengths, 1., params))(query)
    np.testing.assert_allclose(result, expected, rtol=1e-8, atol=1e-7)
    derivative = jax.grad(lambda wavelength: interp.intensity(jnp.atleast_1d(wavelength), 1., params)[0, 0])(3.63)
    assert float(derivative) == pytest.approx(100., rel=1e-7)


def test_inconsistent_grid_parameter_dimensions_are_rejected(emulator):
    # The stored table has no mu column; claiming one must not truncate the query.
    interp = UserSpectrumInterpolator.__new__(UserSpectrumInterpolator)
    interp.__dict__.update(emulator.__dict__)
    interp._has_mu = True
    with pytest.raises(ValueError, match='grid parameter columns'):
        interp.intensity(jnp.array([3.65]), 0.5, jnp.array([5500., 4., 0.]))
