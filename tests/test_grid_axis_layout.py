"""Grid metadata, not coincidental array sizes, defines the lookup layout."""
import itertools
import numpy as np
import pandas as pd
import pytest
import jax.numpy as jnp
import zarr

from spice.spectrum.lazy_zarr_interpolator import (IntensityLazyZarrInterpolator,
                                                   FluxLazyZarrInterpolator)


def write_store(path, order, n_wavelengths, intensity=True, metadata=False):
    axes = ([5000., 6000.], [4., 4.5], [0., 1.]) if intensity else ([5000., 6000.], [4., 4.5])
    names = ['teff', 'logg', 'mu'] if intensity else ['teff', 'logg']
    rows = list(itertools.product(*axes))
    wavelengths = np.linspace(5000., 5006., n_wavelengths)
    flux = np.array([0.001*row[0] + 2*row[1] + (10*row[2] if intensity else 0.)
                     + 0.1*(wavelengths-5000.) for row in rows])
    group = zarr.open_group(str(path), mode='w')
    for name, array in [('wavelength', wavelengths), ('flux', flux), ('continuum', flux+100.)]:
        group.create_array(name, data=array)
    if metadata:
        group.create_array('param_names', data=np.asarray(order, dtype='U8'))
    frame = pd.DataFrame(rows, columns=names)
    frame['row_idx'] = np.arange(len(rows))
    frame.to_parquet(path/'index.parquet')


@pytest.mark.parametrize('order', [('mu', 'teff', 'logg'), ('teff', 'mu', 'logg'), ('teff', 'logg', 'mu')])
@pytest.mark.parametrize('in_memory', [False, True])
@pytest.mark.parametrize('sparse', [False, True])
def test_intensity_uses_grid_axis_order(tmp_path, order, in_memory, sparse):
    path = tmp_path/'axis.zarr'
    write_store(path, order, 5)
    emulator = IntensityLazyZarrInterpolator(str(path), params=list(order),
                                             in_memory=in_memory, sparse=sparse)
    result = emulator.intensity(jnp.log10(jnp.array([5001., 5004.])), 0.25, jnp.array([5500., 4.25]))
    np.testing.assert_allclose(result, [[16.6, 116.6], [16.9, 116.9]], atol=1e-10)
    assert emulator.parameter_names == ['teff', 'logg', 'mu']


@pytest.mark.parametrize('intensity', [False, True])
@pytest.mark.parametrize('in_memory', [False, True])
@pytest.mark.parametrize('sparse', [False, True])
def test_shared_wavelengths_when_model_and_wavelength_counts_match(tmp_path, intensity, in_memory, sparse):
    path = tmp_path/'square.zarr'
    order = ['teff', 'logg', 'mu'] if intensity else ['teff', 'logg']
    write_store(path, order, 8 if intensity else 4, intensity=intensity)
    cls = IntensityLazyZarrInterpolator if intensity else FluxLazyZarrInterpolator
    emulator = cls(str(path), params=order, in_memory=in_memory, sparse=sparse)
    wavelengths = jnp.log10(jnp.array([5001., 5004.]))
    parameters = jnp.array([5500., 4.25])
    if intensity:
        actual = emulator.intensity(wavelengths, 0.25, parameters)
        expected = [[16.6, 116.6], [16.9, 116.9]]
    else:
        actual = emulator.flux(wavelengths, parameters)
        expected = [[14.1, 114.1], [14.4, 114.4]]
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    assert 'wavelength' in emulator.static_data
    assert 'wavelength' not in emulator.rowwise_data


def test_intensity_uses_axis_order_from_stored_metadata(tmp_path):
    path = tmp_path/'metadata.zarr'
    write_store(path, ['mu', 'teff', 'logg'], 5, metadata=True)
    emulator = IntensityLazyZarrInterpolator(str(path), in_memory=True)
    actual = emulator.intensity(jnp.log10(jnp.array([5001.])), 0.25, jnp.array([5500., 4.25]))
    np.testing.assert_allclose(actual, [[16.6, 116.6]], atol=1e-10)
