"""Convolution must preserve shape and use real neighbours across chunk seams."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from spice.spectrum.utils import apply_spectral_resolution, apply_variable_resolution


def spectrum(n=97):
    x = jnp.linspace(3.6, 3.6006, n)
    # Put features on segment boundaries, including the short final segment.
    y = 1 - 0.5 * jnp.exp(-((jnp.arange(n) - 31.) / 1.5)**2)
    return x, y.at[-2].set(0.6)


@pytest.mark.parametrize('segment_size', [1, 32, 97, 128])
def test_constant_resolution_matches_whole_spectrum(segment_size):
    x, y = spectrum()
    expected = apply_spectral_resolution(x, y, 20000.)
    actual = apply_variable_resolution(x, y, jnp.full_like(y, 20000.), segment_size)
    assert actual.shape == y.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_variable_resolution_matches_independent_segment_convolution():
    x, y = spectrum()
    resolving_power = jnp.linspace(10000., 40000., len(y))
    result = np.asarray(apply_variable_resolution(x, y, resolving_power, segment_size=32))
    dx = float(x[1] - x[0])
    expected = np.empty(len(y))
    for start in range(0, len(y), 32):
        end = min(start + 32, len(y))
        R = float(resolving_power[start:end].mean())
        sigma = 1. / (2 * np.sqrt(2 * np.log(2)) * R * np.log(10))
        width = min(int(np.floor(4 * sigma / dx)), 2046)
        offset = np.arange(-width, width + 1)
        kernel = np.exp(-0.5 * (offset * dx / sigma)**2)
        kernel /= kernel.sum()
        padded = np.pad(np.asarray(y), (width, width), constant_values=float(y.mean()))
        full = np.convolve(padded, kernel, mode='valid')
        expected[start:end] = full[start:end]
    np.testing.assert_allclose(result, expected, rtol=1e-9, atol=1e-10)


def test_resolution_is_jittable_and_differentiable():
    x, y = spectrum(35)
    fn = jax.jit(lambda values: apply_variable_resolution(x, values, jnp.full_like(values, 20000.), 16))
    result = fn(y)
    gradient = jax.grad(lambda values: fn(values).sum())(y)
    assert result.shape == y.shape
    assert np.all(np.isfinite(gradient))
    assert np.any(np.asarray(gradient) != 0)


@pytest.mark.parametrize('size', [0, -2])
def test_invalid_segment_size_is_rejected(size):
    x, y = spectrum()
    with pytest.raises(ValueError, match='positive static integer'):
        apply_variable_resolution(x, y, jnp.ones_like(y), size)
