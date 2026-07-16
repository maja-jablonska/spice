"""Tests for :class:`spice.spectrum.grid_node_interpolation.GridNodeInterpolatedEmulator`.

Uses a mock emulator so the tests run without the optional ``aemu`` extra.
The mock deliberately behaves like the memorizing bundle this wrapper exists
for: its output depends on the *nearest grid node* of teff, not teff itself,
so any smooth teff response in the wrapped emulator must come from the
wrapper's own interpolation.
"""
import jax
import numpy as np
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from spice.spectrum.grid_node_interpolation import GridNodeInterpolatedEmulator


class MockNodeEmulator:
    """Emulates a bundle that memorized a 250 K teff grid.

    intensity = (nearest node / 1000)^2 * (1 + mu) at every wavelength, in
    two channels scaled 1x and 2x so channel structure is preserved.
    """

    stellar_parameter_names = ["teff", "logg"]
    min_stellar_parameters = np.array([4000.0, 0.0])
    max_stellar_parameters = np.array([8000.0, 5.0])

    def intensity(self, log_wavelengths, mu, parameters):
        teff = parameters[0]
        node = 250.0 * jnp.round(teff / 250.0)
        base = (node / 1000.0) ** 2 * (1.0 + mu)
        ones = jnp.ones((jnp.shape(log_wavelengths)[0],))
        return jnp.stack([base * ones, 2.0 * base * ones], axis=-1)


@pytest.fixture
def wrapped():
    return GridNodeInterpolatedEmulator(MockNodeEmulator(), parameter="teff",
                                        spacing=250.0, origin=0.0)


LOG_WL = jnp.log10(jnp.linspace(5000.0, 5010.0, 7))


class TestGridNodeInterpolatedEmulator:
    def test_exact_at_nodes(self, wrapped):
        raw = MockNodeEmulator()
        for teff in (5750.0, 6000.0, 6250.0):
            p = jnp.array([teff, 1.8])
            assert jnp.allclose(wrapped.intensity(LOG_WL, 0.7, p),
                                raw.intensity(LOG_WL, 0.7, p))

    def test_log_linear_between_nodes(self, wrapped):
        # Halfway between 5750 and 6000 the wrapper must return the geometric
        # mean of the node intensities (linear in log10).
        p_mid = jnp.array([5875.0, 1.8])
        raw = MockNodeEmulator()
        i0 = raw.intensity(LOG_WL, 1.0, jnp.array([5750.0, 1.8]))
        i1 = raw.intensity(LOG_WL, 1.0, jnp.array([6000.0, 1.8]))
        expected = jnp.sqrt(i0 * i1)
        assert jnp.allclose(wrapped.intensity(LOG_WL, 1.0, p_mid), expected,
                            rtol=1e-10)

    def test_smooth_in_teff(self, wrapped):
        # The mock steps at node midpoints; the wrapper must not.
        teffs = np.linspace(5750.0, 6000.0, 26)
        vals = np.array([
            float(wrapped.intensity(LOG_WL, 1.0, jnp.array([t, 1.8]))[0, 0])
            for t in teffs
        ])
        steps = np.diff(vals)
        assert np.all(steps > 0), "interpolated intensity should be monotone here"
        # No step may exceed 3x the median step (the mock's raw output has one
        # giant jump at 5875 and zeros elsewhere, which would fail this).
        assert steps.max() < 3.0 * np.median(steps)

    def test_extrapolates_from_nearest_inbounds_nodes(self, wrapped):
        # Below the lowest node the wrapper extrapolates from the first
        # bracket (4000, 4250) rather than indexing out of bounds.
        val = wrapped.intensity(LOG_WL, 1.0, jnp.array([3990.0, 1.8]))
        assert bool(jnp.all(jnp.isfinite(val)))

    def test_delegates_other_attributes(self, wrapped):
        assert wrapped.stellar_parameter_names == ["teff", "logg"]
        assert float(wrapped.max_stellar_parameters[0]) == 8000.0

    def test_unknown_parameter_raises(self):
        with pytest.raises(ValueError):
            GridNodeInterpolatedEmulator(MockNodeEmulator(), parameter="vmicro")

    def test_works_under_vmap(self, wrapped):
        # simulate_observed_flux vmaps intensity over (wavelength-rows, mus,
        # parameter-rows); the wrapper must trace cleanly.
        params = jnp.stack([jnp.array([5800.0, 1.8]), jnp.array([5900.0, 1.8])])
        mus = jnp.array([0.5, 1.0])
        wls = jnp.stack([LOG_WL, LOG_WL])
        out = jax.vmap(wrapped.intensity, in_axes=(0, 0, 0))(wls, mus, params)
        assert out.shape == (2, LOG_WL.shape[0], 2)
        assert bool(jnp.all(jnp.isfinite(out)))
