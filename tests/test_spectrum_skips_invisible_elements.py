"""``simulate_observed_flux`` must not spend ``intensity_fn`` evaluations on
elements with zero projected area.

Half of any single star's mesh faces away from the observer. Those elements
have ``visible_cast_areas == 0`` and used to be pushed through the intensity
function anyway -- for an emulator that is half the cost of every synthesis.
The fix orders visible elements first and skips all-zero chunks with
``lax.cond``; these tests pin both the saving and that the flux is unchanged.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np

from spice.models import IcosphereModel
from spice.spectrum.spectrum import simulate_observed_flux


def _counting_intensity(counter):
    def intensity(log_wavelengths, mu, parameters):
        # Runs once per evaluated chunk (vmap batches the callback); never runs
        # inside a branch lax.cond did not take.
        jax.debug.callback(lambda _: counter.append(1), mu)
        val = jnp.broadcast_to(mu, log_wavelengths.shape) * parameters[0]
        return jnp.stack([val, val], axis=-1)
    return intensity


def _mesh():
    return IcosphereModel.construct(1000, 1.0, 1.0, jnp.array([2.0]), ["teff"])


def test_far_side_chunks_are_not_evaluated():
    m = _mesh()
    n = m.mus.shape[0]
    chunk = 64
    n_visible = int(np.sum(np.asarray(m.visible_cast_areas) > 0))
    assert 0 < n_visible < n, "a sphere must have a far side for this test to mean anything"

    counter = []
    jax.block_until_ready(simulate_observed_flux(_counting_intensity(counter), m,
                                                 jnp.linspace(3.6, 3.7, 8), chunk_size=chunk))
    all_chunks = math.ceil(n / chunk)
    needed_chunks = math.ceil(n_visible / chunk)
    # JAX may invoke a vmapped debug callback once per chunk or once per
    # element; either way the count must reflect only the non-empty chunks.
    assert len(counter) in (needed_chunks, needed_chunks * chunk), (len(counter), needed_chunks, chunk)
    assert len(counter) < all_chunks * chunk
    assert len(counter) != all_chunks


def test_flux_unchanged_by_reordering():
    m = _mesh()
    log_wl = jnp.linspace(3.6, 3.7, 8)
    flux = simulate_observed_flux(_counting_intensity([]), m, log_wl, chunk_size=64,
                                  disable_doppler_shift=True)
    areas = np.asarray(m.visible_cast_areas)
    mus = np.where(np.asarray(m.mus) > 0, np.asarray(m.mus), 0.0)
    expected = np.sum(areas * mus * np.asarray(m.parameters)[:, 0]) * 5.08326693599739e-16 / 100.0
    np.testing.assert_allclose(np.asarray(flux), expected, rtol=1e-10)


def test_gradient_flows_through_skipped_layout():
    m = _mesh()
    log_wl = jnp.linspace(3.6, 3.7, 4)

    def total(scale):
        mm = m._replace(parameters=m.parameters * scale)
        return jnp.sum(simulate_observed_flux(_counting_intensity([]), mm, log_wl,
                                              chunk_size=64, disable_doppler_shift=True))
    g = jax.grad(total)(1.0)
    fd = (total(1.0 + 1e-6) - total(1.0 - 1e-6)) / 2e-6
    np.testing.assert_allclose(g, fd, rtol=1e-5)
