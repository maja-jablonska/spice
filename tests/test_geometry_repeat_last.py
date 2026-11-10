"""``repeat_last`` pads a clipped polygon's nan rows with its last valid vertex.

It used to do so with a ``fori_loop`` whose trip count depended on the data,
which JAX cannot reverse-differentiate -- so nothing downstream of the
occlusion (eclipse depths as a function of radii or inclination) had a
gradient. The static form must give identical results and be differentiable.
"""
import jax
import jax.numpy as jnp
import numpy as np

from spice.geometry.utils import repeat_last, last_non_nan, append_to_last_nan, polygon_area


def _loop_version(arr):
    return jax.lax.cond(jnp.any(jnp.all(jnp.isnan(arr), axis=1)),
                        lambda: jax.lax.fori_loop(0, jnp.sum(jnp.all(jnp.isnan(arr), axis=1)),
                                                  lambda i, a: append_to_last_nan(a, last_non_nan(a)), arr),
                        lambda: arr)


def _polygon(n_valid, n_total=8, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.full((n_total, 2), np.nan)
    arr[:n_valid] = rng.normal(size=(n_valid, 2))
    return jnp.asarray(arr)


def test_matches_the_former_loop():
    for n_valid in (1, 3, 5, 8):
        arr = _polygon(n_valid)
        np.testing.assert_array_equal(np.asarray(repeat_last(arr)), np.asarray(_loop_version(arr)))
        assert not np.isnan(np.asarray(repeat_last(arr))).any()


def test_no_nan_rows_is_identity():
    arr = _polygon(8)
    np.testing.assert_array_equal(np.asarray(repeat_last(arr)), np.asarray(arr))


def test_gradient_flows_through_padding():
    valid = _polygon(4)[:4]
    pad = jnp.full((4, 2), jnp.nan)                # as in the clipper: the padding rows are constants

    def area_of_scaled(scale):
        poly = repeat_last(jnp.concatenate([valid * scale, pad]))
        return polygon_area(poly[:, 0], poly[:, 1])
    g = jax.grad(area_of_scaled)(1.0)
    fd = (area_of_scaled(1.0 + 1e-6) - area_of_scaled(1.0 - 1e-6)) / 2e-6
    assert np.isfinite(float(g))
    np.testing.assert_allclose(g, fd, rtol=1e-5)
