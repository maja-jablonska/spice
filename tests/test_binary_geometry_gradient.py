"""Eclipse light curves must be differentiable in the binary geometry, end to end.

``test_geometry_clip_gradient.py`` covers the Sutherland-Hodgman clipper on its own;
this test covers the whole chain a free-geometry fit differentiates: icosphere
construction with a traced radius, the Keplerian orbit with a traced inclination,
occlusion (neighbour search + clipping) and the visible cast areas. Three
successive bugs broke it (a concrete-value requirement, a dynamic ``fori_loop``
in ``repeat_last``, nan cotangents from the clipper's padding rows), the last of
which gave nan gradients while finite differences were finite.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spice.models import IcosphereModel
from spice.models.binary import Binary, _evaluate_orbit, add_orbit

N_VERT = 300
NAMES = ["teff", "logg"]
R1, M1, M2 = 8.3, 2.05, 1.95                  # TZ For-like, R_sun / M_sun
R2_0, INCL_0, P_YR = 3.9, 85.7, 75.67 / 365.25


def _bodies(R2):
    return (IcosphereModel.construct(N_VERT, R1, M1, jnp.array([4900.0, 2.9]), NAMES),
            IcosphereModel.construct(N_VERT, R2, M2, jnp.array([6400.0, 3.5]), NAMES))


@pytest.fixture(scope="module")
def neighbours():
    # from_bodies needs concrete geometry, so the occlusion neighbour counts are fixed up front (as the fits do)
    ref = Binary.from_bodies(*_bodies(R2_0))
    return min(64, int(1.5 * int(ref.n_neighbours1)) + 1), min(64, int(1.5 * int(ref.n_neighbours2)) + 1)


def _light(R2, incl, phase, nn):
    b1, b2 = _bodies(R2)
    bn = Binary(b1, b2, 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., jnp.zeros_like(b1.centers), jnp.zeros_like(b2.centers),
                jnp.zeros_like(b1.velocities), jnp.zeros_like(b2.velocities), *nn)
    bn = add_orbit(bn, P_YR, 0.0, 0.0, jnp.deg2rad(incl), 0.0, 0.0, 0.0, 0.0, 0.0, 100)
    m1, m2 = _evaluate_orbit(bn, phase * P_YR, n_neighbors1=nn[0], n_neighbors2=nn[1])
    proj = lambda m: jnp.sum(m.visible_cast_areas * jnp.where(m.mus > 0, m.mus, 0.0))
    return proj(m1) + proj(m2)


@pytest.mark.parametrize("phase", [0.74, 0.75])        # early ingress (0.2% deep: partial clipping) and mid-eclipse
def test_eclipse_gradient_is_finite_and_matches_finite_differences(phase, neighbours):
    f = lambda R2, incl: _light(R2, incl, phase, neighbours)
    out = float(f(R2_0, INCL_0)); off = float(_light(R2_0, INCL_0, 0.5, neighbours))
    assert out < 0.999 * off, "phase is not in eclipse: the test would not exercise the occlusion"
    g = jax.grad(f, argnums=(0, 1))(R2_0, INCL_0)
    assert np.isfinite(float(g[0])) and np.isfinite(float(g[1])), g
    h = 1e-3
    fd_r = (float(f(R2_0 + h, INCL_0)) - float(f(R2_0 - h, INCL_0))) / (2 * h)
    fd_i = (float(f(R2_0, INCL_0 + h)) - float(f(R2_0, INCL_0 - h))) / (2 * h)
    np.testing.assert_allclose(float(g[0]), fd_r, rtol=1e-3)
    np.testing.assert_allclose(float(g[1]), fd_i, rtol=1e-3)
