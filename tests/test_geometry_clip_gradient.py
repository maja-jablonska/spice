"""The Sutherland-Hodgman clipper must be differentiable in the polygon vertices.

``mesh_view`` differentiates eclipse light curves through ``clip`` (radii,
inclination, ...). The clipper pads polygons with nan rows and special-cases
vertical edges with ``lax.cond``; under ``vmap`` those conds become selects, so
the untaken branches are evaluated too, and their divisions by zero and
``nan * x`` products turned into nan cotangents that contaminated the real
vertices. The area of the intersection must have a finite gradient that matches
finite differences, for every overlap configuration, jitted and vmapped.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spice.geometry import clip
from spice.geometry.utils import polygon_area


def _area(subject, clipping):
    poly = clip(subject, clipping)
    ok = ~jnp.any(jnp.isnan(poly), axis=1)
    x = jnp.where(ok, poly[:, 0], 0.0); y = jnp.where(ok, poly[:, 1], 0.0)
    return polygon_area(x, y)


TRI_A = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
CASES = {
    "partial overlap": np.array([[1.0, -0.5], [3.0, 1.0], [1.0, 1.5]]),
    "clipping inside subject": np.array([[0.3, 0.3], [0.9, 0.3], [0.3, 0.9]]),
    "subject inside clipping": np.array([[-1.0, -1.0], [5.0, -1.0], [-1.0, 5.0]]),
    "crossing vertical edge": np.array([[0.5, -0.5], [2.5, 0.8], [0.5, 1.6]]),
    "disjoint": np.array([[3.0, 3.0], [4.0, 3.0], [3.0, 4.0]]),
}


@pytest.mark.parametrize("name", list(CASES))
def test_area_gradient_is_finite_and_matches_finite_differences(name):
    clipping = jnp.asarray(CASES[name])
    f = lambda s: _area(s, clipping)
    a = float(f(jnp.asarray(TRI_A)))
    g = np.asarray(jax.grad(f)(jnp.asarray(TRI_A)))
    assert np.isfinite(g).all(), f"{name}: nan/inf gradient {g}"
    if name == "disjoint":
        assert a == 0.0 and np.all(g == 0.0)
        return
    assert a > 0
    h = 1e-6
    for (i, k) in [(0, 0), (1, 0), (2, 1)]:
        e = np.zeros_like(TRI_A); e[i, k] = h
        fd = (float(f(jnp.asarray(TRI_A + e))) - float(f(jnp.asarray(TRI_A - e)))) / (2 * h)
        np.testing.assert_allclose(g[i, k], fd, atol=1e-5, err_msg=f"{name}: vertex {i} coord {k}")


def test_coincident_edge_is_a_kink_but_stays_finite():
    # the clipping polygon shares the subject's vertical edge x = 0: the area is not differentiable there
    # (one-sided slopes differ), so only finiteness is required
    clipping = jnp.asarray([[0.0, 0.5], [1.5, 0.5], [0.0, 1.5]])
    g = np.asarray(jax.grad(lambda s: _area(s, clipping))(jnp.asarray(TRI_A)))
    assert np.isfinite(g).all(), g


def test_vmapped_gradient_has_no_nan():
    clips = jnp.asarray(np.stack(list(CASES.values())))
    subjects = jnp.asarray(np.repeat(TRI_A[None], len(CASES), axis=0))
    total = lambda s: jnp.sum(jax.vmap(_area)(s, clips))
    g = np.asarray(jax.jit(jax.grad(total))(subjects))
    assert np.isfinite(g).all(), g
    g_single = np.stack([np.asarray(jax.grad(lambda s: _area(s, c))(jnp.asarray(TRI_A))) for c in clips])
    np.testing.assert_allclose(g, g_single, atol=1e-6)
