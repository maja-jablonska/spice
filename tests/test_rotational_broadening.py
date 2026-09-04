"""Rotation must actually reach the line-of-sight velocity field.

``add_rotation`` used to store only the scalar rotation velocity and axis; the
per-element ``rotation_velocities`` field that produces Doppler broadening was
filled solely by ``evaluate_rotation``. ``binary.py`` never calls that, so every
binary synthesis silently lost v sin i -- measured on a 38 km/s star, the line
FWHM stayed at the non-rotating 16.19 km/s instead of broadening to 64.20 km/s.
The HARPS fit of TZ For independently demanded ~44 km/s of unexplained extra
broadening for its 38 km/s component, which is how this was found.

These tests pin the velocity field rather than a synthesised profile, so they
need no emulator, no network and no grid files.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from spice.models.mesh_model import IcosphereModel
from spice.models.mesh_transform import add_rotation, evaluate_rotation
from spice.models.mesh_view import get_mesh_view

# Equator-on: the default rotation axis is [0, 1, 0] and this line of sight lies
# perpendicular to it, so the projected velocity spans the full +/- v sin i.
LOS = jnp.array([0.0, 0.0, -1.0])
V_ROT = 38.0  # km/s, the TZ For secondary


def _star(v_rot, radius=3.94, mass=1.958):
    mesh = IcosphereModel.construct(
        1000, radius, mass, jnp.array([6650.0, 3.35]), ["teff", "logg"]
    )
    return get_mesh_view(add_rotation(mesh, rotation_velocity=v_rot), LOS)


def test_add_rotation_populates_velocity_field():
    """add_rotation alone must fill rotation_velocities, not just the scalar."""
    still = _star(0.0)
    spinning = _star(V_ROT)
    assert float(jnp.max(jnp.abs(still.rotation_velocities))) == pytest.approx(0.0, abs=1e-9)
    assert float(jnp.max(jnp.abs(spinning.rotation_velocities))) > 0.5 * V_ROT, (
        "add_rotation did not build the per-element velocity field; a mesh that "
        "looks configured would synthesize with no rotational broadening"
    )


def test_los_velocity_spans_vsini():
    """Equator-on, the projected velocity must span about +/- v sin i."""
    spinning = _star(V_ROT)
    los_v = np.asarray(spinning.los_velocities)
    visible = np.asarray(spinning.mus) > 0
    assert los_v[visible].max() == pytest.approx(V_ROT, rel=0.15), (
        f"max LOS velocity {los_v[visible].max():.1f} km/s, expected ~+{V_ROT}"
    )
    assert los_v[visible].min() == pytest.approx(-V_ROT, rel=0.15), (
        f"min LOS velocity {los_v[visible].min():.1f} km/s, expected ~-{V_ROT}"
    )


def test_rotation_survives_into_a_binary():
    """The binary path must not lose the velocity field.

    binary.py never calls evaluate_rotation, so if add_rotation does not fill the
    field the components reach synthesis unbroadened -- exactly what happened to
    every TZ For spectrum.
    """
    from spice.models.binary import Binary

    body1 = _star(5.5, radius=8.28, mass=2.057)
    body2 = _star(V_ROT)
    binary = Binary.from_bodies(body1, body2, n_neighbours1=24, n_neighbours2=24)
    for body, v in ((binary.body1, 5.5), (binary.body2, V_ROT)):
        peak = float(jnp.max(jnp.abs(body.rotation_velocities)))
        assert peak > 0.5 * v, (
            f"binary component lost its rotation field (peak {peak:.2f} km/s "
            f"for v sin i = {v})"
        )


def test_evaluate_rotation_after_add_rotation_is_consistent():
    """Calling evaluate_rotation afterwards must stay valid (t=0 is a no-op)."""
    spinning = _star(V_ROT)
    again = evaluate_rotation(spinning, 0.0)
    np.testing.assert_allclose(
        np.asarray(spinning.rotation_velocities),
        np.asarray(again.rotation_velocities),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(spinning.d_centers), np.asarray(again.d_centers), rtol=1e-10, atol=1e-10
    )
