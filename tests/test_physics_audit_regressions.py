"""Regression tests for the 2026-11 physics audit.

Each test pins a bug that produced plausible-looking wrong results:

1. transits: the occlusion neighbour search was capped at 256 occluder faces
   per occluded face, so a small finely meshed body in front of a coarse star
   lost most of its faces (Rp/R = 0.1: 62% too shallow);
2. non-radial pulsation modes were symmetric about +z whatever the rotation
   axis (default +y), i.e. perpendicular to the spin;
3. evaluate_rotation left pulsation displacements/velocities pointing in their
   old directions;
4. spot (theta, phi) and inclination_to_los_axis assumed a +z rotation axis;
5. the m != 0 horizontal basis was ~10% too large (pole regulariser), and
   negative m / l > 10 were accepted silently.
"""
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from spice.models.binary import _choose_occlusion_search
from spice.models.mesh_model import IcosphereModel
from spice.models.mesh_transform import (add_pulsation, add_rotation, evaluate_pulsations,
                                         evaluate_rotation)
from spice.models.mesh_view import resolve_occlusion
from spice.models.spots import add_spot
from spice.models.utils import (inclination_to_los_axis, lat_to_theta, rotation_axis_frame,
                                spherical_harmonic_gradient_with_tilt, vertex_to_polar)

LOS = jnp.array([0.0, 0.0, -1.0])


def _star(n_vertices=1000, radius=1.0, mass=1.0):
    return IcosphereModel.construct(n_vertices, radius, mass, jnp.array([5000.0, 4.0]), ["teff", "logg"])


def _angle_deg(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


# --- 1. transit occlusion -----------------------------------------------------

def test_transit_of_small_fine_body_has_the_right_depth():
    star = _star(100, 1.0)._replace(los_vector=LOS)
    planet = _star(5000, 0.1, 0.001)._replace(center=jnp.array([0.0, 0.0, 50.0]), los_vector=LOS)
    k, from_occluder = _choose_occlusion_search(star, planet)
    assert from_occluder, "a fine small occluder should be searched from its own faces"
    resolved = resolve_occlusion(star, planet, n_neighbors=k, from_occluder=from_occluder)

    blocked = float(jnp.sum(resolved.occluded_areas))
    # Exact geometric depth for a disc of radius Rp centred on the stellar disc.
    np.testing.assert_allclose(blocked, np.pi * 0.1 ** 2, rtol=5e-3)


def test_both_search_directions_agree_for_similar_meshes():
    eclipsed = _star(1000, 3.94)._replace(los_vector=LOS)
    occulter = _star(1000, 8.28, 2.0)._replace(center=jnp.array([6.0, 0.0, 50.0]), los_vector=LOS)
    a = resolve_occlusion(eclipsed, occulter, n_neighbors=48)
    b = resolve_occlusion(eclipsed, occulter, n_neighbors=48, from_occluder=True)
    np.testing.assert_allclose(np.asarray(a.occluded_areas), np.asarray(b.occluded_areas), atol=1e-10)


# --- shared frame -------------------------------------------------------------

@pytest.mark.parametrize("axis", [[0., 0., 1.], [0., 1., 0.], [1., 2., -0.5], [0., 0., -1.]])
def test_rotation_axis_frame_is_orthonormal_with_axis_as_pole(axis):
    B = np.asarray(rotation_axis_frame(jnp.array(axis)))
    np.testing.assert_allclose(B.T @ B, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(B[:, 2], np.asarray(axis) / np.linalg.norm(axis), atol=1e-12)
    if axis == [0., 0., 1.]:
        np.testing.assert_array_equal(B, np.eye(3))  # z-axis results stay identical


# --- 2. pulsation symmetry axis ----------------------------------------------

def test_axisymmetric_mode_follows_the_rotation_axis():
    star = add_rotation(_star(2000), 20.0)  # default axis is +y
    axis = np.asarray(star.rotation_axis)
    pulsating = evaluate_pulsations(
        add_pulsation(star, 0, 2, 1.0, jnp.array([[[0.05, 0.]], [[0., 0.]], [[0., 0.]]])), 0.0)

    centers = np.asarray(star.d_centers)
    radial = np.einsum("ij,ij->i", np.asarray(pulsating.center_pulsation_offsets),
                       centers / np.linalg.norm(centers, axis=1, keepdims=True))
    cos_to_axis = centers @ axis / np.linalg.norm(centers, axis=1)
    pole = np.abs(cos_to_axis) > 0.98
    equator = np.abs(cos_to_axis) < 0.05
    # Y_2^0 is +1 at the poles and -1/2 on the equator of its own axis.
    assert radial[pole].min() > 0.045
    np.testing.assert_allclose(radial[equator].mean(), -0.025, atol=3e-3)


# --- 3. pulsate, then rotate --------------------------------------------------

def test_rotation_carries_pulsation_displacement_and_velocity():
    star = add_rotation(_star(2000), 50.0, rotation_axis=jnp.array([0., 0., 1.]))
    star = add_pulsation(star, 0, 0, 1.0, jnp.array([[[0.1, 0.]], [[0., 0.]], [[0., 0.]]]))
    pulsated = evaluate_pulsations(star, 0.3)
    quarter_turn = (np.pi / 2) * 695700.0 / 50.0  # seconds for 90 deg at R = 1 R_sun
    rotated = evaluate_rotation(pulsated, quarter_turn)

    offsets = np.asarray(rotated.center_pulsation_offsets)
    centers = np.asarray(rotated.d_centers)
    r_hat = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    # A radial mode stays radial after the surface has turned.
    tangential = offsets - np.einsum("ij,ij->i", offsets, r_hat)[:, None] * r_hat
    assert np.max(np.linalg.norm(tangential, axis=1)) < 1e-3 * np.max(np.linalg.norm(offsets, axis=1))
    v = np.asarray(rotated.pulsation_velocities)
    cos = np.einsum("ij,ij->i", v, r_hat) / np.linalg.norm(v, axis=1)
    assert np.all(np.abs(np.abs(cos) - 1.0) < 1e-6)


# --- 4. spots and inclination -------------------------------------------------

def test_polar_spot_lands_on_the_rotation_pole():
    star = add_rotation(_star(2000), 20.0)  # default +y axis
    spotted = add_spot(star, lat_to_theta(90.), 0., 15., 1000., 0)
    hot = np.asarray(spotted.parameters[:, 0]) > 5500.
    centroid = np.asarray(star.d_centers)[hot].mean(axis=0)
    assert _angle_deg(centroid, star.rotation_axis) < 2.0


@pytest.mark.parametrize("axis", [[0., 1., 0.], [0., 0., 1.], [1., 1., 0.]])
def test_inclination_is_measured_from_the_given_axis(axis):
    los = inclination_to_los_axis(60., jnp.array(axis))
    assert abs(_angle_deg(-np.asarray(los), axis) - 60.) < 1e-6


def test_inclination_without_axis_keeps_the_z_convention_and_warns():
    with pytest.warns(UserWarning, match="rotation_axis"):
        los = inclination_to_los_axis(60.)
    i = np.deg2rad(60.)
    np.testing.assert_allclose(np.asarray(los), [0., np.sin(i), -np.cos(i)], atol=1e-12)


def test_vsini_matches_requested_inclination_on_default_axis():
    star = add_rotation(_star(5000), 50.0)
    viewed = star._replace(los_vector=inclination_to_los_axis(60., star.rotation_axis))
    v = np.asarray(viewed.los_velocities)[np.asarray(viewed.mus) > 0]
    np.testing.assert_allclose(v.max(), 50.0 * np.sin(np.deg2rad(60.)), rtol=5e-3)


# --- 5. m != 0 basis and mode validation -------------------------------------

def test_vertex_to_polar_is_exact_on_the_poles():
    np.testing.assert_allclose(np.asarray(vertex_to_polar(jnp.array([0., 0., -1.]))), [np.pi, 0.])
    np.testing.assert_allclose(np.asarray(vertex_to_polar(jnp.array([0., 0., 2.]))), [0., 0.])


def test_horizontal_m1_basis_is_not_inflated_at_the_pole():
    star = _star(2000)
    def peak(points):
        g = spherical_harmonic_gradient_with_tilt(1, 1, points, jnp.array([0., 0., 1.]), 0.)
        return float(jnp.max(jnp.linalg.norm(g, axis=1)))
    surface = np.linalg.norm(np.asarray(star.d_vertices), axis=1) > 0.5
    ratio = peak(star.d_vertices[surface]) / peak(star.d_centers)
    assert abs(ratio - 1.0) < 0.01  # was 1.10


def test_negative_m_and_high_l_are_rejected():
    star = _star(500)
    params = jnp.array([[[0.05, 0.]], [[0., 0.]], [[0., 0.]]])
    with pytest.raises(ValueError, match="Negative azimuthal order"):
        add_pulsation(star, -1, 2, 1.0, params)
    with pytest.raises(ValueError, match="m <= l"):
        add_pulsation(star, 3, 2, 1.0, params)
    with pytest.raises(ValueError, match="Degree l"):
        add_pulsation(star, 0, 11, 1.0, params)
