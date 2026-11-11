"""Travelling (prograde/retrograde) non-radial pulsation modes.

A travelling mode is Re[Y_l^m e^{-i s w t}]: its pattern keeps a constant shape
and amplitude and moves in longitude at w/m. "Prograde" means in the direction
the star spins; these tests check that against the mesh's own rotation
velocity field rather than a hand-derived sign.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from spice.models.mesh_model import IcosphereModel
from spice.models.utils import rotation_axis_frame
from spice.models.mesh_transform import (add_pulsation, add_pulsations, add_rotation,
                                         evaluate_pulsations, reset_pulsations)

SOLRAD_PER_DAY_TO_KMS = 8.052083333333332
P = 1.0  # days


def _star(axis=None, n=5000):
    m = IcosphereModel.construct(n, 1.0, 1.0, jnp.array([5000.0, 4.0]), ["teff", "logg"])
    return add_rotation(m, 40.0, rotation_axis=axis)


def _radial(m, pulsed):
    c = np.asarray(m.d_centers)
    r_hat = c / np.linalg.norm(c, axis=1, keepdims=True)
    return np.einsum("ij,ij->i", np.asarray(pulsed.center_pulsation_offsets), r_hat)


def _radial_mode(amplitude=0.05, phase=0.0):
    return jnp.array([[[amplitude, phase]], [[0., 0.]], [[0., 0.]]])


def _crest_longitude(star, pulsed, m_order):
    """Longitude (about the rotation axis) of an m-fold crest, from its Fourier phase."""
    B = np.asarray(rotation_axis_frame(star.rotation_axis))
    c = np.asarray(star.d_centers) @ B          # pole = rotation axis
    equator = np.abs(c[:, 2]) < 0.15
    phi = np.arctan2(c[equator, 1], c[equator, 0])
    r = _radial(star, pulsed)[equator]
    # r ~ A cos(m (phi - phi0)) = a cos(m phi) + b sin(m phi), phi0 = atan2(b, a) / m.
    # A least-squares fit is exact on the irregular mesh (a plain Fourier sum leaks).
    design = np.column_stack([np.cos(m_order * phi), np.sin(m_order * phi)])
    (a, b), *_ = np.linalg.lstsq(design, r, rcond=None)
    return np.arctan2(b, a) / m_order


def _spin_sense(star):
    """Sign of the rotation's azimuthal velocity about the rotation axis (+1 = increasing longitude)."""
    a = np.asarray(star.rotation_axis) / np.linalg.norm(star.rotation_axis)
    c = np.asarray(star.d_centers)
    e_phi = np.cross(a, c)
    return np.sign(np.sum(np.einsum("ij,ij->i", np.asarray(star.rotation_velocities), e_phi)))


@pytest.mark.parametrize("axis", [None, jnp.array([0., 0., 1.]), jnp.array([1., 1., 0.5])])
@pytest.mark.parametrize("travelling,sign", [("prograde", 1.0), ("retrograde", -1.0)])
def test_pattern_moves_with_or_against_the_rotation(axis, travelling, sign):
    star = _star(axis)
    m = add_pulsation(star, 2, 2, P, _radial_mode(), travelling=travelling)
    shift = (_crest_longitude(star, evaluate_pulsations(m, P / 40), 2)
             - _crest_longitude(star, evaluate_pulsations(m, 0.0), 2))
    shift = (shift + np.pi / 2) % np.pi - np.pi / 2
    assert abs(shift) > 0.05  # expected w t / m = 2 pi / 80
    assert np.sign(shift) == sign * _spin_sense(star)


def test_travelling_amplitude_is_constant_and_standing_is_not():
    star = _star()
    times = np.linspace(0, P, 9)
    for travelling, expect_constant in [("prograde", True), (None, False)]:
        m = add_pulsation(star, 2, 3, P, _radial_mode(0.05), travelling=travelling)
        peaks = [np.abs(_radial(star, evaluate_pulsations(m, t))).max() for t in times]
        spread = (max(peaks) - min(peaks)) / 0.05
        if expect_constant:
            assert spread < 0.05
            np.testing.assert_allclose(peaks, 0.05, rtol=0.05)
        else:
            assert spread > 0.5


def test_pattern_speed_is_omega_over_m():
    star = _star(jnp.array([0., 0., 1.]))
    m_order, dt = 3, P / 30
    m = add_pulsation(star, m_order, 3, P, _radial_mode(), travelling="retrograde")
    shift = (_crest_longitude(star, evaluate_pulsations(m, dt), m_order)
             - _crest_longitude(star, evaluate_pulsations(m, 0.0), m_order))
    shift = (shift + np.pi / m_order) % (2 * np.pi / m_order) - np.pi / m_order
    # retrograde = against the spin; SPICE spins toward decreasing longitude
    np.testing.assert_allclose(shift, -_spin_sense(star) * 2 * np.pi * dt / P / m_order, rtol=0.03)


def test_velocity_is_the_time_derivative_of_the_displacement():
    star = _star()
    fp = jnp.array([[[0.02, 0.3], [0.006, 1.1]], [[0.01, 0.7], [0., 0.]], [[0.004, 0.2], [0.002, 2.0]]])
    m = add_pulsation(star, 1, 2, P, fp, travelling="prograde")
    t, dt = 0.37, 1e-5
    d_plus = np.asarray(evaluate_pulsations(m, t + dt).center_pulsation_offsets)
    d_minus = np.asarray(evaluate_pulsations(m, t - dt).center_pulsation_offsets)
    fd = (d_plus - d_minus) / (2 * dt) * SOLRAD_PER_DAY_TO_KMS
    v = np.asarray(evaluate_pulsations(m, t).pulsation_velocities)
    # center offsets are face-centroid means of vertex offsets; velocities use centres
    assert np.median(np.linalg.norm(fd - v, axis=1)) < 0.02 * np.max(np.linalg.norm(v, axis=1))


def test_standing_default_and_reset():
    star = _star()
    standing = evaluate_pulsations(add_pulsation(star, 2, 2, P, _radial_mode()), 0.3)
    explicit = evaluate_pulsations(add_pulsation(star, 2, 2, P, _radial_mode(), travelling="standing"), 0.3)
    np.testing.assert_array_equal(np.asarray(standing.vertices), np.asarray(explicit.vertices))
    moving = add_pulsation(star, 2, 2, P, _radial_mode(), travelling="prograde")
    assert float(jnp.max(jnp.abs(moving.pulsation_travelling))) == 1.0
    assert float(jnp.max(jnp.abs(reset_pulsations(moving).pulsation_travelling))) == 0.0


def test_add_pulsations_accepts_per_mode_senses():
    star = _star()
    fp = jnp.stack([_radial_mode(), _radial_mode()])
    m = add_pulsations(star, jnp.array([1, 2]), jnp.array([2, 2]), jnp.array([P, P]), fp,
                       travelling=["prograde", "retrograde"])
    idx = np.array([1, 2]) + int(m.max_pulsation_mode) * 2
    np.testing.assert_array_equal(np.asarray(m.pulsation_travelling)[idx], [-1., 1.])


def test_invalid_travelling_requests_are_rejected():
    star = _star(n=500)
    with pytest.raises(ValueError, match="axisymmetric"):
        add_pulsation(star, 0, 2, P, _radial_mode(), travelling="prograde")
    with pytest.raises(ValueError, match="travelling must be"):
        add_pulsation(star, 1, 2, P, _radial_mode(), travelling="sideways")
