"""``kernel_flux`` must reproduce ``simulate_observed_flux`` for a fixed mesh.

The kernel replaces per-element synthesis with n_mu convolutions. These tests
pin the two approximations it makes (mu interpolation, sub-pixel shift
interpolation) against the exact path on a synthetic intensity function with
narrow lines, a mu-dependent line and a parameter-dependent line depth, on a
rotating star with a bulk line-of-sight velocity so the kernel is asymmetric.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spice.models import IcosphereModel
from spice.models.mesh_transform import add_rotation, evaluate_rotation
from spice.spectrum.spectrum import simulate_observed_flux
from spice.spectrum.synthesis_kernel import build_synthesis_kernel, kernel_flux

X0 = 3.7                       # log10(5012 A)
DLOG = 4e-6                    # ~2.8 km/s per pixel, HARPS-like
LOG_WL = jnp.linspace(X0 - 0.004, X0 + 0.004, 2001)


def intensity(log_wl, mu, params):
    """Limb-darkened continuum with a slope, a parameter-scaled line and a second line.

    Linear in mu overall (the lines carry no mu dependence of their own), so mu
    interpolation is exact and the only kernel error left is the sub-pixel
    shift interpolation.
    """
    x = log_wl
    cont = 1e6 * (1.0 + 50.0 * (x - X0)) * (0.4 + 0.6 * mu)
    line = (1.0 - 0.5 * params[0] * jnp.exp(-((x - X0) / 2e-5) ** 2)
            - 0.3 * jnp.exp(-((x - (X0 + 1.2e-3)) / 3e-5) ** 2))
    return jnp.stack([cont * line, cont], axis=-1)


def _mesh(vrot=50.0, v_bulk=20.0):
    m = IcosphereModel.construct(2000, 1.0, 1.0, jnp.array([1.0]), ["depth"])
    if vrot:
        m = evaluate_rotation(add_rotation(m, vrot), 0.0)
    return m._replace(orbital_velocity=jnp.array([0.0, 0.0, v_bulk]))


def test_matches_exact_synthesis_for_rotating_orbiting_star():
    m = _mesh()
    exact = simulate_observed_flux(intensity, m, LOG_WL, chunk_size=256)
    k = build_synthesis_kernel(m, LOG_WL, n_mu=16, oversample=4)
    got = kernel_flux(intensity, k, m.parameters[0])
    rel = np.abs(np.asarray(got / exact) - 1.0)
    assert rel.max() < 2e-3, rel.max()
    assert np.sqrt(np.mean(rel ** 2)) < 5e-4
    # and the Doppler broadening is real: the unshifted disc spectrum differs a lot
    static = simulate_observed_flux(intensity, m, LOG_WL, chunk_size=256, disable_doppler_shift=True)
    assert np.abs(np.asarray(static / exact) - 1.0).max() > 0.05


def test_exact_when_nothing_moves():
    m = _mesh(vrot=0.0, v_bulk=0.0)
    exact = simulate_observed_flux(intensity, m, LOG_WL, chunk_size=256)
    k = build_synthesis_kernel(m, LOG_WL, n_mu=8, oversample=1)
    got = kernel_flux(intensity, k, m.parameters[0])
    # intensity is linear in mu and every shift is exactly zero pixels
    np.testing.assert_allclose(np.asarray(got), np.asarray(exact), rtol=1e-9)


def test_shift_direction_matches_exact_path():
    m = _mesh(vrot=0.0, v_bulk=80.0)          # bulk motion along the line of sight
    k = build_synthesis_kernel(m, LOG_WL, n_mu=8, oversample=4)
    got = np.asarray(kernel_flux(intensity, k, m.parameters[0]))[:, 0]
    exact = np.asarray(simulate_observed_flux(intensity, m, LOG_WL, chunk_size=256))[:, 0]
    rest = np.asarray(intensity(LOG_WL, 1.0, m.parameters[0]))[:, 0]
    x = np.asarray(LOG_WL)
    # the line moved, and it moved to exactly where the exact path puts it
    assert x[np.argmin(got)] != x[np.argmin(rest)]
    assert x[np.argmin(got)] == x[np.argmin(exact)]
    # in the direction the library's convention dictates (los_velocities < 0 is approaching)
    expected_sign = np.sign(float(jnp.mean(m.los_velocities)))
    assert np.sign(x[np.argmin(got)] - x[np.argmin(rest)]) == expected_sign


def test_kernels_conserve_visible_area():
    m = _mesh()
    k = build_synthesis_kernel(m, LOG_WL, n_mu=16, oversample=4)
    np.testing.assert_allclose(float(k.kernels.sum()), float(jnp.sum(m.visible_cast_areas)), rtol=1e-10)
    assert float(k.kernels.min()) >= 0.0


def test_gradient_wrt_parameters():
    m = _mesh()
    k = build_synthesis_kernel(m, LOG_WL, n_mu=16, oversample=4)
    total = lambda depth: jnp.sum(kernel_flux(intensity, k, jnp.array([depth]))[:, 0])
    g = jax.grad(total)(1.0)
    fd = (total(1.0 + 1e-5) - total(1.0 - 1e-5)) / 2e-5
    np.testing.assert_allclose(g, fd, rtol=1e-5)


def test_kernel_build_is_differentiable_in_velocities():
    m = _mesh(vrot=30.0, v_bulk=0.0)
    k0 = build_synthesis_kernel(m, LOG_WL, n_mu=8, oversample=4)
    win = jnp.abs(LOG_WL - X0) < 1.5e-4

    def line_centroid(v_bulk):
        mm = m._replace(orbital_velocity=jnp.array([0.0, 0.0, v_bulk]))
        k = build_synthesis_kernel(mm, LOG_WL, n_mu=8, oversample=4, half_width=k0.half_width + 40)
        f = kernel_flux(intensity, k, m.parameters[0])
        depth = jnp.where(win, 1.0 - f[:, 0] / f[:, 1], 0.0)     # continuum-normalised line
        return jnp.sum(LOG_WL * depth) / jnp.sum(depth)
    g = jax.grad(line_centroid)(5.0)
    fd = (line_centroid(5.0 + 0.5) - line_centroid(5.0 - 0.5)) / 1.0
    # a 1 km/s shift moves the line by log10(1 + 1/c) = 1.45e-6 in log10(lambda)
    assert abs(float(g)) > 1e-6
    np.testing.assert_allclose(g, fd, rtol=0.05)


def test_rejects_grid_not_uniform_in_log():
    m = _mesh()
    wl = jnp.linspace(4980.0, 5040.0, 500)
    with pytest.raises(ValueError, match="uniform in log10"):
        build_synthesis_kernel(m, jnp.log10(wl))
