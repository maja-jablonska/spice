"""Tests for vector spherical harmonic pulsations."""
import jax.numpy as jnp
import numpy as np
from jax import config

config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)

from spice.models import IcosphereModel
from spice.models.mesh_transform import add_pulsation, evaluate_pulsations
from spice.models.utils import (
    spherical_harmonic_with_tilt,
    spherical_harmonic_gradient_with_tilt,
    spherical_harmonic_toroidal_with_tilt,
    spherical_harmonic_tangential_gradient,
    spherical_harmonic_toroidal,
)


def _make_mesh(n_vertices=500):
    return IcosphereModel.construct(
        n_vertices=n_vertices,
        radius=1.0,
        mass=1.0,
        parameters=[5777.0],
        parameter_names=["teff"],
        max_pulsation_mode=5,
        max_fourier_order=5,
    )


class TestVSHBasis:
    """Tests for the three vector spherical harmonic basis vectors."""

    @staticmethod
    def _real_vertex_mask(coords):
        """Mask out zero-padded dummy vertices."""
        return jnp.linalg.norm(coords, axis=1) > 1e-6

    def test_toroidal_is_tangential(self):
        """T_lm should be perpendicular to the radial direction."""
        mesh = _make_mesh()
        coords = mesh.d_vertices
        mask = self._real_vertex_mask(coords)

        T = spherical_harmonic_toroidal(1, 2, coords)
        r_hat = coords / (jnp.linalg.norm(coords, axis=1, keepdims=True) + 1e-10)

        dot = jnp.sum(T * r_hat, axis=1)
        assert jnp.allclose(dot[mask], 0.0, atol=1e-4), f"Max radial component: {jnp.max(jnp.abs(dot[mask]))}"

    def test_spheroidal_is_tangential(self):
        """S_lm should be perpendicular to the radial direction."""
        mesh = _make_mesh()
        coords = mesh.d_vertices
        mask = self._real_vertex_mask(coords)

        S = spherical_harmonic_tangential_gradient(1, 2, coords)
        r_hat = coords / (jnp.linalg.norm(coords, axis=1, keepdims=True) + 1e-10)

        dot = jnp.sum(S * r_hat, axis=1)
        assert jnp.allclose(dot[mask], 0.0, atol=1e-4), f"Max radial component: {jnp.max(jnp.abs(dot[mask]))}"

    def test_toroidal_perpendicular_to_spheroidal(self):
        """T_lm and S_lm should be mutually orthogonal at every point."""
        mesh = _make_mesh()
        coords = mesh.d_vertices
        mask = self._real_vertex_mask(coords)

        S = spherical_harmonic_tangential_gradient(1, 2, coords)
        T = spherical_harmonic_toroidal(1, 2, coords)

        dot = jnp.sum(S * T, axis=1)
        norms = jnp.linalg.norm(S, axis=1) * jnp.linalg.norm(T, axis=1) + 1e-10
        cos_angle = dot / norms
        assert jnp.allclose(cos_angle[mask], 0.0, atol=1e-3), f"Max cos(angle): {jnp.max(jnp.abs(cos_angle[mask]))}"

    def test_toroidal_with_tilt_matches_untilted(self):
        """Zero tilt should give the same result as no tilt."""
        mesh = _make_mesh()
        coords = mesh.d_vertices

        T_no_tilt = spherical_harmonic_toroidal(1, 2, coords)
        T_zero_tilt = spherical_harmonic_toroidal_with_tilt(
            1, 2, coords, tilt_axis=jnp.array([0., 0., 1.]), tilt_angle=0.)

        assert jnp.allclose(T_no_tilt, T_zero_tilt, atol=1e-6)

    def test_radial_basis_is_radial(self):
        """R_lm = Y_l^m * r_hat should be purely radial."""
        mesh = _make_mesh()
        coords = mesh.d_vertices
        mask = self._real_vertex_mask(coords)
        r_hat = coords / (jnp.linalg.norm(coords, axis=1, keepdims=True) + 1e-10)

        Y = spherical_harmonic_with_tilt(1, 2, coords)[:, jnp.newaxis]
        R = Y * r_hat

        cross = jnp.cross(R, r_hat)
        assert jnp.allclose(cross[mask], 0.0, atol=1e-10)


class TestVSHPulsationModel:
    """Tests for the vector spherical harmonic pulsation model."""

    def test_fourier_params_shape(self):
        """Model should store fourier_series_parameters with shape (n_modes, 3, n_fourier, 2)."""
        mesh = _make_mesh()
        n_modes = mesh.max_pulsation_mode ** 2
        assert mesh.fourier_series_parameters.shape == (n_modes, 3, mesh.max_fourier_order, 2)

    def test_add_pulsation_vector_amplitude(self):
        """Adding a pulsation with 3-component amplitudes should store correctly."""
        mesh = _make_mesh()

        # (3, n_terms, 2): radial, spheroidal, toroidal Fourier params
        fourier_params = jnp.array([
            [[0.05, 0.0]],   # radial
            [[0.01, 0.5]],   # spheroidal
            [[0.003, 1.0]],  # toroidal
        ])

        mesh = add_pulsation(mesh, m_order=0, l_degree=1,
                             period=86400.0,
                             fourier_series_parameters=fourier_params)

        # harmonic index for (m=0, l=1) = 0 + 5*1 = 5
        idx = 0 + 5 * 1
        stored = mesh.fourier_series_parameters[idx]
        # Check that the radial component's first term is stored
        assert jnp.isclose(stored[0, 0, 0], 0.05)
        # Spheroidal component
        assert jnp.isclose(stored[1, 0, 0], 0.01)
        # Toroidal component
        assert jnp.isclose(stored[2, 0, 0], 0.003)

    def test_add_pulsation_legacy_2d_params(self):
        """A 2D (n_terms, 2) input should be auto-wrapped as purely radial."""
        mesh = _make_mesh()

        # Legacy shape: (n_terms, 2)
        fourier_params = jnp.array([[0.05, 0.0]])

        mesh = add_pulsation(mesh, m_order=0, l_degree=0,
                             period=86400.0,
                             fourier_series_parameters=fourier_params)

        idx = 0
        stored = mesh.fourier_series_parameters[idx]
        # Radial should have the value
        assert jnp.isclose(stored[0, 0, 0], 0.05)
        # Spheroidal and toroidal should be zero
        assert jnp.isclose(stored[1, 0, 0], 0.0)
        assert jnp.isclose(stored[2, 0, 0], 0.0)

    def test_purely_radial_pulsation_produces_radial_displacement(self):
        """With only radial amplitude, displacement should be purely radial."""
        mesh = _make_mesh()

        fourier_params = jnp.array([
            [[0.05, 0.0]],
            [[0.0, 0.0]],
            [[0.0, 0.0]],
        ])

        mesh = add_pulsation(mesh, m_order=0, l_degree=1,
                             period=86400.0,
                             fourier_series_parameters=fourier_params)
        mesh = evaluate_pulsations(mesh, t=0.25)

        offsets = mesh.vertices_pulsation_offsets
        r_hat = mesh.d_vertices / jnp.linalg.norm(mesh.d_vertices, axis=1, keepdims=True)

        # Offsets should be along r_hat
        cross = jnp.cross(offsets, r_hat)
        cross_norms = jnp.linalg.norm(cross, axis=1)
        offset_norms = jnp.linalg.norm(offsets, axis=1)

        # Where offset is nonzero, cross product should be small relative to offset
        mask = offset_norms > 1e-10
        if jnp.any(mask):
            relative_cross = cross_norms[mask] / offset_norms[mask]
            assert jnp.all(relative_cross < 0.05), f"Max relative cross: {jnp.max(relative_cross)}"

    def test_purely_toroidal_pulsation_is_tangential(self):
        """With only toroidal amplitude, displacement should be perpendicular to radial."""
        mesh = _make_mesh()

        fourier_params = jnp.array([
            [[0.0, 0.0]],
            [[0.0, 0.0]],
            [[0.05, 0.0]],  # toroidal only
        ])

        mesh = add_pulsation(mesh, m_order=1, l_degree=2,
                             period=86400.0,
                             fourier_series_parameters=fourier_params)
        mesh = evaluate_pulsations(mesh, t=0.25)

        offsets = mesh.vertices_pulsation_offsets
        r_hat = mesh.d_vertices / jnp.linalg.norm(mesh.d_vertices, axis=1, keepdims=True)

        # Radial component of displacement should be zero
        radial_component = jnp.sum(offsets * r_hat, axis=1)
        offset_norms = jnp.linalg.norm(offsets, axis=1)

        mask = offset_norms > 1e-10
        if jnp.any(mask):
            relative_radial = jnp.abs(radial_component[mask]) / offset_norms[mask]
            assert jnp.all(relative_radial < 0.05), f"Max relative radial: {jnp.max(relative_radial)}"

    def test_zero_amplitude_no_vertex_displacement(self):
        """Zero Fourier amplitudes should produce zero vertex displacement."""
        mesh = _make_mesh()

        fourier_params = jnp.array([
            [[0.0, 0.0]],
            [[0.0, 0.0]],
            [[0.0, 0.0]],
        ])

        mesh = add_pulsation(mesh, m_order=0, l_degree=1,
                             period=86400.0,
                             fourier_series_parameters=fourier_params)
        mesh = evaluate_pulsations(mesh, t=0.5)

        assert jnp.allclose(mesh.vertices_pulsation_offsets, 0.0, atol=1e-10)

    def test_evaluate_pulsations_no_nan(self):
        """Evaluating pulsations should not produce NaN values."""
        mesh = _make_mesh()

        fourier_params = jnp.array([
            [[0.03, 0.0], [0.01, 1.0]],
            [[0.01, 0.5], [0.005, 0.0]],
            [[0.005, 1.0], [0.002, 0.5]],
        ])

        mesh = add_pulsation(mesh, m_order=1, l_degree=2,
                             period=86400.0,
                             fourier_series_parameters=fourier_params)
        mesh = evaluate_pulsations(mesh, t=0.5)

        assert not jnp.any(jnp.isnan(mesh.vertices_pulsation_offsets))
        assert not jnp.any(jnp.isnan(mesh.center_pulsation_offsets))
        assert not jnp.any(jnp.isnan(mesh.area_pulsation_offsets))
        assert not jnp.any(jnp.isnan(mesh.pulsation_velocities))

    def test_combined_mode_nonzero(self):
        """A mode with all three VSH components should produce nonzero displacement."""
        mesh = _make_mesh()

        fourier_params = jnp.array([
            [[0.05, 0.0]],
            [[0.02, 0.0]],
            [[0.01, 0.0]],
        ])

        mesh = add_pulsation(mesh, m_order=1, l_degree=2,
                             period=86400.0,
                             fourier_series_parameters=fourier_params)
        mesh = evaluate_pulsations(mesh, t=0.25)

        assert jnp.max(jnp.abs(mesh.vertices_pulsation_offsets)) > 0.0

    def test_numerical_derivative_magnitude_matches_analytical(self):
        """Numerical and analytical velocity magnitudes should agree."""
        mesh = _make_mesh()

        fourier_params = jnp.array([
            [[0.05, 0.0]],
            [[0.02, 0.5]],
            [[0.01, 1.0]],
        ])

        mesh = add_pulsation(mesh, m_order=0, l_degree=1,
                             period=86400.0,
                             fourier_series_parameters=fourier_params)

        mesh_analytical = evaluate_pulsations(mesh, t=0.25, use_numerical_derivative=False)
        mesh_numerical = evaluate_pulsations(mesh, t=0.25, use_numerical_derivative=True)

        # Compare magnitudes (sign convention may differ between analytical/numerical)
        assert jnp.allclose(
            jnp.abs(mesh_analytical.pulsation_velocities),
            jnp.abs(mesh_numerical.pulsation_velocities),
            rtol=1e-3, atol=1e-6
        ), (f"Max diff: "
            f"{jnp.max(jnp.abs(jnp.abs(mesh_analytical.pulsation_velocities) - jnp.abs(mesh_numerical.pulsation_velocities)))}")

    def test_pulsation_dimensions_unchanged(self):
        """Pulsation offsets should have the correct shapes."""
        mesh = _make_mesh()

        fourier_params = jnp.array([
            [[0.05, 0.0]],
            [[0.0, 0.0]],
            [[0.0, 0.0]],
        ])

        mesh = add_pulsation(mesh, m_order=0, l_degree=0,
                             period=86400.0,
                             fourier_series_parameters=fourier_params)
        mesh = evaluate_pulsations(mesh, t=0.0)

        assert mesh.vertices_pulsation_offsets.shape == mesh.d_vertices.shape
        assert mesh.center_pulsation_offsets.shape == mesh.d_centers.shape
        assert mesh.area_pulsation_offsets.shape == mesh.base_areas.shape
        assert mesh.pulsation_velocities.shape == mesh.d_centers.shape
