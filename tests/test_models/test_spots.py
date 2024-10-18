import pickle

import pytest
import jax.numpy as jnp
from spice.models.spots import add_spot, add_spherical_harmonic_spots, add_spherical_harmonic_spot
from tests.test_models.utils import default_icosphere


@pytest.fixture
def mock_mesh():
    return default_icosphere()


@pytest.fixture
def spherical_harmonic_params():
    return {
        "m_values": jnp.array([0, 1]),
        "n_values": jnp.array([0, 1]),
        "param_deltas": jnp.array([100., 0.2]),
        "param_indices": jnp.array([0, 1])
    }


class TestSpotFunctions:

    def test_add_spherical_harmonic_spots_modifies_mesh_model_with_single_spot(self, mock_mesh, spherical_harmonic_params):
        m_values, n_values, param_deltas, param_indices = spherical_harmonic_params.values()
        modified_mesh = add_spherical_harmonic_spots(mock_mesh, m_values[:1], n_values[:1],
                                                     param_deltas[:1], param_indices[:1])
        assert not jnp.allclose(modified_mesh.parameters, mock_mesh.parameters)

    def test_add_spherical_harmonic_spots_modifies_mesh_model_with_multiple_spots(self, mock_mesh, spherical_harmonic_params):
        m_values, n_values, param_deltas, param_indices = spherical_harmonic_params.values()
        modified_mesh = add_spherical_harmonic_spots(mock_mesh, m_values, n_values,
                                                     param_deltas, param_indices)
        assert not jnp.allclose(modified_mesh.parameters, mock_mesh.parameters)

    def test_add_spherical_harmonic_spots_equals_to_two_add_spherical_harmonic_spot(self, mock_mesh, spherical_harmonic_params):
        m_values, n_values, param_deltas, param_indices = spherical_harmonic_params.values()
        modified_mesh = add_spherical_harmonic_spots(mock_mesh, m_values, n_values,
                                                     param_deltas, param_indices)

        modified_mesh1 = add_spherical_harmonic_spot(mock_mesh, m_values[0], n_values[0],
                                                     param_deltas[0], param_indices[0])
        modified_mesh1 = add_spherical_harmonic_spot(modified_mesh1, m_values[1], n_values[1],
                                                     param_deltas[1], param_indices[1])
        assert jnp.allclose(modified_mesh.parameters, modified_mesh1.parameters)

    def test_raises_error_for_m_larger_than_n_add_spherical_harmonic_spot(self, mock_mesh):
        with pytest.raises(ValueError):
            add_spherical_harmonic_spot(mock_mesh, 1, 0, 10, 0)

    def test_raises_error_for_m_larger_than_n_add_spherical_harmonic_spots(self, mock_mesh, spherical_harmonic_params):
        _, _, param_deltas, param_indices = spherical_harmonic_params.values()
        with pytest.raises(ValueError):
            add_spherical_harmonic_spot(mock_mesh, 1, 0, param_deltas[0], param_indices[0])

    def test_raises_error_for_read_only_phoebe_model_add_spherical_harmonic_spots(self, spherical_harmonic_params, shared_datadir):
        phoebe_mesh = pickle.load(open(shared_datadir / "phoebe_model.pkl", "rb"))
        m_values, n_values, param_deltas, param_indices = spherical_harmonic_params.values()
        with pytest.raises(ValueError):
            add_spherical_harmonic_spots(phoebe_mesh, m_values, n_values, param_deltas, param_indices)

    def test_handles_empty_spots_without_modification_add_spherical_harmonic_spots(self, mock_mesh):
        empty_arrays = jnp.array([])
        modified_mesh = add_spherical_harmonic_spots(mock_mesh, empty_arrays, empty_arrays,
                                                     empty_arrays, empty_arrays)
        assert jnp.allclose(modified_mesh.parameters, mock_mesh.parameters)

    def test_add_spot_applies_modification(self, mock_mesh):
        result = add_spot(mock_mesh, 0.0, 0.0, 50.0, 100.0, 0)
        assert not jnp.allclose(result.parameters, mock_mesh.parameters)

    def test_add_spot_raises_error_for_phoebe_model(self, shared_datadir):
        phoebe_mesh = pickle.load(open(shared_datadir / "phoebe_model.pkl", "rb"))
        with pytest.raises(ValueError):
            add_spot(phoebe_mesh, jnp.pi / 2, 0.0, jnp.pi / 4, 1.0, 0)

    def test_add_spot_with_zero_radius_does_not_modify_mesh(self, mock_mesh):
        result = add_spot(mock_mesh, 0.0, 0.0, 0.0, 100.0, 0)
        assert jnp.allclose(result.parameters, mock_mesh.parameters)

    def test_add_spot_with_large_radius_modifies_entire_mesh(self, mock_mesh):
        result = add_spot(mock_mesh, jnp.pi / 2, 0.0, 400., 1.0, 0)
        assert not jnp.any(jnp.isclose(result.parameters[:, 0], mock_mesh.parameters[:, 0]))

    def test_add_tilted_spherical_harmonic_spot(self, mock_mesh):
        base_temp = 5700
        spot_temp = 15000
        tilt_axis = jnp.array([1., 0., 0.])
        tilt_degree = 10.

        modified_mesh = add_spherical_harmonic_spot(
            mesh=mock_mesh, 
            m_order=4, 
            n_degree=4,
            param_delta=spot_temp - base_temp,
            param_index=0, 
            tilt_axis=tilt_axis, 
            tilt_degree=tilt_degree
        )

        assert not jnp.allclose(modified_mesh.parameters, mock_mesh.parameters)
        
        # Check if the tilt has been applied
        untilted_mesh = add_spherical_harmonic_spot(
            mesh=mock_mesh, 
            m_order=4, 
            n_degree=4,
            param_delta=spot_temp - base_temp,
            param_index=0
        )
        
        assert not jnp.allclose(modified_mesh.parameters, untilted_mesh.parameters)

    def test_add_tilted_spherical_harmonic_spot_with_zero_tilt(self, mock_mesh):
        base_temp = 5700
        spot_temp = 15000
        tilt_axis = jnp.array([1., 0., 0.])
        tilt_degree = 0.

        modified_mesh = add_spherical_harmonic_spot(
            mesh=mock_mesh, 
            m_order=4, 
            n_degree=4,
            param_delta=spot_temp - base_temp,
            param_index=0, 
            tilt_axis=tilt_axis, 
            tilt_degree=tilt_degree
        )

        untilted_mesh = add_spherical_harmonic_spot(
            mesh=mock_mesh, 
            m_order=4, 
            n_degree=4,
            param_delta=spot_temp - base_temp,
            param_index=0
        )
        
        assert jnp.allclose(modified_mesh.parameters, untilted_mesh.parameters)

    def test_add_tilted_spherical_harmonic_spot_different_tilt_axes(self, mock_mesh):
        base_temp = 5700
        spot_temp = 15000
        tilt_degree = 45.

        modified_mesh_x = add_spherical_harmonic_spot(
            mesh=mock_mesh, 
            m_order=4, 
            n_degree=4,
            param_delta=spot_temp - base_temp,
            param_index=0, 
            tilt_axis=jnp.array([1., 0., 0.]), 
            tilt_degree=tilt_degree
        )

        modified_mesh_y = add_spherical_harmonic_spot(
            mesh=mock_mesh, 
            m_order=4, 
            n_degree=4,
            param_delta=spot_temp - base_temp,
            param_index=0, 
            tilt_axis=jnp.array([0., 1., 0.]), 
            tilt_degree=tilt_degree
        )

        assert not jnp.allclose(modified_mesh_x.parameters, modified_mesh_y.parameters)
