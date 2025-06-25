import pickle

import pytest
import jax.numpy as jnp
from src.spice.models.mesh_transform import add_pulsations, transform, add_rotation, evaluate_rotation, add_pulsation, reset_pulsations, \
    evaluate_pulsations, update_parameters
from tests.test_models.utils import default_icosphere

import chex


class TestMeshTransformations:
    @pytest.fixture
    def mesh_model(self):
        return default_icosphere()

    @pytest.fixture
    def phoebe_model(self, shared_datadir):
        return pickle.load(open(shared_datadir / "phoebe_model.pkl", "rb"))

    def test_transform_mesh(self, mesh_model):
        vector = jnp.array([1, 1, 1])
        transformed_mesh = transform(mesh_model, vector)
        assert jnp.allclose(transformed_mesh.center, vector), "Mesh center should be transformed by the vector."

    def test_transform_phoebe_model_raises_error(self, phoebe_model):
        vector = jnp.array([1, 1, 1])
        with pytest.raises(ValueError):
            transform(phoebe_model, vector)
            
    def test_update_parameters_single(self, mesh_model):
        parameter_name = "teff"
        parameter_names = ['teff', 'abun']
        new_value = 5000.0
        updated_mesh = update_parameters(mesh_model, [parameter_name], [new_value], parameter_names)
        assert jnp.allclose(updated_mesh.parameters[:, parameter_names.index(parameter_name)], new_value), f"{parameter_name} should be updated to {new_value}"
        
    def test_update_parameters_no_parameter_names(self, mesh_model):
        with pytest.raises(ValueError):
            update_parameters(mesh_model, ["teff"], [5000.0], parameter_names=None)

    def test_update_parameters_multiple(self, mesh_model):
        parameter_names = ["teff", "abun"]
        new_values = [5000.0, 4.5]
        updated_mesh = update_parameters(mesh_model, parameter_names, new_values, parameter_names)
        for name, value in zip(parameter_names, new_values):
            assert jnp.allclose(updated_mesh.parameters[:, parameter_names.index(name)], value), f"{name} should be updated to {value}"

    def test_update_parameters_by_index(self, mesh_model):
        parameter_indices = [0, 1]  # Assuming these are valid indices
        new_values = [5000.0, 4.5]
        updated_mesh = update_parameters(mesh_model, parameter_indices, new_values)
        for idx, value in zip(parameter_indices, new_values):
            assert jnp.allclose(updated_mesh.parameters[:, idx], value), f"Parameter at index {idx} should be updated to {value}"

    def test_update_parameters_phoebe_model_raises_error(self, phoebe_model):
        with pytest.raises(ValueError):
            update_parameters(phoebe_model, ["teff"], [5000.0])

    def test_update_parameters_invalid_name(self, mesh_model):
        with pytest.raises(ValueError):
            update_parameters(mesh_model, ["invalid_parameter"], [5000.0])

    def test_update_parameters_mismatched_lengths(self, mesh_model):
        with pytest.raises(ValueError):
            update_parameters(mesh_model, ["temperature", "gravity"], [5000.0])  # Missing one value

    def test_add_rotation_to_mesh(self, mesh_model):
        rotation_velocity = jnp.array([0.1, 0.1, 0.1])
        rotation_axis = jnp.array([0, 0, 1])
        rotated_mesh = add_rotation(mesh_model, rotation_velocity, rotation_axis)
        assert jnp.allclose(rotated_mesh.rotation_velocity,
                            rotation_velocity), "Mesh rotation velocity should be updated."

    def test_add_rotation_to_phoebe_model_raises_error(self, phoebe_model):
        rotation_velocity = jnp.array([0.1, 0.1, 0.1])
        with pytest.raises(ValueError):
            add_rotation(phoebe_model, rotation_velocity)

    def test_evaluate_rotation_mesh(self, mesh_model):
        t = 3600
        rotation_model = add_rotation(mesh_model, rotation_velocity=100., rotation_axis=jnp.array([1., 0., 0.]))
        evaluated_mesh = evaluate_rotation(rotation_model, t)
        assert not jnp.all(jnp.isclose(rotation_model.cast_centers, evaluated_mesh.cast_centers)), "Evaluate rotation should update the mesh model."

    def test_evaluate_rotation_phoebe_model_raises_error(self, phoebe_model):
        t = 1.0
        with pytest.raises(ValueError):
            evaluate_rotation(phoebe_model, t)

    def test_pulsation_functions(self, mesh_model):
        t = 1.0
        m_order, n_degree = 1, 1
        pulsation_period = 1
        fourier_series_parameters = jnp.array([[1, 1]])
        pulsated_mesh = add_pulsation(mesh_model, m_order, n_degree, pulsation_period,
                                      fourier_series_parameters)
        evaluated_pulsation_mesh = evaluate_pulsations(pulsated_mesh, t)
        assert not jnp.all(jnp.isclose(pulsated_mesh.los_velocities, evaluated_pulsation_mesh.los_velocities)), "Pulsation should be added to the mesh model."
        reset_mesh = reset_pulsations(pulsated_mesh)
        assert jnp.allclose(reset_mesh.vertices_pulsation_offsets,
                            jnp.zeros_like(mesh_model.vertices_pulsation_offsets)), "Pulsations should be reset."
        
    def test_pulsation_functions_fourier_as_array(self, mesh_model):
        t = 1.0
        m_order, n_degree = jnp.array([1]), jnp.array([1])
        pulsation_period = 1
        fourier_series_parameters = jnp.array([[1, 1]])
        pulsated_mesh = add_pulsation(mesh_model, m_order, n_degree, pulsation_period,
                                      fourier_series_parameters)
        evaluated_pulsation_mesh = evaluate_pulsations(pulsated_mesh, t)
        assert not jnp.all(jnp.isclose(pulsated_mesh.los_velocities, evaluated_pulsation_mesh.los_velocities)), "Pulsation should be added to the mesh model."
        reset_mesh = reset_pulsations(pulsated_mesh)
        assert jnp.allclose(reset_mesh.vertices_pulsation_offsets,
                            jnp.zeros_like(mesh_model.vertices_pulsation_offsets)), "Pulsations should be reset."

    def test_pulsation_functions_with_pulsation_period_as_array(self, mesh_model):
        t = 1.0
        spherical_harmonics_parameters = jnp.array([1, 1])
        pulsation_period = jnp.array([1.])
        fourier_series_parameters = jnp.array([[1, 1]])
        pulsated_mesh = add_pulsation(mesh_model, spherical_harmonics_parameters[0], spherical_harmonics_parameters[1], pulsation_period,
                                      fourier_series_parameters)
        evaluated_pulsation_mesh = evaluate_pulsations(pulsated_mesh, t)
        assert not jnp.all(jnp.isclose(pulsated_mesh.los_velocities, evaluated_pulsation_mesh.los_velocities)), "Pulsation should be added to the mesh model."
        reset_mesh = reset_pulsations(pulsated_mesh)
        assert jnp.allclose(reset_mesh.vertices_pulsation_offsets,
                            jnp.zeros_like(mesh_model.vertices_pulsation_offsets)), "Pulsations should be reset."
        
    def test_pulsation_with_tilt(self, mesh_model):
        t = 1800
        m_order, n_degree = 1, 1
        pulsation_period = 3600  # 1 hour
        fourier_series_parameters = jnp.array([[0.1, 0.0]])  # Amplitude 0.1, phase 0.0
        pulsation_axis = jnp.array([0.0, 1.0, 0.0])  # Y-axis
        pulsation_angle = 45.  # 45 degrees tilt

        # Add pulsation with tilt
        pulsated_mesh = add_pulsation(mesh_model, m_order, n_degree, pulsation_period,
                                      fourier_series_parameters, pulsation_axis, pulsation_angle)
        
        # Evaluate pulsation
        evaluated_pulsation_mesh = evaluate_pulsations(pulsated_mesh, t)
        
        # Check if pulsation has been applied
        assert not jnp.allclose(pulsated_mesh.vertices_pulsation_offsets, 
                                evaluated_pulsation_mesh.vertices_pulsation_offsets), \
            "Pulsation offsets should change after evaluation"
        
        # Check if pulsation velocities are non-zero
        assert jnp.any(evaluated_pulsation_mesh.pulsation_velocities != 0), \
            "Pulsation velocities should be non-zero"
        
        # Check if pulsation axis and angle are correctly set
        harmonic_ind = m_order + mesh_model.max_pulsation_mode * n_degree
        assert jnp.allclose(pulsated_mesh.pulsation_axes[harmonic_ind], 
                            pulsation_axis), \
            "Pulsation axis should be correctly set"
        assert jnp.isclose(pulsated_mesh.pulsation_angles[harmonic_ind], 
                           pulsation_angle), \
            "Pulsation angle should be correctly set"

    def test_multiple_pulsations_with_tilt(self, mesh_model):
        t = 1800
        m_orders = jnp.array([1, 2])
        l_degrees = jnp.array([1, 2])
        periods = jnp.array([3600.0, 7200.0])  # 1 hour and 2 hours
        fourier_series_parameters = jnp.array([
            [0.1, 0.0],  # Amplitude 0.1, phase 0.0 for first pulsation
            [0.05, jnp.pi/2]  # Amplitude 0.05, phase pi/2 for second pulsation
        ])
        pulsation_axes = jnp.array([
            [0.0, 1.0, 0.0],  # Y-axis for first pulsation
            [0.0, 0.0, 1.0]   # Z-axis for second pulsation
        ])
        pulsation_angles = jnp.array([45., 30.])  # 45 and 30 degrees tilts

        # Add multiple pulsations with tilts
        pulsated_mesh = add_pulsations(mesh_model, m_orders, l_degrees, periods,
                                       fourier_series_parameters, pulsation_axes, pulsation_angles)
        
        # Evaluate pulsations
        evaluated_pulsation_mesh = evaluate_pulsations(pulsated_mesh, t)
        
        # Check if pulsations have been applied
        assert not jnp.allclose(pulsated_mesh.vertices_pulsation_offsets, 
                                evaluated_pulsation_mesh.vertices_pulsation_offsets), \
            "Pulsation offsets should change after evaluation"
        
        # Check if pulsation velocities are non-zero
        assert jnp.any(evaluated_pulsation_mesh.pulsation_velocities != 0), \
            "Pulsation velocities should be non-zero"
        
        # Check if pulsation axes and angles are correctly set for both pulsations
        for i in range(2):
            harmonic_ind = m_orders[i] + mesh_model.max_pulsation_mode * l_degrees[i]
            assert jnp.allclose(pulsated_mesh.pulsation_axes[harmonic_ind], pulsation_axes[i]), \
                f"Pulsation axis for pulsation {i} should be correctly set"
            assert jnp.isclose(pulsated_mesh.pulsation_angles[harmonic_ind], pulsation_angles[i]), \
                f"Pulsation angle for pulsation {i} should be correctly set"

    def test_transform_dimensions(self, mesh_model):
        """Test dimensions of transform function outputs"""
        vector = jnp.array([1.0, 2.0, 3.0])
        
        # Test unjitted version
        transformed = transform(mesh_model, vector)
        chex.assert_shape(transformed.center, (3,))
        chex.assert_equal_shape([transformed.d_vertices, mesh_model.d_vertices])
        chex.assert_equal_shape([transformed.d_centers, mesh_model.d_centers])

    def test_rotation_dimensions(self, mesh_model):
        """Test dimensions of rotation function outputs"""
        rotation_velocity = 10.0
        rotation_axis = jnp.array([0., 0., 1.])
        t = 1800.0

        # Test add_rotation
        rotated = add_rotation(mesh_model, rotation_velocity, rotation_axis)
        chex.assert_shape(rotated.rotation_axis, (3,))
        chex.assert_shape(rotated.rotation_matrix, (3, 3))
        chex.assert_shape(rotated.rotation_matrix_prim, (3, 3))
        
        # Test evaluate_rotation
        evaluated = evaluate_rotation(rotated, t)
        chex.assert_equal_shape([evaluated.d_vertices, mesh_model.d_vertices])
        chex.assert_equal_shape([evaluated.d_centers, mesh_model.d_centers])
        chex.assert_equal_shape([evaluated.rotation_velocities, mesh_model.d_centers])

    def test_pulsation_dimensions(self, mesh_model):
        """Test dimensions of pulsation function outputs"""
        m_order = 1
        n_degree = 1
        period = 3600.0
        fourier_params = jnp.array([[0.1, 0.0]])
        pulsation_axis = jnp.array([0., 1., 0.])
        pulsation_angle = 45.0
        t = 1800.0

        # Test add_pulsation
        pulsated = add_pulsation(mesh_model, m_order, n_degree, period, 
                                fourier_params, pulsation_axis, pulsation_angle)
        
        max_ind = m_order + mesh_model.max_pulsation_mode * n_degree
        chex.assert_shape(pulsated.pulsation_axes[max_ind], (3,))
        chex.assert_shape(pulsated.fourier_series_parameters[max_ind], 
                         (mesh_model.max_fourier_order, 2))

        # Test evaluate_pulsations  
        evaluated = evaluate_pulsations(pulsated, t)
        chex.assert_equal_shape([evaluated.vertices_pulsation_offsets, 
                                mesh_model.d_vertices])
        chex.assert_equal_shape([evaluated.center_pulsation_offsets,
                                mesh_model.d_centers])
        chex.assert_equal_shape([evaluated.pulsation_velocities,
                                mesh_model.d_centers])
