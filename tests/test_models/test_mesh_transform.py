import pickle

import pytest
import jax.numpy as jnp
from src.spice.models.mesh_transform import transform, add_rotation, evaluate_rotation, add_pulsation, reset_pulsations, \
    evaluate_pulsations
from tests.test_models.utils import default_icosphere


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
        t = 1.0
        rotation_model = add_rotation(mesh_model, rotation_velocity=100.)
        evaluated_mesh = evaluate_rotation(rotation_model, t)
        assert not jnp.all(jnp.isclose(rotation_model.cast_centers, evaluated_mesh.cast_centers)), "Evaluate rotation should update the mesh model."

    def test_evaluate_rotation_phoebe_model_raises_error(self, phoebe_model):
        t = 1.0
        with pytest.raises(ValueError):
            evaluate_rotation(phoebe_model, t)

    def test_pulsation_functions(self, mesh_model):
        t = 1.0
        spherical_harmonics_parameters = jnp.array([1, 1])
        pulsation_period = 1
        fourier_series_parameters = jnp.array([[1, 1]])
        pulsated_mesh = add_pulsation(mesh_model, spherical_harmonics_parameters, pulsation_period,
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
        pulsated_mesh = add_pulsation(mesh_model, spherical_harmonics_parameters, pulsation_period,
                                      fourier_series_parameters)
        evaluated_pulsation_mesh = evaluate_pulsations(pulsated_mesh, t)
        assert not jnp.all(jnp.isclose(pulsated_mesh.los_velocities, evaluated_pulsation_mesh.los_velocities)), "Pulsation should be added to the mesh model."
        reset_mesh = reset_pulsations(pulsated_mesh)
        assert jnp.allclose(reset_mesh.vertices_pulsation_offsets,
                            jnp.zeros_like(mesh_model.vertices_pulsation_offsets)), "Pulsations should be reset."
