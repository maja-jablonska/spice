import pickle

import pytest
import jax.numpy as jnp
from spice.models.spots import add_spot, add_spherical_harmonic_spots, add_spherical_harmonic_spot
from tests.test_models.utils import default_icosphere


@pytest.fixture
def mock_mesh():
    return default_icosphere()


@pytest.fixture
def spot_params():
    return {
        "spot_center_theta": 0.5,
        "spot_center_phi": 1.0,
        "spot_radius": 0.1,
        "parameter_diff": 100.,
        "parameter_index": 0,
        "smoothness": 1.5
    }


@pytest.fixture
def spherical_harmonic_params():
    return {
        "m_values": jnp.array([0, 1]),
        "n_values": jnp.array([0, 1]),
        "param_deltas": jnp.array([100., 0.2]),
        "param_indices": jnp.array([0, 1])
    }


class TestSpotFunctions:
    def test_addSpot_modifiesMeshModel(self, mock_mesh, spot_params):
        modified_mesh = add_spot(mock_mesh, **spot_params)
        assert not jnp.allclose(modified_mesh.parameters, mock_mesh.parameters)

    def test_addSphericalHarmonicSpots_modifiesMeshModelWithSingleSpot(self, mock_mesh, spherical_harmonic_params):
        m_values, n_values, param_deltas, param_indices = spherical_harmonic_params.values()
        modified_mesh = add_spherical_harmonic_spots(mock_mesh, m_values[:1], n_values[:1],
                                                     param_deltas[:1], param_indices[:1])
        assert not jnp.allclose(modified_mesh.parameters, mock_mesh.parameters)

    def test_addSphericalHarmonicSpots_modifiesMeshModelWithMultipleSpots(self, mock_mesh, spherical_harmonic_params):
        m_values, n_values, param_deltas, param_indices = spherical_harmonic_params.values()
        modified_mesh = add_spherical_harmonic_spots(mock_mesh, m_values, n_values,
                                                     param_deltas, param_indices)
        assert not jnp.allclose(modified_mesh.parameters, mock_mesh.parameters)

    def test_addSphericalHarmonicSpots_equalsToTwo_addSphericalHarmonicSpot(self, mock_mesh, spherical_harmonic_params):
        m_values, n_values, param_deltas, param_indices = spherical_harmonic_params.values()
        modified_mesh = add_spherical_harmonic_spots(mock_mesh, m_values, n_values,
                                                     param_deltas, param_indices)

        modified_mesh1 = add_spherical_harmonic_spot(mock_mesh, m_values[0], n_values[0],
                                                     param_deltas[0], param_indices[0])
        modified_mesh1 = add_spherical_harmonic_spot(modified_mesh1, m_values[1], n_values[1],
                                                     param_deltas[1], param_indices[1])
        assert jnp.allclose(modified_mesh.parameters, modified_mesh1.parameters)

    def test_raisesErrorFor_m_LargerThan_n_addSphericalHarmonicSpot(self, mock_mesh, spherical_harmonic_params):
        with pytest.raises(ValueError):
            add_spherical_harmonic_spot(mock_mesh, 1, 0, 10, 0)

    def test_raisesErrorFor_m_LargerThan_n_addSphericalHarmonicSpots(self, mock_mesh, spherical_harmonic_params):
        m_values, n_values, param_deltas, param_indices = spherical_harmonic_params.values()
        with pytest.raises(ValueError):
            add_spherical_harmonic_spots(mock_mesh, 1, 0, param_deltas, param_indices)

    def test_raisesErrorForReadOnlyPhoebeModel_addSpot(self, spot_params, shared_datadir):
        phoebe_mesh = pickle.load(open(shared_datadir / "phoebe_model.pkl", "rb"))
        with pytest.raises(ValueError):
            add_spot(phoebe_mesh, **spot_params)

    def test_raisesErrorForReadOnlyPhoebeModel_addSphericalHarmonicSpots(self, spherical_harmonic_params, shared_datadir):
        phoebe_mesh = pickle.load(open(shared_datadir / "phoebe_model.pkl", "rb"))
        m_values, n_values, param_deltas, param_indices = spherical_harmonic_params.values()
        with pytest.raises(ValueError):
            add_spherical_harmonic_spots(phoebe_mesh, m_values, n_values, param_deltas, param_indices)

    def test_handlesEmptySpotsWithoutModification_addSphericalHarmonicSpots(self, mock_mesh):
        empty_arrays = jnp.array([])
        modified_mesh = add_spherical_harmonic_spots(mock_mesh, empty_arrays, empty_arrays,
                                                     empty_arrays, empty_arrays)
        assert jnp.allclose(modified_mesh.parameters, mock_mesh.parameters)
