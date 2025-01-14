from copy import deepcopy
import pickle

import jax.numpy as jnp
import pytest

from spice.models import IcosphereModel
from spice.models.binary import Binary, PhoebeBinary, evaluate_orbit, evaluate_orbit_at_times, add_orbit
from spice.models.mesh_transform import transform
from spice.models.model import Model
from tests.test_models.utils import default_icosphere

SOLAR_RADIUS_CM = 69570000000.0


@pytest.mark.skip("Skipping binary tests until debugged")
class TestBinaryModels:
    @pytest.fixture
    def setup_binary(self):
        body1 = default_icosphere()
        body2 = transform(default_icosphere(), jnp.array([SOLAR_RADIUS_CM*2, 0., 0.]))
        binary = Binary.from_bodies(body1, body2)
        return binary

    @pytest.fixture
    def setup_phoebe_binary(self, shared_datadir):
        config = pickle.load(open(shared_datadir / "phoebe_binary_config.pkl", "rb"))
        return PhoebeBinary.construct(config, ['teff', 'logg', 'abun'])

    def test_from_bodies(self, setup_binary):
        binary = setup_binary
        assert isinstance(binary, Binary), "Binary object creation failed"

    def test_phoebe_binary_construct(self, setup_phoebe_binary):
        phoebe_binary = setup_phoebe_binary
        assert isinstance(phoebe_binary, PhoebeBinary), "PhoebeBinary object creation failed"

    def test_evaluate_orbit(self, setup_binary):
        _binary = deepcopy(setup_binary)
        binary = add_orbit(_binary, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10)
        time = 0.5
        n_cells = 10
        primary, secondary = evaluate_orbit(binary, time, n_cells)
        assert isinstance(primary, IcosphereModel) and isinstance(secondary, IcosphereModel), "Orbit evaluation failed to return Model instances"

    def test_evaluate_orbit_at_times(self, setup_binary):
        _binary = deepcopy(setup_binary)
        binary = add_orbit(_binary, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10)
        times = jnp.array([0.1, 0.5, 0.9])
        n_cells = 10
        primary_models, secondary_models = evaluate_orbit_at_times(binary, times, n_cells)
        assert len(primary_models) == len(times) and len(secondary_models) == len(times), "Orbit evaluation at times failed to return correct number of models"

    def test_add_orbit_raises_error_for_phoebe_binary(self, setup_phoebe_binary):
        with pytest.raises(ValueError):
            add_orbit(setup_phoebe_binary, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10)