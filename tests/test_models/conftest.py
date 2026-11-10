"""Small deterministic fixtures; no version-specific pickles are needed."""
import jax.numpy as jnp
import pytest

from spice.models.phoebe_model import PhoebeModel
from tests.test_models.utils import default_icosphere


@pytest.fixture
def phoebe_model():
    """An imported mesh for exercising PHOEBE's read-only model contract."""
    mesh = default_icosphere()
    return PhoebeModel(
        time=0.0, mass=mesh.mass, radius=mesh.radius, center=jnp.zeros(3),
        d_vertices=mesh.mesh_elements,
        d_cast_vertices=mesh.cast_vertices[mesh.faces.astype(int)],
        d_centers=mesh.centers, d_cast_centers=mesh.cast_centers,
        d_mus=mesh.mus, d_log_gs=mesh.log_gs,
        d_cast_areas=mesh.visible_cast_areas,
        center_velocities=jnp.zeros_like(mesh.centers),
        rotation_velocity=0.0, rotation_axis=mesh.rotation_axis,
        parameters=mesh.parameters, los_vector=jnp.array([0., 0., 1.]),
        orbital_velocity=jnp.zeros(3),
    )
