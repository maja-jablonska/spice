import pytest
from spice.models import IcosphereModel
import jax.numpy as jnp
import numpy as np

from jax import config
config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)


SOLAR_RADIUS_CM = 69570000000.0


def test_icosphere_no_velocity():
    model = IcosphereModel.construct(
            n_vertices=1000,
            radius=SOLAR_RADIUS_CM,
            mass=1.,
            abs_luminosity=1.,
            parameters=[5777.]
            )
    
    assert jnp.all(jnp.isclose(model.velocities, 0.0))
    assert jnp.all(jnp.isclose(model.los_velocities, 0.0))
    assert jnp.isclose(model.orbital_velocity, 0.0)
    
def test_icosphere_no_offset():
    model = IcosphereModel.construct(
            n_vertices=1000,
            radius=SOLAR_RADIUS_CM,
            mass=1.,
            abs_luminosity=1.,
            parameters=[5777.]
            )
    
    assert jnp.all(jnp.isclose(model.centers, model.d_centers))
    assert jnp.all(jnp.isclose(model.vertices, model.d_vertices))
    
def test_icosphere_parameters_from_list():
    model = IcosphereModel.construct(
            n_vertices=1000,
            radius=SOLAR_RADIUS_CM,
            mass=1.,
            abs_luminosity=1.,
            parameters=[5777., 0.5]
            )
    
    assert jnp.all(jnp.isclose(model.parameters,
                               jnp.vstack([
                                       5777.*jnp.ones_like(model.areas),
                                       0.5*jnp.ones_like(model.areas)
                                       ]).T)
                   )

def test_icosphere_parameters_from_float():
    model = IcosphereModel.construct(
            n_vertices=1000,
            radius=SOLAR_RADIUS_CM,
            mass=1.,
            abs_luminosity=1.,
            parameters=0.5
            )
    
    assert jnp.all(jnp.isclose(model.parameters,
                               0.5*jnp.ones_like(model.areas)))