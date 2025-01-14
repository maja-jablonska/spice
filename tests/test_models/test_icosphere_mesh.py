from spice.models import IcosphereModel
import jax.numpy as jnp

from jax import config
import chex

config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)


SOLAR_RADIUS_CM = 69570000000.0
SOLAR_MASS_KG = 1.988409870698051e+30


class TestIcosphere:

    def test_icosphere_no_velocity(self):
        model = IcosphereModel.construct(
                n_vertices=1000,
                radius=1.,
                mass=1.,
                parameters=[5777.],
                parameter_names=['teff']
                )

        assert jnp.all(jnp.isclose(model.velocities, 0.0))
        assert jnp.all(jnp.isclose(model.los_velocities, 0.0))
        assert jnp.isclose(model.orbital_velocity, 0.0)

    def test_icosphere_no_offset(self):
        model = IcosphereModel.construct(
                n_vertices=1000,
                radius=1.,
                mass=1.,
                parameters=[5777.],
                parameter_names=['teff']
                )

        assert jnp.all(jnp.isclose(model.centers, model.d_centers))
        assert jnp.all(jnp.isclose(model.vertices, model.d_vertices))

    def test_icosphere_parameters_from_list(self):
        model = IcosphereModel.construct(
                n_vertices=1000,
                radius=SOLAR_RADIUS_CM,
                mass=SOLAR_MASS_KG,
                parameters=[5777., 0.5],
                parameter_names=['teff', 'some_param']
                )

        assert jnp.all(jnp.isclose(model.parameters,
                                   jnp.vstack([
                                           5777.*jnp.ones_like(model.areas),
                                           0.5*jnp.ones_like(model.areas)
                                           ]).T)
                       )

    def test_icosphere_parameters_from_list_logg_overriding(self):
        model = IcosphereModel.construct(
                n_vertices=1000,
                radius=1.,
                mass=1.,
                parameters=[5777., 0.5],
                parameter_names=['teff', 'logg']
                )

        assert jnp.all(jnp.isclose(model.parameters,
                                   jnp.vstack([
                                           5777.*jnp.ones_like(model.areas),
                                           jnp.log(6.6743e-11*model.mass*SOLAR_MASS_KG/jnp.power(jnp.linalg.norm(model.d_centers*SOLAR_RADIUS_CM, axis=1)*1e-2, 2)/9.80665)
                                           ]).T)
                       )


    def test_icosphere_parameters_from_list_logg_no_overriding(self):
        model = IcosphereModel.construct(
            n_vertices=1000,
            radius=1.,
            mass=1.,
            parameters=[5777., 0.5],
            parameter_names=['teff', 'logg'],
            override_log_g=False
        )

        assert jnp.all(jnp.isclose(model.parameters,
                                   jnp.vstack([
                                       5777. * jnp.ones_like(model.areas),
                                       0.5 * jnp.ones_like(model.areas)
                                   ]).T)
                       )

    def test_icosphere_parameters_from_float(self):
        model = IcosphereModel.construct(
                n_vertices=1000,
                radius=1.,
                mass=1.,
                parameters=0.5,
                parameter_names=['some_param']
                )

        assert jnp.all(jnp.isclose(model.parameters,
                                   0.5*jnp.ones_like(model.areas)))

    def test_icosphere_dimensions(self):
        model = IcosphereModel.construct(
            n_vertices=1000,
            radius=1.,
            mass=1., 
            parameters=[5777., 0.5],
            parameter_names=['teff', 'logg']
        )

        # Test vertices dimensions
        chex.assert_shape(model.vertices, (model.d_vertices.shape[0], 3))
        chex.assert_shape(model.d_vertices, (model.vertices.shape[0], 3))

        # Test centers dimensions 
        chex.assert_shape(model.centers, (model.d_centers.shape[0], 3))
        chex.assert_shape(model.d_centers, (model.centers.shape[0], 3))

        # Test faces dimensions
        chex.assert_rank(model.faces, 2)
        chex.assert_shape(model.faces, (model.faces.shape[0], 3))

        # Test areas dimensions
        chex.assert_rank(model.areas, 1)
        chex.assert_shape(model.areas, (model.faces.shape[0],))

        # Test parameters dimensions
        chex.assert_rank(model.parameters, 2)
        chex.assert_shape(model.parameters, (model.faces.shape[0], 2))

        # Test velocities dimensions
        chex.assert_shape(model.velocities, (model.faces.shape[0], 3))
        chex.assert_shape(model.rotation_velocities, (model.faces.shape[0], 3))
        chex.assert_shape(model.pulsation_velocities, (model.faces.shape[0], 3))

        # Test pulsation offsets dimensions
        chex.assert_shape(model.vertices_pulsation_offsets, model.vertices.shape)
        chex.assert_shape(model.center_pulsation_offsets, model.centers.shape)
        chex.assert_shape(model.area_pulsation_offsets, model.areas.shape)
