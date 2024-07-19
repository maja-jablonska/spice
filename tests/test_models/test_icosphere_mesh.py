from spice.models import IcosphereModel
import jax.numpy as jnp

from jax import config
config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)


SOLAR_RADIUS_CM = 69570000000.0
SOLAR_MASS_KG = 1.988409870698051e+30


class TestIcosphere:

    def test_icosphere_no_velocity(self):
        model = IcosphereModel.construct(
                n_vertices=1000,
                radius=SOLAR_RADIUS_CM,
                mass=SOLAR_MASS_KG,
                abs_luminosity=1.,
                parameters=[5777.],
                parameter_names=['teff']
                )

        assert jnp.all(jnp.isclose(model.velocities, 0.0))
        assert jnp.all(jnp.isclose(model.los_velocities, 0.0))
        assert jnp.isclose(model.orbital_velocity, 0.0)

    def test_icosphere_no_offset(self):
        model = IcosphereModel.construct(
                n_vertices=1000,
                radius=SOLAR_RADIUS_CM,
                mass=SOLAR_MASS_KG,
                abs_luminosity=1.,
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
                abs_luminosity=1.,
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
                radius=SOLAR_RADIUS_CM,
                mass=SOLAR_MASS_KG,
                abs_luminosity=1.,
                parameters=[5777., 0.5],
                parameter_names=['teff', 'logg']
                )

        assert jnp.all(jnp.isclose(model.parameters,
                                   jnp.vstack([
                                           5777.*jnp.ones_like(model.areas),
                                           jnp.log(6.6743e-11*model.mass/jnp.power(jnp.linalg.norm(model.d_centers, axis=1)*1e-2, 2)/9.80665)
                                           ]).T)
                       )


    def test_icosphere_parameters_from_list_logg_no_overriding(self):
        model = IcosphereModel.construct(
            n_vertices=1000,
            radius=SOLAR_RADIUS_CM,
            mass=SOLAR_MASS_KG,
            abs_luminosity=1.,
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
                radius=SOLAR_RADIUS_CM,
                mass=SOLAR_MASS_KG,
                abs_luminosity=1.,
                parameters=0.5,
                parameter_names=['some_param']
                )

        assert jnp.all(jnp.isclose(model.parameters,
                                   0.5*jnp.ones_like(model.areas)))
