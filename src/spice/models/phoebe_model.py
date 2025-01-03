from jax.typing import ArrayLike
from spice.models.utils import cast_to_los
from .mesh_model import Model
from .phoebe_utils import Component, PhoebeConfig
from typing import List, Optional, Dict
from collections import namedtuple
import numpy as np

R_SOL_CM = 69570000000.0
DAY_TO_S = 86400.0

LOG_G_NAMES: List[str] = ['logg', 'loggs', 'log_g', 'log_gs', 'log g', 'log gs',
                          'surface gravity', 'surface gravities', 'surface_gravity', 'surface_gravities']
TEFF_NAMES: List[str] = ['teff', 't_eff', 't eff', 'teffs', 't_effs', 't effs',
                         'effective_temperature', 'effective_temperatures',
                         'effective temperature', 'effective temperatures']
ABUNDANCE_NAMES: List[str] = ['abundance', 'abundances',
                              'abun', 'abuns',
                              'metallicity', 'metallicities']
MU_NAMES: List[str] = ['mu', 'mus']


class PhoebeModel(Model, namedtuple("PhoebeModel",
                                    ["time", "mass", "radius", "center",
                                     "d_vertices", "d_cast_vertices",
                                     "d_centers",
                                     "d_cast_centers", "d_mus", "d_log_gs",
                                     "d_cast_areas", "center_velocities",
                                     "rotation_velocity", "rotation_axis",
                                     "parameters", "los_vector", "orbital_velocity"
                                     ])):
    time: float
    mass: float
    radius: float
    center: ArrayLike
    d_vertices: ArrayLike
    d_cast_vertices: ArrayLike
    d_centers: ArrayLike
    d_cast_centers: ArrayLike
    d_mus: ArrayLike
    d_log_gs: ArrayLike
    d_cast_areas: ArrayLike
    center_velocities: ArrayLike
    rotation_velocity: float
    rotation_axis: ArrayLike
    parameters: ArrayLike
    los_vector: ArrayLike
    orbital_velocity: ArrayLike

    @property
    def mesh_elements(self) -> ArrayLike:
        return self.d_vertices

    @property
    def centers(self) -> ArrayLike:
        return self.d_centers

    @property
    def velocities(self) -> ArrayLike:
        return self.center_velocities

    @property
    def mus(self) -> ArrayLike:
        return self.d_mus

    @property
    def log_gs(self) -> ArrayLike:
        return self.d_log_gs

    @property
    def los_velocities(self) -> ArrayLike:
        return cast_to_los(self.velocities, self.los_vector)

    @property
    def los_z(self) -> ArrayLike:
        raise NotImplementedError

    @property
    def cast_vertices(self) -> ArrayLike:
        return self.d_cast_vertices

    @property
    def cast_centers(self) -> ArrayLike:
        return self.d_cast_centers

    @property
    def cast_areas(self) -> ArrayLike:
        return self.d_cast_areas
    
    @property
    def visible_cast_areas(self) -> ArrayLike:
        return self.d_cast_areas

    @classmethod
    def construct(cls,
                  phoebe_config: PhoebeConfig,
                  time: float,
                  parameter_names: List[str] = None,
                  parameter_values: Dict[str, float] = None,
                  component: Optional[Component] = None,
                  override_parameters: Optional[ArrayLike] = None) -> "PhoebeModel":
        radius = phoebe_config.get_quantity('requiv', component=component)
        inclination = np.deg2rad(phoebe_config.get_quantity('incl', component=component))
        period = phoebe_config.get_quantity('period', component=component) * DAY_TO_S
        rotation_axis = np.array([0., np.sin(inclination), np.cos(inclination)])

        try:
            yaw = np.deg2rad(phoebe_config.b.get_parameter('yaw', component=str(component)).value)
            rotation_axis = np.matmul(rotation_axis,
                                      np.array([[np.cos(yaw), -np.sin(yaw), 0.],
                                                [np.sin(yaw), np.cos(yaw), 0.],
                                                [0., 0., 1.]])
                                      )
        except ValueError:
            pass

        try:
            pitch = np.deg2rad(phoebe_config.b.get_parameter('pitch', component=str(component)).value) - inclination
            rotation_axis = np.matmul(rotation_axis,
                                      np.array([[np.cos(pitch), 0., np.sin(pitch)],
                                                [0., 1., 0.],
                                                [-np.sin(pitch), 0., np.cos(pitch)]])
                                      )
        except ValueError:
            pass

        los_vector = np.array([0., 0., -1.])

        mus = phoebe_config.get_mus(time, component)

        lin_velocity = 2 * np.pi * radius / period / 1e5  # km/s

        ones_like_centers = np.ones_like(phoebe_config.get_center_velocities(time, component))[:, 0]
        log_gs = ones_like_centers * phoebe_config.b.get_quantity('logg', component=str(component), context='component')

        if override_parameters:
            params = override_parameters
        else:
            params = []
            parameter_values = parameter_values or {}
            parameter_values_keys = [pk.lower() for pk in parameter_values.keys()]
            if parameter_names:
                for pl in parameter_names:
                    if pl.lower() in TEFF_NAMES:
                        params.append(phoebe_config.get_parameter(time, 'teffs', component=component))
                    elif pl.lower() in LOG_G_NAMES:
                        params.append(log_gs)
                    elif pl.lower() in ABUNDANCE_NAMES:
                        params.append(ones_like_centers * phoebe_config.get_quantity('abun', component=component))
                    elif pl.lower() in MU_NAMES:
                        params.append(phoebe_config.get_mus(time, component=component))
                    else:
                        if pl.lower() not in parameter_values_keys:
                            raise ValueError(f"Parameter {pl} not found in parameter_values and couldn't be inferred from the PHOEBE mesh. "
                                             f"Please add it in the parameter_values dictionary")
                        params.append(ones_like_centers * parameter_values[pl])
            if len(params) == 0:
                params = phoebe_config.get_parameter(time, 'teffs', component=component)

        # If binary, retrieve orbit centers
        if phoebe_config.orbit_dataset_name:
            center = phoebe_config.get_orbit_centers(time, component=component)
            orbital_velocity = phoebe_config.get_orbit_velocities(time, component=component)
        else:
            center = np.zeros(3)
            orbital_velocity = np.zeros(3)

        return PhoebeModel.__new__(cls,
                                   time=time,
                                   mass=phoebe_config.get_quantity('mass', component=component),
                                   radius=radius,
                                   center=center,
                                   d_vertices=phoebe_config.get_mesh_projected_vertices(time, component),
                                   d_cast_vertices=phoebe_config.get_mesh_projected_vertices(time, component),
                                   d_centers=phoebe_config.get_mesh_projected_centers(time, component),
                                   d_cast_centers=phoebe_config.get_mesh_projected_centers(time, component),
                                   d_mus=mus,
                                   d_log_gs=log_gs,
                                   d_cast_areas=phoebe_config.get_projected_areas(time, component),
                                   rotation_velocity=lin_velocity,
                                   center_velocities=phoebe_config.get_center_velocities(time, component),
                                   rotation_axis=rotation_axis,
                                   parameters=np.array(params).reshape((mus.shape[0], -1)),
                                   los_vector=los_vector,
                                   orbital_velocity=orbital_velocity
                                   )
