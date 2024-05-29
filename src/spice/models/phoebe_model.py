from jax.typing import ArrayLike
from .mesh_model import Model
from .phoebe_utils import Component, PhoebeConfig
from typing import List, Optional
from collections import namedtuple
import numpy as np


R_SOL_CM = 69570000000.0
DAY_TO_S = 86400.0

PHOEBE_PARAMETERS = ['teff', 'logg']

class PhoebeModel(Model, namedtuple("PhoebeModel",
                                    ["time", "mass", "radius", "d_centers",
                                     "d_cast_centers", "d_mus",
                                     "d_cast_areas", "center_velocities",
                                     "rotation_velocity", "rotation_axis",
                                     "parameters"
                                     ])):
    time: float
    mass: float
    radius: float
    d_centers: ArrayLike
    d_cast_centers: ArrayLike
    d_mus: ArrayLike
    d_cast_areas: ArrayLike
    center_velocities: ArrayLike
    rotation_velocity: float
    rotation_axis: ArrayLike
    parameters: ArrayLike
    
    @property
    def vertices(self) -> ArrayLike:
        raise NotImplementedError
    
    @property
    def centers(self) -> ArrayLike:
        return self.d_centers
        
    @property
    def velocities(self) -> ArrayLike:
        raise NotImplementedError
    
    @property
    def mus(self) -> ArrayLike:
        raise self.get_mus()
    
    @property
    def los_velocities(self) -> ArrayLike:
        raise NotImplementedError
    
    @property
    def los_z(self) -> ArrayLike:
        raise NotImplementedError
    
    @property
    def los_z(self) -> ArrayLike:
        raise NotImplementedError
    
    @property
    def cast_vertices(self) -> ArrayLike:
        raise NotImplementedError
    
    @property
    def cast_centers(self) -> ArrayLike:
        return self.d_cast_centers

    @property
    def cast_areas(self) -> ArrayLike:
        raise self.d_cast_areas
    
    @classmethod
    def construct(cls,
                  phoebe_config: PhoebeConfig,
                  time: float,
                  component: Optional[Component] = None) -> "PhoebeModel":
        radius = phoebe_config.get_quantity('requiv', component=component)*R_SOL_CM
        inclination = phoebe_config.get_quantity('incl', component=component)
        period = phoebe_config.get_quantity('period', component=component)*DAY_TO_S
        rotation_axis = np.array([0., np.sin(inclination), np.cos(inclination)])
        
        lin_velocity = 2*np.pi*radius/period
        
        return PhoebeModel.__new__(cls,
            time=time,
            mass=phoebe_config.get_quantity('mass', component=component),
            radius=radius,
            d_centers=phoebe_config.get_mesh_centers(time, component),
            d_cast_centers=phoebe_config.get_mesh_projected_centers(time, component),
            d_mus=phoebe_config.get_mus(time, component),
            d_cast_areas=phoebe_config.get_projected_areas(time, component),
            rotation_velocity=lin_velocity,
            center_velocities=phoebe_config.get_center_velocities(time, component),
            rotation_axis=rotation_axis,
            parameters=np.zeros(1,)
        )
