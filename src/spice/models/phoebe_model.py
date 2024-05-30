from jax.typing import ArrayLike
from spice.models.utils import cast_to_los
from .mesh_model import Model
from .phoebe_utils import Component, PhoebeConfig
from typing import List, Optional
from collections import namedtuple
import numpy as np


R_SOL_CM = 69570000000.0
DAY_TO_S = 86400.0

PHOEBE_PARAMETERS = ['teffs', 'loggs']

class PhoebeModel(Model, namedtuple("PhoebeModel",
                                    ["time", "mass", "radius",
                                     "d_vertices", "d_cast_vertices",
                                     "d_centers",
                                     "d_cast_centers", "d_mus",
                                     "d_cast_areas", "center_velocities",
                                     "rotation_velocity", "rotation_axis",
                                     "parameters", "los_vector"
                                     ])):
    time: float
    mass: float
    radius: float
    d_vertices: ArrayLike
    d_cast_vertices: ArrayLike
    d_centers: ArrayLike
    d_cast_centers: ArrayLike
    d_mus: ArrayLike
    d_cast_areas: ArrayLike
    center_velocities: ArrayLike
    rotation_velocity: float
    rotation_axis: ArrayLike
    parameters: ArrayLike
    los_vector: ArrayLike
    
    @property
    def vertices(self) -> ArrayLike:
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
    
    @classmethod
    def construct(cls,
                  phoebe_config: PhoebeConfig,
                  time: float,
                  parameter_labels: List[str] = None,
                  component: Optional[Component] = None) -> "PhoebeModel":
        radius = phoebe_config.get_quantity('requiv', component=component)*R_SOL_CM
        inclination = np.deg2rad(phoebe_config.get_quantity('incl', component=component))
        period = phoebe_config.get_quantity('period', component=component)*DAY_TO_S
        los_vector = np.array([0., -np.sin(inclination), np.cos(inclination)])
        mus=phoebe_config.get_mus(time, component)
        
        lin_velocity = 2*np.pi*radius/period
        
        params = []
        if parameter_labels:
            for pl in parameter_labels:
                if pl+'s' in PHOEBE_PARAMETERS:
                    params.append(phoebe_config.get_parameter(time, pl+'s', component=component))
                elif pl in PHOEBE_PARAMETERS:
                    params.append(phoebe_config.get_parameter(time, pl, component=component))
        if len(params)==0:
            params = phoebe_config.get_parameter(time, 'teffs', component=component)
        
        return PhoebeModel.__new__(cls,
            time=time,
            mass=phoebe_config.get_quantity('mass', component=component),
            radius=radius,
            d_vertices=phoebe_config.get_mesh_vertices(time, component),
            d_cast_vertices=phoebe_config.get_mesh_projected_vertices(time, component),
            d_centers=phoebe_config.get_mesh_centers(time, component),
            d_cast_centers=phoebe_config.get_mesh_projected_centers(time, component),
            d_mus=mus,
            d_cast_areas=phoebe_config.get_projected_areas(time, component),
            rotation_velocity=lin_velocity,
            center_velocities=phoebe_config.get_center_velocities(time, component),
            rotation_axis=los_vector,
            parameters=np.array(params).reshape((mus.shape[0], -1)),
            los_vector=-np.array([0., 0., 1.])
        )
