from jax.typing import ArrayLike
from .mesh_model import Model
from .phoebe_utils import Component, PhoebeConfig
from typing import Optional
from collections import namedtuple


class PhoebeModel(Model, namedtuple("PhoebeModel",
                                    ["time", "mass", "radius", "d_centers", "d_cast_centers", "d_mus",
                                     "d_cast_areas"])):
    time: float
    mass: float
    radius: float
    d_centers: ArrayLike
    d_cast_centers: ArrayLike
    d_mus: ArrayLike
    d_cast_areas: ArrayLike
    
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
        return PhoebeModel.__new__(cls,
            time=time,
            mass=phoebe_config.get_quantity('mass'),
            radius=phoebe_config.get_quantity('requiv'),
            d_centers=phoebe_config.get_mesh_centers(time, component),
            d_cast_centers=phoebe_config.get_mesh_projected_centers(time, component),
            d_mus=phoebe_config.get_mus(time, component),
            d_cast_areas=phoebe_config.get_projected_areas(time, component)
        )
