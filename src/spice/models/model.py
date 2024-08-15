from typing import TypeVar


T = TypeVar('T')


class Model:
    @property
    def vertices(self) -> T:
        raise NotImplementedError
    
    @property
    def mesh_elements(self) -> T:
        raise NotImplementedError
    
    @property
    def centers(self) -> T:
        raise NotImplementedError
        
    @property
    def velocities(self) -> T:
        raise NotImplementedError
    
    @property
    def mus(self) -> T:
        raise NotImplementedError
    
    @property
    def los_velocities(self) -> T:
        raise NotImplementedError
    
    @property
    def los_z(self) -> T:
        raise NotImplementedError

    @property
    def radii(self) -> T:
        raise NotImplementedError
    
    @property
    def cast_vertices(self) -> T:
        raise NotImplementedError
    
    @property
    def cast_centers(self) -> T:
        raise NotImplementedError

    @property
    def cast_areas(self) -> T:
        raise NotImplementedError

    @classmethod
    def construct(cls, **kwargs) -> "Model":
        raise NotImplementedError
