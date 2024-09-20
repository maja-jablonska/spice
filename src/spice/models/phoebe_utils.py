import phoebe
import numpy as np
from numpy.typing import ArrayLike
from typing import List, Optional

from enum import auto, Enum


R_SOL_CM = 69570000000.0
R_SURFACE_AREA = 6.082104402130212e+22


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def cos_angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)

class Component(Enum):
    PRIMARY = auto()
    SECONDARY = auto()
    
    def __str__(self):
        return self.name.lower()

class PhoebeConfig:
    def __init__(self,
                 bundle: phoebe.frontend.bundle.Bundle,
                 mesh_dataset_name: str = 'mesh01',
                 orbit_dataset_name: Optional[str] = None):
        self.__b: phoebe.frontend.bundle.Bundle = bundle
        self.__times: np.array = np.array(self.__b.times).astype(np.float64)

        if mesh_dataset_name is not None and mesh_dataset_name not in self.__b.datasets:
            raise ValueError("No dataset with name {} in the bundle. Bundle datasets: {}".format(mesh_dataset_name, ",".join(self.__b.datasets)))
        
        self.__mesh_dataset_name: str = mesh_dataset_name
        
        if orbit_dataset_name is not None and orbit_dataset_name not in self.__b.datasets:
            raise ValueError("No dataset with name {} in the bundle. Bundle datasets: {}".format(orbit_dataset_name, ",".join(self.__b.datasets)))
        
        if orbit_dataset_name is None and 'orb01' in self.__b.datasets:
            orbit_dataset_name = 'orb01'
        
        self.__orbit_dataset_name: str = orbit_dataset_name
            
    @property
    def times(self) -> np.array:
        return self.__times
        
    @property
    def mesh_dataset_name(self) -> str:
        return self.__mesh_dataset_name
    
    @property
    def orbit_dataset_name(self) -> str:
        return self.__orbit_dataset_name
    
    @property
    def b(self) -> str:
        return self.__b
    
    def get_parameter(self, time: float, qualifier: str, component: Optional[Component] = None, **kwargs) -> np.array:
        if component is not None:
            if str(component) not in self.b.components:
                raise ValueError("No component {} in the bundle. Bundle components: {}".format(str(component), ",".join(self.b.components)))
            return self.b.get_parameter(qualifier=qualifier,
                                        component=str(component),
                                        dataset=self.mesh_dataset_name,
                                        kind='mesh',
                                        time=time,
                                        **kwargs).value
        else:
            return self.b.get_parameter(qualifier=qualifier,
                            dataset=self.mesh_dataset_name,
                            kind='mesh',
                            time=time,
                            **kwargs).value
            
    def list_quantities(self) -> List[str]:
        return [param._qualifier for param in self.b._params]
            
    def get_quantity(self, qualifier: str, component: Optional[Component] = None, **kwargs) -> float:
        if component is not None:
            if str(component) not in self.b.components:
                raise ValueError("No component {} in the bundle. Bundle components: {}".format(str(component), ",".join(self.b.components)))
            return self.b.get_quantity(qualifier=qualifier, component=str(component), context='component', **kwargs).value
        else:
            return self.b.get_quantity(qualifier=qualifier, **kwargs).value
        
    def get_mesh_vertices(self, time: float, component: Optional[Component] = None) -> np.array:
        return self.get_parameter(time, 'xyz_elements', component)
    
    def get_mesh_projected_vertices(self, time: float, component: Optional[Component] = None) -> np.array:
        return self.get_parameter(time, 'uvw_elements', component)
    
    def get_mesh_projected_centers(self, time: float, component: Optional[Component] = None) -> np.array:
        return np.concatenate([self.get_parameter(time, 'us', component).reshape((-1, 1)),
                               self.get_parameter(time, 'vs', component).reshape((-1, 1)),
                               self.get_parameter(time, 'ws', component).reshape((-1, 1))], axis=1)
        
    def get_mesh_centers(self, time: float, component: Optional[Component] = None) -> np.array:
        return np.concatenate([self.get_parameter(time, 'xs', component).reshape((-1, 1)),
                               self.get_parameter(time, 'ys', component).reshape((-1, 1)),
                               self.get_parameter(time, 'zs', component).reshape((-1, 1))], axis=1)
        
    def get_center_velocities(self, time: float, component: Optional[Component] = None) -> np.array:
        return np.concatenate([self.get_parameter(time, 'vus', component).reshape((-1, 1)),
                               self.get_parameter(time, 'vvs', component).reshape((-1, 1)),
                               self.get_parameter(time, 'vws', component).reshape((-1, 1))], axis=1)
        
    
    def get_mesh_normals(self, time: float, component: Optional[Component] = None) -> np.array:
        return self.get_parameter(time, 'uvw_normals', component)
    
    def get_projected_areas(self, time: float, component: Optional[Component] = None) -> np.array:
        areas = self.get_parameter(time, 'areas', component)
        visibilities = self.get_parameter(time, 'visibilities', component)
        return areas*self.get_mus(time, component)*visibilities
    
    def get_radial_velocities(self, time: float, component: Optional[Component] = None) -> np.array:
        return self.get_parameter(time, 'vws', component)
    
    def get_loggs(self, time: float, component: Optional[Component] = None) -> np.array:
        return self.get_parameter(time, 'loggs', component)
    
    def get_teffs(self, time: float, component: Optional[Component] = None) -> np.array:
        return self.get_parameter(time, 'teffs', component)

    def get_mus(self, time: float, component: Optional[Component] = None) -> np.array:
        return self.get_parameter(time, 'mus', component)
    
    def get_all_orbit_centers(self, component: Component) -> np.array:
        if self.orbit_dataset_name is None:
            raise ValueError("No orbit dataset name provided in the constructor.")
        
        return np.concatenate([self.b.get_parameter(qualifier='us', component=str(component),
                                                    dataset=self.orbit_dataset_name, kind='orb').value.reshape((-1, 1)),
                               self.b.get_parameter(qualifier='vs', component=str(component),
                                                    dataset=self.orbit_dataset_name, kind='orb').value.reshape((-1, 1)),
                               self.b.get_parameter(qualifier='ws', component=str(component),
                                                    dataset=self.orbit_dataset_name, kind='orb').value.reshape((-1, 1))], axis=1)
        
    def get_all_orbit_velocities(self, component: Component) -> np.array:
        if self.orbit_dataset_name is None:
            raise ValueError("No orbit dataset name provided in the constructor.")
        
        return np.concatenate([self.b.get_parameter(qualifier='vus', component=str(component),
                                                    dataset=self.orbit_dataset_name, kind='orb').value.reshape((-1, 1)),
                               self.b.get_parameter(qualifier='vvs', component=str(component),
                                                    dataset=self.orbit_dataset_name, kind='orb').value.reshape((-1, 1)),
                               self.b.get_parameter(qualifier='vws', component=str(component),
                                                    dataset=self.orbit_dataset_name, kind='orb').value.reshape((-1, 1))], axis=1)
        
    def get_orbit_centers(self, time: float, component: Component) -> np.array:
        if self.orbit_dataset_name is None:
            raise ValueError("No orbit dataset name provided in the constructor.")
        
        time_ind = np.argmin(np.abs(self.times - time))
        
        return self.get_all_orbit_centers(component)[time_ind]

    def get_orbit_velocities(self, time: float, component: Component) -> np.array:
        if self.orbit_dataset_name is None:
            raise ValueError("No orbit dataset name provided in the constructor.")
        
        time_ind = np.argmin(np.abs(self.times - time))
        
        return self.get_all_orbit_velocities(component)[time_ind]
