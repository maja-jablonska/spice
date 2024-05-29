import phoebe
import numpy as np
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

LOS = np.array([0, 0, 1])

class Component(Enum):
    PRIMARY = auto()
    SECONDARY = auto()
    
    def __str__(self):
        return self.name.lower()

class PhoebeConfig:
    def __init__(self,
                 bundle: phoebe.frontend.bundle.Bundle,
                 mesh_dataset_name: str):
        self.__b: phoebe.frontend.bundle.Bundle = bundle
        self.__times: np.array = np.array(self.__b.times)

        self.__dataset_name: str = mesh_dataset_name
            
    @property
    def times(self) -> np.array:
        return self.__times
        
    @property
    def dataset_name(self) -> str:
        return self.__dataset_name
    
    @property
    def b(self) -> str:
        return self.__b
    
    def get_parameter(self, time: float, qualifier: str, component: Optional[Component] = None) -> np.array:
        if component is not None:
            if str(component) not in self.b.components:
                raise ValueError("No component {} in the bundle. Bundle components: {}".format(str(component), ",".join(self.b.components)))
            return self.b.get_parameter(qualifier=qualifier,
                                        component=str(component),
                                        dataset=self.dataset_name,
                                        kind='mesh',
                                        time=time).value
        else:
            return self.b.get_parameter(qualifier=qualifier,
                            dataset=self.dataset_name,
                            kind='mesh',
                            time=time).value
            
    def list_quantities(self) -> List[str]:
        return [param._qualifier for param in self.b._params]
            
    def get_quantity(self, qualifier: str, component: Optional[Component] = None) -> float:
        if component is not None:
            if str(component) not in self.b.components:
                raise ValueError("No component {} in the bundle. Bundle components: {}".format(str(component), ",".join(self.b.components)))
            return self.b.get_quantity(qualifier=qualifier, component=component, context='component')
        else:
            return self.b.get_quantity(qualifier=qualifier).value
    
    def get_mesh_projected_centers(self, time: float, component: Optional[Component] = None) -> np.array:
        return np.concatenate([self.get_parameter(time, 'us', component).reshape((-1, 1)),
                               self.get_parameter(time, 'vs', component).reshape((-1, 1)),
                               self.get_parameter(time, 'ws', component).reshape((-1, 1))], axis=1)*R_SOL_CM
        
    def get_mesh_centers(self, time: float, component: Optional[Component] = None) -> np.array:
        return np.concatenate([self.get_parameter(time, 'xs', component).reshape((-1, 1)),
                               self.get_parameter(time, 'ys', component).reshape((-1, 1)),
                               self.get_parameter(time, 'zs', component).reshape((-1, 1))], axis=1)*R_SOL_CM
        
    def get_center_velocities(self, time: float, component: Optional[Component] = None) -> np.array:
        return np.concatenate([self.get_parameter(time, 'vxs', component).reshape((-1, 1)),
                               self.get_parameter(time, 'vys', component).reshape((-1, 1)),
                               self.get_parameter(time, 'vzs', component).reshape((-1, 1))], axis=1)
    
    def get_mesh_normals(self, time: float, component: Optional[Component] = None) -> np.array:
        return self.get_parameter(time, 'uvw_normals', component)
    
    def get_projected_areas(self, time: float, component: Optional[Component] = None) -> np.array:
        mesh_normals = self.get_mesh_normals(time, component)
        areas = self.get_parameter(time, 'areas', component)
        return areas*np.array([cos_angle_between(mn, LOS) for mn in mesh_normals])*R_SURFACE_AREA
    
    def get_radial_velocities(self, time: float, component: Optional[Component] = None) -> np.array:
        return self.get_parameter(time, 'vws', component)
    
    def get_loggs(self, time: float, component: Optional[Component] = None) -> np.array:
        return self.get_parameter(time, 'loggs', component)
    
    def get_teffs(self, time: float, component: Optional[Component] = None) -> np.array:
        return self.get_parameter(time, 'teffs', component)

    def get_mus(self, time: float, component: Optional[Component] = None) -> np.array:
        return self.get_parameter(time, 'mus', component)
