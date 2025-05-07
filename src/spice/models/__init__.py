from .model import Model
from .mesh_generation import icosphere
from .mesh_model import IcosphereModel, MeshModel
from .spots import add_spherical_harmonic_spot, add_spherical_harmonic_spots
from .binary import Binary
from .utils import lat_to_theta, lon_to_phi, theta_to_lat, phi_to_lon
from .mesh_view_kdtree import resolve_occlusion, get_optimal_search_radius

# Make PHOEBE-related imports optional
try:
    from .phoebe_model import PhoebeModel
    from .binary import PhoebeBinary
    PHOEBE_AVAILABLE = True
except ImportError:
    PHOEBE_AVAILABLE = False