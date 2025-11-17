from .model import Model
from .mesh_generation import icosphere
from .mesh_model import IcosphereModel, MeshModel
from .spots import add_ring_spot, add_spherical_harmonic_spot, add_spherical_harmonic_spots

try:  # Binary utilities require optional geometry dependencies
    from .binary import Binary
except ImportError:  # pragma: no cover - optional dependency (jaxkd)
    Binary = None
from .utils import lat_to_theta, lon_to_phi, theta_to_lat, phi_to_lon
from .eclipse_utils import find_eclipses

# Make PHOEBE-related imports optional
try:
    from .phoebe_model import PhoebeModel
    from .binary import PhoebeBinary
    PHOEBE_AVAILABLE = True
except ImportError:
    PHOEBE_AVAILABLE = False
