from jax.typing import ArrayLike
from spice.models.utils import cast_to_los
from .mesh_model import Model
from .phoebe_utils import Component, PhoebeConfig, PHOEBE_AVAILABLE
from typing import List, Optional, Dict
from collections import namedtuple
import numpy as np

from spice.constants import SOLAR_RAD_CM as R_SOL_CM, DAY_TO_S

LOG_G_NAMES: List[str] = ['logg', 'loggs', 'log_g', 'log_gs', 'log g', 'log gs',
                          'surface gravity', 'surface gravities', 'surface_gravity', 'surface_gravities']
TEFF_NAMES: List[str] = ['teff', 't_eff', 't eff', 'teffs', 't_effs', 't effs',
                         'effective_temperature', 'effective_temperatures',
                         'effective temperature', 'effective temperatures']
ABUNDANCE_NAMES: List[str] = ['abundance', 'abundances',
                              'abun', 'abuns',
                              'metallicity', 'metallicities']
MU_NAMES: List[str] = ['mu', 'mus']

# Emulator bundles often qualify their parameter names with the atmosphere grid
# they were trained on -- the Aug-2026 ``RozanskiT/TPayne-spice-harps`` retrain
# exposes ``marcs_teff`` / ``marcs_logg`` rather than ``teff`` / ``logg``.
# Without stripping the prefix those names match nothing below, and
# ``construct`` either raises or (if the caller silences it by passing them in
# ``parameter_values``) bakes them in as *constants* -- silently discarding
# PHOEBE's per-element, gravity-darkened Teff and log g, which is the whole
# reason for importing a PHOEBE mesh in the first place.
MODEL_GRID_PREFIXES: List[str] = ['marcs', 'atlas', 'atlas9', 'atlas12',
                                  'phoenix', 'kurucz', 'tlusty']


def _canonical_parameter_name(name: str) -> str:
    """Lower-case ``name``, dropping a leading atmosphere-grid qualifier.

    ``'marcs_teff' -> 'teff'``; anything without a recognised prefix is
    returned unchanged, so unrelated names can never be collapsed onto a
    mesh column by accident.
    """
    lowered = name.lower()
    prefix, sep, rest = lowered.partition('_')
    if sep and prefix in MODEL_GRID_PREFIXES:
        return rest
    return lowered


def _stack_per_element(params, n_elements):
    """Shape per-parameter columns into ``(n_elements, n_parameters)`` rows.

    ``params`` is either a single 1-D array (the no-``parameter_names`` path,
    just teffs) or a *list* of 1-D arrays, one per parameter, each of length
    ``n_elements`` -- i.e. ``(n_parameters, n_elements)``.

    This used to be ``np.array(params).reshape((n_elements, -1))``, but reshape
    does not transpose: it reinterprets the flat buffer, so row 0 came out as
    the first ``n_parameters`` *teff* values rather than element 0's
    ``(teff, logg, feh, ...)``. Every element was then emulated at meaningless
    parameters. Harmless only in the single-parameter case, where the two
    layouts coincide -- which is why it survived.
    """
    arr = np.asarray(params)
    if arr.ndim == 1:
        return arr.reshape((n_elements, 1))
    return arr.T


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
        return -self.center_velocities

    @property
    def mus(self) -> ArrayLike:
        return self.d_mus

    @property
    def log_gs(self) -> ArrayLike:
        return self.d_log_gs

    @property
    def los_velocities(self) -> ArrayLike:
        # Sign convention: approaching (blueshifted) LOS velocity is negative.
        return -cast_to_los(self.velocities, self.los_vector)

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
        if not PHOEBE_AVAILABLE:
            raise ImportError("PHOEBE is not installed. Please install it with 'pip install stellar-spice[phoebe]'")
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

        # PHOEBE's uvw frame has +w toward the observer, and its radial velocity
        # is -vws (verified against a PHOEBE rv dataset: rv = -126.483 km/s while
        # the visible-area-weighted vws = +126.482). SPICE computes
        #   los_velocities = -cast_to_los(velocities, los_vector)
        # with velocities = -center_velocities and cast_to_los = -dot(v, los),
        # which reduces to +dot(-center_velocities, los_vector). Choosing
        # los_vector = +w therefore gives -vws, matching PHOEBE exactly.
        # (With [0, 0, -1] it gave +vws -- the right magnitude, wrong sign.)
        # Safe to differ from the MeshModel default here: for PhoebeModel the
        # LOS is used ONLY by los_velocities -- mus, cast_vertices/centers/areas
        # all come straight from PHOEBE, and los_z is not implemented.
        los_vector = np.array([0., 0., 1.])

        mus = phoebe_config.get_mus(time, component)

        # requiv comes back in solar radii and period was converted to
        # seconds above, so the R_sun -> cm factor is required for km/s.
        # Without it this was 6.957e10 times too small (7.96e-11 km/s
        # instead of 5.54 km/s for the TZ For primary).
        lin_velocity = 2 * np.pi * radius * R_SOL_CM / period / 1e5  # km/s

        ones_like_centers = np.ones_like(phoebe_config.get_center_velocities(time, component))[:, 0]
        log_gs = ones_like_centers * phoebe_config.b.get_quantity('loggs', component=str(component), time=time)

        if override_parameters:
            params = override_parameters
        else:
            params = []
            parameter_values = parameter_values or {}
            parameter_values_keys = [pk.lower() for pk in parameter_values.keys()]
            if parameter_names:
                for pl in parameter_names:
                    canonical = _canonical_parameter_name(pl)
                    if canonical in TEFF_NAMES:
                        params.append(phoebe_config.get_parameter(time, 'teffs', component=component))
                    elif canonical in LOG_G_NAMES:
                        params.append(log_gs)
                    elif canonical in ABUNDANCE_NAMES:
                        params.append(ones_like_centers * phoebe_config.get_quantity('abun', component=component))
                    elif canonical in MU_NAMES:
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
                                   parameters=_stack_per_element(params, mus.shape[0]),
                                   los_vector=los_vector,
                                   orbital_velocity=orbital_velocity
                                   )
