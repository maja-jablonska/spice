import jax
import jax.numpy as jnp
import numpy as np
import pickle
import warnings
from typing import List, Optional, Dict, Union, Tuple
from numpy.typing import ArrayLike

from spice.spectrum.spectrum_emulator import SpectrumEmulator
from spice.spectrum.utils import linear_multivariate_interpolation
from spice.utils import log as _log
from overrides import override


def _import_h5py():
    """Import h5py only when an HDF5 grid is read or written.

    Keeps ``import spice.spectrum.user_spectrum_interpolator`` (and pickle
    grids) working when h5py is missing or built against another NumPy.
    """
    try:
        import h5py
    except Exception as e:
        raise ImportError(
            "h5py is required to read or write .h5 spectrum grids "
            f"(import failed: {e}). Install or rebuild it with `pip install h5py`."
        ) from e
    return h5py

# Parameter naming conventions to support flexible input formats
METALLICITY_NAMES = ["m/h", "default_metals_h", "metals/h", "me/h",
                     "[metals/h]", "[me/h]", "metallicity", "metals_h", "me_h",
                     "[m/h]", "m_h", "mh", "feh", "fe/h", "[fe/h]", "fe_h"]

LOG_G_NAMES: List[str] = ['logg', 'log_g', 'log g',
                          'surface gravity',
                          'surface_gravity']

TEFF_NAMES: List[str] = ['teff', 't_eff', 't eff',
                         'effective_temperature',
                         'effective temperature']

DEFAULT_PARAMETER_NAMES = ["teff", "logg", "m/h"]

# Common limb darkening law coefficients
LIMB_DARKENING_NAMES = ["limb_darkening", "limb_dark", "limbdark", "ld"]

# Limb darkening laws
def linear_limb_darkening(mu: jnp.ndarray, coeff: float) -> jnp.ndarray:
    """Linear limb darkening law: I(mu)/I(1) = 1 - u*(1-mu)"""
    return 1.0 - coeff * (1.0 - mu)

def quadratic_limb_darkening(mu: jnp.ndarray, coeffs: Tuple[float, float]) -> jnp.ndarray:
    """Quadratic limb darkening law: I(mu)/I(1) = 1 - a*(1-mu) - b*(1-mu)^2"""
    a, b = coeffs
    one_minus_mu = 1.0 - mu
    return 1.0 - a * one_minus_mu - b * one_minus_mu**2

def nonlinear_limb_darkening(mu: jnp.ndarray, coeffs: Tuple[float, float, float, float]) -> jnp.ndarray:
    """Non-linear limb darkening law (4-parameter)"""
    a, b, c, d = coeffs
    return 1.0 - a * (1.0 - mu**0.5) - b * (1.0 - mu) - c * (1.0 - mu**1.5) - d * (1.0 - mu**2)


@jax.jit
def _interpolate_spectrum(log10_wavelengths, continuum_wavelengths, intensities, continuum_intensities, 
                         all_parameters, parameters, log10_wavelength):
    """
    Interpolate a spectrum for a given set of parameters and wavelength.
    
    Args:
        log10_wavelengths: Array of log10 wavelengths for line spectrum
        continuum_wavelengths: Array of log10 wavelengths for continuum
        intensities: Array of intensities (with absorption lines)
        continuum_intensities: Array of continuum intensities
        all_parameters: All grid parameters including mu values
        parameters: Target parameters to interpolate at (including mu)
        log10_wavelength: Target log10 wavelength to interpolate at
        
    Returns:
        Tuple of (line intensity, continuum intensity) at the target wavelength and parameters
    """
    wave_idx = jnp.searchsorted(log10_wavelengths, log10_wavelength)
    wave_indices = jnp.clip(jnp.array([wave_idx - 1, wave_idx]), 0, len(log10_wavelengths) - 1)
    
    continuum_wave_idx = jnp.searchsorted(continuum_wavelengths, log10_wavelength)
    continuum_wave_indices = jnp.clip(jnp.array([continuum_wave_idx - 1, continuum_wave_idx]), 0, len(continuum_wavelengths) - 1)
    
    repeated_params = jnp.repeat(all_parameters, 2, axis=0)
    repeated_log10_wavelengths = jnp.tile(log10_wavelengths[wave_indices],
                                        (all_parameters.shape[0], 1)).reshape((-1, 1))
    repeated_continuum_wavelengths = jnp.tile(continuum_wavelengths[continuum_wave_indices],
                                            (all_parameters.shape[0], 1)).reshape((-1, 1))
    
    params_with_wavelength = jnp.hstack([parameters, log10_wavelength]).reshape(1, -1)
    if params_with_wavelength.shape[1] != all_parameters.shape[1] + 1:
        raise ValueError("Query parameters must match the grid parameter columns (including mu only when present)")

    return jnp.concatenate([
        linear_multivariate_interpolation(
            jnp.hstack([repeated_params, repeated_log10_wavelengths]),
            intensities[:, wave_indices].flatten(),
            params_with_wavelength
        ),
        linear_multivariate_interpolation(
            jnp.hstack([repeated_params, repeated_continuum_wavelengths]),
            continuum_intensities[:, continuum_wave_indices].flatten(),
            params_with_wavelength
        )
    ])

# Vectorize the interpolation function for multiple wavelengths
interpolate_spectrum = jax.vmap(_interpolate_spectrum, in_axes=(None, None, None, None, None, None, 0))

# Above this many cells the rows are treated as scattered rather than a
# (possibly incomplete) rectilinear grid.
_MAX_DENSE_GRID_CELLS = 20_000_000


def _rectilinear_index(parameters: np.ndarray):
    """Axes and a dense row lookup for grid rows lying on a rectilinear grid.

    Returns ``(axes, lookup)`` with ``lookup[i0, i1, ...]`` the row index at
    those axis nodes, or -1 where the grid has no model. Returns ``None`` when
    the rows are too scattered for a dense lookup to be reasonable.
    """
    axes = [np.unique(parameters[:, j]) for j in range(parameters.shape[1])]
    shape = tuple(len(a) for a in axes)
    if int(np.prod(shape, dtype=np.float64)) > _MAX_DENSE_GRID_CELLS:
        return None
    lookup = -np.ones(shape, dtype=np.int32)
    idx = tuple(np.searchsorted(axes[j], parameters[:, j]) for j in range(parameters.shape[1]))
    lookup[idx] = np.arange(parameters.shape[0], dtype=np.int32)
    return tuple(jnp.asarray(a) for a in axes), jnp.asarray(lookup)


def _bracket(axis, value):
    """Lower node index and fraction along one axis; clamps outside the axis."""
    n = axis.shape[0]
    if n == 1:
        return jnp.int32(0), jnp.int32(0), jnp.zeros((), dtype=axis.dtype)
    lo = jnp.clip(jnp.searchsorted(axis, value, side="right") - 1, 0, n - 2)
    t = jnp.clip((value - axis[lo]) / (axis[lo + 1] - axis[lo]), 0.0, 1.0)
    return lo, lo + 1, t


def _multilinear_spectrum(axes, lookup, log10_wavelengths, continuum_wavelengths,
                          intensities, continuum_intensities, parameters, log10_wavelength):
    """Multilinear interpolation over the grid axes, linear in log10 wavelength.

    Exact on grid nodes and for functions linear in each parameter. Corners
    missing from an incomplete grid are dropped and the weights renormalised
    (as in the zarr interpolators); outside the grid the edge node is used.
    """
    d = len(axes)
    los, his, ts = zip(*[_bracket(axes[j], parameters[j]) for j in range(d)])
    lo, hi, t = jnp.stack(los), jnp.stack(his), jnp.stack(ts)
    corners = jnp.array(np.array(np.meshgrid(*[[0, 1]] * d, indexing="ij")).reshape(d, -1).T)
    indices = jnp.where(corners == 0, lo, hi)                       # (2^d, d)
    weights = jnp.prod(jnp.where(corners == 0, 1.0 - t, t), axis=1)  # (2^d,)
    rows = lookup[tuple(indices.T)]
    weights = jnp.where(rows >= 0, weights, 0.0)
    weights = weights / jnp.where(jnp.sum(weights) > 0, jnp.sum(weights), 1.0)
    rows = jnp.maximum(rows, 0)

    def along_wavelength(grid_wl, spectra):
        i = jnp.clip(jnp.searchsorted(grid_wl, log10_wavelength, side="right") - 1, 0, grid_wl.shape[0] - 2)
        f = jnp.clip((log10_wavelength - grid_wl[i]) / (grid_wl[i + 1] - grid_wl[i]), 0.0, 1.0)
        values = spectra[rows, i] * (1.0 - f) + spectra[rows, i + 1] * f
        return jnp.sum(weights * values)

    return jnp.array([along_wavelength(log10_wavelengths, intensities),
                      along_wavelength(continuum_wavelengths, continuum_intensities)])


multilinear_spectrum = jax.jit(jax.vmap(_multilinear_spectrum,
                                        in_axes=(None, None, None, None, None, None, None, 0)))


class UserSpectrumInterpolator(SpectrumEmulator[ArrayLike]):
    """
    A spectrum interpolator for user-provided grids of spectra.
    
    This class allows users to load their own grid of spectra and perform
    interpolation to get intensities at arbitrary stellar parameters and wavelengths.
    
    The grid should contain:
    - parameters: Array of shape [n_models, n_params] containing parameter values (including mu)
    - intensities: Array of spectra with absorption lines
    - log10_wavelengths: Array of wavelengths in log10 scale
    - continuum_intensities (optional): Array of continuum spectra (without lines)
    - continuum_wavelengths (optional): Array of wavelengths for continuum in log10 scale
    
    If the grid doesn't include mu values, limb darkening can be applied using coefficients.
    """
    
    def __init__(self, 
                grid_path: str, 
                parameter_names: Optional[List[str]] = None,
                has_continuum: bool = True,
                has_mu: bool = True,
                limb_darkening_law: str = "linear"):
        """
        Initialize the interpolator with a user-provided grid.
        
        Args:
            grid_path: Path to the HDF5 or pickle file containing the grid data
            parameter_names: List of parameter names (default: ["teff", "logg", "m/h"])
            has_continuum: Whether the grid includes continuum spectra
            has_mu: Whether the grid includes mu as a parameter
            limb_darkening_law: Type of limb darkening law to use if has_mu=False
                                Options: "linear", "quadratic", "nonlinear"
        """
        self._parameter_names = parameter_names or DEFAULT_PARAMETER_NAMES
        self._has_continuum = has_continuum
        self._has_mu = has_mu
        
        if not has_mu:
            _log.info(f"No mu values provided, using limb darkening law: {limb_darkening_law}")
            # Set limb darkening law
            if limb_darkening_law.lower() == "linear":
                self._limb_darkening_func = linear_limb_darkening
            elif limb_darkening_law.lower() == "quadratic":
                self._limb_darkening_func = quadratic_limb_darkening
            elif limb_darkening_law.lower() == "nonlinear":
                self._limb_darkening_func = nonlinear_limb_darkening
            else:
                raise ValueError(f"Unsupported limb darkening law: {limb_darkening_law}. "
                            f"Options are: linear, quadratic, nonlinear")
        
        # Load the grid data
        try:
            if grid_path.endswith('.h5'):
                self._load_h5_grid(grid_path)
            elif grid_path.endswith('.pkl') or grid_path.endswith('.pickle'):
                self._load_pickle_grid(grid_path)
            else:
                raise ValueError(f"Unsupported file format: {grid_path}. Use .h5 or .pkl/.pickle")
        except Exception as e:
            raise RuntimeError(f"Failed to load grid from {grid_path}: {e}")

        index = _rectilinear_index(np.asarray(self.parameters))
        if index is None:
            warnings.warn(
                "Grid rows do not form a rectilinear grid; falling back to inverse-distance "
                "weighting over all rows, which is NOT linear interpolation (off-node values are "
                "biased toward the grid mean).")
            self._axes, self._lookup = None, None
        else:
            self._axes, self._lookup = index
    
    def _load_h5_grid(self, grid_path: str):
        """Load grid data from an HDF5 file."""
        h5py = _import_h5py()
        with h5py.File(grid_path, 'r') as f:
            # Required fields
            required_keys = ['parameters', 'sp_intensity', 'log10_sp_wave']
            if not all(k in f for k in required_keys):
                missing = [k for k in required_keys if k not in f]
                raise ValueError(f"Missing required keys in HDF5 file: {missing}")
            
            self.parameters = jnp.array(f['parameters'])
            self.intensities = jnp.array(f['sp_intensity'])
            self.log10_wavelengths = jnp.array(f['log10_sp_wave'])
            
            # Optional continuum fields
            if self._has_continuum:
                continuum_keys = ['sp_no_lines_intensity', 'log10_sp_no_lines_wave']
                if not all(k in f for k in continuum_keys):
                    missing = [k for k in continuum_keys if k not in f]
                    raise ValueError(f"Missing continuum keys in HDF5 file: {missing}")
                
                self.continuum_intensities = jnp.array(f['sp_no_lines_intensity'])
                self.continuum_wavelengths = jnp.array(f['log10_sp_no_lines_wave'])
            else:
                # Use the line spectrum as continuum if not provided
                self.continuum_intensities = self.intensities
                self.continuum_wavelengths = self.log10_wavelengths
    
    def _load_pickle_grid(self, grid_path: str):
        """Load grid data from a pickle file."""
        with open(grid_path, 'rb') as f:
            data = pickle.load(f)
        
        # Validate required fields
        required_keys = ['parameters', 'intensities', 'log10_wavelengths']
        if not all(k in data for k in required_keys):
            missing = [k for k in required_keys if k not in data]
            raise ValueError(f"Missing required keys in pickle file: {missing}")
        
        self.parameters = jnp.array(data['parameters'])
        self.intensities = jnp.array(data['intensities'])
        self.log10_wavelengths = jnp.array(data['log10_wavelengths'])
        
        # Optional continuum fields
        if self._has_continuum:
            continuum_keys = ['continuum_intensities', 'continuum_wavelengths']
            if not all(k in data for k in continuum_keys):
                missing = [k for k in continuum_keys if k not in data]
                warnings.warn(f"Missing continuum keys in pickle file: {missing}. Using line spectrum as continuum.")
                self.continuum_intensities = self.intensities
                self.continuum_wavelengths = self.log10_wavelengths
            else:
                self.continuum_intensities = jnp.array(data['continuum_intensities'])
                self.continuum_wavelengths = jnp.array(data['continuum_wavelengths'])
        else:
            # Use the line spectrum as continuum if not provided
            self.continuum_intensities = self.intensities
            self.continuum_wavelengths = self.log10_wavelengths
    
    def _interpolate(self, query: ArrayLike, log_wavelengths: ArrayLike) -> ArrayLike:
        query = jnp.asarray(query)
        if query.shape[-1] != self.parameters.shape[1]:
            raise ValueError("Query parameters must match the grid parameter columns "
                             "(including mu only when present)")
        if self._axes is None:
            return interpolate_spectrum(self.log10_wavelengths, self.continuum_wavelengths,
                                        self.intensities, self.continuum_intensities,
                                        self.parameters, query, log_wavelengths)
        return multilinear_spectrum(self._axes, self._lookup, self.log10_wavelengths,
                                    self.continuum_wavelengths, self.intensities,
                                    self.continuum_intensities, query, jnp.asarray(log_wavelengths))

    @property
    def parameter_names(self) -> List[str]:
        """Get names of all parameters."""
        return self._parameter_names
    
    @property
    @override
    def stellar_parameter_names(self) -> List[str]:
        """Get labels of stellar parameters (no geometry-related parameters)."""
        return self._parameter_names
    
    @override
    def to_parameters(self, parameters: Union[ArrayLike, Dict, None] = None) -> ArrayLike:
        """
        Convert parameters to the standard format used by the interpolator.
        
        Args:
            parameters: Parameters as array, list, or dictionary
            
        Returns:
            Standardized parameter array
        """
        if parameters is None:
            parameters = {}
        if not isinstance(parameters, dict):
            if not isinstance(parameters, (list, tuple, np.ndarray, jax.Array, jax.core.Tracer)):
                raise ValueError("Parameters must be an array, list, or dictionary")
            values = jnp.asarray(parameters)
            if values.ndim not in (1, 2) or values.shape[-1] != len(self._parameter_names):
                raise ValueError(
                    f"Parameters must have {len(self._parameter_names)} columns; "
                    f"got shape {values.shape}")
            return values

        if isinstance(parameters, dict):
            # Alias names are matched case-insensitively (Teff == teff); any other name must be
            # one of the grid's own parameters, otherwise it would be silently ignored.
            # Limb-darkening coefficients travel in the same dict (consumed by intensity()).
            alias_names = set(TEFF_NAMES) | set(LOG_G_NAMES) | set(METALLICITY_NAMES) | set(LIMB_DARKENING_NAMES)
            normalised, unknown = {}, []
            for key, value in parameters.items():
                name = key.lower() if isinstance(key, str) and key.lower() in alias_names else key
                if name in alias_names or name in self._parameter_names:
                    normalised[name] = value
                else:
                    unknown.append(key)
            if unknown:
                raise ValueError(
                    f"Unknown parameter name(s) {unknown}; this interpolator's parameters are "
                    f"{list(self._parameter_names)} (temperature, gravity and metallicity also accept "
                    "the usual aliases, e.g. teff/Teff, logg, m/h; limb-darkening coefficients as ld)")
            parameters = normalised

            # Fill the grid's own columns, in the grid's own order. Temperature,
            # gravity and metallicity columns also accept their aliases (so a
            # 'feh' column takes 'm/h' and vice versa). Previously the result was
            # always [teff, logg, m/h, ...] whatever the grid's order, and a grid
            # column named e.g. 'feh' was never read.
            categories = ((TEFF_NAMES, 5777.0), (LOG_G_NAMES, 4.44), (METALLICITY_NAMES, 0.0))
            values = []
            for column, name in enumerate(self._parameter_names):
                lowered = name.lower() if isinstance(name, str) else name
                if name in parameters:
                    values.append(parameters[name])
                    continue
                category = next((c for c in categories if lowered in c[0]), None)
                if category is not None:
                    found = next((parameters[a] for a in category[0] if a in parameters), None)
                    values.append(category[1] if found is None else found)
                    continue
                grid_values = self.parameters[:, column]
                midpoint = (jnp.min(grid_values) + jnp.max(grid_values)) / 2
                warnings.warn(f"Parameter {name} not provided, using midpoint value: {midpoint}")
                values.append(midpoint)
            return jnp.array(values)
        else:
            raise ValueError(
                f"Parameters must be an array, list, or dictionary, got {type(parameters)}")
    
    @override
    def flux(self, log_wavelengths: ArrayLike, parameters: ArrayLike) -> ArrayLike:
        """
        Calculate the flux for given wavelengths and parameters.
        This is not implemented in this class as it requires integration over the stellar surface.
        
        Args:
            log_wavelengths: Log10 of wavelengths in Angstroms
            parameters: Stellar parameters
            
        Raises:
            NotImplementedError: This method is not implemented
        """
        raise NotImplementedError("UserSpectrumInterpolator does not directly support flux calculation")
    
    @override
    def intensity(self, log_wavelengths: ArrayLike, mu: float, parameters: ArrayLike) -> ArrayLike:
        """
        Calculate the intensity for given wavelengths, mu, and parameters.
        
        Args:
            log_wavelengths: Log10 of wavelengths in Angstroms
            mu: Cosine of the angle between the line of sight and the surface normal
            parameters: Stellar parameters
            
        Returns:
            Array of intensities [erg/s/cm2/angstrom/sr] at the specified wavelengths
        """
        # Read auxiliary metadata before converting the stellar parameters.
        limb_darkening_coeffs = None
        if isinstance(parameters, dict):
            for ld_name in LIMB_DARKENING_NAMES:
                if ld_name in parameters:
                    limb_darkening_coeffs = parameters[ld_name]
                    break
        parameters = self.to_parameters(parameters)

        if self._has_mu:
            # Standard case - grid includes mu values
            # Add mu to parameters for interpolation
            params_with_mu = jnp.hstack([parameters, jnp.atleast_1d(mu)])
            
            return self._interpolate(params_with_mu, log_wavelengths)
        else:
            # Grid doesn't include mu - use limb darkening
            if limb_darkening_coeffs is None:
                # No coefficients provided, use defaults based on limb darkening law
                if self._limb_darkening_func == linear_limb_darkening:
                    limb_darkening_coeffs = 0.6  # Typical value for solar-type stars
                elif self._limb_darkening_func == quadratic_limb_darkening:
                    limb_darkening_coeffs = (0.4, 0.2)  # Typical values
                elif self._limb_darkening_func == nonlinear_limb_darkening:
                    limb_darkening_coeffs = (0.5, 0.2, 0.3, 0.1)  # Typical values
                elif self._limb_darkening_func == 'none':
                    limb_darkening_coeffs = None
                warnings.warn(f"No limb darkening coefficients provided. Using default values: {limb_darkening_coeffs}")
            
            # The grid already stores disk-center intensity without a mu axis.
            # Appending mu here would displace the wavelength in the query.
            
            # Perform disk center interpolation
            disk_center_intensities = self._interpolate(parameters, log_wavelengths)
            
            # Apply limb darkening to both line and continuum intensities
            ld_factor = self._limb_darkening_func(mu, limb_darkening_coeffs)
            
            # Return the intensity adjusted by limb darkening
            return disk_center_intensities * ld_factor
    
    @staticmethod
    def create_grid_from_spectra(
        parameters: List[List[float]],
        wavelengths: List[float],
        intensities: List[List[float]],
        continuum_wavelengths: Optional[List[float]] = None,
        continuum_intensities: Optional[List[List[float]]] = None,
        parameter_names: Optional[List[str]] = None,
        output_path: str = "user_spectrum_grid.h5"
    ) -> str:
        """
        Create a spectrum grid from user-provided spectra.
        
        Args:
            parameters: List of parameter sets, each containing values for teff, logg, [M/H], etc.
            wavelengths: List of wavelengths (in Angstroms)
            intensities: List of intensity arrays corresponding to each parameter set
            continuum_wavelengths: Optional list of continuum wavelengths
            continuum_intensities: Optional list of continuum intensities
            parameter_names: Names of the parameters (default: ["teff", "logg", "m/h"])
            output_path: Path to save the grid
            
        Returns:
            Path to the saved grid file
        """
        # Convert to numpy arrays
        parameters_array = np.array(parameters)
        wavelengths_array = np.array(wavelengths)
        intensities_array = np.array(intensities)
        
        # Convert wavelengths to log10 scale
        log10_wavelengths = np.log10(wavelengths_array)
        
        # Handle continuum data
        has_continuum = continuum_wavelengths is not None and continuum_intensities is not None
        if has_continuum:
            continuum_wavelengths_array = np.array(continuum_wavelengths)
            continuum_intensities_array = np.array(continuum_intensities)
            log10_continuum_wavelengths = np.log10(continuum_wavelengths_array)
        else:
            # Use the line spectrum as continuum if not provided
            log10_continuum_wavelengths = log10_wavelengths
            continuum_intensities_array = intensities_array
        
        # Save to HDF5 file
        h5py = _import_h5py()
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('parameters', data=parameters_array)
            f.create_dataset('sp_intensity', data=intensities_array)
            f.create_dataset('log10_sp_wave', data=log10_wavelengths)
            f.create_dataset('sp_no_lines_intensity', data=continuum_intensities_array)
            f.create_dataset('log10_sp_no_lines_wave', data=log10_continuum_wavelengths)
            
            # Store parameter names as attributes
            if parameter_names:
                f.attrs['parameter_names'] = parameter_names
        
        return output_path