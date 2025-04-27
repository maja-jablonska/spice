import jax
import jax.numpy as jnp
import numpy as np
import h5py
import pickle
import warnings
from typing import List, Optional, Dict, Union, Tuple
from numpy.typing import ArrayLike

from spice.spectrum.spectrum_emulator import SpectrumEmulator
from spice.spectrum.utils import linear_multivariate_interpolation
from overrides import override

# Parameter naming conventions to support flexible input formats
METALLICITY_NAMES = ["default_metals_h", "metals/h", "me/h",
                     "[metals/h]", "[me/h]", "metallicity", "metals_h", "me_h"]

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
    # The error is due to a shape mismatch between the points and query_points in linear_multivariate_interpolation
    # Make sure the dimensions match by ensuring params_with_wavelength has the same number of features as repeated_params + wavelength
    
    # Get the number of parameters in the input
    num_params = repeated_params.shape[1]
    
    # Make sure params_with_wavelength has the same number of columns as the points we're comparing against
    if params_with_wavelength.shape[1] != num_params + 1:
        # Adjust the shape to match what's expected
        params_with_wavelength = params_with_wavelength[:, :num_params+1]
    
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
            print("No mu values provided, using limb darkening law: ", limb_darkening_law)
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
    
    def _load_h5_grid(self, grid_path: str):
        """Load grid data from an HDF5 file."""
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
        parameters = parameters or {}
        if isinstance(parameters, (list, tuple)):
            parameters = jnp.array(parameters)
            
        if isinstance(parameters, np.ndarray):
            if len(parameters.shape) > 1:
                if parameters.shape[1] != len(self._parameter_names):
                    raise ValueError(
                        f"Parameters must have {len(self._parameter_names)} columns, got {parameters.shape[1]}")
                return parameters
            return parameters
        
        elif isinstance(parameters, dict):
            # Extract Teff from various possible keys
            teff = None
            for key in TEFF_NAMES:
                if key in parameters:
                    teff = parameters[key]
                    break
            if teff is None:
                teff = 5777.0  # Default solar Teff
            
            logg = None
            for key in LOG_G_NAMES:
                if key in parameters:
                    logg = parameters[key]
                    break
            if logg is None:
                logg = 4.44  # Default solar logg
            
            mh = None
            for key in METALLICITY_NAMES:
                if key in parameters:
                    mh = parameters[key]
                    break
            if mh is None:
                mh = 0.0  # Default solar metallicity
            
            # Support custom parameter sets beyond the standard teff, logg, [M/H]
            if len(self._parameter_names) > 3:
                custom_params = []
                for param_name in self._parameter_names[3:]:
                    if param_name in parameters:
                        custom_params.append(parameters[param_name])
                    else:
                        # Use the midpoint of the grid range for missing parameters
                        param_values = self.parameters[:, 3 + len(custom_params)]
                        midpoint = (jnp.min(param_values) + jnp.max(param_values)) / 2
                        custom_params.append(midpoint)
                        warnings.warn(f"Parameter {param_name} not provided, using midpoint value: {midpoint}")
                
                return jnp.array([teff, logg, mh] + custom_params)
            else:
                return jnp.array([teff, logg, mh])
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
            Array of intensities [erg/cm2/s/angstrom] at the specified wavelengths
        """
        # Convert parameters to the correct format if needed
        if isinstance(parameters, dict) or (isinstance(parameters, np.ndarray) and parameters.shape[-1] != len(self._parameter_names)):
            parameters = self.to_parameters(parameters)
        
        # Check if parameters include limb darkening coefficients
        limb_darkening_coeffs = None
        if isinstance(parameters, dict):
            # Extract limb darkening coefficients from parameters
            for ld_name in LIMB_DARKENING_NAMES:
                if ld_name in parameters:
                    limb_darkening_coeffs = parameters[ld_name]
                    break
        
        if self._has_mu:
            # Standard case - grid includes mu values
            # Add mu to parameters for interpolation
            params_with_mu = jnp.hstack([parameters, jnp.atleast_1d(mu)])
            
            # Perform interpolation
            return interpolate_spectrum(
                self.log10_wavelengths, 
                self.continuum_wavelengths, 
                self.intensities, 
                self.continuum_intensities, 
                self.parameters,
                params_with_mu, 
                log_wavelengths
            )
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
            
            # Interpolate at disk center (mu=1.0) - this is the reference intensity
            params_with_mu_center = jnp.hstack([parameters, jnp.atleast_1d(1.0)])
            
            # Perform disk center interpolation
            disk_center_intensities = interpolate_spectrum(
                self.log10_wavelengths, 
                self.continuum_wavelengths, 
                self.intensities, 
                self.continuum_intensities, 
                self.parameters,
                params_with_mu_center, 
                log_wavelengths
            )
            
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