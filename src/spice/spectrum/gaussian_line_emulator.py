"""
Gaussian Line Emulator

A simple spectrum emulator with Gaussian absorption lines that can be used
as an alternative to TransformerPayne for testing and demonstration purposes.
"""

from typing import Any, Dict, List
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

# Physical constants
h = 6.62607015e-27  # Planck's constant [erg*s]
c = 2.99792458e10   # Speed of light [cm/s]
k = 1.380649e-16    # Boltzmann constant [erg/K]


def _blackbody(wavelength_cm: ArrayLike, T: float) -> ArrayLike:
    """Planck function for blackbody radiation
    
    Args:
        wavelength_cm: Wavelength in cm
        T: Temperature in Kelvin
        
    Returns:
        Intensity in erg/s/cm^2/steradian/cm
    """
    return 2 * h * c**2 / wavelength_cm**5 / (jnp.exp(h * c / (wavelength_cm * k * T)) - 1)


@jax.jit
def _gaussian_line_intensity(log_wavelengths: ArrayLike, mu: float,
                            spectral_parameters: ArrayLike,
                            line_centers: ArrayLike,
                            line_widths: ArrayLike,
                            line_depths: ArrayLike) -> ArrayLike:
    """Calculate intensity with Gaussian absorption lines
    
    Args:
        log_wavelengths: log10 of wavelengths in Angstroms
        mu: cosine of angle (not used for blackbody, but kept for interface)
        spectral_parameters: [Teff]
        line_centers: Center wavelengths for lines (Angstroms)
        line_widths: Widths (sigma) for Gaussian lines (Angstroms)
        line_depths: Depths of lines (0 = no line, 1 = complete absorption)
        
    Returns:
        Intensity array with shape (n_wavelengths, 2)
    """
    # Convert log wavelengths to cm
    wavelengths_angstrom = 10 ** log_wavelengths
    wavelengths_cm = wavelengths_angstrom * 1e-8
    
    # Extract temperature
    Teff = spectral_parameters[0]
    
    # Calculate blackbody continuum
    continuum = _blackbody(wavelengths_cm, Teff)
    
    # Convert to per-Angstrom units
    continuum = continuum * 1e-8
    
    # Apply Gaussian absorption lines
    absorption = jnp.ones_like(wavelengths_angstrom)
    
    for i in range(len(line_centers)):
        # Gaussian profile: depth * exp(-(lambda - lambda0)^2 / (2*sigma^2))
        gaussian = jnp.exp(-0.5 * ((wavelengths_angstrom - line_centers[i]) / line_widths[i])**2)
        absorption = absorption * (1 - line_depths[i] * gaussian)
    
    # Apply absorption to continuum
    intensity_with_lines = continuum * absorption
    
    # Stack results: [intensity with lines, continuum]
    result = jnp.stack([intensity_with_lines, continuum], axis=1)
    
    # Handle any NaN values
    result = jnp.where(jnp.isnan(result), 0.0, result)
    
    return result


@partial(jax.jit, static_argnums=(0, 1, 2))
def _flux(roots: tuple, weights: tuple, intensity_method: callable,
          log_wavelengths: ArrayLike, spectral_parameters: ArrayLike) -> ArrayLike:
    """Integrate intensity over stellar disk using Gaussian quadrature"""
    roots, weights = jnp.array(roots), jnp.array(weights)
    return 2 * jnp.pi * jnp.sum(
        jax.vmap(intensity_method, in_axes=(None, 0, None))(
            log_wavelengths, roots, spectral_parameters
        ) * roots[:, jnp.newaxis, jnp.newaxis] * weights[:, jnp.newaxis, jnp.newaxis],
        axis=0
    )


class GaussianLineEmulator:
    """A simple spectrum emulator with Gaussian absorption lines.
    
    This emulator creates spectra with a blackbody continuum and Gaussian
    absorption lines. It can be used as a drop-in replacement for more
    complex emulators like TransformerPayne for testing and demonstration.
    
    Parameters:
        line_centers: List of wavelengths (Angstroms) for line centers
        line_widths: List of line widths (Angstroms) for each line
        line_depths: List of line depths (0-1) for each line
    """
    
    def __init__(self, 
                 line_centers: List[float] = None,
                 line_widths: List[float] = None,
                 line_depths: List[float] = None):
        
        # Default: single line at 5500 Angstroms
        self.line_centers = jnp.array(line_centers if line_centers else [5500.0])
        self.line_widths = jnp.array(line_widths if line_widths else [0.5])
        self.line_depths = jnp.array(line_depths if line_depths else [0.3])
        
        if len(self.line_centers) != len(self.line_widths) or \
           len(self.line_centers) != len(self.line_depths):
            raise ValueError("line_centers, line_widths, and line_depths must have the same length")
    
    @property
    def parameter_names(self) -> List[str]:
        """Get labels of spectrum model parameters"""
        return ["Teff"]
    
    @property
    def stellar_parameter_names(self) -> List[str]:
        """Get labels of stellar parameters (for compatibility)"""
        return self.parameter_names
    
    @property
    def solar_parameters(self) -> ArrayLike:
        """Solar parameters for the spectrum model"""
        return jnp.array([5777.0])  # Solar temperature
    
    def to_parameters(self, parameter_values: Dict[str, Any] = None) -> ArrayLike:
        """Convert passed values to the accepted parameters format
        
        Args:
            parameter_values: Dictionary with parameter values {'Teff': value}
            
        Returns:
            Array of parameters [Teff]
        """
        if not parameter_values:
            return self.solar_parameters
        
        Teff = parameter_values.get('Teff', self.solar_parameters[0])
        return jnp.array([Teff])
    
    def intensity(self, log_wavelengths: ArrayLike, mu: float, 
                  spectral_parameters: ArrayLike) -> ArrayLike:
        """Calculate the intensity for given wavelengths and mu
        
        Args:
            log_wavelengths: log10 of wavelengths in Angstroms
            mu: cosine of angle between surface normal and line of sight
            spectral_parameters: Array of parameters [Teff]
            
        Returns:
            Array of shape (n_wavelengths, 2) with:
                [:, 0]: intensity with absorption lines
                [:, 1]: continuum intensity (no lines)
        """
        return _gaussian_line_intensity(
            log_wavelengths, mu, spectral_parameters,
            self.line_centers, self.line_widths, self.line_depths
        )
    
    def flux(self, log_wavelengths: ArrayLike, spectral_parameters: ArrayLike,
             mus_number: int = 20) -> ArrayLike:
        """Compute the flux by integrating intensity over the stellar disk
        
        Args:
            log_wavelengths: log10 of wavelengths in Angstroms
            spectral_parameters: Array of parameters [Teff]
            mus_number: Number of mu points for Gaussian quadrature
            
        Returns:
            Array of fluxes
        """
        roots, weights = np.polynomial.legendre.leggauss(mus_number)
        roots = (roots + 1) / 2
        weights /= 2
        return _flux(tuple(roots), tuple(weights), self.intensity,
                    log_wavelengths, spectral_parameters)

