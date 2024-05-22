from typing import Any, Dict, List
from functools import partial
import numpy as np

from transformer_payne.spectrum_emulator import SpectrumEmulator

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

    
h = 6.62607015e-27  # Planck's constant [erg*s]
c = 2.99792458e10   # Speed of light [cm/s]
k = 1.380649e-16    # Boltzmann constant [erg/K]


class TwoLayersWithLine:
    @property
    def parameter_names(self) -> List[str]:
        """Get labels of spectrum model parameters

        Returns:
            List[str]:
        """
        return ["T_bg", "T_fg", "line_width", "rho0"]
    
    @property
    def min_parameters(self) -> ArrayLike:
        """Minimum values supported by the spectrum model

        Returns:
            ArrayLike:
        """
        return jnp.array([0., 0., 0., 0.], dtype=jnp.float32)
    
    @property
    def max_parameters(self) -> ArrayLike:
        """Maximum values supported by the spectrum model

        Returns:
            ArrayLike:
        """
        return jnp.array([jnp.inf, jnp.inf, jnp.inf, jnp.inf], dtype=jnp.float32)
    
    @property
    def stellar_parameter_names(self) -> ArrayLike:
        return self.parameter_names()
    
    @property
    def min_stellar_parameters(self) -> ArrayLike:
        return self.min_stellar_parameters()
    
    @property
    def max_stellar_parameters(self) -> ArrayLike:
        return self.max_stellar_parameters()
    
    @staticmethod
    def is_in_bounds(parameters: ArrayLike) -> bool:
        """Check if parameters are within the bounds of the spectrum model

        Args:
            parameters (ArrayLike):

        Returns:
            bool:
        """
        return jnp.all(parameters >= 0.)
    
    @property
    def solar_parameters(self) -> ArrayLike:
        """Solar parameters for the spectrum model

        Returns:
            ArrayLike:
        """
        return jnp.array([5777., 5000., 0.1, 3e-11])
    
    def to_parameters(self, parameter_values: Dict[str, Any] = None) -> ArrayLike:
        """Convert passed values to the accepted parameters format

        Args:
            parameter_values (Dict[str, Any], optional): parameter values in the format of {'parameter_name': value}. Unset parameters will be set to solar values.

        Raises:
            ValueError: when the parameters are out of accepted bounds

        Returns:
            ArrayLike:
        """
        if not parameter_values:
            return self.solar_parameters
        
        T_bg = parameter_values.get('T_bg', self.solar_parameters[0])
        T_fg = parameter_values.get('T_fg', self.solar_parameters[1])
        line_width = parameter_values.get('line_width', self.solar_parameters[2])
        rho0 = parameter_values.get('rho0', self.solar_parameters[3])
        
        return jnp.array([T_bg, T_fg, line_width, rho0])
    
    @staticmethod
    def flux(log_wavelengths: ArrayLike, spectral_parameters: ArrayLike, mus_number: int = 20) -> ArrayLike:
        """Compute the blackbody flux.

        Args:
            log_wavelengths (ArrayLike): Array of logarithmic wavelengths (log10 of wavelength in Angstroms).
            parameters (ArrayLike): Array of parameters. In this case, only one element is used which represents the temperature in Kelvin.

        Returns:
            ArrayLike: Array of blackbody monochromatic fluxes in erg/s/cm3
        """
        roots, weights = np.polynomial.legendre.leggauss(mus_number)
        roots = (roots + 1) / 2
        weights /= 2
        return _flux(tuple(roots), tuple(weights), _intensity, log_wavelengths, spectral_parameters)

    @staticmethod
    def intensity(log_wavelengths: ArrayLike, mu: float, spectral_parameters: ArrayLike) -> ArrayLike:
        """Calculate the intensity for given wavelengths and mus

        Args:
            log_wavelengths (ArrayLike): [log(angstrom)]
            mu (float): cosine of the angle between the star's radius and the line of sight. As the blackbody radiation field is isotropic, this parameter is not used.
            spectral_parameters (ArrayLike): an array of predefined stellar parameters. In this case, only one element is used which represents the temperature in Kelvin.

        Returns:
            ArrayLike: monochromatic intensities corresponding to passed wavelengths [erg/s/cm3/steradian]
        """

        return _intensity(log_wavelengths, mu, spectral_parameters)

    @staticmethod
    def intensity_linear_limb_darkening(log_wavelengths: ArrayLike, mu: float, spectral_parameters: ArrayLike, limb_darkening_coeff: float = 0.6) -> ArrayLike:
        """Calculate the intensity for given wavelengths and mus

        Args:
            log_wavelengths (ArrayLike): [log(angstrom)]
            mu (float): cosine of the angle between the star's radius and the line of sight. As the blackbody radiation field is isotropic, this parameter is not used.
            spectral_parameters (ArrayLike): an array of predefined stellar parameters. In this case, only one element is used which represents the temperature in Kelvin.

        Returns:
            ArrayLike: monochromatic intensities corresponding to passed wavelengths [erg/s/cm3/steradian]
        """

        return _intensity(log_wavelengths, 1.0, spectral_parameters) * (1 - limb_darkening_coeff*(1 - mu))

    
    @staticmethod
    def flux_linear_limb_darkening(log_wavelengths: ArrayLike, spectral_parameters: ArrayLike, mus_number: int = 20, limb_darkening_coeff: float = 0.6) -> ArrayLike:
        """Compute the blackbody flux.

        Args:
            log_wavelengths (ArrayLike): Array of logarithmic wavelengths (log10 of wavelength in Angstroms).
            parameters (ArrayLike): Array of parameters. In this case, only one element is used which represents the temperature in Kelvin.

        Returns:
            ArrayLike: Array of blackbody monochromatic fluxes in erg/s/cm3
        """
        roots, weights = np.polynomial.legendre.leggauss(mus_number)
        roots = (roots + 1) / 2
        weights /= 2
        _intensity_lld = jax.jit(lambda log_wavelengths, mu, spectral_parameters: _intensity(log_wavelengths, 1.0, spectral_parameters) * (1 - limb_darkening_coeff*(1 - mu)))
        return _flux(tuple(roots), tuple(weights), _intensity_lld , log_wavelengths, spectral_parameters)
    

def B_w(wave, T):
    return 2*h*c**2/wave**5/(jnp.exp(h*c/(wave*k*T))-1)

@jax.jit
def _intensity(log_wavelengths, mu, spectral_parameters):
    # Convert log wavelength from angstroms to cm
    # wave_cm = 10 ** (log_wavelengths - 8)  # 1 Angstrom = 1e-8 cm
    # Convert log wavelength from angstroms to frequency in Hz
    wave_cm = 10 ** (log_wavelengths - 8)

    # Get parameters for the intenisty function
    dz = 5e8 # cm
    wave_0 = 5500.0 * 1e-8 # cm
    eps = 1e-10
    # ---
    T_bg = spectral_parameters[0] # K - balckground radiation temperature
    T_fg = spectral_parameters[1] # K - foreground matter temperature
    line_width = spectral_parameters[2] * 1e-8 # cm - line width in angstroms
    rho0 = spectral_parameters[3] # g/cm3 - foreground matter density

    # Compute opacity of a layer: line
    
    kappa_cont = 1e-1 * 1e-7 / (jnp.pi*line_width)   # constant continuum opacity [cm2/g] = 1e-1 * maximum line opacity
    kappa_line = 1e-7 * line_width / jnp.pi / ((wave_cm - wave_0)**2 + line_width**2)
    kappa = kappa_line + kappa_cont

    exptau_cont = jnp.exp(- kappa_cont * rho0 * dz / mu )
    exptau = jnp.exp(- kappa * rho0 * dz / mu )

    intensity_cont = B_w(wave_cm, T_bg) * exptau_cont + (1 - exptau_cont) * B_w(wave_cm, T_fg)
    intensity = B_w(wave_cm, T_bg) * exptau + (1 - exptau) * B_w(wave_cm, T_fg)

    # Stack into one array
    y = jnp.stack([intensity, intensity_cont], axis=1)
    y = jnp.where(jnp.isnan(y), 0.0, y)
    return y

@partial(jax.jit, static_argnums=(0, 1, 2))
def _flux(roots: ArrayLike, weights: ArrayLike, intensity_method: callable,
          log_wavelengths: ArrayLike, spectral_parameters: ArrayLike):
    roots, weights = jnp.array(roots), jnp.array(weights)
    return 2*jnp.pi*jnp.sum(
        jax.vmap(intensity_method, in_axes=(None, 0, None))(log_wavelengths, roots, spectral_parameters)*
        roots[:, jnp.newaxis, jnp.newaxis]*weights[:, jnp.newaxis, jnp.newaxis],
        axis=0
    )