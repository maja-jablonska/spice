from typing import Callable, List
from overrides import override
from jax.typing import ArrayLike
import jax.numpy as jnp

from spice.spectrum.spectrum_emulator import SpectrumEmulator


# Define constants
h = 6.62607015e-27  # Planck's constant [erg*s]
c = 2.99792458e10   # Speed of light [cm/s]
k = 1.380649e-16    # Boltzmann constant [erg/K]


def blackbody_intensity(log_wave: ArrayLike, mu: float, parameters: ArrayLike) -> ArrayLike:
    """
    Compute the blackbody intensity.

    Args:
        log_wave (ArrayLike): Array of logarithmic wavelengths (log10 of wavelength in Angstroms).
        mu (float): Cosine of the angle between the line of sight and the surface normal. It ranges from -1 to 1.
        parameters (ArrayLike): Array of parameters. In this case, only one element is used which represents the temperature in Kelvin.

    Returns:
        ArrayLike: Array of blackbody intensities in erg/s/cm2/A
    """
    # Convert log wavelength from angstroms to cm
    wave_cm = 10 ** (log_wave - 8)  # 1 Angstrom = 1e-8 cm

    # Extract temperature from parameters
    T = parameters[0]

    # Compute blackbody intensity
    intensity = ((2 * h * c ** 2 / wave_cm ** 5 * 1 / (jnp.exp(h * c / (wave_cm * k * T)) - 1)))*1e-8

    return jnp.tile(intensity, (2, 1))


class BlackbodySpectrum(SpectrumEmulator[ArrayLike]):
    @override
    @staticmethod
    def get_label_names() -> List[str]:
        return ['Teff']
    
    @override
    @staticmethod
    def get_default_parameters() -> ArrayLike:
        return jnp.array([5000.])
    
    @override(check_signature=False)
    @staticmethod
    def to_parameters(Teff: float) -> ArrayLike:
        return jnp.array([Teff])
    
    @override
    @staticmethod
    def flux_method() -> Callable[..., ArrayLike]:
        return blackbody_intensity
