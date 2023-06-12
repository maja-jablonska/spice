from jax.typing import ArrayLike
import jax.numpy as jnp

# Define constants
h = 6.62607015e-27  # Planck's constant [erg*s]
c = 2.99792458e10   # Speed of light [cm/s]
k = 1.380649e-16    # Boltzmann constant [erg/K]

def blackbody_intensity(log_wave: float, mu: float, parameters: ArrayLike) -> ArrayLike:
    """
    Compute the blackbody intensity.

    Parameters:
    log_wave (jnp.array): Array of logarithmic wavelengths (log10 of wavelength in Angstroms).
    mu (float): Cosine of the angle between the line of sight and the surface normal. It ranges from -1 to 1.
    parameters (jnp.array): Array of parameters. In this case, only one element is used which represents the temperature in Kelvin.

    Returns:
    jnp.array: Array of blackbody intensities.
    """
    # Convert log wavelength from angstroms to cm
    wave_cm = 10 ** (log_wave - 8)  # 1 Angstrom = 1e-8 cm

    # Extract temperature from parameters
    T = parameters[0]

    # Compute blackbody intensity
    intensity = 2 * h * c ** 2 / wave_cm ** 5 * 1 / (jnp.exp(h * c / (wave_cm * k * T)) - 1)

    return jnp.tile(intensity, (2, 1))


