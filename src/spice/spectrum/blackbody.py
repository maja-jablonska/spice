from typing import Any, Dict, List, Optional, Sequence

from jax.typing import ArrayLike
import jax.numpy as jnp

from spice.spectrum.limb_darkening import limb_darkening, get_limb_darkening_law_id
from spice.spectrum.solar_parameters import TEFF


# Define constants
h = 6.62607015e-27  # Planck's constant [erg*s]
c = 2.99792458e10   # Speed of light [cm/s]
k = 1.380649e-16    # Boltzmann constant [erg/K]


def blackbody_intensity(
    log_wave: ArrayLike,
    mu: float,
    parameters: ArrayLike,
    ld_law: Optional[str] = None,
    ld_coeffs: Optional[ArrayLike] = None,
) -> ArrayLike:
    """
    Compute the blackbody intensity, optionally attenuated by a limb-darkening law.

    Args:
        log_wave (ArrayLike): Array of logarithmic wavelengths (log10 of wavelength in Angstroms).
        mu (float): Cosine of the angle between the line of sight and the surface normal. It ranges from -1 to 1.
        parameters (ArrayLike): Array of parameters. In this case, only one element is used which represents the temperature in Kelvin.
        ld_law (Optional[str]): Limb-darkening law name from ``spice.spectrum.limb_darkening``
            (``"linear"``, ``"quadratic"``, ``"square_root"``, ``"logarithmic"``, ``"nonlinear"``).
            ``None`` (default) disables limb darkening, recovering the bare blackbody.
        ld_coeffs (Optional[ArrayLike]): Coefficients for the selected law. Required when ``ld_law`` is given.

    Returns:
        ArrayLike: Array of blackbody intensities in erg/s/cm2/A
    """
    # Convert log wavelength from angstroms to cm
    wave_cm = 10 ** (log_wave - 8)  # 1 Angstrom = 1e-8 cm

    # Extract temperature from parameters
    T = parameters[0]

    # Compute blackbody intensity
    intensity = ((2 * h * c ** 2 / wave_cm ** 5 * 1 / (jnp.exp(h * c / (wave_cm * k * T)) - 1)))*1e-8

    if ld_law is not None:
        if ld_coeffs is None:
            raise ValueError("ld_coeffs must be provided when ld_law is set")
        ld = limb_darkening(mu, get_limb_darkening_law_id(ld_law), jnp.asarray(ld_coeffs))
        intensity = intensity * ld

    return jnp.stack([intensity, intensity], axis=-1)


class Blackbody:
    """Blackbody spectrum emulator parameterised by effective temperature only.

    Optional ``ld_law`` / ``ld_coeffs`` bind a limb-darkening law into
    :meth:`intensity` so it can be used directly with
    :func:`spice.spectrum.simulate_observed_flux` without re-passing the LD config.
    """

    def __init__(
        self,
        ld_law: Optional[str] = None,
        ld_coeffs: Optional[Sequence[float]] = None,
    ):
        if ld_law is not None and ld_coeffs is None:
            raise ValueError("ld_coeffs must be provided when ld_law is set")
        self.ld_law = ld_law
        self.ld_coeffs = jnp.asarray(ld_coeffs) if ld_coeffs is not None else None

    @property
    def parameter_names(self) -> List[str]:
        return ["Teff"]

    @property
    def stellar_parameter_names(self) -> List[str]:
        return self.parameter_names

    @property
    def solar_parameters(self) -> ArrayLike:
        return jnp.array([TEFF])

    def to_parameters(self, parameter_values: Optional[Dict[str, Any]] = None) -> ArrayLike:
        """Convert a ``{name: value}`` dict to the ``[Teff]`` parameter vector.

        Accepts either ``"Teff"`` or ``"teff"`` (the key used by ``SOLAR_PARAMETERS``)."""
        if not parameter_values:
            return self.solar_parameters
        Teff = parameter_values.get("Teff", parameter_values.get("teff", float(self.solar_parameters[0])))
        return jnp.array([Teff])

    def intensity(
        self,
        log_wavelengths: ArrayLike,
        mu: float,
        parameters: ArrayLike,
    ) -> ArrayLike:
        return blackbody_intensity(
            log_wavelengths, mu, parameters,
            ld_law=self.ld_law, ld_coeffs=self.ld_coeffs,
        )

    def flux(self, log_wavelengths: ArrayLike, parameters: ArrayLike) -> ArrayLike:
        """Disc-integrated flux at μ=1 (no geometric integration here — callers
        wanting a true flux should use :func:`spice.spectrum.simulate_observed_flux`)."""
        return blackbody_intensity(
            log_wavelengths, 1.0, parameters,
            ld_law=self.ld_law, ld_coeffs=self.ld_coeffs,
        )
