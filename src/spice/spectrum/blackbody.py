from typing import Any, Dict, List, Optional, Sequence

from jax.typing import ArrayLike
import jax
import jax.numpy as jnp
import numpy as np

from spice.spectrum.limb_darkening import limb_darkening, get_limb_darkening_law_id
from spice.spectrum.solar_parameters import TEFF


# Physical constants (cgs), shared across SPICE.
from spice.constants import H_ERG_S as h, C_CM_S as c, K_B_ERG_K as k


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
        ArrayLike: Two identical intensity channels in erg/s/cm2/Angstrom/sr.
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

        Accepts either ``"Teff"`` or ``"teff"`` (the key used by ``SOLAR_PARAMETERS``); any
        other name raises ``ValueError``."""
        if parameter_values is None:
            return self.solar_parameters
        from .parameters import resolve_parameter_names
        values = resolve_parameter_names(parameter_values, self.parameter_names)
        Teff = values.get("Teff", float(self.solar_parameters[0]))
        return jnp.array([Teff])

    def intensity(
        self,
        log_wavelengths: ArrayLike,
        mu: float,
        parameters: ArrayLike,
    ) -> ArrayLike:
        """Planck specific intensity, limb-darkened by the bound law if one is set.

        Args:
            log_wavelengths (ArrayLike): log10 of the wavelengths [Angstrom].
            mu (float): Cosine of the angle between the surface normal and the
                direction to the observer.
            parameters (ArrayLike): ``[Teff]`` [K].

        Returns:
            ArrayLike: ``(n_wavelengths, 2)`` intensities [erg/s/cm2/Angstrom/sr];
            both columns are the same (line and continuum channels).
        """
        return blackbody_intensity(
            log_wavelengths, mu, parameters,
            ld_law=self.ld_law, ld_coeffs=self.ld_coeffs,
        )

    def flux(self, log_wavelengths: ArrayLike, parameters: ArrayLike,
             mus_number: int = 20) -> ArrayLike:
        """Emergent surface flux, :math:`F_\\lambda = 2\\pi\\int_0^1 I_\\lambda(\\mu)\\mu\\,d\\mu`.

        Without limb darkening this is exactly :math:`\\pi B_\\lambda(T)`. With a
        limb-darkening law the angular integral is evaluated by ``mus_number``-point
        Gauss--Legendre quadrature on ``mu`` in [0, 1], as in the grid interpolators.
        This is the flux expected by the luminosity functions; use
        :func:`spice.spectrum.simulate_observed_flux` with :meth:`intensity` for the
        disc-integrated flux at an observer's distance.

        Args:
            log_wavelengths (ArrayLike): log10 of the wavelengths [Angstrom].
            parameters (ArrayLike): ``[Teff]`` [K].
            mus_number (int): Quadrature nodes used when a limb-darkening law is set.

        Returns:
            ArrayLike: ``(n_wavelengths, 2)`` surface flux [erg/s/cm2/Angstrom];
            both columns are the same (line and continuum channels).
        """
        if self.ld_law is None:
            return jnp.pi * blackbody_intensity(log_wavelengths, 1.0, parameters)
        roots, weights = np.polynomial.legendre.leggauss(mus_number)
        mus = jnp.asarray((roots + 1.0) / 2.0)
        weights = jnp.asarray(weights / 2.0)
        intensities = jax.vmap(self.intensity, in_axes=(None, 0, None))(
            log_wavelengths, mus, parameters)
        return 2.0 * jnp.pi * jnp.sum(
            intensities * (mus * weights)[:, jnp.newaxis, jnp.newaxis], axis=0)
