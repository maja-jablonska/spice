from functools import partial
import jax.numpy as jnp
from jax import jit
from typing import Dict, Any, Sequence
from spice.spectrum.limb_darkening import limb_darkening, get_limb_darkening_law_id
from spice.spectrum.line_profile import line_profile, get_line_profile_id

# -------------------------------------------------------
# Physical line broadening utilities
# -------------------------------------------------------
def thermal_width(line_center: float, Teff: float, atomic_mass: float) -> float:
    """Thermal Doppler width (Angstroms)."""
    k_B = 1.380649e-23  # J/K
    m_p = 1.6726219e-27
    c = 2.99792458e8
    v_th = jnp.sqrt(2 * k_B * Teff / (atomic_mass * m_p))
    return line_center * (v_th / c)

# -------------------------------------------------------
# Continuum (simple Planck-like slope)
# -------------------------------------------------------
def continuum(log_wavelengths: jnp.ndarray, Teff: float) -> jnp.ndarray:
    wl = 10.0 ** log_wavelengths
    B = (wl ** -5) / (jnp.exp(1.4388e8 / (wl * Teff)) - 1.0)
    return B / jnp.max(B)

# -------------------------------------------------------
# Main emulator
# -------------------------------------------------------
class PhysicalLineEmulator:
    """Single-line emulator with configurable limb darkening and realistic line profiles.

    Parameters now expected in spectral_parameters:
        [Teff [K], logg [cgs], mu]
    """

    def __init__(
        self,
        line_center: float = 5500.0,
        line_depth: float = 0.3,
        atomic_mass: float = 56.0,   # e.g. Fe
        v_micro: float = 2.0,        # km/s
        gamma0: float = 0.05,        # Lorentzian baseline [Å] near solar Teff/logg
        line_profile_kind: str = "voigt",
        limb_law: str = "linear",
        limb_coeffs: Sequence[float] = (0.6,),
        use_convective_shift: bool = False,
        v_conv0: float = 0.5,        # km/s at μ=0
        # gravity scaling knob for pressure broadening: gamma ∝ (g/g_sun)^alpha
        gamma_g_alpha: float = 0.5,
    ):
        self.line_center = line_center
        self.line_depth = line_depth
        self.atomic_mass = atomic_mass
        self.v_micro = v_micro
        self.gamma0 = gamma0
        self.line_profile_kind = get_line_profile_id(line_profile_kind)
        self.limb_law = get_limb_darkening_law_id(limb_law)
        self.limb_coeffs = jnp.array(limb_coeffs)
        self.use_convective_shift = use_convective_shift
        self.v_conv0 = v_conv0
        self.gamma_g_alpha = gamma_g_alpha

    @property
    def parameter_names(self):
        # Vector order is fixed: Teff, logg, mu
        return ["Teff", "logg", "mu"]

    def to_parameters(self, parameter_values: Dict[str, Any] | None = None) -> jnp.ndarray:
        if parameter_values is None:
            return jnp.array([5777.0, 4.44, 0.8], dtype=jnp.float64)
        Teff = float(parameter_values.get("Teff", 5777.0))
        logg = float(parameter_values.get("logg", 4.44))
        mu   = float(parameter_values.get("mu",   0.8))
        return jnp.array([Teff, logg, mu], dtype=jnp.float64)

    # ---------------------------------------------------
    # Core intensity calculation (new API: μ comes from parameters)
    # ---------------------------------------------------
    @partial(jit, static_argnums=(0,))
    def __intensity(self, log_wavelengths: jnp.ndarray, spectral_parameters: jnp.ndarray):
        # Unpack parameters
        Teff = spectral_parameters[0]
        logg = spectral_parameters[1]
        mu   = spectral_parameters[2]

        wl = 10.0 ** log_wavelengths
        cont = continuum(log_wavelengths, Teff)

        # --- Effective Gaussian width (thermal + microturbulence) ---
        sigma_th = thermal_width(self.line_center, Teff, self.atomic_mass)
        c_kms = 2.99792458e5  # km/s
        sigma_micro = self.v_micro * self.line_center / c_kms
        sigma = jnp.sqrt(sigma_th ** 2 + sigma_micro ** 2)

        # --- Lorentzian parameter (pressure + radiative proxy)
        # Temperature scaling (radiative damping trend) and gravity scaling (pressure broadening)
        # gamma ~ gamma0 * (Teff/5777)^-0.7 * (g/g_sun)^(gamma_g_alpha), with g/g_sun = 10^(logg-4.44)
        gamma_T = (Teff / 5777.0) ** (-0.7)
        g_ratio = 10.0 ** (logg - 4.44)
        gamma_g = g_ratio ** self.gamma_g_alpha
        gamma = self.gamma0 * gamma_T * gamma_g

        # --- Optional convective blueshift (μ-dependent)
        if self.use_convective_shift:
            delta_lambda = -self.line_center * (self.v_conv0 * (1.0 - mu) / c_kms)
        else:
            delta_lambda = 0.0
        line_center_mu = self.line_center + delta_lambda

        # --- Line profile ---
        prof = line_profile(
            wl,
            line_center_mu,
            sigma,
            self.line_depth,
            kind_id=self.line_profile_kind,
            gamma=gamma
        )

        # --- Limb darkening ---
        ld = limb_darkening(mu, self.limb_law, self.limb_coeffs)
        I = cont * prof * ld

        # Return stacked [line-attenuated intensity, pure continuum] at this μ
        return jnp.stack([I, cont * ld], axis=-1)

    def intensity(self, log_wavelengths: jnp.ndarray, mu: float, spectral_parameters: jnp.ndarray):
        """Compatibility layer for code that still passes μ separately.
        Builds a [Teff, logg, mu] vector without any Python branching."""
        sp = jnp.atleast_1d(spectral_parameters)
        # Pad with defaults so first two always exist (works for len 0, 1, 2, or >2)
        defaults = jnp.array([5777.0, 4.44], dtype=sp.dtype)
        padded = jnp.concatenate([sp, defaults], axis=0)

        teff = padded[0]           # sp[0] if present, else 5777.0
        logg = padded[1]           # sp[1] if present, else 4.44
        mu_v = jnp.asarray(mu, dtype=sp.dtype)

        sp3 = jnp.stack([teff, logg, mu_v])
        return self.__intensity(log_wavelengths, sp3)
