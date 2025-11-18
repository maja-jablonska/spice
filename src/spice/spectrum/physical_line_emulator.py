from functools import partial
from typing import Dict, Any, Sequence
import wave

import jax.numpy as jnp
from jax import jit

from spice.spectrum.limb_darkening import limb_darkening, get_limb_darkening_law_id
from spice.spectrum.line_profile import line_profile, get_line_profile_id


# -------------------------------------------------------
# Physical constants (cgs)
# -------------------------------------------------------
# h     : Planck constant            [erg s]
# c     : speed of light             [cm s^-1]
# k_B   : Boltzmann constant         [erg K^-1]
# m_p   : proton mass                [g]
H_CGS  = 6.62607015e-27     # erg s
C_CGS  = 2.99792458e10      # cm s^-1
KB_CGS = 1.380649e-16       # erg K^-1
M_P    = 1.6726219e-24      # g
C_KMS  = 2.99792458e5       # km s^-1


# -------------------------------------------------------
# Thermal + microturbulent Doppler width
# -------------------------------------------------------
def thermal_width(line_center_A: float, Teff: float, atomic_mass: float) -> float:
    """
    Thermal Doppler width (1-sigma) at line_center.

    Parameters
    ----------
    line_center_A : float
        Line centre wavelength [Å].
    Teff : float
        Effective temperature [K].
    atomic_mass : float
        Atomic mass in units of proton mass (i.e. A for an isotope).

    Returns
    -------
    sigma_th : float
        Thermal Doppler width [Å].
    """
    # Convert to cgs
    # v_th = sqrt(2 k_B T / (A m_p))     [cm s^-1]
    v_th = jnp.sqrt(2.0 * KB_CGS * Teff / (atomic_mass * M_P))

    # Doppler shift Δλ/λ = v/c
    # σ_λ = λ * (v_th / c)               [cm]
    sigma_lambda_cm = (line_center_A * 1e-8) * (v_th / C_CGS)

    # Convert back to Å
    return sigma_lambda_cm * 1e8


# -------------------------------------------------------
# Continuum (Planck function)
# -------------------------------------------------------
def planck_continuum(
    log_wavelengths: jnp.ndarray,
    Teff: float,
) -> jnp.ndarray:
    """
    Planck continuum B_λ at the stellar surface.

    Parameters
    ----------
    log_wavelengths : array_like
        log10(λ / Å), i.e. λ in Angstroms.
    Teff : float
        Effective temperature [K].

    Returns
    -------
    B_lambda : jnp.ndarray
        Monochromatic specific intensity B_λ at the surface
        in units of:

            erg s^-1 cm^-2 Å^-1 sr^-1

        (per unit wavelength, per unit area, per unit time, per steradian).
    """
    # λ [Å] → λ [cm]
    lam_A  = 10.0 ** log_wavelengths
    lam_cm = lam_A * 1e-8

    # x = h c / (λ k_B T)  (dimensionless)
    x = (H_CGS * C_CGS) / (lam_cm * KB_CGS * Teff)

    # Planck function B_λ [erg s^-1 cm^-2 cm^-1 sr^-1]
    B_lambda_cm = (2.0 * H_CGS * C_CGS**2) / (
        lam_cm**5 * (jnp.exp(x) - 1.0)
    )

    # Convert from per-cm to per-Å:
    #   B_λ(Å) dλ_Å = B_λ(cm) dλ_cm
    #   dλ_cm = 1e-8 dλ_Å  →  B_λ(Å) = B_λ(cm) * 1e-8
    B_lambda_A = B_lambda_cm * 1e-8

    return B_lambda_A  # [erg s^-1 cm^-2 Å^-1 sr^-1]


# -------------------------------------------------------
# Main emulator
# -------------------------------------------------------
class PhysicalLineEmulator:
    """
    Single-line physical emulator with limb darkening and Voigt-like profiles.

    Parameters
    ----------
    line_center : float, default 5500.0
        Rest wavelength of the line [Å].
    line_depth : float, default 0.3
        Central line depth (0 < depth < 1); multiplicative absorption.
    atomic_mass : float, default 56.0
        Atomic mass in units of proton mass (e.g. 56 for Fe).
    v_micro : float, default 2.0
        Microturbulent velocity [km s^-1].
    gamma0 : float, default 0.05
        Baseline Lorentzian width [Å] near solar Teff/logg.
    line_profile_kind : {"gaussian", "lorentzian", "voigt"}, default "voigt"
        Profile kind identifier used by `spice.spectrum.line_profile`.
    limb_law : str, default "linear"
        Limb darkening law name for `spice.spectrum.limb_darkening`.
    limb_coeffs : sequence of float, default (0.6,)
        Limb darkening coefficients.
    use_convective_shift : bool, default False
        Whether to include a simple μ-dependent convective blueshift.
    v_conv0 : float, default 0.5
        Convective blueshift amplitude at μ = 0 [km s^-1].
    gamma_g_alpha : float, default 0.5
        Exponent for gravity scaling of Lorentzian width.

    Spectral parameter vector
    -------------------------
    The `spectral_parameters` vector is always ordered as:

        [Teff [K], logg [cgs], mu]

    Output
    ------
    For an input wavelength grid of length n, all `intensity(...)` calls return
    an array of shape [n, 2]:

        [:, 0]  full line intensity with limb darkening
        [:, 1]  pure continuum intensity with limb darkening

    Units of both columns:

        erg s^-1 cm^-2 Å^-1 sr^-1

    (monochromatic specific intensity at the stellar surface, along
     the ray at cosine μ).
    """

    def __init__(
        self,
        line_center: float = 5500.0,   # [Å]
        line_depth: float = 0.3,
        atomic_mass: float = 56.0,     # in units of proton mass
        v_micro: float = 2.0,          # [km s^-1]
        gamma0: float = 0.05,          # [Å]
        line_profile_kind: str = "voigt",
        limb_law: str = "linear",
        limb_coeffs: Sequence[float] = (0.6,),
        use_convective_shift: bool = False,
        v_conv0: float = 0.5,          # [km s^-1] at μ = 0
        gamma_g_alpha: float = 0.5,
    ):
        self.line_center = float(line_center)
        self.line_depth = float(line_depth)
        self.atomic_mass = float(atomic_mass)
        self.v_micro = float(v_micro)
        self.gamma0 = float(gamma0)

        self.line_profile_kind = get_line_profile_id(line_profile_kind)
        self.limb_law = get_limb_darkening_law_id(limb_law)
        self.limb_coeffs = jnp.array(limb_coeffs, dtype=jnp.float64)

        self.use_convective_shift = bool(use_convective_shift)
        self.v_conv0 = float(v_conv0)
        self.gamma_g_alpha = float(gamma_g_alpha)

    # ------------------------------------------------------------------
    # Parameter handling
    # ------------------------------------------------------------------
    @property
    def parameter_names(self):
        # Fixed ordering
        return ["Teff", "logg", "mu"]

    def to_parameters(
        self,
        parameter_values: Dict[str, Any] | None = None,
    ) -> jnp.ndarray:
        """
        Convert a parameter dict {name: value} to a 3-element vector
        [Teff, logg, mu], all as float64.
        """
        if parameter_values is None:
            return jnp.array([5777.0, 4.44, 0.8], dtype=jnp.float64)

        Teff = float(parameter_values.get("Teff", 5777.0))
        logg = float(parameter_values.get("logg", 4.44))
        mu   = float(parameter_values.get("mu",   0.8))
        return jnp.array([Teff, logg, mu], dtype=jnp.float64)

    # ------------------------------------------------------------------
    # Core intensity calculation
    # ------------------------------------------------------------------
    @partial(jit, static_argnums=(0,))
    def _intensity_core(
        self,
        log_wavelengths: jnp.ndarray,
        spectral_parameters: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Core calculation.

        Parameters
        ----------
        log_wavelengths : jnp.ndarray
            log10(λ / Å), shape [n].
        spectral_parameters : jnp.ndarray
            [Teff [K], logg [cgs], mu], shape [3].

        Returns
        -------
        intensities : jnp.ndarray
            Shape [n, 2], columns:
                0 : full line intensity with limb darkening
                1 : continuum intensity with limb darkening
            Units:
                erg s^-1 cm^-2 Å^-1 sr^-1
        """
        # Unpack parameters
        Teff = spectral_parameters[0]
        logg = spectral_parameters[1]
        mu   = spectral_parameters[2]

        # Wavelength grid [Å]
        wl_A = 10.0 ** log_wavelengths

        # Continuum specific intensity (no limb darkening yet)
        I_cont = planck_continuum(log_wavelengths, Teff)  # [erg s^-1 cm^-2 Å^-1 sr^-1]

        # --- Effective Gaussian width (thermal + microturbulence) ---
        sigma_th = thermal_width(self.line_center, Teff, self.atomic_mass)  # [Å]
        sigma_micro = self.v_micro * self.line_center / C_KMS              # [Å]
        sigma = jnp.sqrt(sigma_th**2 + sigma_micro**2)                      # [Å]

        # --- Lorentzian width scaling (temperature + gravity) ---
        # gamma ~ gamma0 * (Teff/5777)^-0.7 * (g/g_sun)^(gamma_g_alpha)
        gamma_T = (Teff / 5777.0) ** (-0.7)
        g_ratio = 10.0 ** (logg - 4.44)
        gamma_g = g_ratio**self.gamma_g_alpha
        gamma = self.gamma0 * gamma_T * gamma_g  # [Å]

        # --- μ-dependent convective blueshift [Å] ---
        if self.use_convective_shift:
            delta_lambda = -self.line_center * (self.v_conv0 * (1.0 - mu) / C_KMS)
        else:
            delta_lambda = 0.0

        line_center_mu = self.line_center + delta_lambda  # [Å]

        # Limb darkening factor at this μ (dimensionless)
        ld = limb_darkening(mu, self.limb_law, self.limb_coeffs)  # dimensionless

        # --- Line profile kernel (Voigt / etc.) ---
        # SPICE's `line_profile` returns a profile with units baked into the amplitude,
        # but here we only need its *shape* for building a multiplicative absorption.
        raw_profile = line_profile(
            wl_A,
            line_center_mu,
            sigma,
            self.line_depth,
            kind_id=self.line_profile_kind,
            gamma=gamma,
        )

        # raw_profile is multiplicative: wings≈1, core≈1-depth
        prof = raw_profile

        I_line_mu = I_cont * prof * ld
        I_cont_mu = I_cont * ld
        return jnp.stack([I_line_mu, I_cont_mu], axis=-1)



    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def intensity(
        self,
        log_wavelengths: jnp.ndarray,
        mu: float,
        spectral_parameters: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Public interface compatible with older code that passes μ separately.

        Parameters
        ----------
        log_wavelengths : jnp.ndarray
            log10(λ / Å), shape [n].
        mu : float
            Cosine of the viewing angle (0 <= μ <= 1).
        spectral_parameters : jnp.ndarray
            Can be:
              - [],            → [5777, 4.44, mu]
              - [Teff],        → [Teff, 4.44, mu]
              - [Teff, logg],  → [Teff, logg, mu]
              - [Teff, logg, mu]
            All values are assumed in [K, cgs, dimensionless].

        Returns
        -------
        intensities : jnp.ndarray
            Shape [n, 2], columns:
                0 : full line intensity with limb darkening
                1 : continuum intensity with limb darkening
            Units:
                erg s^-1 cm^-2 Å^-1 sr^-1
        """
        sp = jnp.atleast_1d(spectral_parameters)
        if sp.shape[0] == 0:
            dtype = jnp.float64
            teff = jnp.array(5777.0, dtype=dtype)
            logg = jnp.array(4.44, dtype=dtype)
        else:
            dtype = sp.dtype
            teff = sp[0]
            logg = sp[1] if sp.shape[0] > 1 else jnp.array(4.44, dtype=dtype)

        mu_v = jnp.squeeze(jnp.asarray(mu, dtype=dtype))

        sp3 = jnp.stack([teff, logg, mu_v])  # [3]
        return self._intensity_core(log_wavelengths, sp3)
