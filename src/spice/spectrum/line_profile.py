import jax
import jax.numpy as jnp

KIND_MAP = {"gaussian": 0, "lorentzian": 1, "voigt": 2}

def get_line_profile_id(kind: str) -> int:
    return KIND_MAP[kind]


def _pseudo_voigt_shape(Δλ: jnp.ndarray, sigma: float, gamma: float) -> jnp.ndarray:
    """
    Pseudo-Voigt kernel shape S(Δλ) with S(0)=1, wings→0.

    Uses the Thompson, Cox & Hastings (1987) approximation.
    """
    # FWHM of Gaussian and Lorentzian
    ln2 = jnp.log(2.0)
    F_G = 2.0 * jnp.sqrt(2.0 * ln2) * sigma   # Gaussian FWHM
    F_L = 2.0 * gamma                         # Lorentzian FWHM

    # Overall FWHM of Voigt
    F = (F_L**5 + 2.69269 * F_L**4 * F_G + 2.42843 * F_L**3 * F_G**2
         + 4.47163 * F_L**2 * F_G**3 + 0.07842 * F_L * F_G**4 + F_G**5) ** (1.0 / 5.0)

    # Mixing parameter eta
    eta = 1.36603 * (F_L / F) - 0.47719 * (F_L / F) ** 2 + 0.11116 * (F_L / F) ** 3
    eta = jnp.clip(eta, 0.0, 1.0)

    x = 2.0 * Δλ / F  # dimensionless

    # Normalized Gaussian and Lorentzian parts with peak = 1
    G = jnp.exp(-4.0 * ln2 * x**2)
    L = 1.0 / (1.0 + x**2)

    S = eta * L + (1.0 - eta) * G  # still S(0)=1
    return S


@jax.jit
def line_profile(
    wavelengths: jnp.ndarray,
    center: float,
    sigma: float,
    depth: float,
    kind_id: int = 0,
    gamma: float = 0.05,
) -> jnp.ndarray:
    """
    Normalized absorption line profile (multiplicative).

    Returns I/I_cont with:
        wings ~ 1
        core  ~ 1 - depth
    """
    Δλ = wavelengths - center
    depth_clipped = jnp.clip(depth, 0.0, 1.0)

    def gaussian(_):
        # shape S: 1 at centre, 0 in wings
        return jnp.exp(-0.5 * (Δλ / sigma) ** 2)

    def lorentzian(_):
        return 1.0 / (1.0 + (Δλ / gamma) ** 2)

    def voigt(_):
        return _pseudo_voigt_shape(Δλ, sigma, gamma)

    # S(Δλ) ∈ [0, 1]
    shape = jax.lax.switch(kind_id, [gaussian, lorentzian, voigt], None)
    shape = jnp.clip(shape, 0.0, 1.0)

    # wings: S≈0 → 1 - d*0 = 1
    # core : S=1 → 1 - d*1 = 1 - d
    return 1.0 - depth_clipped * shape
