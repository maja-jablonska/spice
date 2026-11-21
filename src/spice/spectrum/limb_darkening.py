"""Dimensionless limb-darkening laws f(mu) for per-mu intensity emulators.

This module returns the *dimensionless* factor f(mu) = I(mu) / I(1) for a chosen
law; emulators that already produce a per-mu specific intensity multiply by it
directly. It is deliberately **distinct** from
:mod:`spice.spectrum.flux_limb_darkening`, which instead converts a
disc-integrated *flux* into a specific intensity with flux-conservation
normalisation, I(mu) = F * f(mu) / (2*pi * integral of f(mu')*mu' dmu').

The two modules therefore use different ``LAW_IDS`` mappings on purpose:

- here: ``{linear:0, quadratic:1, square_root:2, logarithmic:3, nonlinear:4}``
  (5 laws; ``nonlinear`` is the Claret 4-parameter law);
- ``flux_limb_darkening``: ``{linear:0, quadratic:1, nonlinear_4:2}``.

Always obtain a law id from the matching module's ``get_limb_darkening_law_id``
rather than hard-coding the integer.
"""

import jax
import jax.numpy as jnp

LAW_IDS = {"linear": 0, "quadratic": 1, "square_root": 2, "logarithmic": 3, "nonlinear": 4}

def get_limb_darkening_law_id(law: str) -> int:
    """Integer id of a limb-darkening law for :func:`limb_darkening`.

    Args:
        law (str): ``"linear"``, ``"quadratic"``, ``"square_root"``,
            ``"logarithmic"`` or ``"nonlinear"`` (Claret 4-parameter).

    Returns:
        int: The law id. Raises ``KeyError`` for an unknown law.
    """
    return LAW_IDS[law]

@jax.jit
def limb_darkening(mu, law_id: int, coeffs):
    """Dimensionless limb-darkening factor f(mu) = I(mu) / I(1).

    Args:
        mu: Cosine(s) of the angle between the surface normal and the direction
            to the observer; clipped to [1e-6, 1].
        law_id (int): From :func:`get_limb_darkening_law_id`.
        coeffs: Law coefficients: ``a`` (linear), ``a, b`` (quadratic,
            square-root, logarithmic) or ``a1..a4`` (nonlinear). Up to 4 values.

    Returns:
        f(mu), with the shape of ``mu``. The laws are ``1 - a(1-mu)``,
        ``1 - a(1-mu) - b(1-mu)^2``, ``1 - a(1-mu) - b(1-sqrt(mu))``,
        ``1 - a(1-mu) - b mu ln(mu)`` and ``1 - sum_i a_i (1 - mu^(i/2))``.
    """
    mu = jnp.clip(mu, 1e-6, 1.0)
    # Pad coeffs to length 4 for JIT shape consistency
    coeffs = jnp.pad(coeffs, (0, 4 - coeffs.shape[0]))

    a, b = coeffs[0], coeffs[1]

    def linear():
        return 1.0 - a * (1.0 - mu)

    def quadratic():
        return 1.0 - a * (1.0 - mu) - b * (1.0 - mu) ** 2

    def square_root():
        return 1.0 - a * (1.0 - mu) - b * (1.0 - jnp.sqrt(mu))

    def logarithmic():
        return 1.0 - a * (1.0 - mu) - b * mu * jnp.log(mu)

    def nonlinear():
        # Claret 4-param: I/I(1) = 1 - Σ a_i * (1 - μ^(i/2))
        idx = jnp.arange(4)
        mu_exp = mu[..., None] ** ((idx + 1) / 2.0)
        term = coeffs[:4] * (1.0 - mu_exp)
        return 1.0 - jnp.sum(term, axis=-1)

    return jax.lax.switch(
        law_id, [linear, quadratic, square_root, logarithmic, nonlinear]
    )
