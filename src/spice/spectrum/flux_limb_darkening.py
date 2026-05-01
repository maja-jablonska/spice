"""Flux-conservation limb darkening: convert a disc-integrated flux F into
the specific intensity I(μ) at angle μ such that 2π·∫₀¹ I(μ)·μ dμ = F.

Distinct from `spice.spectrum.limb_darkening`, which exposes a different,
dimensionless f(μ) used by intensity emulators that already operate per-μ.
"""

from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import checkify
from jaxtyping import ArrayLike


def _linear(mu, coeffs):
    u = coeffs[0]
    return 1.0 - u * (1.0 - mu)


def _quadratic(mu, coeffs):
    u1, u2 = coeffs[0], coeffs[1]
    return 1.0 - (
        u1 * (1.0 - mu) +
        u2 * (1.0 - mu) ** 2
    )


def _nonlinear_4(mu, coeffs):
    c1, c2, c3, c4 = coeffs
    return 1.0 - (
        c1 * (1 - jnp.sqrt(mu)) +
        c2 * (1 - mu) +
        c3 * (1 - mu ** 1.5) +
        c4 * (1 - mu ** 2)
    )


# Flux-conservation normalisation: the factor 2π·∫₀¹ f(μ)·μ dμ that maps a
# disc-integrated flux F to the specific intensity at disc centre via
# I(μ) = F · f(μ) / norm. Derived analytically per law from the polynomial
# (or √μ for nonlinear_4) integrals.
def _linear_norm(coeffs):
    # f(μ)=1−u(1−μ); ∫₀¹ μ·f dμ = (3−u)/6 → norm = π(3−u)/3
    u = coeffs[0]
    return jnp.pi * (3.0 - u) / 3.0


def _quadratic_norm(coeffs):
    # f(μ)=1−u1(1−μ)−u2(1−μ)²; ∫₀¹ μ·f dμ = (6−2u1−u2)/12
    u1, u2 = coeffs[0], coeffs[1]
    return jnp.pi * (6.0 - 2.0 * u1 - u2) / 6.0


def _nonlinear_4_norm(coeffs):
    # f(μ)=1−c1(1−√μ)−c2(1−μ)−c3(1−μ^1.5)−c4(1−μ²);
    # ∫₀¹ μ·f dμ = 1/2 − c1/10 − c2/6 − 3c3/14 − c4/4
    c1, c2, c3, c4 = coeffs
    return 2.0 * jnp.pi * (
        0.5 - c1 / 10.0 - c2 / 6.0 - 3.0 * c3 / 14.0 - c4 / 4.0
    )


_LD_FUNCS = (_linear, _quadratic, _nonlinear_4)
_LD_NORMS = (_linear_norm, _quadratic_norm, _nonlinear_4_norm)
LAW_IDS = {
    'linear': 0,
    'quadratic': 1,
    'nonlinear_4': 2,
}


def limb_darkening(mu, law_id, coeffs):
    """Dimensionless f(μ) for the selected law.

    mu: [..., n_cells]
    law_id: int (0=linear, 1=quadratic, 2=nonlinear_4)
    coeffs: array-like (length 4; unused entries ignored)
    """
    return lax.switch(law_id, _LD_FUNCS, mu, coeffs)


def limb_darkening_norm(law_id, coeffs):
    """Flux-conservation factor 2π·∫₀¹ f(μ)·μ dμ for the selected law."""
    return lax.switch(law_id, _LD_NORMS, coeffs)


def get_limb_darkening_law_id(law: str) -> int:
    return LAW_IDS[law]


def _coerce_coeffs(coeffs: Optional[ArrayLike]) -> jnp.ndarray:
    if coeffs is None:
        return jnp.array([0.6, 0.6, 0.0, 0.0])
    arr = jnp.atleast_1d(jnp.asarray(coeffs))
    if arr.shape[0] < 4:
        arr = jnp.pad(arr, (0, 4 - arr.shape[0]), constant_values=0.0)
    return arr


def _ld_coeffs_predicates(coeffs: ArrayLike):
    """Return (predicate, message) pairs that callers can plug into either
    Python ``raise`` or ``checkify.check`` flows. Predicates are JAX scalars
    (or concrete bools, depending on input)."""
    finite = jnp.all(jnp.isfinite(coeffs))
    in_range = jnp.all(coeffs >= -1.0) & jnp.all(coeffs <= 2.0)
    norms = jnp.array([
        _linear_norm(coeffs),
        _quadratic_norm(coeffs),
        _nonlinear_4_norm(coeffs),
    ])
    norm_ok = jnp.all(norms > 1e-6)
    return [
        (finite, "ld_coeffs contains non-finite values"),
        (in_range, "ld_coeffs outside the typical [-1, 2] range"),
        (norm_ok, "ld_coeffs produce a near-zero flux-conservation norm; "
                  "intensity I(μ) = F·f(μ)/norm will diverge"),
    ]


def check_ld_coeffs(coeffs: ArrayLike) -> None:
    """Eager Python validation for limb-darkening coefficients.

    Skips silently when ``coeffs`` is a JAX tracer — ``checkify.check`` would
    raise during MLIR lowering unless the call site is wrapped with
    ``checkify.checkify``, which we don't want to require. Use
    :func:`checkify_ld_coeffs` instead for in-jit predicates."""
    if isinstance(coeffs, jax.core.Tracer):
        return
    arr = np.asarray(coeffs)
    for pred, msg in _ld_coeffs_predicates(arr):
        if not bool(pred):
            raise ValueError(f"{msg}: {arr}")


def checkify_ld_coeffs(coeffs: ArrayLike) -> None:
    """Register :func:`jax.experimental.checkify.check` predicates on
    ``coeffs``. Only meaningful when the surrounding call is wrapped with
    ``checkify.checkify(...)``::

        from jax.experimental import checkify
        from spice.spectrum.flux_limb_darkening import checkify_ld_coeffs

        def my_intensity(lw, mu, p, ld_coeffs):
            checkify_ld_coeffs(ld_coeffs)
            return em.intensity(lw, mu, p, ld_coeffs=ld_coeffs)

        err, flux = checkify.checkify(simulate_observed_flux)(my_intensity, m, lw,
                                                              ld_coeffs=coeffs)
        err.throw()
    """
    for pred, msg in _ld_coeffs_predicates(coeffs):
        checkify.check(pred, msg + ": {coeffs}", coeffs=coeffs)


class FluxLimbDarkening:
    """Bundle a limb-darkening law and its coefficients with an ``apply`` step
    that converts disc-integrated flux to specific intensity at angle μ while
    preserving the disc-integrated flux: I(μ) = F · f(μ) / (2π·∫₀¹ f(μ′)·μ′ dμ′).
    """

    def __init__(
        self,
        law: Optional[str] = None,
        coeffs: Optional[ArrayLike] = None,
    ):
        self.law_id = get_limb_darkening_law_id(law if law is not None else 'linear')
        self.coeffs = _coerce_coeffs(coeffs)
        # Register checks once at construction. Inside an outer ``vmap`` the
        # constructor is traced once, so this is a single-shot predicate per
        # ``simulate_observed_flux`` call rather than per mesh element.
        check_ld_coeffs(self.coeffs)

    def apply(self, flux: ArrayLike, mu: ArrayLike) -> ArrayLike:
        f = limb_darkening(mu, self.law_id, self.coeffs)
        norm = limb_darkening_norm(self.law_id, self.coeffs)
        return flux * f / norm


def apply_flux_limb_darkening(
    flux: ArrayLike,
    mu: ArrayLike,
    ld_law: Optional[str] = None,
    ld_coeffs: Optional[ArrayLike] = None,
) -> ArrayLike:
    """Convenience wrapper: build a one-shot :class:`FluxLimbDarkening` and
    apply it. Centralises the (law, coeffs) → I(μ) pattern used by every
    flux-emulator's ``intensity`` method."""
    return FluxLimbDarkening(ld_law, ld_coeffs).apply(flux, mu)


class LdBoundIntensity:
    """Hashable adapter that binds ``ld_law`` / ``ld_coeffs`` into an
    ``intensity_fn`` so the resulting callable produces a stable jit cache
    key when reused across calls.

    A bare ``functools.partial`` would also work functionally, but its
    ``__hash__`` is identity-based, so passing it as a static argument to
    :func:`jax.jit` triggers a recompile on every public call. This adapter
    hashes by ``(intensity_fn, ld_law, tuple(ld_coeffs))`` instead.
    """

    __slots__ = ("fn", "ld_law", "_coeffs", "_coeffs_key")

    def __init__(self, fn, ld_law: Optional[str], ld_coeffs: Optional[ArrayLike]):
        self.fn = fn
        self.ld_law = ld_law
        if ld_coeffs is None:
            self._coeffs = None
            self._coeffs_key = None
        else:
            self._coeffs = jnp.asarray(ld_coeffs)
            # numpy round-trip → tuple-of-floats key for cross-call hashing.
            self._coeffs_key = tuple(np.asarray(ld_coeffs, dtype=np.float64)
                                       .reshape(-1).tolist())

    def __call__(self, *args, **kwargs):
        if self.ld_law is not None:
            kwargs.setdefault("ld_law", self.ld_law)
        if self._coeffs is not None:
            kwargs.setdefault("ld_coeffs", self._coeffs)
        return self.fn(*args, **kwargs)

    def __hash__(self) -> int:
        return hash((self.fn, self.ld_law, self._coeffs_key))

    def __eq__(self, other) -> bool:
        if not isinstance(other, LdBoundIntensity):
            return False
        return (
            self.fn == other.fn
            and self.ld_law == other.ld_law
            and self._coeffs_key == other._coeffs_key
        )

    def __repr__(self) -> str:  # nicer tracebacks
        return (
            f"LdBoundIntensity(fn={self.fn!r}, ld_law={self.ld_law!r}, "
            f"ld_coeffs={self._coeffs_key!r})"
        )
