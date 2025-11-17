"""Utilities for applying a ring-shaped starspot model to SPICE labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import jax.numpy as jnp


@dataclass
class RingSpotConfig:
    """Configuration options for the analytic ring spot model.

    Parameters
    ----------
    sigma_umb_deg:
        Width (1σ) of the cool core in degrees.
    theta0_deg:
        Polar angle of the bright ring centre relative to the spot axis in degrees.
    sigma_plage_deg:
        Width (1σ) of the plage ring in degrees.
    umbra_delta:
        Additive perturbation applied inside the umbral core. Use a negative
        value to cool a temperature column or reduce an abundance.
    plage_delta:
        Additive perturbation applied to the plage ring. Use a positive value to
        brighten or enhance the chosen parameter.
    """

    sigma_umb_deg: float = 17.0
    theta0_deg: float = 50.0
    sigma_plage_deg: float = 8.0
    umbra_delta: float = -1100.0
    plage_delta: float = 150.0


def _angle_between(v1: jnp.ndarray, v2: jnp.ndarray) -> jnp.ndarray:
    """Return the angle (radians) between pairs of vectors."""

    if v2.ndim == 1:
        v2 = jnp.broadcast_to(v2, v1.shape)

    dot = jnp.sum(v1 * v2, axis=-1)
    dot = jnp.clip(dot, -1.0, 1.0)
    return jnp.arccos(dot)


def ring_spot_weights(
    n_hat: jnp.ndarray, s_hat: jnp.ndarray, cfg: RingSpotConfig
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return (umbra, plage) weights for each surface element."""

    theta = _angle_between(n_hat, s_hat)
    deg = jnp.pi / 180.0
    sigma_umb = cfg.sigma_umb_deg * deg
    theta0 = cfg.theta0_deg * deg
    sigma_plage = cfg.sigma_plage_deg * deg

    w_umb = jnp.exp(-0.5 * (theta / sigma_umb) ** 2)
    w_plage = jnp.exp(-0.5 * ((theta - theta0) / sigma_plage) ** 2)
    return w_umb, w_plage


def apply_ring_spot_parameter(
    base: jnp.ndarray | float,
    n_hat: jnp.ndarray,
    s_hat: jnp.ndarray,
    cfg: RingSpotConfig,
    umbra_delta: float | None = None,
    plage_delta: float | None = None,
) -> jnp.ndarray:
    """Apply the ring-spot perturbation to an arbitrary per-element field."""

    base = jnp.asarray(base)
    w_umb, w_plage = ring_spot_weights(n_hat, s_hat, cfg)

    if base.ndim == 0:
        base = jnp.full(w_umb.shape, base)

    du = cfg.umbra_delta if umbra_delta is None else umbra_delta
    dp = cfg.plage_delta if plage_delta is None else plage_delta
    return base + du * w_umb + dp * w_plage


def apply_ring_spot_temperature(
    T_base: jnp.ndarray | float, n_hat: jnp.ndarray, s_hat: jnp.ndarray, cfg: RingSpotConfig
) -> jnp.ndarray:
    """Backwards-compatible alias that targets the temperature column."""

    return apply_ring_spot_parameter(T_base, n_hat, s_hat, cfg)


def apply_ring_spot_to_labels(
    labels: Mapping[str, jnp.ndarray | float],
    n_hat: jnp.ndarray,
    s_hat: jnp.ndarray,
    cfg: RingSpotConfig,
    label_deltas: Mapping[str, Tuple[float, float]] | None = None,
) -> Dict[str, jnp.ndarray]:
    """Return a spotted copy of a label dictionary suitable for SPICE.

    Parameters
    ----------
    labels:
        Mapping of per-element label arrays. Must contain ``"T_eff"``.
    n_hat, s_hat:
        Surface-normal array and spot direction in stellar coordinates.
    label_deltas:
        Optional mapping from label names to ``(umbra_delta, plage_delta)``
        tuples. Defaults to modifying only ``"T_eff"`` using the values stored
        in ``cfg``.
    """

    spotted = dict(labels)
    deltas = label_deltas or {"T_eff": (cfg.umbra_delta, cfg.plage_delta)}

    for key, (du, dp) in deltas.items():
        if key in labels:
            spotted[key] = apply_ring_spot_parameter(labels[key], n_hat, s_hat, cfg, du, dp)

    return spotted


def ca_irt_scale_map(
    n_hat: jnp.ndarray,
    s_hat: jnp.ndarray,
    cfg: RingSpotConfig,
    plage_scale: float = 0.0,
    umbra_scale: float = 0.0,
) -> jnp.ndarray:
    """Return a per-element scaling map for arbitrary masked features."""

    w_umb, w_plage = ring_spot_weights(n_hat, s_hat, cfg)
    return 1.0 + plage_scale * w_plage + umbra_scale * w_umb


def apply_ca_irt_scaling_local(
    I_lambda: jnp.ndarray,
    I_cont: jnp.ndarray,
    S_j: float,
    mask_irt: jnp.ndarray,
) -> jnp.ndarray:
    """Scale Ca II IRT line depths for a single surface element."""

    I_lambda = jnp.asarray(I_lambda)
    I_cont = jnp.asarray(I_cont)
    mask_irt = jnp.asarray(mask_irt, dtype=bool)

    r = 1.0 - I_lambda[mask_irt] / I_cont[mask_irt]
    r_scaled = S_j * r
    return I_lambda.at[mask_irt].set(I_cont[mask_irt] * (1.0 - r_scaled))


__all__ = [
    "RingSpotConfig",
    "ring_spot_weights",
    "apply_ring_spot_parameter",
    "apply_ring_spot_temperature",
    "apply_ring_spot_to_labels",
    "ca_irt_scale_map",
    "apply_ca_irt_scaling_local",
]
