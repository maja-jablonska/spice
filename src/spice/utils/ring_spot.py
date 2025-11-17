"""Utilities for applying a ring-shaped starspot model to SPICE labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple

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
    deltaT_umb:
        Temperature decrement to apply to the umbral core in Kelvin.
    deltaT_plage:
        Temperature increment to apply to the plage ring in Kelvin.
    A_plage:
        Multiplicative scaling applied to Ca II IRT line depths within the ring.
    B_umb:
        Multiplicative scaling applied to Ca II IRT line depths within the umbra.
    dCa_plage:
        Abundance enhancement (dex) applied to the plage.
    dCa_umb:
        Abundance decrement (dex) applied to the umbra.
    """

    sigma_umb_deg: float = 17.0
    theta0_deg: float = 50.0
    sigma_plage_deg: float = 8.0
    deltaT_umb: float = 1100.0
    deltaT_plage: float = 150.0
    A_plage: float = 0.7
    B_umb: float = 0.3
    dCa_plage: float = 0.5
    dCa_umb: float = 0.0


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


def apply_ring_spot_temperature(
    T_base: jnp.ndarray | float, n_hat: jnp.ndarray, s_hat: jnp.ndarray, cfg: RingSpotConfig
) -> jnp.ndarray:
    """Apply the ring spot temperature perturbation to a base field."""

    T_base = jnp.asarray(T_base)
    w_umb, w_plage = ring_spot_weights(n_hat, s_hat, cfg)

    if T_base.ndim == 0:
        T_base = jnp.full(w_umb.shape, T_base)

    return T_base - cfg.deltaT_umb * w_umb + cfg.deltaT_plage * w_plage


def apply_ring_spot_ca_abundance(
    Ca_base: jnp.ndarray | float, n_hat: jnp.ndarray, s_hat: jnp.ndarray, cfg: RingSpotConfig
) -> jnp.ndarray:
    """Apply Ca abundance perturbations for the ring spot."""

    Ca_base = jnp.asarray(Ca_base)
    w_umb, w_plage = ring_spot_weights(n_hat, s_hat, cfg)

    if Ca_base.ndim == 0:
        Ca_base = jnp.full(w_umb.shape, Ca_base)

    dCa = cfg.dCa_plage * w_plage - cfg.dCa_umb * w_umb
    return Ca_base + dCa


def apply_ring_spot_to_labels(
    labels: Mapping[str, jnp.ndarray | float],
    n_hat: jnp.ndarray,
    s_hat: jnp.ndarray,
    cfg: RingSpotConfig,
    ca_keys: Iterable[str] | None = None,
) -> Dict[str, jnp.ndarray]:
    """Return a spotted copy of a label dictionary suitable for SPICE.

    Parameters
    ----------
    labels:
        Mapping of per-element label arrays. Must contain ``"T_eff"``.
    n_hat, s_hat:
        Surface-normal array and spot direction in stellar coordinates.
    ca_keys:
        Optional override for which keys should be treated as Ca abundances.
    """

    spotted = dict(labels)
    spotted["T_eff"] = apply_ring_spot_temperature(labels["T_eff"], n_hat, s_hat, cfg)

    keys = ca_keys if ca_keys is not None else ("Ca_H", "Ca_Fe", "Ca_over_Fe")
    for key in keys:
        if key in labels:
            spotted[key] = apply_ring_spot_ca_abundance(labels[key], n_hat, s_hat, cfg)
            break

    return spotted


def ca_irt_scale_map(
    n_hat: jnp.ndarray, s_hat: jnp.ndarray, cfg: RingSpotConfig
) -> jnp.ndarray:
    """Return a per-element Ca II IRT scaling map."""

    w_umb, w_plage = ring_spot_weights(n_hat, s_hat, cfg)
    return 1.0 + cfg.A_plage * w_plage - cfg.B_umb * w_umb


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
    "apply_ring_spot_temperature",
    "apply_ring_spot_ca_abundance",
    "apply_ring_spot_to_labels",
    "ca_irt_scale_map",
    "apply_ca_irt_scaling_local",
]
