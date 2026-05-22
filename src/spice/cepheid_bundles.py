"""Reusable bundle classes for the Cepheid notebooks.

Defining `CepheidBundle` and `LineSpectra` in a real module (rather than in a
notebook cell) means pickle stores their fully-qualified path as
``cepheid_bundles.CepheidBundle`` instead of ``__main__.CepheidBundle``.
That makes the resulting `.pkl` files loadable from any other notebook,
script, or REPL session — so the build/analysis split below can actually
round-trip through disk.

Public API:

- :class:`CepheidBundle` — one model variant (mesh + per-phase snapshots + t=0 template).
- :class:`LineSpectra` — synthetic spectra for one line across all phases.
- :func:`build_bundle` — construct an icosphere mesh, attach pulsation+rotation, evaluate over time.
- :func:`apply_phase_params` — push phase-dependent Teff/log g into a bundle's snapshots.
- :func:`simulate_line_spectra` — synthesise spectra for a list of line centres.
- :func:`save_pickle` / :func:`load_pickle` — single-call round-trip for any of the above.
"""
from __future__ import annotations

import os
import pickle
from typing import Any, Callable, NamedTuple, Optional, Sequence

import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from spice.models import IcosphereModel
from spice.models.mesh_transform import (
    add_pulsation,
    add_rotation,
    evaluate_pulsations,
    evaluate_rotation,
)
from spice.spectrum import simulate_observed_flux


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
class CepheidBundle(NamedTuple):
    """Three-field record for one model variant.

    Attributes
    ----------
    pulsating
        The base mesh with `add_pulsation` + `add_rotation` attached, before
        any time-dependent evaluation. Useful if you want to re-evaluate at
        new phases.
    snapshots
        List of meshes, one per phase in the build-time `timeseries`.
    template
        The mesh after `evaluate_rotation` at t=0; used as the spectral
        template against which other phases are cross-correlated.
    """

    pulsating: Optional[Any] = None
    snapshots: Optional[list] = None
    template: Optional[Any] = None


class LineSpectra(NamedTuple):
    """Synthetic spectra for one line, across all phase snapshots."""

    wavelengths: Any  # shape (steps,)
    spectra: list     # per-snapshot flux arrays of shape (steps, 2)
    template: Any     # t=0 flux array of shape (steps, 2)


# ---------------------------------------------------------------------------
# Build / modify
# ---------------------------------------------------------------------------
def build_bundle(
    emulator,
    base_params,
    *,
    fourier_params,
    period: float,
    timeseries,
    n_mesh: int = 5000,
    radius: float = 43.06,
    mass: float = 5.26,
    rotation_velocity: float = 0.0,
    max_fourier_order: int = 10,
    param_names_attr: str = "stellar_parameter_names",
    desc: str = "Evaluating",
) -> CepheidBundle:
    """Construct an icosphere mesh, attach pulsation+rotation, evaluate it across `timeseries`."""
    param_names = getattr(emulator, param_names_attr, None) or emulator.parameter_names
    base = IcosphereModel.construct(
        n_mesh, radius, mass, base_params, param_names,
        max_fourier_order=max_fourier_order,
    )
    pulsating = add_rotation(
        add_pulsation(
            base, m_order=0, l_degree=0,
            period=period, fourier_series_parameters=fourier_params,
        ),
        rotation_velocity,
    )
    snapshots = [
        evaluate_rotation(evaluate_pulsations(pulsating, t), t)
        for t in tqdm(timeseries, desc=desc, leave=False)
    ]
    template = evaluate_rotation(pulsating, 0)
    return CepheidBundle(pulsating=pulsating, snapshots=snapshots, template=template)


def apply_phase_params(
    bundle: CepheidBundle,
    phases: Sequence[float],
    modifiers: Sequence[tuple[int, Callable[[float], float]]],
) -> CepheidBundle:
    """Apply phase-dependent parameter modifiers to a bundle's snapshots.

    `modifiers` is a list of `(param_index, fn)` pairs; `fn(phase)` returns
    the new value of that parameter for that snapshot. Returns a new bundle
    (NamedTuples are immutable).
    """
    from spice.utils.parameters import modify_mesh_parameter_from_array

    if bundle.snapshots is None:
        raise ValueError("Bundle has no snapshots to modify.")
    new_snapshots = []
    for snap, ph in zip(bundle.snapshots, phases):
        for idx, fn in modifiers:
            snap = modify_mesh_parameter_from_array(snap, idx, fn(ph))
        new_snapshots.append(snap)
    return bundle._replace(snapshots=new_snapshots)


# ---------------------------------------------------------------------------
# Spectrum synthesis
# ---------------------------------------------------------------------------
def simulate_line_spectra(
    snapshots: list,
    template_snapshot,
    intensity_fn: Callable,
    line_centers: Sequence[float],
    *,
    line_width: float = 2.0,
    steps: int = 500,
    desc: str = "lines",
    ld_law: str | None = None,
    ld_coeffs=None,
    chunk_size: int | None = None,
    wavelengths_chunk_size: int | None = None,
) -> dict[float, LineSpectra]:
    """Synthesise spectra for each line center across all snapshots + a t=0 template.

    ``ld_law`` and ``ld_coeffs`` are forwarded to :func:`simulate_observed_flux`,
    which binds them into ``intensity_fn`` (used with the flux emulator's
    ``intensity`` method to apply a flux-conservation limb-darkening law per
    call without rebuilding the bundle).

    ``chunk_size`` / ``wavelengths_chunk_size`` (when set) override the
    ``simulate_observed_flux`` defaults — useful for fitting the wavelength
    scan into V100/A100 HBM when synthesising wide windows.
    """
    out: dict[float, LineSpectra] = {}
    extra = {}
    if ld_law is not None:
        extra["ld_law"] = ld_law
    if ld_coeffs is not None:
        extra["ld_coeffs"] = ld_coeffs
    if chunk_size is not None:
        extra["chunk_size"] = chunk_size
    if wavelengths_chunk_size is not None:
        extra["wavelengths_chunk_size"] = wavelengths_chunk_size
    for lc in tqdm(line_centers, desc=desc):
        wl = jnp.linspace(lc - line_width, lc + line_width, steps)
        log_wl = jnp.log10(wl)
        per_snapshot = [
            simulate_observed_flux(intensity_fn, m, log_wl, **extra)
            for m in tqdm(snapshots, desc=f"line {lc:.2f}", leave=False)
        ]
        template = simulate_observed_flux(intensity_fn, template_snapshot, log_wl, **extra)
        out[float(lc)] = LineSpectra(
            wavelengths=np.asarray(wl),
            spectra=[np.asarray(s) for s in per_snapshot],
            template=np.asarray(template),
        )
    return out


# ---------------------------------------------------------------------------
# Pickle helpers (one-call round-trip)
# ---------------------------------------------------------------------------
def save_pickle(obj, path: str) -> None:
    # Atomic write: tmp + fsync + os.replace so a SIGKILL/OOM/node crash
    # mid-dump can't leave a truncated .pkl that future resume runs would
    # either mistake for complete or crash on at load time. The tmp filename
    # includes the pid so concurrent workers writing different paths can't
    # collide on the tmp slot.
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())

    size = os.path.getsize(tmp_path)
    if size == 0:
        os.remove(tmp_path)
        raise IOError(f"Failed to save {path}: generated file is 0 bytes.")

    os.replace(tmp_path, path)
    print(f"    [save] {os.path.basename(path)}: {size / 1024**2:.1f} MB")


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
