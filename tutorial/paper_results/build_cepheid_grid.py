#!/usr/bin/env python
"""Build a grid of Cepheid bundles + spectra following ``cepheid_build_vmicro.ipynb``.

For each entry in the configured Cepheid grid this script:

  1. Loads up to four spectrum emulators -- ``IntensityLazyZarrInterpolator``
     (zarr line intensity grid, mu-conditioned), ``FluxLazyZarrInterpolator``
     (zarr flux grid, applies a linear LD law at call time), and two
     pretrained aemu intensity bundles (HARPS-resolution and a small
     iron-line bundle). LD coefficients are bound per call at synthesis
     time, so we don't rebuild emulators or bundles per coefficient.
  2. Constructs up to four mesh bundles per Cepheid -- one per emulator --
     using the Cepheid's stellar parameters and a Fourier-decomposed radius
     pulsation template. Each emulator gets its own bundle because the
     mesh stores stellar parameters in the emulator's training order, which
     differs across the four.
  3. Pushes phase-dependent Teff, logg and vmicro into every snapshot
     (vmicro from the Luck 2018 delta Cep fit; Teff/logg from SPIPS3).
     Modifier indices are looked up by name from each emulator's
     ``stellar_parameter_names``; emulators that don't carry vmicro just
     skip that modifier.
  4. Synthesises 5000-5020 Å spectra for ``with_ld`` (zarr intensity bundle)
     plus a 20-point linear LD coefficient sweep on the zarr flux bundle,
     plus one variant per aemu bundle.
  5. Pickles ``<name>_bundles.pkl`` and ``<name>_spectra.pkl`` to ``--out-dir``.

The default grid spans the classical-Cepheid period range (3-30 d) using the
Bono et al. 2001 PR relation for radius and mass, and the Delta-Cep SPIPS3
template for the pulsation shape and phase-resolved Teff/logg trends. δ Cep
appears twice — ``cep_DeltaCep`` (12 km/s rotation) and ``cep_DeltaCep_norot``
— so rotating and non-rotating cases are built without a custom CSV. Override
or extend the grid via ``--config CSV`` (see :func:`load_grid_csv`).

Synthesis is the slow step (~1-2 min per variant on CPU; 23 variants by
default, so ~30-45 min per Cepheid). The script is restartable: existing
per-Cepheid pickles in ``--out-dir`` are skipped unless ``--force`` is given.
Pass ``--skip-aemu`` to drop the two aemu bundles (the
``astro_emulators_toolkit`` extra is then optional) or ``--skip-zarr`` to
drop the two zarr bundles.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

# CPU-only on macOS, matches src/spice/__init__.py
os.environ.setdefault("JAX_PLATFORMS", "cpu")

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(HERE))

import numpy as np
import jax.numpy as jnp
from astropy.io import fits
from astropy.table import Table
from scipy import optimize
from scipy.interpolate import interp1d

from spice.spectrum.lazy_zarr_interpolator import (
    IntensityLazyZarrInterpolator,
    FluxLazyZarrInterpolator,
)
from cepheid_bundles import (  # noqa: E402  (sys.path patched above)
    CepheidBundle,
    LineSpectra,
    build_bundle,
    apply_phase_params,
    simulate_line_spectra,
    save_pickle,
)


# --- defaults that match the build notebook --------------------------------
# Gadi paths used by the cepheid_build_vmicro.ipynb notebook.
LINE_INTERP_PATH = "/g/data/y89/mj8805/fe_regular_nlte_big.zarr"
FLUX_INTERP_PATH = "/g/data/y89/mj8805/fe_nlte_flux_big.zarr/regular_synthesized_spectra.zarr"

# Pretrained aemu (astro_emulators_toolkit) bundles. Both expose
# ``IntensityPretrainedAemuSpectrumEmulator``. Defaults match
# benchmarks/benchmark_spice_components.py.
DEFAULT_HARPS_AEMU = "RozanskiT/TPayne-spice-harps"
DEFAULT_IRON_AEMU  = "RozanskiT/TPayne-spice-small-random"

# Bundle variant keys that this script knows how to build. Used by `--bundles`
# / `--skip-aemu` / `--skip-zarr`. Order is the build order.
ALL_BUNDLES = ("intensity", "flux_no_ld", "harps", "iron_line")
ZARR_BUNDLES = ("intensity", "flux_no_ld")
AEMU_BUNDLES = ("harps", "iron_line")

INTENSITY_PARAMS = ["teff", "logg", "[Fe/H]", "vmicro", "[a/Fe]",
                    "[C/Fe]", "[N/Fe]", "[O/Fe]", "[r/Fe]", "[s/Fe]", "mu"]
FLUX_PARAMS      = ["teff", "logg", "[Fe/H]", "vmicro", "[a/Fe]",
                    "[C/Fe]", "[N/Fe]", "[O/Fe]", "[r/Fe]", "[s/Fe]"]

SOLAR_INTENSITY = np.array([5777, 4.44, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
SOLAR_FLUX      = np.array([5777, 4.44, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# 20-point LD sweep across [0, 1]; spacing is 1/19 ≈ 0.0526, so 3 decimals in
# the formatted key keep every coefficient distinct.
LD_COEFFS = np.linspace(0.0, 1.0, 20)
LD_FMT = ".3f"
DEFAULT_PULSATION_FITS = HERE / "delta_cep.fits"

# δ Cep projected rotation used in cepheid_grid.csv / cepheid_analysis.ipynb.
DELTA_CEP_ROTATION_KMS = 12.0

WL_MIN, WL_MAX = 5000.0, 5020.0
WL_STEPS = 2000
N_PHASES = 100
N_FOURIER_TERMS = 8


# ---------------------------------------------------------------------------
# Cepheid configuration
# ---------------------------------------------------------------------------
@dataclass
class CepheidConfig:
    """Stellar + pulsation parameters for one Cepheid in the grid."""

    name: str
    period: float                                  # days
    radius: float                                  # R_sun (mean)
    mass: float                                    # M_sun
    teff: Optional[float] = None                   # phase-0 Teff (K); None → take from fits
    logg: Optional[float] = None                   # phase-0 logg;     None → take from fits
    fe_h: float = 0.06
    vmicro: float = 3.5
    # Abundances default to solar (0). The vmicro notebook only sets teff,
    # logg, [Fe/H], vmicro on stellar_params_dict; everything else falls back
    # to the emulator's solar_parameters via parameter_helper. Override here
    # (or in the CSV) for non-solar abundance studies.
    a_fe: float = 0.0
    c_fe: float = 0.0
    n_fe: float = 0.0
    o_fe: float = 0.0
    r_fe: float = 0.0
    s_fe: float = 0.0
    rotation_velocity: float = 0.0                 # km/s; 0 = no rotation
    n_vertices: Optional[int] = None               # None → fall back to --n-mesh
    pulsation_fits: Path = field(default_factory=lambda: DEFAULT_PULSATION_FITS)

    def stellar_params_dict(self, teff_default: float, logg_default: float) -> dict:
        return {
            "teff":   self.teff if self.teff is not None else teff_default,
            "logg":   self.logg if self.logg is not None else logg_default,
            "[Fe/H]": self.fe_h,
            "vmicro": self.vmicro,
            "[a/Fe]": self.a_fe,
            "[C/Fe]": self.c_fe,
            "[N/Fe]": self.n_fe,
            "[O/Fe]": self.o_fe,
            "[r/Fe]": self.r_fe,
            "[s/Fe]": self.s_fe,
        }


def _delta_cep_configs() -> tuple[CepheidConfig, CepheidConfig]:
    """δ Cep literature parameters, rotating and non-rotating (cepheid_grid.csv)."""
    common = dict(
        period=5.366265401100268,
        radius=43.06,
        mass=5.26,
        teff=6562.0,
        logg=1.883,
        fe_h=0.06,
    )
    return (
        CepheidConfig(name="cep_DeltaCep", rotation_velocity=DELTA_CEP_ROTATION_KMS, **common),
        CepheidConfig(name="cep_DeltaCep_norot", rotation_velocity=0.0, **common),
    )


def default_grid() -> list[CepheidConfig]:
    """Six-entry classical-Cepheid grid covering log P = 0.5–1.5.

    Radius scales via Bono et al. 2001 (log R/R_sun = 0.748 log P + 1.10).
    Mass-period passes through Delta-Cep (M=5.26, P=5.366) with slope 0.4 in
    log-log, giving ~4-10 M_sun across P=3-30 d (consistent with pulsation-
    model masses). Teff drops along the IS toward longer period (rough fit
    to Madore & Freedman 1991); logg follows from M and R.

    δ Cep is listed twice (``cep_DeltaCep`` at :data:`DELTA_CEP_ROTATION_KMS`
    km/s and ``cep_DeltaCep_norot`` at 0 km/s); other periods are non-rotating.
    """
    rows: list[CepheidConfig] = []
    for p_days, teff in [(3.0, 6300), (5.366, 6562), (10.0, 5800),
                         (20.0, 5500), (30.0, 5300)]:
        if p_days == 5.366:
            rows.extend(_delta_cep_configs())
            continue
        log_p = np.log10(p_days)
        radius = 10.0 ** (0.748 * log_p + 1.10)
        mass   = 10.0 ** (0.40 * log_p + 0.430)
        # log g = log(GM/R²) in cgs, with G·M_sun/R_sun² → 10^4.44 cgs at solar
        logg = 4.44 + np.log10(mass) - 2.0 * np.log10(radius)
        rows.append(CepheidConfig(
            name=f"cep_P{p_days:05.2f}".replace(".", "p"),
            period=p_days, radius=float(radius), mass=float(mass),
            teff=float(teff), logg=float(logg),
            fe_h=0.0,
        ))
    return rows


def load_grid_csv(path: Path) -> list[CepheidConfig]:
    """Read a CSV with one row per Cepheid.

    Required columns: ``name, period, radius, mass``.
    Optional: ``teff, logg, fe_h, vmicro, a_fe, c_fe, n_fe, o_fe, r_fe, s_fe,
    rotation_velocity, n_vertices, pulsation_fits``.
    """
    configs = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            kwargs = {"name": row["name"],
                      "period": float(row["period"]),
                      "radius": float(row["radius"]),
                      "mass":   float(row["mass"])}
            for key in ("teff", "logg", "fe_h", "vmicro",
                        "a_fe", "c_fe", "n_fe", "o_fe", "r_fe", "s_fe",
                        "rotation_velocity"):
                if key in row and row[key] != "":
                    kwargs[key] = float(row[key])
            if "n_vertices" in row and row["n_vertices"] != "":
                kwargs["n_vertices"] = int(row["n_vertices"])
            if "pulsation_fits" in row and row["pulsation_fits"]:
                kwargs["pulsation_fits"] = Path(row["pulsation_fits"])
            configs.append(CepheidConfig(**kwargs))
    return configs


# ---------------------------------------------------------------------------
# Pulsation template helpers (per-star: Fourier fit + Teff/logg interpolators)
# ---------------------------------------------------------------------------
def _fourier_series(x, a0, *coeffs):
    n_terms = len(coeffs) // 2
    out = a0
    for i in range(n_terms):
        a_i, b_i = coeffs[2 * i], coeffs[2 * i + 1]
        out = out + a_i * np.cos(2 * np.pi * (i + 1) * x) + b_i * np.sin(2 * np.pi * (i + 1) * x)
    return out


def fit_radius_template(pulsation_data: Table, n_terms: int = N_FOURIER_TERMS) -> np.ndarray:
    """Fourier-fit ``R(phase) / R(0)`` and pack into the (3, n_terms, 2) VSH matrix.

    Cepheid pulsation is the radial l=0, m=0 mode, so only the first row of
    the (radial, spheroidal, toroidal) VSH matrix is non-zero.
    """
    phases = np.array(pulsation_data["PHASE"])
    radius = np.array(pulsation_data["R"] / pulsation_data["R"][0])
    sort_idx = np.argsort(phases)
    phases, radius = phases[sort_idx], radius[sort_idx]

    p0 = [float(np.mean(radius))] + [0.0] * (2 * n_terms)
    params, _ = optimize.curve_fit(_fourier_series, phases, radius, p0=p0)
    fc = params[1:]

    radial = np.zeros((n_terms, 2))
    for i in range(n_terms):
        a_i, b_i = fc[2 * i], fc[2 * i + 1]
        radial[i] = [np.sqrt(a_i ** 2 + b_i ** 2), np.arctan2(b_i, a_i)]
    return np.stack([radial, np.zeros_like(radial), np.zeros_like(radial)])


def xi_micro_deltaCep(phase):
    """ξ (km/s) for δ Cep, fit to Luck 2018 (AJ 156, 171) table 3.

    Phase convention: GCVS ephemeris (Samus et al. 2017), phase 0 ≈ maximum
    light; minimum radius near phase ~0.7-0.8. Range 2.5–4.0 km/s, RMS fit
    residual 0.13 km/s.
    """
    phi = phase % 1.0
    c = [3.046, 0.182, -0.507, -0.182, -0.061, -0.091, 0.040]
    val = c[0]
    for k in range(1, 4):
        val += c[2 * k - 1] * np.cos(2 * np.pi * k * phi) + c[2 * k] * np.sin(2 * np.pi * k * phi)
    return val


def make_phase_modifiers(pulsation_data: Table):
    """Cubic-interpolate Teff(phase) and logg(phase) from the SPIPS3 table, and
    pull vmicro(phase) from :func:`xi_micro_deltaCep` (Luck 2018 δ Cep fit)."""
    phase = np.asarray(pulsation_data["PHASE"])
    sort_idx = np.argsort(phase)
    phase  = phase[sort_idx]
    teff   = np.asarray(pulsation_data["Teff"])[sort_idx]
    logg   = np.asarray(pulsation_data["logg"])[sort_idx]
    teff_i = interp1d(phase, teff, kind="cubic", bounds_error=False,
                      fill_value=(float(teff[-1]), float(teff[0])), assume_sorted=True)
    logg_i = interp1d(phase, logg, kind="cubic", bounds_error=False,
                      fill_value=(float(logg[-1]), float(logg[0])), assume_sorted=True)

    teff_at   = lambda p: float(teff_i(p % 1.0))
    logg_at   = lambda p: float(logg_i(p % 1.0))
    vmicro_at = lambda p: float(xi_micro_deltaCep(p))

    return teff_at, logg_at, vmicro_at


# ---------------------------------------------------------------------------
# Emulator setup (built once, shared across all Cepheids)
# ---------------------------------------------------------------------------
def _load_aemu_intensity(name: str):
    """Lazy-import + load an :class:`IntensityPretrainedAemuSpectrumEmulator`.

    Imported here (not at module top) so the script still imports when the
    optional ``aemu`` extra is missing -- relevant when callers pass
    ``--skip-aemu`` to build the two zarr bundles only.
    """
    from spice.spectrum.aemu_spectrum_emulator import (
        IntensityPretrainedAemuSpectrumEmulator,
    )
    return IntensityPretrainedAemuSpectrumEmulator(name)


def build_emulators(
    intensity_path: str,
    flux_path: str,
    *,
    harps_aemu: Optional[str] = DEFAULT_HARPS_AEMU,
    iron_aemu: Optional[str] = DEFAULT_IRON_AEMU,
    wanted: Sequence[str] = ALL_BUNDLES,
) -> dict[str, Any]:
    """Build the emulator backing each requested bundle variant.

    Returns ``{bundle_name: emulator}`` for every key in ``wanted`` (subset of
    :data:`ALL_BUNDLES`). LD coefficients for the ``flux_no_ld`` variant are
    passed per call to ``simulate_observed_flux`` at synthesis time, so we
    don't need a separate flux emulator per coefficient.
    """
    out: dict[str, Any] = {}
    if "intensity" in wanted:
        print(f"Loading intensity grid: {intensity_path}")
        out["intensity"] = IntensityLazyZarrInterpolator(
            intensity_path, params=INTENSITY_PARAMS, solar_parameters=SOLAR_INTENSITY,
            sparse=True, in_memory=False,
        )
    if "flux_no_ld" in wanted:
        print(f"Loading flux grid:      {flux_path}")
        out["flux_no_ld"] = FluxLazyZarrInterpolator(
            flux_path, params=FLUX_PARAMS, solar_parameters=SOLAR_FLUX,
            sparse=True, in_memory=False,
        )
    if "harps" in wanted and harps_aemu is not None:
        print(f"Loading HARPS aemu:     {harps_aemu}")
        out["harps"] = _load_aemu_intensity(harps_aemu)
    if "iron_line" in wanted and iron_aemu is not None:
        print(f"Loading iron-line aemu: {iron_aemu}")
        out["iron_line"] = _load_aemu_intensity(iron_aemu)
    return out


# Names of phase-dependent stellar parameters that the SPIPS3 / Luck-2018
# templates produce. Mapped to bundle parameter indices via
# ``_phase_modifiers_for`` -- only parameters that the emulator actually
# carries get a modifier.
PHASE_MODIFIER_NAMES = ("teff", "logg", "vmicro")


def _phase_modifiers_for(emulator, teff_at, logg_at, vmicro_at):
    """Build a ``[(param_index, fn), ...]`` modifier list for ``emulator``.

    Looks up each phase-dependent parameter by name in the emulator's
    ``stellar_parameter_names`` and skips any that are not present (e.g.
    aemu bundles that don't carry vmicro).
    """
    name_to_fn = {"teff": teff_at, "logg": logg_at, "vmicro": vmicro_at}
    names = list(getattr(emulator, "stellar_parameter_names", []))
    if not names:
        # Last-resort fallback: assume the canonical zarr ordering.
        names = INTENSITY_PARAMS[:-1] if "intensity" in repr(type(emulator)) else FLUX_PARAMS
    out = []
    for n, fn in name_to_fn.items():
        if n in names:
            out.append((names.index(n), fn))
    return out


# ---------------------------------------------------------------------------
# Per-Cepheid build
# ---------------------------------------------------------------------------
def build_one(
    config: CepheidConfig,
    emulators: dict[str, Any],
    out_dir: Path,
    *,
    skip_spectra: bool = False,
    n_mesh: int = 5000,
) -> None:
    bundles_path = out_dir / f"{config.name}_bundles.pkl"
    spectra_path = out_dir / f"{config.name}_spectra.pkl"
    n_mesh_used = config.n_vertices if config.n_vertices is not None else n_mesh
    print(f"\n=== {config.name}  P={config.period:.4f} d  "
          f"R={config.radius:.2f} R_sun  M={config.mass:.2f} M_sun  "
          f"v_rot={config.rotation_velocity:.2f} km/s  n_vertices={n_mesh_used} ===")
    print(f"  variants: {', '.join(emulators.keys())}")

    with fits.open(config.pulsation_fits, ignore_missing_simple=True) as hdul:
        pulsation_data = Table(hdul[2].data)

    fourier = fit_radius_template(pulsation_data)
    teff_at, logg_at, vmicro_at = make_phase_modifiers(pulsation_data)

    teff_default = float(pulsation_data["Teff"][0])
    logg_default = float(pulsation_data["logg"][0])
    sp = config.stellar_params_dict(teff_default, logg_default)
    vm_phase = [vmicro_at(p) for p in np.linspace(0, 1, 64)]
    print(f"  stellar params: Teff={sp['teff']:.0f} K  logg={sp['logg']:.3f}  "
          f"[Fe/H]={sp['[Fe/H]']:+.2f}")
    print(f"  vmicro (Luck 2018 δ Cep fit): "
          f"phase range [{min(vm_phase):.2f}, {max(vm_phase):.2f}] km/s")

    timeseries = jnp.linspace(0, config.period, N_PHASES)

    # 1) Build one bundle per emulator. Each emulator has its own parameter
    # ordering; ``build_bundle`` reads ``stellar_parameter_names`` and stores
    # parameters on the mesh in that emulator's order.
    bundles: dict[str, CepheidBundle] = {}
    for name, emul in emulators.items():
        print(f"  building {name}…")
        bundles[name] = build_bundle(
            emul, emul.to_parameters(sp),
            fourier_params=fourier,
            period=config.period, timeseries=timeseries,
            n_mesh=n_mesh_used, radius=config.radius, mass=config.mass,
            rotation_velocity=config.rotation_velocity,
            param_names_attr="stellar_parameter_names",
            desc=f"  evaluating {name}",
        )

    # 2) Phase-dependent Teff, logg, and vmicro. Modifier indices are looked
    # up by name in each emulator's ``stellar_parameter_names``; aemu bundles
    # that lack a particular parameter (e.g. vmicro) silently skip it.
    phases_per_snapshot = [(t % config.period) / config.period for t in timeseries]
    for name, bundle in bundles.items():
        modifiers = _phase_modifiers_for(
            emulators[name], teff_at, logg_at, vmicro_at,
        )
        bundles[name] = apply_phase_params(
            bundle, phases_per_snapshot, modifiers=modifiers,
        )

    save_pickle(bundles, str(bundles_path))
    print(f"  → saved {bundles_path.name}")

    if skip_spectra:
        print("  (skip-spectra: leaving spectra unbuilt)")
        return

    # 3) Spectrum synthesis. The intensity/aemu bundles each get a single
    # variant; the zarr flux bundle additionally fans out into a 20-point
    # linear-LD coefficient sweep (different ``ld_coeffs`` per call, no
    # rebuild required).
    line_center = 0.5 * (WL_MIN + WL_MAX)
    line_width = 0.5 * (WL_MAX - WL_MIN)

    spectrum_variants: list[tuple[str, Any, CepheidBundle, Optional[Any]]] = []
    if "intensity" in bundles:
        spectrum_variants.append((
            "with_ld",
            emulators["intensity"].intensity,
            bundles["intensity"],
            None,
        ))
    if "flux_no_ld" in bundles:
        for c in LD_COEFFS:
            spectrum_variants.append((
                f"flux_linear_{c:{LD_FMT}}",
                emulators["flux_no_ld"].intensity,
                bundles["flux_no_ld"],
                jnp.array([float(c), 0.0, 0.0, 0.0]),
            ))
    for name in ("harps", "iron_line"):
        if name in bundles:
            spectrum_variants.append((
                name,
                emulators[name].intensity,
                bundles[name],
                None,
            ))

    spectra: dict[str, dict[float, LineSpectra]] = {}
    for name, intensity_fn, bundle, ld_coeffs in spectrum_variants:
        t0 = time.perf_counter()
        spectra[name] = simulate_line_spectra(
            bundle.snapshots, bundle.snapshots[0], intensity_fn, (line_center,),
            line_width=line_width, steps=WL_STEPS,
            ld_coeffs=ld_coeffs,
        )
        print(f"    {name:>20s}: {time.perf_counter() - t0:6.1f} s")

    save_pickle(spectra, str(spectra_path))
    print(f"  → saved {spectra_path.name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out-dir", type=Path, default=HERE / "cepheid_grid",
                   help="Directory for per-star pickles.")
    p.add_argument("--config", type=Path, default=None,
                   help="CSV with custom Cepheid grid (default: built-in 6-entry grid).")
    p.add_argument("--names", nargs="+", default=None,
                   help="Subset of grid names to build (default: all).")
    p.add_argument("--intensity-zarr", type=str, default=LINE_INTERP_PATH,
                   help=f"Intensity grid zarr path (default: {LINE_INTERP_PATH}).")
    p.add_argument("--flux-zarr", type=str, default=FLUX_INTERP_PATH,
                   help=f"Flux grid zarr path (default: {FLUX_INTERP_PATH}).")
    p.add_argument("--harps-aemu", type=str, default=DEFAULT_HARPS_AEMU,
                   help="HARPS aemu bundle: HF repo id or local bundle dir "
                        f"(default: {DEFAULT_HARPS_AEMU}).")
    p.add_argument("--iron-aemu", type=str, default=DEFAULT_IRON_AEMU,
                   help="Iron-line aemu bundle: HF repo id or local bundle dir "
                        f"(default: {DEFAULT_IRON_AEMU}).")
    p.add_argument("--skip-aemu", action="store_true",
                   help="Skip the HARPS and iron-line aemu bundles "
                        "(useful when astro_emulators_toolkit isn't installed).")
    p.add_argument("--skip-zarr", action="store_true",
                   help="Skip the zarr intensity/flux bundles (only build the aemu ones).")
    p.add_argument("--bundles", nargs="+", default=None,
                   choices=list(ALL_BUNDLES),
                   help="Explicit subset of bundle variants to build "
                        f"(default: all of {list(ALL_BUNDLES)}; "
                        "overrides --skip-aemu / --skip-zarr).")
    p.add_argument("--n-mesh", type=int, default=5000,
                   help="Icosphere mesh resolution.")
    p.add_argument("--force", action="store_true",
                   help="Re-build even if output pickles already exist.")
    p.add_argument("--skip-spectra", action="store_true",
                   help="Skip spectrum synthesis (the slow step). Save bundles only.")
    p.add_argument("--continue-on-error", action="store_true",
                   help="If a star fails, log and move on instead of aborting.")
    return p.parse_args()


def _resolve_bundle_set(args) -> tuple[str, ...]:
    """Combine ``--bundles`` / ``--skip-aemu`` / ``--skip-zarr`` into a final
    ordered tuple of bundle keys to build."""
    if args.bundles:
        wanted = tuple(b for b in ALL_BUNDLES if b in set(args.bundles))
    else:
        wanted = ALL_BUNDLES
    if args.skip_aemu:
        wanted = tuple(b for b in wanted if b not in AEMU_BUNDLES)
    if args.skip_zarr:
        wanted = tuple(b for b in wanted if b not in ZARR_BUNDLES)
    return wanted


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    grid = load_grid_csv(args.config) if args.config else default_grid()
    if args.names:
        wanted = set(args.names)
        grid = [c for c in grid if c.name in wanted]
        missing = wanted - {c.name for c in grid}
        if missing:
            print(f"warning: --names entries not in grid: {sorted(missing)}", file=sys.stderr)

    # Skip already-built stars (unless --force).
    pending = []
    for cfg in grid:
        b = args.out_dir / f"{cfg.name}_bundles.pkl"
        s = args.out_dir / f"{cfg.name}_spectra.pkl"
        already = b.exists() and (args.skip_spectra or s.exists())
        if already and not args.force:
            print(f"skip {cfg.name}: outputs already exist (use --force to rebuild)")
            continue
        pending.append(cfg)

    if not pending:
        print("Nothing to do.")
        return 0

    wanted_bundles = _resolve_bundle_set(args)
    if not wanted_bundles:
        print("No bundle variants selected (check --bundles / --skip-aemu / --skip-zarr).",
              file=sys.stderr)
        return 1

    print(f"Will build {len(pending)} Cepheid(s) → {args.out_dir.resolve()}")
    print(f"Bundle variants: {list(wanted_bundles)}")
    emulators = build_emulators(
        args.intensity_zarr, args.flux_zarr,
        harps_aemu=args.harps_aemu, iron_aemu=args.iron_aemu,
        wanted=wanted_bundles,
    )

    failures = []
    grid_t0 = time.perf_counter()
    for cfg in pending:
        star_t0 = time.perf_counter()
        try:
            build_one(
                cfg, emulators, args.out_dir,
                skip_spectra=args.skip_spectra,
                n_mesh=args.n_mesh,
            )
        except Exception as exc:
            failures.append((cfg.name, exc))
            print(f"  !! {cfg.name} failed: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                traceback.print_exc()
                return 1
        else:
            print(f"  {cfg.name}: {time.perf_counter() - star_t0:.0f} s")

    elapsed = time.perf_counter() - grid_t0
    print(f"\nDone — {len(pending) - len(failures)}/{len(pending)} succeeded "
          f"in {elapsed/60:.1f} min.")
    if failures:
        print("Failures:")
        for name, exc in failures:
            print(f"  {name}: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
