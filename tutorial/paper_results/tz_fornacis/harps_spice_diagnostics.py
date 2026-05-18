"""HARPS ↔ SPICE diagnostic plots for the TZ Fornacis comparison.

Run cell-by-cell (VS Code / PyCharm interactive) or
`jupytext --to notebook harps_spice_diagnostics.py` to convert to .ipynb.

What this answers
-----------------
The eclipse-plots notebook already shows per-phase residual RVs spanning
~130 km/s (median −27, min −46, max +84). That scatter cannot be explained
by stellar physics alone — it implies a frame / Doppler-projection bug.
The plots below isolate which step is broken:

* Plot 1 — LOS / units sanity check (does ``body.los_vector`` match the
  eclipse-predictor LOS? is ``orbital_velocity`` in km/s?).
* Plot 2 — Geometric per-body RV curves vs the published Gallenne+2016
  sinusoid. Wrong amplitude = units bug. Wrong sign = LOS flip. Wrong
  phase = mean-anomaly / T_p bug.
* Plot 3 — Spectrum-derived per-body RV (cross-correlation against the
  t=0 template). Should agree with the geometric RV from Plot 2; if it
  doesn't, ``simulate_observed_flux`` isn't applying the Doppler the way
  the geometry says.
* Plot 4 — CCF HARPS vs SPICE composite in a clean metal window. The
  peak position, FWHM, and symmetry are the three sub-diagnostics.
* Plot 5 — Phase-folded Mg b 5183.6 Å core depth: HARPS vs SPICE.
* Plot 6 — Single-line bisector for an unblended Fe I line.
* Plot 7 — Residual auto-correlation length + telluric-band overlay.

Each plot prints a short interpretation guide so the answer is in-band.
"""

# %% [markdown]
# # Setup

# %%
import os
# Force CPU backend before JAX is imported (Metal plugin mismatch errors on this env).
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys
from pathlib import Path
import pickle

import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")  # headless: save PNGs without needing a display
import matplotlib.pyplot as plt
import zarr
from scipy.signal import correlate


def _repo_root(here: Path) -> Path:
    for p in [here, *here.parents]:
        if (p / "src" / "spice").is_dir():
            return p
    raise RuntimeError("Could not find spice repo root (src/spice)")


HERE = Path.cwd()
REPO = _repo_root(HERE)
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spice.spectrum import apply_spectral_resolution

DIAG_DIR = Path(
    "/Users/mjablons/code/spice/tutorial/paper_results/tz_fornacis/diagnostics"
)
DIAG_DIR.mkdir(parents=True, exist_ok=True)
print(f"Saving diagnostics to: {DIAG_DIR}")

C_KMS = 299_792.458
P_DAYS = 75.66647
# matches astropy.units.day -> astropy.units.year (Julian year = 365.25 d) to ~1e-15
P_YR = P_DAYS / 365.25
T_P_HJD = 2_452_599.29040

# Gallenne+2016, Table 4
K1_LIT = 38.71            # km/s, primary semi-amplitude
K2_LIT = 39.59            # km/s, secondary semi-amplitude
GAMMA1_LIT = 17.99
GAMMA2_LIT = 18.35


# %% [markdown]
# # Load SPICE pickle + HARPS zarr

# %%
# Override with `TZ_FOR_PICKLE=/path/to/file.pkl python ...` to point at the
# narrow-band test pickle from tz_fornacis_spectra_test.py.
PICKLE_PATH = Path(
    os.environ.get("TZ_FOR_PICKLE", str(REPO / "data" / "tz_fornacis_data_40000.pkl"))
)
print(f"Loading pickle: {PICKLE_PATH}")
with open(PICKLE_PATH, "rb") as f:
    data = pickle.load(f)

PERIOD_YR = float(data["period_yr"])
BINARY_PARAMS = data["binary_params"]
times_yr = np.asarray(data["times"])
wavelengths = np.asarray(data["wavelengths"])
spectra1 = np.asarray(data["spectra_body1"])
spectra2 = np.asarray(data["spectra_body2"])
pb1 = data["mesh_body1"]
pb2 = data["mesh_body2"]
time_origin = np.asarray(data["time_origin"])

M1, M2 = BINARY_PARAMS["primary_mass"], BINARY_PARAMS["secondary_mass"]
GAMMA_LIT = (M1 * GAMMA1_LIT + M2 * GAMMA2_LIT) / (M1 + M2)


def spice_orbital_phase(times_yr, binary_params=BINARY_PARAMS, period_yr=PERIOD_YR):
    m0 = float(binary_params["mean_anomaly_at_t0_rad"])
    return (np.asarray(times_yr) / period_yr + m0 / (2.0 * np.pi)) % 1.0


SPICE_PHASES = spice_orbital_phase(times_yr)

HARPS_ZARR_PATH = Path(
    "/Users/mjablons/code/spice/tutorial/paper_results/tz_fornacis/harps_data_zarr2.zarr"
)
if not HARPS_ZARR_PATH.exists():
    raise FileNotFoundError(
        f"{HARPS_ZARR_PATH} not found. "
        "Run harps_read.ipynb → harps_suppnet_normalize.ipynb → phase_harps.ipynb first."
    )
harps = zarr.open_group(str(HARPS_ZARR_PATH))
HARPS_PHASES_NORM = np.asarray(harps["phases"][:]) / P_DAYS

print(f"SPICE samples: {len(times_yr)}  ({(time_origin == 'general').sum()} general,",
      f"{(time_origin == 'primary_eclipse').sum()} prim,",
      f"{(time_origin == 'secondary_eclipse').sum()} sec)")
print(f"HARPS exposures: {len(HARPS_PHASES_NORM)}")
print(f"Mass ratio q = M2/M1 = {M2/M1:.4f}")
print(f"Literature γ_sys = {GAMMA_LIT:+.3f} km/s   K1 = {K1_LIT}   K2 = {K2_LIT}")


# %% [markdown]
# # Plot 1 — LOS / units sanity check
#
# Reads `los_vector` and `orbital_velocity` straight off the pickled meshes.
# * `los_vector`: tells you which axis SPICE was actually projecting
#   Doppler onto. The eclipse predictor in `tz_fornacis_spectra.py:170`
#   used `[0, 0, -1]`; the default in `mesh_model.py:31` is `[0, 1, 0]`.
#   If these disagree, the Doppler is projected onto a different axis
#   from the orbit's "line of sight".
# * `orbital_velocity` max magnitude: should be ≲ K1 + K2 ≈ 78 km/s
#   when stored in km/s. If you see ~78,000, the units are m/s and
#   `apply_vrad_log` is shifting by 1000× too much.

# %%
b1, b2 = pb1[0], pb2[0]
print("body1.los_vector =", np.asarray(b1.los_vector))
print("body2.los_vector =", np.asarray(b2.los_vector))
print(f"eclipse-predictor LOS = [0, 0, -1]    (from tz_fornacis_spectra.py:170)")
print(f"default LOS in mesh_model._default_los_vector() = [0, 1, 0]")
print()

v1_norm = np.array([np.linalg.norm(np.asarray(b.orbital_velocity)) for b in pb1])
v2_norm = np.array([np.linalg.norm(np.asarray(b.orbital_velocity)) for b in pb2])
print(f"|body1.orbital_velocity| over time:")
print(f"  min={v1_norm.min():.4g}, max={v1_norm.max():.4g}, median={np.median(v1_norm):.4g}")
print(f"|body2.orbital_velocity| over time:")
print(f"  min={v2_norm.min():.4g}, max={v2_norm.max():.4g}, median={np.median(v2_norm):.4g}")
print()
expected = K1_LIT + abs(GAMMA_LIT)
print(f"Expected order-of-magnitude in km/s: max1 ~ K1 ≈ {K1_LIT}, max2 ~ K2 ≈ {K2_LIT}")
print(f"If observed max is ~{expected*1000:.0f}, orbital_velocity is in m/s and apply_vrad_log "
      f"(which uses C = {C_KMS:.3f} km/s) overshoots by 1000×.")
print(f"If observed max is ~{expected*0.001:.5f}, orbital_velocity is in m/s but you "
      f"accidentally treated it as km/s elsewhere.")


# %% [markdown]
# # Plot 2 — Geometric per-body RV curves
#
# Projects each body's stored `orbital_velocity` onto its `los_vector` using
# SPICE's exact sign convention (`-cast_to_los` from `mesh_model.py:194`).
#
# Also projects onto `[0, 0, -1]` and `[0, 1, 0]` separately, so you can see
# explicitly which axis matches Gallenne's published RV curve. Whichever
# axis reproduces the K1 = 38.71, K2 = 39.59 sinusoids with γ at the right
# phase is the axis SPICE *should* be using.

# %%
def _project(velocity_3, los_dir_3):
    """SPICE's los velocity sign: -v·n̂  (blueshift = negative)."""
    v = np.asarray(velocity_3, dtype=float).reshape(3)
    n = np.asarray(los_dir_3, dtype=float).reshape(3)
    return -float(np.dot(v, n / np.linalg.norm(n)))


rv1_native = np.array([_project(b.orbital_velocity, b.los_vector) for b in pb1])
rv2_native = np.array([_project(b.orbital_velocity, b.los_vector) for b in pb2])
rv1_zminus = np.array([_project(b.orbital_velocity, [0, 0, -1]) for b in pb1])
rv2_zminus = np.array([_project(b.orbital_velocity, [0, 0, -1]) for b in pb2])
rv1_yplus  = np.array([_project(b.orbital_velocity, [0, 1, 0])  for b in pb1])
rv2_yplus  = np.array([_project(b.orbital_velocity, [0, 1, 0])  for b in pb2])

phase_grid = np.linspace(0, 1, 500)
# circular orbit; γ as published; body2 in antiphase to body1
rv1_model_a = GAMMA_LIT + K1_LIT * np.sin(2 * np.pi * phase_grid)
rv2_model_a = GAMMA_LIT - K2_LIT * np.sin(2 * np.pi * phase_grid)
rv1_model_b = GAMMA_LIT - K1_LIT * np.sin(2 * np.pi * phase_grid)
rv2_model_b = GAMMA_LIT + K2_LIT * np.sin(2 * np.pi * phase_grid)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
order = np.argsort(SPICE_PHASES)
for ax, label, rv1, rv2 in [
    (axes[0], "native los (what SPICE actually used)", rv1_native, rv2_native),
    (axes[1], "projected onto [0, 0, -1]", rv1_zminus, rv2_zminus),
    (axes[2], "projected onto [0, 1, 0]",  rv1_yplus,  rv2_yplus),
]:
    ax.plot(SPICE_PHASES[order], rv1[order], "o", color="tab:blue", ms=3, label="SPICE body1", alpha=0.7)
    ax.plot(SPICE_PHASES[order], rv2[order], "o", color="tab:red",  ms=3, label="SPICE body2", alpha=0.7)
    ax.plot(phase_grid, rv1_model_a, "--", color="tab:blue", alpha=0.4, lw=1, label="Gallenne K1 sin")
    ax.plot(phase_grid, rv2_model_a, "--", color="tab:red",  alpha=0.4, lw=1, label="Gallenne −K2 sin")
    ax.plot(phase_grid, rv1_model_b, ":",  color="tab:blue", alpha=0.4, lw=1, label="Gallenne −K1 sin")
    ax.plot(phase_grid, rv2_model_b, ":",  color="tab:red",  alpha=0.4, lw=1, label="Gallenne +K2 sin")
    ax.axhline(GAMMA_LIT, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel("Orbital phase (Gallenne T_p)")
    ax.set_title(label, fontsize=10)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("RV [km/s]")
axes[0].legend(fontsize=7, loc="upper right")
plt.suptitle("Where do the SPICE per-body RV curves match Gallenne+2016?", y=1.02)
plt.tight_layout()
fig.savefig(DIAG_DIR / "plot2_per_body_rv_curves.png", dpi=140, bbox_inches="tight")
plt.show()

for label, rv1, rv2 in [
    ("native",   rv1_native, rv2_native),
    ("[0,0,-1]", rv1_zminus, rv2_zminus),
    ("[0,1,0]",  rv1_yplus,  rv2_yplus),
]:
    print(f"{label:>10s}: body1 med={np.median(rv1):+.2f} range=[{rv1.min():+.1f},{rv1.max():+.1f}]  "
          f"body2 med={np.median(rv2):+.2f} range=[{rv2.min():+.1f},{rv2.max():+.1f}]")
print()
print(f"Target: each body median ≈ γ = {GAMMA_LIT:+.2f} km/s, full range ≈ [γ−K, γ+K].")
print("Whichever projection axis above hits Gallenne is the one simulate_observed_flux "
      "should be using. If `native` doesn't match but `[0,0,-1]` does, you need to set "
      "body.los_vector = [0,0,-1] before synthesizing.")


# %% [markdown]
# # Plot 3 — Spectrum-derived per-body RV
#
# Independently of Plot 2's geometry, measure each per-body spectrum's
# **actual** line shift via cross-correlation against the t=0 template.
# This tells us whether the Doppler that SPICE *thinks* it applied in
# Plot 2 is the same Doppler that made it into the synthesized spectrum.
#
# Interpretation matrix:
# * Plot 3 ≈ Plot 2 (native), neither matches Gallenne → frame setup bug
#   (LOS, T_p, mean_anomaly, or units) — fix upstream.
# * Plot 3 ≈ Gallenne, Plot 2 (native) ≠ Gallenne → `simulate_observed_flux`
#   is already correct; only the diagnostic projection (Plot 2) was wrong.
# * Plot 3 ≠ Plot 2 (native) → the velocity stored on the mesh isn't the
#   velocity actually used during line-by-line Doppler shifting.

# %%
LO_CCF, HI_CCF = 5100.0, 5400.0
LOG_WL_CCF = np.linspace(np.log10(LO_CCF), np.log10(HI_CCF), 4096)
WAVE_CCF = 10.0 ** LOG_WL_CCF
DELTA_LOG_CCF = LOG_WL_CCF[1] - LOG_WL_CCF[0]
V_PER_PIXEL_KMS = DELTA_LOG_CCF * np.log(10.0) * C_KMS


def _resample_spice(s):
    f = s[:, 0] / s[:, 1]
    return np.interp(WAVE_CCF, wavelengths, f)


def _ccf_rv(spectrum_log, template_log, max_lag_pix=600):
    a = spectrum_log - spectrum_log.mean()
    b = template_log - template_log.mean()
    full = correlate(a, b, mode="full")
    center = len(full) // 2
    lo, hi = center - max_lag_pix, center + max_lag_pix + 1
    seg = full[lo:hi]
    lags = np.arange(-max_lag_pix, max_lag_pix + 1, dtype=float)
    k = int(np.argmax(seg))
    if 0 < k < len(seg) - 1:
        y0, y1, y2 = seg[k - 1], seg[k], seg[k + 1]
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-12:
            return (lags[k] + 0.5 * (y0 - y2) / denom) * V_PER_PIXEL_KMS
    return lags[k] * V_PER_PIXEL_KMS


template1 = _resample_spice(spectra1[0])
template2 = _resample_spice(spectra2[0])
rv1_ccf = np.array([_ccf_rv(_resample_spice(s), template1) for s in spectra1])
rv2_ccf = np.array([_ccf_rv(_resample_spice(s), template2) for s in spectra2])

ref1, ref2 = rv1_native[0], rv2_native[0]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
order = np.argsort(SPICE_PHASES)
axes[0].plot(SPICE_PHASES[order], (rv1_native - ref1)[order], "o", color="black",
             ms=3, alpha=0.6, label="Geometric Δv (Plot 2 native)")
axes[0].plot(SPICE_PHASES[order], rv1_ccf[order], "x", color="tab:blue",
             ms=4, alpha=0.7, label="CCF Δv (from spectrum)")
axes[0].set_xlabel("Orbital phase"); axes[0].set_ylabel("RV - RV(t=0) [km/s]")
axes[0].set_title("Body 1: does the Doppler reach the spectrum?")
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

axes[1].plot(SPICE_PHASES[order], (rv2_native - ref2)[order], "o", color="black",
             ms=3, alpha=0.6, label="Geometric Δv (Plot 2 native)")
axes[1].plot(SPICE_PHASES[order], rv2_ccf[order], "x", color="tab:red",
             ms=4, alpha=0.7, label="CCF Δv (from spectrum)")
axes[1].set_xlabel("Orbital phase"); axes[1].set_ylabel("RV - RV(t=0) [km/s]")
axes[1].set_title("Body 2: does the Doppler reach the spectrum?")
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
plt.tight_layout()
fig.savefig(DIAG_DIR / "plot3_spectrum_derived_rv.png", dpi=140, bbox_inches="tight")
plt.show()

delta_v1 = rv1_ccf - (rv1_native - ref1)
delta_v2 = rv2_ccf - (rv2_native - ref2)
print(f"Body 1: |CCF Δv - geometric Δv| median = {np.median(np.abs(delta_v1)):.2f} km/s, "
      f"max = {np.abs(delta_v1).max():.2f} km/s")
print(f"Body 2: |CCF Δv - geometric Δv| median = {np.median(np.abs(delta_v2)):.2f} km/s, "
      f"max = {np.abs(delta_v2).max():.2f} km/s")
print("If median ≪ 1 km/s: the velocity stored on the mesh IS what reaches the spectrum.")
print("If they diverge: simulate_observed_flux's Doppler step is decoupled from "
      "mesh_model.velocities. Check the LOS axis used in `cast_to_los` vs the axis the "
      "orbit was set up in.")


# %% [markdown]
# # Plot 4 — CCF HARPS vs SPICE composite (one phase, telluric-clean window)
#
# The shape of this CCF tells you what kind of physics is missing.
# * Symmetric, single peak, offset from zero → bulk RV shift only.
#   Probably a frame bug (see Plots 1–3) once corrected, peak should
#   center on zero.
# * Symmetric, single peak, FWHM ≫ K1 + intrinsic → SPICE is too narrow:
#   add `add_rotation` (vsini) and macroturbulence.
# * Asymmetric or double-peaked → SB2 signature; the two components are
#   mis-phased in SPICE relative to HARPS.

# %%
HARPS_IDX = 0
phase_t = HARPS_PHASES_NORM[HARPS_IDX]
phase_d = np.minimum(np.abs(SPICE_PHASES - phase_t), 1 - np.abs(SPICE_PHASES - phase_t))
SPICE_MATCH = int(np.argmin(phase_d))

LO4, HI4 = 5100.0, 5800.0
log_wl4 = np.linspace(np.log10(LO4), np.log10(HI4), 8192)
wave4 = 10.0 ** log_wl4
v_per_pixel_4 = (log_wl4[1] - log_wl4[0]) * np.log(10.0) * C_KMS

hw_raw = np.asarray(harps["normalized_wave"][HARPS_IDX])
hf_raw = np.asarray(harps["normalized_flux"][HARPS_IDX])
m = np.isfinite(hw_raw) & np.isfinite(hf_raw)
hf_u = np.interp(wave4, hw_raw[m], hf_raw[m])

s_comp = spectra1[SPICE_MATCH] + spectra2[SPICE_MATCH]
sf_u = np.interp(wave4, wavelengths, s_comp[:, 0] / s_comp[:, 1])

R_DEGRADE = 60_000.0
hf_d = np.asarray(apply_spectral_resolution(jnp.asarray(log_wl4), jnp.asarray(hf_u), R_DEGRADE))
sf_d = np.asarray(apply_spectral_resolution(jnp.asarray(log_wl4), jnp.asarray(sf_u), R_DEGRADE))

a = hf_d - hf_d.mean()
b = sf_d - sf_d.mean()
ccf = correlate(a, b, mode="full")
lags = np.arange(-len(a) + 1, len(a), dtype=float)
center = len(ccf) // 2
max_lag = 800
seg = ccf[center - max_lag : center + max_lag + 1]
lag_v = lags[center - max_lag : center + max_lag + 1] * v_per_pixel_4
seg_norm = seg / seg.max()

k = int(np.argmax(seg_norm))
peak_kms = lag_v[k]
if 0 < k < len(seg_norm) - 1:
    y0, y1, y2 = seg_norm[k - 1], seg_norm[k], seg_norm[k + 1]
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) > 1e-12:
        peak_kms = lag_v[k] + 0.5 * (y0 - y2) / denom * (lag_v[1] - lag_v[0])

half = seg_norm >= 0.5
fwhm_kms = float(lag_v[half].max() - lag_v[half].min()) if half.any() else float("nan")

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(lag_v, seg_norm, color="black")
ax.axvline(peak_kms, color="tab:red", ls="--", label=f"peak = {peak_kms:+.2f} km/s")
ax.axvline(0, color="gray", ls=":", alpha=0.5, label="zero")
ax.axhline(0.5, color="gray", ls=":", alpha=0.3)
ax.set_xlabel("RV shift [km/s]")
ax.set_ylabel("Normalized CCF")
ax.set_title(f"CCF HARPS[idx={HARPS_IDX}, φ={phase_t:.3f}] vs SPICE composite [idx={SPICE_MATCH}, φ={SPICE_PHASES[SPICE_MATCH]:.3f}]\n"
             f"window {LO4:.0f}–{HI4:.0f} Å, R={R_DEGRADE:.0f}, FWHM ≈ {fwhm_kms:.1f} km/s")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(DIAG_DIR / "plot4_ccf_harps_vs_spice.png", dpi=140, bbox_inches="tight")
plt.show()

print(f"CCF peak: {peak_kms:+.2f} km/s   FWHM: {fwhm_kms:.1f} km/s")
print(f"Expected intrinsic FWHM at R={R_DEGRADE:.0f}: ≈ {C_KMS / R_DEGRADE * 2.355:.1f} km/s")
print(f"With K1 ≈ {K1_LIT}, K2 ≈ {K2_LIT}, vsini_2 ≈ 30 km/s, expect FWHM ≈ 30–50 km/s away from eclipses.")
print("If FWHM ≪ that: SPICE composite is too sharp (missing vsini/macroturbulence).")
print("If peak ≠ 0: bulk RV offset — see Plots 1–3.")
print("If peak is asymmetric / has secondary bump: the two components are mis-phased.")


# %% [markdown]
# # Plot 5 — Phase-folded Mg b 5183.6 Å core depth
#
# Eclipses block one component → if the *blocked* component contributes
# the deeper Mg b core, the line depth changes systematically with phase.
# The shape of depth-vs-phase encodes:
# * The flux ratio between the two components.
# * Whether SPICE's occlusion gets the right component blocked at the
#   right phase (it should, since the photometric LC is already
#   validated, but a residual offset can still hide here).
# * Whether SPICE under/overestimates intrinsic line depth (via wrong
#   [Fe/H], log g, micro, or missing rotation).

# %%
LINE_MG = 5183.604
WIN_MG = 5
LO_MG = LINE_MG - WIN_MG
HI_MG = LINE_MG + WIN_MG
LOG_WL_MG = np.linspace(np.log10(LO_MG), np.log10(HI_MG), 2048)
WAVE_MG = 10.0 ** LOG_WL_MG

R_MG = 60_000.0
depths_h = np.full(len(HARPS_PHASES_NORM), np.nan)
depths_s = np.full(len(HARPS_PHASES_NORM), np.nan)
sidx_for = np.zeros(len(HARPS_PHASES_NORM), dtype=int)

for i in range(len(HARPS_PHASES_NORM)):
    hw_i = np.asarray(harps["normalized_wave"][i])
    hf_i = np.asarray(harps["normalized_flux"][i])
    m_i = np.isfinite(hw_i) & np.isfinite(hf_i)
    if not m_i.any():
        continue
    hf_u = np.interp(WAVE_MG, hw_i[m_i], hf_i[m_i])
    hf_d = np.asarray(apply_spectral_resolution(jnp.asarray(LOG_WL_MG), jnp.asarray(hf_u), R_MG))

    d = np.minimum(np.abs(SPICE_PHASES - HARPS_PHASES_NORM[i]),
                   1 - np.abs(SPICE_PHASES - HARPS_PHASES_NORM[i]))
    s_match = int(np.argmin(d))
    sidx_for[i] = s_match
    s = spectra1[s_match] + spectra2[s_match]
    sf_u = np.interp(WAVE_MG, wavelengths, s[:, 0] / s[:, 1])
    sf_d = np.asarray(apply_spectral_resolution(jnp.asarray(LOG_WL_MG), jnp.asarray(sf_u), R_MG))

    win = (WAVE_MG >= LINE_MG - 1.0) & (WAVE_MG <= LINE_MG + 1.0)
    depths_h[i] = 1.0 - float(hf_d[win].min())
    depths_s[i] = 1.0 - float(sf_d[win].min())

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(HARPS_PHASES_NORM, depths_h, c="black", s=22, label="HARPS")
ax.scatter(HARPS_PHASES_NORM, depths_s, c="tab:red", s=22, marker="x", label="SPICE")
for i, (ph, dh, ds) in enumerate(zip(HARPS_PHASES_NORM, depths_h, depths_s)):
    ax.plot([ph, ph], [dh, ds], color="gray", lw=0.4, alpha=0.5)
ax.set_xlabel("Orbital phase (Gallenne T_p)")
ax.set_ylabel(f"Core depth at {LINE_MG} Å (1 − F_min)")
ax.set_title(f"Phase-folded Mg b 5183.6 Å core depth (R = {R_MG:.0f})")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(DIAG_DIR / "plot5_mgb_core_depth_vs_phase.png", dpi=140, bbox_inches="tight")
plt.show()

print(f"HARPS Mg b depth: mean={np.nanmean(depths_h):.3f}, std={np.nanstd(depths_h):.3f}")
print(f"SPICE Mg b depth: mean={np.nanmean(depths_s):.3f}, std={np.nanstd(depths_s):.3f}")
print()
print("Diagnostic reading:")
print("  • SPICE consistently deeper than HARPS → missing vsini/macroturbulence on the secondary,")
print("    and/or SPICE is using a deeper-line continuum than SUPPNet returns for HARPS.")
print("  • SPICE consistently shallower → log(g) or [Fe/H] off, or too much vsini in SPICE.")
print("  • Phase-folded shapes don't agree (depth wiggles at eclipses) → occlusion phase / "
      "flux-ratio between components is wrong.")


# %% [markdown]
# # Plot 6 — Single-line bisector (Fe I 5269.537 Å)
#
# Bisector shape:
# * Lateral offset between HARPS and SPICE bisectors = bulk RV (should
#   go to zero once Plots 1–3 are fixed).
# * Bisector "C-shape" curvature in HARPS that SPICE lacks → real-star
#   3D / granulation signatures (a 1D atmosphere can't reproduce these,
#   but the discrepancy should be small for a giant).
# * Bisector slope (wing-to-core walk) in SPICE that HARPS doesn't show →
#   SPICE's line wings are too narrow ↔ missing rotation/macroturbulence.

# %%
LINE_FE = 5269.537
WIN_FE = 1.5
LO_FE = LINE_FE - WIN_FE
HI_FE = LINE_FE + WIN_FE
LOG_WL_FE = np.linspace(np.log10(LO_FE), np.log10(HI_FE), 2048)
WAVE_FE = 10.0 ** LOG_WL_FE
R_FE = 60_000.0
HARPS_IDX_FE = HARPS_IDX  # reuse from Plot 4

hw_i = np.asarray(harps["normalized_wave"][HARPS_IDX_FE])
hf_i = np.asarray(harps["normalized_flux"][HARPS_IDX_FE])
m_i = np.isfinite(hw_i) & np.isfinite(hf_i)
hf_u = np.interp(WAVE_FE, hw_i[m_i], hf_i[m_i])
hf_d = np.asarray(apply_spectral_resolution(jnp.asarray(LOG_WL_FE), jnp.asarray(hf_u), R_FE))

d = np.minimum(np.abs(SPICE_PHASES - HARPS_PHASES_NORM[HARPS_IDX_FE]),
               1 - np.abs(SPICE_PHASES - HARPS_PHASES_NORM[HARPS_IDX_FE]))
sidx_fe = int(np.argmin(d))
s = spectra1[sidx_fe] + spectra2[sidx_fe]
sf_u = np.interp(WAVE_FE, wavelengths, s[:, 0] / s[:, 1])
sf_d = np.asarray(apply_spectral_resolution(jnp.asarray(LOG_WL_FE), jnp.asarray(sf_u), R_FE))


def _bisector(wl, flux, n_levels=24):
    core_idx = int(np.argmin(flux))
    core_val = float(flux[core_idx])
    cont = float(np.percentile(flux, 95))
    if cont <= core_val:
        return np.full(n_levels, np.nan), np.full(n_levels, np.nan)
    levels = np.linspace(core_val + 0.05 * (cont - core_val),
                         cont       - 0.05 * (cont - core_val), n_levels)
    blue_wl = np.full(n_levels, np.nan)
    red_wl = np.full(n_levels, np.nan)
    for j, lvl in enumerate(levels):
        # blue side: idx 0..core_idx, find last crossing
        bf = flux[: core_idx + 1] - lvl
        cb = np.where(np.diff(np.sign(bf)) != 0)[0]
        if cb.size:
            ib = cb[-1]
            f1, f2 = flux[ib], flux[ib + 1]
            blue_wl[j] = wl[ib] + (lvl - f1) * (wl[ib + 1] - wl[ib]) / (f2 - f1 + 1e-30)
        rf = flux[core_idx:] - lvl
        cr = np.where(np.diff(np.sign(rf)) != 0)[0]
        if cr.size:
            ir = cr[0] + core_idx
            f1, f2 = flux[ir], flux[ir + 1]
            red_wl[j] = wl[ir] + (lvl - f1) * (wl[ir + 1] - wl[ir]) / (f2 - f1 + 1e-30)
    return (blue_wl + red_wl) / 2.0, levels


mid_h, lvl_h = _bisector(WAVE_FE, hf_d)
mid_s, lvl_s = _bisector(WAVE_FE, sf_d)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(WAVE_FE, hf_d, color="black", lw=1, label="HARPS")
axes[0].plot(WAVE_FE, sf_d, color="tab:red", lw=1, alpha=0.8, label="SPICE")
axes[0].axvline(LINE_FE, color="gray", ls=":", alpha=0.5)
axes[0].set_xlabel("Wavelength [Å]"); axes[0].set_ylabel("Normalized flux")
axes[0].set_title(f"Fe I {LINE_FE:.3f} Å, HARPS phase {HARPS_PHASES_NORM[HARPS_IDX_FE]:.3f}")
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(mid_h, lvl_h, "o-", color="black", label="HARPS bisector", ms=4)
axes[1].plot(mid_s, lvl_s, "x-", color="tab:red", label="SPICE bisector", ms=5, alpha=0.85)
axes[1].axvline(LINE_FE, color="gray", ls=":", alpha=0.5, label=f"rest λ = {LINE_FE}")
axes[1].set_xlabel("Bisector wavelength midpoint [Å]")
axes[1].set_ylabel("Flux level")
axes[1].set_title("Bisector comparison")
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)
plt.tight_layout()
fig.savefig(DIAG_DIR / "plot6_bisector_fe5269.png", dpi=140, bbox_inches="tight")
plt.show()

if np.isfinite(mid_h).all() and np.isfinite(mid_s).all():
    bulk_h = np.nanmean(mid_h) - LINE_FE
    bulk_s = np.nanmean(mid_s) - LINE_FE
    span_h = np.nanmax(mid_h) - np.nanmin(mid_h)
    span_s = np.nanmax(mid_s) - np.nanmin(mid_s)
    print(f"HARPS bisector: mean shift = {bulk_h*C_KMS/LINE_FE:+.2f} km/s,  span = {span_h*C_KMS/LINE_FE:.2f} km/s")
    print(f"SPICE bisector: mean shift = {bulk_s*C_KMS/LINE_FE:+.2f} km/s,  span = {span_s*C_KMS/LINE_FE:.2f} km/s")
print("Bisector span (curvature) in SPICE should be ≪ HARPS span if HARPS shows real "
      "convective C-shape and SPICE doesn't model it. A SPICE span that exceeds HARPS "
      "is suspicious — likely a sampling artefact at this resolution.")


# %% [markdown]
# # Plot 7 — Residual auto-correlation length + telluric overlay
#
# Coherence length of (HARPS − SPICE) tells us what *kind* of error
# dominates:
# * Coherence ≲ a few km/s (a few pixels at R = 60k): wavelength solution
#   or sampling — air/vacuum confusion, or coarse SPICE λ grid.
# * Coherence ≈ a line FWHM (∼10–30 km/s): line-shape error → vsini,
#   macroturbulence, microturbulence, [Fe/H].
# * Coherence ≫ a line FWHM (≳ 100 km/s, smooth waves across the band):
#   continuum-normalization mismatch (SUPPNet vs emulator continuum).
#
# Telluric bands are overplotted so you can visually flag residual
# features that are HARPS atmosphere, not stellar physics.

# %%
LO_R, HI_R = 4500.0, 6800.0
LOG_WL_R = np.linspace(np.log10(LO_R), np.log10(HI_R), 16_384)
WAVE_R = 10.0 ** LOG_WL_R
V_PER_PIXEL_R = (LOG_WL_R[1] - LOG_WL_R[0]) * np.log(10.0) * C_KMS
R_RES = 60_000.0

hw_i = np.asarray(harps["normalized_wave"][HARPS_IDX])
hf_i = np.asarray(harps["normalized_flux"][HARPS_IDX])
m_i = np.isfinite(hw_i) & np.isfinite(hf_i)
hf_u = np.interp(WAVE_R, hw_i[m_i], hf_i[m_i])
hf_d = np.asarray(apply_spectral_resolution(jnp.asarray(LOG_WL_R), jnp.asarray(hf_u), R_RES))

d = np.minimum(np.abs(SPICE_PHASES - HARPS_PHASES_NORM[HARPS_IDX]),
               1 - np.abs(SPICE_PHASES - HARPS_PHASES_NORM[HARPS_IDX]))
s_match = int(np.argmin(d))
s = spectra1[s_match] + spectra2[s_match]
sf_u = np.interp(WAVE_R, wavelengths, s[:, 0] / s[:, 1])
sf_d = np.asarray(apply_spectral_resolution(jnp.asarray(LOG_WL_R), jnp.asarray(sf_u), R_RES))

residual = hf_d - sf_d

TELLURIC_BANDS = [
    ("H2O 5880-5980", 5880, 5980),
    ("O2 γ", 6275, 6330),
    ("H2O 6440-6600", 6440, 6600),
    ("O2 B", 6860, 6960),
]

fig, axes = plt.subplots(2, 1, figsize=(13, 7.5))
ax0, ax1 = axes
ax0.axhline(0, color="gray", lw=0.5)
ax0.plot(WAVE_R, residual, color="tab:blue", lw=0.5)
ymin, ymax = float(np.nanpercentile(residual, 1)), float(np.nanpercentile(residual, 99))
ax0.set_ylim(ymin - 0.05, ymax + 0.05)
for name, lo_b, hi_b in TELLURIC_BANDS:
    if lo_b < HI_R and hi_b > LO_R:
        ax0.axvspan(max(lo_b, LO_R), min(hi_b, HI_R), color="tab:orange", alpha=0.18)
        ax0.text((max(lo_b, LO_R) + min(hi_b, HI_R)) / 2, ymax,
                 name, fontsize=7, ha="center", va="top", alpha=0.7)
ax0.set_xlabel("Wavelength [Å]")
ax0.set_ylabel("HARPS − SPICE (normalized flux)")
ax0.set_title(f"Residual at HARPS phase {HARPS_PHASES_NORM[HARPS_IDX]:.3f} "
              f"(R = {R_RES:.0f}); orange bands = uncorrected tellurics")
ax0.grid(alpha=0.3)

# ACF on a clean window (avoid telluric bands)
clean_mask = (WAVE_R >= 5100) & (WAVE_R <= 5800)
r = residual[clean_mask]
r = r - np.mean(r)
acf = correlate(r, r, mode="full")
acf = acf[len(acf) // 2 :]
acf /= acf[0]
lag_kms = np.arange(len(acf)) * V_PER_PIXEL_R

show_pix = int(min(len(acf), 800))
ax1.plot(lag_kms[:show_pix], acf[:show_pix], color="black", lw=1)
ax1.axhline(0, color="gray", lw=0.5)
ax1.axhline(1.0 / np.e, color="gray", ls=":", alpha=0.6, label="1/e")
below = np.where(acf[:show_pix] < 1.0 / np.e)[0]
if below.size:
    coh = lag_kms[below[0]]
    ax1.axvline(coh, color="tab:red", ls="--", label=f"1/e coherence ≈ {coh:.1f} km/s")
ax1.set_xlabel("Lag [km/s]")
ax1.set_ylabel("Auto-correlation of residual (5100–5800 Å)")
ax1.set_title("Residual ACF — coherence length diagnostic")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(DIAG_DIR / "plot7_residual_acf_tellurics.png", dpi=140, bbox_inches="tight")
plt.show()

if below.size:
    coh_kms = float(lag_kms[below[0]])
    fwhm_at_R = C_KMS / R_RES * 2.355
    print(f"Residual 1/e coherence length: {coh_kms:.1f} km/s")
    print(f"Line FWHM at R = {R_RES:.0f}: {fwhm_at_R:.1f} km/s")
    if coh_kms < 0.5 * fwhm_at_R:
        print("→ Sub-FWHM coherence: wavelength solution / sampling / air-vacuum suspect.")
    elif coh_kms < 3.0 * fwhm_at_R:
        print("→ Coherence ~ line FWHM: line-shape error (vsini, macro/microturbulence, [Fe/H]).")
    else:
        print("→ Coherence ≫ line FWHM: continuum-normalization mismatch dominates.")
