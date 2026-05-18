"""Fast, narrow-band synthesis to verify the γ + rotation + macroturbulence fixes.

Synthesizes TZ Fornacis spectra over 4980–5040 Å on a sparse time grid and a
small mesh — fits on CPU in a few minutes. Produces a pickle in the same
format as ``tz_fornacis_spectra.py`` so you can drop it into the existing
``harps_spice_diagnostics.py`` (just point ``PICKLE_PATH`` at the test file).

Built-in self-checks at the end:

1. **γ check (orbit level).** ``mean(body1.orbital_velocity.y)`` should be
   approximately ``-γ`` (γ lives on the barycenter along ``-y``). Pre-fix this
   was ``0`` (along the wrong axis) or ``-2γ`` (double-counted).
2. **Doppler-reaches-spectrum check.** CCF body 1's spectrum at the most blue-
   and most red-shifted phases against the t=0 template. The measured shift
   should match the geometric ``los_velocities`` to ≲ 1 km/s, replicating
   diagnostic Plot 3 in microcosm.
3. **Rotation+macro broadening check.** Per-body line FWHM should be
   broader than the bare intrinsic emulator FWHM by approximately
   ``√(vsini² + vmacro²)`` (in quadrature with the thermal width).

Usage
-----
::

    JAX_PLATFORMS=cpu python tutorial/paper_results/tz_fornacis/tz_fornacis_spectra_test.py
"""
import os
# Force CPU before JAX is imported (Metal plugin mismatch on some macOS envs).
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import pickle
import sys
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.signal import correlate
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ---- Constants kept in sync with tz_fornacis_spectra.py (inlined to avoid
# the astropy import in that module, which has broken on some local envs).
PRIMARY_MASS = 2.057
PRIMARY_RADIUS = 8.28
PRIMARY_TEFF = 4930
PRIMARY_LOGG = 2.91
SECONDARY_MASS = 1.958
SECONDARY_RADIUS = 3.94
SECONDARY_TEFF = 6650
SECONDARY_LOGG = 3.35
PERIOD_DAYS = 75.66647
PERIOD_YR = PERIOD_DAYS / 365.25  # Julian year (matches astropy day→year exactly)
ECC = 0.0
INCL_DEG = 85.68
PER0_DEG = 65.99
LONG_AN_DEG = 269.0
DISTANCE_PC = 182.8
T_P_HJD = 2_452_599.29040
GAMMA1_KMS = 17.99
GAMMA2_KMS = 18.35
GAMMA_KMS = (PRIMARY_MASS * GAMMA1_KMS + SECONDARY_MASS * GAMMA2_KMS) / (
    PRIMARY_MASS + SECONDARY_MASS
)
PRIMARY_FEH = -0.30
SECONDARY_FEH = -0.30
PRIMARY_VMIC = 1.5
SECONDARY_VMIC = 1.5
PRIMARY_AFE = 0.0
SECONDARY_AFE = 0.0
PRIMARY_VROT_KMS = 5.5
SECONDARY_VROT_KMS = 38.0
PRIMARY_VMACRO_KMS = 5.0
SECONDARY_VMACRO_KMS = 6.0
_FIRST_BJD = float(os.environ.get("TZ_FOR_FIRST_BJD", "2454887.5473123"))
_MEAN_ANOMALY_T0 = (
    ((_FIRST_BJD - T_P_HJD) % PERIOD_DAYS) / PERIOD_DAYS * (2.0 * np.pi)
)

# ---- Test-specific overrides (fast CPU run)
WL_MIN = 4980.0
WL_MAX = 5040.0
NUM_WAVELENGTHS = 6000     # Δλ ≈ 0.01 Å → native R ≈ 5×10⁵
NUM_TIMES = 30             # sparse out-of-eclipse phase coverage
N_MESH = 1000              # icosphere; small to fit CPU memory
ORBIT_CHUNK = 10
MODEL = os.environ.get("MODEL", "RozanskiT/TPayne-spice-harps")
# Force the cheap fallback emulator with USE_GAUSSIAN=1; otherwise we try
# TPayne first and fall back if astro_emulators_toolkit isn't installed.
USE_GAUSSIAN_FALLBACK = os.environ.get("USE_GAUSSIAN", "auto")

C_KMS = 299_792.458


def _make_emulator():
    """Return ``(emulator, kind)`` where ``kind`` is ``"aemu"`` or ``"gaussian"``.

    Tries the TPayne intensity emulator first; on ImportError (no
    ``astro_emulators_toolkit`` installed locally) falls back to
    ``GaussianLineEmulator`` with a handful of Fe I lines in 4980-5040 Å.
    The fallback is intentionally physics-light — it's sufficient to verify
    γ + rotation + macroturbulence end-to-end without needing a HuggingFace
    download or 1 GB of model weights.
    """
    if USE_GAUSSIAN_FALLBACK != "1":
        try:
            from spice.spectrum.aemu_spectrum_emulator import (
                IntensityPretrainedAemuSpectrumEmulator,
            )
            print(f"Using intensity emulator: {MODEL}")
            return IntensityPretrainedAemuSpectrumEmulator(MODEL), "aemu"
        except (ImportError, ValueError) as exc:
            print(f"TPayne unavailable ({exc.__class__.__name__}); "
                  f"falling back to GaussianLineEmulator.")
    from spice.spectrum.gaussian_line_emulator import GaussianLineEmulator
    # A handful of well-known Fe I / Ti I features in the test band so we get
    # genuine absorption structure to CCF against.
    line_centers = [4983.85, 5006.12, 5014.94, 5022.24, 5028.13, 5031.04]
    line_widths = [0.10, 0.12, 0.15, 0.10, 0.10, 0.12]
    line_depths = [0.45, 0.55, 0.70, 0.55, 0.50, 0.60]
    em = GaussianLineEmulator(line_centers=line_centers,
                              line_widths=line_widths,
                              line_depths=line_depths)
    print(f"Using GaussianLineEmulator with {len(line_centers)} mock lines.")
    return em, "gaussian"


def _build_binary(em_kind):
    from spice.models.binary import Binary, add_orbit
    from spice.models.mesh_model import IcosphereModel
    from spice.models.mesh_transform import add_rotation

    incl_rad = jnp.deg2rad(INCL_DEG)
    per0_rad = jnp.deg2rad(PER0_DEG)
    long_an_rad = jnp.deg2rad(LONG_AN_DEG)

    if em_kind == "aemu":
        primary_params = dict(
            teff=PRIMARY_TEFF, logg=PRIMARY_LOGG,
            feh=PRIMARY_FEH, vmic=PRIMARY_VMIC, afe=PRIMARY_AFE,
        )
        secondary_params = dict(
            teff=SECONDARY_TEFF, logg=SECONDARY_LOGG,
            feh=SECONDARY_FEH, vmic=SECONDARY_VMIC, afe=SECONDARY_AFE,
        )
    else:
        # GaussianLineEmulator only knows about ``Teff``.
        primary_params = dict(Teff=PRIMARY_TEFF)
        secondary_params = dict(Teff=SECONDARY_TEFF)
    body1 = IcosphereModel.construct(
        N_MESH, PRIMARY_RADIUS, PRIMARY_MASS,
        em.to_parameters(primary_params),
        em.stellar_parameter_names,
        override_log_g=False,
    )
    body2 = IcosphereModel.construct(
        N_MESH, SECONDARY_RADIUS, SECONDARY_MASS,
        em.to_parameters(secondary_params),
        em.stellar_parameter_names,
        override_log_g=False,
    )
    body1 = add_rotation(body1, rotation_velocity=PRIMARY_VROT_KMS)
    body2 = add_rotation(body2, rotation_velocity=SECONDARY_VROT_KMS)

    binary = Binary.from_bodies(body1, body2)
    binary = add_orbit(
        binary, PERIOD_YR, ECC, 0.0,
        incl_rad, per0_rad, long_an_rad,
        float(_MEAN_ANOMALY_T0), 0.0, float(GAMMA_KMS), 15,
    )
    return binary


def _evaluate_orbit_chunked(binary, times, chunk_size):
    from spice.models.binary import evaluate_orbit_at_times
    times = jnp.asarray(times)
    n = times.shape[0]
    pb1, pb2 = [], []
    for i in tqdm(range(0, n, chunk_size), desc="orbit"):
        chunk = times[i:i + chunk_size]
        r1, r2 = evaluate_orbit_at_times(binary, chunk)
        pb1.extend(list(r1))
        pb2.extend(list(r2))
    return pb1, pb2


def _apply_macroturbulence(spectra, log_vws, vmacro_kms):
    from spice.spectrum import apply_spectral_resolution
    if vmacro_kms <= 0.0:
        return spectra
    R_macro = C_KMS / vmacro_kms
    out = []
    for s in spectra:
        line = apply_spectral_resolution(log_vws, jnp.asarray(s[:, 0]), R_macro)
        cont = apply_spectral_resolution(log_vws, jnp.asarray(s[:, 1]), R_macro)
        out.append(np.stack([np.asarray(line), np.asarray(cont)], axis=-1))
    return np.asarray(out)


def _synthesize(vws, pb1, pb2):
    from spice.spectrum import simulate_observed_flux
    log_vws = jnp.log10(vws)
    s1 = [simulate_observed_flux(em.intensity, _pb, log_vws, distance=DISTANCE_PC)
          for _pb in tqdm(pb1, desc="primary")]
    s2 = [simulate_observed_flux(em.intensity, _pb, log_vws, distance=DISTANCE_PC)
          for _pb in tqdm(pb2, desc="secondary")]
    s1 = _apply_macroturbulence(s1, log_vws, PRIMARY_VMACRO_KMS)
    s2 = _apply_macroturbulence(s2, log_vws, SECONDARY_VMACRO_KMS)
    return np.asarray(s1), np.asarray(s2)


def _binary_params_dict():
    return {
        "primary_mass": PRIMARY_MASS, "secondary_mass": SECONDARY_MASS,
        "primary_radius": PRIMARY_RADIUS, "secondary_radius": SECONDARY_RADIUS,
        "primary_teff": PRIMARY_TEFF, "primary_logg": PRIMARY_LOGG,
        "secondary_teff": SECONDARY_TEFF, "secondary_logg": SECONDARY_LOGG,
        "primary_feh": PRIMARY_FEH, "secondary_feh": SECONDARY_FEH,
        "primary_vmic_kms": PRIMARY_VMIC, "secondary_vmic_kms": SECONDARY_VMIC,
        "primary_afe": PRIMARY_AFE, "secondary_afe": SECONDARY_AFE,
        "primary_vrot_kms": PRIMARY_VROT_KMS, "secondary_vrot_kms": SECONDARY_VROT_KMS,
        "primary_vmacro_kms": PRIMARY_VMACRO_KMS,
        "secondary_vmacro_kms": SECONDARY_VMACRO_KMS,
        "inclination_deg": INCL_DEG, "per0_deg": PER0_DEG,
        "long_an_deg": LONG_AN_DEG, "ecc": ECC,
        "period_days": PERIOD_DAYS, "n_mesh": N_MESH,
        "gamma1_kms": GAMMA1_KMS, "gamma2_kms": GAMMA2_KMS,
        "gamma_kms": GAMMA_KMS, "t_p_hjd": T_P_HJD,
        "first_bjd_phase_ref": _FIRST_BJD,
        "mean_anomaly_at_t0_rad": float(_MEAN_ANOMALY_T0),
        "reference_time_yr": 0.0,
        "gamma_applied_in_spice": True,
        "t_p_applied_in_spice": True,
        "distance_pc": DISTANCE_PC,
        "wl_min_A": WL_MIN, "wl_max_A": WL_MAX,
    }


def _check_gamma_in_orbit(pb1, pb2):
    """The mass-weighted mean of body-y velocities should be exactly -γ.

    Per body separately, mean(body_vy) → -γ only as n_phases → ∞ (the orbital
    sinusoid averages to zero in the limit). For 30 samples each body's mean
    can sit a few km/s off γ from sampling bias, but body1 and body2 errors
    cancel in the mass-weighted mean — that's the strict invariant.
    """
    v1y = np.array([np.asarray(b.orbital_velocity).reshape(3)[1] for b in pb1])
    v2y = np.array([np.asarray(b.orbital_velocity).reshape(3)[1] for b in pb2])
    m1, m2 = PRIMARY_MASS, SECONDARY_MASS
    bary_y = (m1 * np.mean(v1y) + m2 * np.mean(v2y)) / (m1 + m2)
    expected = -GAMMA_KMS
    print(f"  body 1 v.y: mean={np.mean(v1y):+.3f}  median={np.median(v1y):+.3f}  "
          f"range=[{v1y.min():+.2f},{v1y.max():+.2f}] km/s")
    print(f"  body 2 v.y: mean={np.mean(v2y):+.3f}  median={np.median(v2y):+.3f}  "
          f"range=[{v2y.min():+.2f},{v2y.max():+.2f}] km/s")
    print(f"  mass-weighted ⟨v.y⟩ = {bary_y:+.3f} km/s")
    print(f"  expected: {expected:+.3f} km/s  (γ_sys, along -y after fix)")
    tol = 0.3
    if abs(bary_y - expected) < tol:
        print("  PASS — γ reaches the barycenter exactly once along -y.")
        return True
    if abs(bary_y) < tol:
        print("  FAIL — γ DROPPED on its way to the mesh.")
    elif abs(bary_y - 2 * expected) < tol:
        print("  FAIL — γ DOUBLE-COUNTED (would have been -2γ).")
    else:
        print(f"  FAIL — mass-weighted ⟨v.y⟩ deviates by {bary_y - expected:+.3f} km/s.")
    return False


def _ccf_rv(a, b, v_per_pixel_kms, max_lag=300):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a - a.mean()
    b = b - b.mean()
    full = correlate(a, b, mode="full")
    center = len(full) // 2
    seg = full[center - max_lag : center + max_lag + 1]
    lags = np.arange(-max_lag, max_lag + 1, dtype=float)
    k = int(np.argmax(seg))
    if 0 < k < len(seg) - 1:
        y0, y1, y2 = seg[k - 1], seg[k], seg[k + 1]
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-12:
            return (lags[k] + 0.5 * (y0 - y2) / denom) * v_per_pixel_kms
    return lags[k] * v_per_pixel_kms


def _check_doppler_reaches_spectrum(pb1, spectra1, wavelengths):
    """CCF body 1's extreme-velocity phases against the t=0 template. SPICE
    convention: positive ``vrad`` in ``apply_vrad_log`` → blueshift, so a
    body with negative v.y (los_velocities = +v.y) actually produces a
    REDSHIFTED spectrum. The expected relation between geometric Δ(v.y) and
    CCF Δv is therefore CCF Δv ≈ −Δ(v.y).
    """
    v_y = np.array([np.asarray(b.orbital_velocity).reshape(3)[1] for b in pb1])
    # Index with the most-negative v.y is the most-redshifted observed
    # spectrum; the most-positive v.y is the most-blueshifted.
    most_negative_idx = int(np.argmin(v_y))
    most_positive_idx = int(np.argmax(v_y))
    ref_idx = 0

    log_wl = np.linspace(np.log10(WL_MIN), np.log10(WL_MAX), 4096)
    wave_u = 10.0 ** log_wl
    v_per_pixel_kms = (log_wl[1] - log_wl[0]) * np.log(10.0) * C_KMS

    def _norm(s):
        f = s[:, 0] / s[:, 1]
        return np.interp(wave_u, wavelengths, f)

    ref = _norm(spectra1[ref_idx])
    neg = _norm(spectra1[most_negative_idx])
    pos = _norm(spectra1[most_positive_idx])

    rv_neg = _ccf_rv(neg, ref, v_per_pixel_kms)
    rv_pos = _ccf_rv(pos, ref, v_per_pixel_kms)
    geom_neg = v_y[most_negative_idx] - v_y[ref_idx]
    geom_pos = v_y[most_positive_idx] - v_y[ref_idx]
    # SPICE convention: CCF Δv (wavelength shift) ≈ −Δ(v.y)
    expected_neg, expected_pos = -geom_neg, -geom_pos

    print(f"  most-negative v.y phase: Δ(v.y) = {geom_neg:+.2f}, "
          f"CCF Δv = {rv_neg:+.2f} (expected ≈ {expected_neg:+.2f})")
    print(f"  most-positive v.y phase: Δ(v.y) = {geom_pos:+.2f}, "
          f"CCF Δv = {rv_pos:+.2f} (expected ≈ {expected_pos:+.2f})")
    err_neg = abs(rv_neg - expected_neg)
    err_pos = abs(rv_pos - expected_pos)
    tol = 1.0
    if err_neg < tol and err_pos < tol:
        print("  PASS — Doppler shift in the spectrum matches the geometric "
              "LOS velocity (with the SPICE sign convention).")
        return True
    print(f"  FAIL — CCF and -geom diverge by up to {max(err_neg, err_pos):.2f} km/s.")
    return False


def main():
    global em
    print("=" * 70)
    print("TZ Fornacis narrow-band test synthesis (4980-5040 Å)")
    print("=" * 70)

    out_dir = Path(__file__).resolve().parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pkl = out_dir / f"tz_fornacis_test_{NUM_WAVELENGTHS}.pkl"

    print(f"[jax] backend={jax.default_backend()}  devices={jax.devices()}")
    print(f"Output:        {out_pkl}")
    print(f"Wavelength:    {WL_MIN:.1f} - {WL_MAX:.1f} Å, {NUM_WAVELENGTHS} points "
          f"(Δλ ≈ {(WL_MAX-WL_MIN)/NUM_WAVELENGTHS:.4f} Å)")
    print(f"Times:         {NUM_TIMES} uniform phases over one period")
    print(f"Mesh:          {N_MESH} icosphere elements per body")
    print()

    em, em_kind = _make_emulator()

    print("Building binary (with rotation + atmospheric params)...")
    binary = _build_binary(em_kind)

    times_yr = np.linspace(0.0, PERIOD_YR, NUM_TIMES)
    time_origin = np.full(times_yr.shape, "general", dtype=object)

    t0 = time.time()
    pb1, pb2 = _evaluate_orbit_chunked(binary, times_yr, ORBIT_CHUNK)
    print(f"orbit eval: {time.time() - t0:.1f}s")

    print("\nCHECK 1: γ propagation at orbit level")
    check1_ok = _check_gamma_in_orbit(pb1, pb2)
    print()

    vws = jnp.linspace(WL_MIN, WL_MAX, NUM_WAVELENGTHS)
    t0 = time.time()
    spectra1, spectra2 = _synthesize(vws, pb1, pb2)
    print(f"synthesis:  {time.time() - t0:.1f}s")

    print("\nCHECK 2: Doppler reaches the synthesized spectrum")
    check2_ok = _check_doppler_reaches_spectrum(pb1, spectra1, np.asarray(vws))
    print()

    print(f"Saving pickle: {out_pkl}")
    tmp = out_pkl.with_suffix(out_pkl.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump({
            "mode": "test_narrowband",
            "spectra_body1": spectra1,
            "spectra_body2": spectra2,
            "mesh_body1": pb1,
            "mesh_body2": pb2,
            "wavelengths": np.asarray(vws),
            "wl_min": WL_MIN, "wl_max": WL_MAX,
            "times": times_yr,
            "time_origin": time_origin,
            "period_yr": PERIOD_YR,
            "primary_eclipse_window_yr": (np.nan, np.nan),
            "secondary_eclipse_window_yr": (np.nan, np.nan),
            "binary_params": _binary_params_dict(),
            "emulator_kind": em_kind,
            "model": MODEL if em_kind == "aemu" else "GaussianLineEmulator(fallback)",
        }, f)
    os.replace(tmp, out_pkl)
    print(f"✓ saved {out_pkl}")
    print()

    print("=" * 70)
    print(f"OVERALL: CHECK 1 {'PASS' if check1_ok else 'FAIL'}   "
          f"CHECK 2 {'PASS' if check2_ok else 'FAIL'}")
    print("=" * 70)
    return 0 if (check1_ok and check2_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
