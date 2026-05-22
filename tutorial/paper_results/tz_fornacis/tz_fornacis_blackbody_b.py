"""TZ Fornacis Stromgren b from blackbody spectra — fast check vs Clausen photometry.

Uses the same orbit, time grid, and eclipse-window mapping as ``tz_fornacis_spectra.py``
(Kepler eclipse times shifted to SPICE t=0), but ``spice.spectrum.Blackbody`` instead of
the TPayne emulator. No macroturbulence (not meaningful for a featureless continuum).

Usage
-----
::

    python tz_fornacis_blackbody_b.py --output-dir /path/to/repo
    python tz_fornacis_blackbody_b.py --compare --output-dir /path/to/repo
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import click
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

# Re-use TZ For constants and time-grid helpers from the spectra script.
sys.path.insert(0, str(HERE))
import tz_fornacis_spectra as tzs  # noqa: E402

# Coarser mesh — blackbody is cheap; 500 is enough for eclipse geometry.
tzs.N_MESH = int(os.environ.get("TZ_FOR_BB_N_MESH", "500"))

from spice.spectrum import AB_passband_luminosity, simulate_observed_flux  # noqa: E402
from spice.spectrum.blackbody import Blackbody  # noqa: E402
from spice.spectrum.filter import Stromgrenb  # noqa: E402

# Stromgren b band (same window as ``tz_fornacis_b.pkl`` from the narrow-grid script).
WL_MIN_B = 4520.0
WL_MAX_B = 4820.0

PERIOD_DAYS = tzs.PERIOD_DAYS
T0_PHOT_HJD = 2_445_183.92451
HJD_OFFSET = 2_440_000.0
CLAUSEN_CSV = HERE / "tzfor_lightcurve.csv"

# Quadratic LD (Claret form: I/I(1) = 1 - a*(1-mu) - b*(1-mu)^2) for Strömgren b
# at the TZ For component atmospheric parameters. Values are reasonable
# interpolations of Claret & Bloemen 2011 (A&A 529, A75) for:
#   primary  (K giant,  Teff=4930 K, logg=2.91, [Fe/H]=-0.30): strong linear, small quadratic.
#   secondary (F subg., Teff=6650 K, logg=3.35, [Fe/H]=-0.30): standard F-star LD.
# Refine from the Claret tables if you need tighter ingress/egress agreement.
LD_LAW = "quadratic"
LD_COEFFS_PRIMARY = (0.75, 0.05)
LD_COEFFS_SECONDARY = (0.45, 0.25)


def _synthesize_bb(vws, pb1, pb2, distance_pc: float):
    log_vws = jnp.log10(vws)
    bb1 = Blackbody(ld_law=LD_LAW, ld_coeffs=LD_COEFFS_PRIMARY)
    bb2 = Blackbody(ld_law=LD_LAW, ld_coeffs=LD_COEFFS_SECONDARY)
    s1 = [
        simulate_observed_flux(bb1.intensity, _pb, log_vws, distance=distance_pc)
        for _pb in pb1
    ]
    s2 = [
        simulate_observed_flux(bb2.intensity, _pb, log_vws, distance=distance_pc)
        for _pb in pb2
    ]
    return np.asarray(s1), np.asarray(s2)


def _stromgren_b_mag(wavelengths, spectra1, spectra2):
    filt = Stromgrenb()
    return np.array([
        AB_passband_luminosity(filt, wavelengths, s1[:, 0] + s2[:, 0])
        for s1, s2 in zip(spectra1, spectra2)
    ])


def _oot_median(phase_phot, values, oot_pad=0.05):
    """Median of `values` over phases outside ±oot_pad cycles of the eclipses.

    Both the Clausen and SPICE light curves are folded against Clausen's
    primary T0 (primary at phi=0/1, secondary at phi=0.5). Using this median
    for the Δm zero-point keeps the two curves on a consistent baseline
    despite their different in-eclipse sampling cadence.
    """
    phase = np.asarray(phase_phot)
    oot = (
        ((phase > oot_pad) & (phase < 0.5 - oot_pad))
        | ((phase > 0.5 + oot_pad) & (phase < 1.0 - oot_pad))
    )
    if not oot.any():
        return float(np.median(values))
    return float(np.median(np.asarray(values)[oot]))


def _folded_correlation(phase_spice, dm_spice, phase_obs, dm_obs, n_shift=400):
    phase_spice = np.asarray(phase_spice, dtype=float)
    dm_spice = np.asarray(dm_spice, dtype=float)
    phase_obs = np.asarray(phase_obs, dtype=float)
    dm_obs = np.asarray(dm_obs, dtype=float)
    best_r, best_shift = -2.0, 0.0
    order_s = np.argsort(phase_spice)
    ph_s, dm_s = phase_spice[order_s], dm_spice[order_s]
    order_o = np.argsort(phase_obs)
    ph_o, dm_o = phase_obs[order_o], dm_obs[order_o]
    for shift in np.linspace(0.0, 1.0, n_shift, endpoint=False):
        ph_try = (phase_spice + shift) % 1.0
        order_t = np.argsort(ph_try)
        dm_interp = np.interp(ph_o, ph_try[order_t], dm_spice[order_t], period=1.0)
        r = np.corrcoef(dm_interp, dm_o)[0, 1]
        if r > best_r:
            best_r, best_shift = r, shift
    return best_shift, best_r


def _compare_clausen(times_yr, period_yr, m0, b_mag, t_p_hjd, plot_path: Path | None):
    import pandas as pd
    import matplotlib.pyplot as plt

    phase_spice = (times_yr / period_yr + m0 / (2.0 * np.pi)) % 1.0
    hjd = t_p_hjd + phase_spice * PERIOD_DAYS
    phase_phot = ((hjd - T0_PHOT_HJD) % PERIOD_DAYS) / PERIOD_DAYS

    df = pd.read_csv(CLAUSEN_CSV)
    df["HJD_full"] = df["HJD"] + HJD_OFFSET
    df_c = df.loc[df["HJD"] > 4500.0].copy()
    phase_clausen = (
        (df_c["HJD_full"].to_numpy() - T0_PHOT_HJD) % PERIOD_DAYS
    ) / PERIOD_DAYS

    # Δm zero-points come from the same out-of-eclipse phase mask on both
    # curves; otherwise the medians end up biased by the differing in-eclipse
    # cadence (Clausen's HJD>4500 subset is eclipse-campaign dense, SPICE's
    # time grid mixes uniform sampling with dense eclipse windows).
    dm = b_mag - _oot_median(phase_phot, b_mag)
    dm_clausen = df_c["b"].to_numpy() - _oot_median(
        phase_clausen, df_c["b"].to_numpy()
    )

    best_shift, best_r = _folded_correlation(phase_phot, dm, phase_clausen, dm_clausen)
    phase_aligned = (phase_phot + best_shift) % 1.0

    i_max = int(np.argmax(dm))
    print(f"Blackbody Δm range: {dm.min():.4f} – {dm.max():.4f}")
    print(f"Deepest feature at phot. phase {phase_phot[i_max]:.3f} d (Δm={dm[i_max]:.3f})")
    print(f"Best phase shift (SPICE phot + shift): {best_shift:.4f}  ({best_shift * PERIOD_DAYS:.2f} d)")
    print(f"Pearson r vs Clausen Δb: {best_r:.4f}")

    if plot_path is None:
        return best_shift, best_r

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    order_c = np.argsort(phase_clausen)
    order_s = np.argsort(phase_aligned)
    axes[0].plot(phase_clausen[order_c], dm_clausen[order_c], ".", color="0.35", ms=2, alpha=0.4,
                 label="Clausen Δb")
    axes[0].plot(phase_aligned[order_s], dm[order_s], "x-", color="tab:red", ms=5, lw=0.8,
                 label=f"Blackbody Δb (r={best_r:.3f})")
    axes[0].set_ylabel("Δm (from median)")
    axes[0].legend(loc="upper right")
    axes[0].set_title("TZ For — phase-folded Stromgren b (photometric T₀, aligned)")

    dm_c_on_s = np.interp(phase_aligned[order_s], phase_clausen[order_c], dm_clausen[order_c])
    axes[1].plot(phase_aligned[order_s], dm[order_s] - dm_c_on_s, "o-", ms=4, lw=0.8)
    axes[1].set_xlabel("Photometric orbital phase")
    axes[1].set_ylabel("SPICE − Clausen")
    axes[1].set_title("Residual after phase alignment")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    print(f"Wrote {plot_path}")
    return best_shift, best_r


@click.command()
@click.option("--num-times", default=150, show_default=True)
@click.option("--num-eclipse-times", default=80, show_default=True)
@click.option("--num-wavelengths", default=4000, show_default=True,
              help="Wavelength samples across the Stromgren b window.")
@click.option("--orbit-chunk", default=30, show_default=True)
@click.option("--output-dir", default=".", type=click.Path(file_okay=False))
@click.option("--force/--no-force", default=False)
@click.option("--compare/--no-compare", default=True,
              help="Fold against Clausen CSV and write a comparison plot.")
def main(num_times, num_eclipse_times, num_wavelengths, orbit_chunk, output_dir,
         force, compare):
    """Synthesize blackbody spectra and optional Clausen comparison."""
    out_dir = Path(output_dir)
    out_pkl = out_dir / "tz_fornacis_b_blackbody.pkl"
    if out_pkl.exists() and not force:
        print(f"[skip] {out_pkl} exists (use --force)")
        if not compare:
            return
        with open(out_pkl, "rb") as f:
            data = pickle.load(f)
        bp = data["binary_params"]
        _compare_clausen(
            np.asarray(data["times"]), float(data["period_yr"]),
            float(bp["mean_anomaly_at_t0_rad"]), np.asarray(data["b_mag_ab"]),
            float(bp["t_p_hjd"]),
            HERE / "clausen_b_blackbody_compare.png",
        )
        return

    bb = Blackbody()
    print(f"Building binary (N_MESH={tzs.N_MESH}, orbit anchors={tzs.ORBIT_RESOLUTION_POINTS})...")
    t0 = time.time()
    (binary, period_yr, primary_window, secondary_window,
     primary_kepler_yr, secondary_kepler_yr) = tzs._build_binary_and_eclipses(bb)
    times_yr, time_origin = tzs._build_merged_times(
        period_yr, primary_window, secondary_window, num_times, num_eclipse_times,
    )
    print(f"Time grid: {len(times_yr)} points in {time.time() - t0:.1f}s")

    vws = jnp.linspace(WL_MIN_B, WL_MAX_B, num_wavelengths)
    pb1, pb2 = tzs._evaluate_orbit_chunked(binary, times_yr, orbit_chunk)
    print("Synthesizing blackbody spectra...")
    t0 = time.time()
    spectra1, spectra2 = _synthesize_bb(vws, pb1, pb2, tzs.DISTANCE_PC)
    print(f"Synthesis done in {time.time() - t0:.1f}s")

    b_mag = _stromgren_b_mag(np.asarray(vws), spectra1, spectra2)
    bp = tzs._binary_params_dict(wl_min=WL_MIN_B, wl_max=WL_MAX_B)
    bp["spectrum_model"] = "blackbody"
    bp["ld_law"] = LD_LAW
    bp["ld_coeffs_primary"] = tuple(LD_COEFFS_PRIMARY)
    bp["ld_coeffs_secondary"] = tuple(LD_COEFFS_SECONDARY)

    payload = {
        "mode": "merged_blackbody",
        "spectra_body1": spectra1,
        "spectra_body2": spectra2,
        "wavelengths": np.asarray(vws),
        "wl_min": WL_MIN_B,
        "wl_max": WL_MAX_B,
        "b_mag_ab": b_mag,
        "times": times_yr,
        "time_origin": time_origin,
        "period_yr": period_yr,
        "primary_eclipse_window_yr": primary_window,
        "secondary_eclipse_window_yr": secondary_window,
        "primary_eclipse_window_kepler_yr": primary_kepler_yr,
        "secondary_eclipse_window_kepler_yr": secondary_kepler_yr,
        "binary_params": bp,
    }
    tmp = out_pkl.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f)
    os.replace(tmp, out_pkl)
    print(f"Saved {out_pkl}")

    if compare:
        _compare_clausen(
            times_yr, period_yr, float(bp["mean_anomaly_at_t0_rad"]), b_mag,
            float(bp["t_p_hjd"]), HERE / "clausen_b_blackbody_compare.png",
        )


if __name__ == "__main__":
    main()
