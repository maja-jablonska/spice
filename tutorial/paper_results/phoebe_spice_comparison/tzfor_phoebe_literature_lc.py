"""TZ For literature light curves computed by PHOEBE with its own atmospheres, scored exactly like the SPICE fit.

Question: are the TZ For photometric residuals (b rms 3.0, y 2.9 mmag, chi2/N 0.714 with the phase-3 emulator) something
SPICE introduces, or what any code gets from the literature parameters? PHOEBE builds the system itself (Roche geometry,
Lucy gravity darkening, its own irradiation, ck2004 / phoenix intensities with interpolated limb darkening) at the same
parameters as the SPICE literature run whose ``--dump-model`` npz is given, and its Stromgren b / y magnitudes at the
Clausen (1991) observation times go through the SPICE fit's own scoring (tzfor_fullrange_fit.band_residuals): global
median zero point, per-night zero points profiled with the same Gaussian prior, the same clipped points, the same sigma.
The SPICE model in the dump is re-scored the same way first, so the two numbers are directly comparable (and the
re-score must reproduce the dump's chi2).

    python tzfor_phoebe_literature_lc.py --dump tzfor_lit_exact_p3_gallenne.npz --params gallenne --atm ck2004 \
        --photometry tzfor_lightcurve.csv --out out/phoebe_gallenne_ck2004.npz

Parameter sets (Gallenne et al. 2016 geometry in all):
  base           [M/H] -0.30 / -0.30, v_rot 5.5 / 38 km/s   (our earlier FITTED metallicity, not a literature value)
  gallenne       [M/H] +0.02 / -0.05, v sin i 6.1 / 45.7 km/s (Gallenne et al. 2016, adopted Teff2 6350)
  gallenne_spec  as gallenne with their spectroscopic Teff2 6650 K (Table 2 as published)
  andersen       Teff 5000 / 6350, Fe/H +0.1 / +0.1, v sin i 5.5 / 38 km/s (Andersen et al. 1991)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for p in (HERE.parent / "tz_fornacis", Path(os.environ.get("TZFOR_DIR", ""))):
    if p and (p / "tzfor_constants.py").is_file():
        sys.path.insert(0, str(p))
import tzfor_constants as K  # noqa: E402

HJD_OFFSET = 2440000.0   # Clausen (1991): truncated HJD 244xxxx (as in the fit scripts' load_photometry)
SIG_PH = {"b": 0.0041, "y": 0.0035}   # tzfor_fullrange_fit.py
PARAMS = {
    "base": dict(teff=(4930.0, 6350.0), abun=(-0.30, -0.30), vrot=(K.PRIMARY_VROT_KMS, K.SECONDARY_VROT_KMS)),
    "gallenne": dict(teff=(4930.0, 6350.0), abun=(0.02, -0.05),
                     vrot=tuple(v / np.sin(np.deg2rad(K.INCL_DEG)) for v in (6.1, 45.7))),
    "gallenne_spec": dict(teff=(4930.0, 6650.0), abun=(0.02, -0.05),
                          vrot=tuple(v / np.sin(np.deg2rad(K.INCL_DEG)) for v in (6.1, 45.7))),
    "andersen": dict(teff=(5000.0, 6350.0), abun=(0.10, 0.10),
                     vrot=tuple(v / np.sin(np.deg2rad(K.INCL_DEG)) for v in (5.5, 38.0))),
}


def build(params, atm, n_mesh, irrad, gravb, radii=None):
    import phoebe
    from astropy import units as u
    phoebe.logger(clevel="ERROR")
    b = phoebe.default_binary()
    b.flip_constraint("mass@primary", solve_for="sma")
    b.set_value("period@binary@component", K.PERIOD_DAYS)
    b.set_value("q@binary@component", K.SECONDARY_MASS / K.PRIMARY_MASS)
    b.set_value("ecc@binary@component", K.ECC)
    b.set_value("mass@primary@component", K.PRIMARY_MASS)
    b.set_value_all("incl@binary", K.INCL_DEG)
    b.set_value("per0@binary@component", K.PER0_DEG)
    b.set_value("long_an@binary@component", K.LONG_AN_DEG)
    # T_P_HJD is the DEEPER minimum; PHOEBE's t0_supconj is the other conjunction (see tzfor_phoebe_aemu_photometry.py)
    b.set_value("t0_supconj@binary@component", K.T_P_HJD - K.PERIOD_DAYS / 2.0)
    b.set_value("distance", K.DISTANCE_PC * u.pc)
    for i, comp in enumerate(("primary", "secondary")):
        radius = (K.PRIMARY_RADIUS, K.SECONDARY_RADIUS)[i] if radii is None else radii[i]
        b.set_value(f"requiv@{comp}@component", radius)
        b.set_value(f"teff@{comp}@component", params["teff"][i])
        b.set_value(f"abun@{comp}@component", params["abun"][i])
        v_sync = 2 * np.pi * radius * 6.957e5 / (K.PERIOD_DAYS * 86400.0)   # syncpar default 1 under-rotates the secondary 14x
        b.set_value(f"syncpar@{comp}@component", params["vrot"][i] / v_sync)
    b.set_value_all("distortion_method", "roche")
    b.set_value_all("ntriangles", n_mesh)
    b.set_value_all("gravb_bol", gravb)
    b.set_value_all("irrad_method", irrad)
    b.set_value_all("atm", atm)
    b.atm_choice = atm   # ld_mode is a per-dataset parameter: set when the datasets are added (phoebe_mags)
    return b


def phoebe_mags(b, times, n_mesh_label=""):
    """Stromgren b / y magnitudes (arbitrary zero point) at the given times."""
    order = np.argsort(times); ts = times[order]   # PHOEBE sorts compute_times internally
    for band in ("b", "y"):
        b.add_dataset("lc", compute_times=ts, passband=f"Stromgren:{band}", dataset=f"lc{band}", overwrite=True)
        if getattr(b, "atm_choice", "blackbody") != "blackbody":
            b.set_value_all("ld_mode", dataset=f"lc{band}", value="interp")
    t0 = time.time()
    b.run_compute(model="phmodel", overwrite=True)
    out = {}
    for band in ("b", "y"):
        f = np.asarray(b.get_value(f"fluxes@lc{band}@phmodel"), float)
        m = np.empty_like(f); m[order] = -2.5 * np.log10(f)
        out[band] = m
    print(f"  PHOEBE {len(times)} times {n_mesh_label} in {time.time() - t0:.0f} s", flush=True)
    return out


def score(model_at_obs, obs, w, night_idx, n_nights, sig, sig_night):
    """tzfor_fullrange_fit.band_residuals + its chi2 term, for one band."""
    res = obs - model_at_obs; res = res - np.median(res)
    ww = w / sig ** 2
    z = np.bincount(night_idx, ww * res, n_nights) / (np.bincount(night_idx, ww, n_nights) + 1.0 / sig_night ** 2)
    res = res - z[night_idx]
    c2 = float(np.sum(w * res ** 2) / sig ** 2 + np.sum((z / sig_night) ** 2))
    rms = float(np.sqrt(np.sum(w * res ** 2) / np.sum(w)))
    return c2, rms, res, z


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dump", required=True, help="SPICE literature run --dump-model npz (same parameter set)")
    ap.add_argument("--params", choices=sorted(PARAMS), required=True)
    ap.add_argument("--atm", default="ck2004", choices=["ck2004", "phoenix", "blackbody"])
    ap.add_argument("--irrad", default="horvat", choices=["horvat", "wilson", "none"])
    ap.add_argument("--gravb", type=float, default=0.32, help="PHOEBE gravb_bol (Lucy 0.32 = the SPICE run's beta 0.079 x 4)")
    ap.add_argument("--n-mesh", type=int, default=3000)
    ap.add_argument("--photometry", required=True)
    ap.add_argument("--nightly-sigma", type=float, default=3.0, help="mmag, as --phot-nightly-sigma in the SPICE run")
    ap.add_argument("--max-points", type=int, default=0, help="smoke: only the first N observations")
    ap.add_argument("--out", required=True)
    ap.add_argument("--requiv", type=float, nargs=2, default=None, metavar=("R1", "R2"),
                    help="volume-equivalent radii [Rsun] instead of tzfor_constants (Gallenne's light-curve radii: 8.34 3.97)")
    ap.add_argument("--vrot", type=float, nargs=2, default=None, metavar=("V1", "V2"),
                    help="equatorial rotation [km/s] instead of the parameter set's (sets syncpar, hence the Roche flattening)")
    a = ap.parse_args()
    import pandas as pd
    D = np.load(a.dump, allow_pickle=True)
    df = pd.read_csv(a.photometry)
    hjd = df["HJD"].values.astype(float) + HJD_OFFSET
    ph = ((hjd - K.T_P_HJD) % K.PERIOD_DAYS) / K.PERIOD_DAYS
    n = len(hjd)
    if D["lc_b_obs_phase"].size != n or not np.allclose(D["lc_b_obs_phase"], ph, atol=1e-9):
        raise SystemExit(f"dump observation phases ({D['lc_b_obs_phase'].size}) do not match {a.photometry} ({n} rows)")
    _, night_idx = np.unique(np.floor(df["HJD"].values), return_inverse=True); n_nights = int(night_idx.max()) + 1
    sel = np.arange(n) if not a.max_points else np.arange(min(n, a.max_points))
    sig_night = a.nightly_sigma / 1000.0
    N_ph = int(sum(D[f"lc_{bd}_weight"].sum() for bd in "by"))

    # 1. re-score the SPICE model from the dump (must reproduce the dump's chi2)
    res = {"params": a.params, "atm": a.atm, "irrad": a.irrad, "gravb": a.gravb, "n_mesh": a.n_mesh, "N_ph": N_ph}
    c_sp = 0.0
    for bd in "by":
        c2, rms, r, z = score(D[f"lc_{bd}_model_at_obs"], df[bd].values, D[f"lc_{bd}_weight"], night_idx, n_nights, SIG_PH[bd], sig_night)
        c_sp += c2; res[f"spice_{bd}_rms_mmag"] = 1e3 * rms; res[f"spice_{bd}_resid"] = r
    res["spice_chi2_phot_N"] = c_sp / N_ph
    print(f"SPICE ({a.dump}): chi2_phot/N {c_sp / N_ph:.3f} (dump {float(D['chi2_phot_exact']) / int(D['N_ph']):.3f}); "
          f"b rms {res['spice_b_rms_mmag']:.2f}  y rms {res['spice_y_rms_mmag']:.2f} mmag", flush=True)
    if not a.max_points and abs(c_sp / N_ph - float(D["chi2_phot_exact"]) / int(D["N_ph"])) > 2e-3:
        raise SystemExit("re-scoring the SPICE model does not reproduce the dump's chi2: the scoring differs from the fit")

    # 2. PHOEBE at the observation times and at the SPICE model-curve phases
    P = dict(PARAMS[a.params])
    if a.vrot is not None: P["vrot"] = tuple(a.vrot)
    print(f"PHOEBE {a.params}: Teff {P['teff']}, abun {P['abun']}, v_rot {tuple(round(v, 2) for v in P['vrot'])} km/s, "
          f"atm {a.atm}, irrad {a.irrad}, gravb {a.gravb}, roche, {a.n_mesh} triangles", flush=True)
    b = build(P, a.atm, a.n_mesh, a.irrad, a.gravb, radii=a.requiv)
    res["requiv"] = np.asarray(a.requiv if a.requiv is not None else (K.PRIMARY_RADIUS, K.SECONDARY_RADIUS)); res["vrot"] = np.asarray(P["vrot"])
    m_obs = phoebe_mags(b, hjd[sel], "(observations)")
    mph = D["lc_b_model_phase"]
    m_curve = phoebe_mags(b, K.T_P_HJD + mph * K.PERIOD_DAYS, "(SPICE model phases)")
    c_ph = 0.0
    for bd in "by":
        obs = df[bd].values[sel]; w = D[f"lc_{bd}_weight"][sel]
        c2, rms, r, z = score(m_obs[bd], obs, w, night_idx[sel], n_nights, SIG_PH[bd], sig_night)
        c_ph += c2; res[f"phoebe_{bd}_rms_mmag"] = 1e3 * rms; res[f"phoebe_{bd}_resid"] = r
        res[f"phoebe_{bd}_at_obs"] = m_obs[bd]; res[f"phoebe_{bd}_curve"] = m_curve[bd]
        res[f"spice_{bd}_curve"] = D[f"lc_{bd}_model_mag"]
        # model-vs-model on the SPICE phases, each curve relative to its own median (zero points are arbitrary)
        dmm = (m_curve[bd] - np.median(m_curve[bd])) - (D[f"lc_{bd}_model_mag"] - np.median(D[f"lc_{bd}_model_mag"]))
        res[f"phoebe_minus_spice_{bd}_curve"] = dmm
        print(f"  {bd}: PHOEBE rms {1e3 * rms:.2f} mmag (SPICE {res[f'spice_{bd}_rms_mmag']:.2f}); PHOEBE - SPICE model curve "
              f"max {1e3 * np.max(np.abs(dmm)):.2f} mmag, rms {1e3 * np.sqrt(np.mean(dmm ** 2)):.2f} mmag", flush=True)
    n_used = int(sum(D[f"lc_{bd}_weight"][sel].sum() for bd in "by"))
    res["phoebe_chi2_phot_N"] = c_ph / n_used
    res["model_phase"] = mph
    print(f"RESULT {a.params} {a.atm} irrad={a.irrad}: chi2_phot/N PHOEBE {c_ph / n_used:.3f} vs SPICE {c_sp / N_ph:.3f}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **{k: np.asarray(v) for k, v in res.items()})
    print("wrote", a.out)


if __name__ == "__main__":
    main()
