"""Score TZ For light-curve models against Clausen (1991) b, y with a photometric-ephemeris offset and a choice of nightly zero points.

The model curves (SPICE: a tzfor_fullrange_fit.py --dump-model npz; PHOEBE: tzfor_phoebe_literature_lc.py outputs, whose curves
are on the same model phases) are interpolated at (phase - dphi) exactly as tzfor_fullrange_fit.band_residuals does, a global
median zero point is removed, points more than --clip mmag from the model are dropped, and then per night:

    none            no nightly zero points
    out-of-eclipse  a zero point per night from that night's points outside both eclipses (median, >= 3 points), so an eclipse
                    misfit cannot be absorbed
    profiled        the fit's closed-form profiled offset with a Gaussian prior of --nightly-sigma mmag (all points)

dphi is either given or fitted per model by a grid search on a truncated quadratic loss (a dropped point costs the clip level) (the photometric ephemeris of Clausen's 1976-83 data relative to the
spectroscopic T_P of Gallenne et al. 2016 is a nuisance parameter). Reports chi2/N and the rms in and out of eclipse.

    python tzfor_phot_rescore.py --dump DUMP.npz --csv tzfor_lightcurve.csv [--phoebe P1.npz ...] [--dphi fit|VALUE]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tzfor_constants as K  # noqa: E402

SIG = {"b": 0.0041, "y": 0.0035}          # tzfor_fullrange_fit.py catalogue sigmas
HJD_OFFSET = 2440000.0                     # Clausen (1991) truncated HJD
ECL_HALF = 0.012                           # half-width of each eclipse in phase (both eclipses last ~0.02 in phase)


def residuals(curve_phase, curve_mag, ph, mag, dphi, clip_mmag, nightly, night, ecl, sig, sig_night):
    o = np.argsort(curve_phase % 1.0); mp, mm = (curve_phase % 1.0)[o], np.asarray(curve_mag)[o]
    r = mag - np.interp((ph - dphi) % 1.0, mp, mm, period=1.0); r = r - np.median(r)
    keep = np.abs(r) <= clip_mmag / 1000.0
    prior = 0.0
    for k in np.unique(night):
        m = night == k
        if nightly == "out-of-eclipse":
            u = m & keep & ~ecl
            if u.sum() >= 3:
                r[m] -= np.median(r[u])
        elif nightly == "profiled":
            u = m & keep; w = u.sum() / sig ** 2
            z = np.sum(r[u]) / sig ** 2 / (w + 1.0 / sig_night ** 2) if u.any() else 0.0
            r[m] -= z; prior += (z / sig_night) ** 2
    return r, keep, prior


def score(curves, ph, obs, dphi, a, night, ecl):
    """(chi2/N over the kept points, N, per-band stats, truncated loss). The truncated loss sum min((r/s)^2, (clip/s)^2) / N_all
    charges a dropped point the clip level, so a dphi search cannot lower it by dropping points; dphi is chosen on it."""
    c2 = 0.0; n = 0; out = {}; trunc = 0.0; n_all = 0
    for b in "by":
        r, keep, prior = residuals(curves[b][0], curves[b][1], ph, obs[b], dphi, a.clip, a.nightly, night, ecl, SIG[b], a.nightly_sigma / 1e3)
        c2 += np.sum((r[keep] / SIG[b]) ** 2) + prior; n += int(keep.sum())
        trunc += np.sum(np.minimum((r / SIG[b]) ** 2, (a.clip / 1e3 / SIG[b]) ** 2)) + prior; n_all += r.size
        out[b] = dict(rms=1e3 * r[keep].std(), rms_eclipse=1e3 * r[keep & ecl].std(), mean_eclipse=1e3 * r[keep & ecl].mean(),
                      rms_outside=1e3 * r[keep & ~ecl].std(), clipped=int((~keep).sum()))
    return c2 / n, n, out, trunc / n_all


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dump", required=True); ap.add_argument("--csv", required=True); ap.add_argument("--phoebe", nargs="*", default=[])
    ap.add_argument("--dphi", default="fit", help="'fit' (grid search per model) or a value in cycles")
    ap.add_argument("--nightly", choices=("none", "out-of-eclipse", "profiled"), default="none")
    ap.add_argument("--nightly-sigma", type=float, default=3.0); ap.add_argument("--clip", type=float, default=15.0)
    ap.add_argument("--json", default=None)
    ap.add_argument("--ephemeris", type=float, nargs=2, default=None, metavar=("T0_HJD", "P_D"),
                    help="phase the photometry with this published ephemeris (e.g. Andersen et al. 1991: 2445032.6091 75.66675) "
                         "instead of Gallenne's T_P / P; use with --dphi 0 for a model with nothing adjusted but the zero points")
    a = ap.parse_args()
    import pandas as pd
    df = pd.read_csv(a.csv); hjd = df["HJD"].values.astype(float) + HJD_OFFSET
    T0, Pd = (K.T_P_HJD, K.PERIOD_DAYS) if a.ephemeris is None else a.ephemeris
    ph = ((hjd - T0) % Pd) / Pd
    night = np.floor(df["HJD"].values).astype(int)
    ecl = (np.minimum(ph, 1 - ph) < ECL_HALF) | (np.abs(ph - 0.5) < ECL_HALF)
    obs = {b: df[b].values.astype(float) for b in "by"}
    D = np.load(a.dump, allow_pickle=True)
    pn = [str(x) for x in D["pnames"]]; dphi_dump = float(np.asarray(D["theta"], float)[pn.index("dphi")]) if "dphi" in pn else 0.0
    dd = ((D["lc_b_obs_phase"] - (ph - dphi_dump)) + 0.5) % 1.0 - 0.5          # the dump stores (phase - its dphi)
    if np.max(np.abs(dd)) > 1e-9:
        raise SystemExit("dump observation phases do not match the CSV")
    models = {"SPICE " + Path(a.dump).stem: {b: (D[f"lc_{b}_model_phase"], D[f"lc_{b}_model_mag"]) for b in "by"}}
    for p in a.phoebe:
        Q = np.load(p, allow_pickle=True)
        if not np.allclose(Q["model_phase"], D["lc_b_model_phase"]):
            raise SystemExit(f"{p}: PHOEBE curve phases differ from the dump's model phases")
        models["PHOEBE " + Path(p).stem.replace("phoebe_", "")] = {b: (Q["model_phase"], Q[f"phoebe_{b}_curve"]) for b in "by"}
    grid = np.linspace(-0.0010, 0.0005, 301)
    res = {}
    for nm, cv in models.items():
        if a.dphi == "fit":
            vals = [score(cv, ph, obs, x, a, night, ecl)[3] for x in grid]; i = int(np.argmin(vals)); dphi = float(grid[i])
        else:
            dphi = float(a.dphi)
        c, n, out, _ = score(cv, ph, obs, dphi, a, night, ecl)
        c0, _, _, _ = score(cv, ph, obs, 0.0, a, night, ecl)
        res[nm] = dict(dphi=dphi, dphi_days=dphi * K.PERIOD_DAYS, chi2_N=c, n=n, chi2_N_dphi0=c0, bands=out)
        print(f"{nm:45s} nightly {a.nightly:14s} dphi {dphi:+.5f} ({24 * dphi * K.PERIOD_DAYS:+.2f} h): chi2/N {c:.3f} (at dphi 0: {c0:.3f}), "
              + "; ".join(f"{b} rms {v['rms']:.2f} [eclipse {v['rms_eclipse']:.2f}, outside {v['rms_outside']:.2f}] clipped {v['clipped']}" for b, v in out.items()))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
