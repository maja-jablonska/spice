"""PHOEBE's best geometry for TZ For: R1, R2 and the inclination fitted to Clausen (1991) b, y at the parameters of
Gallenne et al. (2016) (Teff 4930 / 6350 K, [Fe/H] +0.02 / -0.05, equatorial rotation v sin i / sin i, Roche, Lucy gravity
darkening, Horvat irradiation), with the scoring used for every TZ For light curve in the paper
(tz_fornacis/tzfor_phot_rescore.py): one photometric-ephemeris offset dphi profiled on a grid, no nightly zero points,
a truncated quadratic at 15 mmag (a point beyond it costs the clip level, so the fit cannot gain by dropping points).

The light curve is computed on a fixed phase grid (0.0005 in phase across both eclipses, 24 points outside) and
interpolated to the observations, as SPICE's are. Nelder-Mead from Gallenne's radii and inclination. The result (JSON)
is what tzfor_phoebe_aemu_photometry.py --mode export-lc / export-spectra turns into SPICE meshes (TZ_FOR_* env).

    python tzfor_phoebe_geom_fit.py --atm phoenix --photometry tzfor_lightcurve.csv --out geomfit_phoenix.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.environ.get("TZFOR_DIR", str(Path(__file__).resolve().parents[1] / "tz_fornacis")))
import tzfor_constants as K  # noqa: E402

SIG = {"b": 0.0041, "y": 0.0035}
HJD_OFFSET = 2440000.0
VSINI = (6.1, 45.7); TEFF = (4930.0, 6350.0); ABUN = (0.02, -0.05)


def phase_grid():
    e = np.arange(-0.013, 0.01301, 0.0005)
    return np.unique(np.r_[e % 1.0, (0.5 + e) % 1.0, np.linspace(0.03, 0.47, 12), np.linspace(0.53, 0.97, 12)])


def build(atm, irrad, n_mesh, R1, R2, incl):
    import phoebe
    from astropy import units as u
    phoebe.logger(clevel="ERROR")
    b = phoebe.default_binary()
    b.flip_constraint("mass@primary", solve_for="sma")
    b.set_value("period@binary@component", K.PERIOD_DAYS)
    b.set_value("q@binary@component", K.SECONDARY_MASS / K.PRIMARY_MASS)
    b.set_value("ecc@binary@component", K.ECC)
    b.set_value("mass@primary@component", K.PRIMARY_MASS)
    b.set_value_all("incl@binary", incl)
    b.set_value("per0@binary@component", K.PER0_DEG)
    b.set_value("long_an@binary@component", K.LONG_AN_DEG)
    b.set_value("t0_supconj@binary@component", K.T_P_HJD - K.PERIOD_DAYS / 2.0)   # T_P = deeper minimum
    b.set_value("distance", K.DISTANCE_PC * u.pc)
    for i, comp in enumerate(("primary", "secondary")):
        R = (R1, R2)[i]
        b.set_value(f"requiv@{comp}@component", R)
        b.set_value(f"teff@{comp}@component", TEFF[i]); b.set_value(f"abun@{comp}@component", ABUN[i])
        v_sync = 2 * np.pi * R * 6.957e5 / (K.PERIOD_DAYS * 86400.0)
        b.set_value(f"syncpar@{comp}@component", VSINI[i] / np.sin(np.deg2rad(incl)) / v_sync)
    b.set_value_all("distortion_method", "roche"); b.set_value_all("ntriangles", n_mesh)
    b.set_value_all("gravb_bol", 0.32); b.set_value_all("irrad_method", irrad); b.set_value_all("atm", atm)
    for band in ("b", "y"):
        b.add_dataset("lc", compute_times=K.T_P_HJD + phase_grid() * K.PERIOD_DAYS, passband=f"Stromgren:{band}", dataset=f"lc{band}")
        if atm != "blackbody":
            b.set_value_all("ld_mode", dataset=f"lc{band}", value="interp")
    return b


def curves(b):
    b.run_compute(model="phmodel", overwrite=True)
    ph = phase_grid()
    return {band: (ph, -2.5 * np.log10(np.asarray(b.get_value(f"fluxes@lc{band}@phmodel"), float))) for band in ("b", "y")}


def loss(cv, ph_obs, obs, clip=15.0):
    """min over dphi of the truncated chi2 / N (no nightly zero points) -> (value, dphi, chi2/N kept, rms b, rms y, n clipped)."""
    best = None
    for dphi in np.linspace(-0.0010, 0.0005, 151):
        tr = 0.0; c2 = 0.0; nk = 0; rms = []; ncl = 0
        for band in ("b", "y"):
            mp, mm = cv[band]
            r = obs[band] - np.interp((ph_obs - dphi) % 1.0, mp, mm, period=1.0); r = r - np.median(r)
            k = np.abs(r) <= clip / 1e3
            tr += np.sum(np.minimum((r / SIG[band]) ** 2, (clip / 1e3 / SIG[band]) ** 2)); c2 += np.sum((r[k] / SIG[band]) ** 2)
            nk += int(k.sum()); rms.append(1e3 * r[k].std()); ncl += int((~k).sum())
        v = tr / (2 * ph_obs.size)
        if best is None or v < best[0]:
            best = (v, float(dphi), c2 / nk, rms[0], rms[1], ncl)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--atm", default="phoenix", choices=["ck2004", "phoenix"]); ap.add_argument("--irrad", default="horvat")
    ap.add_argument("--n-mesh", type=int, default=3000); ap.add_argument("--photometry", required=True)
    ap.add_argument("--maxfev", type=int, default=160); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import pandas as pd
    df = pd.read_csv(a.photometry); hjd = df["HJD"].values.astype(float) + HJD_OFFSET
    ph_obs = ((hjd - K.T_P_HJD) % K.PERIOD_DAYS) / K.PERIOD_DAYS; obs = {band: df[band].values.astype(float) for band in ("b", "y")}
    x0 = np.array([K.PRIMARY_RADIUS, K.SECONDARY_RADIUS, K.INCL_DEG]); hist = []

    def f(x):
        t = time.time(); R1, R2, inc = x
        if not (6.0 < R1 < 11.0 and 2.5 < R2 < 6.0 and 80.0 < inc < 90.0):
            return 1e9
        v = loss(curves(build(a.atm, a.irrad, a.n_mesh, R1, R2, inc)), ph_obs, obs)
        hist.append(dict(R1=R1, R2=R2, incl=inc, loss=v[0], dphi=v[1], chi2_N=v[2], rms_b=v[3], rms_y=v[4], clipped=v[5]))
        print(f"  eval {len(hist):3d}: R1 {R1:.4f} R2 {R2:.4f} i {inc:.4f} | loss {v[0]:.5f} chi2/N {v[2]:.4f} dphi {v[1]:+.5f} "
              f"rms {v[3]:.2f}/{v[4]:.2f} clipped {v[5]} ({time.time() - t:.0f}s)", flush=True)
        json.dump(dict(atm=a.atm, irrad=a.irrad, n_mesh=a.n_mesh, history=hist), open(a.out + ".progress", "w"))
        return v[0]

    print(f"PHOEBE geometry fit, atm {a.atm}, irrad {a.irrad}, {a.n_mesh} triangles, {phase_grid().size} model phases; start "
          f"R1 {x0[0]} R2 {x0[1]} i {x0[2]}", flush=True)
    f0 = f(x0)
    simplex = np.array([x0, x0 + [0.10, 0, 0], x0 + [0, 0.06, 0], x0 + [0, 0, 0.15]])
    r = minimize(f, x0, method="Nelder-Mead", options=dict(initial_simplex=simplex, maxfev=a.maxfev, xatol=2e-3, fatol=1e-4))
    best = min(hist, key=lambda h: h["loss"])
    lit = hist[0]
    res = dict(atm=a.atm, irrad=a.irrad, n_mesh=a.n_mesh, best=best, literature=lit, n_eval=len(hist), message=str(r.message),
               vrot=[VSINI[i] / np.sin(np.deg2rad(best["incl"])) for i in (0, 1)], history=hist)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"BEST {a.atm}: R1 {best['R1']:.4f} R2 {best['R2']:.4f} i {best['incl']:.4f} dphi {best['dphi']:+.5f} chi2/N {best['chi2_N']:.4f} "
          f"rms {best['rms_b']:.2f}/{best['rms_y']:.2f} (Gallenne geometry: chi2/N {lit['chi2_N']:.4f}); {len(hist)} evaluations", flush=True)


if __name__ == "__main__":
    main()
