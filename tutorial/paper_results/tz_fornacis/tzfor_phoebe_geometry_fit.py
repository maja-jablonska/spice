"""Does PHOEBE, with its Roche geometry, agree with SPICE's free-geometry radii?

The SPICE joint fit (``tzfor_fullrange_fit.py --free-geometry``) models both stars as
spheres and lets R1, R2 and the inclination float; it lands at R1 8.118, R2 4.084 R☉
against Andersen's 8.28 / 3.94, driven by the spectroscopic light ratio. Two objections
have to be answered separately:

1. **Is the sphere approximation to blame?** The projected area sets the light ratio, and
   for TZ For's PHOEBE Roche meshes it differs from a sphere of the equivalent radius by
   +0.02 % (primary) and -0.13 % (secondary), against the +11.8 % change in the area ratio that
   the SPICE fit made. The surfaces themselves depart from spheres by 0.08 % and 0.75 % in radius
   (the secondary's is rotational flattening from its unsynchronised 38 km/s spin, not tides).
   This script re-measures both from the bundle it builds.
2. **Does the photometry alone prefer either geometry?** This script evaluates PHOEBE's own
   chi2 (Roche meshes, ck2004 atmospheres, irradiation, gravity darkening) at the literature
   geometry, at the SPICE free-geometry solution and at the line-strength-corrected one, then
   runs PHOEBE's Nelder-Mead with requiv1, requiv2, incl and Teff2 free to find its own best.

CPU only (PHOEBE has no GPU path). Writes a pickle with every chi2 and the fitted values.
"""
import argparse, math, os, pickle, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import tzfor_constants as K

# (label, R1, R2, incl, Teff1, Teff2) -- the SPICE solutions this compares against
SPICE_SOLUTIONS = [
    ("literature (Andersen 1991 / Gallenne 2016)", K.PRIMARY_RADIUS, K.SECONDARY_RADIUS, K.INCL_DEG, 4896.0, 6396.0),
    ("SPICE free geometry, 5% mask", 8.118, 4.084, 85.732, 4904.0, 6275.0),
    ("SPICE free geometry, delta map", 8.087, 4.045, 85.770, 4879.0, 6211.0),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ntriangles", type=int, default=800); ap.add_argument("--n-phase", type=int, default=400)
    ap.add_argument("--abun", type=float, default=-0.175); ap.add_argument("--nprocs", type=int, default=8)
    ap.add_argument("--nm-maxiter", type=int, default=60); ap.add_argument("--nm-teff-maxiter", type=int, default=40)
    ap.add_argument("--free-teff1", action="store_true", help="also let Teff1 float in the Nelder-Mead")
    ap.add_argument("--skip-fit", action="store_true", help="only the chi2 comparison at the fixed geometries")
    ap.add_argument("--scan-ratio", action="store_true", help="map PHOEBE's photometric chi2 against R2/R1 at fixed R1+R2, re-optimising Teff2 at each ratio")
    ap.add_argument("--ratio-range", type=float, nargs=3, default=(0.450, 0.530, 17), help="min max n for the R2/R1 scan")
    ap.add_argument("--out", default=str(HERE / "tzfor_aemu_out" / "phoebe_geometry_fit.pkl"))
    args = ap.parse_args()
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import phoebe; phoebe.logger(clevel="ERROR")
    try: phoebe.progressbars_off()
    except Exception: pass
    try: phoebe.multiprocessing_set_nprocs(args.nprocs)
    except Exception: pass
    import tzfor_phoebe_aemu_photometry as M

    SIG = {"b": 0.0041, "y": 0.0035}
    df = pd.read_csv(HERE / "tzfor_lightcurve.csv"); t_obs = df["HJD"].values + 2450000.0
    b = M.build_bundle(args.ntriangles, "roche")
    for comp in ("primary", "secondary"):
        try: b.set_value(f"abun@{comp}@component", args.abun)
        except Exception as e: print("abun:", e)  # noqa: BLE001
    for band, ds, key in (("Stromgren:b", "lcb", "b"), ("Stromgren:y", "lcy", "y")):
        m = df[key].values; flux = 10 ** (-0.4 * (m - np.median(m))); sig = flux * SIG[key] * np.log(10) / 2.5
        b.add_dataset("lc", times=t_obs, fluxes=flux, sigmas=sig, passband=band, dataset=ds)
        b.flip_constraint(f"compute_phases@{ds}", solve_for="compute_times")
        b.set_value("compute_phases", dataset=ds, context="dataset", value=np.linspace(0.0, 1.0, args.n_phase, endpoint=False))
        b.set_value_all("pblum_mode", dataset=ds, value="dataset-scaled")   # only the eclipse shapes matter, not the zero point
    b.set_value_all("irrad_method", "horvat"); b.set_value_all("ltte", False)
    N = 2 * len(t_obs)
    print(f"{len(t_obs)} epochs x 2 bands = {N} photometric points", flush=True)

    def chi2_at(R1, R2, incl, T1, T2, tag):
        b.set_value("requiv@primary@component", R1); b.set_value("requiv@secondary@component", R2)
        b.set_value_all("incl@binary", incl)
        b.set_value("teff@primary@component", T1); b.set_value("teff@secondary@component", T2)
        t = time.time(); b.run_compute(model="chk", overwrite=True)
        c = float(np.sum(b.calculate_chi2(model="chk")))
        print(f"  {tag:46s} R1 {R1:.3f} R2 {R2:.3f} i {incl:.3f} T2 {T2:.0f} -> chi2/N {c / N:.4f}  ({time.time() - t:.0f}s)", flush=True)
        return c

    # ---- anchor the ephemeris: the SPICE fits all needed a 0.1585 phase shift, and a chi2 computed with the
    # eclipses at the wrong phase compares nothing at all (chi2/N ~ 200 instead of ~1.5) ----
    b.set_value("requiv@primary@component", K.PRIMARY_RADIUS); b.set_value("requiv@secondary@component", K.SECONDARY_RADIUS)
    b.set_value_all("incl@binary", K.INCL_DEG)
    b.set_value("teff@primary@component", 4896.0); b.set_value("teff@secondary@component", 6396.0)
    t0_base = b.get_value("t0_supconj@binary@component"); shift = 0.1585 * K.PERIOD_DAYS; best_t0 = None
    for sgn in (-1.0, 0.0, 1.0):
        b.set_value("t0_supconj@binary@component", t0_base + sgn * shift)
        b.run_compute(model="chk", overwrite=True); c = float(np.sum(b.calculate_chi2(model="chk")))
        print(f"t0_supconj = t0 {sgn:+.0f} x {shift:.3f} d: chi2/N {c / N:.3f}", flush=True)
        if best_t0 is None or c < best_t0[0]: best_t0 = (c, sgn)
    b.set_value("t0_supconj@binary@component", t0_base + best_t0[1] * shift)
    print(f"anchor: shift sign {best_t0[1]:+.0f}, chi2/N {best_t0[0] / N:.3f}\n", flush=True)

    # ---- how distorted are the Roche meshes, and does it show in the projected area? ----
    b.add_dataset("mesh", compute_times=[K.T_P_HJD + 0.25 * K.PERIOD_DAYS], dataset="mesh01",
                  columns=["areas", "mus", "rs", "visibilities"], overwrite=True)
    b.run_compute(model="meshchk", overwrite=True)
    distortion = {}
    for comp, Req in (("primary", K.PRIMARY_RADIUS), ("secondary", K.SECONDARY_RADIUS)):
        rs = np.asarray(b.get_value("rs", component=comp, dataset="mesh01", model="meshchk"))     # already in R_sun
        ar = np.asarray(b.get_value("areas", component=comp, dataset="mesh01", model="meshchk"))
        mus = np.asarray(b.get_value("mus", component=comp, dataset="mesh01", model="meshchk"))
        vis = np.asarray(b.get_value("visibilities", component=comp, dataset="mesh01", model="meshchk"))
        proj = float(np.sum(ar * np.clip(mus, 0, None) * vis))
        distortion[comp] = dict(r_min=float(rs.min()), r_max=float(rs.max()), r_mean=float(rs.mean()),
                                pp_percent=float(100 * (rs.max() - rs.min()) / rs.mean()), proj_over_piR2=proj / (math.pi * Req ** 2))
        d = distortion[comp]
        print(f"Roche mesh {comp:9s}: r {d['r_min']:.4f}-{d['r_max']:.4f} R☉ (peak-to-peak {d['pp_percent']:.2f}% of the mean); "
              f"projected area / pi R_equiv^2 = {d['proj_over_piR2']:.5f}", flush=True)
    b.remove_dataset("mesh01")

    print("\nPHOEBE chi2 (Roche, ck2004, irradiation) at each geometry, as given:", flush=True)
    fixed = {}
    for label, R1, R2, incl, T1, T2 in SPICE_SOLUTIONS:
        fixed[label] = chi2_at(R1, R2, incl, T1, T2, label)

    # A geometry must not be judged at temperatures tuned to a different atmosphere table: give each one its
    # own Teff2 and ephemeris anchor before comparing, so the difference that remains is geometry alone.
    print("\nSame geometries after PHOEBE re-optimises Teff2 and t0_supconj for each:", flush=True)
    relaxed = {}
    t0_anchor = b.get_value("t0_supconj@binary@component")
    b.add_solver("optimizer.nelder_mead", solver="nmT", compute="phoebe01",
                 fit_parameters=["teff@secondary@component", "t0_supconj@binary@component"],
                 maxiter=args.nm_teff_maxiter, progress_every_niters=0)
    for label, R1, R2, incl, T1, T2 in SPICE_SOLUTIONS:
        b.set_value("requiv@primary@component", R1); b.set_value("requiv@secondary@component", R2)
        b.set_value_all("incl@binary", incl); b.set_value("teff@primary@component", T1)
        b.set_value("teff@secondary@component", T2); b.set_value("t0_supconj@binary@component", t0_anchor)
        t = time.time(); b.run_solver("nmT", solution="nmT_sol", overwrite=True); b.adopt_solution("nmT_sol")
        b.run_compute(model="relax", overwrite=True); c = float(np.sum(b.calculate_chi2(model="relax")))
        relaxed[label] = dict(chi2=c, T2=float(b.get_value("teff@secondary@component")), t0=float(b.get_value("t0_supconj@binary@component")))
        print(f"  {label:46s} -> Teff2 {relaxed[label]['T2']:.0f}, chi2/N {c / N:.4f}  ({time.time() - t:.0f}s)", flush=True)
    b.set_value("t0_supconj@binary@component", t0_anchor)

    base = relaxed[SPICE_SOLUTIONS[0][0]]["chi2"]
    print("\nGeometry comparison (each at its own best Teff2 and ephemeris):", flush=True)
    for label in (s_[0] for s_ in SPICE_SOLUTIONS):
        c = relaxed[label]["chi2"]
        print(f"  {label:46s} Delta chi2 = {c - base:+9.1f} ({(c - base) / N:+.4f} per point, "
              f"{(c - base) / max(abs(base), 1) * 100:+.2f}% of chi2)", flush=True)

    result = dict(distortion=distortion, chi2_fixed=fixed, chi2_relaxed=relaxed, N=N, solutions=SPICE_SOLUTIONS, args=vars(args))
    pickle.dump(result, open(args.out, "wb"), protocol=4)

    # ---- the photometric constraint on the radius RATIO, which is what the spectra moved ----
    # The eclipse duration fixes R1 + R2, so the ratio is the free direction; Teff2 is re-optimised at
    # every ratio because a smaller secondary is compensated by a hotter one at fixed eclipse depth.
    if args.scan_ratio:
        R_sum = K.PRIMARY_RADIUS + K.SECONDARY_RADIUS
        lo, hi, nr = args.ratio_range; ratios = np.linspace(lo, hi, int(nr))
        b.add_solver("optimizer.nelder_mead", solver="nmT2", compute="phoebe01",
                     fit_parameters=["teff@secondary@component"], maxiter=args.nm_teff_maxiter, progress_every_niters=0)
        b.set_value("t0_supconj@binary@component", t0_anchor); b.set_value("teff@primary@component", 4896.0)
        scan = []
        print(f"\nPHOEBE photometric chi2 vs R2/R1 at fixed R1+R2 = {R_sum:.3f} R_sun:", flush=True)
        for q in ratios:
            R1 = R_sum / (1.0 + q); R2 = R_sum - R1
            b.set_value("requiv@primary@component", R1); b.set_value("requiv@secondary@component", R2)
            b.set_value_all("incl@binary", K.INCL_DEG); b.set_value("teff@secondary@component", 6396.0)
            t = time.time(); b.run_solver("nmT2", solution="nmT2_sol", overwrite=True); b.adopt_solution("nmT2_sol")
            b.run_compute(model="scanm", overwrite=True); c = float(np.sum(b.calculate_chi2(model="scanm")))
            T2 = float(b.get_value("teff@secondary@component"))
            scan.append((float(q), R1, R2, T2, c))
            print(f"  R2/R1 {q:.4f}  (R1 {R1:.3f}, R2 {R2:.3f})  Teff2 {T2:.0f}  chi2/N {c / N:.4f}  ({time.time() - t:.0f}s)", flush=True)
            result["ratio_scan"] = np.array(scan); pickle.dump(result, open(args.out, "wb"), protocol=4)
        A = np.array(scan); k = int(np.argmin(A[:, 4]))
        print(f"  minimum at R2/R1 = {A[k, 0]:.4f} (literature {K.SECONDARY_RADIUS / K.PRIMARY_RADIUS:.4f}, SPICE free geometry {4.084 / 8.118:.4f})", flush=True)

    if args.skip_fit:
        print("saved", args.out); return

    # ---- PHOEBE's own best geometry ----
    b.set_value("requiv@primary@component", K.PRIMARY_RADIUS); b.set_value("requiv@secondary@component", K.SECONDARY_RADIUS)
    b.set_value_all("incl@binary", K.INCL_DEG); b.set_value("teff@primary@component", 4896.0); b.set_value("teff@secondary@component", 6396.0)
    fit_params = ["requiv@primary@component", "requiv@secondary@component", "incl@binary@component", "teff@secondary@component"]
    if args.free_teff1: fit_params.append("teff@primary@component")
    b.add_solver("optimizer.nelder_mead", solver="nm", compute="phoebe01", fit_parameters=fit_params,
                 maxiter=args.nm_maxiter, progress_every_niters=0)
    t = time.time(); b.run_solver("nm", solution="nm_sol", overwrite=True)
    print(f"\nNelder-Mead ({len(fit_params)} free): {time.time() - t:.0f}s  message: {b.get_value('message@nm_sol')}", flush=True)
    b.adopt_solution("nm_sol")
    best = dict(R1=b.get_value("requiv@primary@component"), R2=b.get_value("requiv@secondary@component"),
                incl=b.get_value("incl@binary@component"), T1=b.get_value("teff@primary@component"), T2=b.get_value("teff@secondary@component"))
    b.run_compute(model="nmfit", overwrite=True); best["chi2"] = float(np.sum(b.calculate_chi2(model="nmfit")))
    print(f"PHOEBE best geometry: R1 {best['R1']:.3f}  R2 {best['R2']:.3f}  i {best['incl']:.3f}  Teff1 {best['T1']:.0f}  Teff2 {best['T2']:.0f}  "
          f"chi2/N {best['chi2'] / N:.4f}  (literature {base / N:.4f})", flush=True)
    print(f"  radius ratio: PHOEBE {best['R2'] / best['R1']:.4f} | literature {K.SECONDARY_RADIUS / K.PRIMARY_RADIUS:.4f} | SPICE free geometry {4.084 / 8.118:.4f}", flush=True)
    result["phoebe_best"] = best
    pickle.dump(result, open(args.out, "wb"), protocol=4); print("saved", args.out)


if __name__ == "__main__":
    main()
