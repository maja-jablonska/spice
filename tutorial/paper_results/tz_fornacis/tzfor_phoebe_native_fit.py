"""PHOEBE-native fit of TZ For's temperatures from Clausen b, y.

Uses PHOEBE's own solver framework rather than a hand-rolled chi2 surface:
optimizer.nelder_mead on (teff@primary, teff@secondary, t0_supconj) with
dataset-scaled passband luminosities (zero points marginalised), followed by
sampler.emcee with uniform priors, so the posterior widths and the
Teff1-Teff2 correlation come from PHOEBE itself. Geometry fixed at the
literature values (same as every SPICE fit), ck2004 atmospheres.

CPU only (PHOEBE has no GPU path); uses PHOEBE's multiprocessing over walkers.
"""
import argparse, os, pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import tzfor_constants as K

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ntriangles", type=int, default=800)
    ap.add_argument("--n-phase", type=int, default=600)
    ap.add_argument("--abun", type=float, default=-0.23)
    ap.add_argument("--nm-maxiter", type=int, default=80)
    ap.add_argument("--nwalkers", type=int, default=12)
    ap.add_argument("--niters", type=int, default=150)
    ap.add_argument("--nprocs", type=int, default=8)
    ap.add_argument("--out", default=str(HERE / "tzfor_aemu_out" / "phoebe_native_fit.pkl"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.nm_maxiter, args.nwalkers, args.niters = 3, 6, 4
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import phoebe; phoebe.logger(clevel="ERROR")
    phoebe.multiprocessing_set_nprocs(args.nprocs)
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
        b.set_value_all("pblum_mode", dataset=ds, value="dataset-scaled")
    b.set_value_all("irrad_method", "horvat"); b.set_value_all("ltte", False)

    # ---- anchor t0_supconj: the SPICE fits needed a +0.1585 phase shift; check the sign with PHOEBE's own chi2 ----
    t0_base = b.get_value("t0_supconj@binary@component"); shift = 0.1585 * K.PERIOD_DAYS
    best = None
    for s in (-1.0, 0.0, 1.0):
        b.set_value("t0_supconj@binary@component", t0_base + s * shift)
        t = time.time(); b.run_compute(model="chk", overwrite=True); c = float(np.sum(b.calculate_chi2(model="chk")))
        print(f"t0_supconj = t0 {s:+.0f} x {shift:.3f} d: chi2/N {c / (2 * len(t_obs)):.3f} ({time.time() - t:.0f}s)", flush=True)
        if best is None or c < best[0]: best = (c, s)
    t0 = t0_base + best[1] * shift; b.set_value("t0_supconj@binary@component", t0)
    print(f"anchor: t0_supconj = {t0:.4f} (shift sign {best[1]:+.0f})", flush=True)

    # ---- Nelder-Mead on the temperatures and the ephemeris anchor ----
    b.add_solver("optimizer.nelder_mead", solver="nm", compute="phoebe01",
                 fit_parameters=["teff@primary@component", "teff@secondary@component", "t0_supconj@binary@component"],
                 maxiter=args.nm_maxiter, xatol=1.0, fatol=1e-3, progress_every_niters=5)
    t = time.time(); b.run_solver("nm", solution="nm_sol", overwrite=True)
    print(f"Nelder-Mead: {time.time() - t:.0f}s  message: {b.get_value('message@nm_sol')}", flush=True)
    b.adopt_solution("nm_sol")
    T1, T2, t0f = (b.get_value("teff@primary@component"), b.get_value("teff@secondary@component"), b.get_value("t0_supconj@binary@component"))
    b.run_compute(model="nmfit", overwrite=True); c = float(np.sum(b.calculate_chi2(model="nmfit")))
    print(f"NM solution: Teff1 {T1:.0f}  Teff2 {T2:.0f}  t0 {t0f:.4f}  chi2/N {c / (2 * len(t_obs)):.3f}", flush=True)

    # ---- emcee: uniform priors, walkers initialised around the NM solution ----
    b.add_distribution("teff@primary@component", phoebe.uniform(4300., 5600.), distribution="pri")
    b.add_distribution("teff@secondary@component", phoebe.uniform(5600., 7400.), distribution="pri")
    b.add_distribution("t0_supconj@binary@component", phoebe.uniform(t0f - 0.3, t0f + 0.3), distribution="pri")
    b.add_distribution("teff@primary@component", phoebe.gaussian(T1, 40.), distribution="init")
    b.add_distribution("teff@secondary@component", phoebe.gaussian(T2, 60.), distribution="init")
    b.add_distribution("t0_supconj@binary@component", phoebe.gaussian(t0f, 0.003), distribution="init")
    b.add_solver("sampler.emcee", solver="mc", compute="phoebe01", init_from="init", priors="pri",
                 nwalkers=args.nwalkers, niters=args.niters, progress_every_niters=5)
    t = time.time(); b.run_solver("mc", solution="mc_sol", overwrite=True)
    print(f"emcee: {args.nwalkers} walkers x {args.niters} iterations in {time.time() - t:.0f}s", flush=True)
    samples = np.asarray(b.get_value("samples@mc_sol")); lnp = np.asarray(b.get_value("lnprobabilities@mc_sol"))
    twigs = list(b.get_value("fitted_twigs@mc_sol")); burn = int(b.get_value("burnin@mc_sol")) if "burnin" in [p.qualifier for p in b.get_solution("mc_sol").to_list()] else args.niters // 3
    print("fitted:", twigs, "samples shape", samples.shape, "burn-in", burn, flush=True)
    post = samples[burn:].reshape(-1, samples.shape[-1])
    mean, std = post.mean(0), post.std(0, ddof=1); corr = np.corrcoef(post.T)
    for i, tw in enumerate(twigs):
        print(f"  {tw}: {mean[i]:.4f} +- {std[i]:.4f}")
    print(f"  corr(Teff1, Teff2) = {corr[0, 1]:+.3f}")
    print(f"  acceptance fraction: {np.mean(np.asarray(b.get_value('acceptance_fractions@mc_sol'))):.2f}" if "acceptance_fractions" in [p.qualifier for p in b.get_solution("mc_sol").to_list()] else "")
    pickle.dump(dict(nm=(T1, T2, t0f, c / (2 * len(t_obs))), samples=samples, lnp=lnp, twigs=twigs, burn=burn, mean=mean, std=std, corr=corr, args=vars(args)),
                open(args.out, "wb"), protocol=4)
    print("saved", args.out)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("fork", force=True)
    main()
