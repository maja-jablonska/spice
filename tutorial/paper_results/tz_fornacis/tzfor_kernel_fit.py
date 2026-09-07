"""TZ For joint photometry + spectroscopy fit through SynthesisKernel, with gravity
darkening free.

Successor of tzfor_joint_fit.py (sparse operators, two windows). The kernel
path evaluates the emulator once per (mu, log g) node per star and reuses it
for every epoch, so all 13 HARPS line windows fit in memory together, and the
per-element temperatures follow Teff = Teff_ref (g/g_ref)^beta with beta free.

Parameters: theta = [Teff1_ref, Teff2_ref, [Fe/H], beta1, beta2, dphi].
Geometry is the PHOEBE mesh, fixed. Loss = chi2_phot + chi2_spec as in
tzfor_joint_fit.py (Clausen scatter per band; HARPS pixel noise per epoch and
window plus --spec-floor; closed-form linear continuum correction per epoch and
window; instrumental R = 115000 and fixed macroturbulence).

Both the starting point and the best fit are checked against exact
per-element synthesis (simulate_observed_flux with the same gravity law applied
element by element). --jackknife repeats the fit leaving one window out at a
time, warm-started from the all-window optimum.
"""
import argparse
import math
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize

import tzfor_grad_inference as GI
import tzfor_constants as K
from spice.spectrum.synthesis_kernel import (build_synthesis_kernel, gravity_darkened_rows,
                                             kernel_flux_multi)

C_KMS = 299792.458
R_HARPS = 115000.0


def gauss_kernel_pix(sigma_pix):
    half = int(math.ceil(4 * sigma_pix))
    x = np.arange(-half, half + 1)
    k = np.exp(-0.5 * (x / sigma_pix) ** 2)
    return jnp.asarray(k / k.sum()), half


def broadener(dlog, vmacro_kms):
    """Gaussian in velocity = Gaussian in log-lambda pixels; edge-padded, no wrap."""
    fwhm = math.hypot(C_KMS / R_HARPS, vmacro_kms)
    sigma_pix = fwhm / (C_KMS * math.log(10.0) * dlog) / 2.3548
    kern, half = gauss_kernel_pix(sigma_pix)

    def apply(y):                                     # y (..., n)
        yp = jnp.pad(y, [(0, 0)] * (y.ndim - 1) + [(half, half)], mode="edge")
        return jnp.apply_along_axis(lambda r: jnp.convolve(r, kern, mode="valid"), -1, yp)
    return apply


def load_photometry(csv, bands=("b", "y")):
    import pandas as pd
    df = pd.read_csv(csv)
    hjd = df["HJD"].values + 2450000.0
    ph = ((hjd - K.T_P_HJD) % K.PERIOD_DAYS) / K.PERIOD_DAYS
    return jnp.asarray(ph), {b: jnp.asarray(df[b].values) for b in bands}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lc-meshes", default=str(HERE / "tzfor_aemu_out" / "tzfor_aemu_roche_n1200_lc6650_meshes.pkl"))
    ap.add_argument("--spec-meshes", default=str(HERE / "tzfor_aemu_out" / "tzfor_aemu_roche_n1500_spectra_meshes.pkl"))
    ap.add_argument("--harps", default=str(HERE / "tzfor_aemu_out" / "harps_windows_log.npz"))
    ap.add_argument("--photometry", default=str(HERE / "tzfor_lightcurve.csv"))
    ap.add_argument("--out", default=str(HERE / "tzfor_aemu_out" / "kernel_fit_result.pkl"))
    ap.add_argument("--n-mu", type=int, default=32)
    ap.add_argument("--n-g", type=int, default=3, help="log g nodes per star")
    ap.add_argument("--oversample", type=int, default=2)
    ap.add_argument("--windows", type=float, nargs="+", default=None, help="window START wavelengths to keep")
    ap.add_argument("--drop-windows", type=float, nargs="+", default=[5560.0], help="window START wavelengths to drop")
    ap.add_argument("--spec-floor", type=float, default=0.005)
    ap.add_argument("--spec-weight", type=float, default=1.0)
    ap.add_argument("--vmacro", type=float, nargs=2, default=(5.0, 6.0))
    ap.add_argument("--dv-sys", type=float, default=0.25, help="systemic velocity offset measured by cross-correlation [km/s]")
    ap.add_argument("--fix-beta", action="store_true")
    ap.add_argument("--maxiter", type=int, default=200)
    ap.add_argument("--jackknife", action="store_true")
    ap.add_argument("--surrogate-tol", type=float, default=0.01)
    ap.add_argument("--n-epochs", type=int, default=None, help="smoke tests: first N spectral epochs")
    ap.add_argument("--n-lc-epochs", type=int, default=None, help="smoke tests: first N light-curve epochs")
    ap.add_argument("--skip-validation", action="store_true")
    ap.add_argument("--chunk", type=int, default=32768)
    ap.add_argument("--n-phot-wl", type=int, default=16000, help="wavelength samples for the passband integrals (16000 converges to 0.05 mmag)")
    args = ap.parse_args()
    print("devices:", jax.devices(), flush=True)

    emu = GI.make_emulator()
    LC = pickle.load(open(args.lc_meshes, "rb"))
    SP = pickle.load(open(args.spec_meshes, "rb"))
    H = np.load(args.harps, allow_pickle=True); H = {k: H[k] for k in H.files}
    names = LC["parameter_names"]; iT, iG, iF = names.index("marcs_teff"), names.index("marcs_logg"), names.index("feh")
    assert SP["parameter_names"] == names
    assert np.allclose(H["times"], SP["times"])
    if args.n_epochs:
        SP["models"] = SP["models"][:args.n_epochs]; SP["times"] = SP["times"][:args.n_epochs]
        for k in list(H):
            if k.startswith("obs_"): H[k] = H[k][:args.n_epochs]
        H["sigma_win"] = H["sigma_win"][:args.n_epochs]
    if args.n_lc_epochs:
        LC["models"] = LC["models"][:args.n_lc_epochs]; LC["times"] = LC["times"][:args.n_lc_epochs]
    n_ep, n_lc = len(SP["models"]), len(LC["models"])

    # ---- windows ----
    all_windows = [tuple(w) for w in H["windows"]]
    keep = [i for i, w in enumerate(all_windows)
            if (args.windows is None or any(abs(w[0] - s) < 1 for s in args.windows))
            and not any(abs(w[0] - s) < 1 for s in args.drop_windows)]
    print("windows:", [all_windows[i] for i in keep], flush=True)
    dlog = float(H["dlog"])
    shift_sys = math.log10(1.0 + args.dv_sys / C_KMS)

    # ---- surface coordinate (log g) nodes and reference values, per star ----
    def star_geometry(models, s):
        ms = [pair[s] for pair in models]
        m0 = ms[0]
        vis = np.asarray(m0.mus) > 0; a = np.asarray(m0.visible_cast_areas)[vis]
        p = np.asarray(m0.parameters)[vis]
        logg_ref = float(np.average(p[:, iG], weights=a))
        allg = np.concatenate([np.asarray(m.parameters)[np.asarray(m.mus) > 0, iG] for m in ms])
        gnodes = np.linspace(allg.min() - 0.005, allg.max() + 0.005, args.n_g)
        base = jnp.asarray(np.average(p, axis=0, weights=a))
        return base, logg_ref, jnp.asarray(gnodes)
    base1, gref1, gnodes1 = star_geometry(SP["models"], 0)
    base2, gref2, gnodes2 = star_geometry(SP["models"], 1)
    T1_0, T2_0, feh_0 = float(base1[iT]), float(base2[iT]), float(base1[iF])
    print(f"start: Teff1 {T1_0:.1f} (logg_ref {gref1:.3f}, nodes {np.asarray(gnodes1).round(3)})  "
          f"Teff2 {T2_0:.1f} (logg_ref {gref2:.3f}, nodes {np.asarray(gnodes2).round(3)})  [Fe/H] {feh_0:+.2f}", flush=True)

    def rows(theta):
        b1 = base1.at[iF].set(theta[2]); b2 = base2.at[iF].set(theta[2])
        r1 = gravity_darkened_rows(b1, iT, iG, gnodes1, theta[0], gref1, theta[3])
        r2 = gravity_darkened_rows(b2, iT, iG, gnodes2, theta[1], gref2, theta[4])
        return r1, r2

    # ---- kernels: spectroscopy (per window, per epoch, per star) ----
    t = time.time()
    spec_k = {}
    for wi in keep:
        lw_model = jnp.asarray(H[f"logwl_{wi}"] - shift_sys)      # model grid shifted by the systemic velocity
        k1 = [build_synthesis_kernel(m1, lw_model, args.n_mu, args.oversample,
                                     element_coordinate=jnp.asarray(m1.parameters)[:, iG], coordinate_nodes=gnodes1)
              for m1, _ in SP["models"]]
        k2 = [build_synthesis_kernel(m2, lw_model, args.n_mu, args.oversample,
                                     element_coordinate=jnp.asarray(m2.parameters)[:, iG], coordinate_nodes=gnodes2)
              for _, m2 in SP["models"]]
        spec_k[wi] = (k1, k2)
    # ---- kernels: photometry (per light-curve epoch, per star) ----
    # The aemu spectrum is line-rich at R = 115000: passband integrals sampled
    # every ~10 A alias lines into 3-5 mmag epoch-to-epoch errors (Clausen's
    # noise is 4 mmag). 16000 points (0.1 A) converge b and y to 0.05 mmag.
    lw_ph = jnp.linspace(math.log10(4300.0), math.log10(5900.0), args.n_phot_wl)
    ph_k1 = [build_synthesis_kernel(m1, lw_ph, args.n_mu, 1, element_coordinate=jnp.asarray(m1.parameters)[:, iG], coordinate_nodes=gnodes1)
             for m1, _ in LC["models"]]
    ph_k2 = [build_synthesis_kernel(m2, lw_ph, args.n_mu, 1, element_coordinate=jnp.asarray(m2.parameters)[:, iG], coordinate_nodes=gnodes2)
             for _, m2 in LC["models"]]
    jax.block_until_ready(ph_k2[-1].kernels)
    print(f"built {2 * n_ep * len(keep)} spectroscopic + {2 * n_lc} photometric kernels in {time.time() - t:.0f}s", flush=True)

    # ---- photometry data and passbands ----
    ph_o, mag_o = load_photometry(args.photometry)
    wl_ph = 10.0 ** lw_ph
    resp = {}
    for n in ("Stromgren:b", "Stromgren:y"):
        w, f = LC["passbands"][n]
        resp[n.split(":")[1]] = jnp.interp(wl_ph, jnp.asarray(w), jnp.asarray(f), left=0., right=0.)
    sig_ph = {"b": 0.0041, "y": 0.0035}
    N_ph = sum(len(v) for v in mag_o.values())
    P, T_P = K.PERIOD_DAYS, K.T_P_HJD
    model_phase = jnp.asarray(((np.asarray(LC["times"]) - T_P) % P) / P)
    order = jnp.argsort(model_phase); mph = model_phase[order]

    def chi2_phot(theta):
        r1, r2 = rows(theta)
        flux = kernel_flux_multi(emu.intensity, ph_k1, r1)[..., 0] + kernel_flux_multi(emu.intensity, ph_k2, r2)[..., 0]
        tot = 0.0
        for b, r in resp.items():
            mags = jax.vmap(lambda f: GI.passband_mag(f, wl_ph, r))(flux)[order]
            mags = mags - jnp.median(mags)
            mi = jnp.interp((ph_o - theta[5]) % 1.0, mph, mags, period=1.0)
            res = mag_o[b] - mi; res = res - jnp.median(res)
            tot = tot + jnp.sum(res ** 2) / sig_ph[b] ** 2
        return tot

    # ---- spectroscopy ----
    br1, br2 = broadener(dlog, args.vmacro[0]), broadener(dlog, args.vmacro[1])
    obs, good, sig, xw = {}, {}, {}, {}
    N_sp = 0
    for wi in keep:
        o = np.array(H[f"obs_{wi}"], float); g = np.isfinite(o)
        obs[wi] = jnp.asarray(np.where(g, o, 1.0)); good[wi] = jnp.asarray(g)
        sig[wi] = jnp.asarray(np.hypot(H["sigma_win"][:, wi], args.spec_floor))[:, None]
        xw[wi] = jnp.linspace(-1.0, 1.0, o.shape[1]); N_sp += int(g.sum())

    def window_model(wi, r1, r2):
        k1, k2 = spec_k[wi]
        s1 = kernel_flux_multi(emu.intensity, k1, r1, wavelength_chunk_size=args.chunk)   # (n_ep, n, 2)
        s2 = kernel_flux_multi(emu.intensity, k2, r2, wavelength_chunk_size=args.chunk)
        return (br1(s1[..., 0]) + br2(s2[..., 0])) / (br1(s1[..., 1]) + br2(s2[..., 1]))

    def continuum_fix(model, ob, gd, x):
        w = jnp.where(gd, 1.0, 0.0)
        A = jnp.stack([model, model * x], 1) * w[:, None]
        coef = jnp.linalg.solve(A.T @ A + 1e-12 * jnp.eye(2), A.T @ (ob * w))
        return model * (coef[0] + coef[1] * x)

    def chi2_spec(theta, active):
        r1, r2 = rows(theta)
        tot = 0.0
        for wi in keep:
            mod = window_model(wi, r1, r2)
            mod = jax.vmap(lambda m, o, g: continuum_fix(m, o, g, xw[wi]))(mod, obs[wi], good[wi])
            res = jnp.where(good[wi], (obs[wi] - mod) / sig[wi], 0.0)
            tot = tot + active[wi] * jnp.sum(res ** 2)
        return tot

    theta0 = jnp.array([T1_0, T2_0, feh_0, 0.08, 0.08, 0.158])
    S = jnp.array([100., 100., 0.1, 0.1, 0.1, 0.001])

    def total(x, active):
        th = theta0 + x * S
        c_ph = chi2_phot(th)
        c_sp = chi2_spec(th, active)
        return c_ph + args.spec_weight * c_sp, (c_ph, c_sp)
    vg = jax.jit(jax.value_and_grad(total, has_aux=True))

    # ---- exact-synthesis self-check: epoch 0, first kept window, per-element gravity law ----
    from spice.spectrum.spectrum import simulate_observed_flux
    wi0 = keep[0]
    lw0 = jnp.asarray(H[f"logwl_{wi0}"] - shift_sys)

    def exact_window(theta, e):
        m1, m2 = SP["models"][e]
        def one(mm, base, gref, T, beta, vm_br):
            p = jnp.asarray(mm.parameters); logg = p[:, iG]
            teff = T * jnp.power(10.0, beta * (logg - gref))
            p = p.at[:, iT].set(teff).at[:, iF].set(theta[2])
            return simulate_observed_flux(emu.intensity, mm._replace(parameters=p), lw0, chunk_size=256)
        s1 = one(m1, base1, gref1, theta[0], theta[3], br1); s2 = one(m2, base2, gref2, theta[1], theta[4], br2)
        return (br1(s1[:, 0]) + br2(s2[:, 0])) / (br1(s1[:, 1]) + br2(s2[:, 1]))

    def check(theta, label, epochs=(0,)):
        r1, r2 = rows(theta)
        surr = np.asarray(jax.jit(lambda a, b: window_model(wi0, a, b))(r1, r2))
        worst = 0.0
        for e in epochs:
            ex = np.asarray(jax.jit(lambda th: exact_window(th, e))(theta))
            d = np.abs(surr[e] - ex); worst = max(worst, d.max())
            print(f"{label}: epoch {e:2d} window {all_windows[wi0][0]:.0f}: kernel vs exact max {d.max():.5f} rms {np.sqrt((d ** 2).mean()):.6f}", flush=True)
        return worst
    if not args.skip_validation:
        t = time.time(); worst = check(theta0, "start"); print(f"  (exact check took {time.time() - t:.0f}s)", flush=True)
        if worst > args.surrogate_tol:
            print("ABORT: kernel model outside tolerance"); sys.exit(2)

    # ---- optimisation ----
    active_all = {wi: 1.0 for wi in keep}
    t = time.time(); (v0, aux0), g0 = vg(jnp.zeros(6), active_all); jax.block_until_ready(g0)
    print(f"first value+grad (compile) {time.time() - t:.0f}s", flush=True)
    t = time.time(); jax.block_until_ready(vg(jnp.zeros(6), active_all)[1]); print(f"warm value+grad {time.time() - t:.2f}s", flush=True)
    print(f"start: chi2_phot/N {float(aux0[0]) / N_ph:.3f}  chi2_spec/N {float(aux0[1]) / N_sp:.3f}", flush=True)

    lo = [(3800 - T1_0) / 100, (3800 - T2_0) / 100, (-1.5 - feh_0) / 0.1, -0.08 / 0.1, -0.08 / 0.1, (-0.05 - 0.158) / 0.001]
    hi = [(6900 - T1_0) / 100, (6900 - T2_0) / 100, (0.5 - feh_0) / 0.1, 0.22 / 0.1, 0.22 / 0.1, (0.35 - 0.158) / 0.001]
    if args.fix_beta:
        lo[3] = hi[3] = 0.0; lo[4] = hi[4] = 0.0

    def fit(x0, active, maxiter, label):
        hist = []
        def f(x):
            (v, _), g = vg(jnp.asarray(x), active); hist.append(float(v)); return float(v), np.asarray(g, dtype=float)
        t = time.time()
        r = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi)), options=dict(maxiter=maxiter))
        (_, aux), _ = vg(jnp.asarray(r.x), active)
        th = np.asarray(theta0) + np.asarray(r.x) * np.asarray(S)
        print(f"{label}: {r.nit} it, {len(hist)} eval, {time.time() - t:.0f}s, {r.message[:40]} | chi2_phot/N {float(aux[0]) / N_ph:.3f} "
              f"chi2_spec/N {float(aux[1]) / N_sp:.3f} | Teff1 {th[0]:.0f} Teff2 {th[1]:.0f} [Fe/H] {th[2]:+.3f} beta1 {th[3]:.3f} beta2 {th[4]:.3f} dphi {th[5]:.4f}", flush=True)
        return r, th, aux

    r, th, aux = fit(np.zeros(6), active_all, args.maxiter, "all windows")
    result = dict(theta=th, x=np.asarray(r.x), S=np.asarray(S), chi2_phot=float(aux[0]), chi2_spec=float(aux[1]),
                  N_ph=N_ph, N_sp=N_sp, windows=[all_windows[i] for i in keep], args=vars(args),
                  gnodes=(np.asarray(gnodes1), np.asarray(gnodes2)), logg_ref=(gref1, gref2))
    with open(args.out, "wb") as fh:
        pickle.dump(result, fh, protocol=4)
    try:
        Hm = np.asarray(jax.hessian(lambda x: total(x, active_all)[0])(jnp.asarray(r.x))); ev = np.linalg.eigvalsh(Hm)
        print(f"Hessian eigenvalues (scaled units): {np.array2string(ev, precision=3)}", flush=True)
        result["hessian"] = Hm
        if np.all(ev > 0):
            cov = 2.0 * np.linalg.inv(Hm); sg = np.sqrt(np.diag(cov)) * np.asarray(S)
            Cc = cov / np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))
            print(f"  formal 1-sigma: Teff1 {sg[0]:.0f} Teff2 {sg[1]:.0f} [Fe/H] {sg[2]:.3f} beta1 {sg[3]:.3f} beta2 {sg[4]:.3f} dphi {sg[5]:.5f}")
            print(f"  corr(Teff1,Teff2) {Cc[0, 1]:+.2f}  corr(Teff2,[Fe/H]) {Cc[1, 2]:+.2f}  corr(Teff2,beta2) {Cc[1, 4]:+.2f}  corr(Teff1,beta1) {Cc[0, 3]:+.2f}", flush=True)
            result.update(cov=cov, sigma=sg, corr=Cc)
    except Exception as e:  # noqa: BLE001
        print(f"Hessian failed: {type(e).__name__}: {str(e)[:200]}", flush=True)

    if not args.skip_validation:
        check(jnp.asarray(th), "best fit", epochs=(0, n_ep // 2, n_ep - 1))

    if args.jackknife and len(keep) > 1:
        print("\n=== leave-one-window-out jackknife (warm start from the all-window optimum) ===", flush=True)
        jk = {}
        for wi in keep:
            active = {w: (0.0 if w == wi else 1.0) for w in keep}
            rj, thj, auxj = fit(np.asarray(r.x), active, 60, f"drop {all_windows[wi][0]:.0f}-{all_windows[wi][1]:.0f}")
            jk[all_windows[wi]] = thj
        A = np.array(list(jk.values())); n = A.shape[0]
        jk_err = np.sqrt((n - 1) / n * np.sum((A - A.mean(0)) ** 2, axis=0))
        print(f"jackknife error: Teff1 {jk_err[0]:.0f} K  Teff2 {jk_err[1]:.0f} K  [Fe/H] {jk_err[2]:.3f}  beta1 {jk_err[3]:.3f}  beta2 {jk_err[4]:.3f}", flush=True)
        result["jackknife"] = jk; result["jackknife_err"] = jk_err
    with open(args.out, "wb") as fh:
        pickle.dump(result, fh, protocol=4)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
