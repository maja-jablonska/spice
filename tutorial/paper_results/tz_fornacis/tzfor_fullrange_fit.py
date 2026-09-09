"""TZ For joint fit over the FULL HARPS range through SynthesisKernel (GPU).

One log-uniform grid 4450-6750 A at the HARPS pixel (harps_fullrange_log.npz;
Balmer cores, Na D, tellurics and the detector gap pre-masked), Clausen b, y
photometry, PHOEBE geometry fixed, gravity darkening fixed (beta unconstrained
on this system). Compared with tzfor_kernel_fit.py:

  * spline continuum per epoch (cubic B-splines, knots every --knot-spacing A)
    instead of a polynomial per 40 A window;
  * --free-vmacro: macroturbulence per star through the broadening kernel;
  * --free-vsini: a scale on the rotational part of each mesh's velocity field
    (los = v_orb + s * (los - v_orb)), kernels rebuilt inside the loss;
  * --delta: a per-pixel line-strength correction map of the PRIMARY in its
    rest frame, multiplying the emulator's line channel before the kernels so
    it Doppler-shifts with the star; Gaussian prior (--delta-sigma) plus a
    second-difference smoothness penalty; fitted by alternating with theta;
  * --derive-mask-out: after the fit, stack the residual in the primary's rest
    frame over --mask-epochs and write a per-epoch mask (threshold, dilation);
    --mask-in applies such a mask; --fit-epochs restricts the fit to odd/even
    epochs for the circularity test;
  * --jackknife-blocks N: leave-one-contiguous-block-out refits.
"""
import argparse, math, os, pickle, sys, time
from pathlib import Path
from typing import NamedTuple

os.environ.setdefault("JAX_ENABLE_X64", "1")
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, jax, jax.numpy as jnp
from scipy.optimize import minimize
from scipy.interpolate import BSpline
import tzfor_grad_inference as GI, tzfor_constants as K
from tzfor_kernel_fit import load_photometry, C_KMS, R_HARPS
from spice.spectrum.synthesis_kernel import build_synthesis_kernel, gravity_darkened_rows, kernel_flux_multi


class Geom(NamedTuple):
    """The three arrays build_synthesis_kernel reads from a mesh."""
    visible_cast_areas: jnp.ndarray
    mus: jnp.ndarray
    los_velocities: jnp.ndarray


def epoch_subset(n, which):
    idx = np.arange(n)
    return {"all": idx, "odd": idx[1::2], "even": idx[::2]}[which]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--harps", default=str(HERE / "tzfor_aemu_out" / "harps_fullrange_log.npz"))
    ap.add_argument("--spec-meshes", default=str(HERE / "tzfor_aemu_out" / "tzfor_aemu_roche_n1500_spectra_meshes.pkl"))
    ap.add_argument("--lc-meshes", default=str(HERE / "tzfor_aemu_out" / "tzfor_aemu_roche_n1200_lc6650_meshes.pkl"))
    ap.add_argument("--photometry", default=str(HERE / "tzfor_lightcurve.csv"))
    ap.add_argument("--out", default=str(HERE / "tzfor_aemu_out" / "fullrange_fit.pkl"))
    ap.add_argument("--result-in", default=None, help="warm start (theta, delta) from a previous result")
    ap.add_argument("--n-mu", type=int, default=16); ap.add_argument("--n-g", type=int, default=2); ap.add_argument("--oversample", type=int, default=2)
    ap.add_argument("--beta", type=float, default=0.079, help="gravity-darkening exponent on g, fixed (unconstrained on TZ For)")
    ap.add_argument("--vmacro0", type=float, nargs=2, default=(5.0, 6.0)); ap.add_argument("--free-vmacro", action="store_true")
    ap.add_argument("--free-vsini", action="store_true")
    ap.add_argument("--dv-sys", type=float, default=0.25)
    ap.add_argument("--knot-spacing", type=float, default=40.0, help="continuum spline knot spacing [A]")
    ap.add_argument("--spec-floor", type=float, default=0.005)
    ap.add_argument("--mask-in", default=None); ap.add_argument("--derive-mask-out", default=None)
    ap.add_argument("--mask-threshold", type=float, default=0.05); ap.add_argument("--mask-nsigma", type=float, default=4.0); ap.add_argument("--mask-dilate", type=int, default=2)
    ap.add_argument("--mask-epochs", default="all", choices=["all", "odd", "even"]); ap.add_argument("--fit-epochs", default="all", choices=["all", "odd", "even"])
    ap.add_argument("--delta", action="store_true"); ap.add_argument("--delta-sigma", type=float, default=0.03); ap.add_argument("--delta-smooth", type=float, default=1.0)
    ap.add_argument("--delta-alternations", type=int, default=3); ap.add_argument("--delta-maxiter", type=int, default=60)
    ap.add_argument("--delta-mode", default="joint", choices=["joint", "fit-only", "fixed"],
                    help="joint: alternate theta and delta (degenerate: delta can mimic the primary's line profiles); "
                         "fit-only: fit delta once at the warm-start theta on --fit-epochs and stop; fixed: use delta from --result-in unchanged")
    ap.add_argument("--delta-in", default=None, help="result pkl whose delta map is used (delta-mode fixed)")
    ap.add_argument("--jackknife-blocks", type=int, default=0)
    ap.add_argument("--noise-check", action="store_true", help="measure the float32 roundoff of the objective and compare AD with finite differences at the start point")
    ap.add_argument("--stall-retries", type=int, default=2, help="L-BFGS-B restarts from a perturbed point when a fit stalls in its first line search")
    ap.add_argument("--maxiter", type=int, default=150); ap.add_argument("--n-phot-wl", type=int, default=16000); ap.add_argument("--chunk", type=int, default=32768)
    ap.add_argument("--n-epochs", type=int, default=None); ap.add_argument("--n-lc-epochs", type=int, default=None); ap.add_argument("--skip-validation", action="store_true")
    args = ap.parse_args()
    print("devices:", jax.devices(), flush=True)

    emu = GI.make_emulator()
    SP = pickle.load(open(args.spec_meshes, "rb")); LC = pickle.load(open(args.lc_meshes, "rb"))
    H = np.load(args.harps, allow_pickle=True); H = {k: H[k] for k in H.files}
    names = SP["parameter_names"]; iT, iG, iF = names.index("marcs_teff"), names.index("marcs_logg"), names.index("feh")
    assert np.allclose(H["times"], SP["times"])
    if args.n_epochs:
        SP["models"] = SP["models"][:args.n_epochs]; SP["times"] = SP["times"][:args.n_epochs]; H["obs"] = H["obs"][:args.n_epochs]; H["sigma_blocks"] = H["sigma_blocks"][:args.n_epochs]
    if args.n_lc_epochs:
        LC["models"] = LC["models"][:args.n_lc_epochs]; LC["times"] = LC["times"][:args.n_lc_epochs]
    n_ep, n_lc = len(SP["models"]), len(LC["models"])
    lw_obs = np.asarray(H["logwl"], float); n = lw_obs.size; wl = 10.0 ** lw_obs; dlog = float(H["dlog"])
    shift_sys = math.log10(1.0 + args.dv_sys / C_KMS); lw_model = jnp.asarray(lw_obs - shift_sys)

    # ---- per-star geometry: base rows, log g nodes, per-epoch element arrays ----
    def star_setup(models, s):
        m0 = models[0][s]; vis = np.asarray(m0.mus) > 0; a = np.asarray(m0.visible_cast_areas)[vis]; p = np.asarray(m0.parameters)[vis]
        allg = np.concatenate([np.asarray(m.parameters)[np.asarray(m.mus) > 0, iG] for m in (pair[s] for pair in models)])
        return (jnp.asarray(np.average(p, axis=0, weights=a)), float(np.average(p[:, iG], weights=a)), jnp.asarray(np.linspace(allg.min() - 0.005, allg.max() + 0.005, args.n_g)))
    base = [None, None]; gref = [0.0, 0.0]; gn = [None, None]
    for s in (0, 1): base[s], gref[s], gn[s] = star_setup(SP["models"], s)
    base_lc = [star_setup(LC["models"], s)[0] for s in (0, 1)]
    T0 = [float(base[0][iT]), float(base[1][iT])]; feh0 = float(base[0][iF])

    def elems(m):
        a = np.asarray(m.visible_cast_areas); mu = np.asarray(m.mus); v = np.asarray(m.los_velocities); g = np.asarray(m.parameters)[:, iG]
        vis = (mu > 0) & (a > 0); v_orb = float(np.average(v[vis], weights=a[vis]))
        return jnp.asarray(a), jnp.asarray(mu), jnp.asarray(v), jnp.asarray(g), v_orb
    E = [[elems(pair[s]) for pair in SP["models"]] for s in (0, 1)]
    rv = np.array([[E[s][e][4] + args.dv_sys for e in range(n_ep)] for s in (0, 1)])          # component RVs per epoch
    print(f"start: Teff1 {T0[0]:.1f}  Teff2 {T0[1]:.1f}  [Fe/H] {feh0:+.2f}  | RV1 {rv[0].min():+.0f}..{rv[0].max():+.0f}  RV2 {rv[1].min():+.0f}..{rv[1].max():+.0f} km/s", flush=True)

    # kernel half width sized for a vsini scale up to 2.5
    dfine = dlog / args.oversample
    HW = [int(math.ceil(math.log10(1.0 + (max(abs(E[s][e][4]) for e in range(n_ep)) + 2.5 * max(float(jnp.max(jnp.abs(E[s][e][2] - E[s][e][4]))) for e in range(n_ep)) + 3.0) / C_KMS) / dfine)) + 1 for s in (0, 1)]

    def kernels_for(s, vscale):
        ks = []
        for e in range(n_ep):
            a, mu, v, g, v_orb = E[s][e]
            los = v_orb + vscale * (v - v_orb)
            ks.append(build_synthesis_kernel(Geom(a, mu, los), lw_model, args.n_mu, args.oversample, half_width=HW[s], element_coordinate=g, coordinate_nodes=gn[s]))
        return ks
    fixed_kernels = None if args.free_vsini else [kernels_for(0, 1.0), kernels_for(1, 1.0)]
    lw_ph = jnp.linspace(math.log10(4300.0), math.log10(5900.0), args.n_phot_wl); wl_ph = 10.0 ** lw_ph
    ph_k = [[build_synthesis_kernel(pair[s], lw_ph, args.n_mu, 1, element_coordinate=jnp.asarray(pair[s].parameters)[:, iG], coordinate_nodes=gn[s]) for pair in LC["models"]] for s in (0, 1)]
    resp = {nm.split(":")[1]: jnp.interp(wl_ph, jnp.asarray(LC["passbands"][nm][0]), jnp.asarray(LC["passbands"][nm][1]), left=0., right=0.) for nm in ("Stromgren:b", "Stromgren:y")}
    print(f"kernels: half widths {HW} px, {'rebuilt in-loss (free vsini)' if args.free_vsini else 'prebuilt'}", flush=True)

    # ---- data, masks, noise, continuum basis ----
    obs_np = np.array(H["obs"], float); good_np = np.isfinite(obs_np)
    if args.mask_in:
        m = np.asarray(np.load(args.mask_in, allow_pickle=True)["mask"], bool)[:n_ep]; good_np &= ~m
        print(f"mask-in: dropping {100 * m[np.isfinite(obs_np)].mean():.1f}% of usable pixels", flush=True)
    fit_e = epoch_subset(n_ep, args.fit_epochs); ep_w = np.zeros(n_ep); ep_w[fit_e] = 1.0
    edges = np.asarray(H["block_edges"]); blk = np.clip(np.searchsorted(edges, wl) - 1, 0, edges.size - 2)
    sig_np = np.hypot(np.nan_to_num(H["sigma_blocks"][:n_ep, blk], nan=0.01), args.spec_floor)
    obs = jnp.asarray(np.where(good_np, obs_np, 1.0)); good = jnp.asarray(good_np); sig = jnp.asarray(sig_np); ep_w = jnp.asarray(ep_w); blk_j = jnp.asarray(blk)
    N_sp = int(good_np[fit_e].sum())
    kn = np.arange(lw_obs[0], lw_obs[-1], math.log10(1 + args.knot_spacing / 5500.0)); interior = kn[(kn > lw_obs[0] + 1e-9) & (kn < lw_obs[-1] - 1e-9)]
    t_kn = np.concatenate([[lw_obs[0]] * 4, interior, [lw_obs[-1]] * 4])      # clamped cubic knot vector: k+1 copies at each end
    Bmat = jnp.asarray(BSpline.design_matrix(lw_obs, t_kn, 3).toarray()); nB = Bmat.shape[1]
    print(f"continuum: {nB} cubic B-spline coefficients per epoch ({args.knot_spacing} A knots); fitting {len(fit_e)} epochs, {N_sp} pixels", flush=True)
    ph_o, mag_o = load_photometry(args.photometry); sig_ph = {"b": 0.0041, "y": 0.0035}; N_ph = sum(len(v) for v in mag_o.values())
    model_phase = ((np.asarray(LC["times"]) - K.T_P_HJD) % K.PERIOD_DAYS) / K.PERIOD_DAYS; order = np.argsort(model_phase); mph = jnp.asarray(model_phase[order])

    # ---- parameter vector ----
    pnames = ["Teff1", "Teff2", "feh", "dphi"]; theta0 = [T0[0], T0[1], feh0, 0.1585]; S = [100., 100., 0.1, 0.001]; lo = [3800., 3800., -1.5, 0.10]; hi = [6900., 6900., 0.5, 0.22]
    if args.free_vmacro: pnames += ["vmac1", "vmac2"]; theta0 += list(args.vmacro0); S += [1., 1.]; lo += [0.5, 0.5]; hi += [20., 20.]
    if args.free_vsini: pnames += ["vsini_scale1", "vsini_scale2"]; theta0 += [1.0, 1.0]; S += [0.1, 0.1]; lo += [0.5, 0.5]; hi += [2.0, 2.0]
    theta0 = np.array(theta0); S = np.array(S); n_par = len(pnames)
    delta0 = np.zeros(n)
    if args.result_in:
        Rin = pickle.load(open(args.result_in, "rb"))
        for i, nm in enumerate(pnames):
            if nm in Rin["pnames"]: theta0[i] = Rin["theta"][Rin["pnames"].index(nm)]
        if Rin.get("delta") is not None and args.delta: delta0 = np.asarray(Rin["delta"])
        print("warm start from", args.result_in, flush=True)
    P = {nm: i for i, nm in enumerate(pnames)}

    # ---- model ----
    HALF_BR = 60
    xb = jnp.arange(-HALF_BR, HALF_BR + 1)
    def broaden(y, vmacro):
        sigma_pix = math.hypot(C_KMS / R_HARPS, 1.0) * 0 + jnp.hypot(C_KMS / R_HARPS, vmacro) / (C_KMS * math.log(10.0) * dlog) / 2.3548
        k = jnp.exp(-0.5 * (xb / sigma_pix) ** 2); k = k / k.sum()
        yp = jnp.pad(y, [(0, 0)] * (y.ndim - 1) + [(HALF_BR, HALF_BR)], mode="edge")
        return jax.vmap(lambda r: jnp.convolve(r, k, mode="valid"))(yp.reshape(-1, yp.shape[-1])).reshape(y.shape)

    def intensity_with_delta(delta):
        def fn(x, mu, prow):
            out = emu.intensity(x, mu, prow)
            return out.at[:, 0].multiply(1.0 + jnp.interp(x, lw_model, delta))
        return fn

    def rows(theta):
        r = []
        for s in (0, 1):
            b = base[s].at[iF].set(theta[P["feh"]]); r.append(gravity_darkened_rows(b, iT, iG, gn[s], theta[P[f"Teff{s+1}"]], gref[s], args.beta))
        return r

    def spectra(theta, delta):
        r = rows(theta); ks = fixed_kernels if fixed_kernels is not None else [kernels_for(0, theta[P["vsini_scale1"]]), kernels_for(1, theta[P["vsini_scale2"]])]
        vm = [theta[P["vmac1"]], theta[P["vmac2"]]] if args.free_vmacro else list(args.vmacro0)
        s1 = kernel_flux_multi(intensity_with_delta(delta), ks[0], r[0], wavelength_chunk_size=args.chunk)
        s2 = kernel_flux_multi(emu.intensity, ks[1], r[1], wavelength_chunk_size=args.chunk)
        F = broaden(s1[..., 0], vm[0]) + broaden(s2[..., 0], vm[1]); C = broaden(s1[..., 1], vm[0]) + broaden(s2[..., 1], vm[1])
        return F / C                                                              # (n_ep, n)

    def continuum_fix(model, ob, gd):
        w = jnp.where(gd, 1.0, 0.0); A = model[:, None] * Bmat * w[:, None]
        coef = jnp.linalg.solve(A.T @ A + 1e-9 * jnp.eye(nB), A.T @ (ob * w)); return model * (Bmat @ coef)

    def chi2_spec(theta, delta, block_w):
        mod = jax.vmap(continuum_fix)(spectra(theta, delta), obs, good)
        r = jnp.where(good, (obs - mod) / sig, 0.0)
        per_epoch = jnp.sum(block_w[blk_j][None, :] * r ** 2, axis=1)      # block weights mapped to pixels
        return jnp.sum(ep_w * per_epoch)                                     # two-level sum: float32 roundoff ~1e-6 of chi2, not 1e-5

    def chi2_phot(theta):
        r = []
        for s in (0, 1):
            b = base_lc[s].at[iF].set(theta[P["feh"]]); r.append(gravity_darkened_rows(b, iT, iG, gn[s], theta[P[f"Teff{s+1}"]], gref[s], args.beta))
        flux = kernel_flux_multi(emu.intensity, ph_k[0], r[0])[..., 0] + kernel_flux_multi(emu.intensity, ph_k[1], r[1])[..., 0]
        tot = 0.0
        for b, rr in resp.items():
            mags = jax.vmap(lambda f: GI.passband_mag(f, wl_ph, rr))(flux)[order]; mags = mags - jnp.median(mags)
            mi = jnp.interp((ph_o - theta[P["dphi"]]) % 1.0, mph, mags, period=1.0); res = mag_o[b] - mi; res = res - jnp.median(res)
            tot = tot + jnp.sum(res ** 2) / sig_ph[b] ** 2
        return tot

    def delta_prior(delta):
        return jnp.sum((delta / args.delta_sigma) ** 2) + args.delta_smooth * jnp.sum((jnp.diff(delta, 2) / args.delta_sigma) ** 2)

    def total(x, dlt, block_w):
        th = jnp.asarray(theta0) + x * jnp.asarray(S)
        c_ph = chi2_phot(th); c_sp = chi2_spec(th, dlt, block_w)
        return c_ph + c_sp + (delta_prior(dlt) if args.delta else 0.0), (c_ph, c_sp)
    vg_theta = jax.jit(jax.value_and_grad(total, argnums=0, has_aux=True))
    vg_delta = jax.jit(jax.value_and_grad(total, argnums=1, has_aux=True))
    blocks_all = jnp.ones(edges.size - 1)

    # ---- validation against exact per-element synthesis (epoch 0, theta0, vsini 1) ----
    if not args.skip_validation:
        from spice.spectrum.spectrum import simulate_observed_flux
        th = jnp.asarray(theta0); r = rows(th); m1, m2 = SP["models"][0]; vm = list(args.vmacro0)
        def exact(mm, rr, s):
            p = jnp.asarray(mm.parameters); teff = th[P[f"Teff{s+1}"]] * jnp.power(10.0, args.beta * (p[:, iG] - gref[s]))
            return simulate_observed_flux(emu.intensity, mm._replace(parameters=p.at[:, iT].set(teff).at[:, iF].set(th[P["feh"]])), lw_model, chunk_size=256)
        t = time.time(); e1 = exact(m1, r[0], 0); e2 = exact(m2, r[1], 1)
        ex = np.asarray((broaden(e1[None, :, 0], vm[0]) + broaden(e2[None, :, 0], vm[1])) / (broaden(e1[None, :, 1], vm[0]) + broaden(e2[None, :, 1], vm[1])))[0]
        su = np.asarray(jax.jit(lambda t_: spectra(t_, jnp.zeros(n))[0])(th)); g0 = good_np[0]
        d = np.abs(su - ex)[g0]; print(f"kernel vs exact (epoch 0, full range, {n} px): max {d.max():.5f} rms {np.sqrt((d ** 2).mean()):.6f}  ({time.time() - t:.0f}s)", flush=True)
        if d.max() > 0.01: print("ABORT: kernel model outside tolerance"); sys.exit(2)

    # ---- optimisation ----
    x0 = np.zeros(n_par); dlt = jnp.asarray(delta0)
    t = time.time(); (v, aux), g = vg_theta(jnp.asarray(x0), dlt, blocks_all); jax.block_until_ready(g); print(f"first value+grad (compile) {time.time() - t:.0f}s", flush=True)
    t = time.time(); jax.block_until_ready(vg_theta(jnp.asarray(x0), dlt, blocks_all)[1]); print(f"warm value+grad {time.time() - t:.1f}s", flush=True)
    print(f"start: chi2_phot/N {float(aux[0]) / N_ph:.3f}  chi2_spec/N {float(aux[1]) / N_sp:.3f}", flush=True)
    bounds = list(zip((np.array(lo) - theta0) / S, (np.array(hi) - theta0) / S))
    if args.noise_check:
        fval = lambda x_: float(vg_theta(jnp.asarray(x_), dlt, blocks_all)[0][0])
        rep = [fval(x0) for _ in range(3)]; print(f"noise check: 3 repeat evaluations {rep} -> spread {max(rep) - min(rep):.3g} (chi2 total {rep[0]:.6g})", flush=True)
        for i, nm in enumerate(pnames):
            for h in (1e-3, 1e-2, 1e-1):
                xp = x0.copy(); xp[i] += h; xm = x0.copy(); xm[i] -= h; fp, fm = fval(xp), fval(xm)
                print(f"  {nm:12s} h={h:<5g} f(+h)-f(-h) = {fp - fm:+11.4g}   FD grad {(fp - fm) / (2 * h):+12.6g}   AD grad {float(g[i]):+12.6g}", flush=True)

    def fit_theta(x_start, dlt, block_w, maxiter, label):
        hist = []
        def f(x):
            (v, _), g = vg_theta(jnp.asarray(x), dlt, block_w); hist.append(float(v)); return float(v), np.asarray(g, float)
        # L-BFGS-B stalls ("ABNORMAL", 0-1 iterations) when the start is already within the float32 roundoff of the
        # objective (~10 units of chi2 out of 1e7): the achievable decrease along the first steepest-descent line is below
        # the noise. Restart from a perturbed point (0.3 scaled units: 30 K, 0.03 dex, 0.3 km/s) so the fit re-enters the
        # basin with real decreases and settles to the noise-limited optimum. Keep the best point seen.
        t = time.time(); x_s = np.asarray(x_start, float); best = None; rng = np.random.default_rng(0); tries = 0; nit = 0
        lo_b = np.array([b[0] for b in bounds]); hi_b = np.array([b[1] for b in bounds])
        while True:
            r = minimize(f, x_s, jac=True, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=maxiter)); nit += r.nit
            if best is None or r.fun < best.fun: best = r
            stalled = r.nit <= 1 and "ABNORMAL" in r.message
            if not stalled or tries >= args.stall_retries: break
            tries += 1; x_s = np.clip(np.asarray(best.x) + rng.normal(0.0, 0.3, size=x_s.size), lo_b, hi_b)
            print(f"  {label}: stalled ({r.message[:30].strip()}) -> retry {tries} from a perturbed start", flush=True)
        r = best; r.nit = nit
        (_, aux), _ = vg_theta(jnp.asarray(r.x), dlt, block_w); th = theta0 + np.asarray(r.x) * S
        print(f"{label}: {r.nit} it, {len(hist)} eval, {time.time() - t:.0f}s, {r.message[:30]} | chi2_phot/N {float(aux[0]) / N_ph:.3f} chi2_spec/N {float(aux[1]) / N_sp:.3f} | "
              + " ".join(f"{nm} {val:.4g}" for nm, val in zip(pnames, th)), flush=True)
        return np.asarray(r.x), th, aux

    def fit_delta(x, dlt, maxiter):
        def f(d):
            (v, _), g = vg_delta(jnp.asarray(x), jnp.asarray(d), blocks_all); return float(v), np.asarray(g, float)
        t = time.time(); r = minimize(f, np.asarray(dlt), jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter))
        d = np.asarray(r.x); print(f"delta fit: {r.nit} it, {time.time() - t:.0f}s, rms delta {100 * d.std():.2f}% (prior {100 * args.delta_sigma:.0f}%)", flush=True)
        return jnp.asarray(d)

    if args.delta and args.delta_mode == "fixed":
        dlt = jnp.asarray(pickle.load(open(args.delta_in, "rb"))["delta"]); print(f"delta held fixed from {args.delta_in} (rms {100 * float(jnp.std(dlt)):.2f}%)", flush=True)
    if args.delta and args.delta_mode == "fit-only":
        dlt = fit_delta(x0, dlt, args.delta_maxiter); th = theta0.copy(); x = x0
        (_, aux), _ = vg_theta(jnp.asarray(x0), dlt, blocks_all); th_nodelta = th.copy()
        print("delta fitted at the warm-start theta; theta not refitted (use delta-mode fixed on the other epoch subset)", flush=True)
    else:
        x, th, aux = fit_theta(x0, dlt, blocks_all, args.maxiter, "pass 1" + (" (delta fixed)" if args.delta and args.delta_mode == "fixed" else " (delta = 0)"))
        th_nodelta = th.copy()
        if args.delta and args.delta_mode == "joint":
            for it in range(args.delta_alternations):
                dlt = fit_delta(x, dlt, args.delta_maxiter)
                x, th, aux = fit_theta(x, dlt, blocks_all, 40, f"alternation {it + 1}: theta | delta")
            print("delta effect on theta: " + " ".join(f"{nm} {b - a:+.4g}" for nm, a, b in zip(pnames, th_nodelta, th)), flush=True)

    result = dict(pnames=pnames, theta=th, theta_nodelta=th_nodelta, x=x, S=S, theta0=theta0, chi2_phot=float(aux[0]), chi2_spec=float(aux[1]), N_ph=N_ph, N_sp=N_sp,
                  delta=(np.asarray(dlt) if args.delta else None), args=vars(args), rv=rv, fit_epochs=fit_e, beta=args.beta, delta_mode=args.delta_mode)
    pickle.dump(result, open(args.out, "wb"), protocol=4)

    # ---- residual stack -> learned mask (optional) ----
    if args.derive_mask_out:
        mod = np.asarray(jax.jit(lambda t_, d_: jax.vmap(continuum_fix)(spectra(t_, d_), obs, good))(jnp.asarray(th), dlt))
        res = np.where(good_np, obs_np - mod, np.nan); me = epoch_subset(n_ep, args.mask_epochs)
        def stack(rvs):
            Sx = np.full((len(me), n), np.nan)
            for j, e in enumerate(me):
                Sx[j] = np.interp(lw_obs + math.log10(1.0 + rvs[e] / C_KMS), lw_obs, res[e], left=np.nan, right=np.nan)
            med = np.nanmedian(Sx, 0); mad = 1.4826 * np.nanmedian(np.abs(Sx - med), 0); cnt = np.isfinite(Sx).sum(0)
            bad = (np.abs(med) > args.mask_threshold) & (np.abs(med) > args.mask_nsigma * mad / np.sqrt(np.maximum(cnt, 1)))
            B = np.zeros(n, bool)
            for i in np.where(bad)[0]: B[max(0, i - args.mask_dilate):i + args.mask_dilate + 1] = True
            return B
        B1, B2 = stack(rv[0]), stack(rv[1]); mask = np.zeros((n_ep, n), bool)
        for e in range(n_ep):
            for B, rvs in ((B1, rv[0]), (B2, rv[1])):
                mask[e] |= np.interp(lw_obs - math.log10(1.0 + rvs[e] / C_KMS), lw_obs, B.astype(float), left=0, right=0) > 0.5
        rb = np.sqrt(np.nanmean(res ** 2)); ra = np.sqrt(np.nanmean(np.where(mask, np.nan, res) ** 2))
        np.savez(args.derive_mask_out, mask=mask, threshold=args.mask_threshold, mask_epochs=me, rv=rv)
        print(f"mask derived from {args.mask_epochs} epochs: primary-frame bad px {B1.sum()}, secondary {B2.sum()}; masks {100 * mask[good_np].mean():.1f}% of usable pixels; residual rms {100 * rb:.2f}% -> {100 * ra:.2f}%; saved {args.derive_mask_out}", flush=True)

    # ---- block jackknife ----
    if args.jackknife_blocks:
        nb = edges.size - 1; per = nb // args.jackknife_blocks; jk = []
        for j in range(args.jackknife_blocks):
            bw = np.ones(nb); bw[j * per:(j + 1) * per] = 0.0
            _, thj, _ = fit_theta(x, dlt, jnp.asarray(bw), 40, f"drop block {j + 1}/{args.jackknife_blocks} ({edges[j * per]:.0f}-{edges[min((j + 1) * per, nb)]:.0f} A)")
            jk.append(thj)
        A = np.array(jk); m = A.shape[0]; err = np.sqrt((m - 1) / m * np.sum((A - A.mean(0)) ** 2, 0))
        print("jackknife error: " + " ".join(f"{nm} {e_:.3g}" for nm, e_ in zip(pnames, err)), flush=True)
        result["jackknife"] = A; result["jackknife_err"] = err
        pickle.dump(result, open(args.out, "wb"), protocol=4)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
