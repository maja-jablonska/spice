"""Fit Gaia FGK benchmark stars with the TZ For spectroscopic estimator, to calibrate it against fundamental Teff.

Each star is fitted the way the TZ For components are: the HARPS emulator (SPICE_HARPS_BUNDLE, an 11-input
interpolated-atmosphere bundle; default mjablonska/TPayne-spice-harps-interp), a SPICE icosphere with rigid
rotation synthesised element by element (simulate_observed_flux, each element at its own mu and Doppler shift), Gaussian
instrumental + macroturbulent broadening, the cubic-spline continuum with 40 A knots fitted in closed
form, the 0.005 noise floor, the four 40 A windows of the headline fit, and scaled-solar abundances
with fixed microturbulence (1.5 km/s as for TZ For, or the GBS value with --vmicro gbs). log g is fixed
at the GBS value (from mass and radius), as the geometry fixes it for TZ For. Free: Teff, [Fe/H],
vmacro, vsini and a velocity offset. Pass 1 fits every pixel; pass 2 masks pixels whose residual
exceeds --mask-threshold and --mask-nsigma sigma (dilated), the single-spectrum version of the TZ For
learned mask; a leave-one-window-out jackknife gives the statistical error.

The fitted minus GBS Teff and [Fe/H] measure the estimator's systematic error, emulator, line list,
1D LTE MARCS atmospheres and continuum treatment together.

Usage: python gbs_fit.py --data gbs_harps.npz --targets targets.csv --harps-grid harps_fullrange_log.npz --out OUT.json
           [--vmicro 1.5|gbs] [--stars HD168723,HD61421] [--gpu-precision highest]
"""
import argparse, csv, json, math, os, sys, time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
HERE = Path(__file__).resolve().parent
import numpy as np, jax, jax.numpy as jnp
from scipy.optimize import minimize
from scipy.interpolate import BSpline
from spice.models import IcosphereModel
from spice.spectrum import simulate_observed_flux

C_KMS = 299792.458; R_HARPS = 115000.0
NAMES = ["marcs_teff", "marcs_logg", "feh", "vmicro", "a", "c", "n", "o", "r", "s"]
HALF_BR = 60                                     # broadening padding [px] on each side of a window


def make_emulator(bundle=None):
    """The HARPS intensity emulator. SPICE_HARPS_BUNDLE (hub name or local bundle directory) overrides the default."""
    from spice.spectrum.aemu_spectrum_emulator import IntensityPretrainedAemuSpectrumEmulator
    bundle = bundle or os.environ.get("SPICE_HARPS_BUNDLE", "mjablonska/TPayne-spice-harps-interp")
    print("HARPS emulator bundle:", bundle, flush=True)
    emu = IntensityPretrainedAemuSpectrumEmulator(bundle)
    if "marcs_fe_h" in emu.stellar_parameter_names:
        raise SystemExit("12-input bundles with a separate atmosphere [Fe/H] (v2b) are not supported; use an 11-input bundle")
    return emu


def conv_valid(y, k):
    """jnp.convolve(y, k, 'valid') along the last axis as shifted slices (no cuDNN: V100-safe)."""
    n = y.shape[-1] - k.shape[0] + 1
    return sum(k[k.shape[0] - 1 - j] * y[..., j:j + n] for j in range(k.shape[0]))


def envelope(wl, f, deg=2, iters=8):
    """Upper-envelope polynomial of one window (fit, keep points above fit - 0.5 sigma, repeat)."""
    x = (wl - wl.mean()) / (np.ptp(wl) / 2); m = np.isfinite(f) & (f > 0)
    for _ in range(iters):
        c = np.polyfit(x[m], f[m], deg); r = f - np.polyval(c, x); s = np.std(r[m]); m2 = m & (r > -0.5 * s)
        if m2.sum() < 20 or m2.sum() == m.sum(): break
        m = m2
    return np.polyval(np.polyfit(x[m], f[m], deg), x)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True); ap.add_argument("--targets", required=True); ap.add_argument("--harps-grid", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--stars", default="")
    ap.add_argument("--windows", type=float, nargs="*", default=[5160, 5200, 5380, 5420, 6100, 6140, 6400, 6440])
    ap.add_argument("--vmicro", default="1.5", help="microturbulence [km/s] for every star, or 'gbs' for the table value")
    ap.add_argument("--n-vert", type=int, default=2000, help="icosphere vertices (emulator cost scales with the visible elements)")
    ap.add_argument("--chunk", type=int, default=256, help="surface elements per compiled chunk of simulate_observed_flux")
    ap.add_argument("--spec-floor", type=float, default=0.005); ap.add_argument("--knot-spacing", type=float, default=40.0)
    ap.add_argument("--maxiter", type=int, default=150); ap.add_argument("--jk-maxiter", type=int, default=60)
    ap.add_argument("--mask-threshold", type=float, default=0.05); ap.add_argument("--mask-nsigma", type=float, default=4.0); ap.add_argument("--mask-dilate", type=int, default=2)
    ap.add_argument("--fix-teff", action="store_true", help="hold Teff at the GBS value (does the [Fe/H] offset survive without the Teff-[Fe/H] trade-off?)")
    args = ap.parse_args()
    print("devices:", jax.devices(), flush=True)
    emu = make_emulator()

    # ---- pixel grid: the TZ For HARPS log grid restricted to the windows; one padded uniform segment per window ----
    H = np.load(args.harps_grid, allow_pickle=True); lw_full = np.asarray(H["logwl"], float); dlog = float(H["dlog"]); wl_full = 10.0 ** lw_full
    wins = np.asarray(args.windows, float).reshape(-1, 2)
    runs = [np.where((wl_full >= a) & (wl_full <= b))[0] for a, b in wins]
    pix = np.concatenate(runs); lw_obs = lw_full[pix]; n = pix.size; seg = [r.size for r in runs]
    win_of_px = np.concatenate([np.full(r.size, i) for i, r in enumerate(runs)])
    lw_seg = [jnp.asarray(lw_full[r[0]] + dlog * np.arange(-HALF_BR, r.size + HALF_BR)) for r in runs]
    kn = np.arange(lw_obs[0], lw_obs[-1], math.log10(1 + args.knot_spacing / 5500.0)); interior = kn[(kn > lw_obs[0] + 1e-9) & (kn < lw_obs[-1] - 1e-9)]
    t_kn = np.concatenate([[lw_obs[0]] * 4, interior, [lw_obs[-1]] * 4]); Bm = BSpline.design_matrix(lw_obs, t_kn, 3).toarray()
    Bmat = jnp.asarray(Bm[:, (Bm > 1e-3).sum(0) >= 30]); nB = Bmat.shape[1]
    print(f"windows {wins.tolist()}: {n} px ({seg}), {nB} continuum splines", flush=True)

    # ---- geometry: a unit icosphere seen equator-on; rigid rotation about y gives v_los = vsini * x / R ----
    mesh = IcosphereModel.construct(args.n_vert, 1.0, 1.0, np.array([5000.0, 3.0, 0.0, 1.5, 0, 0, 0, 0, 0, 0]), NAMES)
    vis = (np.asarray(mesh.mus) > 0) & (np.asarray(mesh.visible_cast_areas) > 0)
    xc = np.asarray(mesh.d_centers)[:, 0]; X = jnp.asarray(xc / np.max(np.abs(xc[vis])))
    # unit velocity field along the line of sight: los_velocities = v . los_vector, so v = X * los_vector gives
    # los_velocities = X (|los_vector| = 1); scaled by vsini and offset by dv inside the model
    LOS = jnp.asarray(mesh.los_vector, float); V_UNIT = X[:, None] * LOS[None, :]; V_OFF = LOS[None, :] * jnp.ones_like(X)[:, None]
    assert np.allclose(np.asarray(mesh._replace(rotation_velocities=V_UNIT).los_velocities), np.asarray(X)), "LOS sign convention"
    print(f"mesh: {int(vis.sum())} visible elements", flush=True)
    xb = jnp.arange(-HALF_BR, HALF_BR + 1)

    def model(theta, base):
        """Continuum-normalised spectrum on the window pixels. theta = (Teff, [Fe/H], vmacro, vsini, dv)."""
        row = base.at[0].set(theta[0]).at[2].set(theta[1])
        m = mesh._replace(parameters=jnp.broadcast_to(row, mesh.parameters.shape),
                          rotation_velocities=theta[3] * V_UNIT, pulsation_velocities=theta[4] * V_OFF)
        sig = jnp.hypot(C_KMS / R_HARPS, theta[2]) / (C_KMS * math.log(10.0) * dlog) / 2.3548
        k = jnp.exp(-0.5 * (xb / sig) ** 2); k = k / k.sum(); out = []
        for lw in lw_seg:
            f = simulate_observed_flux(emu.intensity, m, lw, chunk_size=args.chunk)
            out.append(conv_valid(f[:, 0], k) / conv_valid(f[:, 1], k))
        return jnp.concatenate(out)

    def continuum_fix(mod, ob, w):
        Aw = mod[:, None] * Bmat * w[:, None]
        coef = jnp.linalg.solve(Aw.T @ Aw + 1e-9 * jnp.eye(nB), Aw.T @ (ob * w)); return mod * (Bmat @ coef)

    def chi2(x, x0, S, base, ob, sg, w):
        th = x0 + x * S; m = continuum_fix(model(th, base), ob, w)
        return jnp.sum(w * ((ob - m) / sg) ** 2)
    vg = jax.jit(jax.value_and_grad(chi2))
    fitted_model = jax.jit(lambda th, base, ob, w: continuum_fix(model(th, base), ob, w))
    raw_model = jax.jit(model)

    D = np.load(args.data, allow_pickle=True); targets = list(csv.DictReader(open(args.targets)))
    only = [s for s in args.stars.split(",") if s]
    results, dumps = {}, {}
    S = np.array([100.0, 0.1, 1.0, 1.0, 0.3]); lo = np.array([4000.0, -3.5, 0.3, 0.3, -10.0]); hi = np.array([7500.0, 0.5, 20.0, 35.0, 10.0])
    for t in targets:
        star = t["star"]
        if (only and star not in only) or f"{star}/wl" not in D.files:
            continue
        t0 = time.time()
        vmic = float(t["vmic"]) if args.vmicro == "gbs" else float(args.vmicro)
        alpha = float(t.get("alpha") or 0.0)            # [alpha/Fe]: 0 (scaled solar) except the extended metal-poor group (GBS [Mg/Fe])
        base = jnp.asarray([float(t["teff"]), float(t["logg"]), float(t["feh"]), vmic, alpha, 0, 0, 0, 0, 0], float)
        ref = np.array([float(t["teff"]), float(t["feh"]), max(float(t["vmac"]), 1.0), max(float(t["vsini"]), 1.0), 0.0])
        wl, fl, er = (np.asarray(D[f"{star}/{k}"], float) for k in ("wl", "flux", "err"))

        # radial velocity: CCF of the observed spectrum against the model at the GBS parameters, Mg b window
        mref = np.asarray(raw_model(jnp.asarray(ref), base)); m0 = slice(0, seg[0]); wl0 = 10.0 ** lw_obs[m0]
        def ccf(v):
            f = np.interp(wl0 * (1 + v / C_KMS), wl, fl); f = f / envelope(wl0, f)
            return float(np.corrcoef(1 - f, 1 - mref[m0])[0, 1])
        vg_ = np.arange(-200.0, 200.0, 1.0); c = np.array([ccf(v) for v in vg_]); j = int(np.argmax(c))
        vf = np.arange(vg_[j] - 2, vg_[j] + 2, 0.05); cf = np.array([ccf(v) for v in vf]); rv = float(vf[np.argmax(cf)])
        # rest-frame observed spectrum on the window pixels, normalised per window by its upper envelope
        wl_px = 10.0 ** lw_obs
        ob = np.interp(wl_px * (1 + rv / C_KMS), wl, fl, left=np.nan, right=np.nan); eo = np.interp(wl_px * (1 + rv / C_KMS), wl, er, left=np.nan, right=np.nan)
        for i in range(len(runs)):
            m = win_of_px == i; env = envelope(wl_px[m], ob[m]); ob[m] /= env; eo[m] /= env
        good = np.isfinite(ob) & np.isfinite(eo) & (eo > 0)
        sg = np.hypot(np.where(good, eo, 1.0), args.spec_floor); obj = jnp.asarray(np.where(good, ob, 1.0)); sgj = jnp.asarray(sg)

        def fit(x_start, w, maxiter):
            x0j, Sj = jnp.asarray(ref), jnp.asarray(S)
            f = lambda x: tuple(np.asarray(v, float) for v in vg(jnp.asarray(x), x0j, Sj, base, obj, sgj, jnp.asarray(w)))
            bounds = list(zip((lo - ref) / S, (hi - ref) / S))
            if args.fix_teff:
                bounds[0] = (0.0, 0.0)
            r = minimize(f, x_start, jac=True, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=maxiter))
            return np.asarray(r.x), ref + np.asarray(r.x) * S, float(r.fun), r
        w1 = good.astype(float)
        x1, th1, c1, r1 = fit(np.zeros(5), w1, args.maxiter)
        res1 = np.where(good, ob - np.asarray(fitted_model(jnp.asarray(th1), base, obj, jnp.asarray(w1))), np.nan)
        bad = (np.abs(res1) > args.mask_threshold) & (np.abs(res1) > args.mask_nsigma * sg); bad = np.nan_to_num(bad).astype(bool)
        masked = np.zeros(n, bool)
        for i in np.where(bad)[0]: masked[max(0, i - args.mask_dilate):i + args.mask_dilate + 1] = True
        w2 = (good & ~masked).astype(float)
        x2, th2, c2, r2 = fit(x1, w2, args.maxiter)
        jk = []
        for i in range(len(runs)):
            _, thj, _, _ = fit(x2, w2 * (win_of_px != i), args.jk_maxiter); jk.append(thj)
        jk = np.array(jk); m_ = jk.shape[0]; jk_err = np.sqrt((m_ - 1) / m_ * np.sum((jk - jk.mean(0)) ** 2, 0))
        mod2 = np.asarray(fitted_model(jnp.asarray(th2), base, obj, jnp.asarray(w2))); res2 = np.where(w2 > 0, ob - mod2, np.nan)
        pn = ["teff", "feh", "vmac", "vsini", "dv"]
        results[star] = dict(kind=t["kind"], ref=dict(teff=float(t["teff"]), eteff=float(t["eteff"]), logg=float(t["logg"]), feh=float(t["feh"]), efeh=float(t["efeh"]),
                                                     vmic=float(t["vmic"]), vmac=float(t["vmac"]), vsini=float(t["vsini"])),
                             vmicro_used=vmic, alpha_used=alpha, rv=rv, n_px=int(good.sum()), masked_frac=float(masked[good].mean()),
                             plain=dict(zip(pn, th1.tolist()), chi2_N=c1 / max(w1.sum(), 1), nit=int(r1.nit)),
                             masked=dict(zip(pn, th2.tolist()), chi2_N=c2 / max(w2.sum(), 1), nit=int(r2.nit), rms=float(np.nanstd(res2))),
                             jackknife_err=dict(zip(pn, jk_err.tolist())), jackknife=jk.tolist(),
                             dteff=float(th2[0] - float(t["teff"])), dfeh=float(th2[1] - float(t["feh"])),
                             dteff_plain=float(th1[0] - float(t["teff"])), dfeh_plain=float(th1[1] - float(t["feh"])), seconds=time.time() - t0)
        dumps[f"{star}/obs"] = ob; dumps[f"{star}/model"] = mod2; dumps[f"{star}/good"] = good; dumps[f"{star}/masked"] = masked; dumps[f"{star}/sigma"] = sg
        R = results[star]
        print(f"{star:9s} {t['kind']:5s} GBS {float(t['teff']):6.0f}+-{float(t['eteff']):3.0f} {float(t['feh']):+.2f} | rv {rv:+7.2f} | plain {th1[0]:6.0f} {th1[1]:+.3f} "
              f"| masked ({100 * R['masked_frac']:.1f}%) {th2[0]:6.0f}+-{jk_err[0]:3.0f} {th2[1]:+.3f}+-{jk_err[1]:.3f} vmac {th2[2]:.1f} vsini {th2[3]:.1f} "
              f"| dTeff {R['dteff']:+5.0f} d[Fe/H] {R['dfeh']:+.3f} | chi2/N {R['masked']['chi2_N']:.2f} rms {R['masked']['rms']:.4f} | {time.time() - t0:.0f}s", flush=True)
        json.dump(dict(args=vars(args), bundle=os.environ.get("SPICE_HARPS_BUNDLE"), results=results), open(args.out, "w"), indent=1)

    for kind in ("giant", "fstar", "sun", "dwarf", "metalpoor", "giant_w", "dwarf_w"):
        d = np.array([[r["dteff"], r["dfeh"], r["dteff_plain"], r["dfeh_plain"]] for r in results.values() if r["kind"] == kind])
        if d.size:
            k = d.shape[0]
            print(f"{kind}: N {k}  masked dTeff mean {d[:, 0].mean():+.0f} +- {d[:, 0].std(ddof=1) / math.sqrt(k) if k > 1 else float('nan'):.0f} (scatter {d[:, 0].std(ddof=1) if k > 1 else float('nan'):.0f}) K, "
                  f"d[Fe/H] {d[:, 1].mean():+.3f} (scatter {d[:, 1].std(ddof=1) if k > 1 else float('nan'):.3f}) | plain dTeff {d[:, 2].mean():+.0f} d[Fe/H] {d[:, 3].mean():+.3f}", flush=True)
    np.savez_compressed(os.path.splitext(args.out)[0] + "_spectra.npz", logwl=lw_obs, **dumps)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
