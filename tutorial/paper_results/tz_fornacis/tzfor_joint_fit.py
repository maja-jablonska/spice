"""Joint photometry + spectroscopy gradient fit for TZ For (GPU job).

Photometry alone cannot separate Teff1 from Teff2 (corr = +1.00: differential
b, y only fix the light ratio). Spectroscopy breaks that through line-depth
ratios, so this fits Clausen b, y AND the 21 HARPS epochs together, with the
PHOEBE geometry fixed and the atmospheric parameters differentiable.

Spectroscopic term: the disc integral with EXACT per-element Doppler shifts is
linear in the emulator output once I(lambda, mu) is interpolated in mu and in
wavelength, so each epoch is a fixed sparse operator built once (numpy) from
the mesh. The emulator then runs n_mu times per star per gradient step on an
oversampled grid instead of ~1700 times per star per epoch. Both surrogates are
checked against the exact per-element synthesis before the optimisation starts
and again at the best fit.

Assumptions (stated, not hidden):
  * loss = chi2_phot + chi2_spec with real sigmas (Clausen scatter per band;
    HARPS pixel noise per epoch with a --spec-floor to absorb normalisation /
    emulator systematics). --spec-weight rescales the spectroscopic term.
  * per epoch and window a linear continuum correction (a + b*x) is solved in
    closed form -- it absorbs SuppNet normalisation tilt, not line depths.
  * a single systemic velocity is measured by cross-correlation against the
    starting model and baked into the operators; macroturbulence is fixed
    (5, 6 km/s) and instrumental R = 115000, as in tzfor_compare_spectra.
"""
import argparse
import json
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
from tzfor_compare_spectra import velocity_offset

C_KMS = 299792.458
R_HARPS = 115000.0


# ---------------------------------------------------------------------------
# Exact-Doppler linear operator
# ---------------------------------------------------------------------------

def build_operator(mus, areas, vlos, wl_out, wl_fine, n_mu, dv_sys=0.0):
    """Sparse (rows, cols, vals) with flux[k] = sum vals * I_flat[cols] into rows.

    I_flat is I(wl_fine, mu_nodes) flattened as mu-major. Linear in mu between
    nodes and linear in rest wavelength on the fine grid; the Doppler shift of
    every element is applied exactly to its sampling wavelengths.
    """
    mus = np.asarray(mus); areas = np.asarray(areas); vlos = np.asarray(vlos) + dv_sys
    vis = (mus > 0) & (areas > 0)
    mus, areas, vlos = mus[vis], areas[vis], vlos[vis]
    mu_nodes = np.linspace(0.0, 1.0, n_mu)
    im = np.clip(np.searchsorted(mu_nodes, mus) - 1, 0, n_mu - 2)
    fm = (mus - mu_nodes[im]) / (mu_nodes[im + 1] - mu_nodes[im])
    n_out, n_fine = wl_out.size, wl_fine.size
    lam_rest = wl_out[None, :] / (1.0 + vlos[:, None] / C_KMS)
    j = np.clip(np.searchsorted(wl_fine, lam_rest.ravel()) - 1, 0, n_fine - 2).reshape(lam_rest.shape)
    fw = (lam_rest - wl_fine[j]) / (wl_fine[j + 1] - wl_fine[j])
    k = np.broadcast_to(np.arange(n_out)[None, :], lam_rest.shape)
    rows, cols, vals = [], [], []
    for dm, wm in ((0, 1 - fm), (1, fm)):
        base = ((im + dm) * n_fine)[:, None]
        for dj, wj in ((0, 1 - fw), (1, fw)):
            rows.append(k.ravel()); cols.append((base + j + dj).ravel())
            vals.append((areas[:, None] * wm[:, None] * wj).ravel())
    rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals)
    key = rows.astype(np.int64) * (n_mu * n_fine) + cols
    uk, inv = np.unique(key, return_inverse=True)
    v = np.zeros(uk.size); np.add.at(v, inv, vals)
    return (uk // (n_mu * n_fine)).astype(np.int32), (uk % (n_mu * n_fine)).astype(np.int32), v


def stack_operators(ops):
    """Pad to a common nnz (zero vals contribute nothing) so lax.map can scan epochs."""
    nnz = max(o[2].size for o in ops)
    n_out = int(max(o[0].max() for o in ops)) + 1
    # rows come out of np.unique sorted; pad with the LAST row so they stay
    # sorted and segment_sum can use a segmented reduction instead of atomics.
    R = np.full((len(ops), nnz), n_out - 1, np.int32); Cc = np.zeros((len(ops), nnz), np.int32); V = np.zeros((len(ops), nnz))
    for e, (r, c, v) in enumerate(ops):
        R[e, :r.size] = r; Cc[e, :c.size] = c; V[e, :v.size] = v
    return jnp.asarray(R), jnp.asarray(Cc), jnp.asarray(V)


def apply_op(r, c, v, x, n_out):
    return jax.ops.segment_sum(v * jnp.take(x, c), r, num_segments=n_out, indices_are_sorted=True)


# ---------------------------------------------------------------------------
# Broadening (instrumental + macroturbulence), per window
# ---------------------------------------------------------------------------

def gauss_kernel(sigma_pix):
    half = int(math.ceil(4 * sigma_pix))
    x = np.arange(-half, half + 1)
    k = np.exp(-0.5 * (x / sigma_pix) ** 2)
    return jnp.asarray(k / k.sum()), half


def make_broadener(wl_out, windows, vmacro_kms):
    """Returns broaden(y) applying a per-window Gaussian (edge-padded, so no wrap)."""
    segs = []
    for lo, hi in windows:
        idx = np.where((wl_out >= lo) & (wl_out <= hi))[0]
        w = wl_out[idx]; lam0, dlam = 0.5 * (w[0] + w[-1]), float(w[1] - w[0])
        fwhm = math.hypot(C_KMS / R_HARPS, vmacro_kms)
        sigma_pix = lam0 * fwhm / C_KMS / 2.3548 / dlam
        kern, half = gauss_kernel(sigma_pix)
        segs.append((int(idx[0]), int(idx[-1] + 1), kern, half))

    def broaden(y):
        out = []
        for a, b, kern, half in segs:
            seg = jnp.pad(y[a:b], half, mode="edge")
            out.append(jnp.convolve(seg, kern, mode="valid"))
        return jnp.concatenate(out)
    return broaden


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

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
    ap.add_argument("--harps", default=str(HERE / "tzfor_aemu_out" / "harps_windows.npz"))
    ap.add_argument("--photometry", default=str(HERE / "tzfor_lightcurve.csv"))
    ap.add_argument("--out", default=str(HERE / "tzfor_aemu_out" / "joint_fit_result.pkl"))
    ap.add_argument("--n-mu", type=int, default=32)
    ap.add_argument("--oversample", type=int, default=4, help="fine-grid oversampling vs HARPS 0.02 A (4: 0.28%% max / 0.04%% rms vs exact)")
    ap.add_argument("--spec-floor", type=float, default=0.005, help="error floor added in quadrature to HARPS pixel noise")
    ap.add_argument("--spec-weight", type=float, default=1.0, help="multiplier on chi2_spec")
    ap.add_argument("--no-continuum-fix", action="store_true")
    ap.add_argument("--vmacro", type=float, nargs=2, default=(5.0, 6.0))
    ap.add_argument("--phot-only", action="store_true")
    ap.add_argument("--spec-only", action="store_true")
    ap.add_argument("--maxiter", type=int, default=200)
    ap.add_argument("--surrogate-tol", type=float, default=0.01, help="abort if surrogate vs exact max error exceeds this")
    ap.add_argument("--n-epochs", type=int, default=None, help="use only the first N spectral epochs (smoke tests)")
    ap.add_argument("--windows", type=float, nargs="+", default=None, metavar="A",
                    help="keep only these windows (lo hi lo hi ...) of the HARPS file; operators are built for them alone")
    ap.add_argument("--mask", type=float, nargs="+", default=[], metavar="A",
                    help="wavelength ranges (lo hi lo hi ...) excluded from the spectroscopic term, e.g. the LTE H-alpha core")
    ap.add_argument("--skip-validation", action="store_true")
    args = ap.parse_args()
    print("devices:", jax.devices(), flush=True)

    emu = GI.make_emulator()
    LC = pickle.load(open(args.lc_meshes, "rb"))
    SP = pickle.load(open(args.spec_meshes, "rb"))
    H = np.load(args.harps, allow_pickle=True)
    names = LC["parameter_names"]; iT, iF = names.index("marcs_teff"), names.index("feh")
    assert SP["parameter_names"] == names
    assert np.allclose(np.asarray(H["times"]), np.asarray(SP["times"])), "HARPS windows file does not match spectra meshes"
    H = {k: np.asarray(H[k]) for k in H.files}
    if args.n_epochs:
        SP["models"] = SP["models"][:args.n_epochs]; SP["times"] = SP["times"][:args.n_epochs]
        H = {k: (v[:args.n_epochs] if v.ndim and v.shape[0] == len(H["times"]) and k != "wavelengths" else v) for k, v in H.items()}

    # ---- baselines: absolute parameter rows, shared by both payloads ----
    p1_0, p2_0 = GI.mean_params(SP["models"][0][0]), GI.mean_params(SP["models"][0][1])
    T1_0, T2_0, feh_0 = float(p1_0[iT]), float(p2_0[iT]), float(p1_0[iF])
    print(f"start: Teff1 {T1_0:.1f}  Teff2 {T2_0:.1f}  [Fe/H] {feh_0:+.2f}", flush=True)

    def rows(theta):                     # theta = [Teff1, Teff2, feh, dphi]
        p1 = p1_0.at[iT].set(theta[0]).at[iF].set(theta[2])
        p2 = p2_0.at[iT].set(theta[1]).at[iF].set(theta[2])
        return p1, p2

    # ---- photometry (mu-binned surrogate from tzfor_grad_inference) ----
    ph_o, mag_o = load_photometry(args.photometry)
    wl_ph = jnp.linspace(4300., 5900., 160); resp = {}
    for n in ("Stromgren:b", "Stromgren:y"):
        w, f = LC["passbands"][n]
        resp[n.split(":")[1]] = jnp.interp(wl_ph, jnp.asarray(w), jnp.asarray(f), left=0., right=0.)
    sig_ph = {"b": 0.0041, "y": 0.0035}
    lc_p1_0, lc_p2_0 = GI.mean_params(LC["models"][0][0]), GI.mean_params(LC["models"][0][1])
    loss_ph_rel, _ = GI.make_photometric_objective(emu, LC, ph_o, mag_o, resp, wl_ph, iT, iF,
                                                   n_mu=args.n_mu, weights=sig_ph)
    N_ph = sum(len(v) for v in mag_o.values())

    def chi2_phot(theta):
        rel = jnp.array([theta[0] - lc_p1_0[iT], theta[1] - lc_p2_0[iT], theta[2] - lc_p1_0[iF], theta[3]])
        return loss_ph_rel(rel) * N_ph

    # ---- spectroscopy: operators ----
    wl_out = np.asarray(H["wavelengths"])
    windows = [tuple(w) for w in np.asarray(H["windows"])] if "windows" in H else [(5160., 5200.), (6540., 6580.)]
    if args.windows:
        assert len(args.windows) % 2 == 0, "--windows takes pairs: lo hi [lo hi ...]"
        keep_w = list(zip(args.windows[::2], args.windows[1::2]))
        windows = [w for w in windows if any(abs(w[0] - lo) < 1 and abs(w[1] - hi) < 1 for lo, hi in keep_w)]
        assert windows, f"none of {keep_w} matches the file's windows"
        sel = np.zeros(wl_out.size, bool)
        for lo, hi in windows:
            sel |= (wl_out >= lo) & (wl_out <= hi)
        wl_out = wl_out[sel]; H["obs"] = H["obs"][:, sel]
        if "sigma_win" in H:
            wi = [i for i, w in enumerate(np.asarray(H["windows"])) if any(abs(w[0] - lo) < 1 for lo, hi in windows)]
            H["sigma"] = np.nanmedian(H["sigma_win"][:, wi], axis=1)
    print("windows:", windows, flush=True)
    n_out = wl_out.size
    ov = args.oversample
    wl_fine = np.concatenate([np.linspace(lo - 2, hi + 2, int((hi - lo + 4) / 0.02 * ov)) for lo, hi in windows])
    lwf = jnp.log10(wl_fine); mu_nodes = jnp.linspace(0.0, 1.0, args.n_mu)
    obs_np = np.array(H["obs"], dtype=float)
    assert len(args.mask) % 2 == 0, "--mask takes pairs: lo hi [lo hi ...]"
    for lo, hi in zip(args.mask[::2], args.mask[1::2]):
        obs_np[:, (wl_out >= lo) & (wl_out <= hi)] = np.nan
        print(f"masking {lo:.1f}-{hi:.1f} A", flush=True)
    obs = jnp.asarray(obs_np); good = jnp.isfinite(obs); obs = jnp.where(good, obs, 1.0)
    sig_sp = jnp.asarray(np.hypot(np.asarray(H["sigma"]), args.spec_floor))[:, None]
    N_sp = int(np.isfinite(obs_np).sum())
    H["obs"] = np.where(np.isfinite(obs_np), obs_np, np.nan)   # velocity CC and later uses see the mask too
    br1, br2 = make_broadener(wl_out, windows, args.vmacro[0]), make_broadener(wl_out, windows, args.vmacro[1])
    xw = jnp.concatenate([jnp.linspace(-1, 1, int(((wl_out >= lo) & (wl_out <= hi)).sum())) for lo, hi in windows])
    winid = np.concatenate([np.full(int(((wl_out >= lo) & (wl_out <= hi)).sum()), i) for i, (lo, hi) in enumerate(windows)])

    def build_all(dv_sys):
        t = time.time(); o1, o2 = [], []
        for m1, m2 in SP["models"]:
            o1.append(build_operator(m1.mus, m1.visible_cast_areas, m1.los_velocities, wl_out, wl_fine, args.n_mu, dv_sys))
            o2.append(build_operator(m2.mus, m2.visible_cast_areas, m2.los_velocities, wl_out, wl_fine, args.n_mu, dv_sys))
        print(f"built {2*len(o1)} operators (dv_sys {dv_sys:+.2f} km/s) in {time.time()-t:.0f}s, nnz max {max(o[2].size for o in o1+o2)}", flush=True)
        return stack_operators(o1), stack_operators(o2)

    def intensities(p1, p2):
        # map + checkpoint over mu nodes: differentiating the network through
        # n_mu * n_fine (~0.5-1M) wavelength points at once keeps every layer's
        # activations alive (44 GB OOM on an A100); per-node rematerialisation
        # caps that at one node's worth.
        I1 = jax.lax.map(jax.checkpoint(lambda m: emu.intensity(lwf, m, p1)), mu_nodes)
        I2 = jax.lax.map(jax.checkpoint(lambda m: emu.intensity(lwf, m, p2)), mu_nodes)
        return (I1[..., 0].reshape(-1), I1[..., 1].reshape(-1), I2[..., 0].reshape(-1), I2[..., 1].reshape(-1))

    def epoch_spectrum(op1, op2, I):
        l1, c1, l2, c2 = I
        F1, C1 = apply_op(*op1, l1, n_out), apply_op(*op1, c1, n_out)
        F2, C2 = apply_op(*op2, l2, n_out), apply_op(*op2, c2, n_out)
        return (br1(F1) + br2(F2)) / (br1(C1) + br2(C2))

    def model_spectra(theta, OPS):
        p1, p2 = rows(theta); I = intensities(p1, p2)
        O1, O2 = OPS
        return jax.lax.map(lambda o: epoch_spectrum((o[0], o[1], o[2]), (o[3], o[4], o[5]), I),
                           (O1[0], O1[1], O1[2], O2[0], O2[1], O2[2]))

    def continuum_fix(model, ob, gd):
        """Closed-form per-window linear rescale of the model: minimise sum (ob - m(a+bx))^2."""
        if args.no_continuum_fix:
            return model
        out = model
        for i in range(len(windows)):
            sel = jnp.asarray(winid == i)
            w = jnp.where(sel & gd, 1.0, 0.0)
            A = jnp.stack([model, model * xw], 1) * w[:, None]
            ATA = A.T @ A; ATb = A.T @ (ob * w)
            coef = jnp.linalg.solve(ATA + 1e-12 * jnp.eye(2), ATb)
            out = jnp.where(sel, model * (coef[0] + coef[1] * xw), out)
        return out

    def chi2_spec(theta, OPS):
        mod = model_spectra(theta, OPS)
        mod = jax.vmap(continuum_fix)(mod, obs, good)
        r = jnp.where(good, (obs - mod) / sig_sp, 0.0)
        return jnp.sum(r ** 2)

    theta0 = jnp.array([T1_0, T2_0, feh_0, 0.158])

    # ---- systemic velocity from the starting model (window 1, Mg b) ----
    OPS = build_all(0.0)
    mod0 = np.asarray(jax.jit(model_spectra)(theta0, OPS))
    sel1 = (wl_out >= windows[0][0]) & (wl_out <= windows[0][1]); dvs = []
    for e in range(mod0.shape[0]):
        dv, cc, rail = velocity_offset(wl_out[sel1], mod0[e, sel1], np.asarray(H["obs"])[e, sel1])
        dvs.append(dv)
    dvs = np.asarray(dvs); dv_sys = float(np.nanmedian(dvs))
    print(f"per-epoch velocity offset: median {dv_sys:+.2f} km/s, scatter {np.nanstd(dvs):.2f} km/s "
          f"(a phase-dependent pattern here would mean an orbit problem)", flush=True)
    if abs(dv_sys) > 0.3:
        OPS = build_all(dv_sys)

    # ---- surrogate self-check against exact per-element synthesis (epoch 0) ----
    from spice.spectrum.spectrum import simulate_observed_flux
    p1, p2 = rows(theta0); m1, m2 = SP["models"][0]
    lw_out = jnp.log10(wl_out)
    dv_used = dv_sys if abs(dv_sys) > 0.3 else 0.0
    # A systemic velocity is the same shift for every element, so apply it to
    # the sampling grid: simulate_observed_flux shifts log_wl by each element's
    # own velocity on top (PhoebeModel derives los_velocities; it is not a field).
    lw_sys = lw_out - jnp.log10(1.0 + dv_used / C_KMS)
    def exact_norm(_dv=None):
        def one(mm, row):
            mm = mm._replace(parameters=jnp.broadcast_to(row, mm.parameters.shape))
            return simulate_observed_flux(emu.intensity, mm, lw_sys, chunk_size=256)
        s1, s2 = one(m1, p1), one(m2, p2)
        return (br1(s1[:, 0]) + br2(s2[:, 0])) / (br1(s1[:, 1]) + br2(s2[:, 1]))
    t = time.time(); ex = np.asarray(jax.jit(exact_norm)()); t_ex = time.time() - t
    su = np.asarray(jax.jit(model_spectra)(theta0, OPS))[0]
    err = np.abs(su - ex); print(f"surrogate vs exact (epoch 0, normalised): max {err.max():.5f}  rms {np.sqrt((err**2).mean()):.6f}  "
                                 f"[exact synthesis of both stars took {t_ex:.1f}s incl. compile]", flush=True)
    if err.max() > args.surrogate_tol:
        print("ABORT: spectroscopic surrogate outside tolerance"); sys.exit(2)

    # ---- joint objective ----
    S = jnp.array([100., 100., 0.1, 0.001])
    def total(x):
        th = theta0 + x * S
        c_ph = 0.0 if args.spec_only else chi2_phot(th)
        c_sp = 0.0 if args.phot_only else chi2_spec(th, OPS)
        return c_ph + args.spec_weight * c_sp, (c_ph, c_sp)
    # One compiled executable for value, gradient AND the two chi2 parts: a
    # second jitted function of the same size ran the A100 out of memory
    # ("Failed to load in-memory CUBIN ... CUDA_ERROR_OUT_OF_MEMORY").
    vg = jax.jit(jax.value_and_grad(total, has_aux=True))
    parts_cache = {}
    def parts(x):
        (_, aux), _ = vg(jnp.asarray(x)); return aux
    t = time.time(); (v0, _), g0 = vg(jnp.zeros(4)); jax.block_until_ready(g0); print(f"first value+grad (compile) {time.time()-t:.0f}s", flush=True)
    t = time.time(); jax.block_until_ready(vg(jnp.zeros(4))[1]); print(f"warm value+grad {time.time()-t:.2f}s", flush=True)
    c0 = parts(jnp.zeros(4)); print(f"start: chi2_phot/N {float(c0[0])/N_ph:.3f}  chi2_spec/N {float(c0[1])/N_sp:.3f}", flush=True)

    hist = []
    def f(x):
        (v, _), g = vg(jnp.asarray(x)); hist.append(float(v)); return float(v), np.asarray(g, dtype=float)
    lo = [(3800 - T1_0) / 100, (3800 - T2_0) / 100, (-1.5 - feh_0) / 0.1, -0.05 / 0.001]
    hi = [(6900 - T1_0) / 100, (6900 - T2_0) / 100, (0.5 - feh_0) / 0.1, 0.35 / 0.001]
    x0 = np.zeros(4)
    if args.spec_only:                  # dphi is unconstrained by spectra: pin it
        lo[3] = hi[3] = 0.0
    t = time.time()
    r = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi)), options=dict(maxiter=args.maxiter))
    print(f"\noptimisation: {r.nit} iterations, {len(hist)} evaluations, {time.time()-t:.0f}s, success={r.success}, {r.message}", flush=True)
    th = np.asarray(theta0) + np.asarray(r.x) * np.asarray(S)
    c = parts(jnp.asarray(r.x))
    print(f"end:   chi2_phot/N {float(c[0])/N_ph:.3f}  chi2_spec/N {float(c[1])/N_sp:.3f}")
    print(f"  Teff1 = {th[0]:8.1f} K   Teff2 = {th[1]:8.1f} K   [Fe/H] = {th[2]:+.3f}   dphi = {th[3]:.4f}")

    result = dict(theta=th, x=np.asarray(r.x), S=np.asarray(S), chi2_phot=float(c[0]), chi2_spec=float(c[1]),
                  N_ph=N_ph, N_sp=N_sp, dv_sys=dv_sys, dv_per_epoch=dvs, hist=hist, args=vars(args))
    with open(args.out, "wb") as fh:
        pickle.dump(result, fh, protocol=4)          # the fit is safe even if the Hessian below OOMs
    try:
        Hm = np.asarray(jax.hessian(lambda x: total(x)[0])(jnp.asarray(r.x))); ev = np.linalg.eigvalsh(Hm)
        print(f"Hessian eigenvalues (scaled units): {np.array2string(ev, precision=3)}")
        result["hessian"] = Hm
    except Exception as e:                            # noqa: BLE001
        print(f"Hessian failed: {type(e).__name__}: {str(e)[:200]}"); ev = np.array([-1.0])
    if np.all(ev > 0):
        cov = 2.0 * np.linalg.inv(Hm)          # chi2 curvature -> covariance
        sig = np.sqrt(np.diag(cov)) * np.asarray(S)
        Cc = cov / np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))
        print(f"  1-sigma (formal, chi2+1): Teff1 +-{sig[0]:.0f}  Teff2 +-{sig[1]:.0f}  [Fe/H] +-{sig[2]:.3f}  dphi +-{sig[3]:.5f}")
        print(f"  corr(Teff1,Teff2) = {Cc[0,1]:+.3f}   corr(Teff2,[Fe/H]) = {Cc[1,2]:+.3f}   corr(Teff1,[Fe/H]) = {Cc[0,2]:+.3f}")
        result.update(cov=cov, sigma=sig, corr=Cc)

    # ---- validate the best fit against exact synthesis (3 epochs) ----
    p1, p2 = rows(jnp.asarray(th)); su_all = np.asarray(jax.jit(model_spectra)(jnp.asarray(th), OPS))
    for e in ([] if args.skip_validation else sorted({0, len(SP["models"]) // 2, len(SP["models"]) - 1})):
        m1, m2 = SP["models"][e]
        def one(mm, row):
            mm = mm._replace(parameters=jnp.broadcast_to(row, mm.parameters.shape))
            return simulate_observed_flux(emu.intensity, mm, lw_sys, chunk_size=256)
        s1, s2 = one(m1, p1), one(m2, p2)
        exn = np.asarray((br1(s1[:, 0]) + br2(s2[:, 0])) / (br1(s1[:, 1]) + br2(s2[:, 1])))
        d = np.abs(su_all[e] - exn); print(f"best fit, epoch {e:2d}: surrogate vs exact max {d.max():.5f} rms {np.sqrt((d**2).mean()):.6f}")
    result["model_spectra"] = su_all; result["wavelengths"] = wl_out
    with open(args.out, "wb") as fh:
        pickle.dump(result, fh, protocol=4)
    print("saved", args.out)


if __name__ == "__main__":
    main()
