"""Learn a line mask from the phase-independent part of the spectral residuals.

At the joint-fit solution the residual HARPS - model is 2-5 % rms and the same
at every epoch. A temperature or metallicity error would move *all* lines
coherently; an emulator line-list error is a feature fixed in the REST FRAME of
one component. So: shift each epoch's residual into each star's rest frame
(its orbital velocity from the mesh), stack over the 21 epochs, and flag pixels
whose median residual is significant. Flagged features are dilated and mapped
back into the observed frame of every epoch as a per-epoch boolean mask.

GPU job (model spectra at all epochs through SynthesisKernel); the stacking is
numpy. Writes line_mask.npz (per window: mask (n_epoch, n_pix), True = drop)
plus diagnostics.
"""
import argparse, math, os, pickle, sys
from pathlib import Path
os.environ.setdefault("JAX_ENABLE_X64", "1")
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, jax, jax.numpy as jnp
import tzfor_grad_inference as GI, tzfor_constants as K
from tzfor_kernel_fit import broadener, C_KMS
from spice.spectrum.synthesis_kernel import build_synthesis_kernel, gravity_darkened_rows, kernel_flux_multi

ap = argparse.ArgumentParser()
ap.add_argument("--result", default=str(HERE / "tzfor_aemu_out" / "kernel_fit_result_phot16k.pkl"))
ap.add_argument("--spec-meshes", default=str(HERE / "tzfor_aemu_out" / "tzfor_aemu_roche_n1500_spectra_meshes.pkl"))
ap.add_argument("--harps", default=str(HERE / "tzfor_aemu_out" / "harps_windows_log.npz"))
ap.add_argument("--threshold", type=float, default=0.01, help="|median rest-frame residual| above which a pixel is a bad line")
ap.add_argument("--nsigma", type=float, default=4.0, help="and above nsigma x (MAD / sqrt(n_epoch))")
ap.add_argument("--dilate", type=int, default=3, help="pixels added on each side of a flagged feature")
ap.add_argument("--out", default=str(HERE / "tzfor_aemu_out" / "line_mask.npz"))
args = ap.parse_args()
print("devices:", jax.devices(), flush=True)

R = pickle.load(open(args.result, "rb")); th = np.asarray(R["theta"]); A = R["args"]
emu = GI.make_emulator(); SP = pickle.load(open(args.spec_meshes, "rb"))
H = np.load(args.harps, allow_pickle=True); H = {k: H[k] for k in H.files}
names = SP["parameter_names"]; iT, iG, iF = names.index("marcs_teff"), names.index("marcs_logg"), names.index("feh")
all_windows = [tuple(w) for w in H["windows"]]
keep = [i for i, w in enumerate(all_windows) if any(abs(w[0] - kw[0]) < 1 for kw in R["windows"])]
gn1, gn2 = (jnp.asarray(g) for g in R["gnodes"]); gref1, gref2 = R["logg_ref"]
def base(s):
    m0 = SP["models"][0][s]; vis = np.asarray(m0.mus) > 0; a = np.asarray(m0.visible_cast_areas)[vis]
    return jnp.asarray(np.average(np.asarray(m0.parameters)[vis], axis=0, weights=a))
r1 = gravity_darkened_rows(base(0).at[iF].set(th[2]), iT, iG, gn1, th[0], gref1, th[3])
r2 = gravity_darkened_rows(base(1).at[iF].set(th[2]), iT, iG, gn2, th[1], gref2, th[4])
dlog = float(H["dlog"]); shift_sys = math.log10(1.0 + A["dv_sys"] / C_KMS)
br1, br2 = broadener(dlog, A["vmacro"][0]), broadener(dlog, A["vmacro"][1])
n_ep = len(SP["models"])
# per-epoch component velocities: area-weighted mean LOS velocity of the visible disc (+ systemic)
def rv(m):
    vis = np.asarray(m.mus) > 0; a = np.asarray(m.visible_cast_areas)[vis]
    return float(np.average(np.asarray(m.los_velocities)[vis], weights=a)) + A["dv_sys"]
rv1 = np.array([rv(m1) for m1, _ in SP["models"]]); rv2 = np.array([rv(m2) for _, m2 in SP["models"]])
print(f"component RVs: primary {rv1.min():+.1f}..{rv1.max():+.1f}  secondary {rv2.min():+.1f}..{rv2.max():+.1f} km/s", flush=True)

out = {"windows": np.array([all_windows[i] for i in keep]), "rv1": rv1, "rv2": rv2, "threshold": args.threshold, "nsigma": args.nsigma, "dilate": args.dilate}
tot_px = tot_masked = 0; rms_before = []; rms_after = []
for wi in keep:
    lw_obs = np.asarray(H[f"logwl_{wi}"]); lw = jnp.asarray(lw_obs - shift_sys)
    k1 = [build_synthesis_kernel(m1, lw, A["n_mu"], A["oversample"], element_coordinate=jnp.asarray(m1.parameters)[:, iG], coordinate_nodes=gn1) for m1, _ in SP["models"]]
    k2 = [build_synthesis_kernel(m2, lw, A["n_mu"], A["oversample"], element_coordinate=jnp.asarray(m2.parameters)[:, iG], coordinate_nodes=gn2) for _, m2 in SP["models"]]
    s1 = kernel_flux_multi(emu.intensity, k1, r1); s2 = kernel_flux_multi(emu.intensity, k2, r2)
    mod = np.asarray((br1(s1[..., 0]) + br2(s2[..., 0])) / (br1(s1[..., 1]) + br2(s2[..., 1])))      # (n_ep, n)
    obs = np.array(H[f"obs_{wi}"], float); n = obs.shape[1]; x = np.linspace(-1, 1, n)
    res = np.full_like(obs, np.nan)
    for e in range(n_ep):
        g = np.isfinite(obs[e]); Am = np.stack([mod[e], mod[e] * x], 1)[g]; c = np.linalg.lstsq(Am, obs[e][g], rcond=None)[0]
        res[e] = obs[e] - mod[e] * (c[0] + c[1] * x)
    # rest-frame stacks: a feature at observed log-lambda x sits at x - log10(1+v/c) in the star's rest frame
    def stack(rvs):
        S = np.full_like(res, np.nan)
        for e in range(n_ep):
            sh = math.log10(1.0 + rvs[e] / C_KMS)
            S[e] = np.interp(lw_obs + sh, lw_obs, res[e], left=np.nan, right=np.nan)   # residual sampled at rest-frame pixels
        med = np.nanmedian(S, 0); mad = 1.4826 * np.nanmedian(np.abs(S - med), 0); cnt = np.isfinite(S).sum(0)
        bad = (np.abs(med) > args.threshold) & (np.abs(med) > args.nsigma * mad / np.sqrt(np.maximum(cnt, 1)))
        # dilate
        idx = np.where(bad)[0]; B = np.zeros(n, bool)
        for i in idx: B[max(0, i - args.dilate):i + args.dilate + 1] = True
        return B, med
    B1, med1 = stack(rv1); B2, med2 = stack(rv2)
    # map rest-frame bad pixels back to each epoch's observed frame
    mask = np.zeros_like(obs, bool)
    for e in range(n_ep):
        for B, rvs in ((B1, rv1), (B2, rv2)):
            sh = math.log10(1.0 + rvs[e] / C_KMS)
            mask[e] |= np.interp(lw_obs - sh, lw_obs, B.astype(float), left=0, right=0) > 0.5
    g = np.isfinite(obs); rb = np.sqrt(np.nanmean(res[g] ** 2)); ra = np.sqrt(np.nanmean(res[g & ~mask] ** 2))
    tot_px += g.sum(); tot_masked += (g & mask).sum(); rms_before.append(rb); rms_after.append(ra)
    out[f"mask_{wi}"] = mask; out[f"med1_{wi}"] = med1; out[f"med2_{wi}"] = med2; out[f"res_{wi}"] = res
    print(f"window {all_windows[wi][0]:.0f}: bad rest-frame pixels primary {B1.sum():4d} secondary {B2.sum():4d} | masked {100*(g & mask).sum()/g.sum():5.1f}% of pixels | residual rms {100*rb:.2f}% -> {100*ra:.2f}%", flush=True)
print(f"\ntotal masked {100*tot_masked/tot_px:.1f}% of pixels; mean residual rms {100*np.mean(rms_before):.2f}% -> {100*np.mean(rms_after):.2f}%", flush=True)
np.savez(args.out, **out); print("saved", args.out)
