"""Model curves at the final joint solution, for the paper figures (GPU job).

Outputs (npz): the blended, broadened, continuum-corrected model spectrum and
the HARPS data for one out-of-eclipse epoch in every kept window, and the
b, y model light curves at the 56 photometric epochs with the Clausen data.
Same machinery as tzfor_kernel_fit.py (SynthesisKernel, (mu, log g) nodes,
converged 16000-point passband grid).
"""
import argparse, math, os, pickle, sys
from pathlib import Path
os.environ.setdefault("JAX_ENABLE_X64", "1")
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, jax, jax.numpy as jnp
import tzfor_grad_inference as GI, tzfor_constants as K
from tzfor_kernel_fit import broadener, load_photometry, C_KMS
from spice.spectrum.synthesis_kernel import build_synthesis_kernel, gravity_darkened_rows, kernel_flux_multi

ap = argparse.ArgumentParser()
ap.add_argument("--result", default=str(HERE / "tzfor_aemu_out" / "kernel_fit_result_phot16k.pkl"))
ap.add_argument("--lc-meshes", default=str(HERE / "tzfor_aemu_out" / "tzfor_aemu_roche_n1200_lc6650_meshes.pkl"))
ap.add_argument("--spec-meshes", default=str(HERE / "tzfor_aemu_out" / "tzfor_aemu_roche_n1500_spectra_meshes.pkl"))
ap.add_argument("--harps", default=str(HERE / "tzfor_aemu_out" / "harps_windows_log.npz"))
ap.add_argument("--photometry", default=str(HERE / "tzfor_lightcurve.csv"))
ap.add_argument("--epoch", type=int, default=12, help="spectral epoch to export (12 = phase 0.25)")
ap.add_argument("--line-mask", default=None, help="line_mask npz: masked pixels are exported for shading")
ap.add_argument("--out", default=str(HERE / "tzfor_aemu_out" / "paper_models.npz"))
args = ap.parse_args()
print("devices:", jax.devices(), flush=True)

R = pickle.load(open(args.result, "rb")); th = np.asarray(R["theta"]); A = R["args"]
emu = GI.make_emulator()
LC = pickle.load(open(args.lc_meshes, "rb")); SP = pickle.load(open(args.spec_meshes, "rb"))
H = np.load(args.harps, allow_pickle=True); H = {k: H[k] for k in H.files}
names = LC["parameter_names"]; iT, iG, iF = names.index("marcs_teff"), names.index("marcs_logg"), names.index("feh")
all_windows = [tuple(w) for w in H["windows"]]
keep = [i for i, w in enumerate(all_windows) if any(abs(w[0] - kw[0]) < 1 for kw in R["windows"])]
gn1, gn2 = (jnp.asarray(g) for g in R["gnodes"]); gref1, gref2 = R["logg_ref"]
def base(models, s):
    m0 = models[0][s]; vis = np.asarray(m0.mus) > 0; a = np.asarray(m0.visible_cast_areas)[vis]
    return jnp.asarray(np.average(np.asarray(m0.parameters)[vis], axis=0, weights=a))
b1, b2 = base(SP["models"], 0), base(SP["models"], 1)
r1 = gravity_darkened_rows(b1.at[iF].set(th[2]), iT, iG, gn1, th[0], gref1, th[3])
r2 = gravity_darkened_rows(b2.at[iF].set(th[2]), iT, iG, gn2, th[1], gref2, th[4])
dlog = float(H["dlog"]); shift_sys = math.log10(1.0 + A["dv_sys"] / C_KMS)
br1, br2 = broadener(dlog, A["vmacro"][0]), broadener(dlog, A["vmacro"][1])

out = {"theta": th, "epoch": args.epoch, "phase": ((SP["times"][args.epoch] - K.T_P_HJD) % K.PERIOD_DAYS) / K.PERIOD_DAYS}
LM = np.load(args.line_mask, allow_pickle=True) if args.line_mask else None
EXTRA = R.get("extra", [])
if EXTRA:
    iX = [names.index(k) for k in EXTRA]
    b1 = b1.at[jnp.asarray(iX[:5])].set(jnp.asarray(th[6:11])); b2 = b2.at[iX[5]].set(th[11])
    r1 = gravity_darkened_rows(b1.at[iF].set(th[2]), iT, iG, gn1, th[0], gref1, th[3]); r2 = gravity_darkened_rows(b2.at[iF].set(th[2]), iT, iG, gn2, th[1], gref2, th[4])
e = args.epoch; m1, m2 = SP["models"][e]
for wi in keep:
    lw = jnp.asarray(H[f"logwl_{wi}"] - shift_sys)
    k1 = build_synthesis_kernel(m1, lw, A["n_mu"], A["oversample"], element_coordinate=jnp.asarray(m1.parameters)[:, iG], coordinate_nodes=gn1)
    k2 = build_synthesis_kernel(m2, lw, A["n_mu"], A["oversample"], element_coordinate=jnp.asarray(m2.parameters)[:, iG], coordinate_nodes=gn2)
    s1 = kernel_flux_multi(emu.intensity, [k1], r1)[0]; s2 = kernel_flux_multi(emu.intensity, [k2], r2)[0]
    mod = np.asarray((br1(s1[:, 0]) + br2(s2[:, 0])) / (br1(s1[:, 1]) + br2(s2[:, 1])))
    ob = np.asarray(H[f"obs_{wi}"][e], float); g = np.isfinite(ob); x = np.linspace(-1, 1, ob.size)
    msk = np.asarray(LM[f"mask_{wi}"][e], bool) if (LM is not None and f"mask_{wi}" in LM.files) else np.zeros(ob.size, bool)
    order = int(A.get("continuum_order", 1)); basis = np.stack([x ** k for k in range(order + 1)], 1)
    Am = (mod[:, None] * basis)[g & ~msk]; coef = np.linalg.lstsq(Am, ob[g & ~msk], rcond=None)[0]; modc = mod * (basis @ coef)
    out[f"mask_{wi}"] = msk
    # component contributions (continuum-normalised to the blend) for the figure
    c1 = np.asarray(br1(s1[:, 1])); c2 = np.asarray(br2(s2[:, 1])); f1 = np.asarray(br1(s1[:, 0])); f2 = np.asarray(br2(s2[:, 0]))
    out[f"wl_{wi}"] = 10.0 ** np.asarray(H[f"logwl_{wi}"]); out[f"obs_{wi}"] = ob; out[f"model_{wi}"] = modc
    out[f"primary_{wi}"] = f1 / (c1 + c2) * (coef[0] + coef[1] * x); out[f"secondary_{wi}"] = f2 / (c1 + c2) * (coef[0] + coef[1] * x)
    print(f"window {all_windows[wi][0]:.0f}: rms residual {100 * np.sqrt(np.mean((ob[g] - modc[g]) ** 2)):.2f}% (unmasked pixels {100 * np.sqrt(np.mean((ob - modc)[g & ~msk] ** 2)):.2f}%)", flush=True)
out["windows"] = np.array([all_windows[i] for i in keep])

# light curves
lw_ph = jnp.linspace(math.log10(4300.0), math.log10(5900.0), A["n_phot_wl"]); wl_ph = 10.0 ** lw_ph
resp = {n.split(":")[1]: jnp.interp(wl_ph, jnp.asarray(LC["passbands"][n][0]), jnp.asarray(LC["passbands"][n][1]), left=0., right=0.) for n in ("Stromgren:b", "Stromgren:y")}
bl1, bl2 = base(LC["models"], 0), base(LC["models"], 1)
q1 = gravity_darkened_rows(bl1.at[iF].set(th[2]), iT, iG, gn1, th[0], gref1, th[3]); q2 = gravity_darkened_rows(bl2.at[iF].set(th[2]), iT, iG, gn2, th[1], gref2, th[4])
pk1 = [build_synthesis_kernel(m, lw_ph, A["n_mu"], 1, element_coordinate=jnp.asarray(m.parameters)[:, iG], coordinate_nodes=gn1) for m, _ in LC["models"]]
pk2 = [build_synthesis_kernel(m, lw_ph, A["n_mu"], 1, element_coordinate=jnp.asarray(m.parameters)[:, iG], coordinate_nodes=gn2) for _, m in LC["models"]]
flux = kernel_flux_multi(emu.intensity, pk1, q1)[..., 0] + kernel_flux_multi(emu.intensity, pk2, q2)[..., 0]
ph_o, mag_o = load_photometry(args.photometry)
model_phase = ((np.asarray(LC["times"]) - K.T_P_HJD) % K.PERIOD_DAYS) / K.PERIOD_DAYS; order = np.argsort(model_phase)
out["lc_model_phase"] = model_phase[order]; out["obs_phase"] = np.asarray(ph_o)
for b, r in resp.items():
    mags = np.asarray(jax.vmap(lambda f: GI.passband_mag(f, wl_ph, r))(flux))[order]; mags = mags - np.median(mags)
    mi = np.interp((np.asarray(ph_o) - th[5]) % 1.0, model_phase[order], mags, period=1.0)
    res = np.asarray(mag_o[b]) - mi; zp = np.median(res)
    out[f"lc_model_{b}"] = mags + zp; out[f"lc_obs_{b}"] = np.asarray(mag_o[b]); out[f"lc_resid_{b}"] = res - zp
    print(f"{b}: residual rms {1000 * np.std(res - zp):.2f} mmag", flush=True)
out["dphi"] = th[5]
np.savez(args.out, **out); print("saved", args.out)
