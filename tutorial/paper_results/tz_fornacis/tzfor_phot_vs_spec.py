"""How well do photometry and spectroscopy agree on TZ For's temperatures?

Uses the mu-binned photometric surrogate of tzfor_grad_inference with a
converged passband grid (16000 points; 160 points aliased the line-rich aemu
spectrum into 3-5 mmag errors). Reports: photometric chi2 at the spectroscopic
solution, the photometry-only optimum, the photometric valley Teff2(Teff1), and
the out-of-eclipse light ratios. GPU job; ~minutes.
"""
import argparse, os, sys, pickle
from pathlib import Path
os.environ.setdefault("JAX_ENABLE_X64", "1")
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, jax, jax.numpy as jnp, pandas as pd
from scipy.optimize import minimize, minimize_scalar
import tzfor_grad_inference as GI, tzfor_constants as K

ap = argparse.ArgumentParser()
ap.add_argument("--lc-meshes", default=str(HERE / "tzfor_aemu_out" / "tzfor_aemu_roche_n1200_lc6650_meshes.pkl"))
ap.add_argument("--photometry", default=str(HERE / "tzfor_lightcurve.csv"))
ap.add_argument("--spec", type=float, nargs=4, default=(4888., 6415., -0.233, 0.1584), metavar=("T1", "T2", "FEH", "DPHI"))
ap.add_argument("--n-wl", type=int, default=16000)
ap.add_argument("--out", default=str(HERE / "tzfor_aemu_out" / "phot_vs_spec.pkl"))
args = ap.parse_args()
print("devices:", jax.devices(), flush=True)

emu = GI.make_emulator()
d = pickle.load(open(args.lc_meshes, "rb"))
names = d["parameter_names"]; iT, iF = names.index("marcs_teff"), names.index("feh")
df = pd.read_csv(args.photometry)
ph = ((df["HJD"].values + 2450000.0 - K.T_P_HJD) % K.PERIOD_DAYS) / K.PERIOD_DAYS
obs = {b: jnp.asarray(df[b].values) for b in ("b", "y")}
wl = jnp.linspace(4300., 5900., args.n_wl)
resp = {n.split(":")[1]: jnp.interp(wl, jnp.asarray(d["passbands"][n][0]), jnp.asarray(d["passbands"][n][1]), left=0., right=0.)
        for n in ("Stromgren:b", "Stromgren:y")}
loss, forward = GI.make_photometric_objective(emu, d, jnp.asarray(ph), obs, resp, wl, iT, iF, n_mu=32, weights={"b": 0.0041, "y": 0.0035})
loss = jax.jit(loss)
p1_0, p2_0 = GI.mean_params(d["models"][0][0]), GI.mean_params(d["models"][0][1])
T1_0, T2_0, F_0 = float(p1_0[iT]), float(p2_0[iT]), float(p1_0[iF])
N = sum(len(v) for v in obs.values())
def chi2(T1, T2, feh, dphi):
    return float(loss(jnp.array([T1 - T1_0, T2 - T2_0, feh - F_0, dphi])))       # chi2 / N

T1s, T2s, Fs, Ds = args.spec
c_spec = chi2(T1s, T2s, Fs, Ds)
print(f"photometry at the spectroscopic solution ({T1s:.0f}, {T2s:.0f}, {Fs:+.3f}, {Ds:.4f}): chi2/N = {c_spec:.3f}  (residual rms = {np.sqrt(c_spec):.2f} x Clausen noise)", flush=True)
r = minimize(lambda x: chi2(x[0] * 100 + T1s, x[1] * 100 + T2s, x[2] * 0.1 + Fs, x[3] * 0.001 + Ds),
             np.zeros(4), method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-7, maxiter=600))
T1p, T2p, Fp, Dp = r.x[0] * 100 + T1s, r.x[1] * 100 + T2s, r.x[2] * 0.1 + Fs, r.x[3] * 0.001 + Ds
print(f"photometry-only optimum (all four free): Teff1 {T1p:.0f} Teff2 {T2p:.0f} [Fe/H] {Fp:+.3f} dphi {Dp:.4f}  chi2/N = {r.fun:.3f}", flush=True)
print(f"  delta chi2 (total, N={N}) spectroscopic solution vs photometric optimum: {(c_spec - r.fun) * N:.1f}", flush=True)
print(f"\nphotometric valley Teff2_phot(Teff1) at [Fe/H] {Fs:+.3f}, dphi {Ds:.4f}   (spectroscopy: Teff2 = {T2s:.0f} +- 33)", flush=True)
valley = []
for T1 in (T1s - 90, T1s - 45, T1s, T1s + 45, T1s + 90):
    rr = minimize_scalar(lambda T2: chi2(T1, T2, Fs, Ds), bounds=(5800., 6900.), method="bounded", options=dict(xatol=0.3))
    h = 20.; c0 = rr.fun * N; cp = chi2(T1, rr.x + h, Fs, Ds) * N; cm = chi2(T1, rr.x - h, Fs, Ds) * N
    curv = (cp + cm - 2 * c0) / h ** 2; sig = np.sqrt(2.0 / curv) if curv > 0 else np.nan
    valley.append((T1, rr.x, sig, rr.fun))
    print(f"  Teff1 {T1:.0f}: Teff2_phot = {rr.x:.0f} +- {sig:.0f} K (formal)   chi2/N {rr.fun:.3f}", flush=True)
slope = np.polyfit([v[0] for v in valley], [v[1] for v in valley], 1)[0]
print(f"  valley slope dTeff2/dTeff1 = {slope:.2f}", flush=True)

def light_ratio(T1, T2, feh):
    nodes, W1, W2 = GI.precompute_geometry(d, 32); lw = jnp.log10(wl)
    p1 = p1_0.at[iT].set(T1).at[iF].set(feh); p2 = p2_0.at[iT].set(T2).at[iF].set(feh)
    I1 = jax.lax.map(lambda m: emu.intensity(lw, m, p1)[:, 0], nodes); I2 = jax.lax.map(lambda m: emu.intensity(lw, m, p2)[:, 0], nodes)
    e = int(np.argmin(np.abs(((np.asarray(d["times"]) - K.T_P_HJD) % K.PERIOD_DAYS) / K.PERIOD_DAYS - 0.30)))
    f1, f2 = W1[e] @ I1, W2[e] @ I2
    return {b: float(jnp.trapezoid(f2 * rr, wl) / jnp.trapezoid(f1 * rr, wl)) for b, rr in resp.items()}
lr = {"spectroscopic": light_ratio(T1s, T2s, Fs), "photometric": light_ratio(T1p, T2p, Fp), "literature": light_ratio(T1_0, T2_0, F_0)}
print("\nlight ratio L2/L1 out of eclipse (phase 0.30):")
for k, v in lr.items():
    print(f"  {k:14s}: b {v['b']:.3f}   y {v['y']:.3f}", flush=True)
pickle.dump(dict(spec=args.spec, chi2_spec=c_spec, phot_opt=(T1p, T2p, Fp, Dp, r.fun), valley=valley, slope=slope, light_ratio=lr, n_wl=args.n_wl, N=N),
            open(args.out, "wb"), protocol=4)
print("saved", args.out)
