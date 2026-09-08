"""NUTS posterior for the full-range fit (tzfor_fullrange_fit.py) on a frozen grid.

Same idea as tzfor_nuts.py: precompute each star's broadened disc spectra over
the whole range at every epoch on a (Teff, [Fe/H]) grid, with vmacro, the vsini
scales and the delta map held at the fit's optimum; cubic interpolation; spline
continuum per epoch as in the fit; one noise-inflation factor per --n-blocks
contiguous wavelength blocks and per band; numpyro NUTS.
"""
import argparse, math, os, pickle, sys, time
from pathlib import Path
os.environ.setdefault("JAX_ENABLE_X64", "1")
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, jax, jax.numpy as jnp
from scipy.interpolate import BSpline
import tzfor_grad_inference as GI, tzfor_constants as K
from tzfor_kernel_fit import load_photometry, C_KMS, R_HARPS
from tzfor_fullrange_fit import Geom
from spice.spectrum.synthesis_kernel import build_synthesis_kernel, gravity_darkened_rows, kernel_flux_multi

ap = argparse.ArgumentParser()
ap.add_argument("--result", required=True); ap.add_argument("--harps", default=str(HERE / "tzfor_aemu_out" / "harps_fullrange_log.npz"))
ap.add_argument("--spec-meshes", default=str(HERE / "tzfor_aemu_out" / "tzfor_aemu_roche_n1500_spectra_meshes.pkl"))
ap.add_argument("--lc-meshes", default=str(HERE / "tzfor_aemu_out" / "tzfor_aemu_roche_n1200_lc6650_meshes.pkl"))
ap.add_argument("--photometry", default=str(HERE / "tzfor_lightcurve.csv")); ap.add_argument("--mask-in", default=None)
ap.add_argument("--teff-halfwidth", type=float, default=200.0); ap.add_argument("--teff-step", type=float, default=25.0)
ap.add_argument("--feh", type=float, nargs=3, default=(-0.5, 0.1, 0.1)); ap.add_argument("--n-blocks", type=int, default=6)
ap.add_argument("--num-warmup", type=int, default=400); ap.add_argument("--num-samples", type=int, default=1500); ap.add_argument("--chains", type=int, default=4)
ap.add_argument("--out", required=True)
args = ap.parse_args()
print("devices:", jax.devices(), flush=True)

R = pickle.load(open(args.result, "rb")); A = R["args"]; pn = R["pnames"]; th = np.asarray(R["theta"]); Pi = {nm: i for i, nm in enumerate(pn)}
emu = GI.make_emulator(); SP = pickle.load(open(args.spec_meshes, "rb")); LC = pickle.load(open(args.lc_meshes, "rb"))
H = np.load(args.harps, allow_pickle=True); H = {k: H[k] for k in H.files}
names = SP["parameter_names"]; iT, iG, iF = names.index("marcs_teff"), names.index("marcs_logg"), names.index("feh")
n_ep, n_lc = len(SP["models"]), len(LC["models"]); lw_obs = np.asarray(H["logwl"], float); n = lw_obs.size; wl = 10.0 ** lw_obs; dlog = float(H["dlog"])
shift_sys = math.log10(1.0 + A["dv_sys"] / C_KMS); lw_model = jnp.asarray(lw_obs - shift_sys); beta = R["beta"]
delta = jnp.asarray(R["delta"]) if R.get("delta") is not None else jnp.zeros(n)
vm = [th[Pi["vmac1"]], th[Pi["vmac2"]]] if "vmac1" in Pi else list(A["vmacro0"]); vs = [th[Pi["vsini_scale1"]], th[Pi["vsini_scale2"]]] if "vsini_scale1" in Pi else [1.0, 1.0]
print(f"optimum: {dict(zip(pn, np.round(th, 4)))}; delta {'fitted' if R.get('delta') is not None else 'none'}", flush=True)

def star_setup(models, s):
    m0 = models[0][s]; vis = np.asarray(m0.mus) > 0; a = np.asarray(m0.visible_cast_areas)[vis]; p = np.asarray(m0.parameters)[vis]
    allg = np.concatenate([np.asarray(m.parameters)[np.asarray(m.mus) > 0, iG] for m in (pair[s] for pair in models)])
    return jnp.asarray(np.average(p, axis=0, weights=a)), float(np.average(p[:, iG], weights=a)), jnp.asarray(np.linspace(allg.min() - 0.005, allg.max() + 0.005, A["n_g"]))
base, gref, gn = zip(*[star_setup(SP["models"], s) for s in (0, 1)]); base_lc = [star_setup(LC["models"], s)[0] for s in (0, 1)]
def elems(m):
    a = np.asarray(m.visible_cast_areas); mu = np.asarray(m.mus); v = np.asarray(m.los_velocities); g = np.asarray(m.parameters)[:, iG]
    vis = (mu > 0) & (a > 0); v_orb = float(np.average(v[vis], weights=a[vis])); return jnp.asarray(a), jnp.asarray(mu), jnp.asarray(v), jnp.asarray(g), v_orb
dfine = dlog / A["oversample"]
ks = []
for s in (0, 1):
    E = [elems(pair[s]) for pair in SP["models"]]
    hw = int(math.ceil(math.log10(1.0 + (max(abs(e[4]) for e in E) + 2.5 * max(float(jnp.max(jnp.abs(e[2] - e[4]))) for e in E) + 3.0) / C_KMS) / dfine)) + 1
    ks.append([build_synthesis_kernel(Geom(a, mu, v_orb + vs[s] * (v - v_orb)), lw_model, A["n_mu"], A["oversample"], half_width=hw, element_coordinate=g, coordinate_nodes=gn[s]) for a, mu, v, g, v_orb in E])
lw_ph = jnp.linspace(math.log10(4300.0), math.log10(5900.0), A["n_phot_wl"]); wl_ph = 10.0 ** lw_ph
ph_k = [[build_synthesis_kernel(pair[s], lw_ph, A["n_mu"], 1, element_coordinate=jnp.asarray(pair[s].parameters)[:, iG], coordinate_nodes=gn[s]) for pair in LC["models"]] for s in (0, 1)]
resp = {nm.split(":")[1]: jnp.interp(wl_ph, jnp.asarray(LC["passbands"][nm][0]), jnp.asarray(LC["passbands"][nm][1]), left=0., right=0.) for nm in ("Stromgren:b", "Stromgren:y")}

HALF_BR = 60; xb = jnp.arange(-HALF_BR, HALF_BR + 1)
def broaden(y, vmacro):
    sp = jnp.hypot(C_KMS / R_HARPS, vmacro) / (C_KMS * math.log(10.0) * dlog) / 2.3548; k = jnp.exp(-0.5 * (xb / sp) ** 2); k = k / k.sum()
    yp = jnp.pad(y, [(0, 0)] * (y.ndim - 1) + [(HALF_BR, HALF_BR)], mode="edge")
    return jax.vmap(lambda r: jnp.convolve(r, k, mode="valid"))(yp.reshape(-1, yp.shape[-1])).reshape(y.shape)
def int_delta(x, mu, prow):
    out = emu.intensity(x, mu, prow); return out.at[:, 0].multiply(1.0 + jnp.interp(x, lw_model, delta))

obs_np = np.array(H["obs"], float); good_np = np.isfinite(obs_np)
if args.mask_in: good_np &= ~np.asarray(np.load(args.mask_in, allow_pickle=True)["mask"], bool)[:n_ep]
edges = np.asarray(H["block_edges"]); blk = np.clip(np.searchsorted(edges, wl) - 1, 0, edges.size - 2)
sig = jnp.asarray(np.hypot(np.nan_to_num(H["sigma_blocks"][:n_ep, blk], nan=0.01), A["spec_floor"])); obs = jnp.asarray(np.where(good_np, obs_np, 1.0)); good = jnp.asarray(good_np)
jblk = jnp.asarray(np.minimum(blk * args.n_blocks // (edges.size - 1), args.n_blocks - 1))
kn = np.arange(lw_obs[0], lw_obs[-1] + 1e-12, math.log10(1 + A["knot_spacing"] / 5500.0)); t_kn = np.concatenate([[lw_obs[0]] * 3, kn, [lw_obs[-1]] * 3])
Bmat = jnp.asarray(BSpline.design_matrix(lw_obs, t_kn, 3).toarray()); nB = Bmat.shape[1]
ph_o, mag_o = load_photometry(args.photometry); sig_ph = {"b": 0.0041, "y": 0.0035}
model_phase = ((np.asarray(LC["times"]) - K.T_P_HJD) % K.PERIOD_DAYS) / K.PERIOD_DAYS; order = np.argsort(model_phase); mph = jnp.asarray(model_phase[order])

# ---- frozen grid ----
T_nodes = [th[Pi[f"Teff{s+1}"]] + np.arange(-args.teff_halfwidth, args.teff_halfwidth + 1e-6, args.teff_step) for s in (0, 1)]
F_nodes = np.arange(args.feh[0], args.feh[1] + 1e-9, args.feh[2]); nT, nF = len(T_nodes[0]), len(F_nodes)
print(f"grid: {nT} Teff x {nF} [Fe/H] per star; spectra {n_ep} x {n} px float32 = {nT * nF * n_ep * n * 2 * 4 / 2**30:.1f} GiB per star", flush=True)
def star_curves(s, T, f):
    r = gravity_darkened_rows(base[s].at[iF].set(f), iT, iG, gn[s], T, gref[s], beta)
    sp = kernel_flux_multi(int_delta if s == 0 else emu.intensity, ks[s], r)
    spec = jnp.stack([broaden(sp[..., 0], vm[s]), broaden(sp[..., 1], vm[s])], -1).astype(jnp.float32)
    rl = gravity_darkened_rows(base_lc[s].at[iF].set(f), iT, iG, gn[s], T, gref[s], beta)
    flux = kernel_flux_multi(emu.intensity, ph_k[s], rl)[..., 0]
    return spec, jnp.stack([jax.vmap(lambda fl: jnp.trapezoid(fl * r_, wl_ph))(flux) for r_ in resp.values()], 0)
star_curves_j = jax.jit(star_curves, static_argnums=(0,))
GS = [np.zeros((nT, nF, n_ep, n, 2), np.float32) for s in (0, 1)]; GB = [np.zeros((nT, nF, 2, n_lc)) for s in (0, 1)]
t0 = time.time()
for s in (0, 1):
    for i, T in enumerate(T_nodes[s]):
        for j, f in enumerate(F_nodes):
            sp, bd = star_curves_j(s, float(T), float(f)); GS[s][i, j] = np.asarray(sp); GB[s][i, j] = np.asarray(bd)
    print(f"star {s + 1} grid done ({time.time() - t0:.0f}s)", flush=True)
GS = [jnp.asarray(g) for g in GS]; GB = [jnp.asarray(g) for g in GB]

def catmull_rom(u): return jnp.stack([-0.5 * u ** 3 + u ** 2 - 0.5 * u, 1.5 * u ** 3 - 2.5 * u ** 2 + 1.0, -1.5 * u ** 3 + 2.0 * u ** 2 + 0.5 * u, 0.5 * u ** 3 - 0.5 * u ** 2])
def locate(nodes, value):
    x = (value - nodes[0]) / (nodes[1] - nodes[0]); i = jnp.clip(jnp.floor(x), 1, len(nodes) - 3).astype(jnp.int32); return i - 1, catmull_rom(x - i)
def interp2(grid, i0, wT, j0, wF):
    return sum(wT[a] * sum(wF[b] * grid[i0 + a, j0 + b] for b in range(4)) for a in range(4))
def star_model(s, T, f):
    i0, wT = locate(jnp.asarray(T_nodes[s]), T); j0, wF = locate(jnp.asarray(F_nodes), f); return interp2(GS[s], i0, wT, j0, wF), interp2(GB[s], i0, wT, j0, wF)
def continuum_fix(model, ob, gd):
    w = jnp.where(gd, 1.0, 0.0); Am = model[:, None] * Bmat * w[:, None]; coef = jnp.linalg.solve(Am.T @ Am + 1e-9 * jnp.eye(nB), Am.T @ (ob * w)); return model * (Bmat @ coef)
def spec_chi2_blocks(T1, T2, f):
    s1, _ = star_model(0, T1, f); s2, _ = star_model(1, T2, f); mod = (s1[..., 0] + s2[..., 0]) / (s1[..., 1] + s2[..., 1])
    mod = jax.vmap(continuum_fix)(mod, obs, good); r2 = jnp.where(good, ((obs - mod) / sig) ** 2, 0.0)
    return jax.ops.segment_sum(r2.sum(0), jblk, num_segments=args.n_blocks)
def phot_chi2(T1, T2, f, dphi):
    _, b1 = star_model(0, T1, f); _, b2 = star_model(1, T2, f); tot = []
    for k, b in enumerate(resp):
        mags = -2.5 * jnp.log10(b1[k] + b2[k])[order]; mags = mags - jnp.median(mags); mi = jnp.interp((ph_o - dphi) % 1.0, mph, mags, period=1.0)
        res = mag_o[b] - mi; res = res - jnp.median(res); tot.append(jnp.sum(res ** 2) / sig_ph[b] ** 2)
    return jnp.stack(tot)
N_blk = jax.ops.segment_sum(jnp.asarray(good_np.sum(0), float), jblk, num_segments=args.n_blocks); N_b = jnp.asarray([float(len(mag_o[b])) for b in resp])

rng = np.random.default_rng(1)
for _ in range(2):
    T1 = th[Pi["Teff1"]] + rng.uniform(-120, 120); T2 = th[Pi["Teff2"]] + rng.uniform(-120, 120); f = rng.uniform(args.feh[0] + 0.15, args.feh[1] - 0.15)
    s1, _ = star_model(0, T1, f); s2, _ = star_model(1, T2, f); d1, _ = star_curves_j(0, float(T1), float(f)); d2, _ = star_curves_j(1, float(T2), float(f))
    mi = (s1[..., 0] + s2[..., 0]) / (s1[..., 1] + s2[..., 1]); md = (d1[..., 0] + d2[..., 0]) / (d1[..., 1] + d2[..., 1])
    print(f"interpolation check ({T1:.0f}, {T2:.0f}, {f:+.2f}): max |interp - direct| = {float(jnp.max(jnp.abs(mi - md))):.5f}", flush=True)

import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
def model():
    T1 = numpyro.sample("Teff1", dist.Uniform(T_nodes[0][1], T_nodes[0][-2])); T2 = numpyro.sample("Teff2", dist.Uniform(T_nodes[1][1], T_nodes[1][-2]))
    f = numpyro.sample("feh", dist.Uniform(F_nodes[1], F_nodes[-2])); dphi = numpyro.sample("dphi", dist.Uniform(th[Pi["dphi"]] - 0.01, th[Pi["dphi"]] + 0.01))
    ls_w = numpyro.sample("log_s_spec", dist.Normal(jnp.log(2.0), 1.0).expand([args.n_blocks])); ls_b = numpyro.sample("log_s_phot", dist.Normal(0.0, 1.0).expand([2]))
    numpyro.factor("spec", jnp.sum(-0.5 * spec_chi2_blocks(T1, T2, f) * jnp.exp(-2 * ls_w) - N_blk * ls_w))
    numpyro.factor("phot", jnp.sum(-0.5 * phot_chi2(T1, T2, f, dphi) * jnp.exp(-2 * ls_b) - N_b * ls_b))
mcmc = MCMC(NUTS(model, target_accept_prob=0.85, max_tree_depth=8), num_warmup=args.num_warmup, num_samples=args.num_samples, num_chains=args.chains, chain_method="sequential", progress_bar=False)
t0 = time.time(); mcmc.run(jax.random.PRNGKey(0)); print(f"NUTS: {args.chains} x ({args.num_warmup} + {args.num_samples}) in {time.time() - t0:.0f}s", flush=True); mcmc.print_summary()
S = mcmc.get_samples(group_by_chain=True); flat = {k: np.asarray(v).reshape(-1, *np.asarray(v).shape[2:]) for k, v in S.items()}
print(f"posterior: Teff1 {flat['Teff1'].mean():.0f} +- {flat['Teff1'].std():.0f}   Teff2 {flat['Teff2'].mean():.0f} +- {flat['Teff2'].std():.0f}   [Fe/H] {flat['feh'].mean():+.3f} +- {flat['feh'].std():.3f}")
C = np.corrcoef(np.stack([flat["Teff1"], flat["Teff2"], flat["feh"]])); print(f"corr(Teff1,Teff2) {C[0,1]:+.2f}  corr(Teff2,feh) {C[1,2]:+.2f}")
print("noise inflation per block:", np.round(np.exp(flat["log_s_spec"].mean(0)), 2), " photometry:", np.round(np.exp(flat["log_s_phot"].mean(0)), 2))
np.savez(args.out, **{k: np.asarray(v) for k, v in S.items()}, theta_map=th, pnames=np.array(pn)); print("saved", args.out)
