"""NUTS posterior for TZ For on a frozen local emulator grid (GPU job).

A joint gradient through the kernel path costs ~10 s, too slow for the 10^4-10^5
gradients a sampler needs. But the emulator enters only through the
disc-integrated spectra of each star, which depend on (Teff, [Fe/H]) of that
star alone. So: precompute, for each star, the broadened line and continuum
disc spectra at every epoch and window, and the band-integrated fluxes at every
light-curve epoch, on a (Teff, [Fe/H]) grid around the optimum; interpolate
them with a Catmull-Rom cubic (smooth gradients); and sample (Teff1, Teff2,
[Fe/H], dphi) plus one noise-inflation factor per window and per band with
numpyro's NUTS. Every likelihood is then interpolation + FFT-free algebra.

The noise inflation makes the posterior honest about the chi2/N ~ 10 residual
floor; it does not model correlated line-list systematics (see the learned
line mask and the window jackknife for that). Interpolation is validated
against the direct kernel model at random points before sampling.
"""
import argparse, math, os, pickle, sys, time
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
ap.add_argument("--line-mask", default=None)
ap.add_argument("--teff-halfwidth", type=float, default=250.0)
ap.add_argument("--teff-step", type=float, default=25.0)
ap.add_argument("--feh", type=float, nargs=3, default=(-0.55, 0.05, 0.1), metavar=("LO", "HI", "STEP"))
ap.add_argument("--num-warmup", type=int, default=500)
ap.add_argument("--num-samples", type=int, default=2000)
ap.add_argument("--chains", type=int, default=4)
ap.add_argument("--spec-floor", type=float, default=0.005)
ap.add_argument("--out", default=str(HERE / "tzfor_aemu_out" / "nuts_posterior.npz"))
args = ap.parse_args()
print("devices:", jax.devices(), flush=True)

R = pickle.load(open(args.result, "rb")); th = np.asarray(R["theta"]); A = R["args"]
emu = GI.make_emulator()
LC = pickle.load(open(args.lc_meshes, "rb")); SP = pickle.load(open(args.spec_meshes, "rb"))
H = np.load(args.harps, allow_pickle=True); H = {k: H[k] for k in H.files}
names = LC["parameter_names"]; iT, iG, iF = names.index("marcs_teff"), names.index("marcs_logg"), names.index("feh")
all_windows = [tuple(w) for w in H["windows"]]
keep = [i for i, w in enumerate(all_windows) if any(abs(w[0] - kw[0]) < 1 for kw in R["windows"])]
gn = [jnp.asarray(g) for g in R["gnodes"]]; gref = list(R["logg_ref"]); beta = [th[3], th[4]]
n_ep, n_lc = len(SP["models"]), len(LC["models"])
dlog = float(H["dlog"]); shift_sys = math.log10(1.0 + A["dv_sys"] / C_KMS)
br = [broadener(dlog, A["vmacro"][0]), broadener(dlog, A["vmacro"][1])]

def base_row(models, s):
    m0 = models[0][s]; vis = np.asarray(m0.mus) > 0; a = np.asarray(m0.visible_cast_areas)[vis]
    return jnp.asarray(np.average(np.asarray(m0.parameters)[vis], axis=0, weights=a))
base_sp = [base_row(SP["models"], 0), base_row(SP["models"], 1)]
base_lc = [base_row(LC["models"], 0), base_row(LC["models"], 1)]

# ---- kernels ----
t0 = time.time()
spec_k = {wi: [[build_synthesis_kernel(pair[s], jnp.asarray(H[f"logwl_{wi}"] - shift_sys), A["n_mu"], A["oversample"],
                                       element_coordinate=jnp.asarray(pair[s].parameters)[:, iG], coordinate_nodes=gn[s]) for pair in SP["models"]]
               for s in (0, 1)] for wi in keep}
lw_ph = jnp.linspace(math.log10(4300.0), math.log10(5900.0), A["n_phot_wl"]); wl_ph = 10.0 ** lw_ph
ph_k = [[build_synthesis_kernel(pair[s], lw_ph, A["n_mu"], 1, element_coordinate=jnp.asarray(pair[s].parameters)[:, iG], coordinate_nodes=gn[s])
         for pair in LC["models"]] for s in (0, 1)]
resp = {n.split(":")[1]: jnp.interp(wl_ph, jnp.asarray(LC["passbands"][n][0]), jnp.asarray(LC["passbands"][n][1]), left=0., right=0.) for n in ("Stromgren:b", "Stromgren:y")}
print(f"kernels built in {time.time() - t0:.0f}s", flush=True)

# ---- data ----
LM = np.load(args.line_mask, allow_pickle=True) if args.line_mask else None
obs, good, sig, xw = {}, {}, {}, {}
for wi in keep:
    o = np.array(H[f"obs_{wi}"], float); g = np.isfinite(o)
    if LM is not None and f"mask_{wi}" in LM.files: g &= ~np.asarray(LM[f"mask_{wi}"], bool)
    obs[wi] = jnp.asarray(np.where(g, o, 1.0)); good[wi] = jnp.asarray(g)
    sig[wi] = jnp.asarray(np.hypot(H["sigma_win"][:, wi], args.spec_floor))[:, None]; xw[wi] = jnp.linspace(-1, 1, o.shape[1])
ph_o, mag_o = load_photometry(args.photometry); sig_ph = {"b": 0.0041, "y": 0.0035}
model_phase = ((np.asarray(LC["times"]) - K.T_P_HJD) % K.PERIOD_DAYS) / K.PERIOD_DAYS; order = np.argsort(model_phase); mph = jnp.asarray(model_phase[order])

# ---- frozen grid: per star, (Teff, feh) -> broadened disc spectra per window/epoch and band fluxes per LC epoch ----
T_nodes = [th[s] + np.arange(-args.teff_halfwidth, args.teff_halfwidth + 1e-6, args.teff_step) for s in (0, 1)]
F_nodes = np.arange(args.feh[0], args.feh[1] + 1e-9, args.feh[2])
nT, nF = len(T_nodes[0]), len(F_nodes)
print(f"grid: {nT} Teff x {nF} [Fe/H] nodes per star ({args.teff_step} K, {args.feh[2]} dex)", flush=True)

def star_curves(s, T, f):
    rows = gravity_darkened_rows(base_sp[s].at[iF].set(f), iT, iG, gn[s], T, gref[s], beta[s])
    spec = []
    for wi in keep:
        sp = kernel_flux_multi(emu.intensity, spec_k[wi][s], rows)                      # (n_ep, n, 2)
        spec.append(jnp.stack([br[s](sp[..., 0]), br[s](sp[..., 1])], -1).astype(jnp.float32))
    rows_lc = gravity_darkened_rows(base_lc[s].at[iF].set(f), iT, iG, gn[s], T, gref[s], beta[s])
    flux = kernel_flux_multi(emu.intensity, ph_k[s], rows_lc)[..., 0]                   # (n_lc, n_wl)
    band = jnp.stack([jax.vmap(lambda fl: jnp.trapezoid(fl * r, wl_ph))(flux) for r in resp.values()], 0)  # (2, n_lc)
    return spec, band
star_curves_j = jax.jit(star_curves, static_argnums=(0,))
GRID_SPEC = [[np.zeros((nT, nF, n_ep, int(H[f"logwl_{wi}"].size), 2), np.float32) for wi in keep] for s in (0, 1)]
GRID_BAND = [np.zeros((nT, nF, 2, n_lc)) for s in (0, 1)]
t0 = time.time()
for s in (0, 1):
    for i, T in enumerate(T_nodes[s]):
        for j, f in enumerate(F_nodes):
            spec, band = star_curves_j(s, float(T), float(f))
            for w, sp in enumerate(spec): GRID_SPEC[s][w][i, j] = np.asarray(sp)
            GRID_BAND[s][i, j] = np.asarray(band)
    print(f"star {s + 1} grid done ({time.time() - t0:.0f}s)", flush=True)
GRID_SPEC = [[jnp.asarray(g) for g in gs] for gs in GRID_SPEC]; GRID_BAND = [jnp.asarray(g) for g in GRID_BAND]

def catmull_rom(u):
    """Weights for the 4 nodes around fractional position u in [0,1)."""
    return jnp.stack([-0.5 * u ** 3 + u ** 2 - 0.5 * u, 1.5 * u ** 3 - 2.5 * u ** 2 + 1.0,
                      -1.5 * u ** 3 + 2.0 * u ** 2 + 0.5 * u, 0.5 * u ** 3 - 0.5 * u ** 2])

def interp2(grid, i0, wT, j0, wF):
    """Bicubic tensor interpolation of grid[i, j, ...] with 4x4 neighbours from (i0, j0)."""
    out = 0.0
    for a in range(4):
        row = 0.0
        for b in range(4):
            row = row + wF[b] * grid[i0 + a, j0 + b]
        out = out + wT[a] * row
    return out

def locate(nodes, value):
    x = (value - nodes[0]) / (nodes[1] - nodes[0]); i = jnp.clip(jnp.floor(x), 1, len(nodes) - 3).astype(jnp.int32)
    return i - 1, catmull_rom(x - i)

def star_model(s, T, f):
    i0, wT = locate(jnp.asarray(T_nodes[s]), T); j0, wF = locate(jnp.asarray(F_nodes), f)
    return [interp2(g, i0, wT, j0, wF) for g in GRID_SPEC[s]], interp2(GRID_BAND[s], i0, wT, j0, wF)

def continuum_fix(model, ob, gd, x):
    w = jnp.where(gd, 1.0, 0.0); Am = jnp.stack([model, model * x], 1) * w[:, None]
    coef = jnp.linalg.solve(Am.T @ Am + 1e-12 * jnp.eye(2), Am.T @ (ob * w)); return model * (coef[0] + coef[1] * x)

def spectra_chi2(T1, T2, f):
    s1, _ = star_model(0, T1, f); s2, _ = star_model(1, T2, f); out = []
    for w, wi in enumerate(keep):
        mod = (s1[w][..., 0] + s2[w][..., 0]) / (s1[w][..., 1] + s2[w][..., 1])
        mod = jax.vmap(lambda m, o, g: continuum_fix(m, o, g, xw[wi]))(mod, obs[wi], good[wi])
        r = jnp.where(good[wi], (obs[wi] - mod) / sig[wi], 0.0); out.append(jnp.sum(r ** 2))
    return jnp.stack(out)                                                            # per window

def phot_chi2(T1, T2, f, dphi):
    _, b1 = star_model(0, T1, f); _, b2 = star_model(1, T2, f); tot = []
    for k, b in enumerate(resp):
        mags = -2.5 * jnp.log10(b1[k] + b2[k])[order]; mags = mags - jnp.median(mags)
        mi = jnp.interp((ph_o - dphi) % 1.0, mph, mags, period=1.0); res = mag_o[b] - mi; res = res - jnp.median(res)
        tot.append(jnp.sum(res ** 2) / sig_ph[b] ** 2)
    return jnp.stack(tot)
N_w = jnp.asarray([float(good[wi].sum()) for wi in keep]); N_b = jnp.asarray([float(len(mag_o[b])) for b in resp])

# ---- validation of the interpolation against the direct kernel model ----
rng = np.random.default_rng(1)
for trial in range(3):
    T1 = th[0] + rng.uniform(-150, 150); T2 = th[1] + rng.uniform(-150, 150); f = rng.uniform(args.feh[0] + 0.15, args.feh[1] - 0.15)
    s1, _ = star_model(0, T1, f); s2, _ = star_model(1, T2, f)
    d1, _ = star_curves_j(0, float(T1), float(f)); d2, _ = star_curves_j(1, float(T2), float(f))
    worst = 0.0
    for w in range(len(keep)):
        mi = (s1[w][..., 0] + s2[w][..., 0]) / (s1[w][..., 1] + s2[w][..., 1]); md = (d1[w][..., 0] + d2[w][..., 0]) / (d1[w][..., 1] + d2[w][..., 1])
        worst = max(worst, float(jnp.max(jnp.abs(mi - md))))
    print(f"interpolation check ({T1:.0f}, {T2:.0f}, {f:+.2f}): max |interp - direct| = {worst:.5f} (normalised flux)", flush=True)

# ---- numpyro NUTS ----
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
lo = [T_nodes[s][1] for s in (0, 1)]; hi = [T_nodes[s][-2] for s in (0, 1)]
def model():
    T1 = numpyro.sample("Teff1", dist.Uniform(lo[0], hi[0])); T2 = numpyro.sample("Teff2", dist.Uniform(lo[1], hi[1]))
    f = numpyro.sample("feh", dist.Uniform(F_nodes[1], F_nodes[-2])); dphi = numpyro.sample("dphi", dist.Uniform(th[5] - 0.01, th[5] + 0.01))
    ls_w = numpyro.sample("log_s_spec", dist.Normal(jnp.log(3.0), 1.0).expand([len(keep)]))    # noise inflation per window
    ls_b = numpyro.sample("log_s_phot", dist.Normal(0.0, 1.0).expand([2]))
    c_w = spectra_chi2(T1, T2, f); c_b = phot_chi2(T1, T2, f, dphi)
    numpyro.factor("spec", jnp.sum(-0.5 * c_w * jnp.exp(-2 * ls_w) - N_w * ls_w))
    numpyro.factor("phot", jnp.sum(-0.5 * c_b * jnp.exp(-2 * ls_b) - N_b * ls_b))
kernel = NUTS(model, target_accept_prob=0.85, max_tree_depth=8)
mcmc = MCMC(kernel, num_warmup=args.num_warmup, num_samples=args.num_samples, num_chains=args.chains, chain_method="sequential", progress_bar=False)
t0 = time.time(); mcmc.run(jax.random.PRNGKey(0)); print(f"NUTS: {args.chains} chains x ({args.num_warmup} + {args.num_samples}) in {time.time() - t0:.0f}s", flush=True)
mcmc.print_summary()
S = mcmc.get_samples(group_by_chain=True)
flat = {k: np.asarray(v).reshape(-1, *np.asarray(v).shape[2:]) for k, v in S.items()}
print(f"\nposterior: Teff1 {flat['Teff1'].mean():.0f} +- {flat['Teff1'].std():.0f}   Teff2 {flat['Teff2'].mean():.0f} +- {flat['Teff2'].std():.0f}   "
      f"[Fe/H] {flat['feh'].mean():+.3f} +- {flat['feh'].std():.3f}   dphi {flat['dphi'].mean():.5f} +- {flat['dphi'].std():.5f}")
C = np.corrcoef(np.stack([flat["Teff1"], flat["Teff2"], flat["feh"]])); print(f"corr(Teff1,Teff2) {C[0,1]:+.2f}  corr(Teff2,feh) {C[1,2]:+.2f}  corr(Teff1,feh) {C[0,2]:+.2f}")
print("noise inflation per window (exp mean log_s):", np.round(np.exp(flat["log_s_spec"].mean(0)), 2), " photometry:", np.round(np.exp(flat["log_s_phot"].mean(0)), 2))
np.savez(args.out, **{k: np.asarray(v) for k, v in S.items()}, windows=np.array([all_windows[i] for i in keep]), theta_map=th, T_nodes=np.array(T_nodes), F_nodes=F_nodes)
print("saved", args.out)
