"""Feasibility of free geometry in the TZ For joint fit: light curves from SPICE's own binary.

The joint fit so far takes its meshes (and hence R1, R2, i) from PHOEBE. To free the
geometry, the photometric model must be built inside the loss from SPICE's own binary
(two icospheres on a Keplerian orbit, SPICE occlusion). This job, at the best-fit
temperatures of the masked full-range fit:

1. finds where SPICE puts the eclipses in its own phase convention (coarse scan), so the
   fit can align its clock with the PHOEBE-mesh one (deep eclipse at phase 0);
2. computes dense Stromgren b, y light curves from the SPICE binary at the literature
   geometry and compares them with the PHOEBE-mesh model at the same temperatures, same
   kernel treatment (n_mu = 16, no gravity darkening): depths, durations, residual rms;
3. times the forward light curve per phase and a full gradient of a light-curve chi2 with
   respect to (R1, R2, i) through the occlusion, at two mesh resolutions.

Writes an npz with both light curves for a local figure. GPU job, well under an hour.
"""
import argparse, math, os, pickle, sys, time
from pathlib import Path
os.environ.setdefault("JAX_ENABLE_X64", "1")
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, jax, jax.numpy as jnp
import tzfor_grad_inference as GI, tzfor_constants as K
from spice.models import IcosphereModel
from spice.models.binary import Binary, add_orbit, evaluate_orbit
from spice.spectrum.synthesis_kernel import build_synthesis_kernel, kernel_flux_multi

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--lc-meshes", default=str(HERE / "tzfor_aemu_out" / "tzfor_aemu_roche_n1200_lc6650_meshes.pkl"))
ap.add_argument("--result", default=str(HERE / "tzfor_aemu_out" / "fullrange_maskall_hp.pkl"))
ap.add_argument("--out", default=str(HERE / "tzfor_aemu_out" / "spice_lc_feasibility.npz"))
ap.add_argument("--n-vert", type=int, nargs="+", default=[1000, 2000]); ap.add_argument("--n-mu", type=int, default=16)
ap.add_argument("--n-phot-wl", type=int, default=16000); ap.add_argument("--n-grad-phases", type=int, default=24)
args = ap.parse_args()
print("devices:", jax.devices(), flush=True)

emu = GI.make_emulator()
LC = pickle.load(open(args.lc_meshes, "rb")); names = LC["parameter_names"]; iT, iG, iF = names.index("marcs_teff"), names.index("marcs_logg"), names.index("feh")
R = pickle.load(open(args.result, "rb")); th = {k: float(v) for k, v in zip(R["pnames"], R["theta"])}
print(f"theta: Teff1 {th['Teff1']:.0f} Teff2 {th['Teff2']:.0f} [Fe/H] {th['feh']:+.3f} dphi {th['dphi']:.4f}", flush=True)

def mean_row(s):
    m = LC["models"][0][s]; vis = np.asarray(m.mus) > 0; a = np.asarray(m.visible_cast_areas)[vis]; p = np.asarray(m.parameters)[vis]
    return np.average(p, axis=0, weights=a)
rows = []
for s, T in ((0, th["Teff1"]), (1, th["Teff2"])):
    r = mean_row(s); r[iT] = T; r[iF] = th["feh"]; rows.append(jnp.asarray(r))
lw_ph = jnp.linspace(math.log10(4300.0), math.log10(5900.0), args.n_phot_wl); wl_ph = 10.0 ** lw_ph
resp = {nm.split(":")[1]: jnp.interp(wl_ph, jnp.asarray(LC["passbands"][nm][0]), jnp.asarray(LC["passbands"][nm][1]), left=0., right=0.) for nm in ("Stromgren:b", "Stromgren:y")}
M1, M2, P_yr = K.PRIMARY_MASS, K.SECONDARY_MASS, K.PERIOD_YR
R1_0, R2_0, INC0 = K.PRIMARY_RADIUS, K.SECONDARY_RADIUS, K.INCL_DEG

def mags_from_kernels(ks1, ks2):
    """Band magnitudes (b, y) for lists of per-phase kernels of the two stars: emulator evaluated once per star."""
    f = kernel_flux_multi(emu.intensity, ks1, rows[0])[..., 0] + kernel_flux_multi(emu.intensity, ks2, rows[1])[..., 0]
    return {b: jax.vmap(lambda ff: GI.passband_mag(ff, wl_ph, rr))(f) for b, rr in resp.items()}

# ---- PHOEBE-mesh reference at the same rows / kernel treatment ----
ph_lc = ((np.asarray(LC["times"]) - K.T_P_HJD) % K.PERIOD_DAYS) / K.PERIOD_DAYS
ks_ph = [[build_synthesis_kernel(pair[s], lw_ph, args.n_mu, 1) for pair in LC["models"]] for s in (0, 1)]
t = time.time(); m_ph = {b: np.asarray(v) for b, v in mags_from_kernels(ks_ph[0], ks_ph[1]).items()}; print(f"PHOEBE-mesh model: {len(ph_lc)} phases in {time.time() - t:.0f}s", flush=True)

# ---- SPICE binary ----
def binary_at(n_vert, R1, R2, inc_deg, mean_anom=0.0):
    b1 = IcosphereModel.construct(n_vert, R1, M1, rows[0], names)
    b2 = IcosphereModel.construct(n_vert, R2, M2, rows[1], names)
    return add_orbit(Binary.from_bodies(b1, b2), P_yr, 0.0, 0.0, jnp.deg2rad(inc_deg), 0.0, 0.0, mean_anom, 0.0, 0.0, 400)

def kernels_at(binary, phases):
    ks1, ks2 = [], []
    for ph in phases:
        m1, m2 = evaluate_orbit(binary, ph * P_yr)
        ks1.append(build_synthesis_kernel(m1, lw_ph, args.n_mu, 1, half_width=8)); ks2.append(build_synthesis_kernel(m2, lw_ph, args.n_mu, 1, half_width=8))
    return ks1, ks2

out = dict(ph_phoebe=ph_lc, b_phoebe=m_ph["b"], y_phoebe=m_ph["y"], theta=np.asarray(R["theta"]), pnames=np.array(R["pnames"]))
for nv in args.n_vert:
    print(f"\n===== SPICE binary, {nv} vertices per star =====", flush=True)
    # 1. coarse scan: where are the eclipses in SPICE's clock?
    coarse = np.linspace(0.0, 1.0, 100, endpoint=False); bin0 = binary_at(nv, R1_0, R2_0, INC0)
    t = time.time(); k1, k2 = kernels_at(bin0, coarse); mc = {b: np.asarray(v) for b, v in mags_from_kernels(k1, k2).items()}
    print(f"coarse scan: 100 phases in {time.time() - t:.0f}s ({(time.time() - t) / 100:.2f} s/phase incl. compile)", flush=True)
    i_deep = int(np.argmax(mc["b"])); ph_deep = coarse[i_deep]; depth_b = mc["b"][i_deep] - np.median(mc["b"])
    others = np.abs(((coarse - ph_deep + 0.5) % 1.0) - 0.5) > 0.1; i_sh = int(np.argmax(np.where(others, mc["b"], -np.inf))); ph_sh = coarse[i_sh]
    print(f"deep minimum at SPICE phase {ph_deep:.2f} (depth {depth_b:.3f} mag in b), shallow at {ph_sh:.2f} (depth {mc['b'][i_sh] - np.median(mc['b']):.3f})", flush=True)
    # 2. aligned clock (deep eclipse at phase 0) and a dense light curve at the PHOEBE-mesh phases + eclipse profiles
    mean_anom = 2.0 * math.pi * ph_deep; bin_al = binary_at(nv, R1_0, R2_0, INC0, mean_anom)
    dense = np.unique(np.concatenate([ph_lc, np.linspace(-0.03, 0.03, 61) % 1.0, 0.5 + np.linspace(-0.03, 0.03, 61)]))
    t = time.time(); k1, k2 = kernels_at(bin_al, dense); md = {b: np.asarray(v) for b, v in mags_from_kernels(k1, k2).items()}
    print(f"dense light curve: {dense.size} phases in {time.time() - t:.0f}s", flush=True)
    for b in ("b", "y"):
        s_sp = np.interp(ph_lc, dense, md[b] - np.median(md[b]), period=1.0); s_ph = m_ph[b] - np.median(m_ph[b])
        d = s_sp - s_ph; ecl = (np.abs(((ph_lc + 0.5) % 1.0) - 0.5) < 0.02) | (np.abs(ph_lc - 0.5) < 0.02)
        print(f"  {b}: SPICE - PHOEBE at the 56 mesh phases: rms {1000 * np.sqrt(np.mean(d ** 2)):.2f} mmag (in eclipse {1000 * np.sqrt(np.mean(d[ecl] ** 2)):.2f}, outside {1000 * np.sqrt(np.mean(d[~ecl] ** 2)):.2f}); "
              f"deep-eclipse depth SPICE {s_sp[np.argmin(np.abs(ph_lc))]:.4f} vs PHOEBE {s_ph[np.argmin(np.abs(ph_lc))]:.4f}", flush=True)
    out[f"ph_spice_{nv}"] = dense; out[f"b_spice_{nv}"] = md["b"]; out[f"y_spice_{nv}"] = md["y"]; out[f"mean_anom_{nv}"] = mean_anom
    np.savez(args.out, **out)
    # 3. gradient timing through the occlusion: chi2 of the in-eclipse points against the PHOEBE-mesh model
    sel = np.argsort(np.abs(((ph_lc + 0.5) % 1.0) - 0.5))[: args.n_grad_phases // 2].tolist() + np.argsort(np.abs(ph_lc - 0.5))[: args.n_grad_phases // 2].tolist()
    ph_g = jnp.asarray(ph_lc[sel]); ref = {b: jnp.asarray(m_ph[b][sel]) for b in ("b", "y")}
    def chi2(R1, R2, inc):
        bn = binary_at(nv, R1, R2, inc, mean_anom); ks1, ks2 = kernels_at(bn, [ph_g[j] for j in range(len(sel))]); mg = mags_from_kernels(ks1, ks2)
        return sum(jnp.sum(((mg[b] - jnp.median(mg[b])) - (ref[b] - jnp.median(ref[b]))) ** 2) / 0.004 ** 2 for b in ("b", "y"))
    vg = jax.value_and_grad(chi2, argnums=(0, 1, 2))
    t = time.time(); (v, g) = vg(R1_0, R2_0, INC0); jax.block_until_ready(g); t1 = time.time() - t
    t = time.time(); (v, g) = vg(R1_0 + 0.01, R2_0, INC0); jax.block_until_ready(g); t2 = time.time() - t
    print(f"gradient over {len(sel)} in-eclipse phases: first call {t1:.0f}s (compile), second {t2:.1f}s; chi2 {float(v):.1f}; "
          f"d chi2/dR1 {float(g[0]):+.3g} /Rsun, d/dR2 {float(g[1]):+.3g} /Rsun, d/d i {float(g[2]):+.3g} /deg; finite {all(np.isfinite(float(x)) for x in g)}", flush=True)
    h = 0.02; fd = (chi2(R1_0 + h, R2_0, INC0) - chi2(R1_0 - h, R2_0, INC0)) / (2 * h); print(f"  finite-difference d chi2/dR1 {float(fd):+.3g}", flush=True)
    out[f"grad_time_{nv}"] = t2; np.savez(args.out, **out)
print("===== DONE =====")
