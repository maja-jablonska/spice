"""Projection factor of delta Cep implied by the HARPS-N Mg b line positions, with its statistical and systematic
errors (paper Sect. 3.2.1, referee round 2).

The literature-model run (exact per-element synthesis, nothing fitted; dcep_out/*.pkl) stores, per epoch, the CCF velocity of the observed
Mg b window against the model spectrum of the same epoch (dv_start), the per-epoch systemic velocity of the
Nardetto et al. (2024) orbit (dv_epoch) and the projected-area-weighted line-of-sight pulsation velocity of the mesh
(vpuls_model = 2/3 V_puls for a radially pulsating sphere; checked against the SPIPS Vpuls curve). The residual
y = dv_start - dv_epoch regressed on x = vpuls_model has slope s, so the observed radial velocity is
V_puls (1/p0 + 2/3 s) and p = (1/p0 + 2/3 s)^-1, with p0 the model's own CCF projection factor in the Mg b window
(5160-5200 A, Turbospectrum synthesis: 1.35 against the phase-0 template, 1.34 against a rest-frame template).

Statistical error: epochs within a night are strongly correlated (residual correlation ~ +1), so the slope error is
cluster-robust by night (CR1), checked with a bootstrap over whole nights within each season.
Systematic error: quadrature sum of (i) the p0 template (half the 1.34/1.35 difference), (ii) the definition of the
velocity scale when observed and model curves differ in shape (half the range of OLS slope, first-harmonic, rms and
peak-to-peak amplitude ratios), (iii) Teff curve / [Fe/H] (variant runs), (iv) the orbit drift within a season
(+-20 per cent), (v) emulator vs Turbospectrum p0 (Table 6: 0.001).

    python dcep_lit_pfactor_errors.py [--dir dcep_out] [--nboot 20000]

The four model results in dcep_out/ come from the Hugging Face dataset mjablonska/spice-paper-data
(python tutorial/paper_results/fetch_paper_data.py --only dcep).
"""
import argparse
import pickle
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P0 = (1.35, 1.34)                         # Mg b window CCF p0: phase-0 template, rest-frame template
P0_EMU_TS = 0.001                         # emulator - Turbospectrum p0 difference (Table 6)
ADOPTED = "dcep_v12V13_own_trahin_p3"
VARIANTS = ("dcep_v12V13_own_luck_p3", "dcep_v12V13_all_trahin_p3", "dcep_v12V13_all_luck_p3")


def p_of(s, p0):
    return 1.0 / (1.0 / p0 + 2.0 / 3.0 * s)


def load(path, orbit_scale=1.0):
    r = pickle.load(open(path, "rb"))
    dv = np.asarray(r["dv_start"], float); x = np.asarray(r["vpuls_model"], float)
    ss = np.asarray(r["season_spec"]).astype(str); de = np.asarray(r["dv_epoch"], float).copy()
    for s in np.unique(ss):                                    # optionally rescale the orbit's drift within a season
        m = ss == s; de[m] = de[m].mean() + orbit_scale * (de[m] - de[m].mean())
    y = dv - de; m = np.isfinite(y)
    for s in np.unique(ss):                                    # the run's outlier cut: 10 km/s from the season median
        k = (ss == s) & m; m &= ~((ss == s) & (np.abs(y - np.median(y[k])) >= 10.0))
    night = np.floor(np.asarray(r["times_spec"], float) - 0.5).astype(int)   # MJD; La Palma nights split at noon UT
    return dict(x=x[m], y=y[m], season=ss[m], night=night[m], ph=(np.asarray(r["phases_spec"], float) % 1.0)[m])


def fit(d):
    """OLS slope with plain and night-clustered (CR1) standard errors."""
    A = np.c_[d["x"], np.ones_like(d["x"])]; c = np.linalg.lstsq(A, d["y"], rcond=None)[0]; e = d["y"] - A @ c
    n, k = A.shape; XtXi = np.linalg.inv(A.T @ A)
    se = np.sqrt(XtXi[0, 0] * e @ e / (n - k))
    G = np.unique(d["night"]); meat = sum(np.outer(A[d["night"] == g].T @ e[d["night"] == g],
                                                   A[d["night"] == g].T @ e[d["night"] == g]) for g in G)
    se_cr = np.sqrt(len(G) / (len(G) - 1) * (n - 1) / (n - k) * (XtXi @ meat @ XtXi)[0, 0])
    return c[0], se, se_cr, e, len(G)


def night_bootstrap(d, nb, rng):
    groups = [[np.flatnonzero((d["season"] == s) & (d["night"] == g)) for g in np.unique(d["night"][d["season"] == s])]
              for s in np.unique(d["season"])]
    out = np.empty(nb)
    for b in range(nb):
        idx = np.concatenate([np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))]) for g in groups])
        out[b] = np.polyfit(d["x"][idx], d["y"][idx], 1)[0]
    return out


def scale_estimators(d, p0):
    """Observed / model velocity amplitude under several definitions (model CCF velocity = x 1.5 / p0)."""
    F = lambda p, k: np.stack([np.ones_like(p)] + [f(2 * np.pi * j * p) for j in range(1, k + 1) for f in (np.cos, np.sin)], 1)
    vm = d["x"] * 1.5 / p0; vo = vm + d["y"]; pp = np.linspace(0, 1, 2001)
    c4m, c4o = (np.linalg.lstsq(F(d["ph"], 4), v, rcond=None)[0] for v in (vm, vo))
    c6m, c6o = (np.linalg.lstsq(F(d["ph"], 6), v, rcond=None)[0] for v in (vm, vo))
    return {"OLS slope": 1 + np.polyfit(d["x"], d["y"], 1)[0] * 2 / 3 * p0,
            "first-harmonic amplitude": np.hypot(*c4o[1:3]) / np.hypot(*c4m[1:3]),
            "rms amplitude (Fourier 6)": (F(pp, 6) @ c6o).std() / (F(pp, 6) @ c6m).std(),
            "peak-to-peak (Fourier 6)": np.ptp(F(pp, 6) @ c6o) / np.ptp(F(pp, 6) @ c6m)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", default=str(HERE / "dcep_out")); ap.add_argument("--nboot", type=int, default=20000)
    a = ap.parse_args(); D = Path(a.dir); rng = np.random.default_rng(1)
    p0 = np.mean(P0)
    d = load(D / f"{ADOPTED}.pkl")
    s, se, se_cr, e, ng = fit(d); p = p_of(s, p0); dpds = p ** 2 * 2.0 / 3.0
    print(f"{ADOPTED}: {len(d['x'])} epochs, {ng} nights; s = {s:.4f}; p0 = {p0:.3f} (mean of {P0}); p = {p:.4f}; "
          f"observed/model velocity ratio {1 + s * 2 / 3 * p0:.4f}")
    same = d["night"][:, None] == d["night"][None, :]; iu = np.triu_indices(len(e), 1); pr = same[iu]
    print(f"  within-night residual correlation {np.corrcoef(e[iu[0][pr]], e[iu[1][pr]])[0, 1]:+.2f}")
    bs = night_bootstrap(d, a.nboot, rng)
    print(f"  STAT  slope SE: OLS {se:.4f}, night-clustered {se_cr:.4f}, night bootstrap {bs.std():.4f}"
          f"  -> p: {dpds * se:.4f} / {dpds * se_cr:.4f} / {p_of(bs, p0).std():.4f}")
    for sn in np.unique(d["season"]):
        k = d["season"] == sn; dd = {kk: v[k] for kk, v in d.items()}; ss_, _, sc, _, n_ = fit(dd)
        print(f"        season {sn}: {k.sum()} epochs, {n_} nights, p = {p_of(ss_, p0):.4f} +- {p_of(ss_, p0) ** 2 * 2 / 3 * sc:.4f}")
    terms = {}
    terms["p0 template (1.34 / 1.35)"] = abs(p_of(s, P0[0]) - p_of(s, P0[1])) / 2
    est = scale_estimators(d, p0); pe = {k: 1.0 / (v / p0) for k, v in est.items()}
    for k, v in pe.items():
        print(f"        scale definition {k:28s} p = {v:.4f}")
    terms["velocity-scale definition"] = (max(pe.values()) - min(pe.values())) / 2
    pv = [p_of(fit(load(D / f"{v}.pkl"))[0], p0) for v in VARIANTS if (D / f"{v}.pkl").exists()]
    terms["Teff curve / [Fe/H] variants"] = max(abs(np.array(pv) - p)) if pv else 0.0
    po = [p_of(fit(load(D / f"{ADOPTED}.pkl", f))[0], p0) for f in (0.8, 1.2)]
    terms["orbit drift +-20 %"] = max(abs(np.array(po) - p))
    terms["emulator vs Turbospectrum p0"] = P0_EMU_TS * (p / p0) ** 2
    for k, v in terms.items():
        print(f"  SYST  {k:32s} {v:.4f}")
    syst = np.sqrt(sum(v ** 2 for v in terms.values())); stat = dpds * se_cr
    print(f"RESULT p = {p:.3f} +- {stat:.3f} (stat., night-clustered) +- {syst:.3f} (syst.); total {np.hypot(stat, syst):.3f}; "
          f"f_grad f_o-g = p/p0 = {p / p0:.3f}")


if __name__ == "__main__":
    main()
