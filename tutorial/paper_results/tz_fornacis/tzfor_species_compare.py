"""Where does the TZ For [Fe/H] zero-point come from? Literature-model misfit split by line class, with and without a free
continuum level.

On Gallenne-like Fe I lines the unshifted Gallenne parameters fit best, while the full spectrum prefers [Fe/H] ~0.2 dex lower
(tzfor_fe1_compare.py). This splits each model's residuals into pixel classes, in the primary's rest frame:
  Fe I / Fe II / other neutral / other ionised : +-0.12 A windows around isolated, moderate lines (observed depth 0.05-0.45;
                                                 no other line within +-0.2 A stronger than this one minus 1 dex)
  blended / forest                             : other pixels with observed depth > 0.05
  continuum                                    : observed depth < 0.05
For each class: chi2/N and the mean model - observation (> 0: model lines too shallow). The whole split is then repeated after
rescaling each model in every 25 A block of every epoch by the least-squares factor (a free continuum level per block), so a
pseudo-continuum offset cannot masquerade as a line-strength difference.

    python tzfor_species_compare.py LINELIST A.npz [B.npz ...]   (the first dump supplies observations and velocities)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

C_KMS = 299792.458
THETA = 5040.0 / 4930.0
DEPTH = (0.05, 0.45)
ISO_A, ISO_DEX, HALF_WIN, BLOCK_A = 0.2, 1.0, 0.12, 25.0
# solar photospheric abundances (Asplund et al. 2009), for the isolation strength proxy only
A_SUN = dict(Li=1.05, C=8.43, N=7.83, O=8.69, Na=6.24, Mg=7.60, Al=6.45, Si=7.51, S=7.12, K=5.03, Ca=6.34, Sc=3.15, Ti=4.95,
             V=3.93, Cr=5.64, Mn=5.43, Fe=7.50, Co=4.99, Ni=6.22, Cu=4.19, Zn=4.56, Sr=2.87, Y=2.21, Zr=2.58, Ba=2.18, La=1.10,
             Ce=1.58, Nd=1.42, Sm=0.96, Eu=0.52)


def read_bsyn(path):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tzfor_fe1_compare import read_bsyn as rb
    return rb(path)


def main():
    linelist, *dumps = sys.argv[1:]
    names, wl, chi, gf = read_bsyn(linelist)
    elem = np.array([n.split()[0] for n in names]); ion = np.array([n.split()[1] if len(n.split()) > 1 else "I" for n in names])
    strength = gf - THETA * chi + np.array([A_SUN.get(e, 2.0) - 7.50 for e in elem]) + np.where(ion == "II", 1.0, 0.0)
    order = np.argsort(wl); wl_s, st_s = wl[order], strength[order]
    D = {Path(d).stem.replace("tzfor_lit_nomask_p3_", ""): np.load(d, allow_pickle=True) for d in dumps}
    R = next(iter(D.values())); obs, sig = R["obs"], R["sigma"]; lam = 10 ** R["logwl"]; v1 = R["rv"][0]; n_ep = obs.shape[0]
    use = np.logical_and.reduce([X["good"] & np.isfinite(X["model"]) for X in D.values()] + [np.isfinite(obs)])
    for X in D.values():
        assert np.array_equal(X["obs"], obs, equal_nan=True)
    rest = lam[None, :] / (1 + v1[:, None] / C_KMS)                      # primary rest-frame wavelength per epoch, pixel
    depth = 1.0 - obs

    def isolated(i):
        lo, hi = np.searchsorted(wl_s, [wl[i] - ISO_A, wl[i] + ISO_A])
        others = st_s[lo:hi][np.abs(wl_s[lo:hi] - wl[i]) > 1e-4]
        return not np.any(others > strength[i] - ISO_DEX)

    def windows(sel):
        """(n_ep, n_px) mask of +-HALF_WIN windows around lines sel whose median observed depth is moderate."""
        keep = []
        for i in sel:
            if not isolated(i):
                continue
            j = np.searchsorted(lam, wl[i] * (1 + v1 / C_KMS)); ok = (j > 0) & (j < lam.size)
            dep = [depth[e, j[e]] for e in range(n_ep) if ok[e] and use[e, j[e]]]
            if dep and DEPTH[0] <= np.median(dep) <= DEPTH[1]:
                keep.append(wl[i])
        keep = np.sort(np.asarray(keep)); m = np.zeros_like(use)
        if keep.size == 0:
            return m, 0
        for e in range(n_ep):
            k = np.clip(np.searchsorted(keep, rest[e]), 1, keep.size - 1)
            m[e] = np.minimum(np.abs(rest[e] - keep[k - 1]), np.abs(rest[e] - keep[k])) <= HALF_WIN
        return m, keep.size

    inrange = (wl > lam.min()) & (wl < lam.max())
    classes = {}
    for lab, sel in (("Fe I", (elem == "Fe") & (ion == "I")), ("Fe II", (elem == "Fe") & (ion == "II")),
                     ("other neutral", (elem != "Fe") & (ion == "I")), ("other ionised", (elem != "Fe") & (ion != "I"))):
        m, n = windows(np.where(sel & inrange)[0]); classes[f"{lab} ({n} lines)"] = m & use
    iso_any = np.logical_or.reduce(list(classes.values()))
    classes["blended / forest (depth > 0.05)"] = use & (depth > 0.05) & ~iso_any
    classes["continuum (depth < 0.05)"] = use & (depth <= 0.05) & ~iso_any

    def rescaled(model):
        """Model times the least-squares factor per 25 A block and epoch (free continuum level)."""
        out = model.copy(); edges = np.arange(lam.min(), lam.max() + BLOCK_A, BLOCK_A)
        b = np.digitize(lam, edges)
        for e in range(n_ep):
            for k in np.unique(b):
                px = (b == k) & use[e]
                if px.sum() > 20:
                    w = 1.0 / sig[e, px] ** 2; f = np.sum(w * model[e, px] * obs[e, px]) / np.sum(w * model[e, px] ** 2)
                    out[e, b == k] = model[e, b == k] * f
        return out

    for title, tf in (("AS SYNTHESISED", lambda m: m), ("FREE CONTINUUM LEVEL PER 25 A BLOCK", rescaled)):
        print(f"\n=== {title}: chi2/N  [mean model - obs]  per pixel class")
        print(f"{'class':38s} {'px':>9s} " + " ".join(f"{k[:22]:>24s}" for k in D))
        M = {k: tf(X["model"]) for k, X in D.items()}
        for lab, m in list(classes.items()) + [("ALL", use)]:
            cells = []
            for k, X in D.items():
                r = (M[k] - obs)[m]
                cells.append(f"{np.mean((r / sig[m]) ** 2):8.2f} [{np.mean(r):+.4f}]")
            print(f"{lab:38s} {int(m.sum()):9,d} " + " ".join(f"{c:>24s}" for c in cells))


if __name__ == "__main__":
    main()
