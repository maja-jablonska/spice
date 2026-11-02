"""TZ For literature models scored on Gallenne-like Fe I lines only.

Gallenne et al. (2016) derived [Fe/H] and the microturbulence from Fe I equivalent widths on disentangled spectra
(4125-6840 A, MOOG + ATLAS9), rejecting strong lines whose profiles were not Gaussian. Their line list is not published, so
this rebuilds an equivalent one from our own Turbospectrum list: Fe I lines in 4125-6840 A that are moderately deep in the
OBSERVED composite spectrum (depth 0.05-0.45 at the primary's line centre, median over epochs: no model chooses the lines) and isolated (no other atomic line within
+-0.2 A with a strength proxy log gf - theta chi above that of the Fe I line minus 1 dex, theta = 5040 / 4930 K). Each model
dump is then scored only in +-0.12 A windows around those lines in the primary's rest frame, on the pixels every model
uses. The mean model - observation in the windows says whether the lines are too deep (< 0) or too shallow (> 0).

    python tzfor_fe1_compare.py LINELIST REF.npz A.npz [B.npz ...]

REF supplies the observations and the primary velocities used to choose the lines; every listed dump is scored on the same windows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

C_KMS = 299792.458
THETA = 5040.0 / 4930.0
WL_RANGE = (4125.0, 6840.0)
DEPTH = (0.05, 0.45)
ISO_A, ISO_DEX, HALF_WIN = 0.2, 1.0, 0.12


def read_bsyn(path):
    """Turbospectrum atomic line list -> (species name, wavelength [A], chi [eV], log gf) for every line."""
    names, wl, chi, gf = [], [], [], []
    lines = open(path, errors="replace").read().splitlines()
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("'") and i + 1 < len(lines) and lines[i + 1].strip().startswith("'"):
            head = s.split("'")
            try:
                n_lines = int(head[2].split()[1])
            except (IndexError, ValueError):
                i += 1; continue
            name = lines[i + 1].strip().strip("'").split()
            name = " ".join(name[:2]) if len(name) >= 2 else "".join(name)
            for row in lines[i + 2:i + 2 + n_lines]:
                f = row.split()
                try:
                    wl.append(float(f[0])); chi.append(float(f[1])); gf.append(float(f[2])); names.append(name)
                except (IndexError, ValueError):
                    pass
            i += 2 + n_lines
        else:
            i += 1
    return np.asarray(names), np.asarray(wl), np.asarray(chi), np.asarray(gf)


def main():
    linelist, ref, *dumps = sys.argv[1:]
    names, wl, chi, gf = read_bsyn(linelist)
    strength = gf - THETA * chi
    fe = (names == "Fe I") & (wl >= WL_RANGE[0]) & (wl <= WL_RANGE[1])
    print(f"line list {Path(linelist).name}: {wl.size:,} atomic lines, {int(fe.sum()):,} Fe I in {WL_RANGE[0]:.0f}-{WL_RANGE[1]:.0f} A")
    order = np.argsort(wl); wl_s, st_s = wl[order], strength[order]
    iso = np.zeros(wl.size, bool)
    for i in np.where(fe)[0]:
        lo, hi = np.searchsorted(wl_s, [wl[i] - ISO_A, wl[i] + ISO_A])
        others = st_s[lo:hi][np.abs(wl_s[lo:hi] - wl[i]) > 1e-4]
        iso[i] = not np.any(others > strength[i] - ISO_DEX)
    R = np.load(ref, allow_pickle=True)
    lam = 10 ** R["logwl"]; v1 = R["rv"][0]
    use_all = [R["good"] & np.isfinite(R["obs"])]
    D = {Path(d).stem: np.load(d, allow_pickle=True) for d in dumps}
    for X in D.values():
        assert np.array_equal(X["obs"], R["obs"], equal_nan=True), "dumps must share the observations"
        use_all.append(X["good"] & np.isfinite(X["model"]))
    use = np.logical_and.reduce(use_all)
    sel = []
    for i in np.where(fe & iso)[0]:
        dep = []
        for e in range(R["obs"].shape[0]):
            lc = wl[i] * (1 + v1[e] / C_KMS); j = np.searchsorted(lam, lc)
            if 0 < j < lam.size and use[e, j]:
                dep.append(1.0 - R["obs"][e, j])
        if dep and DEPTH[0] <= np.median(dep) <= DEPTH[1]:
            sel.append(i)
    sel = np.asarray(sel)
    print(f"selected {sel.size} isolated, moderate Fe I lines (of {int((fe & iso).sum())} isolated); windows +-{HALF_WIN} A in the primary frame")
    win = np.zeros_like(use)
    for e in range(R["obs"].shape[0]):
        rest = lam / (1 + v1[e] / C_KMS)
        k = np.searchsorted(wl[sel], rest)
        near = np.minimum(np.abs(rest - wl[sel][np.clip(k - 1, 0, sel.size - 1)]), np.abs(rest - wl[sel][np.clip(k, 0, sel.size - 1)]))
        win[e] = near <= HALF_WIN
    m = win & use
    print(f"{int(m.sum()):,} pixels in the windows")
    print(f"{'model':45s} {'chi2/N Fe I':>12s} {'chi2/N all':>11s} {'mean(model-obs) Fe I':>21s}")
    for nm, X in D.items():
        r = (X["model"] - X["obs"]) / X["sigma"]
        print(f"{nm:45s} {np.mean(r[m] ** 2):12.3f} {np.mean(r[use] ** 2):11.3f} {np.mean((X['model'] - X['obs'])[m]):+21.4f}")
    np.savez_compressed(Path(ref).with_name("tzfor_fe1_lines.npz"), wl=wl[sel], chi=chi[sel], loggf=gf[sel])


if __name__ == "__main__":
    main()
