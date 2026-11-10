"""Which lines carry the solar [Fe/H] offset? Per-line results of ts_direct_check.py matched to the Turbospectrum line list.

For every fitted line (best [Fe/H] from the line's own chi2), the dominant contributor within +-0.06 A is found
with a population-weighted strength proxy at a solar-like line-forming temperature (theta = 5040/T = 1.0):
  s = log gf + A(X) - theta * chi_low - (1.2 for neutral atoms whose ionisation energy is below 9 eV)
(neutrals of easily ionised metals are minority species in the Sun). A line is 'clean' when that contributor
exceeds the next one by more than --clean dex. Results are grouped by species, and for Fe I by excitation and
strength. Also counts duplicate entries (same species, wavelength, excitation and log gf) in the list.

Usage (Gadi): python ts_line_species.py --linelist FILE --abund-log BSYN_LOG --results ts_direct_check.json [--results ...]
              [--xi 1.09] [--groups Moon Ceres Vesta Ganymede KPNO2005]
"""
import argparse, collections, json, re

import numpy as np

IONISATION_BELOW_9EV = {3, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 37, 38, 39, 40, 41, 42, 56, 57, 58, 59, 60, 62, 63, 64, 66}
ELEMENTS = "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi".split()


def read_linelist(path):
    """Turbospectrum (NLTE-format) atomic list -> arrays of (Z, ion, wl, chi, loggf)."""
    Z, ION, WL, CHI, GF = [], [], [], [], []
    with open(path) as fh:
        lines = fh.readlines()
    i = 0
    while i < len(lines):
        head = lines[i]
        if not head.startswith("'"):
            i += 1; continue
        code = float(head.split("'")[1]); fields = head.split("'")[2].split()
        z = int(code); ion = int(fields[0]); n = int(fields[1])         # header: 'Z.iso'  ion  n_lines
        for row in lines[i + 2:i + 2 + n]:
            p = row.split()
            Z.append(z); ION.append(ion); WL.append(float(p[0])); CHI.append(float(p[1])); GF.append(float(p[2]))
        i += 2 + n
    return np.array(Z), np.array(ION), np.array(WL), np.array(CHI), np.array(GF)


def read_abund(log):
    """The 'ELEMENT  ATOMIC NUMBER  LOG(ABUNDANCE)' table written by babsma/bsyn."""
    A = {}
    for m in re.finditer(r"^\s+([A-Z][a-z]?)\s+(\d+)\s+(-?\d+\.\d+)\s*$", open(log).read(), re.M):
        A.setdefault(int(m.group(2)), float(m.group(3)))
    return A


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--linelist", required=True); ap.add_argument("--abund-log", required=True)
    ap.add_argument("--results", nargs="+", required=True); ap.add_argument("--xi", default="1.09")
    ap.add_argument("--groups", nargs="*", default=["Moon", "Ceres", "Vesta", "Ganymede", "KPNO2005"])
    ap.add_argument("--clean", type=float, default=0.7); ap.add_argument("--window", type=float, default=0.06)
    ap.add_argument("--theta", type=float, default=1.0)
    args = ap.parse_args()
    Z, ION, WL, CHI, GF = read_linelist(args.linelist); A = read_abund(args.abund_log)
    print(f"line list: {WL.size} lines, {len(set(zip(Z, ION)))} species; abundances for {len(A)} elements (Fe {A.get(26)})")

    # duplicates: same species, wavelength (1 mA), excitation (1 meV) and log gf (0.001)
    key = collections.Counter(zip(Z, ION, np.round(WL, 3), np.round(CHI, 3), np.round(GF, 3)))
    dup = {k: c for k, c in key.items() if c > 1}
    strength = lambda z, ion, chi, gf: gf + A.get(z, -9) - args.theta * chi - (1.2 if ion == 1 and z in IONISATION_BELOW_9EV else 0.0)
    strong_dup = [(k, c) for k, c in dup.items() if strength(k[0], k[1], k[3], k[4]) > 5.0 and 5100 < k[2] < 6500]
    print(f"exact duplicate entries: {sum(c - 1 for c in dup.values())} extra copies of {len(dup)} lines; "
          f"{len(strong_dup)} of them non-negligible (proxy > 5) in 5100-6500 A")
    for k, c in sorted(strong_dup, key=lambda kc: -strength(kc[0][0], kc[0][1], kc[0][3], kc[0][4]))[:15]:
        print(f"   {ELEMENTS[k[0] - 1]} {'I' * k[1]} {k[2]:.3f} chi {k[3]:.3f} loggf {k[4]:+.3f} x{c}  proxy {strength(k[0], k[1], k[3], k[4]):.2f}")
    # near-duplicates: same species and wavelength (10 mA) and excitation, log gf different (e.g. two sources merged)
    near = collections.defaultdict(list)
    for z, ion, w, chi, gf in zip(Z, ION, WL, CHI, GF):
        if 5100 < w < 6500:
            near[(z, ion, round(w, 2), round(chi, 2))].append(gf)
    nd = [(k, v) for k, v in near.items() if len(v) > 1 and max(strength(k[0], k[1], k[3], g) for g in v) > 5.0]
    print(f"same species + wavelength (0.01 A) + excitation, >1 entry, proxy > 5: {len(nd)}")
    for k, v in sorted(nd, key=lambda kv: -max(kv[1]))[:10]:
        print(f"   {ELEMENTS[k[0] - 1]} {'I' * k[1]} {k[2]:.2f} chi {k[3]:.2f} loggf {sorted(v)}")

    sw = np.array([strength(z, ion, chi, gf) for z, ion, chi, gf in zip(Z, ION, CHI, GF)])
    order = np.argsort(WL); WLs, sws = WL[order], sw[order]
    rows = []
    for path in args.results:
        R = json.load(open(path))
        for k, r in R.items():
            if k.split("_")[0] not in args.groups or args.xi not in r["xi"]:
                continue
            for l in r["xi"][args.xi]["lines"]:
                if l["edge"]:
                    continue
                a, b = np.searchsorted(WLs, [l["wl"] - args.window, l["wl"] + args.window])
                if b - a == 0:
                    continue
                idx = order[a:b][np.argsort(-sws[a:b])]
                top = idx[0]; gap = sw[top] - (sw[idx[1]] if idx.size > 1 else -99)
                rows.append(dict(spec=k, wl=l["wl"], depth=l["depth"], feh=l["feh"], z=int(Z[top]), ion=int(ION[top]), chi=float(CHI[top]),
                                 gf=float(GF[top]), proxy=float(sw[top]), clean=bool(gap > args.clean)))
    print(f"\nfitted lines matched: {len(rows)} from {len({r['spec'] for r in rows})} spectra; clean (dominant by > {args.clean} dex): {sum(r['clean'] for r in rows)}")
    fe_scale = A.get(26, 7.51) - 7.45
    print(f"(the grid's A(Fe) {A.get(26)} vs 7.45 accounts for {-fe_scale:+.2f} dex of any offset)")
    for label, sel in (("all", lambda r: True), ("clean", lambda r: r["clean"])):
        print(f"\n{label} lines: median best [Fe/H] {np.median([r['feh'] for r in rows if sel(r)]):+.3f}")
        by = collections.defaultdict(list)
        for r in rows:
            if sel(r):
                by[(r["z"], r["ion"])].append(r["feh"])
        for (z, ion), v in sorted(by.items(), key=lambda kv: -len(kv[1]))[:14]:
            print(f"   {ELEMENTS[z - 1]:2s} {'I' * ion:3s} n {len(v):4d}  median {np.median(v):+.3f}  IQR {np.percentile(v, 75) - np.percentile(v, 25):.3f}")
    fe1 = [r for r in rows if r["clean"] and r["z"] == 26 and r["ion"] == 1]
    if fe1:
        print("\nclean Fe I lines by lower-level excitation and by depth:")
        for lo, hi in ((0, 2), (2, 3), (3, 4), (4, 6)):
            v = [r["feh"] for r in fe1 if lo <= r["chi"] < hi]
            if v: print(f"   chi {lo}-{hi} eV: n {len(v):4d} median {np.median(v):+.3f}")
        for lo, hi in ((0.05, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.0)):
            v = [r["feh"] for r in fe1 if lo <= r["depth"] < hi]
            if v: print(f"   depth {lo:.2f}-{hi:.2f}: n {len(v):4d} median {np.median(v):+.3f}")
    json.dump(rows, open(args.results[0].replace(".json", f"_species_xi{args.xi}.json"), "w"), indent=0)


if __name__ == "__main__":
    main()
