"""Combine the benchmark-star fits of all selections into one calibration summary.

Reads <out>/gbs_{vmic15,vmicgbs,fixteff}<tag>.json for the tags in TAGS (later tags override earlier ones for the same
star, e.g. the re-fits with clean exposures), drops EXCLUDE, groups stars into giants / F stars / G-K dwarfs /
metal-poor, and prints mean offsets (fit - benchmark) with errors of the mean and scatter, plus straight-line trends of
the [Fe/H] offset (Teff fixed) against Teff, log g and [Fe/H].

Usage: python gbs_combine.py OUT_DIR [--tags _all_p4] [--json combined.json]   (default tags: the P3 selections)
"""
import argparse, json, os

import numpy as np

TAGS = ["_p3", "_ext_p3", "_mp_p3", "_wide_p3", "_fix_p3", "_fix2_p3"]
# alpha Cen A: archive files are alpha Cen B; alpha Cen B: every exposure blended with A (4" in 2014-16) or saturated
EXCLUDE = {"HD128620", "HD128621"}
RUNS = ("vmic15", "vmicgbs", "fixteff")


def group(kind, ref):
    if ref["feh"] < -0.7 or kind == "metalpoor":
        return "metal-poor"
    if kind in ("giant", "giant_w") or ref["logg"] < 3.5:
        return "giants"
    if ref["teff"] >= 5900:
        return "F stars"
    return "G-K dwarfs"


def load(out, tags=TAGS):
    stars = {}
    for tag in tags:
        for run in RUNS:
            p = os.path.join(out, f"gbs_{run}{tag}.json")
            if not os.path.exists(p):
                continue
            for k, r in json.load(open(p))["results"].items():
                if k in EXCLUDE or "masked" not in r:
                    continue
                s = stars.setdefault(k, dict(kind=r["kind"], ref=r["ref"], src={}))
                s["ref"] = r["ref"]; s["src"][run] = tag
                s[run] = dict(dteff=r["masked"]["teff"] - r["ref"]["teff"], dfeh=r["masked"]["feh"] - r["ref"]["feh"],
                              chi2=r["masked"]["chi2_N"], mask=r["masked_frac"])
    return stars


def stats(v):
    v = np.asarray(v, float)
    if v.size == 0:
        return "—"
    if v.size == 1:
        return f"{v[0]:+.2f}" if abs(v[0]) < 5 else f"{v[0]:+.0f}"
    m, sc = v.mean(), v.std(ddof=1); e = sc / np.sqrt(v.size)
    return f"{m:+.3f} ± {e:.3f} (sc {sc:.3f})" if abs(m) < 5 else f"{m:+.0f} ± {e:.0f} (sc {sc:.0f})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out"); ap.add_argument("--json"); ap.add_argument("--tags", nargs="+", default=TAGS)
    args = ap.parse_args()
    S = load(args.out, args.tags)
    groups = {}
    for k, s in S.items():
        groups.setdefault(group(s["kind"], s["ref"]), []).append(k)
    print(f"{len(S)} stars (excluded {sorted(EXCLUDE)})\n")
    print(f"{'group':12s} {'N':>3s}  {'dTeff (xi 1.5)':24s} {'d[Fe/H] xi 1.5':26s} {'d[Fe/H] xi GBS':26s} {'d[Fe/H] Teff fixed':26s}")
    for g in ("giants", "F stars", "G-K dwarfs", "metal-poor"):
        ks = sorted(groups.get(g, []))
        col = lambda run, q: [S[k][run][q] for k in ks if run in S[k]]
        print(f"{g:12s} {len(ks):3d}  {stats(col('vmic15', 'dteff')):24s} {stats(col('vmic15', 'dfeh')):26s} "
              f"{stats(col('vmicgbs', 'dfeh')):26s} {stats(col('fixteff', 'dfeh')):26s}")
    print("\nTrends of d[Fe/H] (Teff fixed), all except metal-poor: slope per unit, from a straight-line fit")
    ks = [k for k in S if "fixteff" in S[k] and group(S[k]["kind"], S[k]["ref"]) != "metal-poor"]
    y = np.array([S[k]["fixteff"]["dfeh"] for k in ks])
    for q, unit, scale in (("teff", "1000 K", 1000.0), ("logg", "dex", 1.0), ("feh", "dex", 1.0)):
        x = np.array([S[k]["ref"][q] for k in ks]) / scale
        A = np.vstack([x, np.ones_like(x)]).T; c, res, *_ = np.linalg.lstsq(A, y, rcond=None)
        sig = np.sqrt(np.sum((y - A @ c) ** 2) / (len(y) - 2) * np.linalg.inv(A.T @ A)[0, 0])
        print(f"   vs {q:4s}: {c[0]:+.3f} ± {sig:.3f} per {unit}")
    for g in ("giants", "G-K dwarfs"):
        sub = [k for k in ks if group(S[k]["kind"], S[k]["ref"]) == g]
        x = np.array([S[k]["ref"]["teff"] for k in sub]) / 1000; yy = np.array([S[k]["fixteff"]["dfeh"] for k in sub])
        A = np.vstack([x, np.ones_like(x)]).T; c, *_ = np.linalg.lstsq(A, yy, rcond=None)
        sig = np.sqrt(np.sum((yy - A @ c) ** 2) / (len(yy) - 2) * np.linalg.inv(A.T @ A)[0, 0])
        print(f"   {g} only, vs teff: {c[0]:+.3f} ± {sig:.3f} per 1000 K (N {len(sub)}, Teff {1000 * x.min():.0f}-{1000 * x.max():.0f})")
    print("\nper star (Teff fixed d[Fe/H], chi2/N; xi 1.5 dTeff):")
    for g in ("giants", "F stars", "G-K dwarfs", "metal-poor"):
        for k in sorted(groups.get(g, []), key=lambda k: S[k]["ref"]["teff"]):
            s = S[k]; f = s.get("fixteff", {}); v = s.get("vmic15", {})
            print(f"   {g:10s} {k:9s} {s['ref']['teff']:5.0f} {s['ref']['logg']:4.2f} {s['ref']['feh']:+.2f} | "
                  f"fix {f.get('dfeh', np.nan):+.3f} chi2 {f.get('chi2', np.nan):5.1f} | dT {v.get('dteff', np.nan):+5.0f} | {s['src'].get('fixteff')}")
    if args.json:
        json.dump(dict(stars=S, groups=groups), open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
