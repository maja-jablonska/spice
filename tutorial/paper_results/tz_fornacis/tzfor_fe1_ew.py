"""TZ For [Fe/H] from Fe I equivalent widths, the way Gallenne et al. (2016) measured it, but with our models.

For each isolated Fe I line the equivalent width is measured on the observed composite spectrum and on two model dumps that
differ only in [Fe/H] (A and B, e.g. Gallenne +0.02 / -0.05 and the same shifted by -0.2): sum of (1 - F) over a
+-EW_HALF window in the primary's rest frame, averaged over the epochs where the whole window is good. Interpolating
log EW linearly in [Fe/H] between the two models gives each line's [Fe/H] (the composite dilution by the secondary is
the same in data and models). Lines are rejected below 5 mA or where the model EW hardly responds to [Fe/H].
Reported: median and mean +- error of the per-line values, and their slopes against excitation potential (Teff balance)
and reduced EW log(EW / lambda) (microturbulence balance), for a strict and a broad line selection.

    python tzfor_fe1_ew.py LINELIST OUT_PREFIX LABEL:VT:A.npz:B.npz:FEH_A:FEH_B [...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tzfor_fe1_compare import read_bsyn  # noqa: E402
from tzfor_species_compare import A_SUN, THETA  # noqa: E402

C_KMS = 299792.458
ISO_A, ISO_DEX = 0.30, 1.0
WL_RANGE = (4125.0, 6840.0)


def select(names, wl, chi, gf, weighted):
    """Isolated Fe I lines: no other line within +-ISO_A stronger than this one minus ISO_DEX (strength log gf - theta chi,
    plus log abundance relative to Fe and +1 for ions if weighted)."""
    elem = np.array([n.split()[0] for n in names]); ion = np.array([n.split()[1] if len(n.split()) > 1 else "I" for n in names])
    st = gf - THETA * chi
    if weighted:
        st = st + np.array([A_SUN.get(e, 2.0) - 7.50 for e in elem]) + np.where(ion == "II", 1.0, 0.0)
    o = np.argsort(wl); ws, ss = wl[o], st[o]
    fe = np.where((elem == "Fe") & (ion == "I") & (wl > WL_RANGE[0]) & (wl < WL_RANGE[1]))[0]
    keep = []
    for i in fe:
        lo, hi = np.searchsorted(ws, [wl[i] - ISO_A, wl[i] + ISO_A])
        oth = ss[lo:hi][np.abs(ws[lo:hi] - wl[i]) > 1e-4]
        if not np.any(oth > st[i] - ISO_DEX):
            keep.append(i)
    return np.asarray(keep)


CUTS = {"default": dict(half=0.20, min_ep=5, min_ew=5.0, min_slope=0.2),
        "relaxed": dict(half=0.15, min_ep=3, min_ew=3.0, min_slope=0.1)}


def ews(lines, lam, v1, good, spectra, half, min_ep):
    """Mean EW [mA] per line for each spectrum array (n_ep, n_px), over epochs where the +-half window is fully good;
    also the number of such epochs."""
    out = {k: np.full(lines.size, np.nan) for k in spectra}; n_ok = np.zeros(lines.size, int)
    dlam = np.gradient(lam)
    for j, w0 in enumerate(lines):
        acc = {k: [] for k in spectra}
        for e in range(good.shape[0]):
            rest = lam / (1 + v1[e] / C_KMS); m = np.abs(rest - w0) <= half
            if m.sum() < 8 or not good[e, m].all():
                continue
            dl = dlam[m] / (1 + v1[e] / C_KMS)
            for k, y in spectra.items():
                acc[k].append(1e3 * np.sum((1.0 - y[e, m]) * dl))
        n_ok[j] = len(acc[next(iter(spectra))])
        if n_ok[j] >= min_ep:
            for k in spectra:
                out[k][j] = np.mean(acc[k])
    return out, n_ok


def per_line_feh(E, fa, fb, n_ok, cut):
    la, lb, lo = np.log10(E["A"]), np.log10(E["B"]), np.log10(E["obs"])
    slope = (la - lb) / (fa - fb)
    reasons = dict(epochs=n_ok < cut["min_ep"], weak=(E["obs"] <= cut["min_ew"]) | (E["A"] <= cut["min_ew"]),
                   insensitive=~(slope > cut["min_slope"]))
    ok = np.isfinite(la) & np.isfinite(lb) & np.isfinite(lo) & ~reasons["epochs"] & ~reasons["weak"] & ~reasons["insensitive"]
    return np.where(ok, fa + (lo - la) / slope, np.nan), ok, reasons


def summarise(lines, chi, E, feh, ok):
    f = feh[ok]; x = chi[ok]; rew = np.log10(E["obs"][ok] * 1e-3 / lines[ok])
    if f.size < 5:
        return None
    med = np.median(f); mad = 1.4826 * np.median(np.abs(f - med)); clip = np.abs(f - med) < 3 * max(mad, 0.02)
    f, x, rew = f[clip], x[clip], rew[clip]
    return dict(n=int(f.size), n_usable=int(ok.sum()), median=float(np.median(f)), mean=float(f.mean()),
                err=float(f.std(ddof=1) / np.sqrt(f.size)), sd=float(f.std(ddof=1)),
                s_ep=float(np.polyfit(x, f, 1)[0]), s_rew=float(np.polyfit(rew, f, 1)[0]))


def main():
    """Arguments: LINELIST OUT_PREFIX then pairs LABEL:VT:A.npz:B.npz:FEH_A:FEH_B (A, B differ only in [Fe/H])."""
    linelist, prefix, *specs = sys.argv[1:]
    names, wl, chi, gf = read_bsyn(linelist)
    pairs = []
    for sp in specs:
        lab, vt, a_p, b_p, fa, fb = sp.split(":")
        pairs.append((lab, float(vt), a_p, b_p, float(fa), float(fb)))
    rows = {}
    for weighted in (False, True):
        sel = select(names, wl, chi, gf, weighted); sname = "broad" if weighted else "strict"
        for cname, cut in CUTS.items():
            print(f"\n=== {sname} selection ({sel.size} isolated Fe I lines), {cname} cuts {cut}")
            for lab, vt, ap, bp, fa, fb in pairs:
                A, B = np.load(ap, allow_pickle=True), np.load(bp, allow_pickle=True)
                assert np.array_equal(A["obs"], B["obs"], equal_nan=True)
                lam = 10 ** A["logwl"]; good = A["good"] & B["good"] & np.isfinite(A["obs"])
                E, n_ok = ews(wl[sel], lam, A["rv"][0], good, {"obs": A["obs"], "A": A["model"], "B": B["model"]}, cut["half"], cut["min_ep"])
                feh, ok, why = per_line_feh(E, fa, fb, n_ok, cut)
                r = summarise(wl[sel], chi[sel], E, feh, ok)
                lost = ", ".join(f"{k} {int(v.sum())}" for k, v in why.items())
                if r is None:
                    print(f"  {lab:14s} vt {vt:4.2f}: only {int(ok.sum())} usable (removed by: {lost})"); continue
                rows[(sname, cname, lab)] = dict(vt=vt, **r)
                print(f"  {lab:14s} vt {vt:4.2f}: {r['n']:3d} lines ({r['n_usable']} usable; removed by: {lost}): [Fe/H] median {r['median']:+.3f}, "
                      f"mean {r['mean']:+.3f} +- {r['err']:.3f} (sd {r['sd']:.3f}); slope vs EP {r['s_ep']:+.4f} dex/eV, vs log(EW/lambda) {r['s_rew']:+.3f}")
                np.savez(f"{prefix}_{sname}_{cname}_{lab}.npz", wl=wl[sel], chi=chi[sel], loggf=gf[sel], feh=feh, ok=ok, n_epochs=n_ok,
                         ew_obs=E["obs"], ew_A=E["A"], ew_B=E["B"])
            # microturbulence balance: the vt where the trend with reduced EW vanishes, and [Fe/H] there
            pts = sorted((v["vt"], v["s_rew"], v["median"], v["mean"]) for (sn, cn, _), v in rows.items() if sn == sname and cn == cname)
            if len(pts) >= 2:
                vts, srew, med, mean = map(np.array, zip(*pts))
                if srew.min() < 0 < srew.max():
                    vt0 = float(np.interp(0.0, srew[::-1], vts[::-1]) if srew[0] > srew[-1] else np.interp(0.0, srew, vts))
                    print(f"  BALANCE: trend vs reduced EW vanishes at vt = {vt0:.2f} km/s -> [Fe/H] median {np.interp(vt0, vts, med):+.3f}, "
                          f"mean {np.interp(vt0, vts, mean):+.3f}")
                else:
                    print(f"  BALANCE: trend vs reduced EW does not change sign over vt {vts.min():.2f}-{vts.max():.2f} (slopes {np.round(srew, 3).tolist()})")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for lab, vt, *_ in pairs:
            z = np.load(f"{prefix}_broad_relaxed_{lab}.npz") if Path(f"{prefix}_broad_relaxed_{lab}.npz").exists() else None
            if z is None:
                continue
            k = z["ok"]
            ax[0].plot(z["chi"][k], z["feh"][k], "o", ms=3, label=f"vt {vt}"); ax[1].plot(np.log10(z["ew_obs"][k] * 1e-3 / z["wl"][k]), z["feh"][k], "o", ms=3)
        ax[0].set_xlabel("excitation potential [eV]"); ax[1].set_xlabel("log(EW / lambda)")
        for a in ax: a.set_ylabel("[Fe/H] per line"); a.axhline(0.02, color="k", lw=0.8, ls="--")
        ax[0].legend(fontsize=8); fig.suptitle("TZ For Fe I EW abundances (broad selection, relaxed cuts); dashed: Gallenne +0.02")
        fig.tight_layout(); fig.savefig(f"{prefix}_broad_relaxed.png", dpi=130); print("wrote", f"{prefix}_broad_relaxed.png")
    except Exception as exc:
        print("plot skipped:", exc)


if __name__ == "__main__":
    main()
