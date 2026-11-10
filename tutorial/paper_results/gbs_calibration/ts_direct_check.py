"""Emulator-free check of the [Fe/H] zero-point: Turbospectrum flux spectra of the Sun (and of HD168723, the
TZ For giant's benchmark twin) compared directly with the HARPS spectra.

The benchmark fits (gbs_fit.py) return [Fe/H] about 0.2 dex below the GBS values, the Sun included, with Teff
right. The emulator matches Turbospectrum to 0.2 % rms (injection test), so the question is whether the
training-grid physics itself (MARCS, line list, abundances, 1D LTE) makes the lines too strong.

synth:   Turbospectrum disc-integrated flux on MARCS atmospheres interpolated (interpol_modeles, t02, plane-parallel
         at log g >= 3, spherical below, as the interpolated-atmosphere grid) to the exact Teff/log g and to each
         [Fe/H] of the grid, the production line lists and abundance block (scaled solar, Magg 2022), for
         [Fe/H] in --feh and microturbulence in --xi. Writes OUT/ts_<star>.npz.
compare: for each observed spectrum: RV by CCF, then for every model a broadening scan (rotation vsini fixed at
         the GBS value + a free Gaussian), a quadratic continuum per window (linear least squares, as the fits),
         sigma = hypot(err, 0.005). chi2([Fe/H]) per window and overall -> best [Fe/H] (parabola); residual
         vs model line depth; best [Fe/H] line by line (minima deeper than 0.05) in bins of line depth.

Usage (Gadi; offnode_synth driver, astro env):
  python ts_direct_check.py synth --out DIR [--workers 24]
  python ts_direct_check.py compare --out DIR --sun-npz data/gbs_harps_sun.npz --gbs-npz data/gbs_harps.npz
"""
import argparse, json, math, os, sys, time
from multiprocessing import Pool

import numpy as np

C_KMS = 299792.458; R_HARPS = 115000.0
WINDOWS = [(5160, 5200), (5380, 5420), (6100, 6140), (6400, 6440)]
LAM = (5140.0, 6460.0)
STARS = {"sun": dict(teff=5771, logg=4.44, vsini=1.6), "HD168723": dict(teff=4801, logg=2.92, vsini=1.0)}
ABUND = dict(a=0.0, c=0.0, n=0.0, o=0.0, r=0.0, s=0.0)


def synth(args):
    sys.path.insert(0, "/scratch/y89/mj8805/emu_check")
    import offnode_synth as OS
    RT = OS.RT
    store_labels = RT._scan_available_turb_labels(OS.STORE)
    jobs, meta = [], {}
    for star, p in STARS.items():
        for feh in args.feh:
            for xi in args.xi:
                name = f"{star}_z{feh:+.2f}_x{xi:.2f}"
                root = os.path.join(args.out, "runs", name); atm = os.path.join(root, "atm"); os.makedirs(atm, exist_ok=True)
                paths, geom, lab, corners = OS.find_brackets(p["teff"], p["logg"], feh, "02", store_labels)
                if lab != "02" or geom != ("p" if p["logg"] >= 3 else "s"):
                    print(f"[note] {name}: bracket {geom} t{lab} differs from the grid convention", flush=True)
                target = os.path.join(atm, RT.get_model_filename(p["teff"], p["logg"], feh, lab))
                ok, log = OS.interpolate(paths, p["teff"], p["logg"], feh, target, root)
                if not ok:
                    raise RuntimeError(f"interpolation failed for {name}: {log}")
                config = OS.build_config(root, atm, "", lam=LAM, flux_mode=True)
                jobs.append((name, dict(teff=p["teff"], logg=p["logg"], feh=feh, turbvel=xi, t_value=lab, **ABUND), config))
                meta[name] = dict(star=star, feh=feh, xi=xi, geom=geom, label=lab, corners=corners)
                print(f"[prepared] {name}: {geom} t{lab} corners {corners}", flush=True)
    t0 = time.time()
    with Pool(max(1, min(args.workers, len(jobs)))) as pool:
        results = pool.map(OS._run_one, jobs, chunksize=1)
    print(f"synthesis wall {time.time() - t0:.0f}s for {len(jobs)} runs", flush=True)
    spec = {}
    for (name, _, _), res in zip(jobs, results):
        path = res.get("output_path")
        if not (path and os.path.exists(path)):
            raise RuntimeError(f"{name} failed: {str(res.get('message', ''))[:300]}")
        wl, _, F, Fn = OS.parse_flux(path); spec[name] = (wl, Fn[0])
    for star in STARS:
        wl0 = spec[f"{star}_z{args.feh[0]:+.2f}_x{args.xi[0]:.2f}"][0]
        flux = np.zeros((len(args.feh), len(args.xi), wl0.size), np.float32)
        for i, feh in enumerate(args.feh):
            for j, xi in enumerate(args.xi):
                wl, fn = spec[f"{star}_z{feh:+.2f}_x{xi:.2f}"]; assert np.array_equal(wl, wl0); flux[i, j] = fn
        np.savez(os.path.join(args.out, f"ts_{star}.npz"), wl=wl0, flux=flux, feh=np.asarray(args.feh), xi=np.asarray(args.xi),
                 meta=json.dumps({k: v for k, v in meta.items() if v["star"] == star}, default=str), **STARS[star])
        print(f"wrote ts_{star}.npz: {flux.shape}", flush=True)


def rot_kernel(vsini, dv, eps=0.6):
    n = max(1, int(math.ceil(vsini / dv))); x = np.arange(-n, n + 1) * dv / max(vsini, 1e-6)
    k = np.where(np.abs(x) < 1, (2 * (1 - eps) * np.sqrt(np.clip(1 - x * x, 0, None)) + 0.5 * math.pi * eps * (1 - x * x)), 0.0)
    return k / k.sum()


def gauss_kernel(sig_v, dv):
    n = int(math.ceil(5 * sig_v / dv)); x = np.arange(-n, n + 1) * dv; k = np.exp(-0.5 * (x / sig_v) ** 2); return k / k.sum()


class Models:
    """TS normalised flux on a uniform log-lambda grid, broadened on demand (cached)."""
    def __init__(self, path):
        d = np.load(path, allow_pickle=True)
        self.feh, self.xi, self.vsini = d["feh"], d["xi"], float(d["vsini"])
        wl = d["wl"]; self.dlog = math.log10(1 + 0.25 / C_KMS)            # 0.25 km/s pixels
        self.lw = np.arange(math.log10(wl[0]), math.log10(wl[-1]), self.dlog); self.dv = 0.25
        self.flux = np.array([[np.interp(10 ** self.lw, wl, d["flux"][i, j]) for j in range(len(self.xi))] for i in range(len(self.feh))])
        self.rot = np.array([[np.convolve(self.flux[i, j], rot_kernel(self.vsini, self.dv), "same") for j in range(len(self.xi))]
                             for i in range(len(self.feh))])
        self._c = {}

    def get(self, i, j, fwhm):
        key = (i, j, round(fwhm, 3))
        if key not in self._c:
            self._c[key] = np.convolve(self.rot[i, j], gauss_kernel(fwhm / 2.3548, self.dv), "same")
        return self._c[key]

    def at(self, i, j, fwhm, wl_obs, rv):
        return np.interp(np.log10(wl_obs / (1 + rv / C_KMS)), self.lw, self.get(i, j, fwhm))


def cont_fit(ob, mod, x, w, deg=2):
    A = mod[:, None] * np.vander(x, deg + 1) * w[:, None]
    c, *_ = np.linalg.lstsq(A, ob * w, rcond=None); return mod * (np.vander(x, deg + 1) @ c)


def window_chi2(M, i, j, fwhm, rv, wl, ob, sg, good):
    out, fits = [], []
    for a, b in WINDOWS:
        m = (wl >= a) & (wl <= b) & good
        x = (wl[m] - 0.5 * (a + b)) / (0.5 * (b - a)); mod = cont_fit(ob[m], M.at(i, j, fwhm, wl[m], rv), x, 1 / sg[m])
        out.append(float(np.sum(((ob[m] - mod) / sg[m]) ** 2))); fits.append((m, mod))
    return np.array(out), fits


def parabola_min(x, y):
    k = int(np.argmin(y))
    if k == 0 or k == len(y) - 1:
        return float(x[k]), True                       # at the grid edge
    a, b, c = np.polyfit(x[k - 1:k + 2], y[k - 1:k + 2], 2); return float(-b / (2 * a)), False


def compare_one(M, wl, fl, er, label):
    good = np.isfinite(fl) & np.isfinite(er) & (er > 0)
    sel = np.zeros_like(good)
    for a, b in WINDOWS:
        sel |= (wl >= a - 2) & (wl <= b + 2)
    wl, fl, er, good = wl[sel], fl[sel], er[sel], good[sel]
    sg = np.hypot(np.where(good, er, 1.0), 0.005)
    i0 = int(np.argmin(np.abs(M.feh - 0.0))); j0 = len(M.xi) - 1 if len(M.xi) == 1 else 0
    # RV: CCF against the solar-metallicity model in the Mg b window
    m = (wl >= 5160) & (wl <= 5200) & good
    def ccf(v):
        mod = M.at(i0, j0, 5.0, wl[m], v); o = fl[m] / np.percentile(fl[m], 95)
        return np.corrcoef(1 - o, 1 - mod)[0, 1]
    vg = np.arange(-150, 150, 1.0); j = int(np.argmax([ccf(v) for v in vg])); vf = np.arange(vg[j] - 2, vg[j] + 2, 0.05)
    rv = float(vf[np.argmax([ccf(v) for v in vf])])
    fwhms = np.arange(0.5, 12.01, 0.25)                   # from 0.5: an FTS atlas has no instrumental broadening
    res = dict(label=label, rv=rv, xi={})
    for jx, xi in enumerate(M.xi):
        chi = np.zeros((len(M.feh), len(WINDOWS))); bw = np.zeros(len(M.feh))
        for i in range(len(M.feh)):
            tot = [window_chi2(M, i, jx, fw, rv, wl, fl, sg, good)[0].sum() for fw in fwhms]
            k = int(np.argmin(tot)); bw[i] = fwhms[k]; chi[i] = window_chi2(M, i, jx, bw[i], rv, wl, fl, sg, good)[0]
        best, edge = parabola_min(M.feh, chi.sum(1)); per_win = [parabola_min(M.feh, chi[:, w])[0] for w in range(len(WINDOWS))]
        # masked pass: drop pixels the best model misses by > 0.05 and > 4 sigma (as the fits), then redo
        ib = int(np.argmin(chi.sum(1))); _, fits = window_chi2(M, ib, jx, bw[ib], rv, wl, fl, sg, good)
        good2 = good.copy()
        for mm, mod in fits:
            idx = np.where(mm)[0]; bad = (np.abs(fl[idx] - mod) > 0.05) & (np.abs(fl[idx] - mod) > 4 * sg[idx]); good2[idx[bad]] = False
        chi2m = np.array([window_chi2(M, i, jx, bw[i], rv, wl, fl, sg, good2)[0].sum() for i in range(len(M.feh))])
        best_m, edge_m = parabola_min(M.feh, chi2m)
        # residual vs model depth, at [Fe/H] = 0 and at the grid point nearest the best fit
        depth = {}
        for tag, ii in (("feh0", i0), ("best", int(np.argmin(np.abs(M.feh - best))))):
            _, fits = window_chi2(M, ii, jx, bw[ii], rv, wl, fl, sg, good2)
            dd, rr = [], []
            for mm, mod in fits:
                fn = M.at(ii, jx, bw[ii], wl[mm], rv); cont = mod / fn               # fitted continuum polynomial
                dd.append(1 - fn); rr.append((fl[mm] - mod) / cont)                  # residual in units of the continuum
            dd, rr = np.concatenate(dd), np.concatenate(rr); edges = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0])
            depth[tag] = [dict(lo=float(a), hi=float(b), n=int(((dd >= a) & (dd < b)).sum()),
                               mean_resid=float(np.mean(rr[(dd >= a) & (dd < b)])) if ((dd >= a) & (dd < b)).any() else None)
                          for a, b in zip(edges[:-1], edges[1:])]
        # line by line: minima of the solar-metallicity model deeper than 0.05, +-2 FWHM, continuum fixed from the best overall fit
        lines = []
        mods = []
        for i in range(len(M.feh)):
            _, fits = window_chi2(M, i, jx, bw[i], rv, wl, fl, sg, good2); mods.append(fits)
        for w, (mm, _) in enumerate(mods[i0]):
            idx = np.where(mm)[0]; f0 = M.at(i0, jx, bw[i0], wl[idx], rv)
            mins = [k for k in range(2, idx.size - 2) if f0[k] < f0[k - 1] and f0[k] <= f0[k + 1] and 1 - f0[k] > 0.05]
            half = max(3, int(round(2 * bw[i0] / C_KMS * wl[idx].mean() / 0.01)))
            for k in mins:
                sl = slice(max(0, k - half), min(idx.size, k + half + 1))
                y = np.array([np.sum(((fl[idx][sl] - mods[i][w][1][sl]) / sg[idx][sl]) ** 2) for i in range(len(M.feh))])
                b_, e_ = parabola_min(M.feh, y); lines.append(dict(wl=float(wl[idx][k]), depth=float(1 - f0[k]), feh=b_, edge=e_))
        bins = [(0.05, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.0)]
        by_depth = [dict(lo=a, hi=b, n=len(L), median_feh=float(np.median([l["feh"] for l in L])) if L else None,
                         frac_edge=float(np.mean([l["edge"] for l in L])) if L else None)
                    for a, b in bins for L in [[l for l in lines if a <= l["depth"] < b]]]
        res["xi"][f"{xi:.2f}"] = dict(best_feh=best, best_feh_masked=best_m, at_edge=edge or edge_m, fwhm=bw.tolist(), chi2=chi.sum(1).tolist(),
                                      chi2_masked=chi2m.tolist(), per_window=per_win, masked_frac=float(1 - good2.sum() / max(good.sum(), 1)),
                                      depth_resid=depth, lines_by_depth=by_depth, n_lines=len(lines), lines=lines)
        print(f"{label:11s} rv {rv:+7.2f} xi {xi:.2f}: best [Fe/H] {best:+.3f} (masked {best_m:+.3f}){' EDGE' if edge or edge_m else ''} "
              f"FWHM {bw[ib]:.2f} km/s | windows " + " ".join(f"{v:+.2f}" for v in per_win) + " | lines by depth "
              + " ".join(f"[{d['lo']:.2f},{d['hi']:.2f}) n{d['n']} {d['median_feh']:+.2f}" if d["n"] else f"[{d['lo']:.2f},{d['hi']:.2f}) -" for d in by_depth), flush=True)
    return res


def compare(args):
    out = {}
    Msun = Models(os.path.join(args.out, "ts_sun.npz")); print(f"sun models [Fe/H] {Msun.feh.tolist()} xi {Msun.xi.tolist()}", flush=True)
    S = np.load(args.sun_npz, allow_pickle=True)
    for k in sorted({f.split("/")[0] for f in S.files if "/" in f}):
        out[k] = compare_one(Msun, *(np.asarray(S[f"{k}/{q}"], float) for q in ("wl", "flux", "err")), k)
    if args.gbs_npz:
        Mg = Models(os.path.join(args.out, "ts_HD168723.npz")); G = np.load(args.gbs_npz, allow_pickle=True)
        out["HD168723"] = compare_one(Mg, *(np.asarray(G[f"HD168723/{q}"], float) for q in ("wl", "flux", "err")), "HD168723")
    print("summary: best [Fe/H] (plain / masked) per xi; GBS values Sun 0.00, HD168723 -0.30", flush=True)
    for xi in sorted({x for r in out.values() for x in r["xi"]}):
        for grp in dict.fromkeys(k.split("_")[0] for k in out):
            v = [(r["xi"][xi]["best_feh"], r["xi"][xi]["best_feh_masked"]) for k, r in out.items() if k.split("_")[0] == grp and xi in r["xi"]]
            if v:
                print(f"  xi {xi} {grp:9s} n{len(v)} {np.mean([a for a, _ in v]):+.3f} / {np.mean([b for _, b in v]):+.3f}", flush=True)
    path = os.path.join(args.out, f"ts_direct_check{args.tag}.json"); json.dump(out, open(path, "w"), indent=1)
    print("saved", path, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["synth", "compare"]); ap.add_argument("--out", required=True)
    ap.add_argument("--feh", type=float, nargs="*", default=[-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1])
    ap.add_argument("--xi", type=float, nargs="*", default=[1.09, 1.5])
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--sun-npz"); ap.add_argument("--gbs-npz"); ap.add_argument("--tag", default="", help="output suffix")
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
    synth(args) if args.mode == "synth" else compare(args)


if __name__ == "__main__":
    main()
