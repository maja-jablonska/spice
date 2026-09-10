"""What PHOEBE alone can say about TZ For's temperatures.

PHOEBE (ck2004 atmospheres, same fixed geometry as the SPICE fits) has no
line-profile synthesis, so its temperature information is Clausen's b, y
photometry alone. Map chi2 over the (Teff1, Teff2) plane -- zero points and the
phase offset optimised per model, third light 0 -- fit a quadratic, and read off
the covariance: the marginal errors on either temperature, and the conditional
error on Teff2 when Teff1 is pinned externally. Compare with the SPICE joint
photometry + spectroscopy jackknife errors (Teff1 +-46 K, Teff2 +-38 K).

CPU-only by necessity (PHOEBE has no GPU path and is not on Gadi).
"""
import argparse, os, pickle, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import tzfor_constants as K
from tzfor_fit_radii import observed
SIG = {"b": 0.0041, "y": 0.0035}          # Clausen out-of-eclipse scatter per band


def chi2_weighted(ph_m, flux_m, ph_o, mag_o, dphi_grid):
    """chi2 in units of the per-band noise, zero point (median) and phase offset optimised."""
    best = None
    for dphi in dphi_grid:
        tot = 0.0
        for k, v in flux_m.items():
            f = np.interp((ph_o - dphi) % 1.0, ph_m, v, period=1.0)
            m = -2.5 * np.log10(f / np.median(f))
            resid = mag_o[k] - m
            tot += float(np.sum((resid - np.median(resid)) ** 2)) / SIG[k] ** 2
        if best is None or tot < best[0]:
            best = (tot, float(dphi))
    return best

ap = argparse.ArgumentParser()
ap.add_argument("--teff1", type=float, nargs="+", default=[4740., 4815., 4890., 4965., 5040.])
ap.add_argument("--offsets", type=float, nargs="+", default=[-150., -90., -30., 30., 90., 150.], help="Teff2 offsets from the SPICE photometric valley 6360 + 1.66 (Teff1 - 4888)")
ap.add_argument("--abun", type=float, default=-0.23)
ap.add_argument("--n-phase", type=int, default=600)
ap.add_argument("--ntriangles", type=int, default=800)
ap.add_argument("--out", default=str(HERE / "tzfor_aemu_out" / "phoebe_teff_surface.pkl"))
args = ap.parse_args()
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import phoebe; phoebe.logger(clevel="ERROR")
import tzfor_phoebe_aemu_photometry as M

ph_o, mag_o = observed()
N = sum(len(v) for v in mag_o.values())
dphi_grid = np.linspace(0.648, 0.668, 201)
b = M.build_bundle(args.ntriangles, "roche")
for comp in ("primary", "secondary"):
    try:
        b.set_value(f"abun@{comp}@component", args.abun)
    except Exception as e:  # noqa: BLE001
        print(f"abun not set for {comp}: {e}")
t0 = b.get_value("t0_supconj@binary@component")
ts = t0 + np.linspace(0.0, K.PERIOD_DAYS, args.n_phase, endpoint=False)
for band, ds in (("Stromgren:b", "lcb"), ("Stromgren:y", "lcy")):
    b.add_dataset("lc", compute_times=ts, passband=band, dataset=ds)
    b.set_value_all("pblum_mode", dataset=ds, value="absolute")
ph_m = ((ts - t0) % K.PERIOD_DAYS) / K.PERIOD_DAYS; order = np.argsort(ph_m); ph_m = ph_m[order]

rows = []; curves = {}
for T1 in args.teff1:
    for off in args.offsets:
        T2 = 6360.0 + 1.66 * (T1 - 4888.0) + off
        t = time.time()
        b.set_value("teff@primary@component", T1); b.set_value("teff@secondary@component", T2)
        b.run_compute(irrad_method="horvat", ltte=False)
        flux = {k: np.asarray(b.get_value(f"fluxes@{ds}@model"))[order] for k, ds in (("b", "lcb"), ("y", "lcy"))}
        chi2, dphi = chi2_weighted(ph_m, flux, ph_o, mag_o, dphi_grid)
        rows.append((T1, T2, chi2, dphi)); curves[(T1, T2)] = flux
        print(f"Teff1 {T1:.0f} Teff2 {T2:.0f}: chi2/N {chi2 / N:.4f}  dphi {dphi:.4f}  ({time.time() - t:.0f}s)", flush=True)
        pickle.dump(dict(rows=rows, N=N, args=vars(args), ph_m=ph_m, curves=curves, ph_o=np.asarray(ph_o), mag_o={k: np.asarray(v) for k, v in mag_o.items()}), open(args.out, "wb"), protocol=4)

# ---- chi2 surfaces for data subsets -> covariance each; spread = PHOEBE's systematics-inclusive error ----
def surface(sel_bands, phase_mask):
    pts = []
    for (T1, T2), flux in curves.items():
        best = None
        for dphi in dphi_grid:
            tot = 0.0
            for k in sel_bands:
                f = np.interp((ph_o - dphi) % 1.0, ph_m, flux[k], period=1.0)
                m = -2.5 * np.log10(f / np.median(f)); resid = (mag_o[k] - m)[phase_mask]
                tot += float(np.sum((resid - np.median(resid)) ** 2)) / SIG[k] ** 2
            if best is None or tot < best: best = tot
        pts.append((T1, T2, best))
    A = np.array(pts); x, y, c = A[:, 0] - 4890., A[:, 1] - 6400., A[:, 2]
    X = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], 1); coef, *_ = np.linalg.lstsq(X, c, rcond=None)
    Hs = np.array([[2 * coef[3], coef[4]], [coef[4], 2 * coef[5]]]); ev = np.linalg.eigvalsh(Hs)
    if not np.all(ev > 0):
        return None
    xmin = np.linalg.solve(Hs, -coef[1:3]); cmin = coef[0] + coef[1:3] @ xmin + 0.5 * xmin @ Hs @ xmin
    cov = 2.0 * np.linalg.inv(Hs); sig = np.sqrt(np.diag(cov))
    return dict(T1=4890 + xmin[0], T2=6400 + xmin[1], chi2N=cmin / int(phase_mask.sum()) / len(sel_bands) * 1.0, sig=sig, corr=cov[0, 1] / (sig[0] * sig[1]),
                cond=np.sqrt(2.0 / Hs[1, 1]), slope=-Hs[0, 1] / Hs[1, 1], rms=np.std(c - X @ coef))
ph_o = np.asarray(ph_o); allm = np.ones(ph_o.size, bool)
deep = np.abs(((ph_o - 0.158) + 0.5) % 1.0 - 0.5) < 0.04; shallow = np.abs(((ph_o - 0.658) + 0.5) % 1.0 - 0.5) < 0.04
subsets = {"all (b+y)": (("b", "y"), allm), "b only": (("b",), allm), "y only": (("y",), allm),
           "no deep eclipse": (("b", "y"), ~deep), "no shallow eclipse": (("b", "y"), ~shallow), "eclipses only": (("b", "y"), deep | shallow)}
print(f"\n{'subset':<20}{'Teff1':>7}{'Teff2':>7}{'sig1':>6}{'sig2':>6}{'corr':>7}{'T2|T1':>7}{'slope':>7}")
res = {}
for name, (bands, mask) in subsets.items():
    r = surface(bands, mask); res[name] = r
    if r is None:
        print(f"{name:<20}   surface not convex within the grid"); continue
    print(f"{name:<20}{r['T1']:7.0f}{r['T2']:7.0f}{r['sig'][0]:6.0f}{r['sig'][1]:6.0f}{r['corr']:7.3f}{r['cond']:7.0f}{r['slope']:7.2f}")
ok = [r for r in res.values() if r is not None]
T = np.array([[r["T1"], r["T2"]] for r in ok])
print(f"\nspread across data subsets (std): Teff1 {T[:, 0].std(ddof=1):.0f} K  Teff2 {T[:, 1].std(ddof=1):.0f} K   (range Teff1 {T[:, 0].min():.0f}-{T[:, 0].max():.0f}, Teff2 {T[:, 1].min():.0f}-{T[:, 1].max():.0f})")
print("formal Delta-chi2=1 errors above are for comparison with SPICE's formal +-1 K; the subset spread is the analogue of SPICE's window jackknife (+-46 / +-38 K)")
pickle.dump(dict(rows=rows, N=N, args=vars(args), ph_m=ph_m, curves=curves, subsets=res), open(args.out, "wb"), protocol=4)
print("saved", args.out)
