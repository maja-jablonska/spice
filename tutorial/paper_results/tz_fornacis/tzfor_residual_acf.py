"""Correlation length of the TZ For spectral residuals, and the photometry / spectroscopy weight it implies.

The fit's loss is chi2_phot + spec_scale * chi2_spec with per-pixel sigmas. Adjacent HARPS pixels are not independent (about
3 pixels per resolution element, and model-mismatch residuals span whole line profiles), and chi2_spec/N ~ 9 at the optimum,
so the plain sum lets 3 M spectral pixels outvote 2,300 photometric points. Standard fix: rescale each dataset to reduced
chi2 = 1 and count the spectra by their effective number of independent points N_eff = N / L, with L the integrated
autocorrelation length of the normalised residuals r = (model - obs) / sigma along wavelength:

    L = 1 + 2 sum_{k>=1} rho(k)   (summed until rho first drops below 0, Sokal-style window)

    spec_scale = (chi2_phot / N_phot) / ((chi2_spec / N_spec) * L)

rho(k) is pooled over all epochs, using only pixel pairs (i, i+k) that are both good.

    python tzfor_residual_acf.py DUMP.npz CHI2NU_PHOT OUT.json [MAX_LAG]
"""
from __future__ import annotations

import json
import sys

import numpy as np


def acf(r, good, max_lag):
    """Pooled autocorrelation of r (n_ep, n_px) over pairs of good pixels, lags 0..max_lag (mean removed per epoch)."""
    r = np.where(good, r, 0.0)
    mu = np.array([r[e, good[e]].mean() if good[e].any() else 0.0 for e in range(r.shape[0])])
    r = np.where(good, r - mu[:, None], 0.0)
    var = np.sum(r * r) / good.sum()
    rho = np.ones(max_lag + 1)
    for k in range(1, max_lag + 1):
        g = good[:, :-k] & good[:, k:]
        rho[k] = np.sum((r[:, :-k] * r[:, k:])[g]) / g.sum() / var
    return rho


def main():
    dump, chi2nu_ph, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    max_lag = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    D = np.load(dump, allow_pickle=True)
    good = D["good"] & np.isfinite(D["model"]) & np.isfinite(D["obs"]) & (D["sigma"] > 0)
    r = (D["model"] - D["obs"]) / np.where(good, D["sigma"], 1.0)
    chi2nu_sp = float(np.mean(r[good] ** 2))
    rho = acf(r, good, max_lag)
    neg = np.where(rho[1:] < 0)[0]
    kmax = int(neg[0]) if neg.size else max_lag          # last lag before the first negative value
    L = float(1.0 + 2.0 * np.sum(rho[1:kmax + 1]))
    dlogwl = float(np.median(np.diff(D["logwl"])))
    px_kms = dlogwl * np.log(10) * 299792.458
    res = dict(dump=dump, n_spec_pixels=int(good.sum()), chi2nu_spec=chi2nu_sp, chi2nu_phot=chi2nu_ph, L_pixels=L,
               window_lag=kmax, pixel_kms=px_kms, L_kms=L * px_kms, rho_first10=[float(x) for x in rho[:11]],
               spec_scale_white=chi2nu_ph / chi2nu_sp, spec_scale=chi2nu_ph / (chi2nu_sp * L))
    # the same per epoch, to see whether one epoch dominates
    res["L_per_epoch"] = []
    for e in range(r.shape[0]):
        re = acf(r[e:e + 1], good[e:e + 1], min(max_lag, 60)); ne = np.where(re[1:] < 0)[0]; ke = int(ne[0]) if ne.size else re.size - 1
        res["L_per_epoch"].append(float(1 + 2 * np.sum(re[1:ke + 1])))
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "L_per_epoch"}, indent=1))
    print("L per epoch:", np.round(res["L_per_epoch"], 1).tolist())
    print(f"SPEC_SCALE {res['spec_scale']:.6g}")


if __name__ == "__main__":
    main()
