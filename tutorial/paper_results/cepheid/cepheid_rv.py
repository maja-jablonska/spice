"""Radial-velocity / line-profile estimators for the Cepheid p-factor analysis.

Single source of truth for the estimators that had been copy-pasted (and
silently diverged) between ``cepheid_pfactor_estimators.ipynb`` and
``cepheid_three_sources_three_lines.ipynb``. These are the canonical
``cepheid_pfactor_estimators.ipynb`` implementations; importing them keeps the
two notebooks numerically consistent (e.g. the float64 cast in ``rv_fft`` and
the ``slope_p`` ``g.sum() < 5`` cutoff now apply in both places).
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal.windows import tukey

C_KMS = 299792.458


def slope_p(rv, truth, min_rv=1e-3):
    rv = np.asarray(rv)[1:]; tr = np.asarray(truth)[1:]
    g = np.isfinite(rv) & np.isfinite(tr) & (np.abs(rv) > min_rv)
    if g.sum() < 5: return float('nan')
    return float(np.sum(rv[g] * tr[g]) / np.sum(rv[g] ** 2))


def _gauss_peak_fit(v, c, top_frac=0.5, fallback=None):
    k = int(np.nanargmax(c))
    if fallback is None: fallback = float(v[k])
    c_med = float(np.nanmedian(c)); c_max = float(c[k])
    if not np.isfinite(c_max) or c_max <= c_med: return fallback
    thresh = c_med + top_frac * (c_max - c_med); left = k
    while left > 0 and c[left-1] > thresh: left -= 1
    right = k
    while right < len(c) - 1 and c[right+1] > thresh: right += 1
    pad = 2; left = max(0, left - pad); right = min(len(c) - 1, right + pad)
    if right - left + 1 < 5: return fallback
    vv, cc = v[left:right+1], c[left:right+1]
    def m(vv, A, v0, sig, c0): return A * np.exp(-0.5 * ((vv - v0) / sig) ** 2) + c0
    dv = abs(v[1] - v[0]); w = max(2 * dv, 0.25 * (vv[-1] - vv[0]))
    p0 = [c_max - c_med, fallback, w, c_med]
    bnds = ([0., vv[0], dv / 2, -np.inf], [np.inf, vv[-1], max(vv[-1] - vv[0], dv), np.inf])
    try: popt, _ = curve_fit(m, vv, cc, p0=p0, bounds=bnds, maxfev=5000)
    except Exception: return fallback
    return float(popt[1])


def _normalized_lag_ccf(t, o, max_lag):
    lags = np.arange(-max_lag, max_lag + 1)
    ccf = np.full_like(lags, np.nan, dtype=float)
    for j, lag in enumerate(lags):
        if lag > 0: tt, oo = t[:-lag], o[lag:]
        elif lag < 0: tt, oo = t[-lag:], o[:lag]
        else: tt, oo = t, o
        good = np.isfinite(tt) & np.isfinite(oo); tt, oo = tt[good], oo[good]
        if tt.size < 5: continue
        tt = tt - np.mean(tt); oo = oo - np.mean(oo)
        d = np.sqrt(np.sum(tt ** 2) * np.sum(oo ** 2))
        if d > 0: ccf[j] = np.sum(tt * oo) / d
    return lags, ccf


def rv_fft(template, observed, wl, velocity_max=150.0, oversample=4, apodize=0.1):
    # CAST TO FLOAT64. ls.wavelengths is stored as float32; np.log(float32) loses
    # precision in the (10⁻⁷, 10⁻⁶) range that dv_pix lives in -> the velocity
    # grid spacing collapses to a single float32 step and recovered velocities
    # come out 2x too large.
    wl = np.asarray(wl, dtype=float)
    template = np.asarray(template, dtype=float)
    observed = np.asarray(observed, dtype=float)
    lnwl = np.log(wl); n = int(oversample * len(wl)); n += n % 2
    ln = np.linspace(lnwl[0], lnwl[-1], n); dv = C_KMS * (ln[1] - ln[0])
    t = interp1d(lnwl, template, kind='linear', bounds_error=False, fill_value=np.nan)(ln)
    o = interp1d(lnwl, observed,  kind='linear', bounds_error=False, fill_value=np.nan)(ln)
    g = np.isfinite(t) & np.isfinite(o); idx = np.flatnonzero(g)
    s = slice(idx[0], idx[-1] + 1); t = t[s].copy(); o = o[s].copy()
    for a in (t, o):
        bad = ~np.isfinite(a)
        if bad.any(): a[bad] = np.nanmedian(a)
    t = 1.0 - t; o = 1.0 - o
    t -= np.nanmedian(t); o -= np.nanmedian(o)
    w = tukey(t.size, apodize); t *= w; o *= w
    max_lag = min(int(np.ceil(velocity_max / dv)), int(np.floor(t.size * 0.5)))
    lags, ccf = _normalized_lag_ccf(t, o, max_lag)
    velocities = lags * dv
    k = int(np.nanargmax(ccf))
    return _gauss_peak_fit(velocities, ccf, fallback=velocities[k])


def gauss_core_center(wl, flux, lam0, search_A=1.5, fit_window_A=0.40):
    """Fit a Gaussian to the deepest pixel within ±search_A of lam0, then
    refine in a tight ±fit_window_A around the local minimum."""
    m_s = (wl >= lam0 - search_A) & (wl <= lam0 + search_A)
    if m_s.sum() < 9: return np.nan
    x_s, y_s = wl[m_s], flux[m_s]
    lam_min = x_s[int(np.argmin(y_s))]
    m = (wl >= lam_min - fit_window_A) & (wl <= lam_min + fit_window_A)
    if m.sum() < 7: return np.nan
    x, y = wl[m], flux[m]
    core = float(np.min(y))
    if 1.0 - core < 0.05: return np.nan
    def mg(x, a, c, s, b): return b - a * np.exp(-0.5 * ((x - c) / s) ** 2)
    try:
        popt, _ = curve_fit(mg, x, y, p0=[1.0 - core, lam_min, 0.15, 1.0],
                            bounds=([0., x[0], 0.01, 0.5],
                                    [2., x[-1], 1.0, 1.5]), maxfev=5000)
        return float(popt[1])
    except Exception:
        return np.nan


def line_bisector(wl, flux, lam0, search_A=1.5, fit_half_A=0.6,
                  depths=(0.2, 0.4, 0.6)):
    """Mean bisector wavelength at three absorption depths (20/40/60 %)."""
    m_s = (wl >= lam0 - search_A) & (wl <= lam0 + search_A)
    if m_s.sum() < 9: return np.nan
    x_s, y_s = wl[m_s], flux[m_s]
    lam_min = x_s[int(np.argmin(y_s))]
    m = (wl >= lam_min - fit_half_A) & (wl <= lam_min + fit_half_A)
    if m.sum() < 9: return np.nan
    x, y = wl[m], flux[m]
    k = int(np.argmin(y))
    if k == 0 or k == len(y) - 1: return np.nan
    cont = 1.0; ld = cont - float(y[k])
    if ld < 0.05: return np.nan
    bs = []
    for d in depths:
        target = cont - d * ld
        left, xl = y[:k+1], x[:k+1]
        idx = np.where(left > target)[0]
        if len(idx) == 0 or idx[-1] >= len(left) - 1: continue
        il = idx[-1]
        if left[il] == left[il+1]: continue
        lam_b = xl[il] + (target - left[il]) / (left[il+1] - left[il]) * (xl[il+1] - xl[il])
        right, xr = y[k:], x[k:]
        idx = np.where(right > target)[0]
        if len(idx) == 0 or idx[0] == 0: continue
        ir = idx[0]
        if right[ir-1] == right[ir]: continue
        lam_r = xr[ir-1] + (target - right[ir-1]) / (right[ir] - right[ir-1]) * (xr[ir] - xr[ir-1])
        bs.append(0.5 * (lam_b + lam_r))
    return float(np.mean(bs)) if bs else np.nan


def broaden_template_gauss(wl, template, V_puls_phase, oversample=8):
    """Symmetric Gaussian broadening with σ = |V_puls|/√18 -- the std of the
    radial-pulsation wedge velocity distribution."""
    if abs(V_puls_phase) < 0.5: return template.copy()
    sigma_kms = abs(V_puls_phase) / np.sqrt(18.0)
    lnwl = np.log(np.asarray(wl, dtype=float))
    n = int(oversample * len(wl)); n += n % 2
    ln = np.linspace(lnwl[0], lnwl[-1], n); dv = C_KMS * (ln[1] - ln[0])
    t = interp1d(lnwl, template, kind='linear', bounds_error=False, fill_value=np.nan)(ln)
    bad = ~np.isfinite(t)
    if bad.any(): t[bad] = np.nanmedian(t)
    kn = int(8 * sigma_kms / dv) + 1
    if kn % 2 == 0: kn += 1
    kv = (np.arange(kn) - kn // 2) * dv
    kernel = np.exp(-0.5 * (kv / sigma_kms) ** 2); kernel /= kernel.sum()
    t_broad = np.convolve(t, kernel, mode='same')
    return interp1d(ln, t_broad, kind='linear', bounds_error=False, fill_value=1.0)(lnwl)
