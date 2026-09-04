"""Compare SPICE+aemu synthetic spectra against the SuppNet-normalised HARPS data.

Model side: ``tzfor_phoebe_aemu_photometry.py --mode synth-spectra`` (PHOEBE
meshes -> aemu intensity emulator, Doppler on, components stored separately).
Observed side: ``build_harps_zarr.py`` then ``harps_normalize_npz.py``.

The one subtlety worth stating: a blended binary's normalised spectrum is

    (F1 + F2) / (C1 + C2)

**not** the mean of the two normalised spectra. Each star's lines are diluted by
the other's continuum, and by a different amount at each wavelength -- which is
precisely the spectroscopic light ratio. Storing components separately (with
their own continua) is what makes this recoverable.

Both sides are on the same footing: HARPS wavelengths are barycentric and in
air, and the model carries vgamma = +18.17 km/s, so no extra shift is applied.
That assumption is *tested* here (the reported cross-correlation offset should
be consistent with zero) rather than assumed.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

C_KMS = 299792.458
R_HARPS = 115000.0          # HARPS resolving power

HERE = Path(__file__).resolve().parent


def _broaden(y, lam0, dlam, vmacro_kms, resolving_power=R_HARPS):
    """Instrumental profile + macroturbulence, as a Gaussian in velocity.

    Applied post-emulator, which is where macroturbulence belongs here (the
    bundle emulates the intrinsic photospheric spectrum; the emulator's mu
    dependence already supplies limb darkening, and rotation comes from the
    PHOEBE mesh velocities). Both terms conserve equivalent width, so they can
    only redistribute a line, never change its strength.
    """
    fwhm_kms = float(np.hypot(C_KMS / resolving_power, vmacro_kms))
    sigma_pix = lam0 * fwhm_kms / C_KMS / 2.3548 / dlam
    return gaussian_filter1d(y, sigma_pix, mode="nearest")


def composite_normalized(spectra, i, wl=None, sel=None,
                         vmacro=(5.0, 6.0), broaden=True):
    """(F1+F2)/(C1+C2) for epoch i -- the blended, diluted model spectrum.

    Each component is broadened with its *own* macroturbulence before blending,
    and flux and continuum are broadened separately so the ratio stays a proper
    normalised spectrum.
    """
    if sel is None:
        sel = slice(None)
    f1, c1 = spectra["primary"][i][sel], spectra["primary_continuum"][i][sel]
    f2, c2 = spectra["secondary"][i][sel], spectra["secondary_continuum"][i][sel]
    if broaden and wl is not None:
        w = wl[sel]
        lam0, dlam = 0.5 * (w[0] + w[-1]), float(w[1] - w[0])
        f1, c1 = (_broaden(f1, lam0, dlam, vmacro[0]),
                  _broaden(c1, lam0, dlam, vmacro[0]))
        f2, c2 = (_broaden(f2, lam0, dlam, vmacro[1]),
                  _broaden(c2, lam0, dlam, vmacro[1]))
    return (f1 + f2) / (c1 + c2)


def light_ratio(spectra, i):
    """Continuum light ratio L2/L1 at this epoch (wavelength-averaged)."""
    c1 = np.mean(spectra["primary_continuum"][i])
    c2 = np.mean(spectra["secondary_continuum"][i])
    return c2 / c1


def velocity_offset(wl, model, obs, max_shift_kms=80.0, n=641):
    """Cross-correlate model against observed within ONE contiguous window.

    Must be called per window: the synthesis grid is two disjoint bands
    (5160-5200 and 6540-6580 A), and interpolating a shift across that 1340 A
    gap mixes them, which makes the correlation slide monotonically to the
    search bound instead of peaking.

    Returns (best_shift_kms, peak_correlation, hit_rail).
    """
    C = 299792.458
    m = np.isfinite(model) & np.isfinite(obs)
    if m.sum() < 50:
        return np.nan, np.nan, False
    wlm, om = wl[m], obs[m] - np.mean(obs[m])
    shifts = np.linspace(-max_shift_kms, max_shift_kms, n)
    ccs = np.full(n, -np.inf)
    for k, s in enumerate(shifts):
        shifted = np.interp(wlm, wlm * (1 + s / C), model[m])
        sm = shifted - np.mean(shifted)
        denom = np.sqrt(np.sum(sm**2) * np.sum(om**2))
        if denom > 0:
            ccs[k] = np.sum(sm * om) / denom
    k = int(np.argmax(ccs))
    return float(shifts[k]), float(ccs[k]), k in (0, n - 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="…_spectra_n<N>.pkl from Gadi")
    ap.add_argument("--observed", default=str(HERE / "harps_normalized.npz"))
    ap.add_argument("--no-broaden", action="store_true",
                    help="skip the instrumental+macroturbulent convolution "
                         "(measured to change core residuals by <10%, and EW "
                         "not at all, but it is the physically correct model)")
    args = ap.parse_args()

    with open(args.model, "rb") as f:
        M = pickle.load(f)
    O = np.load(args.observed, allow_pickle=True)

    wl = np.asarray(M["wavelengths"])
    spectra = M["spectra"]
    model_files = list(M.get("harps_files") or [])
    obs_files = [str(x) for x in O["files"]]

    print(f"model epochs: {len(M['times'])}   observed: {len(obs_files)}")
    if model_files and model_files != obs_files:
        print("  [warn] file lists differ; joining on filename")

    windows = M.get("windows") or [(5160., 5200., "win1"), (6540., 6580., "win2")]
    print(f"\nper window ({len(windows)} bands), joined on filename\n")
    summary = {}
    for lo, hi, wname in windows:
        sel = (wl >= lo) & (wl <= hi)
        print(f"--- {lo:.0f}-{hi:.0f} A  ({wname}) ---")
        print(f"  {'file':<36}{'phase':>7}{'dv[km/s]':>10}{'ccmax':>8}"
              f"{'rms':>9}{'obs d':>8}{'mod d':>8}")
        rows = []
        for i, fname in enumerate(model_files or obs_files):
            if fname not in obs_files:
                continue
            j = obs_files.index(fname)
            ow, onf = np.asarray(O["wavelengths"][j]), np.asarray(O["normed_flux"][j])
            ok = np.isfinite(ow) & np.isfinite(onf)
            obs_i = np.interp(wl[sel], ow[ok], onf[ok], left=np.nan, right=np.nan)
            mod_i = composite_normalized(spectra, i, wl, sel,
                                         broaden=not args.no_broaden)
            dv, cc, rail = velocity_offset(wl[sel], mod_i, obs_i)
            good = np.isfinite(mod_i) & np.isfinite(obs_i)
            rms = float(np.sqrt(np.mean((mod_i[good] - obs_i[good])**2)))
            phase = ((M["times"][i] - 2452599.29040) % 75.66647) / 75.66647
            rows.append((dv, cc, rms, rail))
            flag = " RAIL" if rail else ""
            print(f"  {fname:<36}{phase:>7.3f}{dv:>10.2f}{cc:>8.3f}{rms:>9.4f}"
                  f"{1-np.nanmin(obs_i):>8.3f}{1-np.nanmin(mod_i):>8.3f}{flag}")
        dv = np.array([r[0] for r in rows]); cc = np.array([r[1] for r in rows])
        rms = np.array([r[2] for r in rows]); rails = sum(r[3] for r in rows)
        summary[wname] = (np.nanmean(dv), np.nanstd(dv), np.nanmean(cc),
                          rms.mean(), rails, len(rows))
        print(f"  => dv {np.nanmean(dv):+7.2f} +- {np.nanstd(dv):5.2f} km/s   "
              f"ccmax {np.nanmean(cc):.3f}   rms {rms.mean():.4f}   "
              f"rail-hits {rails}/{len(rows)}\n")

    print("light ratio L2/L1 (continuum): "
          f"{np.mean([light_ratio(spectra, i) for i in range(len(M['times']))]):.4f}")
    print("\nA nonzero, phase-independent dv would mean a frame/convention offset;")
    print("rail-hits mean the correlation found no peak and the number is meaningless.")


if __name__ == "__main__":
    main()
