"""Explain a residual feature in the TZ For joint fit: whose spectrum is it, and what drives it?

Given an observed wavelength (or a rest wavelength in one star's frame), this walks the four
questions worth asking about an outlying line, using only the fitted model dump plus, optionally,
the emulator itself:

1. **Whose line is it?** The residuals of all 21 epochs are stacked in three frames -- the giant's,
   the secondary's, and the observatory's. The component RVs span 80 km s^-1, so a feature belonging
   to one star adds up coherently in that star's frame and washes out in the others. A telluric or
   detector artefact would instead be sharp in the observatory frame.
2. **Is it real or an artefact of one exposure?** The per-epoch amplitudes are listed.
3. **Is the model line spurious or merely too strong?** The predicted and observed line depths are
   measured against the local continuum.
4. **What controls it?** With ``--emulator``, the line's depth is re-evaluated across temperature and
   against each of the emulator's abundance inputs, which separates an iron-peak line from a
   molecular one and shows whether it is saturated (a strong microturbulence response).

Example: ``python tzfor_diagnose_line.py --rest 5034.46 --emulator``
"""
import argparse, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import tzfor_constants as K

C_KMS = 299792.458
DEFAULT_DUMP = HERE / "tzfor_aemu_out" / "fr_model_maskall_hp.npz"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default=str(DEFAULT_DUMP))
    ap.add_argument("--rest", type=float, help="rest wavelength in the primary's frame [A]")
    ap.add_argument("--observed", type=float, help="observed wavelength [A]; --epoch says in which exposure")
    ap.add_argument("--epoch", type=int, default=0, help="epoch that --observed refers to")
    ap.add_argument("--half-width", type=float, default=1.0, help="search half-width [A]")
    ap.add_argument("--emulator", action="store_true", help="also ask the emulator what drives the line (CPU, ~1 min)")
    args = ap.parse_args()

    d = np.load(args.dump, allow_pickle=True)
    wl = 10.0 ** d["logwl"]; obs, mod, prim = d["obs"], d["model"], d["primary"]
    rv = d["rv"]; n_ep = obs.shape[0]
    masked = d["masked"] if "masked" in d.files else ~d["good"] & np.isfinite(obs)
    res = np.where(np.isfinite(obs), obs - mod, np.nan)
    th, pn = d["theta"], list(d["pnames"])
    T1 = float(th[pn.index("Teff1")]); feh = float(th[pn.index("feh")])
    logg1 = 2.915

    lam0 = args.rest if args.rest else args.observed / (1 + rv[0, args.epoch] / C_KMS)
    print(f"feature at {lam0:.3f} A in the primary's rest frame\n")

    # ---- 1. whose frame? ----
    win = (wl > lam0 - args.half_width) & (wl < lam0 + args.half_width); grid = wl[win]
    frames = [("primary (K giant)", rv[0]), ("secondary", rv[1]), ("observatory (telluric/detector)", np.zeros(n_ep))]
    best = None
    print("1. stacking the 21 epochs in each rest frame (component RVs span 80 km/s):")
    for nm, frv in frames:
        S = np.full((n_ep, grid.size), np.nan)
        for e in range(n_ep):
            S[e] = np.interp(grid * (1 + frv[e] / C_KMS), wl, res[e], left=np.nan, right=np.nan)
        med = np.nanmedian(S, 0); sd = np.nanstd(S, 0); k = int(np.nanargmax(np.abs(med)))
        signif = abs(med[k]) / max(sd[k], 1e-6)
        print(f"   {nm:32s} peak {med[k]:+.3f} at {grid[k]:.3f} A, epoch scatter {sd[k]:.3f}  -> coherence {signif:5.1f}")
        if best is None or signif > best[0]: best = (signif, nm, grid[k], med[k], S)
    signif, nm, lam_pk, amp, S = best
    print(f"   => the feature belongs to the {nm.split(' (')[0]}, rest wavelength {lam_pk:.3f} A\n")

    # ---- 2. every epoch, or one bad exposure? ----
    kk = int(np.argmin(np.abs(grid - lam_pk))); v = S[:, kk]; ok = np.isfinite(v)
    print(f"2. present in {int(np.sum(np.abs(v[ok]) > 0.05))} of {int(ok.sum())} epochs above 5%; "
          f"amplitudes {np.round(v[ok], 2)}")
    e_ref = int(np.nanargmax(np.abs(v)))
    lam_obs = lam_pk * (1 + rv[0, e_ref] / C_KMS)
    i = int(np.argmin(np.abs(wl - lam_obs)))
    print(f"   masked out of the fit: {bool(masked[e_ref, i])}\n")

    # ---- 3. spurious, or just too strong? ----
    loc = slice(max(0, i - 30), i + 31)
    dm = float(np.nanmax(mod[e_ref][loc]) - mod[e_ref][i])
    do = float(np.nanmax(obs[e_ref][loc]) - obs[e_ref][i])
    dp = float(np.nanmax(prim[e_ref][loc]) - prim[e_ref][i])
    print(f"3. depth against the local continuum, epoch {e_ref} (observed {lam_obs:.3f} A):")
    print(f"   model blend {dm:.3f}   observed {max(do, 0.0):.3f}   the giant's own contribution {dp:.3f}")
    print(f"   => the model line is {'entirely spurious' if do < 0.2 * dm else f'about {dm / max(do, 1e-3):.1f}x too strong'}\n")

    # ---- 4. what drives it? ----
    if args.emulator:
        import jax.numpy as jnp
        from spice.spectrum.aemu_spectrum_emulator import IntensityPretrainedAemuSpectrumEmulator
        emu = IntensityPretrainedAemuSpectrumEmulator("RozanskiT/TPayne-spice-harps")
        lw = jnp.linspace(np.log10(lam_pk - 0.5), np.log10(lam_pk + 0.5), 300); w = 10.0 ** np.asarray(lw)
        names = ["marcs_teff", "marcs_logg", "feh", "vmicro", "a", "c", "n", "o", "r", "s"]
        base = [T1, logg1, feh, 1.5, 0., 0., 0., 0., 0., 0.]

        def depth(row):
            o = np.asarray(emu.intensity(lw, 0.7, jnp.array(row))); n_ = o[:, 0] / o[:, 1]
            k_ = int(np.argmin(np.abs(w - lam_pk))); sl = slice(max(0, k_ - 30), k_ + 31)
            return float(np.max(n_[sl]) - np.min(n_[sl]))

        print("4. what the emulator says drives this line (normalised depth at mu = 0.7):")
        print("   temperature:", "  ".join(f"{t:.0f}K {depth([t] + base[1:]):.3f}" for t in (4300, 4600, T1, 5200, 5800, 6400)))
        d0 = depth(base)
        for j, nm_ in enumerate(names):
            if nm_ in ("marcs_teff", "marcs_logg"): continue
            step = 0.5 if nm_ == "vmicro" else 0.3
            hi = list(base); hi[j] += step; lo = list(base); lo[j] -= step
            ch = depth(hi) - depth(lo)
            flag = "  <-- controls the line" if abs(ch) > 0.05 else ""
            print(f"   {nm_:8s} ±{step}: {depth(lo):.3f} -> {d0:.3f} -> {depth(hi):.3f}   change {ch:+.3f}{flag}")


if __name__ == "__main__":
    main()
