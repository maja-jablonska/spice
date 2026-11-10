"""Coadd the downloaded HARPS spectra of each benchmark star into one spectrum (data/gbs_harps.npz).

The ESO Phase 3 s1d products are already merged and rebinned to 0.01 A in the barycentric frame
(SPECSYS = BARYCENT, checked here), so exposures of one star are combined directly: each is scaled to
its median flux over 5000-6000 A, interpolated onto the wavelength grid of the first file and averaged
with inverse-variance weights. The star's own radial velocity is left in; gbs_fit.py measures and
removes it.

Usage: python gbs_prepare.py   (astro env; reads data/targets.csv and data/harps/<star>/*.fits)
"""
import csv, glob, json, os, re

import numpy as np
from astropy.io import fits

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")


def read_adp(path):
    with fits.open(path) as h:
        hdr0, hdr1, tab = h[0].header, h[1].header, h[1].data
        wl = np.asarray(tab["WAVE"][0], float); fl = np.asarray(tab["FLUX"][0], float); er = np.asarray(tab["ERR"][0], float)
        meta = dict(file=os.path.basename(path), object=hdr0.get("OBJECT"), snr=hdr0.get("SNR"), specsys=hdr0.get("SPECSYS", hdr1.get("SPECSYS")),
                    berv=hdr0.get("HIERARCH ESO DRS BERV"), mjd=hdr0.get("MJD-OBS"), origfile=hdr0.get("ORIGFILE"))
    if not np.isfinite(er).any():
        # DRS 3.5/3.8 s1d products leave ERR empty: photon noise sigma ~ sqrt(F), scaled so the 5500-5600 A S/N equals the header SNR
        sq = np.sqrt(np.clip(fl, 1e-10 * np.nanmax(fl), None)); band = (wl > 5500) & (wl < 5600)
        er = sq * np.nanmedian(sq[band]) / float(meta["snr"]); meta["err_from"] = "sqrt(F), header SNR"
    return wl, fl, er, meta


def catalogue_ids(name):
    """'HD025457' / 'HD 25457' -> 'HD25457'; returns the HD/HIP id of an OBJECT string, or None for other names."""
    m = re.fullmatch(r"(HD|HIP)0*(\d+)[A-Z]?", re.sub(r"[^A-Z0-9]", "", str(name).upper()))
    return f"{m.group(1)}{m.group(2)}" if m else None


def main():
    targets = list(csv.DictReader(open(os.path.join(DATA, os.environ.get("GBS_TARGETS", "targets.csv")))))
    out, summary = {}, {}
    for t in targets:
        files = sorted(glob.glob(os.path.join(DATA, "harps", t["star"], "ADP*.fits")))
        if not files:
            print(f"{t['star']}: no spectra, skipped", flush=True); continue
        spec = [read_adp(f) for f in files]
        # saturated / corrupt exposures: header S/N at the ~600 cap of the 2014 reprocessing, or negative flux in the fitted range
        broken = [m["file"] for wl, fl, _, m in spec if float(m["snr"] or 0) >= 580 or np.nanmin(fl[(wl > 5100) & (wl < 6500)]) < 0]
        if broken:
            print(f"{t['star']}: dropped {len(broken)} saturated/corrupt exposure(s) {broken}", flush=True)
            spec = [x for x in spec if x[3]["file"] not in broken]
        # the archive cone (72 arcsec) can catch a companion (alpha Cen A query -> HD128621 = alpha Cen B): drop exposures whose
        # OBJECT is a different HD/HIP number than the star's own
        own = {catalogue_ids(t["star"]), catalogue_ids("HIP" + str(t.get("hip", "")).replace("HIP", ""))}
        comp = lambda o: re.search(r"[0-9]B$", re.sub(r"[^A-Z0-9]", "", str(o).upper())) is not None and not t["star"].upper().endswith("B")
        other = [m["file"] for *_, m in spec if catalogue_ids(m["object"]) not in (None, *own) or comp(m["object"])]   # also 'GJ872B' etc.
        if other:
            print(f"{t['star']}: dropped {len(other)} exposure(s) of another object {sorted({str(m['object']) for *_, m in spec if m['file'] in other})}", flush=True)
            spec = [x for x in spec if x[3]["file"] not in other]
        if not spec:
            print(f"{t['star']}: no spectra of the star itself, skipped", flush=True); continue
        bad = [m["file"] for _, _, _, m in spec if m["specsys"] not in (None, "BARYCENT")]
        assert not bad, f"{t['star']}: not barycentric: {bad}"
        wl0 = spec[0][0]; num = np.zeros_like(wl0); den = np.zeros_like(wl0)
        for wl, fl, er, m in spec:
            s = np.nanmedian(fl[(wl > 5000) & (wl < 6000)]); f = np.interp(wl0, wl, fl / s, left=np.nan, right=np.nan); e = np.interp(wl0, wl, er / s, left=np.nan, right=np.nan)
            w = np.where(np.isfinite(f) & np.isfinite(e) & (e > 0), 1.0 / np.where(e > 0, e, 1.0) ** 2, 0.0)
            num += w * np.nan_to_num(f); den += w
        flux = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan); err = np.where(den > 0, 1.0 / np.sqrt(np.where(den > 0, den, 1.0)), np.nan)
        k = t["star"]; out[f"{k}/wl"] = wl0; out[f"{k}/flux"] = flux; out[f"{k}/err"] = err
        snr = float(np.nanmedian((flux / err)[(wl0 > 5500) & (wl0 < 5600)]))
        summary[k] = dict(n_files=len(files), snr_5550=round(snr, 1), files=[m["file"] for *_, m in spec], snr_files=[m["snr"] for *_, m in spec],
                          objects=sorted({str(m["object"]) for *_, m in spec}))
        print(f"{k}: {len(files)} spectra {summary[k]['objects']}, coadded S/N {snr:.0f} at 5550 A", flush=True)
    out["summary_json"] = json.dumps(summary)
    path = os.path.join(DATA, os.environ.get("GBS_NPZ", "gbs_harps.npz")); np.savez_compressed(path, **out)
    print(f"wrote {path} ({len(summary)} stars)", flush=True)


if __name__ == "__main__":
    main()
