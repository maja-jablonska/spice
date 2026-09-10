"""Rebuild ``harps_data.zarr`` from the ADP FITS archive.

Script form of the store-building cells in ``harps_read.ipynb``, so the
normalisation pipeline can be re-run without opening the notebook. The schema
is kept byte-for-byte compatible with what ``harps_suppnet_normalize.py``
expects:

* rows ordered by ``sorted(glob(...))`` -- ``raw_source_files`` is the canonical
  row -> FITS mapping and downstream code joins on row index, so the order must
  be filename-sorted and not directory order;
* ragged spectra padded to a common length with NaN;
* provenance in attrs: HARPS delivers barycentric, **air** wavelengths.

One file in the archive is 0 bytes; it is skipped with a logged reason rather
than aborting, and it simply does not get a row.
"""
import glob
from pathlib import Path

import numpy as np
import zarr
from astropy.io import fits

OUT = Path(__file__).resolve().parent / "harps_data.zarr"


def main():
    waves, fluxes, errors, snrs, bjds, files, objects = [], [], [], [], [], [], []
    skipped = []
    for path in sorted(glob.glob(str(Path(__file__).resolve().parent / "data" / "*.fits"))):
        name = Path(path).name
        try:
            with fits.open(path) as h:
                hd, d = h[0].header, h[1].data
                w = np.asarray(d["WAVE"][0], dtype=np.float64)
                f = np.asarray(d["FLUX"][0], dtype=np.float64)
                try:
                    e = np.asarray(d["ERR"][0], dtype=np.float64)
                except Exception:
                    e = np.full_like(f, np.nan)
                mjd = hd["MJD-OBS"]
            waves.append(w); fluxes.append(f); errors.append(e)
            snrs.append(float(hd.get("SNR") or np.nan))
            bjds.append(float(mjd) + 2400000.5)
            files.append(name); objects.append(str(hd.get("OBJECT")))
        except Exception as exc:
            skipped.append((name, f"{type(exc).__name__}: {exc}"))

    for name, why in skipped:
        print(f"  [skip] {name}: {why}")
    if not waves:
        raise RuntimeError("no readable FITS in data/")

    maxlen = max(len(w) for w in waves)
    pad = lambda a: np.array([np.pad(x, (0, maxlen - len(x)),
                                     constant_values=np.nan) for x in a])
    root = zarr.open_group(str(OUT), mode="w")
    root.create_array("wavelengths", shape=(len(waves), maxlen), dtype="f8")[:] = pad(waves)
    root.create_array("fluxes", shape=(len(waves), maxlen), dtype="f8")[:] = pad(fluxes)
    root.create_array("errors", shape=(len(waves), maxlen), dtype="f8")[:] = pad(errors)
    root.create_array("snrs", shape=(len(waves),), dtype="f8")[:] = np.array(snrs)
    root.create_array("bjds", shape=(len(waves),), dtype="f8")[:] = np.array(bjds)

    root.attrs["raw_source_files"] = files
    root.attrs["objects"] = objects
    root.attrs["wave_frame"] = "BARYCENT"
    root.attrs["wave_units"] = "Angstrom"
    root.attrs["wave_air_or_vacuum"] = "air"
    root.attrs["bjd_scale"] = "BJD_UTC"
    root.attrs["notes"] = ("Rebuilt by build_harps_zarr.py; rows sorted by "
                           "filename, ragged spectra NaN-padded.")

    assert files == sorted(files), "raw_source_files must be sorted"
    print(f"wrote {OUT}: {len(files)} spectra x {maxlen} points "
          f"({sum(1 for _ in skipped)} skipped)")


if __name__ == "__main__":
    main()
