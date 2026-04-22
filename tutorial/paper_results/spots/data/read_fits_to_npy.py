#!/usr/bin/env python3
"""Read all FITS files and save HJD, WAVE, FLUX to a single NPY file."""

import glob
import os
import warnings
import numpy as np
from astropy.io import fits

warnings.filterwarnings("ignore", module="astropy.io.fits")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def get_hjd(header):
    """Get HJD from header. Use BJD if available, else JD from MJD-OBS."""
    if "ESO DRS BJD" in header:
        # BJD is very close to HJD; use when available (HARPS)
        return float(header["ESO DRS BJD"])
    # JD = MJD + 2400000.5 (for files without BJD, e.g. UVES/FEROS)
    mjd = float(header["MJD-OBS"])
    return mjd + 2400000.5


def read_fits(path):
    """Read HJD, WAVE, FLUX from a single FITS file."""
    with fits.open(path) as hdul:
        header = hdul[0].header
        data = hdul[1].data

        hjd = get_hjd(header)

        # Handle different column names (HARPS: FLUX, UVES: FLUX or FLUX_REDUCED)
        if "FLUX" in data.columns.names:
            flux = np.asarray(data["FLUX"][0])
        elif "FLUX_REDUCED" in data.columns.names:
            flux = np.asarray(data["FLUX_REDUCED"][0])
        else:
            raise ValueError(f"No FLUX column in {path}")

        wave = np.asarray(data["WAVE"][0])

    return hjd, wave, flux


def main():
    fits_files = sorted(
        f
        for f in glob.glob(os.path.join(DATA_DIR, "ADP*.fits"))
        if not f.endswith(".fits.1")
        and os.path.getsize(f) > 500_000  # skip small/corrupt (e.g. 114688)
    )

    hjds = []
    waves = []
    fluxes = []

    for i, path in enumerate(fits_files):
        try:
            hjd, wave, flux = read_fits(path)
            hjds.append(hjd)
            waves.append(wave)
            fluxes.append(flux)
            if (i + 1) % 5 == 0:
                print(f"  Read {i + 1}/{len(fits_files)} files...")
        except Exception as e:
            print(f"Warning: skipping {os.path.basename(path)}: {e}")

    # Stack into arrays - WAVE/FLUX may have different lengths per spectrum
    hjds = np.array(hjds)
    waves_arr = np.array(waves, dtype=object)
    fluxes_arr = np.array(fluxes, dtype=object)

    out_path = os.path.join(DATA_DIR, "spectra_hjd_wave_flux.npy")
    np.save(out_path, {"HJD": hjds, "WAVE": waves_arr, "FLUX": fluxes_arr}, allow_pickle=True)
    print(f"Saved {len(hjds)} spectra to {out_path}")
    print(f"  HJD: shape {hjds.shape}")
    print(f"  WAVE: shape {waves_arr.shape}")
    print(f"  FLUX: shape {fluxes_arr.shape}")


if __name__ == "__main__":
    main()
