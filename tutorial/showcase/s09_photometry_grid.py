"""Showcase 9 (compute) - spectra for the interactive "photometry from a spectrum" page.

Disc-integrated absolute (10 pc) spectra of three dwarfs over the emulator's whole
HARPS range (3780-6910 A) plus SPICE's own AB magnitudes in the Stromgren v, b, y
passbands (all fully inside the range -- the band-coverage rule), so the page's
JavaScript integral can be checked against SPICE (s09_build_html.py embeds both and
shows the difference).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

STARS = [("K dwarf", 4500.0, 4.6, 0.75), ("Sun", 5777.0, 4.44, 1.0), ("A/F star", 7500.0, 4.2, 1.7)]
WL = (3790.0, 6900.0)


def compute(args, out):
    C.setup_jax()
    from spice.spectrum import AB_passband_luminosity
    from spice.spectrum.filter import Stromgrenv, Stromgrenb, Stromgreny

    emu = C.load_emulator()
    wl, lw = C.wl_grid(*WL, 600 if args.quick else 15000)
    filters = {"Strömgren v": Stromgrenv(), "Strömgren b": Stromgrenb(), "Strömgren y": Stromgreny()}
    flux, cont, mags, curves = [], [], [], {}
    for name, T, g, R in STARS:
        p = C.stellar_params(emu, T, g, 0.0, 1.2)
        m = C.make_star(p, emu.stellar_parameter_names, n_vertices=500 if args.quick else 2000, radius=R,
                        inclination=90.0, vrot=2.0)
        f, c = C.synth(emu, m, lw)
        flux.append(f); cont.append(c)
        mags.append([float(AB_passband_luminosity(fl, wl, f)) for fl in filters.values()])
        print(name, dict(zip(filters, np.round(mags[-1], 4))), flush=True)
    for k, fl in filters.items():
        curves[k] = np.asarray(fl.filter_responses_for_wavelengths(wl))
        assert not fl.non_photonic, "page replicates the photonic AB integral only"
    zps = [float(fl.AB_zeropoint) for fl in filters.values()]
    np.savez_compressed(out / "data.npz", wl=wl, flux=np.array(flux), cont=np.array(cont), mags=np.array(mags),
                        star_names=np.array([s[0] for s in STARS]), star_params=np.array([s[1:] for s in STARS]),
                        filter_names=np.array(list(filters)), filter_curves=np.array(list(curves.values())), filter_zps=np.array(zps))


if __name__ == "__main__":
    a = C.parser(__doc__).parse_args()
    compute(a, C.outdir(a))
