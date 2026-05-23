"""Blackbody passband luminosities / magnitudes over a range of mesh resolutions.

Merges the former ``blackbody_luminosity.py`` and ``blackbody_luminosity_offsets.py``
scripts (their ``run_blackbody`` was identical apart from the Doppler-shift and
precision settings) behind a small CLI:

    # was blackbody_luminosity.py  (Doppler shift on, single precision)
    python blackbody_luminosity.py --output blackbody_luminosity_single_precision.csv

    # was blackbody_luminosity_offsets.py  (Doppler shift disabled)
    python blackbody_luminosity.py --disable-doppler-shift
"""
import argparse

from spice.models import IcosphereModel
from spice.spectrum import simulate_observed_flux
from spice.spectrum.spectrum import simulate_monochromatic_luminosity, ST_passband_luminosity
import astropy.units as u
from transformer_payne import Blackbody
from synphot.models import Empirical1D
from synphot import SourceSpectrum, units, SpectralElement, Observation
from spice.spectrum.filter import *
from spice.spectrum.spectrum import AB_passband_luminosity
from spice.spectrum import luminosity, absolute_bol_luminosity
import pandas as pd
from tqdm import tqdm

from jax import config

config.update('jax_platform_name', 'cpu')


def run_blackbody(n_vertices: int, disable_doppler_shift: bool):
    bb = Blackbody()
    model = IcosphereModel.construct(n_vertices, 1., 1., bb.solar_parameters, bb.parameter_names)

    vws = jnp.linspace(1., 100000., 100000)
    i_bb = simulate_observed_flux(bb.intensity, model, jnp.log10(vws), 10., chunk_size=1000,
                                  disable_doppler_shift=disable_doppler_shift)

    bb_synphot = SourceSpectrum(Empirical1D, points=vws, lookup_table=i_bb[:, 0]/1e8*units.FLAM)
    i = SpectralElement.from_filter('johnson_v')
    obs = Observation(bb_synphot, i, binset=vws)

    bessel_b = JohnsonCousinsB()
    bessel_i = JohnsonCousinsI()
    gaia_g = GaiaG()
    johnson_v = JohnsonCousinsV()

    solar_luminosity = luminosity(bb.flux, model, vws)

    return {
        'n_vertices': len(model.d_vertices),
        'areas_to_solar': np.sum(model.areas) / u.solRad.to(u.cm) ** 2 * 4 * np.pi,
        'cast_areas_to_solar': np.sum(np.where(model.mus > 0., model.cast_areas, 0.)) / (np.pi * u.solRad.to(u.cm) ** 2),
        'synphot_johnson_v_ab': obs.effstim(u.ABmag, wavelengths=vws),
        'AB_solar_apparent_mag_B': AB_passband_luminosity(bessel_b, vws, i_bb[:, 0]),
        'AB_solar_apparent_mag_I': AB_passband_luminosity(bessel_i, vws, i_bb[:, 0]),
        'AB_solar_apparent_mag_G': AB_passband_luminosity(gaia_g, vws, i_bb[:, 0]),
        'AB_solar_johnson_v': AB_passband_luminosity(johnson_v, vws, i_bb[:, 0]),
        'ST_solar_apparent_mag_B': ST_passband_luminosity(bessel_b, vws, i_bb[:, 0]),
        'ST_solar_apparent_mag_I': ST_passband_luminosity(bessel_i, vws, i_bb[:, 0]),
        'ST_solar_apparent_mag_G': ST_passband_luminosity(gaia_g, vws, i_bb[:, 0]),
        'ST_solar_johnson_v': ST_passband_luminosity(johnson_v, vws, i_bb[:, 0]),
        'solar_luminosity': solar_luminosity,
        'solar_absolute_bol_luminosity': absolute_bol_luminosity(solar_luminosity)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--disable-doppler-shift', action='store_true',
                        help='Disable the Doppler shift in simulate_observed_flux '
                             '(former blackbody_luminosity_offsets.py behaviour).')
    parser.add_argument('--double-precision', action='store_true',
                        help='Enable jax_enable_x64 (64-bit) before computing.')
    parser.add_argument('--output', default='blackbody_luminosity.csv',
                        help='Output CSV path.')
    parser.add_argument('--n-vertices', type=int, nargs='+', default=[100, 1000, 5000, 10000],
                        help='Mesh vertex counts to evaluate.')
    args = parser.parse_args()

    config.update('jax_enable_x64', args.double_precision)

    vertex_data = [run_blackbody(n_vertices, args.disable_doppler_shift)
                   for n_vertices in tqdm(args.n_vertices)]
    pd.DataFrame.from_records(vertex_data).to_csv(args.output, index=False)
