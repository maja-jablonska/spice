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

DOUBLE_PRECISION = False

config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", DOUBLE_PRECISION)


def run_blackbody(n_vertices: int):
    bb = Blackbody()
    model = IcosphereModel.construct(n_vertices, 1., 1., bb.solar_parameters, bb.parameter_names)

    vws = jnp.linspace(1., 100000., 100000)
    i_bb = simulate_observed_flux(bb.intensity, model, jnp.log10(vws), 10., chunk_size=1000)

    bb_synphot = SourceSpectrum(Empirical1D, points=vws, lookup_table=i_bb[:, 0]/1e8*units.FLAM)
    i = SpectralElement.from_filter('johnson_v')
    obs = Observation(bb_synphot, i, binset=vws)

    bessel_b = BesselB()
    bessel_i = BesselI()
    gaia_g = GaiaG()
    johnson_v = JohnsonV()

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
    vertex_data = [run_blackbody(n_vertices) for n_vertices in tqdm([100, 1000, 5000, 10000])]
    pd.DataFrame.from_records(vertex_data).to_csv('blackbody_luminosity_single_precision.csv', index=False)
