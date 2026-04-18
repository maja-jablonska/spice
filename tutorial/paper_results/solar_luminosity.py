from spice.models import IcosphereModel
from spice.spectrum import simulate_observed_flux
from spice.spectrum.spectrum import AB_passband_luminosity, ST_passband_luminosity, Vega_passband_luminosity
import jax.numpy as jnp
from transformer_payne import Blackbody
from jax import config
import numpy as np
config.update('jax_platform_name', 'cpu')

if __name__ == "__main__":

    bb = Blackbody()
    model = IcosphereModel.construct(20000, 1., 1., [8000.], bb.parameter_names)
    bb_data = np.genfromtxt('/Users/mjablons/Documents/stellar-mesh-integration/tutorial/data/BB_8000.flx', delimiter=',')
    vws = bb_data[:, 0]  # First column contains wavelengths

    from spice.spectrum.filter import *

    #observed_flux = simulate_observed_flux(bb.intensity, model, jnp.log10(vws))
    observed_flux = bb_data[:, 1]*5.08659e-10

    # Calculate apparent magnitudes for all filters
    filter_results = {}
    for filter_class in FILTERS:
        filter_instance = filter_class()
        filter_name = filter_instance.name.replace('-', '_').replace(' ', '_').lower()
        ab_mag = AB_passband_luminosity(filter_instance, vws, observed_flux)
        try:
            st_mag = ST_passband_luminosity(filter_instance, vws, observed_flux)
        except:
            st_mag = 0.0
        try:
            vega_mag = Vega_passband_luminosity(filter_instance, vws, observed_flux)
        except:
            vega_mag = 0.0
        filter_results[f'{filter_name}_ab'] = ab_mag
        filter_results[f'{filter_name}_st'] = st_mag
        filter_results[f'{filter_name}_vega'] = vega_mag
        import csv

    # Prepare data for CSV
    csv_data = [['Filter', 'AB Magnitude', 'ST Magnitude', 'Vega Magnitude']]
    for filter_name, values in filter_results.items():
        if filter_name.endswith('_ab'):
            base_name = filter_name[:-3]
            ab_mag = values
            st_mag = filter_results[f'{base_name}_st']
            vega_mag = filter_results[f'{base_name}_vega']
            csv_data.append([base_name, ab_mag, st_mag, vega_mag])

    # Write to CSV file
    with open('solar_luminosity_results_exact_8000.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_data)
