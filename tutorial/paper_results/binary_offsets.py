import phoebe
import numpy as np
from spice.spectrum import simulate_observed_flux
from transformer_payne import Blackbody
from phoebe.parameters.dataset import _mesh_columns
import astropy.units as u
from spice.models.binary import PhoebeBinary
from spice.models.phoebe_utils import PhoebeConfig
from spice.spectrum import AB_passband_luminosity
from spice.spectrum.filter import *
import argparse
import pickle
import sys

from jax import config
config.update('jax_platform_name', 'cpu')


def load_phoebe_passband(fname: str):
    with open(fname, 'rb') as f:
        struct = pickle.load(f)
        return struct
    
def get_transmission_curve(phoebe_passband):
    return phoebe_passband['ptf_func'][0]*1e11, phoebe_passband['ptf_func'][1]


def create_phoebe_filter(phoebe_passband):
    # Extract wavelengths and transmission values from PHOEBE passband
    wavelengths, transmission = get_transmission_curve(phoebe_passband)
    wavelengths = wavelengths * 1e-1
    
    # Create transmission curve as a 2D array
    transmission_curve = np.array([wavelengths, transmission])
    
    # Create a Filter object
    filter_name = f"{phoebe_passband['pbset']}:{phoebe_passband['pbname']}"
    phoebe_filter = Filter(transmission_curve, name=filter_name)
    
    return phoebe_filter


def offsets(incl: float, period: float):
    b = phoebe.default_binary()
    times = np.linspace(0, period, 100)
    COLUMNS = _mesh_columns
    b.add_dataset('mesh', times=times, columns=COLUMNS, dataset='mesh01')
    b.add_dataset('orb', compute_times=times, dataset='orb01')
    b.add_dataset('lc', compute_times=times, passband='Johnson:U', dataset='lc_johnson_u')
    b.add_dataset('lc', compute_times=times, passband='Johnson:V', dataset='lc_johnson_v')
    b.add_dataset('lc', compute_times=times, passband='Johnson:I', dataset='lc_johnson_i')
    b.add_dataset('lc', compute_times=times, passband='Stromgren:v', dataset='lc_stromgren_v')
    b.add_dataset('lc', compute_times=times, passband='Gaia:G', dataset='lc_gaia_g')
    b.add_dataset('lc', compute_times=times, passband='Bolometric:900-40000', dataset='lc_bolometric')
    
    b.set_value_all('incl@binary', incl)
    b.set_value('period@binary', period)
    b.set_value_all('ld_mode', 'manual')
    b.set_value_all('ld_func', 'linear')
    b.set_value_all('ld_coeffs', [0.])
    b.set_value_all('ld_mode_bol', 'manual')
    b.set_value_all('ld_func_bol', 'linear')
    b.set_value_all('ld_coeffs_bol', [0.])
    b.set_value_all('atm', 'blackbody')
    b.run_compute(irrad_method='none', coordinates='uvw', ltte=False)
    
    bb = Blackbody()
    p1 = PhoebeConfig(b, 'mesh01', 'orb01')
    pb = PhoebeBinary.construct(p1, bb.parameter_names, {pn: sp for pn, sp in zip(bb.parameter_names, bb.solar_parameters)})
    from spice.models.binary import evaluate_orbit 
    
    pbs = [evaluate_orbit(pb, t) for t in times]
    
    ws = np.linspace(900, 40000, 1000)
    s1s_no_d = [simulate_observed_flux(bb.intensity, pb1, np.log10(ws), disable_doppler_shift=True) for pb1, _ in pbs]
    s2s_no_d = [simulate_observed_flux(bb.intensity, pb2, np.log10(ws), disable_doppler_shift=True) for _, pb2 in pbs]
    
    diffs_phoebe = []
    for dataset in ['lc_johnson_u', 'lc_johnson_v', 'lc_stromgren_v', 'lc_gaia_g', 'lc_bolometric']:
        fluxes = b.get_parameter(f'fluxes@{dataset}@model').value
        diffs_phoebe.append(-2.5 * np.log10(fluxes))
    
    filters = [
        create_phoebe_filter(load_phoebe_passband('phoebe_passbands/johnson_u.pb3')),
        create_phoebe_filter(load_phoebe_passband('phoebe_passbands/johnson_v.pb3')),
        create_phoebe_filter(load_phoebe_passband('phoebe_passbands/stromgren_v.pb3')), 
        create_phoebe_filter(load_phoebe_passband('phoebe_passbands/gaia_g.pb3')),
        create_phoebe_filter(load_phoebe_passband('phoebe_passbands/bolometric.pb3'))
    ]
    spice_luminosities = [[AB_passband_luminosity(f, ws, s1[:, 0]+s2[:, 0]) for s1, s2 in zip(s1s_no_d, s2s_no_d)] for f in filters]
    diffs_spice= [np.array(sl)-sl[0] for sl in spice_luminosities]
    diffs_phoebe= [np.array(sl)-sl[0] for sl in diffs_phoebe]
    
    import os
    import csv

    # Create directory for this inclination and period combination
    directory = f'results/incl_{incl:.2f}_period_{period:.2f}'
    os.makedirs(directory, exist_ok=True)

    max_residuals = []
    for i, f in enumerate(filters):
        residuals = np.abs(diffs_phoebe[i]-diffs_spice[i])
        np.savetxt(os.path.join(directory, f'{f.__class__.__name__}.txt'), residuals)
        
        max_residual = np.max(residuals)
        max_residuals.append((f.__class__.__name__, max_residual))

    # Save biggest residuals for each filter in one csv file
    with open(os.path.join(directory, 'max_residuals.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Filter', 'Max Residual'])
        writer.writerows(max_residuals)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate binary offsets.')
    parser.add_argument('--incl', type=float, required=True, help='Inclination in degrees')
    parser.add_argument('--period', type=float, required=True, help='Period in days')
    args = parser.parse_args()

    offsets(args.incl, args.period)
