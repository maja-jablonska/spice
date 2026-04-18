import phoebe
import numpy as np
from spice.spectrum import simulate_observed_flux
from transformer_payne import Blackbody
from phoebe.parameters.dataset import _mesh_columns
from spice.models.binary import PhoebeBinary
from spice.models.phoebe_utils import PhoebeConfig
from spice.spectrum import AB_passband_luminosity
from spice.spectrum.filter import *
import argparse
import pickle
import logging

from jax import config
config.update('jax_enable_x64', True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_phoebe_passband(fname: str):
    logger.info(f"Loading PHOEBE passband from {fname}")
    with open(fname, 'rb') as f:
        struct = pickle.load(f)
        logger.info(f"Successfully loaded passband: {struct['pbset']}:{struct['pbname']}")
        return struct
    
def get_transmission_curve(phoebe_passband):
    logger.info("Extracting transmission curve from PHOEBE passband")
    return phoebe_passband['ptf_func'][0]*1e11, phoebe_passband['ptf_func'][1]


def create_phoebe_filter(phoebe_passband):
    logger.info("Creating PHOEBE filter")
    # Extract wavelengths and transmission values from PHOEBE passband
    wavelengths, transmission = get_transmission_curve(phoebe_passband)
    wavelengths = wavelengths * 1e-1
    
    # Create transmission curve as a 2D array
    transmission_curve = np.array([wavelengths, transmission])
    
    # Create a Filter object
    filter_name = f"{phoebe_passband['pbset']}:{phoebe_passband['pbname']}"
    logger.info(f"Creating filter: {filter_name}")
    phoebe_filter = Filter(transmission_curve, name=filter_name)
    
    return phoebe_filter


def offsets(incl: float, period: float, q: float, ecc: float, primary_mass: float):
    logger.info(f"Starting offsets calculation with incl={incl}°, period={period} days")
    
    logger.info("Creating default binary system")
    b = phoebe.default_binary()
    times = np.linspace(0, period, 100)
    COLUMNS = _mesh_columns
    
    logger.info("Adding datasets to binary system")
    b.add_dataset('mesh', times=times, columns=COLUMNS, dataset='mesh01')
    b.add_dataset('orb', compute_times=times, dataset='orb01')
    b.add_dataset('lc', compute_times=times, passband='Johnson:U', dataset='lc_johnson_u')
    b.add_dataset('lc', compute_times=times, passband='Johnson:V', dataset='lc_johnson_v')
    b.add_dataset('lc', compute_times=times, passband='Stromgren:v', dataset='lc_stromgren_v')
    b.add_dataset('lc', compute_times=times, passband='Gaia:G', dataset='lc_gaia_g')
    b.add_dataset('lc', compute_times=times, passband='Bolometric:900-40000', dataset='lc_bolometric')
    
    logger.info("Setting binary system parameters")
    b.set_value_all('incl@binary', incl)
    b.set_value('period@binary', period)
    b.set_value('q@binary@component', q)
    b.set_value('ecc@binary@component', ecc)
    b.flip_constraint('mass@primary', solve_for='sma')
    b.set_value('mass@primary@component', primary_mass)
    b.set_value_all('ld_mode', 'manual')
    b.set_value_all('ld_func', 'linear')
    b.set_value_all('ld_coeffs', [0.])
    b.set_value_all('ld_mode_bol', 'manual')
    b.set_value_all('ld_func_bol', 'linear')
    b.set_value_all('ld_coeffs_bol', [0.])
    b.set_value_all('atm', 'blackbody')
    
    logger.info("Running PHOEBE compute")
    b.run_compute(irrad_method='none', coordinates='uvw', ltte=False)
    
    logger.info("Creating Blackbody model")
    bb = Blackbody()
    p1 = PhoebeConfig(b, 'mesh01', 'orb01')
    pb = PhoebeBinary.construct(p1, bb.parameter_names, {pn: sp for pn, sp in zip(bb.parameter_names, bb.solar_parameters)})
    from spice.models.binary import evaluate_orbit 
    
    logger.info("Evaluating orbits")
    pbs = [evaluate_orbit(pb, t) for t in times]
    
    logger.info("Simulating observed fluxes")
    ws = np.linspace(900, 40000, 1000)
    s1s_no_d = [simulate_observed_flux(bb.intensity, pb1, np.log10(ws), disable_doppler_shift=True) for pb1, _ in pbs]
    s2s_no_d = [simulate_observed_flux(bb.intensity, pb2, np.log10(ws), disable_doppler_shift=True) for _, pb2 in pbs]
    
    logger.info("Calculating PHOEBE differences")
    diffs_phoebe = []
    for dataset in ['lc_johnson_u', 'lc_johnson_v', 'lc_stromgren_v', 'lc_gaia_g', 'lc_bolometric']:
        fluxes = b.get_parameter(f'fluxes@{dataset}@model').value
        diffs_phoebe.append(-2.5 * np.log10(fluxes))
    
    logger.info("Creating filters")
    filters = [
        create_phoebe_filter(load_phoebe_passband('phoebe_passbands/johnson_u.pb3')),
        create_phoebe_filter(load_phoebe_passband('phoebe_passbands/johnson_v.pb3')),
        create_phoebe_filter(load_phoebe_passband('phoebe_passbands/stromgren_v.pb3')), 
        create_phoebe_filter(load_phoebe_passband('phoebe_passbands/gaia_g.pb3')),
        create_phoebe_filter(load_phoebe_passband('phoebe_passbands/bolometric.pb3'))
    ]
    
    logger.info("Calculating SPICE luminosities")
    spice_luminosities = [[AB_passband_luminosity(f, ws, s1[:, 0]+s2[:, 0]) for s1, s2 in zip(s1s_no_d, s2s_no_d)] for f in filters]
    diffs_spice= [np.array(sl)-sl[0] for sl in spice_luminosities]
    diffs_phoebe= [np.array(sl)-sl[0] for sl in diffs_phoebe]
    
    import os
    import csv

    # Create directory for this inclination and period combination
    directory = f'results/incl_{incl:.2f}_period_{period:.2f}_q_{q:.2f}_ecc_{ecc:.2f}_primary_mass_{primary_mass:.2f}'
    logger.info(f"Creating output directory: {directory}")
    os.makedirs(directory, exist_ok=True)

    max_residuals = []
    logger.info("Calculating and saving residuals")
    for i, f in enumerate(filters):
        residuals = np.abs(diffs_phoebe[i]-diffs_spice[i])
        output_file = os.path.join(directory, f'{f.name}.txt')
        logger.info(f"Saving residuals to {output_file}")
        np.savetxt(output_file, residuals)
        
        max_residual = np.max(residuals)
        max_residuals.append((f.name, max_residual))
        logger.info(f"Max residual for {f.name}: {max_residual}")

    # Save biggest residuals for each filter in one csv file
    csv_file = os.path.join(directory, 'max_residuals.csv')
    logger.info(f"Saving max residuals to {csv_file}")
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Filter', 'Max Residual'])
        writer.writerows(max_residuals)
    
    logger.info("Offset calculation completed successfully")

if __name__ == "__main__":
    logger.info("Starting binary offsets calculation")
    parser = argparse.ArgumentParser(description='Calculate binary offsets.')
    parser.add_argument('--incl', type=float, required=True, help='Inclination in degrees')
    parser.add_argument('--period', type=float, required=True, help='Period in days')
    parser.add_argument('--q', type=float, required=True, help='Mass ratio')
    parser.add_argument('--ecc', type=float, required=True, help='Eccentricity')
    parser.add_argument('--primary_mass', type=float, required=True, help='Primary mass')
    args = parser.parse_args()

    logger.info(f"Received arguments: inclination={args.incl}°, period={args.period} days")
    offsets(args.incl, args.period, args.q, args.ecc, args.primary_mass)
