import csv
import phoebe
import numpy as np
from tqdm import tqdm
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
print("Imports completed successfully")

def load_phoebe_passband(fname: str):
    print(f"Loading PHOEBE passband from {fname}")
    with open(fname, 'rb') as f:
        struct = pickle.load(f)
        print(f"Successfully loaded passband: {struct['pbset']}:{struct['pbname']}")
        return struct
    
def get_transmission_curve(phoebe_passband):
    print("Extracting transmission curve from PHOEBE passband")
    return phoebe_passband['ptf_func'][0]*1e11, phoebe_passband['ptf_func'][1]


def create_phoebe_filter(phoebe_passband):
    print("Creating PHOEBE filter")
    # Extract wavelengths and transmission values from PHOEBE passband
    wavelengths, transmission = get_transmission_curve(phoebe_passband)
    wavelengths = wavelengths * 1e-1
    
    # Create transmission curve as a 2D array
    transmission_curve = np.array([wavelengths, transmission])
    
    # Create a Filter object
    filter_name = f"{phoebe_passband['pbset']}:{phoebe_passband['pbname']}"
    print(f"Creating filter: {filter_name}")
    phoebe_filter = Filter(transmission_curve, name=filter_name)
    
    return phoebe_filter


print("Loading and creating filters...")
filters = [
    create_phoebe_filter(load_phoebe_passband('phoebe_passbands/johnson_u.pb3')),
    create_phoebe_filter(load_phoebe_passband('phoebe_passbands/johnson_v.pb3')),
    create_phoebe_filter(load_phoebe_passband('phoebe_passbands/stromgren_v.pb3')), 
    create_phoebe_filter(load_phoebe_passband('phoebe_passbands/gaia_g.pb3')),
    create_phoebe_filter(load_phoebe_passband('phoebe_passbands/bolometric.pb3'))
]
print(f"Created {len(filters)} filters")


def offsets(incl: float, period: float):
    print(f"\nStarting offset calculation for inclination={incl}, period={period}")
    b = phoebe.default_binary()
    print("Created default PHOEBE binary system")
    
    times = np.linspace(0, period, 100)
    print(f"Generated times array with {len(times)} points")
    
    COLUMNS = _mesh_columns
    b.add_dataset('mesh', times=times, columns=COLUMNS, dataset='mesh01')
    b.add_dataset('orb', compute_times=times, dataset='orb01')
    b.add_dataset('lc', compute_times=times, passband='Johnson:U', dataset='lc_johnson_u')
    b.add_dataset('lc', compute_times=times, passband='Johnson:V', dataset='lc_johnson_v')
    b.add_dataset('lc', compute_times=times, passband='Johnson:I', dataset='lc_johnson_i')
    b.add_dataset('lc', compute_times=times, passband='Stromgren:v', dataset='lc_stromgren_v')
    b.add_dataset('lc', compute_times=times, passband='Gaia:G', dataset='lc_gaia_g')
    b.add_dataset('lc', compute_times=times, passband='Bolometric:900-40000', dataset='lc_bolometric')
    print("Added all datasets to PHOEBE system")
    
    b.set_value_all('incl@binary', incl)
    b.set_value('period@binary', period)
    b.set_value_all('ld_mode', 'manual')
    b.set_value_all('ld_func', 'linear')
    b.set_value_all('ld_coeffs', [0.])
    b.set_value_all('ld_mode_bol', 'manual')
    b.set_value_all('ld_func_bol', 'linear')
    b.set_value_all('ld_coeffs_bol', [0.])
    b.set_value_all('atm', 'blackbody')
    print("Set all PHOEBE parameters")
    
    print("Running PHOEBE compute...")
    b.run_compute(irrad_method='none', coordinates='uvw', ltte=False)
    print("PHOEBE compute completed")
    
    bb = Blackbody()
    print("Created Blackbody instance")
    
    p1 = PhoebeConfig(b, 'mesh01', 'orb01')
    pb = PhoebeBinary.construct(p1, bb.parameter_names, {pn: sp for pn, sp in zip(bb.parameter_names, bb.solar_parameters)})
    print("Constructed PhoebeBinary instance")
    
    from spice.models.binary import evaluate_orbit 
    print("Evaluating orbits...")
    pbs = [evaluate_orbit(pb, t) for t in times]
    print(f"Evaluated {len(pbs)} orbits")
    
    ws = np.linspace(900, 40000, 1000)
    print(f"Generated wavelength array with {len(ws)} points")
    
    print("Simulating observed fluxes without Doppler shift...")
    s1s_no_d = [simulate_observed_flux(bb.intensity, pb1, np.log10(ws), disable_doppler_shift=True) for pb1, _ in pbs]
    s2s_no_d = [simulate_observed_flux(bb.intensity, pb2, np.log10(ws), disable_doppler_shift=True) for _, pb2 in pbs]
    print("Flux simulation completed")
    
    diffs_phoebe = []
    print("Calculating PHOEBE differences...")
    for dataset in ['lc_johnson_u', 'lc_johnson_v', 'lc_stromgren_v', 'lc_gaia_g', 'lc_bolometric']:
        fluxes = b.get_parameter(f'fluxes@{dataset}@model').value
        diffs_phoebe.append(-2.5 * np.log10(fluxes))
    print("PHOEBE differences calculated")

    
    print("Calculating SPICE luminosities...")
    spice_luminosities = [[AB_passband_luminosity(f, ws, s1[:, 0]+s2[:, 0]) for s1, s2 in zip(s1s_no_d, s2s_no_d)] for f in filters]
    diffs_spice= [np.array(sl)-sl[0] for sl in spice_luminosities]
    diffs_phoebe= [np.array(sl)-sl[0] for sl in diffs_phoebe]
    print("Luminosity calculations completed")

    max_residuals = []
    print("Calculating maximum residuals...")
    for i, f in enumerate(filters):
        residuals = np.abs(diffs_phoebe[i]-diffs_spice[i])
        max_residual = np.max(residuals)
        max_residuals.append((f.__class__.__name__, max_residual))
        print(f"Max residual for {f.__class__.__name__}: {max_residual}")
        
    print("Cleaning up PHOEBE system...")
    b.cleanup()
    del b
    print("Cleanup completed")
        
    return max_residuals

if __name__ == "__main__":
    print("Starting main execution")
    inclinations = np.linspace(0., 90, 10)
    periods = np.linspace(1., 100, 10)
    print(f"Generated {len(inclinations)} inclinations and {len(periods)} periods")
    
    residuals = []
    
    print("Starting main calculation loop...")
    for incl in tqdm(inclinations):
        for period in periods:
            print(f"\nProcessing inclination={incl}, period={period}")
            residuals.append(offsets(incl, period))
    
    print("Writing results to CSV...")
    with open('binary_offsets.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Inclination', 'Period', 'Filter', 'Max Residual'])
        for i, incl in enumerate(inclinations):
            for j, period in enumerate(periods):
                for filter_name, max_residual in residuals[i * len(periods) + j]:
                    writer.writerow([incl, period, filter_name, max_residual])
    print("CSV file written successfully")
    print("Program completed")
