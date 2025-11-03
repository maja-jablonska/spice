import jax
jax.config.update('jax_enable_x64', True)
# Disable preallocation to reduce memory footprint
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

import multiprocessing as mp
# Only set spawn if not already set to avoid conflicts
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method("spawn")

import argparse
import phoebe
import numpy as np
import matplotlib.pyplot as plt
from phoebe.parameters.dataset import _mesh_columns
import jax
import jax.numpy as jnp
from tqdm import tqdm
from transformer_payne import Blackbody

from spice.models.mesh_model import IcosphereModel
from spice.models.binary import Binary, add_orbit, evaluate_orbit_at_times
from spice.models.mesh_view import get_mesh_view
from spice.spectrum.filter import Bolometric, JohnsonCousinsI, JohnsonCousinsU, JohnsonCousinsV, Stromgrenv, GaiaG
from spice.spectrum.spectrum import AB_passband_luminosity, simulate_observed_flux
from astropy import units as u

# Memory monitoring imports
import psutil
import os
import time
import gc

phoebe.multiprocessing_set_nprocs(1)

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # Convert to MB

def log_memory_usage(stage_name, start_time=None):
    """Log memory usage at a specific stage."""
    current_time = time.time()
    memory_mb = get_memory_usage()
    
    if start_time is not None:
        elapsed = current_time - start_time
        print(f"[MEMORY] {stage_name}: {memory_mb:.1f} MB (elapsed: {elapsed:.1f}s)")
        return current_time
    else:
        print(f"[MEMORY] {memory_mb:.1f} MB")
        return current_time

def force_garbage_collection():
    """Force garbage collection and log memory change."""
    before = get_memory_usage()
    gc.collect()
    # Clear JAX compilation cache to free GPU/CPU memory
    jax.clear_caches()
    # Additional JAX memory clearing
    gc.collect()  # Run again after JAX cleanup
    after = get_memory_usage()
    freed = before - after
    if freed > 1.0:  # Only log if significant memory was freed
        print(f"[MEMORY] Garbage collection freed {freed:.1f} MB")

def main(inclination, period, ntriangles, q, ecc, n_times, primary_mass, output_path):
    """
    Compares PHOEBE and SPICE velocities and positions for a binary system.

    Args:
        inclination (float): Inclination of the binary system in degrees.
        period (float): Period of the binary system in days.
    """
    filename = os.path.join(output_path, f'eclipses_incl_{inclination}_period_{period}_q_{q}_ecc_{ecc}_primary_mass_{primary_mass}.pkl')
    if os.path.isfile(filename):
        print(f"File {filename} already exists. Exiting.")
        return
    
    start_time = time.time()
    print(f"Running comparison for inclination={inclination} deg, period={period} days")
    log_memory_usage("Script start")

    # Create PHOEBE bundle
    b = phoebe.default_binary()
    bb = Blackbody()
    log_memory_usage("After PHOEBE bundle creation", start_time)

    COLUMNS = _mesh_columns

    # Set parameters from command line arguments
    b.set_value('period@binary@component', period)
    b.set_value('q@binary@component', q)
    b.set_value('ecc@binary@component', ecc)
    b.flip_constraint('mass@primary', solve_for='sma')
    b.set_value('mass@primary@component', primary_mass)
    b.set_value_all('incl@binary', inclination)
    print(f"PHOEBE parameters set: inclination={b['incl@binary']}, period={b['period@binary']}, q={b['q@binary']}, ecc={b['ecc@binary']}")
    log_memory_usage("After parameter setting")
    
    from spice.models.orbit_utils import eclipse_timestamps_kepler

    _, t1_p, _, _, t4_p, _, t1_s, _, _, t4_s = \
        eclipse_timestamps_kepler(b.get_parameter('mass@primary@component').value,
                                  b.get_parameter('mass@secondary@component').value,
                                  b.get_parameter('period@binary@component').value*0.0027378507871321013,
                                  b.get_parameter('ecc@binary@component').value,
                                  b.get_parameter('t0_perpass@binary@component').value*0.0027378507871321013,
                                  jnp.deg2rad(b.get_parameter('incl@binary@component').value),
                                  b.get_parameter('per0@binary@component').value*0.017453292519943295,
                                  b.get_parameter('long_an@binary@component').value*0.0027378507871321013,
                                  b.get_parameter('requiv@primary@component').value,
                                  b.get_parameter('requiv@secondary@component').value,
                                  pad=1.1,
                                  los_vector=jnp.array([0., 0., -1.]))
        
    print(f"PHOEBE eclipse timestamps: {t1_p*365.25}, {t4_p*365.25}, {t1_s*365.25}, {t4_s*365.25}")
    log_memory_usage("After eclipse timestamp calculation")
    
    times = np.concatenate([np.linspace(t1_p/0.0027378507871321013, t4_p/0.0027378507871321013, int(n_times/2)),
                            np.linspace(t1_s/0.0027378507871321013, t4_s/0.0027378507871321013, int(n_times/2))])
    log_memory_usage("After time array creation")

    def default_icosphere(mass=1., radius=1.):
        # Reduced mesh resolution to 1000 points to reduce memory usage
        return get_mesh_view(IcosphereModel.construct(ntriangles, radius, mass, bb.solar_parameters, bb.parameter_names), jnp.array([0., 0., -1.]))

    times_yr = times * 0.0027378507871321013
    
    b.add_dataset('mesh', times=times, columns=COLUMNS, dataset='mesh01')
    b.add_dataset('orb', compute_times=times, dataset='orb01')
    b.add_dataset('lc', compute_times=times, passband='Bolometric:900-40000', dataset='lc_bolometric')
    b.add_dataset('lc', compute_times=times, passband='Johnson:I', dataset='lc_johnson_i')
    b.add_dataset('lc', compute_times=times, passband='Johnson:U', dataset='lc_johnson_u')
    b.add_dataset('lc', compute_times=times, passband='Johnson:V', dataset='lc_johnson_v')
    b.add_dataset('lc', compute_times=times, passband='Stromgren:v', dataset='lc_stromgren_v')
    b.add_dataset('lc', compute_times=times, passband='Gaia:G', dataset='lc_gaia_g')
    b.set_value_all('pblum_mode', dataset='lc_bolometric', value='absolute')
    b.set_value_all('pblum_mode', dataset='lc_johnson_i', value='absolute')
    b.set_value_all('pblum_mode', dataset='lc_johnson_u', value='absolute')
    b.set_value_all('pblum_mode', dataset='lc_johnson_v', value='absolute')
    b.set_value_all('pblum_mode', dataset='lc_stromgren_v', value='absolute')
    b.set_value_all('pblum_mode', dataset='lc_gaia_g', value='absolute')
    b.set_value_all('gravb_bol', 0.0)
    b.set_value('distance', 10*u.pc)     # and set any l3 / l3_frac if relevant
    b.set_value_all('ld_mode', 'manual')
    b.set_value_all('ld_func', 'linear')
    b.set_value_all('ld_coeffs', [0.])
    b.set_value_all('ld_mode_bol', 'manual')
    b.set_value_all('ld_func_bol', 'linear')
    b.set_value_all('ld_coeffs_bol', [0.])
    b.set_value_all('atm', 'blackbody')
    b.set_value_all('irrad_method', 'none')
    b.set_value_all('distortion_method', 'sphere')
    b.set_value_all('ntriangles', ntriangles)
    log_memory_usage("Before PHOEBE compute")
    b.compute_pblums(pbflux=True, set_value=True)  # updates pblums (and optionally pbflux)
    b.compute_ld_coeffs(ld_mode='manual', ld_func='linear', ld_coeffs=[0.])
    b.run_compute(irrad_method='none', coordinates='uvw', ltte=False)
    log_memory_usage("After PHOEBE compute")
    force_garbage_collection()

    body1 = default_icosphere(b.get_parameter('mass@primary@component').value, b.get_parameter('requiv@primary@component').value)
    log_memory_usage("After creating body1")
    force_garbage_collection()
    
    body2 = default_icosphere(b.get_parameter('mass@secondary@component').value, b.get_parameter('requiv@secondary@component').value)
    log_memory_usage("After creating body2")
    force_garbage_collection()
    
    binary = Binary.from_bodies(body1, body2)
    log_memory_usage("After creating binary")
    force_garbage_collection()

    binary = add_orbit(binary, P = b.get_parameter('period@binary@component').value*0.0027378507871321013,
                       ecc = b.get_parameter('ecc@binary@component').value,
                       T = b.get_parameter('t0_perpass@binary@component').value*0.0027378507871321013,
                        i = jnp.deg2rad(b.get_parameter('incl@binary@component').value),
                        omega = b.get_parameter('per0@binary@component').value*0.017453292519943295,
                        Omega = b.get_parameter('long_an@binary@component').value*0.0027378507871321013,
                        vgamma = b.get_parameter('vgamma').value,
                        reference_time = b.get_parameter('t0_ref@binary@component').value*0.0027378507871321013,
                        mean_anomaly = b.get_parameter('mean_anom@binary@component').value*0.017453292519943295,
                        orbit_resolution_points = len(times_yr))
    log_memory_usage("After adding orbit")

    print(f"Evaluating orbit at times: {times_yr}")
    pb1, pb2 = evaluate_orbit_at_times(binary, times_yr)
    log_memory_usage("After orbit evaluation")
    force_garbage_collection()
    
    fluxes_phoebe = b.get_parameter(f'fluxes@lc_bolometric@model').value
    fluxes_phoebe = -2.5 * np.log10(fluxes_phoebe)
    fluxes_phoebe = fluxes_phoebe - fluxes_phoebe[0]
    
    fluxes_phoebe_i = b.get_parameter(f'fluxes@lc_johnson_i@model').value
    fluxes_phoebe_i = -2.5 * np.log10(fluxes_phoebe_i)
    fluxes_phoebe_i = fluxes_phoebe_i - fluxes_phoebe_i[0]
    fluxes_phoebe_u = b.get_parameter(f'fluxes@lc_johnson_u@model').value
    fluxes_phoebe_u = -2.5 * np.log10(fluxes_phoebe_u)
    fluxes_phoebe_u = fluxes_phoebe_u - fluxes_phoebe_u[0]
    fluxes_phoebe_v = b.get_parameter(f'fluxes@lc_johnson_v@model').value
    fluxes_phoebe_v = -2.5 * np.log10(fluxes_phoebe_v)
    fluxes_phoebe_v = fluxes_phoebe_v - fluxes_phoebe_v[0]
    fluxes_phoebe_stromgren_v = b.get_parameter(f'fluxes@lc_stromgren_v@model').value   
    fluxes_phoebe_stromgren_v = -2.5 * np.log10(fluxes_phoebe_stromgren_v)
    fluxes_phoebe_stromgren_v = fluxes_phoebe_stromgren_v - fluxes_phoebe_stromgren_v[0]
    fluxes_phoebe_gaia_g = b.get_parameter(f'fluxes@lc_gaia_g@model').value
    fluxes_phoebe_gaia_g = -2.5 * np.log10(fluxes_phoebe_gaia_g)
    fluxes_phoebe_gaia_g = fluxes_phoebe_gaia_g - fluxes_phoebe_gaia_g[0]
    
    # Reduced wavelength sampling for bolometric flux (1000 points is sufficient)
    vws = jnp.linspace(900, 40000, 2000)
    log_memory_usage("Before spectrum simulation")
    
    bol = Bolometric()
    johnson_i = JohnsonCousinsI()
    johnson_u = JohnsonCousinsU()
    johnson_v = JohnsonCousinsV()
    stromgren_v = Stromgrenv()
    gaia_g = GaiaG()
    
    print("[MEMORY] Computing bolometric luminosities incrementally...")
    # Process one time step at a time to avoid storing all spectra in memory
    bol_lum = []
    bol_lum_i = []
    bol_lum_u = []
    bol_lum_v = []
    bol_lum_stromgren_v = []
    bol_lum_gaia_g = []
    for i, (_pb1, _pb2) in enumerate(tqdm(zip(pb1, pb2), total=len(pb1))):
        spec1 = simulate_observed_flux(bb.intensity, _pb1, jnp.log10(vws), disable_doppler_shift=True)
        spec2 = simulate_observed_flux(bb.intensity, _pb2, jnp.log10(vws), disable_doppler_shift=True)
        lum = AB_passband_luminosity(bol, vws, spec1[:, 0] + spec2[:, 0])
        lum_i = AB_passband_luminosity(johnson_i, vws, spec1[:, 0] + spec2[:, 0])
        lum_u = AB_passband_luminosity(johnson_u, vws, spec1[:, 0] + spec2[:, 0])
        lum_v = AB_passband_luminosity(johnson_v, vws, spec1[:, 0] + spec2[:, 0])
        lum_stromgren_v = AB_passband_luminosity(stromgren_v, vws, spec1[:, 0] + spec2[:, 0])
        lum_gaia_g = AB_passband_luminosity(gaia_g, vws, spec1[:, 0] + spec2[:, 0])
        bol_lum.append(lum)
        bol_lum_i.append(lum_i)
        bol_lum_u.append(lum_u)
        bol_lum_v.append(lum_v)
        bol_lum_stromgren_v.append(lum_stromgren_v)
        bol_lum_gaia_g.append(lum_gaia_g)
        
        # Force garbage collection every 5 iterations
        if (i + 1) % 5 == 0:
            force_garbage_collection()
    
    bol_lum = np.array(bol_lum)
    log_memory_usage("After bolometric luminosity calculation")
    force_garbage_collection()

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump({
            'phoebe_binary': b,
            'spice_binary': binary,
            'fluxes_phoebe': fluxes_phoebe,
            'fluxes_phoebe_i': fluxes_phoebe_i,
            'fluxes_phoebe_u': fluxes_phoebe_u,
            'fluxes_phoebe_v': fluxes_phoebe_v,
            'fluxes_phoebe_stromgren_v': fluxes_phoebe_stromgren_v,
            'fluxes_phoebe_gaia_g': fluxes_phoebe_gaia_g,
            'bol_lum': bol_lum - bol_lum[0],
            'lum_i': bol_lum_i - bol_lum_i[0],
            'lum_u': bol_lum_u - bol_lum_u[0],
            'lum_v': bol_lum_v - bol_lum_v[0],
            'lum_stromgren_v': bol_lum_stromgren_v - bol_lum_stromgren_v[0],
            'lum_gaia_g': bol_lum_gaia_g - bol_lum_gaia_g[0],
            'times': times,
            'n_times': n_times,
            'sma': b.get_parameter('sma@binary@component').value,
            'primary_mass': primary_mass,
            'secondary_mass': b.get_parameter('mass@secondary@component').value,
            'inclination': inclination,
            'period': period,
            'q': q,
            'ecc': ecc
        }, f)
    print(f"Lightcurves saved to {filename}")
    log_memory_usage("Script completion", start_time)
    print(f"[MEMORY] Total execution time: {time.time() - start_time:.1f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare PHOEBE and SPICE binary models.')
    parser.add_argument('inclination', type=float, help='Inclination in degrees')
    parser.add_argument('period', type=float, help='Period in days')
    parser.add_argument('ntriangles', type=int, help='Number of triangles')
    parser.add_argument('--n_times', type=int, default=10, help='Number of time points to evaluate')
    parser.add_argument('--q', type=float, default=1.0, help='Mass ratio')
    parser.add_argument('--ecc', type=float, default=0.0, help='Eccentricity')
    parser.add_argument('--primary_mass', type=float, default=1.0, help='Primary mass')
    parser.add_argument('--output_path', type=str, default='lc_eclipse', help='Output directory for results')
    args = parser.parse_args()

    main(args.inclination, args.period, args.ntriangles, args.q, args.ecc, args.n_times, args.primary_mass, args.output_path)
