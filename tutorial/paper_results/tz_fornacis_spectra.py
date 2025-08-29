from spice.models.binary import Binary, add_orbit, evaluate_orbit_at_times
from spice.models.mesh_model import IcosphereModel
from spice.models.mesh_view import get_mesh_view
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from spice.spectrum import simulate_observed_flux
from transformer_payne import TransformerPayne, METALS
import jax.numpy as jnp
import pickle
import os
from tqdm import tqdm


import click

@click.command()
@click.option('--num-times', default=150, help='Number of time points to sample')
@click.option('--num-wavelengths', default=40000, help='Number of wavelength points')
@click.option('--output-dir', default='data', help='Directory to save output files')
def generate_tz_fornacis_spectra(num_times, num_wavelengths, output_dir):
    """Generate synthetic spectra for TZ Fornacis binary system."""
    click.echo("Loading Transformer Payne model...")
    tp = TransformerPayne.download()
    
    click.echo("Creating primary star model (G-type giant)...")
    body1 = IcosphereModel.construct(500, 8.28, 2.057, tp.to_parameters(dict(logteff=jnp.log10(4930), logg=2.91)), tp.stellar_parameter_names)
    
    click.echo("Creating secondary star model (F-type subgiant)...")
    body2 = IcosphereModel.construct(500, 3.94, 1.958, tp.to_parameters(dict(logteff=jnp.log10(6650), logg=3.35)), tp.stellar_parameter_names)
    
    click.echo("Setting up binary system with orbital parameters...")
    binary = Binary.from_bodies(body1, body2)
    binary = add_orbit(binary, (75.6*u.d).to(u.year).value, 0., 0., jnp.deg2rad(85.68), jnp.deg2rad(65.99), jnp.deg2rad(269.), 0., 0., 0., 15)
    
    click.echo(f"Generating {num_times} time points across orbital period...")
    times = jnp.linspace(0., (75.6*u.d).to(u.year).value, num_times).reshape((10, num_times//10))
    
    click.echo("Evaluating orbital positions at each time point...")
    result = [evaluate_orbit_at_times(binary, t) for t in tqdm(times)]
    times = times.flatten()

    pb1, pb2 = [], []
    for r in result:
        pb1.extend(r[0])
        pb2.extend(r[1])
    
    click.echo(f"Setting up wavelength grid with {num_wavelengths} points...")
    vws = jnp.linspace(3000, 10000, num_wavelengths)
    
    click.echo("Simulating spectra for primary star...")
    spectra_body1 = [simulate_observed_flux(tp.intensity, _pb, jnp.log10(vws)) for _pb in tqdm(pb1)]
    
    click.echo("Simulating spectra for secondary star...")
    spectra_body2 = [simulate_observed_flux(tp.intensity, _pb, jnp.log10(vws)) for _pb in tqdm(pb2)]

    # Create a directory for saving data if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    click.echo(f"Saving spectra data to {output_dir} directory...")
    # Save the spectra data and meshes to pickle files
    with open(f'{output_dir}/tz_fornacis_data_{num_wavelengths}.pkl', 'wb') as f:
        pickle.dump({
            'spectra_body1': spectra_body1,
            'spectra_body2': spectra_body2,
            'mesh_body1': pb1,
            'mesh_body2': pb2,
            'wavelengths': vws,
            'times': times
        }, f)

    click.echo(f"✓ Spectra data successfully saved to pickle files in the '{output_dir}' directory")

if __name__ == '__main__':
    generate_tz_fornacis_spectra()