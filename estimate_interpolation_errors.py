#!/usr/bin/env python

import h5py
import numpy as np
from tqdm import tqdm
import itertools
from jax.scipy.interpolate import RegularGridInterpolator
import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)

def read_hdf5_file_to_dict(filename):
    """
    Reads an HDF5 file and returns its contents as NumPy arrays in a dictionary.

    Parameters:
        filename (str): Path to the HDF5 file.

    Returns:
        dict: A dictionary containing the datasets read from the HDF5 file.
    """
    data_dict = {}
    with h5py.File(filename, 'r') as f:
        for key in ['parameters', 'sp_intensity', 'sp_no_lines_intensity']:
            data_dict[key] = np.array(f[key])
        data_dict["log10_sp_wave"] = np.array(f["log10_sp_wave"])
        data_dict["log10_sp_no_lines_wave"] = np.array(f["log10_sp_no_lines_wave"])
    return data_dict

def main():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # Read the HDF5 file
    filename = '/scratch/y89/tr0953/korg_grid_calculation/ready_grids/regular_grid.h5'
    data = read_hdf5_file_to_dict(filename)

    # Extract wavelengths and intensities
    wavelengths = data['log10_sp_wave']
    intensities = data['sp_intensity']

    # Get unique parameter values
    teff_vals = np.unique(data['parameters'][:, 0])
    logg_vals = np.unique(data['parameters'][:, 1])
    mh_vals = np.unique(data['parameters'][:, 2])
    alpha_vals = np.unique(data['parameters'][:, 3])

    # Create interpolator
    interp = RegularGridInterpolator(
        (teff_vals, logg_vals, mh_vals, alpha_vals, wavelengths),
        intensities.reshape(len(teff_vals), len(logg_vals), len(mh_vals), len(alpha_vals), len(wavelengths))
    )
    interpolate_spectrum = jax.vmap(lambda p, w: interp(jnp.concatenate([p, jnp.array([w])])), in_axes=(None, 0))

    wavelengths = data['log10_sp_wave']

    # Extract unique parameter values for each dimension
    teff_vals = np.unique(data['parameters'][:, 0])
    logg_vals = np.unique(data['parameters'][:, 1]) 
    mh_vals = np.unique(data['parameters'][:, 2])
    alpha_vals = np.unique(data['parameters'][:, 3])

    # Calculate midpoints between grid values
    teff_mid = np.unique((teff_vals[:-1] + teff_vals[1:]) / 2)
    logg_mid = np.unique((logg_vals[:-1] + logg_vals[1:]) / 2)
    mh_mid = np.unique((mh_vals[:-1] + mh_vals[1:]) / 2)
    alpha_mid = np.unique((alpha_vals[:-1] + alpha_vals[1:]) / 2)

    # Create all combinations of midpoints
    mid_points = np.array(list(itertools.product(teff_mid, logg_mid, mh_mid, alpha_mid)))

    # Interpolate spectra at midpoints using previously defined interpolator
    mid_spectra = jnp.array([interpolate_spectrum(mid_points[i], wavelengths) for i in tqdm(range(len(mid_points)))])

    mid_interp = RegularGridInterpolator((teff_mid, logg_mid, mh_mid, alpha_mid, wavelengths), 
                                    mid_spectra.reshape(len(teff_mid), len(logg_mid), len(mh_mid), len(alpha_mid), len(wavelengths)))
    mid_interp_v = jax.vmap(lambda p, w: mid_interp(jnp.concatenate([p, jnp.array([w])])), in_axes=(None, 0))

    # Get original points and interpolate
    original_points = data['parameters']
# Interpolate original spectra at same wavelength points for comparison
    original_spectra = [mid_interp_v(data['parameters'][i], wavelengths) for i in range(len(data['parameters']))]
    print("Interpolating original spectra...")
    original_spectra = jnp.array([mid_interp(jnp.concatenate([original_points[i], [w]]))
                                for i in tqdm(range(len(original_points)))
                                for w in wavelengths]).reshape(len(original_points), len(wavelengths), 1)

    # Calculate RMS error
    rms_error = jnp.sqrt(jnp.mean((original_spectra - mid_spectra[:len(original_points)].reshape(-1, len(wavelengths), 1))**2))
    print(f"RMS Error: {rms_error:.6f}")

    # Save results
    save_data = {
        'parameters': original_points,
        'original_spectra': original_spectra,
        'reconstructed_spectra': mid_spectra[:len(original_points)].reshape(-1, len(wavelengths), 1),
        'wavelengths': wavelengths,
        'rms_error': rms_error
    }

    print("Saving results to spectra_comparison.h5...")
    with h5py.File('spectra_comparison.h5', 'w') as f:
        for key, value in save_data.items():
            f.create_dataset(key, data=value)

    print("Done!")

if __name__ == "__main__":
    main()
