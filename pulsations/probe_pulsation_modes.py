#!/usr/bin/env python3
"""
Script to probe different pulsation modes and demonstrate their effect on spectral lines.
"""

import argparse
import pickle
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import config

from spice.models import IcosphereModel
from spice.models.mesh_transform import add_pulsation, evaluate_pulsations
from spice.spectrum import simulate_observed_flux, GaussianLineEmulator

# Enable 64-bit precision
config.update('jax_enable_x64', True)


def create_gaussian_emulator(line_center=5500.0, line_width=0.3, line_depth=0.5):
    """Create a Gaussian line emulator."""
    return GaussianLineEmulator(
        line_centers=[line_center],
        line_widths=[line_width],
        line_depths=[line_depth]
    )


def create_base_model(n_faces=20000, amplitude=1e-3, ge=None):
    """Create the base IcosphereModel without pulsations."""
    if ge is None:
        ge = create_gaussian_emulator()
    
    m = IcosphereModel.construct(
        n_faces, 1., 1.,
        ge.to_parameters(), 
        ge.parameter_names,
        max_pulsation_mode=20
    )
    return m, ge


def add_mode_and_evaluate(base_model, l, m, amplitude, timestamps):
    """Add a pulsation mode and evaluate at given timestamps."""
    # Add the pulsation mode
    model_with_pulsation = add_pulsation(
        base_model, 
        l, m, 
        1., 
        jnp.array([[amplitude, 0.]])
    )
    
    # Evaluate at different timestamps
    models_at_times = [
        evaluate_pulsations(model_with_pulsation, t) 
        for t in timestamps
    ]
    
    return models_at_times


def compute_spectra(models, ge, wavelengths):
    """Compute spectra for all model states."""
    spectra = [
        simulate_observed_flux(ge.intensity, model, wavelengths)
        for model in models
    ]
    return spectra


def save_results(output_path, data):
    """Save results to a pickle file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Probe pulsation modes and their effect on spectral lines'
    )
    parser.add_argument(
        '--modes', 
        type=str, 
        default='2,6;3,4;4,3',
        help='Comma-separated l,m pairs separated by semicolons (e.g., "2,6;3,4;4,3")'
    )
    parser.add_argument(
        '--amplitude',
        type=float,
        default=1e-3,
        help='Pulsation amplitude (default: 1e-3)'
    )
    parser.add_argument(
        '--n-faces',
        type=int,
        default=20000,
        help='Number of faces in the icosphere mesh (default: 20000)'
    )
    parser.add_argument(
        '--n-times',
        type=int,
        default=20,
        help='Number of time steps (default: 20)'
    )
    parser.add_argument(
        '--period',
        type=float,
        default=2.0,
        help='Period to sample over (default: 2.0)'
    )
    parser.add_argument(
        '--wavelength-min',
        type=float,
        default=5498,
        help='Minimum wavelength (default: 5498)'
    )
    parser.add_argument(
        '--wavelength-max',
        type=float,
        default=5502,
        help='Maximum wavelength (default: 5502)'
    )
    parser.add_argument(
        '--n-wavelengths',
        type=int,
        default=2000,
        help='Number of wavelength points (default: 2000)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='pulsations/pulsation_mode_data.pkl',
        help='Output file path (default: pulsations/pulsation_mode_data.pkl)'
    )
    
    args = parser.parse_args()
    
    # Parse modes
    mode_pairs = []
    for pair in args.modes.split(';'):
        l, m = map(int, pair.split(','))
        mode_pairs.append((l, m))
    
    print(f"Probing {len(mode_pairs)} modes: {mode_pairs}")
    print(f"Amplitude: {args.amplitude}")
    print(f"Mesh resolution: {args.n_faces} faces")
    print(f"Time steps: {args.n_times}")
    
    # Create wavelength grid (in log space)
    wvs = jnp.log10(jnp.linspace(
        args.wavelength_min, 
        args.wavelength_max, 
        args.n_wavelengths
    ))
    
    # Create timestamps
    timestamps = jnp.linspace(0., args.period, args.n_times)
    phases = (timestamps - timestamps[0]) / (timestamps[-1] - timestamps[0])
    
    # Create Gaussian emulator
    ge = create_gaussian_emulator()
    
    # Create base model
    print("Creating base model...")
    base_model, _ = create_base_model(args.n_faces, args.amplitude, ge)
    
    # Store results for all modes
    results = {
        'modes': mode_pairs,
        'amplitude': args.amplitude,
        'timestamps': np.array(timestamps),
        'phases': np.array(phases),
        'wavelengths': 10**np.array(wvs),  # Convert back to linear space
        'wavelengths_log': np.array(wvs),
        'spectra_by_mode': {}
    }
    
    # Process each mode
    for l, m in mode_pairs:
        print(f"Processing mode l={l}, m={m}...")
        
        # Add mode and evaluate
        models_at_times = add_mode_and_evaluate(
            base_model, l, m, args.amplitude, timestamps
        )
        
        # Compute spectra
        print(f"  Computing spectra for {len(timestamps)} time steps...")
        spectra = compute_spectra(models_at_times, ge, wvs)
        
        # Convert to numpy and compute normalized spectra
        spectra_array = np.array([np.array(s) for s in spectra])
        
        # Store results
        results['spectra_by_mode'][(l, m)] = {
            'raw_spectra': spectra_array,  # Shape: (n_times, n_wavelengths, 2)
            'normalized': spectra_array[:, :, 0] / spectra_array[:, :, 1]
        }
        
        print(f"  Done. Spectra shape: {spectra_array.shape}")
    
    # Save results
    save_results(args.output, results)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Modes probed: {len(mode_pairs)}")
    print(f"Time steps per mode: {args.n_times}")
    print(f"Wavelength points: {args.n_wavelengths}")
    print(f"Wavelength range: {args.wavelength_min:.1f} - {args.wavelength_max:.1f} Å")
    print(f"Total data size: {len(results['spectra_by_mode'])} modes")
    print(f"\nOutput saved to: {args.output}")
    print("\nTo load the data:")
    print(f"  import pickle")
    print(f"  with open('{args.output}', 'rb') as f:")
    print(f"      data = pickle.load(f)")


if __name__ == '__main__':
    main()

