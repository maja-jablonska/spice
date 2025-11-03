#!/usr/bin/env python3
"""
Command-line script for calculating selected lines and CCF profiles for Cepheid variable stars
with configurable mesh resolution.

This script simulates a Cepheid variable star using SPICE models and calculates
cross-correlation function (CCF) profiles for selected spectral lines at different
phases of the pulsation cycle.

Usage:
    python cepheid_ccf_analysis_standalone.py --mesh-resolution 1000 --output-dir results/
    python cepheid_ccf_analysis_standalone.py --mesh-resolution 5000 --lines "4896.439,5049.82" --velocity-range -100 100
"""

import argparse
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from tqdm import tqdm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# JAX configuration
from jax import config
config.update("jax_enable_x64", True)

# SPICE imports
from spice.models import IcosphereModel
from spice.models.mesh_transform import add_rotation, evaluate_rotation, add_pulsation, evaluate_pulsations
from spice.spectrum import simulate_observed_flux
from spice.utils.parameters import modify_mesh_parameter_from_array

# Scientific computing
from scipy.signal import correlate
from scipy.interpolate import interp1d
from scipy import optimize
import jax.numpy as jnp
from transformer_payne import TransformerPayne

config.update("jax_platform_name", "cpu")


def air_to_vacuum(wavelength_air: float) -> float:
    """
    Convert wavelength from air to vacuum using Ciddor 1996 formula.
    
    Parameters
    ----------
    wavelength_air : float
        Wavelength in air (Angstroms)
        
    Returns
    -------
    float
        Wavelength in vacuum (Angstroms)
    """
    sigma2 = (1e4 / wavelength_air)**2  # (um^-2)
    n = 1 + 0.0000834254 + 0.02406147/(130 - sigma2) + 0.00015998/(38.9 - sigma2)
    return wavelength_air * n


# Function to calculate radial velocity using cross-correlation
def calculate_radial_velocity(template_spectrum, observed_spectrum, wavelengths, velocity_range=(-100, 100), num_velocities=200):
    """
    Calculate radial velocity using cross-correlation between template and observed spectrum.
    
    Parameters:
    -----------
    template_spectrum : array-like
        Reference spectrum (flux values)
    observed_spectrum : array-like
        Observed spectrum to measure velocity (flux values)
    wavelengths : array-like
        Wavelength array in Angstroms
    velocity_range : tuple
        Min and max velocity to search in km/s
    num_velocities : int
        Number of velocity points to test
        
    Returns:
    --------
    best_velocity : float
        Radial velocity in km/s
    ccf : array
        Cross-correlation function
    velocities : array
        Velocity grid used for cross-correlation
    """
    # Create velocity grid
    velocities = np.linspace(velocity_range[0], velocity_range[1], num_velocities)
    
    # Speed of light in km/s
    c = 299792.458
    
    # Normalize spectra to have zero mean and unit variance
    template_norm = (template_spectrum - np.mean(template_spectrum)) / np.std(template_spectrum)
    observed_norm = (observed_spectrum - np.mean(observed_spectrum)) / np.std(observed_spectrum)
    
    # Calculate cross-correlation for each velocity shift
    ccf = np.zeros(len(velocities))
    
    for i, v in enumerate(velocities):
        # Calculate wavelength shift for this velocity
        doppler_factor = np.sqrt((1 + v/c) / (1 - v/c))  # Relativistic Doppler formula
        shifted_wavelengths = wavelengths * doppler_factor
        
        # Interpolate the template onto the shifted wavelength grid
        shifted_template = interp1d(shifted_wavelengths, template_norm, 
                                    bounds_error=False, fill_value=0)(wavelengths)
        
        # Calculate correlation
        ccf[i] = np.sum(shifted_template * observed_norm)
    
    # Find the velocity with maximum correlation
    best_idx = np.argmax(ccf)
    best_velocity = -velocities[best_idx]  # Flip the sign of the velocity
    
    return best_velocity, ccf, velocities


def create_cepheid_model(mesh_resolution: int, 
                        stellar_params: Dict[str, float],
                        fourier_params: np.ndarray,
                        period: float) -> Tuple[Any, Any]:
    """
    Create a Cepheid model with pulsation using SPICE.
    
    Parameters
    ----------
    mesh_resolution : int
        Number of mesh elements for the icosphere
    stellar_params : dict
        Stellar parameters (Teff, logg, radius, mass, Fe)
    fourier_params : array
        Fourier series parameters for pulsation
    period : float
        Pulsation period in days
        
    Returns
    -------
    tuple
        (base_model, pulsation_model)
    """
    print(f"Creating Cepheid model with {mesh_resolution} mesh elements...")
    
    # Use simple blackbody model
    tp = TransformerPayne.download()
    
    # Create base model
    base_model = IcosphereModel.construct(
        mesh_resolution, 
        stellar_params['radius'], 
        stellar_params['mass'],
        tp.to_parameters({'logteff': np.log10(stellar_params['Teff']), 'logg': stellar_params['logg'], 'Fe': stellar_params['Fe']}),
        tp.stellar_parameter_names,
        max_fourier_order=10
    )
    
    # Add pulsation
    pulsation_model = add_rotation(
        add_pulsation(base_model, m_order=0, l_degree=0, 
                     period=period, fourier_series_parameters=fourier_params), 
        0.0
    )
    
    return base_model, pulsation_model


def simulate_spectra_at_phases(pulsation_model: Any, 
                              base_model: Any,
                              line_centers: List[float],
                              line_width: float,
                              steps: int,
                              phases: np.ndarray,
                              stellar_params: Dict[str, float]) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Simulate spectra at different phases for selected lines.
    
    Parameters
    ----------
    pulsation_model : Any
        Pulsation model
    base_model : Any
        Base model for template
    line_centers : list
        List of line center wavelengths (in air)
    line_width : float
        Width around each line center to simulate
    steps : int
        Number of wavelength steps
    phases : array
        Array of phases to simulate
    stellar_params : dict
        Stellar parameters for temperature/logg interpolation
        
    Returns
    -------
    tuple
        (wavelengths, spectra, templates)
    """
    print("Simulating spectra at different phases...")
    
    # Initialize spectrum model
    tp = TransformerPayne.download()
    intensity_func = tp.intensity
    
    # Convert line centers to vacuum
    line_centers_vacuum = [air_to_vacuum(lc) for lc in line_centers]
    
    wavelengths = []
    spectra = []
    templates = []
    
    for i, lc in enumerate(tqdm(line_centers_vacuum, desc="Processing lines")):
        # Create wavelength grid around line center
        vw = jnp.linspace(lc - line_width, lc + line_width, steps)
        wavelengths.append(vw)
        
        # Simulate spectra for all phases
        current_line_spectra = []
        for phase in tqdm(phases, desc=f"Line {i+1}/{len(line_centers_vacuum)}", leave=False):
            # Evaluate pulsation at this phase
            evaluated_model = evaluate_rotation(evaluate_pulsations(pulsation_model, phase), phase)
            
            # Modify temperature based on phase (simplified interpolation)
            # In a real implementation, you'd interpolate from SPIPS3 data
            teff_phase = stellar_params['Teff'] * (1 + 0.1 * np.sin(2 * np.pi * phase))  # Simplified
            
            # Modify mesh parameters
            evaluated_model = modify_mesh_parameter_from_array(evaluated_model, 0, teff_phase)
            
            # Simulate spectrum
            flux = simulate_observed_flux(intensity_func, evaluated_model, jnp.log10(vw))
            current_line_spectra.append(flux)
        
        spectra.append(current_line_spectra)
        
        # Create template spectrum (phase 0)
        template_flux = simulate_observed_flux(intensity_func, base_model, jnp.log10(vw))
        templates.append(template_flux)
    
    return wavelengths, spectra, templates


def calculate_ccf_profiles(spectra: List[List[np.ndarray]], 
                          templates: List[np.ndarray],
                          wavelengths: List[np.ndarray],
                          velocity_range: Tuple[float, float] = (-50, 50)) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Calculate CCF profiles for all lines and phases.
    
    Parameters
    ----------
    spectra : list
        List of spectra for each line and phase
    templates : list
        List of template spectra for each line
    wavelengths : list
        List of wavelength arrays for each line
    velocity_range : tuple
        Velocity range for CCF calculation
        
    Returns
    -------
    tuple
        (radial_velocities, ccf_profiles)
    """
    print("Calculating CCF profiles...")
    
    radial_velocities = []
    ccf_profiles = []
    
    for j in tqdm(range(len(spectra)), desc="Processing lines"):
        # Use the first spectrum as template
        template_spectrum = templates[j][:, 0] / templates[j][:, 1]
        
        # Calculate radial velocities for all phases
        line_radial_velocities = []
        line_ccf_profiles = []
        
        for i, spectrum in enumerate(spectra[j]):
            if i == 0:  # Skip the template
                line_radial_velocities.append(0.0)
                line_ccf_profiles.append(np.zeros(200))  # Placeholder
                continue
                
            rv, ccf, velocities = calculate_radial_velocity(
                template_spectrum, 
                spectrum[:, 0] / spectrum[:, 1], 
                wavelengths[j], 
                velocity_range=velocity_range
            )
            line_radial_velocities.append(rv)
            line_ccf_profiles.append(ccf)
        
        radial_velocities.append(line_radial_velocities)
        ccf_profiles.append(line_ccf_profiles)
    
    return np.array(radial_velocities), ccf_profiles


def save_results(output_dir: str, 
                radial_velocities: np.ndarray,
                ccf_profiles: List[np.ndarray],
                wavelengths: List[np.ndarray],
                spectra: List[List[np.ndarray]],
                phases: np.ndarray,
                line_centers: List[float],
                mesh_resolution: int) -> None:
    """
    Save results to files.
    
    Parameters
    ----------
    output_dir : str
        Output directory
    radial_velocities : array
        Calculated radial velocities
    ccf_profiles : list
        CCF profiles for each line
    wavelengths : list
        Wavelength arrays
    phases : array
        Phase array
    line_centers : list
        Line center wavelengths
    mesh_resolution : int
        Mesh resolution used
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save data as pickle
    data = {
        'radial_velocities': radial_velocities,
        'ccf_profiles': ccf_profiles,
        'wavelengths': wavelengths,
        'spectra': spectra,
        'phases': phases,
        'line_centers': line_centers,
        'mesh_resolution': mesh_resolution
    }
    
    pickle_file = os.path.join(output_dir, f'cepheid_ccf_mesh_{mesh_resolution}.pkl')
    with open(pickle_file, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Results saved to {pickle_file}")


def create_plots(output_dir: str,
                radial_velocities: np.ndarray,
                ccf_profiles: List[np.ndarray],
                phases: np.ndarray,
                line_centers: List[float],
                mesh_resolution: int) -> None:
    """
    Create and save plots.
    
    Parameters
    ----------
    output_dir : str
        Output directory
    radial_velocities : array
        Calculated radial velocities
    ccf_profiles : list
        CCF profiles for each line
    phases : array
        Phase array
    line_centers : list
        Line center wavelengths
    mesh_resolution : int
        Mesh resolution used
    """
    print("Creating plots...")
    
    # Plot radial velocity curves
    plt.figure(figsize=(12, 8))
    
    for i, (rv, lc) in enumerate(zip(radial_velocities, line_centers)):
        plt.subplot(2, 3, i+1)
        plt.plot(phases[1:], rv[1:], 'o-', label=f'Line {lc:.1f} Å')
        plt.xlabel('Phase')
        plt.ylabel('Radial Velocity (km/s)')
        plt.title(f'Line {lc:.1f} Å')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'radial_velocity_curves_mesh_{mesh_resolution}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot CCF profiles for one line
    if ccf_profiles:
        plt.figure(figsize=(10, 6))
        velocities = np.linspace(-50, 50, len(ccf_profiles[0][0]))
        
        for i, phase in enumerate(phases[1:10]):  # Plot first 9 phases
            plt.plot(velocities, ccf_profiles[0][i+1], alpha=0.7, label=f'Phase {phase:.2f}')
        
        plt.xlabel('Velocity (km/s)')
        plt.ylabel('CCF')
        plt.title(f'CCF Profiles - Line {line_centers[0]:.1f} Å (Mesh: {mesh_resolution})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'ccf_profiles_mesh_{mesh_resolution}.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Plots saved to {output_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Calculate CCF profiles for Cepheid variable stars')
    
    # Required arguments
    parser.add_argument('--mesh-resolution', type=int, required=True,
                       help='Mesh resolution (number of elements)')
    
    # Optional arguments
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results (default: results)')
    parser.add_argument('--lines', type=str, 
                       default='4896.439,5049.82,5044.21,4924.77,6347,6371',
                       help='Comma-separated list of line centers in air (default: Fe I and Si II lines)')
    parser.add_argument('--line-width', type=float, default=2.0,
                       help='Width around line centers in Angstroms (default: 2.0)')
    parser.add_argument('--steps', type=int, default=1000,
                       help='Number of wavelength steps (default: 1000)')
    parser.add_argument('--phases', type=int, default=100,
                       help='Number of phases to simulate (default: 100)')
    parser.add_argument('--velocity-range', nargs=2, type=float, default=[-50, 50],
                       help='Velocity range for CCF calculation in km/s (default: -50 50)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip creating plots')
    
    args = parser.parse_args()
    
    # Parse line centers
    line_centers = [float(x.strip()) for x in args.lines.split(',')]
    
    # Default stellar parameters for Delta Cephei
    stellar_params = {
        'Teff': 6562.0,  # K
        'logg': 1.883,   # dex
        'radius': 41.18, # R_sun
        'mass': 5.26,    # M_sun
        'Fe': 0.08       # dex
    }
    
    # Default Fourier parameters (simplified)
    fourier_params = np.array([
        [5.08410023e-02,  2.52778534e+00],
        [1.07786710e-02,  1.89320108e+00],
        [3.64884277e-03,  1.24266937e+00],
        [1.44494986e-03,  5.48410078e-01],
        [6.17318686e-04, -2.43612157e-01],
        [2.88661671e-04, -1.15420866e+00],
        [1.52119818e-04, -2.10861958e+00],
        [8.50754672e-05, -3.00762780e+00],
        [8.50754672e-05, -3.00762780e+00]
    ])
    
    period = 5.366  # days
    
    print(f"Starting Cepheid CCF analysis with mesh resolution: {args.mesh_resolution}")
    print(f"Line centers: {line_centers}")
    print(f"Output directory: {args.output_dir}")
    
    try:
        # Create models
        base_model, pulsation_model = create_cepheid_model(
            args.mesh_resolution, stellar_params, fourier_params, period
        )
        
        # Create phase array
        phases = np.linspace(0, 1, args.phases)
        
        # Simulate spectra
        wavelengths, spectra, templates = simulate_spectra_at_phases(
            pulsation_model, base_model, line_centers, 
            args.line_width, args.steps, phases, stellar_params
        )
        
        # Calculate CCF profiles
        radial_velocities, ccf_profiles = calculate_ccf_profiles(
            spectra, templates, wavelengths, tuple(args.velocity_range)
        )
        
        # Save results
        save_results(args.output_dir, radial_velocities, ccf_profiles, 
                    wavelengths, spectra, phases, line_centers, args.mesh_resolution)
        
        # Create plots
        if not args.no_plots:
            create_plots(args.output_dir, radial_velocities, ccf_profiles, 
                        phases, line_centers, args.mesh_resolution)
        
        print("Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
