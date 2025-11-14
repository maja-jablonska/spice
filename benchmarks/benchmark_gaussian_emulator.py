#!/usr/bin/env python
"""
Benchmark script for GaussianLineEmulator performance testing.

This script measures the performance of the GaussianLineEmulator across various
mesh resolutions for different operations:
1. Mesh creation + spectrum simulation
2. Spectrum simulation only
3. Rotation evaluation + spectrum simulation
4. Pulsation evaluation + spectrum simulation

Optionally, can also benchmark TransformerPayne for comparison.
"""

import sys
import time
import csv
import argparse
from typing import List, Tuple, Dict, Optional
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add src to path
sys.path.insert(0, '../src')

from spice.models import IcosphereModel
from spice.models.mesh_transform import add_rotation, evaluate_rotation, add_pulsation, evaluate_pulsations
from spice.spectrum.spectrum import simulate_observed_flux
from spice.spectrum.gaussian_line_emulator import GaussianLineEmulator

# Try to import TransformerPayne (optional)
try:
    from transformer_payne import TransformerPayne
    TRANSFORMER_PAYNE_AVAILABLE = True
except (ImportError, NameError, AttributeError) as e:
    TRANSFORMER_PAYNE_AVAILABLE = False
    TransformerPayne = None

# Configure plot style
plt.style.use('seaborn-v0_8-darkgrid')

# Configuration
MESH_RESOLUTIONS = [300, 1000, 5000, 20000]
WAVELENGTH_MIN = 5496.0
WAVELENGTH_MAX = 5504.0
N_WAVELENGTHS = 2000
N_TIMING_RUNS = 5
ROTATION_VELOCITY = 50.0  # km/s
PULSATION_M_ORDER = 4
PULSATION_L_DEGREE = 4
PULSATION_PERIOD = 1.0
N_PULSATION_SAMPLES = 10


def create_emulator(use_transformer_payne: bool = False):
    """Create an emulator (GaussianLineEmulator or TransformerPayne).
    
    Args:
        use_transformer_payne: If True, use TransformerPayne instead of GaussianLineEmulator
        n_wavelengths: Number of wavelengths (used for TransformerPayne wavelength range)
    
    Returns:
        Emulator object (GaussianLineEmulator or TransformerPayne)
    """
    if use_transformer_payne:
        if not TRANSFORMER_PAYNE_AVAILABLE:
            raise ImportError("TransformerPayne is not available. Install it to use this option.")
        print("Loading TransformerPayne...")
        return TransformerPayne.download()
    else:
        return GaussianLineEmulator(
            line_centers=[5500.0],
            line_widths=[0.3],
            line_depths=[0.5]
        )


def create_wavelength_array(use_transformer_payne: bool = False, n_wavelengths: int = N_WAVELENGTHS) -> jnp.ndarray:
    """Create wavelength array for spectrum simulation.
    
    Args:
        use_transformer_payne: If True, use TransformerPayne wavelength range
        n_wavelengths: Number of wavelength points
    
    Returns:
        log10 of wavelengths in Angstroms
    """
    if use_transformer_payne:
        wavelengths = jnp.linspace(WAVELENGTH_MIN, WAVELENGTH_MAX, n_wavelengths)
    else:
        wavelengths = jnp.linspace(WAVELENGTH_MIN, WAVELENGTH_MAX, n_wavelengths)
    return jnp.log10(wavelengths)


def warmup_jit(emulator, log_wavelengths: jnp.ndarray):
    """Warmup JIT compilation with a small mesh.
    
    Args:
        emulator: GaussianLineEmulator or TransformerPayne instance
        log_wavelengths: log10 of wavelengths
    """
    print("Warming up JIT compilation...")
    
    # Get parameters based on emulator type
    if hasattr(emulator, 'to_parameters'):
        params = emulator.to_parameters()
    else:
        # TransformerPayne uses solar_parameters
        params = emulator.solar_parameters
    
    warmup_model = IcosphereModel.construct(
        100, 1.0, 1.0,
        params,
        emulator.stellar_parameter_names
    )
    _ = simulate_observed_flux(emulator.intensity, warmup_model, log_wavelengths)
    print("JIT warmup complete.")


def benchmark_mesh_creation_and_spectrum(
    resolutions: List[int],
    emulator,
    log_wavelengths: jnp.ndarray
) -> Dict[int, Tuple[float, float]]:
    """
    Benchmark Operation 1: Mesh Creation + Spectrum Simulation
    
    Measures the complete workflow from mesh construction to spectrum simulation.
    
    Args:
        resolutions: List of mesh resolutions to test
        emulator: GaussianLineEmulator or TransformerPayne instance
        log_wavelengths: log10 of wavelengths
    
    Returns:
        Dict mapping resolution to (mean_time, std_time)
    """
    print("\n=== Benchmark 1: Mesh Creation + Spectrum Simulation ===")
    results = {}
    
    # Get parameters based on emulator type
    if hasattr(emulator, 'to_parameters'):
        params = emulator.to_parameters()
    else:
        params = emulator.solar_parameters
    
    for resolution in resolutions:
        print(f"Testing resolution {resolution}...")
        times = []
        
        for run in range(N_TIMING_RUNS):
            start_time = time.time()
            
            # Create mesh
            model = IcosphereModel.construct(
                resolution, 1.0, 1.0,
                params,
                emulator.stellar_parameter_names
            )
            
            # Simulate spectrum
            _ = simulate_observed_flux(emulator.intensity, model, log_wavelengths)
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        results[resolution] = (mean_time, std_time)
        print(f"  Resolution {resolution}: {mean_time:.4f} ± {std_time:.4f} s")
    
    return results


def benchmark_spectrum_only(
    resolutions: List[int],
    emulator,
    log_wavelengths: jnp.ndarray
) -> Dict[int, Tuple[float, float]]:
    """
    Benchmark Operation 2: Spectrum Simulation Only
    
    Pre-creates meshes and measures only spectrum computation time.
    
    Args:
        resolutions: List of mesh resolutions to test
        emulator: GaussianLineEmulator or TransformerPayne instance
        log_wavelengths: log10 of wavelengths
    
    Returns:
        Dict mapping resolution to (mean_time, std_time)
    """
    print("\n=== Benchmark 2: Spectrum Simulation Only ===")
    results = {}
    
    # Get parameters based on emulator type
    if hasattr(emulator, 'to_parameters'):
        params = emulator.to_parameters()
    else:
        params = emulator.solar_parameters
    
    # Pre-create all meshes
    models = {}
    for resolution in resolutions:
        print(f"Pre-creating mesh for resolution {resolution}...")
        models[resolution] = IcosphereModel.construct(
            resolution, 1.0, 1.0,
            params,
            emulator.stellar_parameter_names
        )
        # Warmup for this specific mesh size
        _ = simulate_observed_flux(emulator.intensity, models[resolution], log_wavelengths)
    
    # Benchmark spectrum simulation
    for resolution in resolutions:
        print(f"Testing resolution {resolution}...")
        times = []
        model = models[resolution]
        
        for run in range(N_TIMING_RUNS):
            start_time = time.time()
            _ = simulate_observed_flux(emulator.intensity, model, log_wavelengths)
            end_time = time.time()
            times.append(end_time - start_time)
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        results[resolution] = (mean_time, std_time)
        print(f"  Resolution {resolution}: {mean_time:.4f} ± {std_time:.4f} s")
    
    return results


def benchmark_rotation_and_spectrum(
    resolutions: List[int],
    emulator,
    log_wavelengths: jnp.ndarray
) -> Dict[int, Tuple[float, float]]:
    """
    Benchmark Operation 3: Rotation Evaluation + Spectrum
    
    Adds rotation to model, evaluates at phase 0.0, and simulates spectrum.
    
    Args:
        resolutions: List of mesh resolutions to test
        emulator: GaussianLineEmulator or TransformerPayne instance
        log_wavelengths: log10 of wavelengths
    
    Returns:
        Dict mapping resolution to (mean_time, std_time)
    """
    print("\n=== Benchmark 3: Rotation Evaluation + Spectrum ===")
    results = {}
    
    # Get parameters based on emulator type
    if hasattr(emulator, 'to_parameters'):
        params = emulator.to_parameters()
    else:
        params = emulator.solar_parameters
    
    rotation_axis = jnp.array([0., 0., 1.])
    
    for resolution in resolutions:
        print(f"Testing resolution {resolution}...")
        
        # Create base model
        base_model = IcosphereModel.construct(
            resolution, 1.0, 1.0,
            params,
            emulator.stellar_parameter_names
        )
        
        # Add rotation
        rotating_model = add_rotation(base_model, ROTATION_VELOCITY, rotation_axis)
        
        # Warmup
        evaluated = evaluate_rotation(rotating_model, 0.0)
        _ = simulate_observed_flux(emulator.intensity, evaluated, log_wavelengths)
        
        # Benchmark
        times = []
        for run in range(N_TIMING_RUNS):
            start_time = time.time()
            
            # Evaluate rotation
            evaluated = evaluate_rotation(rotating_model, 0.0)
            
            # Simulate spectrum
            _ = simulate_observed_flux(emulator.intensity, evaluated, log_wavelengths)
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        results[resolution] = (mean_time, std_time)
        print(f"  Resolution {resolution}: {mean_time:.4f} ± {std_time:.4f} s")
    
    return results


def benchmark_pulsation_and_spectrum(
    resolutions: List[int],
    emulator,
    log_wavelengths: jnp.ndarray
) -> Dict[int, Tuple[float, float]]:
    """
    Benchmark Operation 4: Pulsation Evaluation + Spectrum
    
    Adds pulsation, evaluates at multiple time points, and simulates spectra.
    
    Args:
        resolutions: List of mesh resolutions to test
        emulator: GaussianLineEmulator or TransformerPayne instance
        log_wavelengths: log10 of wavelengths
    
    Returns:
        Dict mapping resolution to (mean_time, std_time)
    """
    print("\n=== Benchmark 4: Pulsation Evaluation + Spectrum ===")
    results = {}
    
    # Get parameters based on emulator type
    if hasattr(emulator, 'to_parameters'):
        params = emulator.to_parameters()
    else:
        params = emulator.solar_parameters
    
    timeseries = jnp.linspace(0, 1., N_PULSATION_SAMPLES)
    
    for resolution in resolutions:
        print(f"Testing resolution {resolution}...")
        
        # Create base model
        base_model = IcosphereModel.construct(
            resolution, 1.0, 1.0,
            params,
            emulator.stellar_parameter_names
        )
        
        # Add pulsation
        pulsating_model = add_pulsation(
            base_model,
            m_order=PULSATION_M_ORDER,
            l_degree=PULSATION_L_DEGREE,
            period=PULSATION_PERIOD,
            fourier_series_parameters=np.array([[1., 1.]])
        )
        
        # Warmup
        evaluated = evaluate_pulsations(pulsating_model, 0.0)
        _ = simulate_observed_flux(emulator.intensity, evaluated, log_wavelengths)
        
        # Benchmark
        times = []
        for run in range(N_TIMING_RUNS):
            start_time = time.time()
            
            # Evaluate pulsation at all time points and simulate spectra
            for t in timeseries:
                evaluated = evaluate_pulsations(pulsating_model, t)
                _ = simulate_observed_flux(emulator.intensity, evaluated, log_wavelengths)
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        results[resolution] = (mean_time, std_time)
        print(f"  Resolution {resolution}: {mean_time:.4f} ± {std_time:.4f} s " +
              f"(total for {N_PULSATION_SAMPLES} time samples)")
    
    return results


def save_results_to_csv(all_results: Dict[str, Dict[int, Tuple[float, float]]], filename: str):
    """Save all benchmark results to a CSV file."""
    print(f"\nSaving results to {filename}...")
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['mesh_resolution', 'operation', 'mean_time_sec', 'std_time_sec'])
        
        for operation_name, results in all_results.items():
            for resolution, (mean_time, std_time) in results.items():
                writer.writerow([resolution, operation_name, mean_time, std_time])
    
    print(f"Results saved to {filename}")


def plot_results(all_results: Dict[str, Dict[int, Tuple[float, float]]], filename: str):
    """Create a multi-panel plot showing all benchmark results."""
    print(f"\nGenerating plot: {filename}...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    operations = [
        ('mesh_creation_spectrum', 'Mesh Creation + Spectrum'),
        ('spectrum_only', 'Spectrum Simulation Only'),
        ('rotation_spectrum', 'Rotation + Spectrum'),
        ('pulsation_spectrum', 'Pulsation + Spectrum')
    ]
    
    for idx, (op_key, op_title) in enumerate(operations):
        ax = axes[idx]
        results = all_results[op_key]
        
        resolutions = sorted(results.keys())
        mean_times = [results[r][0] for r in resolutions]
        std_times = [results[r][1] for r in resolutions]
        
        # Plot with error bars
        ax.errorbar(resolutions, mean_times, yerr=std_times, 
                   marker='o', linestyle='-', linewidth=2, 
                   markersize=8, capsize=5, capthick=2)
        
        ax.set_xlabel('Mesh Resolution (vertices)', fontsize=11)
        ax.set_ylabel('Time (seconds)', fontsize=11)
        ax.set_title(op_title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        # Add value labels
        for r, m in zip(resolutions, mean_times):
            ax.text(r, m, f'{m:.3f}s', fontsize=8, ha='left', va='bottom')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {filename}")
    plt.show()


def print_summary(all_results: Dict[str, Dict[int, Tuple[float, float]]]):
    """Print a formatted summary of all benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    for operation_name, results in all_results.items():
        print(f"\n{operation_name.upper().replace('_', ' ')}")
        print("-" * 80)
        print(f"{'Resolution':<15} {'Mean Time (s)':<20} {'Std Dev (s)':<20}")
        print("-" * 80)
        
        for resolution in sorted(results.keys()):
            mean_time, std_time = results[resolution]
            print(f"{resolution:<15} {mean_time:<20.4f} {std_time:<20.4f}")
    
    print("\n" + "=" * 80)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Benchmark GaussianLineEmulator or TransformerPayne performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (GaussianLineEmulator)
  python benchmark_gaussian_emulator.py
  
  # Run with TransformerPayne and 1000 wavelengths
  python benchmark_gaussian_emulator.py --use-transformer-payne --n-wavelengths 1000
  
  # Run with custom mesh resolutions
  python benchmark_gaussian_emulator.py --mesh-resolutions 500 2000 10000
  
  # Run with more timing runs for better statistics
  python benchmark_gaussian_emulator.py --n-runs 10
        """
    )
    
    parser.add_argument(
        '--use-transformer-payne',
        action='store_true',
        help='Use TransformerPayne instead of GaussianLineEmulator'
    )
    
    parser.add_argument(
        '--n-wavelengths',
        type=int,
        default=N_WAVELENGTHS,
        help=f'Number of wavelength points (default: {N_WAVELENGTHS})'
    )
    
    parser.add_argument(
        '--mesh-resolutions',
        nargs='+',
        type=int,
        default=MESH_RESOLUTIONS,
        help=f'Mesh resolutions to test (default: {MESH_RESOLUTIONS})'
    )
    
    parser.add_argument(
        '--n-runs',
        type=int,
        default=N_TIMING_RUNS,
        help=f'Number of timing runs per configuration (default: {N_TIMING_RUNS})'
    )
    
    parser.add_argument(
        '--output-prefix',
        type=str,
        default=None,
        help='Prefix for output files (default: auto-generated based on emulator)'
    )
    
    return parser.parse_args()


def main():
    """Main benchmark execution."""
    args = parse_arguments()
    
    # Update global configuration based on arguments
    global N_TIMING_RUNS
    N_TIMING_RUNS = args.n_runs
    
    mesh_resolutions = args.mesh_resolutions
    n_wavelengths = args.n_wavelengths
    use_transformer_payne = args.use_transformer_payne
    
    # Determine emulator name for output
    emulator_name = "TransformerPayne" if use_transformer_payne else "GaussianLineEmulator"
    
    # Set output file prefix
    if args.output_prefix:
        output_prefix = args.output_prefix
    else:
        output_prefix = f"{'tpayne' if use_transformer_payne else 'gaussian'}_emulator"
    
    print(f"{emulator_name} Performance Benchmark")
    print("=" * 80)
    print(f"Emulator: {emulator_name}")
    print(f"Mesh resolutions: {mesh_resolutions}")
    print(f"Number of wavelengths: {n_wavelengths}")
    if use_transformer_payne:
        print(f"Wavelength range: 3000-10000 Å")
    else:
        print(f"Wavelength range: {WAVELENGTH_MIN}-{WAVELENGTH_MAX} Å")
    print(f"Number of timing runs per configuration: {N_TIMING_RUNS}")
    print(f"Rotation velocity: {ROTATION_VELOCITY} km/s")
    print(f"Pulsation config: m={PULSATION_M_ORDER}, l={PULSATION_L_DEGREE}, " +
          f"{N_PULSATION_SAMPLES} time samples")
    print("=" * 80)
    
    # Setup
    emulator = create_emulator(use_transformer_payne)
    log_wavelengths = create_wavelength_array(use_transformer_payne, n_wavelengths)
    
    # Warmup JIT
    warmup_jit(emulator, log_wavelengths)
    
    # Run all benchmarks
    all_results = {}
    
    all_results['mesh_creation_spectrum'] = benchmark_mesh_creation_and_spectrum(
        mesh_resolutions, emulator, log_wavelengths
    )
    
    all_results['spectrum_only'] = benchmark_spectrum_only(
        mesh_resolutions, emulator, log_wavelengths
    )
    
    all_results['rotation_spectrum'] = benchmark_rotation_and_spectrum(
        mesh_resolutions, emulator, log_wavelengths
    )
    
    all_results['pulsation_spectrum'] = benchmark_pulsation_and_spectrum(
        mesh_resolutions, emulator, log_wavelengths
    )
    
    # Save and visualize results
    csv_filename = f'{output_prefix}_benchmark_results.csv'
    plot_filename = f'{output_prefix}_performance.png'
    
    save_results_to_csv(all_results, csv_filename)
    plot_results(all_results, plot_filename)
    print_summary(all_results)
    
    print(f"\nBenchmark complete!")
    print(f"Results saved to: {csv_filename}")
    print(f"Plot saved to: {plot_filename}")


if __name__ == "__main__":
    # Enable 64-bit precision
    jax.config.update("jax_enable_x64", True)
    
    main()

