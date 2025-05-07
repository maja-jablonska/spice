#!/usr/bin/env python
import csv
import time
import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Tuple, Callable
import matplotlib.pyplot as plt
from transformer_payne import TransformerPayne, Blackbody

from spice.models import IcosphereModel
from spice.spectrum.spectrum import simulate_observed_flux, DEFAULT_CHUNK_SIZE


tp = TransformerPayne.download()


def measure_performance(mesh_sizes: List[int], wavelength_count: int = 1000) -> List[Tuple[int, float]]:
    """Measure the performance of spectrum calculation for different mesh sizes."""
    results = []
    
    # Create wavelength array
    log_wavelengths = jnp.linspace(3.5, 4.0, wavelength_count)  # log10 of wavelengths
    
    # Ensure JIT compilation by running once
    warmup_model = IcosphereModel.construct(100, 1., 1., tp.to_parameters(),
                                            tp.stellar_parameter_names)
    _ = simulate_observed_flux(tp.blackbody, warmup_model, log_wavelengths)
    
    for size in mesh_sizes:
        model = IcosphereModel.construct(size, 1., 1., tp.to_parameters(),
                                         tp.stellar_parameter_names)
        _ = simulate_observed_flux(tp.intensity, model, log_wavelengths)
        
        # Measure time (5 runs)
        durations = []
        for _ in range(5):
            start_time = time.time()
            _ = simulate_observed_flux(tp.intensity, model, log_wavelengths)
            end_time = time.time()
            durations.append(end_time - start_time)
        
        # Calculate mean and standard deviation
        mean_duration = np.mean(durations)
        std_duration = np.std(durations)
        results.append((size, mean_duration))
        print(f"Mesh size: {size}, Time: {mean_duration:.4f} ± {std_duration:.4f} seconds")
    
    return results


def plot_results(results: List[Tuple[int, float]]) -> None:
    """Plot the performance results."""
    sizes, times = zip(*results)
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times, 'o-')
    plt.xlabel('Mesh size (number of vertices)')
    plt.ylabel('Computation time (seconds)')
    plt.title('Spectrum Calculation Performance')
    plt.grid(True)
    plt.savefig('spectrum_performance.png')
    plt.show()


def measure_with_different_chunk_sizes(mesh_size: int, chunk_sizes: List[int], wavelength_count: int = 1000) -> List[Tuple[int, float]]:
    """Measure performance with different chunk sizes for a fixed mesh size."""
    results = []
    
    # Create wavelength array
    log_wavelengths = jnp.linspace(3.5, 4.0, wavelength_count)  # log10 of wavelengths
    
    # Create model
    model = IcosphereModel.construct(mesh_size, 1., 1., tp.to_parameters(),
                                     tp.stellar_parameter_names)
    _ = simulate_observed_flux(tp.intensity, model, log_wavelengths, chunk_size=chunk_size)
    
    for chunk_size in chunk_sizes:
        # Measure time (5 runs)
        durations = []
        for _ in range(5):
            start_time = time.time()
            _ = simulate_observed_flux(tp.intensity, model, log_wavelengths, chunk_size=chunk_size)
            end_time = time.time()
            durations.append(end_time - start_time)
        
        # Calculate mean and standard deviation
        mean_duration = np.mean(durations)
        std_duration = np.std(durations)
        results.append((chunk_size, mean_duration))
        print(f"Mesh size: {mesh_size}, Chunk size: {chunk_size}, Time: {mean_duration:.4f} ± {std_duration:.4f} seconds")
    
    return results


def plot_chunk_size_results(results: List[Tuple[int, float]], mesh_size: int) -> None:
    """Plot the performance results for different chunk sizes."""
    chunk_sizes, times = zip(*results)
    
    plt.figure(figsize=(10, 6))
    plt.plot(chunk_sizes, times, 'o-')
    plt.xlabel('Chunk size')
    plt.ylabel('Computation time (seconds)')
    plt.title(f'Spectrum Calculation Performance (Mesh size: {mesh_size})')
    plt.grid(True)
    plt.savefig('chunk_size_performance.png')
    plt.show()


if __name__ == "__main__":
    # Test with different mesh sizes
    print("Testing with different mesh sizes...")
    mesh_sizes = [100, 500, 1000, 5000, 10000, 20000]
    results = measure_performance(mesh_sizes)
    # Test with different chunk sizes for a specific mesh size
    print("\nTesting with different chunk sizes...")
    mesh_size = 10000  # Fixed mesh size
    chunk_sizes = [128, 256, 512, 1024, 2048, 4096]
    chunk_results = measure_with_different_chunk_sizes(mesh_size, chunk_sizes)
    
    # Save results to CSV files
    print("\nSaving results to CSV files...")
    
    # Save mesh size performance results
    with open('mesh_size_performance.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Mesh Size', 'Time (seconds)'])
        for mesh_size, time_taken in results:
            writer.writerow([mesh_size, time_taken])
    print(f"Mesh size results saved to mesh_size_performance.csv")
    
    # Save chunk size performance results
    with open('chunk_size_performance.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Chunk Size', 'Time (seconds)'])
        for chunk_size, time_taken in chunk_results:
            writer.writerow([chunk_size, time_taken])
    print(f"Chunk size results saved to chunk_size_performance.csv")
    
    # Plot the chunk size results
    plot_chunk_size_results(chunk_results, mesh_size)