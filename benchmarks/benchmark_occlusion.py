"""
Benchmark script to compare performance of grid-based and KD-tree-based occlusion detection.
"""

import argparse
import os
import sys

# Prefer the in-repo source tree when running from a checkout (mirrors the
# other benchmark scripts).
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if os.path.isdir(os.path.join(_SRC, "spice")):
    sys.path.insert(0, _SRC)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)  # so sibling _bench_common imports resolve

from _bench_common import _resolve_precisions

import jax.numpy as jnp
import numpy as np
import time
import matplotlib.pyplot as plt
import jax

from spice.models.mesh_model import IcosphereModel
from spice.models.binary import Binary, evaluate_orbit
from spice.models.mesh_transform import transform
from spice.models.mesh_view import Grid, resolve_occlusion as grid_resolve_occlusion
from spice.models.mesh_view_kdtree import resolve_occlusion as kdtree_resolve_occlusion, get_optimal_search_radius

# Configure plot style
plt.style.use('seaborn-v0_8-darkgrid')


def create_test_spheres(resolution=642):
    """Create two test spheres for occlusion detection benchmark."""
    sphere1 = IcosphereModel.construct(
        n_vertices=resolution,
        radius=1.0,      # 1 solar radius
        mass=1.0,        # 1 solar mass
        parameters=jnp.array([5800., 4.5]),  # Teff, log g
        parameter_names=["teff", "logg"]
    )
    
    sphere2 = IcosphereModel.construct(
        n_vertices=resolution,
        radius=0.5,      # 0.5 solar radius
        mass=0.5,        # 0.5 solar mass
        parameters=jnp.array([5000., 4.0]),  # Teff, log g
        parameter_names=["teff", "logg"]
    )
    
    # Position sphere2 in front of sphere1
    sphere2 = transform(sphere2, jnp.array([0.0, 0.0, -2.0]))
    
    return sphere1, sphere2


def benchmark_occlusion_methods(n_runs=5, resolutions=[162, 642, 2562]):
    """
    Benchmark grid-based vs KD-tree-based occlusion detection.
    
    Args:
        n_runs: Number of runs to average timing over
        resolutions: List of mesh resolutions to test
        
    Returns:
        Tuple of timing results
    """
    results = {
        'resolutions': resolutions,
        'grid_times': [],
        'kdtree_times': [],
        'grid_triangles': [],
        'kdtree_triangles': []
    }
    
    for resolution in resolutions:
        print(f"Testing resolution: {resolution}")
        
        # Create test spheres with current resolution
        sphere1, sphere2 = create_test_spheres(resolution)
        
        # Calculate number of triangles
        n_triangles = sphere1.faces.shape[0] + sphere2.faces.shape[0]
        results['grid_triangles'].append(n_triangles)
        results['kdtree_triangles'].append(n_triangles)
        
        # Benchmark grid-based method
        grid_times = []
        for _ in range(n_runs):
            grid = Grid.construct(sphere1, sphere2, 20)
            
            start_time = time.time()
            # Compile and run once to warm up JAX
            _ = grid_resolve_occlusion(sphere1, sphere2, grid)
            end_time = time.time()
            
            # Run the actual benchmark
            start_time = time.time()
            _ = grid_resolve_occlusion(sphere1, sphere2, grid)
            end_time = time.time()
            grid_times.append(end_time - start_time)
        
        results['grid_times'].append(np.mean(grid_times))
        
        # Benchmark KD-tree-based method
        kdtree_times = []
        for _ in range(n_runs):
            search_radius = get_optimal_search_radius(sphere1, sphere2)
            
            start_time = time.time()
            # Compile and run once to warm up JAX
            _ = kdtree_resolve_occlusion(sphere1, sphere2, search_radius)
            end_time = time.time()
            
            # Run the actual benchmark
            start_time = time.time()
            _ = kdtree_resolve_occlusion(sphere1, sphere2, search_radius)
            end_time = time.time()
            kdtree_times.append(end_time - start_time)
        
        results['kdtree_times'].append(np.mean(kdtree_times))
    
    return results


def plot_benchmark_results(results, filename='occlusion_benchmark_results.png'):
    """
    Plot benchmark results.

    Args:
        results: Results dictionary from benchmark_occlusion_methods
        filename: Path for the saved PNG.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot raw times
    ax1.plot(results['resolutions'], results['grid_times'], 'o-', label='Grid-based')
    ax1.plot(results['resolutions'], results['kdtree_times'], 's-', label='KD-tree-based')
    ax1.set_xlabel('Mesh Resolution (vertices)')
    ax1.set_ylabel('Time (s)')
    ax1.set_title('Occlusion Detection Performance')
    ax1.legend()
    ax1.grid(True)
    
    # Plot speedup
    speedup = np.array(results['grid_times']) / np.array(results['kdtree_times'])
    ax2.plot(results['resolutions'], speedup, 'o-')
    ax2.axhline(y=1.0, color='r', linestyle='--')
    ax2.set_xlabel('Mesh Resolution (vertices)')
    ax2.set_ylabel('Speedup (Grid Time / KD-tree Time)')
    ax2.set_title('KD-tree Speedup over Grid-based Method')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Wrote plot to {filename}")


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Benchmark grid-based vs KD-tree-based occlusion detection.")
    p.add_argument("--precision", choices=("x32", "x64", "both"), default="both",
                   help="JAX float precision; 'both' runs the benchmark in x32 and x64 "
                        "(default both)")
    p.add_argument("--n-runs", type=int, default=5, help="runs to average per resolution")
    p.add_argument("--resolutions", type=int, nargs="+", default=[162, 642, 2562],
                   help="mesh resolutions (n_vertices) to test")
    p.add_argument("--plot", default="occlusion_benchmark_results.png",
                   help="output PNG path (precision is appended when running 'both')")
    args = p.parse_args(argv)

    precisions = _resolve_precisions(args.precision)
    for precision in precisions:
        jax.config.update("jax_enable_x64", precision == "x64")
        print(f"\n########## precision: {precision} "
              f"(jax_enable_x64={jax.config.read('jax_enable_x64')}) ##########")
        print("Starting occlusion detection benchmark...")
        results = benchmark_occlusion_methods(n_runs=args.n_runs, resolutions=args.resolutions)

        print("\nBenchmark Results:")
        print("-----------------")
        print("Resolutions:", results['resolutions'])
        print("Grid Times (s):", [f"{t:.4f}" for t in results['grid_times']])
        print("KD-tree Times (s):", [f"{t:.4f}" for t in results['kdtree_times']])

        speedup = np.array(results['grid_times']) / np.array(results['kdtree_times'])
        print("Speedup (Grid/KD-tree):", [f"{s:.2f}x" for s in speedup])

        if len(precisions) > 1:
            root, ext = os.path.splitext(args.plot)
            out = f"{root}_{precision}{ext or '.png'}"
        else:
            out = args.plot
        plot_benchmark_results(results, out)


if __name__ == "__main__":
    main() 