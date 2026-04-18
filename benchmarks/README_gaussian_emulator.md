# GaussianLineEmulator Benchmark

This benchmark tests the performance of spectrum emulators (`GaussianLineEmulator` or `TransformerPayne`) across various mesh resolutions and configuration settings.

## Overview

The benchmark measures performance for four key operations:

1. **Mesh Creation + Spectrum Simulation**: Complete workflow from scratch
2. **Spectrum Simulation Only**: Isolates spectrum computation performance
3. **Rotation + Spectrum**: Tests rotation transform and spectrum computation
4. **Pulsation + Spectrum**: Tests pulsation transform and spectrum computation

## Default Configuration

- **Emulator**: GaussianLineEmulator with single Gaussian line at 5500Å (width=0.3Å, depth=0.5)
- **Mesh Resolutions**: 300, 1000, 5000, 20000 vertices
- **Wavelength Range**: 5496-5504 Å (2000 points for GaussianLineEmulator, 3000-10000 Å for TransformerPayne)
- **Timing Runs**: 5 runs per configuration
- **Rotation**: 50 km/s around z-axis
- **Pulsation**: m=4, l=4, period=1.0, 10 time samples

## Running the Benchmark

### Basic Usage

```bash
cd benchmarks
JAX_PLATFORMS=cpu python benchmark_gaussian_emulator.py
```

**Note**: Use `JAX_PLATFORMS=cpu` to avoid potential GPU/METAL backend issues on some systems.

### Command-Line Options

#### View All Options and Help

```bash
python benchmark_gaussian_emulator.py --help
```

#### 1. Using Different Emulators

**GaussianLineEmulator (default)**:
```bash
python benchmark_gaussian_emulator.py
```

**TransformerPayne**:
```bash
python benchmark_gaussian_emulator.py --use-transformer-payne
```

#### 2. Changing Number of Wavelengths

**With GaussianLineEmulator** (wavelength range: 5496-5504 Å):
```bash
# Test with 500 wavelength points
python benchmark_gaussian_emulator.py --n-wavelengths 500

# Test with 5000 wavelength points
python benchmark_gaussian_emulator.py --n-wavelengths 5000
```

**With TransformerPayne** (wavelength range: 3000-10000 Å):
```bash
# Test with 1000 wavelength points
python benchmark_gaussian_emulator.py --use-transformer-payne --n-wavelengths 1000

# Test with 3000 wavelength points
python benchmark_gaussian_emulator.py --use-transformer-payne --n-wavelengths 3000
```

#### 3. Custom Mesh Resolutions

```bash
# Test with specific resolutions
python benchmark_gaussian_emulator.py --mesh-resolutions 500 2000 10000

# Test with smaller meshes for quick testing
python benchmark_gaussian_emulator.py --mesh-resolutions 100 300 1000

# Test with only one resolution
python benchmark_gaussian_emulator.py --mesh-resolutions 1000
```

#### 4. Changing Number of Timing Runs

```bash
# Quick test with 3 runs per configuration
python benchmark_gaussian_emulator.py --n-runs 3

# More accurate statistics with 10 runs
python benchmark_gaussian_emulator.py --n-runs 10
```

#### 5. Custom Output File Names

```bash
# Specify custom output prefix
python benchmark_gaussian_emulator.py --output-prefix my_test

# This creates: my_test_benchmark_results.csv and my_test_performance.png
```

### Combined Examples

**Quick test with small meshes**:
```bash
python benchmark_gaussian_emulator.py \
    --mesh-resolutions 100 500 1000 \
    --n-runs 3 \
    --n-wavelengths 500
```

**Compare TransformerPayne with different wavelength counts**:
```bash
# Run 1: 500 wavelengths
python benchmark_gaussian_emulator.py \
    --use-transformer-payne \
    --n-wavelengths 500 \
    --output-prefix tpayne_500wl

# Run 2: 2000 wavelengths
python benchmark_gaussian_emulator.py \
    --use-transformer-payne \
    --n-wavelengths 2000 \
    --output-prefix tpayne_2000wl
```

**High-precision benchmark for publication**:
```bash
python benchmark_gaussian_emulator.py \
    --mesh-resolutions 300 1000 5000 20000 \
    --n-runs 10 \
    --n-wavelengths 2000 \
    --output-prefix publication_results
```

## Output Files

The benchmark generates two files (prefix depends on `--output-prefix` or emulator type):

- **CSV file**: `<prefix>_benchmark_results.csv` - Raw timing data with columns:
  - `mesh_resolution`: Number of vertices in the mesh
  - `operation`: Type of operation (mesh_creation_spectrum, spectrum_only, etc.)
  - `mean_time_sec`: Mean execution time in seconds
  - `std_time_sec`: Standard deviation of execution time

- **Plot file**: `<prefix>_performance.png` - 4-panel visualization showing time vs mesh resolution for each operation

### Default Output Names

- GaussianLineEmulator: `gaussian_emulator_benchmark_results.csv` and `gaussian_emulator_performance.png`
- TransformerPayne: `tpayne_emulator_benchmark_results.csv` and `tpayne_emulator_performance.png`

## Performance Characteristics

### GaussianLineEmulator (Typical Results)

- **Mesh Creation + Spectrum**: Scales from ~0.09s (300 vertices) to ~1.5s (20000 vertices)
- **Spectrum Only**: Extremely fast after JIT compilation (< 0.0001s)
- **Rotation + Spectrum**: Very fast after JIT compilation (< 0.0001s)
- **Pulsation + Spectrum**: Most intensive operation, ~1.6s to ~30s for 10 time samples

### TransformerPayne

TransformerPayne is generally slower than GaussianLineEmulator due to its more complex neural network architecture, but provides more realistic stellar spectra. Performance scales similarly with mesh resolution.

## Interpreting Results

1. **Mesh Creation + Spectrum**: Shows total time including mesh construction overhead
2. **Spectrum Only**: Pure computation time after JIT compilation and mesh creation
3. **Rotation + Spectrum**: Additional cost of rotation transform
4. **Pulsation + Spectrum**: Time for complete pulsation evaluation across multiple phases

The difference between "Mesh Creation + Spectrum" and "Spectrum Only" shows the mesh construction overhead. After JIT compilation, subsequent spectrum evaluations are much faster.

## Module Location

The `GaussianLineEmulator` class is available at:
```python
from spice.spectrum import GaussianLineEmulator
# or
from spice.spectrum.gaussian_line_emulator import GaussianLineEmulator
```

This provides a TransformerPayne-free alternative for testing and benchmarking SPICE mesh models.

## Troubleshooting

### JAX Backend Issues

If you encounter JAX backend errors (especially with METAL on macOS), force CPU backend:
```bash
JAX_PLATFORMS=cpu python benchmark_gaussian_emulator.py
```

### TransformerPayne Not Available

If you see "TransformerPayne is not available", install it:
```bash
pip install transformer-payne
```

### Out of Memory

For very large mesh resolutions (> 20000) or many wavelengths (> 5000), you may encounter memory issues. Try:
- Reducing mesh resolutions
- Reducing number of wavelengths
- Using CPU instead of GPU (`JAX_PLATFORMS=cpu`)

