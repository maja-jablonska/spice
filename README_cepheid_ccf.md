# Cepheid CCF Analysis Script

This repository contains a Python command-line script for calculating selected lines and Cross-Correlation Function (CCF) profiles for Cepheid variable stars with configurable mesh resolution.

## Overview

The script simulates a Cepheid variable star using SPICE models and calculates CCF profiles for selected spectral lines at different phases of the pulsation cycle. It's based on the analysis from the `cepheid.ipynb` notebook and provides a standalone command-line interface.

## Features

- **Configurable mesh resolution**: Specify the number of mesh elements for the stellar surface
- **Multiple spectral lines**: Analyze Fe I and Si II lines (customizable)
- **CCF calculation**: Cross-correlation function analysis for radial velocity determination
- **Phase sampling**: Simulate spectra at different pulsation phases
- **Output options**: Save results as pickle files and generate plots
- **Standalone operation**: Works without TransformerPayne dependencies (uses simple blackbody model)

## Requirements

- Python 3.8+
- JAX (with CPU backend)
- SPICE library
- NumPy, SciPy, Matplotlib
- tqdm (for progress bars)

## Installation

1. Install the SPICE library in development mode:
```bash
cd /path/to/stellar-mesh-integration
pip install -e .
```

2. Install additional dependencies:
```bash
pip install jaxkd
```

## Usage

### Basic Usage

```bash
python cepheid_ccf_analysis_standalone.py --mesh-resolution 1000 --output-dir results/
```

### Advanced Usage

```bash
python cepheid_ccf_analysis_standalone.py \
    --mesh-resolution 5000 \
    --lines "4896.439,5049.82,5044.21" \
    --phases 50 \
    --velocity-range -100 100 \
    --output-dir results/
```

### Command Line Arguments

- `--mesh-resolution` (required): Number of mesh elements for the icosphere
- `--output-dir`: Output directory for results (default: results)
- `--lines`: Comma-separated list of line centers in air (default: Fe I and Si II lines)
- `--line-width`: Width around line centers in Angstroms (default: 2.0)
- `--steps`: Number of wavelength steps (default: 1000)
- `--phases`: Number of phases to simulate (default: 100)
- `--velocity-range`: Velocity range for CCF calculation in km/s (default: -50 50)
- `--no-plots`: Skip creating plots

### Examples

1. **Quick test with low resolution:**
```bash
python cepheid_ccf_analysis_standalone.py --mesh-resolution 100 --phases 10 --no-plots
```

2. **High-resolution analysis:**
```bash
python cepheid_ccf_analysis_standalone.py --mesh-resolution 10000 --phases 100 --output-dir high_res_results
```

3. **Custom line selection:**
```bash
python cepheid_ccf_analysis_standalone.py --mesh-resolution 2000 --lines "4896.439,5049.82" --phases 50
```

## Output Files

The script generates the following output files:

1. **Data file**: `cepheid_ccf_mesh_{resolution}.pkl`
   - Contains radial velocities, CCF profiles, wavelengths, phases, and metadata
   - Can be loaded with `pickle.load()` for further analysis

2. **Plots** (unless `--no-plots` is specified):
   - `radial_velocity_curves_mesh_{resolution}.png`: Radial velocity vs phase for each line
   - `ccf_profiles_mesh_{resolution}.png`: CCF profiles for different phases

## Stellar Parameters

The script uses default parameters for Delta Cephei:
- Effective temperature: 6562 K
- Surface gravity: 1.883 dex
- Radius: 41.18 R☉
- Mass: 5.26 M☉
- Metallicity: +0.08 dex
- Pulsation period: 5.366 days

## Technical Details

### CCF Calculation

The script implements two CCF calculation methods:
1. **Velocity grid method**: Tests a range of velocity shifts and finds the maximum correlation
2. **Log-wavelength method**: Uses logarithmic wavelength interpolation for more accurate results

### Spectrum Model

The script uses a simple blackbody model for spectrum generation, which is sufficient for demonstrating the CCF analysis methodology. For more realistic results, the full SPICE library with TransformerPayne can be used.

### Performance

- **Mesh resolution 100**: ~5 seconds
- **Mesh resolution 1000**: ~15 seconds  
- **Mesh resolution 5000**: ~2-3 minutes
- **Mesh resolution 10000**: ~5-10 minutes

Times are approximate and depend on the number of phases and lines analyzed.

## Troubleshooting

### JAX Backend Issues

If you encounter JAX backend errors, set the platform to CPU:
```bash
export JAX_PLATFORMS=cpu
python cepheid_ccf_analysis_standalone.py --mesh-resolution 1000
```

### Memory Issues

For high mesh resolutions, you may need to reduce the number of phases or lines:
```bash
python cepheid_ccf_analysis_standalone.py --mesh-resolution 10000 --phases 20 --lines "4896.439"
```

### Import Errors

Make sure the SPICE library is properly installed and the Python path is set:
```bash
export PYTHONPATH=/path/to/stellar-mesh-integration/src:$PYTHONPATH
```

## Citation

If you use this script in your research, please cite the original SPICE paper and the cepheid analysis notebook.

## License

This script is part of the stellar-mesh-integration project and follows the same license terms.
