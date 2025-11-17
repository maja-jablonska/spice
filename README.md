![spice logo](https://raw.githubusercontent.com/maja-jablonska/spice/main/docs/img/spice_pink.svg)

# SPICE: SPectral Integration Compiled Engine

A comprehensive Python library for modeling and analyzing stellar spectra with inhomogeneous surfaces, supporting rotation, pulsations, spots, and binary star systems.

The paper is not submitted and available as a [preprint](https://arxiv.org/abs/2511.10998).

## Installation

Install from PyPI:

```bash
pip install stellar-spice
```

For PHOEBE integration support:

```bash
pip install stellar-spice[phoebe]
```


## Documentation

📖 Read the full [documentation](https://spice.readthedocs.io) for detailed API reference and tutorials.

## Key Features

### 🌟 **Stellar Surface Modeling**

- **Mesh-based stellar surfaces** using icosphere discretization
- **Inhomogeneous temperature distributions** across stellar surfaces
- **Surface gravity variations** accounting for rotation and shape distortions
- **Line-of-sight velocity calculations** for Doppler shift effects

### 🔄 **Stellar Rotation**

- **Differential rotation** modeling with customizable rotation laws
- **Rotational broadening** effects on spectral lines
- **Surface velocity field** calculations
- **Time-dependent spectral variations** due to rotation

### 🌊 **Stellar Pulsations**

- **Spherical harmonic pulsation modes** (l, m modes)
- **Fourier series parameterization** for complex pulsation patterns
- **Multi-mode pulsations** with different periods and amplitudes
- **Surface displacement and velocity** calculations

### 🌑 **Stellar Spots**

- **Spherical harmonic spot modeling** for complex spot distributions
- **Temperature contrast** between spots and photosphere
- **Time-evolving spot patterns**
- **Spot-induced spectral variations**

### ⭐ **Binary Star Systems**

- **Full orbital dynamics** with Keplerian orbits
- **Mutual eclipses** and occultations
- **Roche lobe geometry** for close binaries
- **Tidal distortion** effects
- **PHOEBE integration** for advanced binary modeling

### 📊 **Spectral Synthesis**

- **Blackbody radiation** for basic stellar modeling
- **ATLAS9 model atmospheres** for realistic stellar spectra
- **Transformer-Payne** integration for ML-based spectral synthesis
- **Custom spectral models** support

### 🔍 **Synthetic Photometry**

- **Multiple passband support** (Johnson, Stromgren, Gaia, etc.)
- **AB magnitude system** calculations
- **Bolometric luminosity** computations
- **Time-series photometry** for variable stars

### 🎯 **Advanced Features**

- **JAX-based computations** for fast, differentiable calculations
- **GPU acceleration** support
- **3D visualization** of stellar surfaces and binary systems
- **Animation capabilities** for time-evolving systems
- **Occlusion handling** for complex geometries

## Quick Start

### Basic Stellar Model

```python
import numpy as np
from spice.models import IcosphereModel
from spice.spectrum import simulate_observed_flux
from spice.spectrum.planck_law import Blackbody

# Create a solar-like star
star = IcosphereModel.construct(
    subdivisions=500,  # Mesh resolution
    radius=1.0,        # Solar radii
    mass=1.0,          # Solar masses
    parameters=Blackbody().solar_parameters,
    parameter_names=Blackbody().parameter_names
)

# Add rotation
star = star.add_rotation(period=25.0)  # 25-day rotation period

# Generate spectrum
wavelengths = np.logspace(3, 4, 1000)  # 1000-10000 Å
spectrum = simulate_observed_flux(Blackbody().intensity, star, wavelengths)
```

### Binary Star System

```python
from spice.models import Binary, add_orbit
from spice.spectrum.filter import GaiaG

# Create binary components
primary = IcosphereModel.construct(500, 1.0, 1.0, bb.solar_parameters, bb.parameter_names)
secondary = IcosphereModel.construct(500, 0.8, 0.8, bb.solar_parameters, bb.parameter_names)

# Create binary system
binary = Binary.from_bodies(primary, secondary)

# Add orbital parameters
binary = add_orbit(
    binary,
    P=1.0,      # 1-year period
    ecc=0.1,    # 10% eccentricity
    i=np.pi/3,  # 60° inclination
    # ... other orbital elements
)

# Calculate light curve
times = np.linspace(0, 1, 100)
light_curve = []
for t in times:
    p1, p2 = evaluate_orbit_at_times(binary, t)
    flux = simulate_observed_flux(bb.intensity, p1, wavelengths) + \
           simulate_observed_flux(bb.intensity, p2, wavelengths)
    light_curve.append(AB_passband_luminosity(GaiaG(), wavelengths, flux))
```

### PHOEBE Integration

```python
import phoebe
from spice.models import PhoebeBinary

# Create PHOEBE binary
b = phoebe.default_binary()
# ... set up PHOEBE parameters

# Convert to SPICE format
pb = PhoebeBinary.construct(b, parameter_names, parameter_values)

# Use SPICE for spectral calculations
spectrum = simulate_observed_flux(intensity_function, pb, wavelengths)
```

## Performance

- **JAX-powered computations** for fast, vectorized operations
- **GPU acceleration** support for large-scale calculations
- **Efficient mesh operations** with optimized occlusion algorithms
- **Memory-efficient** spectral synthesis for time-series data

## Citation

If you use stellar-spice in your research, please cite:

```bibtex
@misc{spice,
      title={SPICE -- modelling synthetic spectra of stars with non-homogeneous surfaces}, 
      author={M. Jabłońska, T. Różański, L. Casagrande, H. Shah, P. A. Kołaczek-Szymański, M. Rychlicki and Yuan-Sen Ting},
      year={2025},
      eprint={2511.10998},
      archivePrefix={arXiv},
      primaryClass={astro-ph.SR},
      url={https://arxiv.org/abs/2511.10998}, 
}
```

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [JAX](https://github.com/google/jax) for fast, differentiable computations
- Integrates with [PHOEBE](https://phoebe-project.org/) for binary star modeling
- Uses [Transformer-Payne](https://github.com/tingyuansen/transformer-payne) for ML-based spectral synthesis

---

**Paper currently in preparation** - Check back for the full scientific publication!
