![spice logo](https://raw.githubusercontent.com/maja-jablonska/spice/main/docs/img/spice_pink.svg)

# SPICE: SPectral Integration Compiled Engine

A comprehensive Python library for modeling and analyzing stellar spectra with inhomogeneous surfaces, supporting rotation, pulsations, spots, and binary star systems.

The paper is submitted and available as a [preprint](https://arxiv.org/abs/2511.10998).

## Installation

Install from PyPI:

```bash
pip install stellar-spice
```

For PHOEBE integration support:

```bash
pip install stellar-spice[phoebe]
```

For zarr-backed model-atmosphere grid interpolation (`LazyZarrInterpolator` and friends):

```bash
pip install stellar-spice[grid]
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

- **Solid-body (rigid) rotation** with a configurable axis and equatorial velocity
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
- **Mutual eclipses** and occultations resolved on the projected meshes
- **PHOEBE integration** — import Roche-lobe geometry and tidally distorted binary meshes computed by PHOEBE

### 📊 **Spectral Synthesis**

- **Blackbody radiation** for basic stellar modeling
- **Model-atmosphere grid interpolation** from precomputed, zarr-backed grids
- **Analytic line-profile emulators** (Gaussian and physical)
- **Neural-network emulator (aemu) bundles** for ML-based spectral synthesis
- **Custom spectral models** via the `SpectrumEmulator` interface

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
from spice.models.mesh_transform import add_rotation
from spice.spectrum import simulate_observed_flux, Blackbody

bb = Blackbody()

# Create a solar-like star
star = IcosphereModel.construct(
    n_vertices=1000,                 # Mesh resolution (number of vertices)
    radius=1.0,                      # Solar radii
    mass=1.0,                        # Solar masses
    parameters=bb.solar_parameters,
    parameter_names=bb.parameter_names,
)

# Add solid-body rotation (equatorial velocity in km/s)
star = add_rotation(star, rotation_velocity=2.0)

# Generate the disc-integrated spectrum. simulate_observed_flux expects *log10*
# wavelengths and returns an (n_wavelengths, 2) array whose columns are the
# disc-integrated emulator channels (flux and continuum; column 0 is the flux).
wavelengths = np.logspace(3, 4, 1000)            # 1000-10000 Å
flux = simulate_observed_flux(bb.intensity, star, np.log10(wavelengths))
```

### Binary Star System

```python
import numpy as np
import jax.numpy as jnp
from spice.models import IcosphereModel, Binary
from spice.models.binary import add_orbit, evaluate_orbit_at_times
from spice.models.mesh_view import get_mesh_view
from spice.spectrum import simulate_observed_flux, Blackbody, AB_passband_luminosity
from spice.spectrum.filter import GaiaG

bb = Blackbody()
los = jnp.array([0.0, 1.0, 0.0])  # line of sight

# Create binary components (cast to the line of sight for occlusion handling)
primary = get_mesh_view(
    IcosphereModel.construct(1000, 1.0, 1.0, bb.solar_parameters, bb.parameter_names), los)
secondary = get_mesh_view(
    IcosphereModel.construct(1000, 0.8, 0.8, bb.solar_parameters, bb.parameter_names), los)

# Assemble the system
binary = Binary.from_bodies(primary, secondary)

# Add orbital elements
binary = add_orbit(
    binary,
    P=1.0,                    # orbital period [years]
    ecc=0.1,                  # eccentricity
    T=0.0,                    # time of periastron passage [years]
    i=np.pi / 3,              # inclination [rad]
    omega=0.0,                # argument of periastron [rad]
    Omega=0.0,                # longitude of the ascending node [rad]
    mean_anomaly=0.0,         # mean anomaly at the reference time [rad]
    reference_time=0.0,       # reference time [years]
    vgamma=0.0,               # systemic velocity [km/s]
    orbit_resolution_points=50,
)

# Evaluate the orbit across phases (eclipses/occlusions resolved internally)
times = jnp.linspace(0.0, 1.0, 100)
primaries, secondaries = evaluate_orbit_at_times(binary, times)

# Combined Gaia G-band light curve
wavelengths = np.linspace(900, 40000, 1000)
gaia_g = GaiaG()
light_curve = [
    AB_passband_luminosity(
        gaia_g,
        wavelengths,
        simulate_observed_flux(bb.intensity, p1, np.log10(wavelengths))[:, 0]
        + simulate_observed_flux(bb.intensity, p2, np.log10(wavelengths))[:, 0],
    )
    for p1, p2 in zip(primaries, secondaries)
]
```

### PHOEBE Integration

```python
import numpy as np
import phoebe
from phoebe.parameters.dataset import _mesh_columns
from spice.models import PhoebeBinary
from spice.models.binary import evaluate_orbit
from spice.models.phoebe_utils import PhoebeConfig
from spice.spectrum import simulate_observed_flux, Blackbody

bb = Blackbody()

# Build a PHOEBE binary and compute a mesh dataset (standard PHOEBE workflow)
b = phoebe.default_binary()
times = np.linspace(0, 1, 10)
b.add_dataset('mesh', compute_times=times, columns=_mesh_columns, dataset='mesh01')
b.run_compute(coordinates='uvw', overwrite=True)

# Wrap the PHOEBE meshes for SPICE (PHOEBE models are read-only inside SPICE)
config = PhoebeConfig(b, 'mesh01')
pb = PhoebeBinary.construct(config, ['teff', 'logg', 'abun'])

# Evaluate the components at a snapshot time, then synthesise the spectrum
primary, secondary = evaluate_orbit(pb, config.times[0])
wavelengths = np.logspace(3, 4, 1000)
flux = simulate_observed_flux(bb.intensity, primary, np.log10(wavelengths))
```

## Performance

- **JAX-powered computations** for fast, vectorized operations
- **GPU acceleration** support for large-scale calculations
- **Efficient mesh operations** with optimized occlusion algorithms
- **Memory-efficient** spectral synthesis for time-series data

## Known internals caveat

### Mesh area conventions (slated for cleanup)

`MeshModel` currently exposes two area arrays that **follow different unit
conventions**:

| Property | Convention | Sum over the mesh |
|---|---|---|
| `m.areas` / `m.base_areas` | unit-sphere normalised | ≈ 4π for any R |
| `m.cast_areas` / `m.visible_cast_areas` | physical R⊙² | ≈ 4π R² / ≈ π R² |

Any code that integrates over the stellar surface must apply the matching
cgs prefactor:

- `simulate_observed_flux` integrates `m.visible_cast_areas` (already in
  R⊙²) and applies only the dimensionless dilution `(R⊙ / pc)² / d_pc²`.
- `simulate_monochromatic_luminosity` integrates `m.areas` (unit-sphere) and
  applies `R² × R⊙²[cm²]` to convert to cgs.

If you add a new integrator over either array, mirror the matching pattern.
This convention asymmetry is a known maintainability hazard and is slated
for future cleanup: once `m.areas` is normalised the same way as
`m.visible_cast_areas`, the cgs prefactor in
`simulate_monochromatic_luminosity` should be reduced to `R⊙²[cm²]` only,
matching the structure of `simulate_observed_flux`.

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
- Uses [astro-emulators-toolkit](https://pypi.org/project/astro-emulators-toolkit/) bundles for ML-based spectral synthesis

---

**A preprint describing SPICE is available on [arXiv](https://arxiv.org/abs/2511.10998).** See [Citation](#citation) if you use it in your work.
