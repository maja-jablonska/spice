# SPICE tutorials

Example notebooks for [SPICE](https://arxiv.org/abs/2511.10998) (`stellar-spice`),
the JAX library for synthesizing spectra of stars with inhomogeneous surfaces.

Most notebooks make the in-tree `spice` package importable on their own (they add
`src/` to `sys.path`), so an editable install (`pip install -e ".[dev]"`) is
recommended but not strictly required.

## Getting started

| Notebook | What it shows |
|---|---|
| [`binary.ipynb`](binary.ipynb) | Binary-star systems — Keplerian orbits and eclipses |
| [`planetary_transit_tutorial.ipynb`](planetary_transit_tutorial.ipynb) | Planetary transit light curves |
| [`phoebe.ipynb`](phoebe.ipynb) | Using PHOEBE models with SPICE |
| [`custom_filter.ipynb`](custom_filter.ipynb) | Defining a custom photometric filter / passband |
| [`synphot.ipynb`](synphot.ipynb) | Synthetic photometry via `synphot` |
| [`solar_oscillations_physical_line.ipynb`](solar_oscillations_physical_line.ipynb) | Solar-like oscillations with the Physical Line Emulator |

## Pulsation & visualization — [`pulsation/`](pulsation/)

| Notebook | What it shows |
|---|---|
| [`plot_vsh_basis_vectors.ipynb`](pulsation/plot_vsh_basis_vectors.ipynb) | Vector spherical harmonic (VSH) basis vectors on a mesh |
| [`pulsation_plot_utils_demo.ipynb`](pulsation/pulsation_plot_utils_demo.ipynb) | Pulsation plotting utilities (`spice.plots`) |
| [`pulsation_horizontal_distortion_3d.ipynb`](pulsation/pulsation_horizontal_distortion_3d.ipynb) | 3D animation of horizontal-mode distortion (produces the `.gif`) |

## Documentation examples — [`docs_examples/`](docs_examples/)

Notebooks that generate figures and grid examples for the documentation
(`docs_imgs.ipynb`, `korg_interpolator_parameters.ipynb`).

## Paper reproduction — [`paper_results/`](paper_results/)

Code and notebooks reproducing the figures/analyses in the SPICE paper, organized
by topic: `cepheid/`, `tz_fornacis/`, `phoebe_spice_comparison/`, `spots/`,
`paper_plots/`, `phoebe_passbands/`, and `technical_checks/`.

> Interpolator benchmarks/diagnostics that used to live here now sit in the
> repository's top-level [`benchmarks/`](../benchmarks/) directory
> (`lazy_zarr_*`, `run_lazy_zarr_decimation_sweep.pbs`).
