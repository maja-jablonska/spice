# SPICE examples

Notebooks and scripts for [SPICE](https://arxiv.org/abs/2511.10998) (`stellar-spice`),
the JAX library for synthesizing spectra of stars with inhomogeneous surfaces. Everything
here is used either by the documentation or by the paper.

Most notebooks make the in-tree `spice` package importable on their own (they add
`src/` to `sys.path`), so an editable install (`pip install -e ".[dev]"`) is
recommended but not strictly required.

## Documentation examples — [`docs_examples/`](docs_examples/)

Companion notebooks for the documentation pages: one notebook per page, one
section per code snippet / figure, so every figure in the docs can be
reproduced by running the matching notebook end-to-end.

| Notebook | Documentation page | Requires |
| --- | --- | --- |
| [`mesh_examples.ipynb`](docs_examples/mesh_examples.ipynb) | Creating a Mesh Model | core install |
| [`spectral_synthesis_examples.ipynb`](docs_examples/spectral_synthesis_examples.ipynb) | Spectral Synthesis | core install |
| [`spectral_grids_examples.ipynb`](docs_examples/spectral_grids_examples.ipynb) | Spectral Grids | `[grid]` extra |
| [`custom_emulators_examples.ipynb`](docs_examples/custom_emulators_examples.ipynb) | Custom Emulators | core install |
| [`synthetic_photometry_examples.ipynb`](docs_examples/synthetic_photometry_examples.ipynb) | Synthetic Photometry | core install |
| [`aemu_examples.ipynb`](docs_examples/aemu_examples.ipynb) | Machine-Learning Emulators (aemu) | `[aemu]` extra |
| [`binaries_examples.ipynb`](docs_examples/binaries_examples.ipynb) | Binaries | `[phoebe]` extra |
| [`phoebe_integration_examples.ipynb`](docs_examples/phoebe_integration_examples.ipynb) | PHOEBE Integration | `[phoebe]` extra |

Figure sections write both light and dark variants directly into `docs/img/`
with the filenames the docs reference.

## Paper reproduction — [`paper_results/`](paper_results/)

Code that produces the figures, tables and numbers of the SPICE paper, by section:

| Directory | Paper content |
|---|---|
| `paper_plots/` | Method figures: rotation velocity field, spot and spherical-harmonic configurations (with their parameter tables), pulsation snapshots, spot smoothness, blackbody pulsation test |
| `spots/` | Sect. 3.1, the Kochukhov (2017) spotted-star test (notebook, Turbospectrum config, grid README) |
| `cepheid/` | Sect. 3.2, delta Cephei: literature-model figures, Mg b projection factor and its errors (`dcep_lit_pfactor_errors.py`), synthetic p-factor pipeline (`dcep_ts_*`, `pfactor_v11/`) |
| `tz_fornacis/` | Sect. 3.3, TZ Fornacis: figures notebook, photometry scoring, PHOEBE mesh export, SuppNet normalisation of the HARPS spectra |
| `phoebe_spice_comparison/` | Sect. 3.4 and Appendix C, SPICE vs PHOEBE |
| `technical_checks/` | Appendices A and D: icosphere properties, solar luminosity/constant, single-spectrum test |
| `emulator_validation/` | Appendix F: emulator vs Turbospectrum over H-beta (Figs. F1-F2) |
| `mesh_resolution_effects.ipynb` | Appendix E: line-profile effects of mesh resolution |

The data these scripts read are in the Hugging Face dataset
[mjablonska/spice-paper-data](https://huggingface.co/datasets/mjablonska/spice-paper-data);
`python tutorial/paper_results/fetch_paper_data.py` downloads them. The paper's timing
benchmarks are in the top-level [`benchmarks/`](../benchmarks/) directory.
