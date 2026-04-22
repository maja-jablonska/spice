# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

SPICE (`stellar-spice` on PyPI) is a JAX-based library for synthesizing spectra of stars with inhomogeneous surfaces — rotation, pulsations, spots, and binary systems. The package lives under `src/spice/` and installs as `spice`. Paper preprint: https://arxiv.org/abs/2511.10998.

## Common commands

```bash
# Editable install for development (use ".[dev,phoebe,huggingface]" for all extras)
pip install -e ".[dev]"

# Run the full test suite (uses src-layout via pyproject pythonpath setting)
pytest

# A single file or a single test
pytest tests/test_lazy_zarr_interpolator.py
pytest tests/test_models/test_binary.py::test_name

# Build a wheel/sdist (release flow is in .github/workflows/release.yml)
python -m build
```

There is no configured linter/formatter; don't introduce one without asking. Tests discover both `test_*.py` files and pytest-BDD-style `Test*`/`Describe*` classes with `test_`/`it_`/`and_`/`but_`/`they_` methods (see `[tool.pytest.ini_options]`).

## Architecture

### Layered subpackages (all lazy-imported)

Every top-level subpackage (`spice`, `spice.models`, `spice.spectrum`) uses a module-level `__getattr__` to defer heavy imports (JAX, flax, phoebe) until the symbol is actually accessed. When adding a new public symbol, register it in the parent package's `__init__.py` `_lazy_imports` map — don't add eager `from .foo import Bar` lines.

- `spice.models` — stellar surface models. Central type is `MeshModel` (`models/mesh_model.py`), a `NamedTuple` subclass registered as a JAX pytree so mesh operations can be jit/vmap'd. State is updated **functionally** via `_replace` / the helpers in `mesh_transform.py`; never mutate a model in place. `Binary` (`models/binary.py`) composes two `Model`s with Keplerian orbit state plus occlusion neighbor indices.
- `spice.spectrum` — spectral synthesis. `SpectrumEmulator` is the abstract interface (see `spectrum_emulator.py`); concrete implementations include `Blackbody` (`planck_law.py`), `GaussianLineEmulator`, `PhysicalLineEmulator`, and the zarr-backed grid interpolators in `lazy_zarr_interpolator.py` (`LazyZarrInterpolator`, `IntensityLazyZarrInterpolator`, `FluxLazyZarrInterpolator`, with `GridIndex` / `SparseGridIndex` for axis lookup). The main entry point is `simulate_observed_flux(intensity_fn, mesh_model, wavelengths)` in `spectrum.py`, which chunks over surface elements with `jax.checkpoint`. Filter/passband code lives in `filter.py`, with data in `spectrum/filter_data/` (packaged via `tool.setuptools.package-data`).
- `spice.geometry` — Sutherland–Hodgman polygon clipping and cast-area utilities used for occlusion/projection.
- `spice.plots` — matplotlib-based helpers for meshes and pulsation modes.
- `spice.utils` — Fourier fitting (used to parameterize pulsation time dependence), logging, parameter handling, warnings.

### JAX conventions

- `spice/__init__.py` forces `JAX_PLATFORMS=cpu` on macOS unless the user sets it explicitly — the Metal backend is slow/broken for this workload. Don't remove this unless you're sure.
- Tests opt into 64-bit precision via `jax_config.update("jax_enable_x64", True)` in `tests/conftest.py`. Model code reads `_float_dtype()` helpers to stay consistent with the x64 flag.
- `conftest.py` also monkey-patches `np.asarray` (for old NumPy `copy=` signature) and `jax_core.ShapedArray.update` (for pickle compatibility with newer JAX metadata). These are load-bearing for the test fixtures in `tests/test_models/data/` — leave them alone unless you're removing the pickles too.

### Optional PHOEBE integration

`phoebe` is an optional extra. `spice.models.phoebe_model`, `spice.models.phoebe_utils`, and the `PhoebeBinary` / `PhoebeModel` entry points are guarded by `try/except ImportError`; `models/binary.py` and `mesh_transform.py` fall back to placeholder classes. Any new PHOEBE-dependent code must follow this pattern and be covered by the guard in `tests/test_phoebe_optional.py`. A lazy anim-to-html monkey-patch (`_patch_phoebe_anim_to_html`) exists in `spice/__init__.py` but is not called automatically.

### Spectral grid data

Large atmosphere grids are stored externally as zarr; the `LazyZarrInterpolator` family and `generate_zarr_index.py` build and consume sparse grid indices. Tests for this path (`test_lazy_zarr_*`, `test_zarr_grid_*`, `test_sparse_grid_index.py`, `test_generate_zarr_index*.py`) are the authoritative reference for expected behavior — read them before changing interpolator code.
