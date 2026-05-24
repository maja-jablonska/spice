# Contributing to SPICE

Thanks for your interest in improving SPICE (`stellar-spice`)! This document
describes how to set up a development environment and the conventions the
project follows.

## Development setup

```bash
# Clone your fork, then from the repository root:
pip install -e ".[dev]"

# To work on the optional integrations as well:
pip install -e ".[dev,phoebe,huggingface,aemu]"
```

SPICE uses a `src/` layout. The test configuration in `pyproject.toml`
(`[tool.pytest.ini_options] pythonpath = ["src"]`) puts the package on the path
for the test run, so an editable install is the most reliable way to develop.

## Running the tests

```bash
pytest                                   # full suite
pytest tests/test_lazy_zarr_interpolator.py        # one file
pytest tests/test_models/test_binary.py::test_name # one test
```

The suite discovers both `test_*.py` files and pytest-BDD-style
`Test*`/`Describe*` classes with `test_`/`it_`/`and_`/`but_`/`they_` methods.

## Conventions

- **No mutation of models.** `MeshModel` is a JAX pytree; update state
  functionally via `_replace` / the helpers in `models/mesh_transform.py`.
- **Lazy imports.** Public symbols are registered in each subpackage's
  `__init__.py` `_lazy_imports` map (or exposed via `__getattr__`); avoid adding
  eager `from .foo import Bar` lines for heavy modules.
- **Optional dependencies** (`phoebe`, `astro_emulators_toolkit`, `h5py`-backed
  paths) must be guarded with `try/except ImportError` and remain importable
  when the extra is absent.
- **Dtype.** Honor the `jax_enable_x64` flag via the shared `_float_dtype`
  helper rather than hardcoding `float64`.
- There is no configured linter/formatter; match the surrounding style.

## Submitting changes

1. Create a feature branch off `main`.
2. Make your change with tests covering it.
3. Ensure `pytest` passes locally.
4. Open a pull request describing the change and the motivation.

## License

By contributing, you agree that your contributions will be licensed under the
[MIT License](LICENSE).
