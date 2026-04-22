import importlib.util
import os
import sys
import types
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest


@lru_cache(maxsize=1)
def _load_lazy_interpolator_module():
    root = Path(__file__).resolve().parents[1]
    emulator_path = root / "src" / "spice" / "spectrum" / "spectrum_emulator.py"
    loader_path = root / "src" / "spice" / "spectrum" / "lazy_zarr_grid_loader.py"
    interpolator_path = root / "src" / "spice" / "spectrum" / "lazy_zarr_grid_interpolator.py"

    target_module_names = [
        "spice",
        "spice.spectrum",
        "spice.spectrum.spectrum_emulator",
        "spice.spectrum.lazy_zarr_grid_loader",
        "spice.spectrum.lazy_zarr_grid_interpolator",
    ]
    missing = object()
    saved_modules = {name: sys.modules.get(name, missing) for name in target_module_names}

    try:
        spice_pkg = types.ModuleType("spice")
        spice_pkg.__path__ = []
        spectrum_pkg = types.ModuleType("spice.spectrum")
        spectrum_pkg.__path__ = []
        spice_pkg.spectrum = spectrum_pkg

        sys.modules["spice"] = spice_pkg
        sys.modules["spice.spectrum"] = spectrum_pkg

        emulator_spec = importlib.util.spec_from_file_location(
            "spice.spectrum.spectrum_emulator", emulator_path
        )
        if emulator_spec is None or emulator_spec.loader is None:
            raise RuntimeError(f"Could not load module spec from {emulator_path}")
        emulator_module = importlib.util.module_from_spec(emulator_spec)
        sys.modules["spice.spectrum.spectrum_emulator"] = emulator_module
        emulator_spec.loader.exec_module(emulator_module)

        loader_spec = importlib.util.spec_from_file_location(
            "spice.spectrum.lazy_zarr_grid_loader", loader_path
        )
        if loader_spec is None or loader_spec.loader is None:
            raise RuntimeError(f"Could not load module spec from {loader_path}")
        loader_module = importlib.util.module_from_spec(loader_spec)
        sys.modules["spice.spectrum.lazy_zarr_grid_loader"] = loader_module
        loader_spec.loader.exec_module(loader_module)

        interpolator_spec = importlib.util.spec_from_file_location(
            "spice.spectrum.lazy_zarr_grid_interpolator", interpolator_path
        )
        if interpolator_spec is None or interpolator_spec.loader is None:
            raise RuntimeError(f"Could not load module spec from {interpolator_path}")
        interpolator_module = importlib.util.module_from_spec(interpolator_spec)
        sys.modules["spice.spectrum.lazy_zarr_grid_interpolator"] = interpolator_module
        interpolator_spec.loader.exec_module(interpolator_module)
        return interpolator_module
    finally:
        for name in target_module_names:
            old_module = saved_modules[name]
            if old_module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module

def _get_lazy_interpolator_class():
    return _load_lazy_interpolator_module().LazyZarrGridInterpolator


def _write_test_store(store_path: Path, include_mu_selected: bool = False):
    import zarr

    group = zarr.open_group(str(store_path), mode="w")

    param_names = np.array([b"teff", b"logg", b"fixed"], dtype="|S16")
    group.create_array("param_names", data=param_names)

    wavelengths = np.array([5000.0, 5001.0, 5002.0, 5003.0], dtype=np.float32)
    group.create_array("wavelength", data=wavelengths)

    teff_values = np.array([5000.0, 6000.0], dtype=np.float32)
    logg_values = np.array([4.0, 4.5, 5.0], dtype=np.float32)
    params = np.asarray(
        [[teff, logg, 1.23] for teff in teff_values for logg in logg_values],
        dtype=np.float32,
    )[::-1]
    group.create_array("params", data=params, chunks=(2, params.shape[1]))

    n_rows = params.shape[0]
    n_wave = wavelengths.shape[0]
    flux = np.empty((n_rows, n_wave), dtype=np.float32)
    continuum = np.empty((n_rows, n_wave), dtype=np.float32)
    mu_selected = np.empty((n_rows,), dtype=np.float32)
    for i, (teff, logg, _) in enumerate(params):
        base = 1e-3 * teff + 10.0 * logg
        flux[i, :] = base + np.arange(n_wave, dtype=np.float32)
        continuum[i, :] = flux[i, :] + 100.0
        mu_selected[i] = 0.2 + 0.0001 * teff + 0.05 * logg

    group.create_array("flux", data=flux, chunks=(2, n_wave))
    group.create_array("continuum", data=continuum, chunks=(2, n_wave))
    if include_mu_selected:
        group.create_array("mu_selected", data=mu_selected, chunks=(2,))


class TestLazyZarrGridInterpolator:
    def test_init_uses_lazy_loader_and_sets_parameter_names(self, monkeypatch, caplog):
        lazy_module = _load_lazy_interpolator_module()
        LazyZarrGridInterpolator = lazy_module.LazyZarrGridInterpolator

        class DummyLoader:
            def __init__(
                self,
                store_path,
                chunk_rows=500,
                row_cache_bytes=None,
                prefetch_axis_chunk_size=None,
            ):
                self.store_path = store_path
                self.chunk_rows = chunk_rows
                self.row_cache_bytes = row_cache_bytes or 0
                self.prefetch_axis_chunk_size = prefetch_axis_chunk_size or 0
                self.axis_names = ["mu", "logg", "teff", "feh", "vmicro"]
                self.row_index_cube = np.zeros((2, 3, 4), dtype=np.int32)
                self.wavelength_jnp = np.linspace(5000.0, 5003.0, 4, dtype=np.float32)
                self.mu_selected_arr = None

        monkeypatch.setattr(lazy_module, "LazyZarrGridLoader", DummyLoader)

        with caplog.at_level("WARNING"):
            interpolator = LazyZarrGridInterpolator(
                "grid.zarr",
                chunk_rows=321,
                row_cache_bytes=1234,
                prefetch_axis_chunk_size=7,
            )

        assert interpolator.grid_loader.store_path == "grid.zarr"
        assert interpolator.grid_loader.chunk_rows == 321
        assert interpolator.grid_loader.row_cache_bytes == 1234
        assert interpolator.grid_loader.prefetch_axis_chunk_size == 7
        assert interpolator.parameter_names == ["mu", "logg", "teff", "feh", "vmicro"]
        assert "Opening lazy spectral grid" in caplog.text
        assert "row_cache=1.21 KiB" in caplog.text
        assert "prefetch_axis_chunk=7" in caplog.text
        assert "Lazy spectral grid ready in" in caplog.text

    def test_intensity_delegates_to_lazy_loader(self):
        LazyZarrGridInterpolator = _get_lazy_interpolator_class()

        class DummyLoader:
            axis_names = ["mu", "logg", "teff", "feh", "vmicro"]
            wavelength_jnp = np.array([100.0, 200.0, 300.0], dtype=np.float32)

            def __init__(self):
                self.last_query = None

            def interpolate_spectrum_and_continuum(self, query_params, validate=True):
                self.last_query = np.asarray(query_params)
                return (
                    np.array([10.0, 20.0, 30.0], dtype=np.float32),
                    np.array([1.0, 2.0, 3.0], dtype=np.float32),
                )

            def interpolate_mu_selected(self, query_params, validate=True):
                return None

        interpolator = LazyZarrGridInterpolator.__new__(LazyZarrGridInterpolator)
        interpolator.grid_loader = DummyLoader()
        interpolator.parameter_names = ["mu", "logg", "teff", "feh", "vmicro"]

        log_wavelengths = np.log10(np.array([100.0, 250.0], dtype=np.float32))
        result = interpolator.intensity(
            log_wavelengths, 0.4, np.array([4.4, 5777.0, 0.0, 0.0], dtype=np.float32)
        )

        np.testing.assert_allclose(
            interpolator.grid_loader.last_query,
            np.array([0.4, 4.4, 5777.0, 0.0, 0.0], dtype=np.float32),
        )
        np.testing.assert_allclose(result[:, 0], np.array([10.0, 25.0], dtype=np.float32), atol=1e-5)
        np.testing.assert_allclose(result[:, 1], np.array([1.0, 2.5], dtype=np.float32), atol=1e-5)

    def test_intensity_uses_lazy_mu_selected_fallback_when_mu_axis_missing(self):
        LazyZarrGridInterpolator = _get_lazy_interpolator_class()

        class DummyLoader:
            axis_names = ["logg", "teff", "feh", "vmicro"]
            wavelength_jnp = np.array([100.0, 200.0, 300.0], dtype=np.float32)

            def __init__(self):
                self.last_query = None
                self.last_mu_query = None

            def interpolate_spectrum_and_continuum(self, query_params, validate=True):
                self.last_query = np.asarray(query_params)
                return (
                    np.array([10.0, 20.0, 30.0], dtype=np.float32),
                    np.array([1.0, 2.0, 3.0], dtype=np.float32),
                )

            def interpolate_mu_selected(self, query_params, validate=True):
                self.last_mu_query = np.asarray(query_params)
                return np.array([0.5], dtype=np.float32)

        interpolator = LazyZarrGridInterpolator.__new__(LazyZarrGridInterpolator)
        interpolator.grid_loader = DummyLoader()
        interpolator.parameter_names = ["logg", "teff", "feh", "vmicro"]

        wavelengths = np.array([100.0, 250.0], dtype=np.float32)
        params = np.array([4.4, 5777.0, 0.0, 0.0], dtype=np.float32)

        result_mu_half = interpolator.intensity(wavelengths, 0.5, params)
        result_mu_one = interpolator.intensity(wavelengths, 1.0, params)

        np.testing.assert_allclose(
            interpolator.grid_loader.last_query,
            np.array([4.4, 5777.0, 0.0, 0.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            interpolator.grid_loader.last_mu_query,
            np.array([4.4, 5777.0, 0.0, 0.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            result_mu_one[:, 0] / result_mu_half[:, 0],
            np.array([1.0 / 0.7, 1.0 / 0.7], dtype=np.float32),
            atol=1e-5,
        )

    def test_to_parameters_none_returns_solar_parameters(self):
        LazyZarrGridInterpolator = _get_lazy_interpolator_class()
        interpolator = LazyZarrGridInterpolator.__new__(LazyZarrGridInterpolator)
        interpolator.parameter_names = ["logg", "teff", "feh", "vmicro"]
        np.testing.assert_allclose(
            np.asarray(interpolator.to_parameters(None)),
            np.asarray(interpolator.solar_parameters),
        )

    def test_flux_raises_not_implemented(self):
        LazyZarrGridInterpolator = _get_lazy_interpolator_class()
        interpolator = LazyZarrGridInterpolator.__new__(LazyZarrGridInterpolator)
        with pytest.raises(NotImplementedError):
            interpolator.flux(np.array([3.7, 3.8]), np.array([4.4, 5777.0, 0.0, 0.0]))

    def test_intensity_supports_jax_jit_and_vmap(self, tmp_path):
        import jax
        import jax.numpy as jnp

        LazyZarrGridInterpolator = _get_lazy_interpolator_class()
        store_path = tmp_path / "lazy_interp_jax.zarr"
        _write_test_store(store_path, include_mu_selected=True)
        interpolator = LazyZarrGridInterpolator(str(store_path), chunk_rows=2)

        base_wavelengths = jnp.log10(jnp.array([5000.0, 5002.5], dtype=jnp.float32))
        batched_wavelengths = jnp.stack([base_wavelengths, base_wavelengths], axis=0)
        mus = jnp.array([[1.0], [0.5]], dtype=jnp.float32)
        parameters = jnp.array(
            [[4.25, 5500.0], [5.0, 6000.0]],
            dtype=jnp.float32,
        )

        compiled = jax.jit(jax.vmap(interpolator.intensity, in_axes=(0, 0, 0)))
        result = compiled(batched_wavelengths, mus, parameters)

        expected0 = interpolator.intensity(
            np.asarray(batched_wavelengths[0]),
            np.asarray(mus[0]),
            np.asarray(parameters[0]),
        )
        expected1 = interpolator.intensity(
            np.asarray(batched_wavelengths[1]),
            np.asarray(mus[1]),
            np.asarray(parameters[1]),
        )

        np.testing.assert_allclose(np.asarray(result[0]), np.asarray(expected0), atol=1e-5)
        np.testing.assert_allclose(np.asarray(result[1]), np.asarray(expected1), atol=1e-5)
