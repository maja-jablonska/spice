import importlib.util
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import zarr

os.environ.setdefault("JAX_PLATFORMS", "cpu")

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "spice"
    / "spectrum"
    / "lazy_zarr_grid_loader.py"
)
SPEC = importlib.util.spec_from_file_location("lazy_zarr_grid_loader_under_test", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load module spec from {MODULE_PATH}")
lazy_zarr_grid_loader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lazy_zarr_grid_loader)

LazyZarrGridLoader = lazy_zarr_grid_loader.LazyZarrGridLoader


def _write_test_store(store_path: Path, include_mu_selected: bool = False):
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
    return params, flux, continuum, mu_selected


class TestLazyZarrGridLoader:
    def test_builds_axis_metadata_and_row_index_cube(self, tmp_path):
        store_path = tmp_path / "lazy_grid.zarr"
        params, _, _, _ = _write_test_store(store_path)

        loader = LazyZarrGridLoader(str(store_path), chunk_rows=2)

        assert loader.axis_names == ["logg", "teff"]
        assert loader.row_index_cube.shape == (3, 2)
        np.testing.assert_array_equal(
            np.sort(loader.row_index_cube.reshape(-1)), np.arange(params.shape[0])
        )

    def test_logs_initialization_progress_and_ready_summary(self, tmp_path, caplog):
        store_path = tmp_path / "lazy_grid_logging.zarr"
        _write_test_store(store_path, include_mu_selected=True)

        with caplog.at_level("WARNING"):
            LazyZarrGridLoader(str(store_path), chunk_rows=2)

        assert "Opening lazy zarr store" in caplog.text
        assert "Lazy grid loader init 1/4 complete" in caplog.text
        assert "metadata_to_read=" in caplog.text
        assert "Lazy grid loader init 4/4 complete" in caplog.text
        assert "row index cube ready; JAX metadata deferred until first use" in caplog.text
        assert "Lazy spectral grid ready in" in caplog.text

    def test_materializes_jax_metadata_lazily(self, tmp_path, caplog):
        store_path = tmp_path / "lazy_grid_jax_metadata.zarr"
        _write_test_store(store_path)

        with caplog.at_level("WARNING"):
            loader = LazyZarrGridLoader(str(store_path), chunk_rows=2)
            assert loader._axes_jnp is None
            assert "Materializing lazy-grid JAX metadata on first use" not in caplog.text
            _ = loader.wavelength_jnp

        assert loader._axes_jnp is not None
        assert "Materializing lazy-grid JAX metadata on first use" in caplog.text

    def test_reuses_lazy_index_cache_on_second_init(self, tmp_path, monkeypatch, caplog):
        store_path = tmp_path / "lazy_grid_cache.zarr"
        _write_test_store(store_path)
        cache_path = tmp_path / "lazy_grid_cache.zarr.spice_lazy_index_cache.npz"

        with caplog.at_level("WARNING"):
            LazyZarrGridLoader(str(store_path), chunk_rows=2)

        assert cache_path.exists()
        assert "Wrote lazy index cache" in caplog.text

        caplog.clear()
        monkeypatch.setattr(
            lazy_zarr_grid_loader,
            "_load_param_names",
            lambda store: (_ for _ in ()).throw(AssertionError("cache should skip param_names")),
        )

        with caplog.at_level("WARNING"):
            loader = LazyZarrGridLoader(str(store_path), chunk_rows=2)

        assert loader.axis_names == ["logg", "teff"]
        assert "Loaded lazy index cache from disk" in caplog.text
        assert "metadata cache hit" in caplog.text

    def test_interpolates_without_materializing_full_grid(self, tmp_path):
        store_path = tmp_path / "lazy_interp_grid.zarr"
        _write_test_store(store_path)

        loader = LazyZarrGridLoader(str(store_path), chunk_rows=2)
        spectrum, continuum = loader.interpolate_spectrum_and_continuum(
            np.array([4.25, 5500.0], dtype=np.float32)
        )

        expected = np.array([48.0, 49.0, 50.0, 51.0], dtype=np.float32)
        np.testing.assert_allclose(np.asarray(spectrum), expected, atol=1e-5)
        np.testing.assert_allclose(np.asarray(continuum), expected + 100.0, atol=1e-5)

    def test_load_rows_preserves_requested_order_and_duplicates(self, tmp_path):
        store_path = tmp_path / "lazy_load_rows.zarr"
        _, flux, _, _ = _write_test_store(store_path)

        loader = LazyZarrGridLoader(str(store_path), chunk_rows=2)
        row_ids = np.array([5, 1, 5, 0], dtype=np.int64)

        loaded = loader._load_rows(loader.flux_arr, row_ids)

        np.testing.assert_allclose(loaded, flux[row_ids])

    def test_load_rows_reuses_in_memory_row_cache(self, tmp_path):
        store_path = tmp_path / "lazy_load_rows_cache.zarr"
        _, flux, _, _ = _write_test_store(store_path)

        loader = LazyZarrGridLoader(str(store_path), chunk_rows=2, row_cache_bytes=4096)

        class CountingArray:
            def __init__(self, wrapped):
                self._wrapped = wrapped
                self.shape = wrapped.shape
                self.dtype = wrapped.dtype
                self.path = wrapped.path
                self.calls = 0

            def get_orthogonal_selection(self, selection):
                self.calls += 1
                return self._wrapped.get_orthogonal_selection(selection)

        counted_flux = CountingArray(loader.flux_arr)
        row_ids = np.array([5, 1, 5, 0], dtype=np.int64)

        loaded_first = loader._load_rows(counted_flux, row_ids)
        loaded_second = loader._load_rows(counted_flux, row_ids)

        np.testing.assert_allclose(loaded_first, flux[row_ids])
        np.testing.assert_allclose(loaded_second, flux[row_ids])
        assert counted_flux.calls == 2

    def test_interpolation_prefetches_grid_chunk_into_row_cache(self, tmp_path):
        store_path = tmp_path / "lazy_prefetch_chunk.zarr"
        _write_test_store(store_path)

        loader = LazyZarrGridLoader(
            str(store_path),
            chunk_rows=2,
            row_cache_bytes=4096,
            prefetch_axis_chunk_size=1,
        )

        spectrum = loader.interpolate_spectrum(np.array([4.25, 5500.0], dtype=np.float32))

        flux_cached_rows = [
            key for key in loader._row_cache.keys() if key[0] == loader.flux_arr.path
        ]

        np.testing.assert_allclose(
            np.asarray(spectrum),
            np.array([48.0, 49.0, 50.0, 51.0], dtype=np.float32),
            atol=1e-5,
        )
        assert len(flux_cached_rows) == 6

    def test_interpolate_spectrum_and_continuum_share_cache_budget(self, tmp_path):
        store_path = tmp_path / "lazy_prefetch_shared_budget.zarr"
        _write_test_store(store_path)

        loader = LazyZarrGridLoader(
            str(store_path),
            chunk_rows=2,
            row_cache_bytes=128,
            prefetch_axis_chunk_size=1,
        )

        spectrum, continuum = loader.interpolate_spectrum_and_continuum(
            np.array([4.25, 5500.0], dtype=np.float32)
        )

        flux_cached_rows = [
            key for key in loader._row_cache.keys() if key[0] == loader.flux_arr.path
        ]
        continuum_cached_rows = [
            key for key in loader._row_cache.keys() if key[0] == loader.continuum_arr.path
        ]

        np.testing.assert_allclose(
            np.asarray(spectrum),
            np.array([48.0, 49.0, 50.0, 51.0], dtype=np.float32),
            atol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(continuum),
            np.array([148.0, 149.0, 150.0, 151.0], dtype=np.float32),
            atol=1e-5,
        )
        assert len(flux_cached_rows) == 4
        assert len(continuum_cached_rows) == 4

    def test_interpolates_mu_selected_lazily(self, tmp_path):
        store_path = tmp_path / "lazy_mu_grid.zarr"
        _write_test_store(store_path, include_mu_selected=True)

        loader = LazyZarrGridLoader(str(store_path), chunk_rows=2)
        mu_selected = loader.interpolate_mu_selected(
            np.array([4.25, 5500.0], dtype=np.float32)
        )

        np.testing.assert_allclose(np.asarray(mu_selected), np.array([0.9625], dtype=np.float32), atol=1e-5)

    def test_interpolates_within_jax_jit(self, tmp_path):
        store_path = tmp_path / "lazy_jax_grid.zarr"
        _write_test_store(store_path, include_mu_selected=True)

        loader = LazyZarrGridLoader(str(store_path), chunk_rows=2)
        query = jnp.array([4.25, 5500.0], dtype=jnp.float32)

        compiled = jax.jit(lambda q: loader.interpolate_spectrum_and_continuum(q))
        spectrum, continuum = compiled(query)
        mu_selected = jax.jit(lambda q: loader.interpolate_mu_selected(q))(query)

        expected = np.array([48.0, 49.0, 50.0, 51.0], dtype=np.float32)
        np.testing.assert_allclose(np.asarray(spectrum), expected, atol=1e-5)
        np.testing.assert_allclose(np.asarray(continuum), expected + 100.0, atol=1e-5)
        np.testing.assert_allclose(np.asarray(mu_selected), np.array([0.9625], dtype=np.float32), atol=1e-5)

    def test_rejects_non_dense_rectilinear_params(self, tmp_path):
        store_path = tmp_path / "lazy_bad_grid.zarr"
        group = zarr.open_group(str(store_path), mode="w")
        group.create_array("param_names", data=np.array([b"teff", b"logg"], dtype="|S16"))
        group.create_array("wavelength", data=np.array([5000.0, 5001.0], dtype=np.float32))
        params = np.array([[5000.0, 4.0], [6000.0, 4.0], [5000.0, 4.5]], dtype=np.float32)
        group.create_array("params", data=params, chunks=(2, 2))
        flux = np.arange(6, dtype=np.float32).reshape(3, 2)
        group.create_array("flux", data=flux, chunks=(2, 2))
        group.create_array("continuum", data=flux + 100.0, chunks=(2, 2))

        with pytest.raises(ValueError, match="dense rectilinear"):
            LazyZarrGridLoader(str(store_path), chunk_rows=2)
