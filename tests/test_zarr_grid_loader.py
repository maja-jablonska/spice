import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import zarr

os.environ.setdefault("JAX_PLATFORMS", "cpu")

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*does not have a Zarr V3 specification.*"
)

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "spice"
    / "spectrum"
    / "zarr_grid_loader.py"
)
SPEC = importlib.util.spec_from_file_location("zarr_grid_loader_under_test", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load module spec from {MODULE_PATH}")
zarr_grid_loader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(zarr_grid_loader)

ZarrGridLoader = zarr_grid_loader.ZarrGridLoader
_effective_chunk_rows = zarr_grid_loader._effective_chunk_rows
_load_dense_2d_array = zarr_grid_loader._load_dense_2d_array
_zarr_row_chunk_size = zarr_grid_loader._zarr_row_chunk_size


def _write_test_store(store_path: Path, continuum_row_delta: int = 0):
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
    for i, (teff, logg, _) in enumerate(params):
        base = 1e-3 * teff + 10.0 * logg
        flux[i, :] = base + np.arange(n_wave, dtype=np.float32)

    continuum_rows = n_rows + continuum_row_delta
    if continuum_rows < 1:
        raise ValueError("continuum_row_delta produced an empty continuum.")
    continuum = flux[:continuum_rows] + 100.0

    group.create_array("flux", data=flux, chunks=(2, n_wave))
    group.create_array("continuum", data=continuum, chunks=(2, n_wave))
    return params, flux, continuum


class TestChunkedLoadingHelpers:
    def test_zarr_row_chunk_size_returns_first_dimension(self, tmp_path):
        group = zarr.open_group(str(tmp_path / "arr.zarr"), mode="w")
        arr = group.create_array(
            "values",
            data=np.arange(24, dtype=np.float32).reshape(6, 4),
            chunks=(4, 4),
        )
        assert _zarr_row_chunk_size(arr) == 4

    def test_effective_chunk_rows_aligns_to_chunk_boundaries(self):
        assert _effective_chunk_rows(500, 2310, 32) == 512
        assert _effective_chunk_rows(5, 6, 4) == 6
        assert _effective_chunk_rows(500, 2310, None) == 500

    def test_effective_chunk_rows_rejects_non_positive_values(self):
        with pytest.raises(ValueError, match="chunk_rows must be a positive integer"):
            _effective_chunk_rows(0, 10, 4)

    def test_load_dense_2d_array_matches_source_data(self, tmp_path):
        group = zarr.open_group(str(tmp_path / "dense.zarr"), mode="w")
        expected = np.arange(35, dtype=np.float32).reshape(7, 5)
        arr = group.create_array("values", data=expected, chunks=(2, 5))

        loaded = _load_dense_2d_array(arr, name="values", chunk_rows=3)

        np.testing.assert_array_equal(loaded, expected)


class TestZarrGridLoaderValidation:
    def test_rejects_non_positive_chunk_rows(self):
        with pytest.raises(ValueError, match="chunk_rows must be a positive integer"):
            ZarrGridLoader("unused.zarr", chunk_rows=0)

    def test_rejects_mismatched_flux_and_continuum_rows(self, tmp_path):
        store_path = tmp_path / "bad_grid.zarr"
        _write_test_store(store_path, continuum_row_delta=-1)

        with pytest.raises(ValueError, match="Flux and continuum row counts must match"):
            ZarrGridLoader(str(store_path), chunk_rows=3)


class TestZarrGridLoaderDenseGrid:
    def test_builds_expected_flux_and_continuum_cubes(self, tmp_path):
        store_path = tmp_path / "dense_grid.zarr"
        params, flux, continuum = _write_test_store(store_path)

        loader = ZarrGridLoader(str(store_path), chunk_rows=3)

        assert loader.axis_names == ["logg", "teff"]
        np.testing.assert_array_equal(
            np.asarray(loader.axes_jnp[0]), np.unique(params[:, 1])
        )
        np.testing.assert_array_equal(
            np.asarray(loader.axes_jnp[1]), np.unique(params[:, 0])
        )

        order = np.lexsort([params[:, 0], params[:, 1]])
        expected_flux_cube = flux[order].reshape(3, 2, -1)
        expected_continuum_cube = continuum[order].reshape(3, 2, -1)

        np.testing.assert_allclose(np.asarray(loader.flux_cube_jnp), expected_flux_cube)
        np.testing.assert_allclose(
            np.asarray(loader.continuum_cube_jnp), expected_continuum_cube
        )
