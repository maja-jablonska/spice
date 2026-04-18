import importlib.util
from pathlib import Path

import numpy as np
import zarr

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "spice"
    / "spectrum"
    / "generate_zarr_index.py"
)
SPEC = importlib.util.spec_from_file_location("generate_zarr_index_under_test", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load module spec from {MODULE_PATH}")
generate_zarr_index = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generate_zarr_index)

write_index_parquet = generate_zarr_index.write_index_parquet
pl = generate_zarr_index._get_polars()


def _write_test_store(store_path: Path, nested: bool = False, include_mu: bool = False):
    root = zarr.open_group(str(store_path), mode="w")
    group = root.create_group("wrapper") if nested else root

    param_names = [b"teff", b"logg", b"fixed"]
    if include_mu:
        param_names = [b"mu", *param_names]
    group.create_array(
        "param_names",
        data=np.array(param_names, dtype="|S16"),
    )
    group.create_array(
        "wavelength",
        data=np.array([5000.0, 5001.0, 5002.0, 5003.0], dtype=np.float32),
    )

    rows = [
        [6000.0, 4.5, 1.23],
        [5000.0, 4.0, 1.23],
        [6000.0, 4.0, 1.23],
    ]
    if include_mu:
        rows = [
            [0.1, *rows[0]],
            [0.5, *rows[1]],
            [0.9, *rows[2]],
        ]
    params = np.asarray(rows, dtype=np.float32)
    group.create_array("params", data=params, chunks=(2, params.shape[1]))
    group.create_array("flux", data=np.ones((params.shape[0], 4), dtype=np.float32))
    group.create_array("continuum", data=np.ones((params.shape[0], 4), dtype=np.float32))
    return params


def test_write_index_parquet_uses_store_default_path(tmp_path):
    store_path = tmp_path / "grid.zarr"
    params = _write_test_store(store_path)

    output_path = write_index_parquet(store_path)

    assert output_path == store_path / "index.parquet"
    frame = pl.read_parquet(output_path)

    assert frame.columns == ["row_idx", "teff", "logg", "fixed"]
    np.testing.assert_array_equal(frame["row_idx"].to_numpy(), np.arange(params.shape[0]))
    np.testing.assert_allclose(frame["teff"].to_numpy(), params[:, 0])
    np.testing.assert_allclose(frame["logg"].to_numpy(), params[:, 1])
    np.testing.assert_allclose(frame["fixed"].to_numpy(), params[:, 2])


def test_write_index_parquet_resolves_nested_wrapper_group(tmp_path):
    store_path = tmp_path / "nested_grid.zarr"
    params = _write_test_store(store_path, nested=True)
    output_path = tmp_path / "custom-index.parquet"

    write_index_parquet(store_path, output_path=output_path, compression="snappy")

    frame = pl.read_parquet(output_path)
    np.testing.assert_allclose(frame["teff"].to_numpy(), params[:, 0])


def test_write_index_parquet_includes_mu_by_default(tmp_path):
    store_path = tmp_path / "mu_grid.zarr"
    params = _write_test_store(store_path, include_mu=True)

    output_path = write_index_parquet(store_path)

    frame = pl.read_parquet(output_path)

    assert frame.columns == ["row_idx", "mu", "teff", "logg", "fixed"]
    np.testing.assert_allclose(frame["mu"].to_numpy(), params[:, 0])


def test_write_index_parquet_can_exclude_mu(tmp_path):
    store_path = tmp_path / "mu_grid_no_mu_index.zarr"
    params = _write_test_store(store_path, include_mu=True)

    output_path = write_index_parquet(store_path, include_mu=False)

    frame = pl.read_parquet(output_path)

    assert frame.columns == ["row_idx", "teff", "logg", "fixed"]
    np.testing.assert_allclose(frame["teff"].to_numpy(), params[:, 1])
