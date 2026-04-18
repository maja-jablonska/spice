import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "spice"
    / "spectrum"
    / "generate_zarr_index.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location("generate_zarr_index", MODULE_PATH)
generate_zarr_index = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC.loader is not None
MODULE_SPEC.loader.exec_module(generate_zarr_index)

build_index_frame_from_zarr = generate_zarr_index.build_index_frame_from_zarr


class DummyFrame:
    def __init__(self, columns):
        self.columns = columns


def _fake_polars():
    return SimpleNamespace(DataFrame=DummyFrame)


def _write_grid_store(tmp_path, params, param_names, mu_selected=None):
    store_path = tmp_path / "grid.zarr"
    group = zarr.open_group(str(store_path), mode="w")
    group.create_dataset("flux", data=np.ones((params.shape[0], 3)))
    group.create_dataset("continuum", data=np.ones((params.shape[0], 3)))
    group.create_dataset("wavelength", data=np.linspace(5000.0, 5002.0, 3))
    group.create_dataset("params", data=np.asarray(params))
    group.create_dataset("param_names", data=np.asarray(param_names, dtype="S"))
    if mu_selected is not None:
        group.create_dataset("mu_selected", data=np.asarray(mu_selected))
    return store_path


def test_build_index_frame_reads_mu_from_mu_selected(tmp_path, monkeypatch):
    monkeypatch.setattr(generate_zarr_index, "_get_polars", _fake_polars)
    store_path = _write_grid_store(
        tmp_path,
        params=np.array([[5000.0, 4.0], [5100.0, 4.1]]),
        param_names=["teff", "logg"],
        mu_selected=np.array([0.25, 0.75]),
    )

    frame = build_index_frame_from_zarr(store_path)

    np.testing.assert_array_equal(frame.columns["row_idx"], np.array([0, 1]))
    np.testing.assert_array_equal(frame.columns["teff"], np.array([5000.0, 5100.0]))
    np.testing.assert_array_equal(frame.columns["logg"], np.array([4.0, 4.1]))
    np.testing.assert_array_equal(frame.columns["mu"], np.array([0.25, 0.75]))


def test_build_index_frame_excludes_mu_selected_when_requested(tmp_path, monkeypatch):
    monkeypatch.setattr(generate_zarr_index, "_get_polars", _fake_polars)
    store_path = _write_grid_store(
        tmp_path,
        params=np.array([[5000.0, 4.0], [5100.0, 4.1]]),
        param_names=["teff", "logg"],
        mu_selected=np.array([0.25, 0.75]),
    )

    frame = build_index_frame_from_zarr(store_path, include_mu=False)

    assert "mu" not in frame.columns


def test_build_index_frame_falls_back_to_mu_param_when_mu_selected_is_missing(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(generate_zarr_index, "_get_polars", _fake_polars)
    store_path = _write_grid_store(
        tmp_path,
        params=np.array([[5000.0, 0.25], [5100.0, 0.75]]),
        param_names=["teff", "mu"],
    )

    frame = build_index_frame_from_zarr(store_path)

    np.testing.assert_array_equal(frame.columns["teff"], np.array([5000.0, 5100.0]))
    np.testing.assert_array_equal(frame.columns["mu"], np.array([0.25, 0.75]))


def test_build_index_frame_rejects_mismatched_mu_selected_length(tmp_path, monkeypatch):
    monkeypatch.setattr(generate_zarr_index, "_get_polars", _fake_polars)
    store_path = _write_grid_store(
        tmp_path,
        params=np.array([[5000.0, 4.0], [5100.0, 4.1]]),
        param_names=["teff", "logg"],
        mu_selected=np.array([0.25]),
    )

    with pytest.raises(ValueError, match="mu_selected"):
        build_index_frame_from_zarr(store_path)
