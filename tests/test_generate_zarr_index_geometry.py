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
geometry_labels_from_logg = generate_zarr_index.geometry_labels_from_logg
PLANE_PARALLEL = generate_zarr_index.PLANE_PARALLEL
SPHERICAL = generate_zarr_index.SPHERICAL


class DummyFrame:
    def __init__(self, columns):
        self.columns = columns


def _fake_polars():
    return SimpleNamespace(DataFrame=DummyFrame)


def _write_grid_store(tmp_path, params, param_names, geometry=None):
    store_path = tmp_path / "grid.zarr"
    group = zarr.open_group(str(store_path), mode="w")
    group.create_dataset("flux", data=np.ones((params.shape[0], 3)))
    group.create_dataset("continuum", data=np.ones((params.shape[0], 3)))
    group.create_dataset("wavelength", data=np.linspace(5000.0, 5002.0, 3))
    group.create_dataset("params", data=np.asarray(params))
    group.create_dataset("param_names", data=np.asarray(param_names, dtype="S"))
    if geometry is not None:
        group.create_dataset("geometry", data=np.asarray(geometry))
    return store_path


def test_geometry_string_array_is_recorded_verbatim(tmp_path, monkeypatch):
    monkeypatch.setattr(generate_zarr_index, "_get_polars", _fake_polars)
    store_path = _write_grid_store(
        tmp_path,
        params=np.array([[5000.0, 4.4], [4500.0, 1.5]]),
        param_names=["teff", "logg"],
        geometry=np.asarray([PLANE_PARALLEL, SPHERICAL], dtype="S"),
    )

    frame = build_index_frame_from_zarr(store_path)

    assert list(frame.columns["geometry"]) == [PLANE_PARALLEL, SPHERICAL]


def test_geometry_integer_codes_are_mapped(tmp_path, monkeypatch):
    monkeypatch.setattr(generate_zarr_index, "_get_polars", _fake_polars)
    store_path = _write_grid_store(
        tmp_path,
        params=np.array([[5000.0, 4.4], [4500.0, 1.5]]),
        param_names=["teff", "logg"],
        geometry=np.array([0, 1]),  # 0 -> plane_parallel, 1 -> spherical
    )

    frame = build_index_frame_from_zarr(store_path)

    assert list(frame.columns["geometry"]) == [PLANE_PARALLEL, SPHERICAL]


def test_geometry_absent_omits_column(tmp_path, monkeypatch):
    monkeypatch.setattr(generate_zarr_index, "_get_polars", _fake_polars)
    store_path = _write_grid_store(
        tmp_path,
        params=np.array([[5000.0, 4.4], [4500.0, 1.5]]),
        param_names=["teff", "logg"],
    )

    frame = build_index_frame_from_zarr(store_path)

    assert "geometry" not in frame.columns


def test_geometry_derived_from_logg_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(generate_zarr_index, "_get_polars", _fake_polars)
    store_path = _write_grid_store(
        tmp_path,
        params=np.array([[5000.0, 4.4], [4500.0, 1.5]]),
        param_names=["teff", "logg"],
    )

    frame = build_index_frame_from_zarr(store_path, geometry_from_logg=3.0)

    assert list(frame.columns["geometry"]) == [PLANE_PARALLEL, SPHERICAL]


def test_stored_geometry_wins_over_logg_derivation(tmp_path, monkeypatch):
    monkeypatch.setattr(generate_zarr_index, "_get_polars", _fake_polars)
    # Stored labels intentionally disagree with what logg<3 would derive.
    store_path = _write_grid_store(
        tmp_path,
        params=np.array([[5000.0, 4.4], [4500.0, 1.5]]),
        param_names=["teff", "logg"],
        geometry=np.asarray([SPHERICAL, PLANE_PARALLEL], dtype="S"),
    )

    frame = build_index_frame_from_zarr(store_path, geometry_from_logg=3.0)

    assert list(frame.columns["geometry"]) == [SPHERICAL, PLANE_PARALLEL]


def test_exclude_geometry_drops_column_even_when_stored(tmp_path, monkeypatch):
    monkeypatch.setattr(generate_zarr_index, "_get_polars", _fake_polars)
    store_path = _write_grid_store(
        tmp_path,
        params=np.array([[5000.0, 4.4], [4500.0, 1.5]]),
        param_names=["teff", "logg"],
        geometry=np.asarray([PLANE_PARALLEL, SPHERICAL], dtype="S"),
    )

    frame = build_index_frame_from_zarr(store_path, include_geometry=False)

    assert "geometry" not in frame.columns


def test_geometry_rejects_mismatched_length(tmp_path, monkeypatch):
    monkeypatch.setattr(generate_zarr_index, "_get_polars", _fake_polars)
    store_path = _write_grid_store(
        tmp_path,
        params=np.array([[5000.0, 4.4], [4500.0, 1.5]]),
        param_names=["teff", "logg"],
        geometry=np.asarray([SPHERICAL], dtype="S"),
    )

    with pytest.raises(ValueError, match="geometry"):
        build_index_frame_from_zarr(store_path)


def test_geometry_from_logg_without_logg_column_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(generate_zarr_index, "_get_polars", _fake_polars)
    store_path = _write_grid_store(
        tmp_path,
        params=np.array([[5000.0, 0.25], [4500.0, 0.75]]),
        param_names=["teff", "mu"],
    )

    with pytest.raises(KeyError, match="logg"):
        build_index_frame_from_zarr(store_path, geometry_from_logg=3.0)


def test_geometry_labels_from_logg_threshold():
    labels = geometry_labels_from_logg([4.5, 3.0, 2.9, 0.0], threshold=3.0)
    # threshold is strict: logg < 3 -> spherical, logg >= 3 -> plane_parallel.
    assert labels == [PLANE_PARALLEL, PLANE_PARALLEL, SPHERICAL, SPHERICAL]
