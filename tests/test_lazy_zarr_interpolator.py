import importlib.util
import os
import sys
import types
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest
import zarr


@lru_cache(maxsize=1)
def _load_module():
    root = Path(__file__).resolve().parents[1]
    emulator_path = root / "src" / "spice" / "spectrum" / "spectrum_emulator.py"
    interp_path = root / "src" / "spice" / "spectrum" / "lazy_zarr_interpolator.py"

    target_module_names = [
        "spice",
        "spice.spectrum",
        "spice.spectrum.spectrum_emulator",
        "spice.spectrum.lazy_zarr_interpolator",
    ]
    missing = object()
    saved_modules = {name: sys.modules.get(name, missing) for name in target_module_names}

    try:
        spice_pkg = types.ModuleType("spice")
        spice_pkg.__path__ = []
        spectrum_pkg = types.ModuleType("spice.spectrum")
        spectrum_pkg.__path__ = []
        spice_pkg.spectrum = spectrum_pkg

        utils_pkg = types.ModuleType("spice.utils")
        utils_pkg.__path__ = []

        class _LogShim:
            class _Timed:
                def __init__(self, *_a, **_k):
                    pass

                def __enter__(self):
                    return self

                def __exit__(self, *_a):
                    return False

            def info(self, *_a, **_k):
                pass

            def warning(self, *_a, **_k):
                pass

            def timed(self, *_a, **_k):
                return self._Timed()

        utils_pkg.log = _LogShim()
        spice_pkg.utils = utils_pkg

        sys.modules["spice"] = spice_pkg
        sys.modules["spice.spectrum"] = spectrum_pkg
        sys.modules["spice.utils"] = utils_pkg

        emulator_spec = importlib.util.spec_from_file_location(
            "spice.spectrum.spectrum_emulator", emulator_path
        )
        emulator_module = importlib.util.module_from_spec(emulator_spec)
        sys.modules["spice.spectrum.spectrum_emulator"] = emulator_module
        emulator_spec.loader.exec_module(emulator_module)

        interp_spec = importlib.util.spec_from_file_location(
            "spice.spectrum.lazy_zarr_interpolator", interp_path
        )
        interp_module = importlib.util.module_from_spec(interp_spec)
        sys.modules["spice.spectrum.lazy_zarr_interpolator"] = interp_module
        interp_spec.loader.exec_module(interp_module)
        return interp_module
    finally:
        for name in target_module_names:
            old_module = saved_modules[name]
            if old_module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


def _write_interpolator_store(store_path: Path):
    import pandas as pd

    group = zarr.open_group(str(store_path), mode="w")

    wavelengths = np.array([5000.0, 5001.0, 5002.0, 5003.0], dtype=np.float32)
    group.create_array("wavelength", data=wavelengths)

    teff_values = np.array([5000.0, 6000.0], dtype=np.float32)
    logg_values = np.array([4.0, 4.5, 5.0], dtype=np.float32)
    rows = [(float(t), float(g)) for t in teff_values for g in logg_values]

    n_rows = len(rows)
    n_wave = wavelengths.shape[0]
    flux = np.empty((n_rows, n_wave), dtype=np.float32)
    continuum = np.empty((n_rows, n_wave), dtype=np.float32)
    for i, (teff, logg) in enumerate(rows):
        base = 1e-3 * teff + 10.0 * logg
        flux[i, :] = base + np.arange(n_wave, dtype=np.float32)
        continuum[i, :] = flux[i, :] + 100.0

    group.create_array("flux", data=flux, chunks=(2, n_wave))
    group.create_array("continuum", data=continuum, chunks=(2, n_wave))

    df = pd.DataFrame(
        {
            "teff": [r[0] for r in rows],
            "logg": [r[1] for r in rows],
            "row_idx": np.arange(n_rows, dtype=np.int32),
        }
    )
    df.to_parquet(f"{store_path}/index.parquet")

    rowwise_bytes = flux.nbytes + continuum.nbytes
    return flux, continuum, rowwise_bytes


def _make_interpolator(store_path: Path, **kwargs):
    mod = _load_module()
    return mod.LazyZarrInterpolator(
        str(store_path),
        solar_parameters=np.array([5500.0, 4.5], dtype=np.float32),
        params=["teff", "logg"],
        **kwargs,
    )


class TestLazyZarrInterpolatorInMemory:
    def test_default_keeps_rowwise_lazy_as_zarr(self, tmp_path):
        store_path = tmp_path / "grid_lazy.zarr"
        _write_interpolator_store(store_path)

        interp = _make_interpolator(store_path)

        assert interp._rowwise_in_memory is False
        assert isinstance(interp.rowwise_data["flux"], zarr.Array)
        assert isinstance(interp.rowwise_data["continuum"], zarr.Array)

    def test_in_memory_true_materializes_rowwise_on_device(self, tmp_path):
        store_path = tmp_path / "grid_eager.zarr"
        _write_interpolator_store(store_path)

        interp = _make_interpolator(store_path, in_memory=True)

        assert interp._rowwise_in_memory is True
        for key in ("flux", "continuum"):
            arr = interp.rowwise_data[key]
            assert not isinstance(arr, zarr.Array)
            # JAX arrays expose .at / .device_buffer; duck-type on .at
            assert hasattr(arr, "at")

    def test_in_memory_auto_loads_when_under_threshold(self, tmp_path):
        store_path = tmp_path / "grid_auto_small.zarr"
        _, _, rowwise_bytes = _write_interpolator_store(store_path)

        interp = _make_interpolator(
            store_path,
            in_memory="auto",
            in_memory_threshold_bytes=rowwise_bytes * 2,
        )

        assert interp._rowwise_in_memory is True
        assert not isinstance(interp.rowwise_data["flux"], zarr.Array)

    def test_in_memory_auto_stays_lazy_when_over_threshold(self, tmp_path):
        store_path = tmp_path / "grid_auto_large.zarr"
        _write_interpolator_store(store_path)

        interp = _make_interpolator(
            store_path,
            in_memory="auto",
            in_memory_threshold_bytes=1,
        )

        assert interp._rowwise_in_memory is False
        assert isinstance(interp.rowwise_data["flux"], zarr.Array)

    def test_invalid_in_memory_raises(self, tmp_path):
        store_path = tmp_path / "grid_invalid.zarr"
        _write_interpolator_store(store_path)

        with pytest.raises(ValueError, match="in_memory must be"):
            _make_interpolator(store_path, in_memory="sometimes")

    def test_in_memory_and_lazy_paths_agree_numerically(self, tmp_path):
        store_path = tmp_path / "grid_equivalence.zarr"
        _write_interpolator_store(store_path)

        interp_lazy = _make_interpolator(store_path, in_memory=False)
        interp_eager = _make_interpolator(store_path, in_memory=True)

        queries = jnp.array(
            [[5500.0, 4.25], [5000.0, 4.0], [5750.0, 4.75]],
            dtype=jnp.float32,
        )

        batch_lazy = interp_lazy.get_weighted_batch(queries)
        batch_eager = interp_eager.get_weighted_batch(queries)

        np.testing.assert_allclose(
            np.asarray(batch_eager["flux"]),
            np.asarray(batch_lazy["flux"]),
            atol=1e-4,
        )
        np.testing.assert_allclose(
            np.asarray(batch_eager["continuum"]),
            np.asarray(batch_lazy["continuum"]),
            atol=1e-4,
        )

    def test_in_memory_accumulation_does_not_use_host_callback(self, tmp_path, monkeypatch):
        store_path = tmp_path / "grid_no_callback.zarr"
        _write_interpolator_store(store_path)

        interp = _make_interpolator(store_path, in_memory=True)

        def _fail(*_a, **_k):
            raise AssertionError("host callback path should not be used when in_memory=True")

        monkeypatch.setattr(interp, "_accumulate_direct", _fail)

        queries = jnp.array([[5500.0, 4.25]], dtype=jnp.float32)
        batch = interp.get_weighted_batch(queries)

        assert batch["flux"].shape == (1, 4)
        assert batch["continuum"].shape == (1, 4)
