import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")


def _load_interpolator_module():
    root = Path(__file__).resolve().parents[1]
    loader_path = root / "src" / "spice" / "spectrum" / "zarr_grid_loader.py"
    emulator_path = root / "src" / "spice" / "spectrum" / "spectrum_emulator.py"
    interpolator_path = (
        root / "src" / "spice" / "spectrum" / "zarr_grid_interpolator.py"
    )

    target_module_names = [
        "spice",
        "spice.spectrum",
        "spice.spectrum.spectrum_emulator",
        "spice.spectrum.zarr_grid_loader",
        "spice.spectrum.zarr_grid_interpolator",
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
            "spice.spectrum.zarr_grid_loader", loader_path
        )
        if loader_spec is None or loader_spec.loader is None:
            raise RuntimeError(f"Could not load module spec from {loader_path}")
        loader_module = importlib.util.module_from_spec(loader_spec)
        sys.modules["spice.spectrum.zarr_grid_loader"] = loader_module
        loader_spec.loader.exec_module(loader_module)

        interpolator_spec = importlib.util.spec_from_file_location(
            "spice.spectrum.zarr_grid_interpolator", interpolator_path
        )
        if interpolator_spec is None or interpolator_spec.loader is None:
            raise RuntimeError(f"Could not load module spec from {interpolator_path}")
        interpolator_module = importlib.util.module_from_spec(interpolator_spec)
        sys.modules["spice.spectrum.zarr_grid_interpolator"] = interpolator_module
        interpolator_spec.loader.exec_module(interpolator_module)
        return interpolator_module
    finally:
        for name in target_module_names:
            old_module = saved_modules[name]
            if old_module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


zarr_grid_interpolator_module = _load_interpolator_module()
ZarrGridInterpolator = zarr_grid_interpolator_module.ZarrGridInterpolator


class TestZarrGridInterpolator:
    def test_init_uses_loader_and_sets_parameter_names(self, monkeypatch):
        class DummyLoader:
            def __init__(self, store_path, chunk_rows=500):
                self.store_path = store_path
                self.chunk_rows = chunk_rows
                self.axis_names = ["mu", "logg", "teff", "feh", "vmicro"]

        monkeypatch.setattr(
            zarr_grid_interpolator_module, "ZarrGridLoader", DummyLoader
        )

        interpolator = ZarrGridInterpolator("grid.zarr", chunk_rows=321)

        assert interpolator.grid_loader.store_path == "grid.zarr"
        assert interpolator.grid_loader.chunk_rows == 321
        assert interpolator.parameter_names == ["mu", "logg", "teff", "feh", "vmicro"]

    def test_stellar_parameter_names_filters_mu(self):
        interpolator = ZarrGridInterpolator.__new__(ZarrGridInterpolator)
        interpolator.parameter_names = ["mu", "logg", "teff", "feh", "vmicro"]
        assert interpolator.stellar_parameter_names == ["logg", "teff", "feh", "vmicro"]

    def test_to_parameters_none_returns_solar_parameters(self):
        interpolator = ZarrGridInterpolator.__new__(ZarrGridInterpolator)
        interpolator.parameter_names = ["logg", "teff", "feh", "vmicro"]
        np.testing.assert_allclose(
            np.asarray(interpolator.to_parameters(None)),
            np.asarray(interpolator.solar_parameters),
        )

    def test_to_parameters_dict_uses_defaults_for_missing_values(self):
        interpolator = ZarrGridInterpolator.__new__(ZarrGridInterpolator)
        interpolator.parameter_names = ["logg", "teff", "feh", "vmicro"]

        params = interpolator.to_parameters({"teff": 6100.0, "feh": -0.2})

        np.testing.assert_allclose(
            np.asarray(params), np.array([4.44, 6100.0, -0.2, 0.0], dtype=np.float32)
        )

    def test_to_parameters_validates_array_shapes(self):
        interpolator = ZarrGridInterpolator.__new__(ZarrGridInterpolator)
        interpolator.parameter_names = ["logg", "teff", "feh", "vmicro"]

        with pytest.raises(ValueError, match="must have length 4"):
            interpolator.to_parameters(np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError, match="must have 4 columns"):
            interpolator.to_parameters(np.ones((2, 3)))
        with pytest.raises(ValueError, match="must be a 1D or 2D array"):
            interpolator.to_parameters(np.ones((1, 1, 1)))
        with pytest.raises(ValueError, match="Parameters must be a dict"):
            interpolator.to_parameters("invalid")

    def test_flux_raises_not_implemented(self):
        interpolator = ZarrGridInterpolator.__new__(ZarrGridInterpolator)
        with pytest.raises(NotImplementedError):
            interpolator.flux(np.array([3.7, 3.8]), np.array([4.4, 5777.0, 0.0, 0.0]))

    def test_intensity_delegates_to_grid_loader(self):
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

        interpolator = ZarrGridInterpolator.__new__(ZarrGridInterpolator)
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
        np.testing.assert_allclose(
            result[:, 0], np.array([10.0, 25.0], dtype=np.float32), atol=1e-5
        )
        np.testing.assert_allclose(
            result[:, 1], np.array([1.0, 2.5], dtype=np.float32), atol=1e-5
        )

    def test_intensity_accepts_linear_wavelength_inputs(self):
        class DummyLoader:
            axis_names = ["mu", "logg", "teff", "feh", "vmicro"]
            wavelength_jnp = np.array([100.0, 200.0, 300.0], dtype=np.float32)

            def interpolate_spectrum_and_continuum(self, query_params, validate=True):
                return (
                    np.array([10.0, 20.0, 30.0], dtype=np.float32),
                    np.array([1.0, 2.0, 3.0], dtype=np.float32),
                )

        interpolator = ZarrGridInterpolator.__new__(ZarrGridInterpolator)
        interpolator.grid_loader = DummyLoader()
        interpolator.parameter_names = ["mu", "logg", "teff", "feh", "vmicro"]

        wavelengths = np.array([100.0, 250.0], dtype=np.float32)
        result = interpolator.intensity(
            wavelengths, 0.4, np.array([4.4, 5777.0, 0.0, 0.0], dtype=np.float32)
        )

        np.testing.assert_allclose(
            result[:, 0], np.array([10.0, 25.0], dtype=np.float32), atol=1e-5
        )
        np.testing.assert_allclose(
            result[:, 1], np.array([1.0, 2.5], dtype=np.float32), atol=1e-5
        )

    def test_intensity_uses_mu_selected_fallback_when_mu_axis_missing(self):
        class DummyLoader:
            axis_names = ["logg", "teff", "feh", "vmicro"]
            wavelength_jnp = np.array([100.0, 200.0, 300.0], dtype=np.float32)
            mu_selected_cube_jnp = np.array([[[[[0.5]]]]], dtype=np.float32)

            def __init__(self):
                self.last_query = None
                self.last_mu_query = None

            def interpolate_spectrum_and_continuum(self, query_params, validate=True):
                self.last_query = np.asarray(query_params)
                return (
                    np.array([10.0, 20.0, 30.0], dtype=np.float32),
                    np.array([1.0, 2.0, 3.0], dtype=np.float32),
                )

            def _interpolate_spectrum_jit(self, query_params, cube):
                self.last_mu_query = np.asarray(query_params)
                return np.array([0.5], dtype=np.float32)

        interpolator = ZarrGridInterpolator.__new__(ZarrGridInterpolator)
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
        # With reference mu=0.5 and default linear LD (u=0.6), mu=1.0 scales by 1/0.7.
        np.testing.assert_allclose(
            result_mu_one[:, 0] / result_mu_half[:, 0],
            np.array([1.0 / 0.7, 1.0 / 0.7], dtype=np.float32),
            atol=1e-5,
        )
