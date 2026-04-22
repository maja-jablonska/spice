import importlib
import os
import sys
from pathlib import Path


os.environ.setdefault("JAX_PLATFORMS", "cpu")


def _clear_spice_modules():
    for module_name in list(sys.modules):
        if module_name == "spice" or module_name.startswith("spice."):
            sys.modules.pop(module_name, None)


def _prepend_src_path(monkeypatch):
    src_path = Path(__file__).resolve().parents[1] / "src"
    monkeypatch.syspath_prepend(str(src_path))
    importlib.invalidate_caches()


class TestPackageImports:
    def test_import_spice_does_not_eagerly_import_subpackages(self, monkeypatch):
        _prepend_src_path(monkeypatch)
        _clear_spice_modules()

        spice = importlib.import_module("spice")

        assert spice.__file__.endswith("src/spice/__init__.py")
        assert "spice.geometry" not in sys.modules
        assert "spice.models" not in sys.modules
        assert "spice.plots" not in sys.modules
        assert "spice.spectrum" not in sys.modules

    def test_models_namespace_stays_lazy_until_symbol_access(self, monkeypatch):
        _prepend_src_path(monkeypatch)
        _clear_spice_modules()

        models = importlib.import_module("spice.models")

        assert "spice.models.mesh_model" not in sys.modules
        assert "spice.models.mesh_generation" not in sys.modules

        assert models.MeshModel is not None

        assert "spice.models.mesh_model" in sys.modules

    def test_plots_namespace_import_is_lazy(self, monkeypatch):
        _prepend_src_path(monkeypatch)
        _clear_spice_modules()

        plots = importlib.import_module("spice.plots")

        assert "spice.plots.plot_mesh" not in sys.modules
        assert "plot_2D" in dir(plots)
