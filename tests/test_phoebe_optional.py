"""Smoke tests must fail on broken core imports and skip absent extras."""
import pytest

from tests.phoebe_helpers import import_phoebe_or_skip


def test_spectrum_functionality():
    from spice.spectrum import Blackbody, SpectrumEmulator, simulate_observed_flux
    from spice.spectrum.spectrum import (
        simulate_monochromatic_luminosity, luminosity,
        AB_passband_luminosity, absolute_bol_luminosity,
    )
    from spice.spectrum.utils import apply_spectral_resolution

    assert callable(Blackbody().intensity)
    assert SpectrumEmulator is not None
    assert all(callable(fn) for fn in (
        simulate_observed_flux, simulate_monochromatic_luminosity, luminosity,
        AB_passband_luminosity, absolute_bol_luminosity, apply_spectral_resolution,
    ))


def test_basic_functionality():
    import jax.numpy as jnp
    from spice.models import IcosphereModel

    model = IcosphereModel.construct(42, 1.0, 1.0, [5000.0], ["teff"])
    assert model.parameters.shape == (model.faces.shape[0], 1)
    assert jnp.all(jnp.isfinite(model.visible_cast_areas))
    assert jnp.sum(model.visible_cast_areas) > 0


def test_phoebe_functionality():
    phoebe = import_phoebe_or_skip()
    from spice.models.phoebe_utils import PHOEBE_AVAILABLE

    assert PHOEBE_AVAILABLE
    bundle = phoebe.default_binary()
    assert "primary" in bundle.components
    assert "secondary" in bundle.components


def test_broken_phoebe_install_is_treated_as_unavailable(monkeypatch):
    """A PHOEBE that is installed but fails on import with a non-ImportError
    (e.g. astropy too old for NumPy: AttributeError on np.in1d) must not break
    spice.models.phoebe_utils / phoebe_model."""
    import importlib
    import importlib.abc
    import sys

    class _BrokenPhoebeFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "phoebe" or name.startswith("phoebe."):
                raise AttributeError("module 'numpy' has no attribute 'in1d'")
            return None

    saved = {name: sys.modules[name] for name in
             ("phoebe", "spice.models.phoebe_utils", "spice.models.phoebe_model")
             if name in sys.modules}
    for name in ("phoebe", "spice.models.phoebe_utils", "spice.models.phoebe_model"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "meta_path", [_BrokenPhoebeFinder(), *sys.meta_path])
    try:
        phoebe_utils = importlib.import_module("spice.models.phoebe_utils")
        assert phoebe_utils.PHOEBE_AVAILABLE is False
        assert phoebe_utils.phoebe is None
        importlib.import_module("spice.models.phoebe_model")
    finally:
        for name in ("spice.models.phoebe_utils", "spice.models.phoebe_model"):
            sys.modules.pop(name, None)
        sys.modules.update(saved)
