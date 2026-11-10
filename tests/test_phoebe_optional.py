"""Smoke tests must fail on broken core imports and skip absent extras."""
import pytest


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
    phoebe = pytest.importorskip("phoebe", reason="PHOEBE is an optional extra")
    from spice.models.phoebe_utils import PHOEBE_AVAILABLE

    assert PHOEBE_AVAILABLE
    bundle = phoebe.default_binary()
    assert "primary" in bundle.components
    assert "secondary" in bundle.components
