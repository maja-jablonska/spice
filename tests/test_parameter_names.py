"""to_parameters rejects names an emulator does not have and treats teff/Teff alike.

A misspelled or foreign name (e.g. ``logteff`` for a bundle that takes ``teff``) used to be
ignored silently, leaving that parameter at its default.
"""
import numpy as np
import pytest

from spice.spectrum import Blackbody, GaussianLineEmulator, PhysicalLineEmulator
from spice.spectrum.parameters import resolve_parameter_names


@pytest.mark.parametrize("emulator", [Blackbody(), GaussianLineEmulator(), PhysicalLineEmulator()],
                         ids=lambda e: type(e).__name__)
def test_teff_and_Teff_are_interchangeable(emulator):
    a = np.asarray(emulator.to_parameters({"Teff": 6100.}))
    b = np.asarray(emulator.to_parameters({"teff": 6100.}))
    np.testing.assert_allclose(a, b)
    assert float(a[0]) == pytest.approx(6100.)


@pytest.mark.parametrize("emulator", [Blackbody(), GaussianLineEmulator(), PhysicalLineEmulator()],
                         ids=lambda e: type(e).__name__)
def test_unknown_name_raises(emulator):
    with pytest.raises(ValueError, match="Unknown parameter name"):
        emulator.to_parameters({"logteff": 3.8})


def test_physical_line_emulator_keeps_other_parameters():
    p = np.asarray(PhysicalLineEmulator().to_parameters({"teff": 6000., "logg": 4.0, "mu": 0.5}))
    np.testing.assert_allclose(p, [6000., 4.0, 0.5])


def test_resolve_parameter_names_maps_temperature_and_aliases():
    out = resolve_parameter_names({"Teff": 5000., "m/h": -0.5}, ["teff", "logg", "feh"], aliases={"m/h": "feh"})
    assert out == {"teff": 5000., "feh": -0.5}


def test_resolve_parameter_names_lists_accepted_names():
    with pytest.raises(ValueError) as err:
        resolve_parameter_names({"t_eff": 5000.}, ["teff", "logg"])
    assert "['teff', 'logg']" in str(err.value)
