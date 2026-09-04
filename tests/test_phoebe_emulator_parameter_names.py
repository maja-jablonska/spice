"""Emulator parameter names must still resolve to PHOEBE's per-element columns.

Bundles qualify their parameter names with the atmosphere grid they were
trained on: the Aug-2026 ``RozanskiT/TPayne-spice-harps`` retrain exposes
``marcs_teff`` / ``marcs_logg`` instead of ``teff`` / ``logg``. Those names
matched none of ``TEFF_NAMES`` / ``LOG_G_NAMES``, so ``PhoebeModel.construct``
raised -- and a caller who silenced it by supplying them through
``parameter_values`` got a *constant* temperature and gravity across the mesh,
silently throwing away PHOEBE's gravity-darkened structure. That defeats the
entire purpose of importing a PHOEBE mesh into SPICE.
"""

import numpy as np
import pytest

from spice.models.phoebe_model import (
    LOG_G_NAMES, TEFF_NAMES, _canonical_parameter_name)


@pytest.mark.parametrize("name,expected", [
    ("marcs_teff", "teff"), ("marcs_logg", "logg"),
    ("MARCS_TEFF", "teff"), ("atlas_teff", "teff"),
    ("phoenix_logg", "logg"), ("kurucz_teff", "teff"),
])
def test_grid_prefix_is_stripped(name, expected):
    assert _canonical_parameter_name(name) == expected


@pytest.mark.parametrize("name", ["teff", "logg", "feh", "vmicro", "a", "mu"])
def test_unprefixed_names_are_untouched(name):
    assert _canonical_parameter_name(name) == name.lower()


@pytest.mark.parametrize("name", ["spot_teff", "secondary_logg", "delta_teff"])
def test_unrelated_prefixes_are_not_collapsed(name):
    """Only known atmosphere-grid families are stripped.

    A name like ``spot_teff`` must NOT silently become the mesh's teff column.
    """
    assert _canonical_parameter_name(name) == name.lower()


def test_marcs_names_now_reach_the_mesh_columns():
    """The regression proper: the bundle's names must hit the teff/logg branches."""
    assert _canonical_parameter_name("marcs_teff") in TEFF_NAMES
    assert _canonical_parameter_name("marcs_logg") in LOG_G_NAMES
    # ...and the raw names did not, which is what made the failure silent.
    assert "marcs_teff" not in TEFF_NAMES
    assert "marcs_logg" not in LOG_G_NAMES
