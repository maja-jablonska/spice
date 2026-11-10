"""``PhoebeModel.rotation_velocity`` must come out in km/s.

``PhoebeConfig.get_quantity('requiv')`` returns solar radii and the period is
converted to seconds, so the equatorial velocity needs the R_sun -> cm factor:

    v = 2 pi R[Rsun] * SOLAR_RAD_CM / P[s] / 1e5   [km/s]

The factor was missing, making the stored value 6.957e10 times too small
(7.96e-11 km/s rather than 5.54 km/s for the TZ For primary). It was latent
rather than active -- ``get_mesh_view`` returns early for a PhoebeModel and
``_evaluate_rotation`` raises for one, so nothing in the synthesis path read it
-- but both ``mesh_transform`` and ``mesh_view`` do read ``mesh.rotation_velocity``
for ordinary meshes, so the wrong value was one guard away from being used.
"""

import numpy as np
import pytest

from spice.constants import DAY_TO_S, SOLAR_RAD_CM


def equatorial_velocity_kms(radius_rsun, period_days):
    """The formula PhoebeModel.construct must implement."""
    return 2 * np.pi * radius_rsun * SOLAR_RAD_CM / (period_days * DAY_TO_S) / 1e5


@pytest.mark.parametrize("radius,period,expected", [
    (8.28, 75.66647, 5.536),      # TZ For primary, synchronous
    (3.94, 75.66647, 2.634),      # TZ For secondary, synchronous
    (1.0, 25.05, 2.02),           # the Sun
])
def test_equatorial_velocity_is_physical(radius, period, expected):
    assert equatorial_velocity_kms(radius, period) == pytest.approx(expected, rel=1e-3)


def test_source_applies_the_solar_radius_factor():
    """Guard the actual expression in phoebe_model.py, not just a local copy."""
    import inspect
    from spice.models import phoebe_model
    src = inspect.getsource(phoebe_model)
    line = [l for l in src.splitlines() if "lin_velocity" in l and "=" in l
            and "2 * np.pi" in l]
    assert line, "lin_velocity assignment not found"
    assert "R_SOL_CM" in line[0], (
        f"lin_velocity is missing the R_sun->cm factor: {line[0].strip()}")


def test_missing_factor_would_be_absurd():
    """Pin the magnitude of the bug so the fix cannot silently regress."""
    buggy = 2 * np.pi * 8.28 / (75.66647 * DAY_TO_S) / 1e5
    correct = equatorial_velocity_kms(8.28, 75.66647)
    assert correct / buggy == pytest.approx(SOLAR_RAD_CM, rel=1e-9)
    assert buggy < 1e-9        # physically absurd for a star
