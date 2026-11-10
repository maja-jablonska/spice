"""PHOEBE-imported models must agree with PHOEBE's own radial velocities.

PHOEBE's uvw frame has +w pointing toward the observer, so an element moving
toward us has ``vws > 0`` but a *negative* radial velocity. Two places got this
backwards:

* ``PhoebeModel`` used ``los_vector = [0, 0, -1]``, which -- given
  ``cast_to_los(v, los) = -dot(v, los)`` and ``velocities = -center_velocities``
  -- yields ``+vws``, i.e. PHOEBE's RV with the sign flipped;
* ``PhoebeConfig.get_radial_velocities`` returned raw ``vws``.

Measured on a default PHOEBE binary at quadrature: the rv dataset gives
-126.483 km/s while the visible-area-weighted ``vws`` is +126.482.

Flipping the LOS is safe for ``PhoebeModel`` specifically because it is the only
consumer of ``los_vector`` there: ``mus`` and ``cast_vertices``/``cast_centers``/
``cast_areas`` all come straight from PHOEBE, ``los_z`` is not implemented, and
the model carries no ``occluded_areas`` -- so SPICE-computed occlusion cannot
reach the synthesis either.
"""

import numpy as np
import pytest

phoebe = pytest.importorskip("phoebe", reason="PHOEBE is an optional extra")

from spice.models import PhoebeModel
from spice.models.phoebe_utils import Component, PhoebeConfig


@pytest.fixture(scope="module")
def quadrature_bundle():
    """A circular, edge-on binary sampled at quadrature, where |RV| is maximal."""
    phoebe.logger(clevel="ERROR")
    b = phoebe.default_binary()
    b.set_value("period@binary", 2.0)
    b.set_value("incl@binary", 90.0)
    b.set_value("q", 1.0)
    b.set_value("sma@binary", 10.0)
    t = 0.5  # quadrature for P = 2 d with t0 = 0
    b.add_dataset("rv", times=[t], dataset="rv01")
    b.add_dataset(
        "mesh", times=[t], dataset="mesh01",
        columns=["vus", "vvs", "vws", "us", "vs", "ws", "mus",
                 "visibilities", "areas", "teffs", "loggs"],
    )
    b.run_compute(irrad_method="none", ltte=False)
    return b, t


def _phoebe_rv(bundle):
    return float(np.asarray(bundle.get_value("rvs@primary@rv01@model"))[0])


def test_los_velocities_match_phoebe_rv(quadrature_bundle):
    """SPICE's los_velocities must reproduce PHOEBE's RV in sign and magnitude."""
    bundle, t = quadrature_bundle
    reference = _phoebe_rv(bundle)
    assert abs(reference) > 10.0, "test geometry should give a large RV"

    model = PhoebeModel.construct(
        PhoebeConfig(bundle, mesh_dataset_name="mesh01"), t,
        parameter_names=["teff", "logg"], component=Component.PRIMARY,
    )
    los_v = np.asarray(model.los_velocities)
    weights = np.clip(np.asarray(model.mus), 0.0, None)
    mean_los_v = float(np.sum(los_v * weights) / np.sum(weights))

    assert np.sign(mean_los_v) == np.sign(reference), (
        f"sign disagrees with PHOEBE: SPICE {mean_los_v:+.3f} vs "
        f"PHOEBE {reference:+.3f} km/s"
    )
    # The weighting differs slightly from PHOEBE's (area * mu * visibility),
    # so compare loosely -- the point of this test is the convention.
    assert mean_los_v == pytest.approx(reference, rel=0.05)


def test_get_radial_velocities_matches_phoebe_rv(quadrature_bundle):
    """The accessor must return an RV, not raw vws (which has the opposite sign)."""
    bundle, t = quadrature_bundle
    reference = _phoebe_rv(bundle)
    config = PhoebeConfig(bundle, mesh_dataset_name="mesh01")

    rvs = np.asarray(config.get_radial_velocities(t, Component.PRIMARY))
    mus = np.asarray(config.get_mus(t, Component.PRIMARY))
    weights = np.clip(mus, 0.0, None)
    mean_rv = float(np.sum(rvs * weights) / np.sum(weights))

    assert np.sign(mean_rv) == np.sign(reference), (
        f"get_radial_velocities returns the negated RV: {mean_rv:+.3f} vs "
        f"PHOEBE {reference:+.3f} km/s"
    )


def test_los_vector_is_only_used_for_velocities():
    """Guard the assumption that makes the flipped LOS safe.

    If PhoebeModel ever grows an ``occluded_areas`` field or computes its
    projections from ``los_vector``, the LOS choice would start affecting
    eclipse geometry and this convention would need revisiting.
    """
    assert not hasattr(PhoebeModel, "occluded_areas"), (
        "PhoebeModel gained occluded_areas: the +w LOS may now affect occlusion"
    )
    with pytest.raises(NotImplementedError):
        PhoebeModel.los_z.fget(None)
