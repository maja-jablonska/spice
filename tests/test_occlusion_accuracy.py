"""Quantitative accuracy tests for mesh-vs-mesh occlusion.

``spice.models.mesh_view.resolve_occlusion`` is what ``binary.py`` uses for every
eclipse, transit and occultation, but until now it was only covered by
qualitative assertions ("some occlusion happened"). That let two bugs through:

* the occulter's *far* hemisphere was included in the neighbour search, so every
  overlap was counted twice and ``jnp.clip`` silently saturated the doubled sum
  at the face area -- over-deepening eclipses by 5.3% / 2.6% / 1.2% of the
  blocked flux fraction at 1280 / 5120 / 20480 faces (worse on coarse meshes,
  which reads deceptively like discretization error converging away);
* filtering those faces *after* the k-nearest-neighbour query instead of before
  it wasted about half of every neighbour slot, starving the search and
  under-counting occlusion by 2-3%.

The tests below pin the blocked *flux* fraction against a direct 2-D quadrature
of the limb-darkened lune, which is independent of the mesh entirely.

Geometry conventions (easy to get backwards when staging a mesh by hand):
with ``los_vector = [0, 0, -1]`` an element is visible when ``mus > 0``, and
``resolve_occlusion`` only applies when the occulter's ``cast_to_los(center)``
exceeds the occluded body's -- which puts the occulter at *positive* z here.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from spice.models.mesh_model import IcosphereModel
from spice.models.mesh_view import resolve_occlusion

LOS = jnp.array([0.0, 0.0, -1.0])
R_ECLIPSED = 3.94      # R_sun, TZ For secondary
R_OCCULTER = 8.28      # R_sun, TZ For primary
U_LD = 0.744           # linear limb darkening of the eclipsed body


def _intensity(mu):
    return 1.0 - U_LD * (1.0 - mu)


def _reference_blocked_fraction(d: float, n: int = 1500) -> float:
    """Blocked fraction of the limb-darkened disc, by direct 2-D quadrature."""
    x = np.linspace(-R_ECLIPSED, R_ECLIPSED, n)
    X, Y = np.meshgrid(x, x)
    r2 = X ** 2 + Y ** 2
    on_disc = r2 <= R_ECLIPSED ** 2
    mu = np.sqrt(np.clip(1.0 - r2 / R_ECLIPSED ** 2, 0.0, 1.0))
    intensity = _intensity(mu) * on_disc
    hidden = on_disc & (((X - d) ** 2 + Y ** 2) <= R_OCCULTER ** 2)
    return float(intensity[hidden].sum() / intensity.sum())


def _body(n_vertices: int, radius: float, mass: float, center):
    mesh = IcosphereModel.construct(
        n_vertices, radius, mass, jnp.array([5000.0, 3.0]), ["teff", "logg"]
    )
    return mesh._replace(center=jnp.asarray(center), los_vector=LOS)


def _spice_blocked_fraction(n_vertices: int, d: float, n_neighbors: int = 20) -> float:
    """Blocked flux fraction through the production occlusion path."""
    eclipsed = _body(n_vertices, R_ECLIPSED, 1.958, [0.0, 0.0, 0.0])
    # Positive z puts the occulter in front under this LOS convention.
    occulter = _body(n_vertices, R_OCCULTER, 2.057, [d, 0.0, 50.0])
    resolved = resolve_occlusion(eclipsed, occulter, n_neighbors=n_neighbors)

    mus = np.asarray(resolved.mus)
    front = mus > 0
    intensity = _intensity(mus)
    cast = np.asarray(resolved.cast_areas)
    visible = np.asarray(resolved.visible_cast_areas)
    unocculted = float(np.sum(cast * intensity * front))
    return 1.0 - float(np.sum(visible * intensity * front)) / unocculted


# Separations spanning a shallow grazing overlap, the TZ For configuration, and
# a deep partial eclipse.
@pytest.mark.parametrize("d", [11.0, 8.947, 6.0])
def test_blocked_fraction_matches_quadrature(d):
    """Coarse mesh must already reproduce the analytic blocked fraction."""
    reference = _reference_blocked_fraction(d)
    got = _spice_blocked_fraction(1000, d)          # 642 vertices / 1280 faces
    assert reference > 0.0, "test geometry produces no occultation"
    assert got == pytest.approx(reference, rel=0.02), (
        f"separation {d}: blocked fraction {got:.5f} vs quadrature {reference:.5f} "
        f"({100 * (got / reference - 1):+.2f}%)"
    )


def test_refinement_does_not_degrade_accuracy():
    """Error must shrink (or hold) with refinement, never grow.

    The far-hemisphere double-count produced errors that *decreased* with
    resolution while remaining large, and neighbour starvation produced errors
    that *grew* with resolution. Both violate this.
    """
    d = 8.947
    reference = _reference_blocked_fraction(d)
    coarse = abs(_spice_blocked_fraction(1000, d) / reference - 1.0)
    fine = abs(_spice_blocked_fraction(5000, d) / reference - 1.0)
    assert fine < 0.01, f"fine mesh error {100 * fine:.2f}% too large"
    assert fine <= coarse + 0.002, (
        f"refinement made it worse: {100 * coarse:.2f}% -> {100 * fine:.2f}%"
    )


def test_far_hemisphere_is_not_counted_twice():
    """Occlusion must never exceed the geometric area that can be hidden.

    With the occulter's far side included, occluded area roughly doubled and was
    only kept plausible by the clip to ``cast_areas``; the blocked flux fraction
    then overshot the analytic value by several percent.
    """
    d = 8.947
    reference = _reference_blocked_fraction(d)
    got = _spice_blocked_fraction(1000, d)
    assert got < reference * 1.02, (
        f"blocked fraction {got:.5f} exceeds quadrature {reference:.5f} -- "
        "far-hemisphere faces are being counted as occluders"
    )


def test_result_is_insensitive_to_neighbour_count():
    """A sane n_neighbors must not change the answer.

    Filtering back-facing occluders after the neighbour query (rather than
    excluding them from it) halves the usable slots, making the result depend
    strongly on ``n_neighbors`` -- 2-3% low at k=10.
    """
    d = 8.947
    small = _spice_blocked_fraction(1000, d, n_neighbors=10)
    large = _spice_blocked_fraction(1000, d, n_neighbors=40)
    assert small == pytest.approx(large, rel=0.01), (
        f"k=10 gives {small:.5f} but k=40 gives {large:.5f}: the neighbour "
        "search is starved"
    )
