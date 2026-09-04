"""A passband must not respond outside its tabulated wavelength range.

``_filter_responses`` interpolates a filter's transmission curve onto the
caller's wavelength grid with ``jnp.interp``, which *clamps to the edge values*
outside the sample range unless ``left``/``right`` are given. Most curves in
``spice.spectrum.filter`` start and end at 0.0, so clamping was a no-op for
them -- but :class:`Bolometric` is flat 1.0 across 1-30000 A, so every
wavelength beyond 30000 A was silently given full transmission and folded into
the integral.

Found by comparing SPICE against PHOEBE for TZ Fornacis: with a grid reaching
into the infrared the bolometric light ratio L2/L1 came out 0.619 against
PHOEBE's 0.444 and an analytic Planck integral's 0.443, because the band had
effectively swallowed the whole grid.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from spice.spectrum.filter import Bolometric, JohnsonCousinsV, Stromgrenb


def test_bolometric_has_no_response_outside_its_range():
    """The regression: a flat-topped curve must still be bounded by its support."""
    filt = Bolometric()
    edges = np.asarray(filt.transmission_curve_wavelengths[0])
    lo, hi = float(edges.min()), float(edges.max())

    outside = jnp.array([lo / 2.0, hi * 1.5, hi * 10.0])
    assert np.allclose(np.asarray(filt.filter_responses_for_wavelengths(outside)), 0.0)

    inside = jnp.array([lo + 0.25 * (hi - lo), lo + 0.75 * (hi - lo)])
    assert np.all(np.asarray(filt.filter_responses_for_wavelengths(inside)) > 0.0)


@pytest.mark.parametrize("filter_cls", [Bolometric, JohnsonCousinsV, Stromgrenb])
def test_response_is_zero_beyond_support(filter_cls):
    filt = filter_cls()
    edges = np.asarray(filt.transmission_curve_wavelengths[0])
    probes = jnp.array([float(edges.min()) - 1.0, float(edges.max()) + 1.0])
    assert np.allclose(np.asarray(filt.filter_responses_for_wavelengths(probes)), 0.0)


def test_extending_the_grid_does_not_change_a_passband_integral():
    """Padding the wavelength grid must not change the flux a filter admits.

    This is the property the bug violated: widening the grid used to widen the
    effective bandpass for Bolometric.
    """
    filt = Bolometric()
    edges = np.asarray(filt.transmission_curve_wavelengths[0])
    lo, hi = float(edges.min()), float(edges.max())

    narrow = jnp.linspace(lo, hi, 4000)
    wide = jnp.linspace(lo, hi * 4.0, 16000)

    def admitted(grid):
        g = np.asarray(grid)
        # A flat spectrum makes the integral read off the band's width directly.
        return float(np.trapezoid(
            np.asarray(filt.filter_responses_for_wavelengths(grid)), g))

    assert admitted(wide) == pytest.approx(admitted(narrow), rel=1e-3)
