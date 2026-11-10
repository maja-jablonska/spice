"""Warnings for wavelengths that an emulator or a passband does not cover.

Two ways to get a silently wrong number:

* asking an emulator for wavelengths outside the range it was trained on. A
  neural emulator returns *something* there, and nothing in its output says it
  is an extrapolation;
* integrating a passband over a wavelength grid that covers only part of its
  response. The band average is then taken over the covered part alone, which
  biases the magnitude by however much the flux in the missing part differs
  from the mean over the rest (e.g. ~9 per cent of Hipparcos H_p lies redward
  of 6910 A, the red edge of the HARPS intensity emulator: 20-30 mmag for an
  F-G star, and 13 mmag of that changes with temperature).

Both checks run only on concrete arrays. Under ``jax.jit`` the wavelengths are
tracers and the checks are skipped, so call the public entry points (or
:meth:`~spice.spectrum.filter.Filter.wavelength_coverage`) once outside the
traced function to see the warnings.
"""
import warnings
from typing import Any, Optional, Tuple

import numpy as np

# Fraction of a passband's response that may lie off the wavelength grid
# before a warning is issued. Johnson V loses 0.2 per cent to the HARPS
# emulator's limits (< 1 mmag) and stays silent; Johnson B loses 0.8 per cent
# (up to ~2 mmag) and warns.
PASSBAND_COVERAGE_TOLERANCE = 0.005

# Relative slack on an emulator's wavelength limits: float32 round trips
# through log10 must not trigger the warning at the exact edge.
_RANGE_SLACK = 1e-6


class WavelengthCoverageWarning(UserWarning):
    """Wavelengths outside an emulator's range, or a passband the grid does not cover."""


def _concrete(values: Any) -> Optional[np.ndarray]:
    """``values`` as a float numpy array, or None for a JAX tracer."""
    try:
        return np.asarray(values, dtype=np.float64)
    except Exception:          # TracerArrayConversionError / ConcretizationTypeError under jit
        return None


def _emulator_of(fn: Any) -> Any:
    """The object an intensity/flux callable belongs to: a bound method's
    ``__self__``, looking through ``LdBoundIntensity`` wrappers."""
    for _ in range(4):
        owner = getattr(fn, "__self__", None)
        if owner is not None:
            return owner
        inner = getattr(fn, "fn", None)
        if inner is None:
            return None
        fn = inner
    return None


def emulator_wavelength_range(fn: Any) -> Optional[Tuple[float, float]]:
    """``(min, max)`` in Angstrom that the emulator behind ``fn`` was trained on, or None."""
    owner = _emulator_of(fn)
    return getattr(owner, "wavelength_range", None) if owner is not None else None


def warn_if_outside_emulator_range(fn: Any, log_wavelengths: Any, stacklevel: int = 3) -> None:
    """Warn when any of ``log_wavelengths`` (log10 Angstrom) falls outside the
    training range of the emulator that ``fn`` (e.g. ``emulator.intensity``) belongs to."""
    rng = emulator_wavelength_range(fn)
    if rng is None:
        return
    lw = _concrete(log_wavelengths)
    if lw is None or lw.size == 0:
        return
    wl = np.power(10.0, lw.ravel())
    lo, hi = rng
    outside = (wl < lo * (1.0 - _RANGE_SLACK)) | (wl > hi * (1.0 + _RANGE_SLACK))
    if outside.any():
        name = type(_emulator_of(fn)).__name__
        warnings.warn(
            f"{int(outside.sum())} of {wl.size} requested wavelengths ({wl.min():.0f}-{wl.max():.0f} A) lie "
            f"outside the {lo:.0f}-{hi:.0f} A range {name} was trained on; its output there is an "
            "extrapolation and should not be used.",
            WavelengthCoverageWarning, stacklevel=stacklevel)


def warn_if_passband_uncovered(filter: Any, wavelengths: Any, stacklevel: int = 3) -> None:
    """Warn when more than ``PASSBAND_COVERAGE_TOLERANCE`` of ``filter``'s
    response lies outside the span of ``wavelengths`` (Angstrom)."""
    wl = _concrete(wavelengths)
    if wl is None or wl.size < 2:
        return
    missing = 1.0 - filter.wavelength_coverage(wl)
    if missing > PASSBAND_COVERAGE_TOLERANCE:
        warnings.warn(
            f"{100 * missing:.1f}% of the {filter.name} passband response lies outside the wavelength grid "
            f"({wl.min():.0f}-{wl.max():.0f} A). The magnitude is the band average over the covered part only; "
            "synthesise the whole passband, with an intensity model that covers it, for an unbiased value.",
            WavelengthCoverageWarning, stacklevel=stacklevel)
