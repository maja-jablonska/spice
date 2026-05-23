"""Small shared helpers for the benchmark scripts.

Single source of truth for the handful of trivial, byte-identical helpers that
were copy-pasted across benchmark_grid_loading.py, benchmark_spice_components.py,
benchmark_occlusion.py, and benchmark_memory.py. Kept dependency-light (numpy +
argparse only) so importing it never drags in the heavier benchmark modules.

(The larger same-named helpers — make_plots, sweep_wavelengths, build_parser,
build_static_spectrum — are intentionally NOT shared: they are genuinely
different per benchmark despite the shared names.)
"""
import argparse

import numpy as np


def _resolve_precisions(value):
    """``('x32',)`` / ``('x64',)`` / ``('x32', 'x64')`` for ``'x32'``/``'x64'``/``'both'``."""
    return ("x32", "x64") if value == "both" else (value,)


def _planck_like(wavelengths_angstrom, teff):
    """A cheap Planck-shaped curve in erg/s/cm^2/A units (only used to fill the
    synthetic grid / static spectrum with something non-trivial)."""
    h = 6.62607015e-27
    c = 2.99792458e10
    k = 1.380649e-16
    w_cm = np.asarray(wavelengths_angstrom) * 1e-8
    return (2 * h * c ** 2 / w_cm ** 5 / (np.exp(h * c / (w_cm * k * teff)) - 1.0)) * 1e-8


def _parse_in_memory(value):
    if value in ("true", "1", "yes"):
        return (True,)
    if value in ("false", "0", "no"):
        return (False,)
    if value == "both":
        return (False, True)
    raise argparse.ArgumentTypeError(f"expected true/false/both, got {value!r}")
