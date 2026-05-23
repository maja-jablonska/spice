"""Pickle-compat shim — implementation lives in :mod:`spice.cepheid_bundles`.

Old pickle files (e.g. ``cepheid_bundles_vmicro.pkl``) embed the FQN
``cepheid_bundles.CepheidBundle`` from before this module was moved into
the spice package. This shim re-exports the names so those pickles still
load when this directory is on ``sys.path``.

New code should import from ``spice.cepheid_bundles`` directly.
"""
from spice.cepheid_bundles import (
    CepheidBundle,
    LineSpectra,
    build_bundle,
    apply_phase_params,
    simulate_line_spectra,
    save_pickle,
    load_pickle,
)

__all__ = [
    "CepheidBundle",
    "LineSpectra",
    "build_bundle",
    "apply_phase_params",
    "simulate_line_spectra",
    "save_pickle",
    "load_pickle",
]
