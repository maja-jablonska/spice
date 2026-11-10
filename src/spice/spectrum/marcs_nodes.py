"""MARCS model-atmosphere node values used by the TPayne-spice-harps training grids.

Since the cno-v2 retrain (2026-10) the HARPS intensity bundle takes the [Fe/H] of the
MARCS structure (``marcs_fe_h``) as an input in addition to the synthesis metallicity
(``feh``). In the training grid ``marcs_fe_h`` is always one of :data:`MARCS_FEH_NODES`;
the network, like ``marcs_teff`` and ``marcs_logg``, also accepts values between them.

Two ways of supplying it are offered (see :func:`resolve_marcs_fe_h`):

* ``"feh"`` (default): ``marcs_fe_h = feh``. The atmosphere metallicity follows the
  synthesis metallicity continuously, which is the analogue of passing a continuous
  Teff as ``marcs_teff``: the network interpolates between the MARCS structures,
  the result is smooth (and differentiable) in [Fe/H].
* ``"snap"``: the nearest MARCS node, reproducing the grid's own behaviour for most
  rows. The production pipeline snaps (Teff, log g, [Fe/H], vmicro) jointly over the
  models that exist, so where MARCS has gaps (about 5 % of grid rows) its node differs
  from the nearest-in-[Fe/H] one; exact replication of a grid row needs the stored
  value, not this helper.
"""
from __future__ import annotations

from typing import Any

import numpy as np

MARCS_FEH_NODES = np.array([-5.0, -4.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
MARCS_FE_H_MODES = ("feh", "snap")


def snap_marcs_feh(feh: Any) -> Any:
    """Nearest MARCS [Fe/H] node (ties go to the lower node). Works on floats, numpy
    arrays and jax arrays (jit-compatible; the gradient is zero almost everywhere)."""
    try:
        import jax.numpy as jnp
        if not isinstance(feh, (float, int, np.ndarray, np.generic)):
            nodes = jnp.asarray(MARCS_FEH_NODES, dtype=jnp.result_type(feh, jnp.float32))
            x = jnp.asarray(feh)
            return nodes[jnp.argmin(jnp.abs(x[..., None] - nodes), axis=-1)]
    except ImportError:  # pragma: no cover - jax is a hard dependency of spice
        pass
    x = np.asarray(feh, dtype=float)
    out = MARCS_FEH_NODES[np.argmin(np.abs(x[..., None] - MARCS_FEH_NODES), axis=-1)]
    return float(out) if np.ndim(out) == 0 else out


def resolve_marcs_fe_h(feh: Any, mode: str = "feh") -> Any:
    """The ``marcs_fe_h`` input to use for a synthesis metallicity ``feh``."""
    if mode == "feh":
        return feh
    if mode == "snap":
        return snap_marcs_feh(feh)
    raise ValueError(f"marcs_fe_h mode must be one of {MARCS_FE_H_MODES}, got {mode!r}")
