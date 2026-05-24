"""Shared floating-point dtype helpers.

SPICE follows the ``jax_enable_x64`` flag everywhere: when 64-bit mode is on
arrays are built as float64, otherwise float32. These helpers centralise that
choice so it isn't reimplemented (slightly differently) in every module.
"""

import jax
import jax.numpy as jnp
import numpy as np


def float_dtype():
    """Return the JAX float dtype matching the active ``jax_enable_x64`` flag."""
    return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


def np_float_dtype():
    """Return the NumPy float dtype matching the active ``jax_enable_x64`` flag."""
    return np.float64 if jax.config.jax_enable_x64 else np.float32
