"""Interpolate an intensity emulator along one parameter between training-grid nodes.

Some pretrained bundles memorize their training grid along a sparsely sampled
parameter instead of interpolating: e.g. ``RozanskiT/TPayne-spice-harps``
reproduces its 250 K Teff nodes on plateaus separated by sharp (~8 K) sigmoid
transitions at the node midpoints. On the nodes the output matches the
training spectra almost exactly, so smooth behaviour can be restored by
evaluating the wrapped emulator at the two bracketing nodes and interpolating
linearly in log10(intensity) — at the cost of two evaluations per call.
"""
import math
from typing import Any

import numpy as np
import jax.numpy as jnp
from jaxtyping import ArrayLike


class GridNodeInterpolatedEmulator:
    """Drop-in intensity-emulator wrapper that interpolates between grid nodes.

    Every attribute other than ``intensity`` (``to_parameters``,
    ``stellar_parameter_names``, ...) is delegated to the wrapped emulator, so
    instances can be passed anywhere an intensity emulator is expected (e.g.
    ``simulate_observed_flux``).

    Args:
        emulator: The intensity emulator to wrap. Must expose
            ``stellar_parameter_names``, ``min_stellar_parameters``,
            ``max_stellar_parameters`` and ``intensity(log_wavelengths, mu,
            parameters)``.
        parameter: Name of the stellar parameter to interpolate along.
        spacing: Node spacing of the training grid in that parameter's units.
        origin: A value the grid nodes are anchored to (any node value, or 0
            when nodes sit at integer multiples of ``spacing``).

    Values whose bracketing interval would leave the emulator's declared
    parameter bounds are linearly extrapolated from the two nearest in-bounds
    nodes.
    """

    def __init__(self, emulator: Any, parameter: str = "teff",
                 spacing: float = 250.0, origin: float = 0.0):
        names = list(emulator.stellar_parameter_names)
        if parameter not in names:
            raise ValueError(
                f"Parameter {parameter!r} not in the emulator's "
                f"stellar_parameter_names: {names!r}"
            )
        if spacing <= 0:
            raise ValueError(f"spacing must be positive, got {spacing}")
        self._emulator = emulator
        self._param_index = names.index(parameter)
        self._spacing = float(spacing)
        self._origin = float(origin)
        lo = float(np.asarray(emulator.min_stellar_parameters)[self._param_index])
        hi = float(np.asarray(emulator.max_stellar_parameters)[self._param_index])
        # Lowest / highest node index inside the declared bounds. Guard the
        # degenerate case of fewer than two in-bounds nodes.
        self._k_min = math.ceil((lo - self._origin) / self._spacing)
        self._k_max = math.floor((hi - self._origin) / self._spacing)
        if self._k_max - self._k_min < 1:
            raise ValueError(
                f"Fewer than two grid nodes with spacing {spacing} inside the "
                f"emulator bounds [{lo}, {hi}] for parameter {parameter!r}."
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._emulator, name)

    def intensity(self, log_wavelengths: ArrayLike, mu: float,
                  parameters: ArrayLike) -> ArrayLike:
        parameters = jnp.asarray(parameters)
        value = parameters[self._param_index]
        k = jnp.clip(
            jnp.floor((value - self._origin) / self._spacing),
            self._k_min, self._k_max - 1,
        )
        v0 = self._origin + k * self._spacing
        w = (value - v0) / self._spacing
        p0 = parameters.at[self._param_index].set(v0)
        p1 = parameters.at[self._param_index].set(v0 + self._spacing)
        i0 = self._emulator.intensity(log_wavelengths, mu, p0)
        i1 = self._emulator.intensity(log_wavelengths, mu, p1)
        # Linear in log10: intensities scale quasi-exponentially with Teff, so
        # log-space interpolation is much closer than linear over a 250 K gap.
        eps = 1e-30
        log_i = (1.0 - w) * jnp.log10(i0 + eps) + w * jnp.log10(i1 + eps)
        return jnp.power(10.0, log_i)
