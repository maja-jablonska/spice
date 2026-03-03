from overrides import override

from spice.spectrum.zarr_grid_loader import ZarrGridLoader
from .spectrum_emulator import SpectrumEmulator
from numpy.typing import ArrayLike
import jax.numpy as jnp
import numpy as np


class ZarrGridInterpolator(SpectrumEmulator[ArrayLike]):
    def __init__(self, grid_path: str, chunk_rows: int = 500):
        self.grid_loader = ZarrGridLoader(grid_path, chunk_rows=chunk_rows)
        self.parameter_names = self.grid_loader.axis_names
    
    @property
    @override
    def stellar_parameter_names(self) -> ArrayLike:
        """Get labels of stellar parameters (no geometry-related parameters, e.g. mu)

        Returns:
            ArrayLike:
        """
        return [pm for pm in self.parameter_names if pm != "mu"]
    
    @property
    def solar_parameters(self) -> ArrayLike:
        return jnp.array([4.44, 5777., 0.0, 0.0])
    
    @override
    def to_parameters(self, parameters: ArrayLike | dict | None = None) -> ArrayLike:
        """
        Convert flexible user input to an appropriate parameter array.
        Accepts array-like objects or dicts mapping names to values.
        """
        if parameters is None:
            return self.solar_parameters

        # Accept dict mapping parameter names to values
        if isinstance(parameters, dict):
            # Build parameter vector in order of self.stellar_parameter_names,
            # defaulting missing values to solar_parameters.
            arr = [parameters.get(
                name, self.solar_parameters[self.stellar_parameter_names.index(name)]
            ) for name in self.stellar_parameter_names]
            return jnp.array(arr)

        # Try array-like next, after dict
        # Use type() to avoid TypeError if jnp.ndarray is not a type
        array_types = (list, tuple, np.ndarray)
        try:
            import jax.numpy as jnp_mod
            array_types = array_types + (jnp_mod.ndarray,)
        except Exception:
            pass

        if isinstance(parameters, array_types):
            arr = jnp.array(parameters)
            if arr.ndim == 1:
                # Single instance: must have correct length
                if arr.shape[0] != len(self.stellar_parameter_names):
                    raise ValueError(
                        f"Parameters vector must have length {len(self.stellar_parameter_names)}, got {arr.shape[0]}"
                    )
            elif arr.ndim == 2:
                # Batch shape: (N, n_params)
                if arr.shape[1] != len(self.stellar_parameter_names):
                    raise ValueError(
                        f"Parameters must have {len(self.stellar_parameter_names)} columns, got {arr.shape[1]}"
                    )
            else:
                raise ValueError(
                    f"Parameters must be a 1D or 2D array, got shape {arr.shape}"
                )
            return arr

        raise ValueError(
            f"Parameters must be a dict mapping names to values, or a list, tuple, or array. Got {type(parameters)}"
        )

    @override
    def flux(self, log_wavelengths: ArrayLike, parameters: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    @override
    def intensity(self, log_wavelengths: ArrayLike, mu: float, parameters: ArrayLike) -> ArrayLike:
        """Calculate the intensity for given wavelengths and mus

        Args:
            log_wavelengths (ArrayLike): [log(angstrom)]
            mu (float): cosine of the angle between the star's radius and the line of sight
            parameters (ArrayLike): stellar parameters

        Returns:
            ArrayLike: intensities corresponding to passed wavelengths [erg/cm2/s/angstrom]
        """
        return self.grid_loader.interpolate_intensity(log_wavelengths, mu, parameters)
