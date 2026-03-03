from overrides import override

from spice.spectrum.zarr_grid_loader import ZarrGridLoader
from .spectrum_emulator import SpectrumEmulator
from numpy.typing import ArrayLike
import jax.numpy as jnp
import numpy as np


class ZarrGridInterpolator(SpectrumEmulator[ArrayLike]):
    _FALLBACK_LIMB_DARKENING_COEFF = 0.6

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
        """Calculate the intensity for given wavelengths and mus.

        Args:
            log_wavelengths (ArrayLike): either wavelengths [angstrom] or log10(wavelengths [angstrom])
            mu (float): cosine of the angle between the star's radius and the line of sight
            parameters (ArrayLike): stellar parameters

        Returns:
            ArrayLike: intensities corresponding to passed wavelengths [erg/cm2/s/angstrom]
        """
        params_arr = self.to_parameters(parameters)
        if params_arr.ndim != 1:
            raise ValueError(
                f"intensity expects a single parameter vector, got shape {params_arr.shape}"
            )
        mu_scalar = float(np.asarray(mu).reshape(-1)[0])

        parameter_lookup = {
            name: params_arr[idx] for idx, name in enumerate(self.stellar_parameter_names)
        }
        query = []
        for axis_name in self.grid_loader.axis_names:
            if axis_name == "mu":
                query.append(mu_scalar)
            else:
                if axis_name not in parameter_lookup:
                    raise ValueError(
                        f"Missing required parameter '{axis_name}' for this grid."
                    )
                query.append(parameter_lookup[axis_name])

        spectrum, continuum = self.grid_loader.interpolate_spectrum_and_continuum(
            jnp.asarray(query)
        )

        # If the grid has no explicit mu axis, use the per-model mu_selected metadata
        # as a reference mu and apply a simple linear limb-darkening rescaling.
        if "mu" not in self.grid_loader.axis_names:
            mu_selected_cube = getattr(self.grid_loader, "mu_selected_cube_jnp", None)
            if mu_selected_cube is not None:
                ref_mu = self.grid_loader._interpolate_spectrum_jit(
                    jnp.asarray(query), mu_selected_cube
                )[0]
                requested_mu = jnp.clip(jnp.asarray(mu_scalar), 0.0, 1.0)
                reference_mu = jnp.clip(ref_mu, 0.0, 1.0)
                u = self._FALLBACK_LIMB_DARKENING_COEFF
                ld_requested = 1.0 - u * (1.0 - requested_mu)
                ld_reference = 1.0 - u * (1.0 - reference_mu)
                scale = ld_requested / jnp.maximum(ld_reference, 1e-8)
                spectrum = spectrum * scale
                continuum = continuum * scale

        # Preserve backwards compatibility with callers that pass wavelengths
        # directly (e.g. 3800..4200) while still supporting log10 wavelengths.
        query_wavelengths = np.asarray(log_wavelengths)
        if np.nanmax(query_wavelengths) > 10.0:
            target_wavelengths = query_wavelengths
        else:
            target_wavelengths = np.power(10.0, query_wavelengths)

        grid_wavelengths = np.asarray(self.grid_loader.wavelength_jnp)
        line_interp = np.interp(target_wavelengths, grid_wavelengths, np.asarray(spectrum))
        continuum_interp = np.interp(
            target_wavelengths, grid_wavelengths, np.asarray(continuum)
        )
        return jnp.stack([jnp.asarray(line_interp), jnp.asarray(continuum_interp)], axis=1)
