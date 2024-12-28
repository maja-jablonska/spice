import pickle
import jax
import numpy as np
from spice.spectrum.spectrum_emulator import SpectrumEmulator
from overrides import override
from typing import List, Optional
from numpy.typing import ArrayLike
import jax.numpy as jnp
from huggingface_hub import hf_hub_download
import h5py

from spice.spectrum.utils import linear_multivariate_interpolation

METALLICITY_NAMES = ["default_metals_h", "metals/h", "me/h",
                      "[metals/h]", "[me/h]", "metallicity", "metals_h", "me_h"]

LOG_G_NAMES: List[str] = ['logg', 'log_g', 'log g',
                          'surface gravity',
                          'surface_gravity']
TEFF_NAMES: List[str] = ['teff', 't_eff', 't eff',
                         'effective_temperature',
                         'effective temperature',]

label_names = ["teff", "logg", "m/h"]

REPO_ID = "mjablonska/spice_korg_interpolator"
REGULAR_FILENAME = "regular_grid.h5"
SMALL_FILENAME = "small_grid.h5"
DEFAULT_CACHE_PATH = '~/.spice_cache'


@jax.jit
def _interpolate_spectrum(log10_wavelengths, continuum_wavelengths, intensities, continuum_intensities, all_parameters, parameters, log10_wavelength):
    wave_idx = jnp.searchsorted(log10_wavelengths, log10_wavelength)
    wave_indices = jnp.clip(jnp.array([wave_idx - 1, wave_idx]), 0, len(log10_wavelengths) - 1)
    
    continuum_wave_idx = jnp.searchsorted(continuum_wavelengths, log10_wavelength)
    continuum_wave_indices = jnp.clip(jnp.array([continuum_wave_idx - 1, continuum_wave_idx]), 0, len(continuum_wavelengths) - 1)
    
    repeated_params = jnp.repeat(all_parameters, 2, axis=0)
    repeated_log10_wavelengths = jnp.tile(log10_wavelengths[wave_indices],
                                            (all_parameters.shape[0], 1)).reshape((-1, 1))
    repeated_continuum_wavelengths = jnp.tile(continuum_wavelengths[continuum_wave_indices],
                                            (all_parameters.shape[0], 1)).reshape((-1, 1))
    
    params_with_wavelength = jnp.hstack([parameters, log10_wavelength]).reshape(1, -1)
    
    return jnp.concatenate([
        linear_multivariate_interpolation(
            jnp.hstack([repeated_params, repeated_log10_wavelengths]),
            intensities[:, [wave_indices]].flatten(),
            params_with_wavelength
        ),
        linear_multivariate_interpolation(
            jnp.hstack([repeated_params, repeated_continuum_wavelengths]),
            continuum_intensities[:, [continuum_wave_indices]].flatten(),
            params_with_wavelength
        )
    ])

interpolate_spectrum = jax.vmap(_interpolate_spectrum, in_axes=(None, None, None, None, None, None, 0))


class KorgSpectrumEmulator(SpectrumEmulator[ArrayLike]):
    def __init__(self, cache_path: str = DEFAULT_CACHE_PATH, model_path: Optional[str] = None, grid_type: str = "small"):
        if model_path is not None:
            try:
                model_path = pickle.load(model_path)
            except Exception as e:
                raise RuntimeError(f"Failed to load model from path: {e}. If no model_path is provided, the model will be downloaded from the Hugging Face Hub.")
        
        else:
            print("Using default cache path:", cache_path)
            self.cache_path = cache_path
            # Check if the file exists in cache
            try:
                print("Attempting to download and load model from Hugging Face Hub...")
                filename = SMALL_FILENAME if grid_type == "small" else REGULAR_FILENAME
                model_path = hf_hub_download(repo_id=REPO_ID, filename=filename)
                print("Model downloaded to:", model_path)
                
                with h5py.File(model_path, 'r') as f:
                    print("Loading model parameters...")
                    self.parameters = jnp.array(f['parameters']) # [teff, logg, metallicity, mu]
                    print("Parameters shape:", self.parameters.shape)
                    
                    print("Loading model intensities...")
                    self.intensities = jnp.array(f['sp_intensity'])
                    print("Intensities shape:", self.intensities.shape)
                    
                    print("Loading wavelengths...")
                    self.log10_wavelengths = jnp.array(f['log10_sp_wave'])
                    print("Wavelengths shape:", self.log10_wavelengths.shape)
                    
                    print("Loading continuum intensities...")
                    self.continuum_intensities = jnp.array(f['sp_no_lines_intensity'])
                    print("Continuum intensities shape:", self.continuum_intensities.shape)
                    
                    print("Loading continuum wavelengths...")
                    self.continuum_wavelengths = jnp.array(f['log10_sp_no_lines_wave'])
                    print("Continuum wavelengths shape:", self.continuum_wavelengths.shape)
                
                print("Model loaded successfully")
            except Exception as e:
                print("Error loading model:", str(e))
                raise RuntimeError(f"Failed to load model from cache: {e}")

    @property
    def parameter_names(self) -> ArrayLike:
        return label_names

    @property
    @override
    def stellar_parameter_names(self) -> ArrayLike:
        return label_names

    @override
    def to_parameters(self, parameters: ArrayLike = None) -> ArrayLike:
        parameters = parameters or {}
        if isinstance(parameters, (list, tuple)):
            parameters = jnp.array(parameters)
        if isinstance(parameters, np.ndarray):
            if len(parameters.shape) > 1:
                if parameters.shape[1] != len(label_names):
                    raise ValueError(
                        f"Parameters must have {len(label_names)} columns, got {parameters.shape[1]}")
                return parameters
            return parameters

        elif isinstance(parameters, dict):
            # Extract Teff from various possible keys
            teff = None
            for key in TEFF_NAMES:
                if key in parameters:
                    teff = parameters[key]
                    break
            if teff is None:
                teff = 5777.0
            
            logg = None
            for key in LOG_G_NAMES:
                if key in parameters:
                    logg = parameters[key]
                    break
            if logg is None:
                logg = 4.44
            
            mh = None
            for key in METALLICITY_NAMES:
                if key in parameters:
                    mh = parameters[key]
                    break
            if mh is None:
                mh = 0.0
            
            return jnp.array([teff, logg, mh])
        else:
            raise ValueError(
                f"Parameters must be an array, list, or dictionary, got {type(parameters)}")

    @override
    def flux(self, log_wavelengths: ArrayLike, parameters: ArrayLike) -> ArrayLike:
        raise NotImplementedError("KorgSpectrumEmulator does not support flux")

    @override
    def intensity(self, log_wavelengths: ArrayLike, mu: float, parameters: ArrayLike) -> ArrayLike:
        return interpolate_spectrum(self.log10_wavelengths, self.continuum_wavelengths, self.intensities, self.continuum_intensities, self.parameters,
                                    jnp.hstack([parameters, jnp.atleast_1d(mu)]), log_wavelengths)
