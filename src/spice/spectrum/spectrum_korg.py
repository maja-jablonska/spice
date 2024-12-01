from functools import lru_cache
import pickle
import warnings
import jax
import numpy as np
from spice.spectrum.spectrum_emulator import SpectrumEmulator
from overrides import override
from typing import List, Optional
from numpy.typing import ArrayLike
import jax.numpy as jnp
from huggingface_hub import hf_hub_download
import joblib

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
FILENAME = "korg_grid_interpolator.pkl"
DEFAULT_CACHE_PATH = '~/.spice_cache'

class KorgSpectrumEmulator(SpectrumEmulator[ArrayLike]):
    def __init__(self, cache_path: str = DEFAULT_CACHE_PATH, model_path: Optional[str] = None):
        if model_path is not None:
            try:
                model_path = pickle.load(model_path)
            except Exception as e:
                raise RuntimeError(f"Failed to load model from path: {e}. If no model_path is provided, the model will be downloaded from the Hugging Face Hub.")
        
        else:
            self.cache_path = cache_path
            # Check if the file exists in cache
            try:
                self.model = joblib.load(hf_hub_download(repo_id=REPO_ID, filename=FILENAME))
            except Exception as e:
                raise RuntimeError(f"Failed to load model from cache: {e}")
        
        self._v_interpolate = jax.vmap(lambda p, mu, w: self.model(jnp.concatenate([p, jnp.atleast_1d(mu), jnp.atleast_1d(w)])), in_axes=(None, None, 0))

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
        return self._v_interpolate(parameters, mu, log_wavelengths).repeat(2, axis=1)
