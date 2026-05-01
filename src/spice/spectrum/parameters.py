from typing import Any, Dict, List
import inspect as _inspect
import time as _time
import sys as _sys
import warnings
from jax.typing import ArrayLike
import jax.numpy as jnp


def parameter_helper(interpolator, parameter_values: Dict[str, Any] = None) -> ArrayLike:
    """Convert passed values to the accepted parameters format

    Args:
        parameter_values (Dict[str, Any], optional): parameter values in the format of {'parameter_name': value}. Unset parameters will be set to solar values.
        relative (bool, optional): if True, the values are treated as relative to the solar values for the abundaces 

    Returns:
        ArrayLike:
    """
    
    if not parameter_values:
        return interpolator.solar_parameters

    # Initialize parameters with solar values
    if isinstance(parameter_values, dict):
        parameters = jnp.array(interpolator.solar_parameters)
        # Convert parameter names to indices for direct access
        parameter_indices = {label: i for i, label in enumerate(interpolator.stellar_parameter_names)}

        for label, value in parameter_values.items():
            idx = parameter_indices[label]
            parameters = parameters.at[idx].set(value)
    else:
        parameters = jnp.array(parameter_values)
    
    if not (jnp.all(parameters >= interpolator.min_stellar_parameters) and jnp.all(parameters <= interpolator.max_stellar_parameters)):
        warnings.warn("Possible exceeding parameter bonds - extrapolating.")
        
    return parameters