"""Utilities for parameter modification and transformation."""

from typing import Any, Callable, Optional, Union, TypeVar
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from ..models.mesh_model import MeshModel

T = TypeVar('T')


def modify_mesh_parameter_from_array(
    mesh: MeshModel,
    parameter_index: int,
    values: ArrayLike
) -> MeshModel:
    """Modify mesh parameters from an array of values.
    
    This function modifies the parameters of a mesh model at a specific index
    using directly provided values. It returns a new mesh with the updated parameters.
    
    Args:
        mesh: The MeshModel instance to modify.
        parameter_index: Index of the parameter to modify in the mesh's parameters array.
        values: Array of values to assign to the parameter at the specified index.
            Must match the shape of the first dimension of mesh.parameters.
            
    Returns:
        Modified MeshModel with updated parameter values.
        
    Example:
        >>> # Set all temperature values (assuming index 0) to a constant
        >>> temperatures = jnp.ones(mesh.parameters.shape[0]) * 5000
        >>> new_mesh = modify_mesh_parameter_from_array(
        ...     mesh,
        ...     parameter_index=0,
        ...     values=temperatures
        ... )
    """
    return _modify_mesh_parameter_impl(mesh, parameter_index, values)


def modify_mesh_parameter(
    mesh: MeshModel,
    parameter_index: int,
    distribution_fn: Callable[[MeshModel, Any], ArrayLike],
    *args
) -> MeshModel:
    """Modify mesh parameters according to a distribution function.
    
    This function applies a distribution function to modify parameters across a mesh,
    optionally using additional arguments (e.g. area, distance from center, etc.).
    The function returns a new mesh with modified parameters.
    
    Args:
        mesh: The MeshModel instance to modify.
        parameter_index: Index of the parameter to modify in the mesh's parameters array.
        distribution_fn: Function that takes the parameter values and optional arguments and returns modified values.
            Should be compatible with JAX transformations.
        *args: Optional positional arguments to pass to the distribution function.
            
    Returns:
        Modified MeshModel with updated parameter values.
        
    Example:
        >>> # Create a temperature distribution that falls off with radius
        >>> def temp_distribution(mesh, scale_height):
        ...     radii = jnp.linalg.norm(mesh.d_centers, axis=1)
        ...     return mesh.parameters[:, 0] * jnp.exp(-radii / scale_height)
        >>> # Modify temperature parameter (assuming index 0)
        >>> new_mesh = modify_mesh_parameter(
        ...     mesh,
        ...     parameter_index=0,
        ...     distribution_fn=temp_distribution,
        ...     0.1  # scale_height as positional argument
        ... )
    """
    # Apply distribution function with args
    modified_values = distribution_fn(mesh, *args)

    return _modify_mesh_parameter_impl(mesh, parameter_index, modified_values)


@jax.jit
def _modify_mesh_parameter_impl(
    mesh: MeshModel,
    parameter_index: int,
    modified_values: ArrayLike
) -> MeshModel:
    """Implementation of modify_mesh_parameter that can be JIT-compiled."""
    # Get current parameter values
    new_parameters = mesh.parameters.at[:, parameter_index].set(modified_values)
    
    return mesh._replace(parameters=new_parameters)

@jax.jit
def constant_luminosity_temperature(mp, original_radius, original_teff):
    """Calculate effective temperature that maintains constant luminosity as radius changes.
    It assumes a uniform temperature distribution.
    
    This function implements the temperature-radius relationship for constant luminosity:
    T ∝ R^(-1/2), derived from L ∝ R²T⁴ = constant.
    
    Args:
        mp: MeshModel or similar object containing radii information.
        original_radius: The reference radius value.
        original_teff: The effective temperature at the reference radius.
        
    Returns:
        float: The new effective temperature that maintains constant luminosity
              given the current radius of the model.
              
    Example:
        >>> # Calculate new temperature for a pulsating star
        >>> new_temp = constant_luminosity_temperature(
        ...     mesh_model, 
        ...     original_radius=1.0,  # Solar radii
        ...     original_teff=5800.0  # Kelvin
        ... )
    """
    
    # Get the current radius (which varies due to pulsation)
    current_radius = mp.radii.mean()  # Taking mean radius as representative
    
    # Calculate the new temperature using T ∝ R^(-1/2) relation
    # T_new = T_original * (R_original/R_new)^(1/2)
    scaling_factor = (original_radius / current_radius) ** 0.5
    new_teff = original_teff * scaling_factor
    return new_teff

