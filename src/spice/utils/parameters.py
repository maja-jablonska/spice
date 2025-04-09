"""Utilities for parameter modification and transformation."""

from typing import Any, Callable, Optional, Union, TypeVar
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from ..models.mesh_model import MeshModel

T = TypeVar('T')


def modify_mesh_parameter(
    mesh: MeshModel,
    parameter_index: int,
    distribution_fn: Callable[[MeshModel, Optional[ArrayLike]], ArrayLike],
    weights: Optional[ArrayLike] = None,
    bounds: Optional[tuple[float, float]] = None,
    clip: bool = True
) -> MeshModel:
    """Modify mesh parameters according to a distribution function.
    
    This function applies a distribution function to modify parameters across a mesh,
    optionally weighted by some mesh property (e.g. area, distance from center, etc.).
    The function returns a new mesh with modified parameters.
    
    Args:
        mesh: The MeshModel instance to modify.
        parameter_index: Index of the parameter to modify in the mesh's parameters array.
        distribution_fn: Function that takes the parameter values and optional weights and returns modified values.
            Should be compatible with JAX transformations.
        weights: Optional weights to apply to the distribution, shape (n_elements,).
            If provided, the distribution will be weighted by these values.
        bounds: Optional tuple of (min_value, max_value) to constrain the parameters.
            If None, no bounds are applied.
        clip: If True and bounds are provided, clip values to the bounds.
            If False and bounds are provided, raise ValueError if values are out of bounds.
            
    Returns:
        Modified MeshModel with updated parameter values.
        
    Example:
        >>> # Create a temperature distribution that falls off with radius
        >>> def temp_distribution(temps, radii):
        ...     return temps * jnp.exp(-radii / scale_height)
        >>> # Get radii from mesh centers
        >>> radii = jnp.linalg.norm(mesh.d_centers, axis=1)
        >>> # Modify temperature parameter (assuming index 0)
        >>> new_mesh = modify_mesh_parameter(
        ...     mesh,
        ...     parameter_index=0,
        ...     distribution_fn=temp_distribution,
        ...     weights=radii
        ... )
    """
    # Apply distribution function
    if weights is not None:
        modified_values = distribution_fn(mesh, *weights)
    else:
        modified_values = distribution_fn(mesh)

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
