from typing import Any, Dict, Iterable, Mapping, Optional
import warnings
from jax.typing import ArrayLike
import jax.numpy as jnp


def resolve_parameter_names(parameter_values: Mapping[str, Any], accepted: Iterable[str],
                            aliases: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """Map the keys of a ``{name: value}`` dict onto an emulator's parameter names.

    ``teff`` and ``Teff`` (any capitalisation of ``teff``) are interchangeable: either
    selects the emulator's temperature parameter. ``aliases`` maps further alternative
    names onto accepted ones. Any other name the emulator does not have raises
    ``ValueError`` instead of being silently ignored (a misspelled name would otherwise
    leave that parameter at its default).

    Returns:
        Dict[str, Any]: the values keyed by the emulator's own parameter names.

    Raises:
        ValueError: for names that are not parameters of the emulator.
    """
    accepted = list(accepted)
    lookup = {name: name for name in accepted}
    teff_names = [name for name in accepted if name.lower() == "teff"]
    if teff_names:
        lookup.update({"teff": teff_names[0], "Teff": teff_names[0], "TEFF": teff_names[0]})
    for alias, target in (aliases or {}).items():
        if target in accepted:
            lookup.setdefault(alias, target)
    resolved, unknown = {}, []
    for key, value in parameter_values.items():
        name = lookup.get(key, lookup.get(key.lower()) if isinstance(key, str) and key.lower() == "teff" else None)
        if name is None:
            unknown.append(key)
        else:
            resolved[name] = value
    if unknown:
        raise ValueError(f"Unknown parameter name(s) {unknown}; this emulator's parameters are {accepted}"
                         " (teff and Teff are interchangeable).")
    return resolved


def parameter_helper(interpolator, parameter_values: Optional[Dict[str, Any]] = None) -> ArrayLike:
    """Convert passed values to the accepted parameters format.

    Args:
        interpolator: the spectrum emulator providing ``solar_parameters``,
            ``stellar_parameter_names`` and the min/max parameter bounds.
        parameter_values (Dict[str, Any], optional): parameter values in the format
            of {'parameter_name': value}. Unset parameters will be set to solar values.

    Returns:
        ArrayLike: the parameters as an array ordered by ``stellar_parameter_names``.
    """

    if parameter_values is None:
        return interpolator.solar_parameters

    # Initialize parameters with solar values
    if isinstance(parameter_values, dict):
        parameters = jnp.array(interpolator.solar_parameters)
        # Convert parameter names to indices for direct access
        parameter_indices = {label: i for i, label in enumerate(interpolator.stellar_parameter_names)}

        for label, value in resolve_parameter_names(parameter_values, interpolator.stellar_parameter_names).items():
            parameters = parameters.at[parameter_indices[label]].set(value)
    else:
        parameters = jnp.array(parameter_values)
    
    if not (jnp.all(parameters >= interpolator.min_stellar_parameters) and jnp.all(parameters <= interpolator.max_stellar_parameters)):
        warnings.warn("Parameters outside the source's bounds: grid interpolators clamp to the "
                      "nearest edge node, neural emulators extrapolate beyond their training range.")
        
    return parameters