import warnings
import numpy as np
from spice.spectrum.spectrum_emulator import SpectrumEmulator
from overrides import override
from typing import Any, List, Optional
from numpy.typing import ArrayLike

from juliacall import Main as jl

jl.seval("using Korg")
Korg = jl.Korg

metallicity_labels = ["default_metals_h", "metals/h", "me/h",
                      "[metals/h]", "[me/h]", "metallicity", "metals_h", "me_h"]
alpha_labels = ["default_alpha_H", "alpha/h", "[alpha/h]", "alpha_h"]

LOG_G_NAMES: List[str] = ['logg', 'log_g', 'log g',
                          'surface gravity',
                          'surface_gravity']
TEFF_NAMES: List[str] = ['teff', 't_eff', 't eff',
                         'effective_temperature',
                         'effective temperature',]

elements_90 = [
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe",
    "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se",
    "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo",
    "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce",
    "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",
    "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb",
    "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U"
]
label_names = ["teff", "logg"] + elements_90


class KorgSpectrumEmulator(SpectrumEmulator[ArrayLike]):
    def __init__(self, line_list: Optional[Any] = None):
        if line_list is None:
            line_list = Korg.get_GALAH_DR3_linelist()
        self.line_list = line_list

    @override
    def stellar_parameter_names(self) -> ArrayLike:
        return label_names

    @override
    def to_parameters(self, parameters: ArrayLike = None) -> ArrayLike:
        parameters = parameters or {}
        if isinstance(parameters, (list, tuple)):
            parameters = np.array(parameters)
        if isinstance(parameters, np.ndarray):
            if len(parameters.shape) > 1:
                if parameters.shape[1] != len(label_names):
                    raise ValueError(
                        f"Parameters must have {len(label_names)} columns, got {parameters.shape[1]}")
                return parameters
            return parameters

        elif isinstance(parameters, dict):
            a_x_params = {}
            default_metals_h = 0.0
            default_alpha_h = 0.0

            if any([name.lower() in parameters.keys() for name in metallicity_labels]):
                default_metals_h = parameters[next(
                    name for name in parameters.keys() if name.lower() in metallicity_labels)]
                if not any([name.lower() in parameters.keys() for name in alpha_labels]):
                    default_alpha_h = default_metals_h
                else:
                    default_alpha_h = parameters[next(
                        name for name in parameters.keys() if name.lower() in alpha_labels)]

            for name in parameters.keys():
                if name.lower() not in [l.lower() for l in label_names]:
                    warnings.warn(f"Unknown parameter: {name}")
                elif name.lower() in [l.lower() for l in elements_90]:
                    a_x_params[name] = parameters[name]

            if not any([name.lower() in parameters.keys() for name in TEFF_NAMES]):
                teff = 5777.0000327042135
            else:
                teff = parameters[next(
                    name for name in parameters.keys() if name.lower() in TEFF_NAMES)]

            if not any([name.lower() in parameters.keys() for name in LOG_G_NAMES]):
                logg = 4.44
            else:
                logg = parameters[next(
                    name for name in parameters.keys() if name.lower() in LOG_G_NAMES)]

            return np.array([teff, logg] + list(Korg.format_A_X(default_metals_h, default_alpha_h, a_x_params)))
        else:
            raise ValueError(
                f"Parameters must be an array, list, or dictionary, got {type(parameters)}")

    def _interpolate_atm(self, parameters: ArrayLike) -> ArrayLike:
        return Korg.interpolate_marcs(parameters[0], parameters[1], parameters[2:])

    def _synthesize_spectrum(self, log_wavelengths: ArrayLike, parameters: ArrayLike, **kwargs) -> ArrayLike:
        start_lambda = 10**log_wavelengths[0]
        end_lambda = 10**log_wavelengths[-1]
        step_lambda = (end_lambda - start_lambda) / len(log_wavelengths)
        return Korg.synthesize(self._interpolate_atm(parameters),
                               self.line_list,
                               parameters[2:],
                               start_lambda,
                               end_lambda,
                               step_lambda,
                               **kwargs)

    @override
    def flux(self, log_wavelengths: ArrayLike, parameters: ArrayLike) -> ArrayLike:
        return np.array(self._synthesize_spectrum(log_wavelengths, parameters).flux)

    @override
    def intensity(self, log_wavelengths: ArrayLike, mu: float, parameters: ArrayLike) -> ArrayLike:
        return np.array(self._synthesize_spectrum(log_wavelengths, parameters, np.array([mu])).intensity(mu))
