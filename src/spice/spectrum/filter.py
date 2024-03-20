from abc import ABC, abstractmethod, abstractproperty
from typing import Optional
import re
import matplotlib.pyplot as plt
from spice.spectrum.utils import wavelengths_to_frequencies, intensity_wavelengths_to_hz, intensity_Jy_to_erg

try:
    import jax.numpy as jnp
    from jax.typing import ArrayLike
    import jax
    
except ImportError:
    raise ImportError("Please install JAX to use SPICE.")


@jax.jit
def _filter_responses(wavelengths: ArrayLike,
                      sample_wavelengths: ArrayLike,
                      sample_responses: ArrayLike) -> ArrayLike:
    return jnp.interp(wavelengths, sample_wavelengths, sample_responses)


class Filter(ABC):
    def __init__(self,
                 transmission_curve: ArrayLike,
                 ab_zeropoint: float = 3.631*1e-20,
                 name: Optional[str] = None):
        self.__transmission_curve = transmission_curve
        
        tc_freq = wavelengths_to_frequencies(transmission_curve[0])
        tc_freq_mask = jnp.argsort(tc_freq)
        self.__transmission_curve_freq = jnp.vstack([tc_freq[tc_freq_mask],
                                                     transmission_curve[1][tc_freq_mask]])
        
        self.__ab_zeropoint = ab_zeropoint
        self.__name = name or re.sub( r"([A-Z])", r" \1", type(self).__name__).split()

    @property
    def transmission_curve_wavelengths(self) -> ArrayLike:
        return self.__transmission_curve
    
    @property
    def transmission_curve_frequencies(self) -> ArrayLike:
        return self.__transmission_curve_freq
    
    @property
    def ab_zeropoint(self) -> float:
        return self.__ab_zeropoint

    @property
    def name(self) -> str:
        return self.__name
        
    def __str__(self):
        return self.name
    
    def filter_responses_for_wavelengths(self, wavelengths: ArrayLike) -> ArrayLike:
        return _filter_responses(wavelengths, self.transmission_curve_wavelengths[0], self.transmission_curve_wavelengths[1])
    
    def filter_responses_for_frequencies(self, wavelengths: ArrayLike) -> ArrayLike:
        return _filter_responses(wavelengths, self.transmission_curve_frequencies[0], self.transmission_curve_frequencies[1])

    def plot_filter_responses_for_wavelengths(self,
                                              wavelengths: ArrayLike,
                                              intensities: ArrayLike,
                                              plot_kwargs: Optional[dict] = None):
        plot_kwargs = plot_kwargs or {}
        if "label" not in plot_kwargs:
            plot_kwargs["label"] = self.name
        
        responses = self.filter_responses_for_wavelengths(wavelengths)
        plt.plot(wavelengths, intensities, color='black')
        plt.plot(wavelengths, intensities*responses, **plot_kwargs)
        plt.gca().set_xlabel("Wavelength [$\\AA$]")
        plt.gca().set_ylabel("Intensity [$\\frac{erg}{s\\ cm^3}$]")
        plt.legend()

    def plot_filter_responses_for_frequencies(self,
                                              frequencies: ArrayLike,
                                              intensities: ArrayLike,
                                              plot_kwargs: Optional[dict] = None):
        plot_kwargs = plot_kwargs or {}
        if "label" not in plot_kwargs:
            plot_kwargs["label"] = self.name
        
        responses = self.filter_responses_for_frequencies(frequencies)
        plt.plot(frequencies, intensities, color='black')
        plt.plot(frequencies, intensities*responses, **plot_kwargs)
        plt.gca().set_xlabel("Frequency [Hz]")
        plt.gca().set_ylabel("Intensity [$\\frac{erg}{s\\ cm^3}$]")
        plt.legend()

