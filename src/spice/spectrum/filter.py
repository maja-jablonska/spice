from abc import ABC
from typing import Optional
import re
import matplotlib.pyplot as plt
from spice.spectrum.utils import wavelengths_to_frequencies
import numpy as np
from importlib import resources as impresources
from . import filter_data

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



# Filter information: http://dx.doi.org/10.1086/132749

class BesselU(Filter):
    def __init__(self):
        transmission_curve = jnp.array([
            [3000., 3050., 3100., 3150., 3200., 3250., 3300., 3350., 3400.,
            3450., 3500., 3550., 3600., 3650., 3700., 3750., 3800., 3850.,
            3900., 3950., 4000., 4050., 4100., 4150., 4200.],
            [0.   , 0.016, 0.068, 0.167, 0.287, 0.423, 0.56 , 0.673, 0.772,
            0.841, 0.905, 0.943, 0.981, 0.993, 1.   , 0.989, 0.916, 0.804,
            0.625, 0.423, 0.238, 0.114, 0.051, 0.019, 0.   ]
        ])
        super().__init__(transmission_curve, name='Bessel U')


class BesselB(Filter):
    def __init__(self):
        transmission_curve = jnp.array([
            [3600., 3700., 3800., 3900., 4000., 4100., 4200., 4300., 4400.,
            4500., 4600., 4700., 4800., 4900., 5000., 5100., 5200., 5300.,
            5400., 5500., 5600.],
            [0.   , 0.03 , 0.134, 0.567, 0.92 , 0.978, 1.   , 0.978, 0.935,
            0.853, 0.74 , 0.64 , 0.536, 0.424, 0.325, 0.235, 0.15 , 0.095,
            0.043, 0.009, 0.   ]
        ])
        super().__init__(transmission_curve, name='Bessel B')
        
class BesselV(Filter):
    def __init__(self):
        transmission_curve = jnp.array([
            [4700., 4800., 4900., 5000., 5100., 5200., 5300., 5400., 5500.,
            5600., 5700., 5800., 5900., 6000., 6100., 6200., 6300., 6400.,
            6500., 6600., 6700., 6800., 6900., 7000.],
            [0.   , 0.03 , 0.163, 0.458, 0.78 , 0.967, 1.   , 0.973, 0.898,
            0.792, 0.684, 0.574, 0.461, 0.359, 0.27 , 0.197, 0.135, 0.081,
            0.045, 0.025, 0.017, 0.013, 0.009, 0.   ]
        ])
        super().__init__(transmission_curve, name='Bessel V')
        
        
class BesselR(Filter):
    def __init__(self):
        transmission_curve = jnp.array([
            [5500., 5600., 5700., 5800., 5900., 6000., 6100., 6200., 6300.,
            6400., 6500., 6600., 6700., 6800., 6900., 7000., 7100., 7200.,
            7300., 7400., 7500., 8000., 8500., 9000.],
            [0.  , 0.23, 0.74, 0.91, 0.98, 1.  , 0.98, 0.96, 0.93, 0.9 , 0.86,
            0.81, 0.78, 0.72, 0.67, 0.61, 0.56, 0.51, 0.46, 0.4 , 0.35, 0.14,
            0.03, 0.  ]
        ])
        super().__init__(transmission_curve, name='Bessel R')
        

class BesselI(Filter):
    def __init__(self):
        transmission_curve = jnp.array([
            [7000., 7100., 7200., 7300., 7400., 7500., 7600., 7700., 7800.,
            7900., 8000., 8100., 8200., 8300., 8400., 8500., 8600., 8700.,
            8800., 8900., 9000., 9100., 9200.],
            [0.   , 0.024, 0.232, 0.555, 0.785, 0.91 , 0.965, 0.985, 0.99 ,
            0.995, 1.   , 1.   , 0.99 , 0.98 , 0.95 , 0.91 , 0.86 , 0.75 ,
            0.56 , 0.33 , 0.15 , 0.03 , 0.   ]
        ])
        super().__init__(transmission_curve, name='Bessel I')
        


class GaiaG(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'GaiaG.dat')).T
        super().__init__(transmission_curve, name='Gaia G')


class GaiaRP(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'GaiaRP.dat')).T
        super().__init__(transmission_curve, name='Gaia RP')


class GaiaBP(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'GaiaBP.dat')).T
        super().__init__(transmission_curve, name='Gaia BP')