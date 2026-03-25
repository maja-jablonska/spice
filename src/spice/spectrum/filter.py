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
    from jaxtyping import Array, Float

except ImportError:
    raise ImportError("Please install JAX to use SPICE.")


@jax.jit
def _filter_responses(wavelengths: ArrayLike,
                      sample_wavelengths: ArrayLike,
                      sample_responses: ArrayLike) -> ArrayLike:
    return jnp.interp(wavelengths, sample_wavelengths, sample_responses)


class Filter(ABC):
    def __init__(self,
                 transmission_curve: Float[Array, "2 n_samples"],
                 name: Optional[str] = None,
                 non_photonic: bool = False,
                 AB_zeropoint: float = -22.40788262039795,
                 ST_zeropoint: float = -21.10,
                 Vega_zeropoint: Optional[float] = None):
        """A class representing an astronomical filter with its transmission curve and properties.

        The Filter class provides functionality to handle astronomical filters, including their
        transmission curves in both wavelength and frequency space, and methods to apply the
        filter response to spectral data.

        Args:
            transmission_curve (Float[Array, "2 n_samples"]): A 2xN array containing wavelengths
                and corresponding transmission values. First row should be wavelengths in Angstroms,
                second row should be transmission values between 0 and 1.
            ab_zeropoint (float, optional): The AB magnitude system zero point in erg/s/cm^2/Hz.
                Defaults to 3.631e-20 (standard AB system zero point).
            name (Optional[str], optional): Name of the filter. If None, will be derived from class name.
                Defaults to None.
            non_photonic (bool, optional): Whether the filter is non-photonic and a different form of synthetic photometry equation is used. Defaults to False.
            AB_zeropoint (float, optional): The zeropoint of the filter in magnitudes in the AB system. Defaults to -22.407 (-2.5 log10(3.631 * 1e-20 * c [km/s] * 1e5)).
            ST_zeropoint (float, optional): The zeropoint of the filter in magnitudes in the ST system. Defaults to -21.10.
            Vega_zeropoint (float, optional): The zeropoint of the filter in magnitudes in the Vega system. Defaults to None.
        Properties:
            transmission_curve_wavelengths: The transmission curve in wavelength space
            transmission_curve_frequencies: The transmission curve in frequency space
            ab_zeropoint: The AB magnitude system zero point
            name: The name of the filter
            non_photonic: Whether the filter is non-photonic and a different form of synthetic photometry equation is used
        Methods:
            filter_responses_for_wavelengths: Get filter response values for given wavelengths
            filter_responses_for_frequencies: Get filter response values for given frequencies
            plot_filter_responses_for_wavelengths: Plot filter response curve with input spectrum
        """
        self.__transmission_curve = transmission_curve

        tc_freq = wavelengths_to_frequencies(transmission_curve[0])
        tc_freq_mask = jnp.argsort(tc_freq)
        self.__transmission_curve_freq = jnp.vstack([tc_freq[tc_freq_mask],
                                                     transmission_curve[1][tc_freq_mask]])

        self.__name = name or re.sub(r"([A-Z])", r" \1", type(self).__name__).split()
        self.__non_photonic = non_photonic
        self.__AB_zeropoint = AB_zeropoint
        self.__ST_zeropoint = ST_zeropoint
        self.__Vega_zeropoint = Vega_zeropoint
    @property
    def transmission_curve_wavelengths(self) -> ArrayLike:
        return self.__transmission_curve

    @property
    def transmission_curve_frequencies(self) -> ArrayLike:
        return self.__transmission_curve_freq

    @property
    def AB_zeropoint(self) -> float:
        return self.__AB_zeropoint
    
    @property
    def ST_zeropoint(self) -> float:
        return self.__ST_zeropoint
    
    @property
    def Vega_zeropoint(self) -> float:
        return self.__Vega_zeropoint

    @property
    def name(self) -> str:
        return self.__name

    @property
    def non_photonic(self) -> bool:
        return self.__non_photonic
    
    @property
    def zeropoint(self) -> float:
        return self.__zeropoint

    def __str__(self):
        return self.name

    def filter_responses_for_wavelengths(self, wavelengths: Float[Array, "n_wavelengths"]) -> Float[Array, "n_wavelengths"]:
        return _filter_responses(wavelengths, self.transmission_curve_wavelengths[0],
                                 self.transmission_curve_wavelengths[1])

    def filter_responses_for_frequencies(self, wavelengths: Float[Array, "n_wavelengths"]) -> Float[Array, "n_wavelengths"]:
        return _filter_responses(wavelengths, self.transmission_curve_frequencies[0],
                                 self.transmission_curve_frequencies[1])

    def plot_filter_responses_for_wavelengths(self,
                                              wavelengths: Float[Array, "n_wavelengths"],
                                              intensities: Float[Array, "n_wavelengths"],
                                              plot_kwargs: Optional[dict] = None):
        plot_kwargs = plot_kwargs or {}
        if "label" not in plot_kwargs:
            plot_kwargs["label"] = self.name

        responses = self.filter_responses_for_wavelengths(wavelengths)
        plt.plot(wavelengths, intensities, color='black')
        plt.plot(wavelengths, intensities * responses, **plot_kwargs)
        plt.gca().set_xlabel("Wavelength [$\\AA$]")
        plt.gca().set_ylabel("Intensity [$\\frac{erg}{s\\ cm^3}$]")
        plt.legend()

    def plot_filter_responses_for_frequencies(self,
                                              frequencies: Float[Array, "n_wavelengths"],
                                              intensities: Float[Array, "n_wavelengths"],
                                              plot_kwargs: Optional[dict] = None):
        plot_kwargs = plot_kwargs or {}
        if "label" not in plot_kwargs:
            plot_kwargs["label"] = self.name

        responses = self.filter_responses_for_frequencies(frequencies)
        plt.plot(frequencies, intensities, color='black')
        plt.plot(frequencies, intensities * responses, **plot_kwargs)
        plt.gca().set_xlabel("Frequency [Hz]")
        plt.gca().set_ylabel("Intensity [$\\frac{erg}{s\\ cm^3}$]")
        plt.legend()


# Filter information: Bessel & Murphy (2012)

class JohnsonCousinsU(Filter):
    def __init__(self):
        transmission_curve = jnp.array([
            [3000., 3050., 3100., 3150., 3200., 3250., 3300., 3350., 3400.,
             3450., 3500., 3550., 3600., 3650., 3700., 3750., 3800., 3850.,
             3900., 3950., 4000., 4050., 4100., 4150., 4200.],
            [0.000, 0.019, 0.068, 0.167, 0.278, 0.398, 0.522, 0.636, 0.735,
             0.813, 0.885, 0.940, 0.980, 1.000, 1.000, 0.974, 0.918, 0.802,
             0.590, 0.355, 0.194, 0.107, 0.046, 0.003, 0.000]
        ])
        super().__init__(transmission_curve, name='Johnson-Cousins U', AB_zeropoint=-22.40788262039795-0.771+0.01, ST_zeropoint=-21.10+0.142)


class JohnsonCousinsB(Filter):
    def __init__(self):
        transmission_curve = jnp.array([
            [3600., 3700., 3800., 3900., 4000., 4100., 4200., 4300., 4400.,
             4500., 4600., 4700., 4800., 4900., 5000., 5100., 5200., 5300.,
             5400., 5500., 5600.],
            [0.000, 0.031, 0.137, 0.584, 0.947, 1.000, 1.000, 0.957, 0.895,
             0.802, 0.682, 0.577, 0.474, 0.369, 0.278, 0.198, 0.125, 0.078,
             0.036, 0.008, 0.000]
        ])
        super().__init__(transmission_curve, name='Johnson-Cousins B', AB_zeropoint=-22.40788262039795+0.138-0.008, ST_zeropoint=-21.10+0.625)


class JohnsonCousinsV(Filter):
    def __init__(self):
        transmission_curve = jnp.array([
            [4700., 4800., 4900., 5000., 5100., 5200., 5300., 5400., 5500.,
             5600., 5700., 5800., 5900., 6000., 6100., 6200., 6300., 6400.,
             6500., 6600., 6700., 6800., 6900., 7000., 7100., 7200., 7300.,
             7400.],
            [0.000, 0.033, 0.176, 0.485, 0.811, 0.986, 1.000, 0.955, 0.865,
             0.750, 0.656, 0.545, 0.434, 0.334, 0.249, 0.180, 0.124, 0.075,
             0.041, 0.022, 0.014, 0.011, 0.008, 0.006, 0.004, 0.002, 0.001,
             0.000]
        ])
        super().__init__(transmission_curve, name='Johnson-Cousins V', AB_zeropoint=-22.40788262039795+0.023-0.003, ST_zeropoint=-21.10+0.019)


class JohnsonCousinsR(Filter):
    def __init__(self):
        transmission_curve = jnp.array([
            [5500., 5600., 5700., 5800., 5900., 6000., 6100., 6200., 6300.,
             6400., 6500., 6600., 6700., 6800., 6900., 7000., 7100., 7200.,
             7300., 7400., 7500., 7600., 7700., 7800., 7900., 8000., 8100.,
             8200., 8300., 8400., 8500., 8600., 8700., 8800., 8900., 9000.,
             9100.],
            [0.000, 0.247, 0.780, 0.942, 0.998, 1.000, 0.974, 0.940, 0.901,
             0.859, 0.814, 0.760, 0.713, 0.662, 0.605, 0.551, 0.497, 0.446,
             0.399, 0.350, 0.301, 0.257, 0.215, 0.177, 0.144, 0.116, 0.089,
             0.066, 0.051, 0.039, 0.030, 0.021, 0.014, 0.008, 0.006, 0.003,
             0.000]
        ])
        super().__init__(transmission_curve, name='Johnson-Cousins R', AB_zeropoint=-22.40788262039795-0.16-0.003, ST_zeropoint=-21.10-0.538)


class JohnsonCousinsI(Filter):
    def __init__(self):
        transmission_curve = jnp.array([
            [7000., 7100., 7200., 7300., 7400., 7500., 7600., 7700., 7800.,
             7900., 8000., 8100., 8200., 8300., 8400., 8500., 8600., 8700.,
             8800., 8900., 9000., 9100., 9200.],
            [0.000, 0.090, 0.356, 0.658, 0.865, 0.960, 1.000, 0.998, 0.985,
             0.973, 0.970, 0.958, 0.932, 0.904, 0.860, 0.810, 0.734, 0.590,
             0.392, 0.203, 0.070, 0.008, 0.000]
        ])
        super().__init__(transmission_curve, name='Johnson-Cousins I', AB_zeropoint=-22.40788262039795-0.402-0.002, ST_zeropoint=-21.10-1.220)


class HipparcosHp(Filter):
    def __init__(self):
        transmission_curve = jnp.array([
            [3400., 3500., 3600., 3700., 3800., 3900., 4000., 4100., 4200.,
             4300., 4400., 4500., 4600., 4700., 4800., 4900., 5000., 5100.,
             5200., 5300., 5400., 5500., 5600., 5700., 5800., 5900., 6000.,
             6100., 6200., 6300., 6400., 6500., 6600., 6700., 6800., 6900.,
             7000., 7100., 7200., 7300., 7400., 7500., 7600., 7700., 7800.,
             7900., 8000., 8100., 8200., 8300., 8400., 8500., 8600., 8700.,
             8800., 8900.],
            [0.000, 0.041, 0.072, 0.133, 0.199, 0.263, 0.347, 0.423, 0.508,
             0.612, 0.726, 0.813, 0.906, 0.966, 0.992, 1.000, 0.992, 0.978,
             0.951, 0.914, 0.880, 0.840, 0.797, 0.755, 0.712, 0.668, 0.626,
             0.583, 0.542, 0.503, 0.465, 0.429, 0.393, 0.359, 0.326, 0.293,
             0.260, 0.230, 0.202, 0.176, 0.152, 0.130, 0.112, 0.095, 0.081,
             0.068, 0.054, 0.042, 0.032, 0.024, 0.018, 0.014, 0.010, 0.006,
             0.002, 0.000]
        ])
        super().__init__(transmission_curve, name='Hipparcos Hp', AB_zeropoint=-22.40788262039795+0.022+0.008, ST_zeropoint=-21.10+0.074)


class TychoBT(Filter):
    def __init__(self):
        transmission_curve = jnp.array([
            [3500., 3550., 3600., 3650., 3700., 3750., 3800., 3850., 3900.,
             3950., 4000., 4050., 4100., 4150., 4200., 4250., 4300., 4350.,
             4400., 4450., 4500., 4550., 4600., 4650., 4700., 4750., 4800.,
             4850., 4900., 4950., 5000., 5050.],
            [0.000, 0.015, 0.063, 0.132, 0.220, 0.323, 0.439, 0.556, 0.664,
             0.751, 0.813, 0.853, 0.880, 0.904, 0.931, 0.960, 0.984, 1.000,
             0.969, 0.852, 0.674, 0.479, 0.309, 0.196, 0.131, 0.097, 0.077,
             0.056, 0.035, 0.015, 0.003, 0.000]
        ])
        super().__init__(transmission_curve, name='Tycho BT', AB_zeropoint=-22.40788262039795+0.09+0.01, ST_zeropoint=-21.10+0.672)
        

class TychoVT(Filter):
    def __init__(self):
        transmission_curve = jnp.array([
            [4550., 4600., 4650., 4700., 4750., 4800., 4850., 4900., 4950.,
             5000., 5050., 5100., 5150., 5200., 5250., 5300., 5350., 5400.,
             5450., 5500., 5550., 5600., 5650., 5700., 5750., 5800., 5850.,
             5900., 5950., 6000., 6050., 6100., 6150., 6200., 6250., 6300.,
             6350., 6400., 6450., 6500., 6550., 6600., 6650., 6700., 6750.],
            [0.000, 0.023, 0.119, 0.308, 0.540, 0.749, 0.882, 0.951, 0.981,
             0.997, 1.000, 0.992, 0.974, 0.946, 0.911, 0.870, 0.827, 0.784,
             0.738, 0.692, 0.645, 0.599, 0.553, 0.504, 0.458, 0.412, 0.368,
             0.324, 0.282, 0.245, 0.209, 0.178, 0.152, 0.129, 0.108, 0.092,
             0.078, 0.066, 0.055, 0.044, 0.036, 0.027, 0.017, 0.008, 0.000]
        ])
        super().__init__(transmission_curve, name='Tycho VT', AB_zeropoint=-22.40788262039795+0.044-0.007, ST_zeropoint=-21.10+0.115)


# # J/A+A/649/A3     Gaia Early Data Release 3 photometric passbands (Riello+, 2021)

class GaiaG(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'gaia_edr3_passband.dat')).T
        zp = np.loadtxt((impresources.files(filter_data) / 'gaia_edr3_zeropt.dat'))
        transmission_curve[transmission_curve == 99.99] = 0.
        super().__init__(jnp.array([transmission_curve[0]*10., transmission_curve[1]]), name='Gaia G', AB_zeropoint=zp[1, 0], Vega_zeropoint=zp[0, 0])


class GaiaBP(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'gaia_edr3_passband.dat')).T
        zp = np.loadtxt((impresources.files(filter_data) / 'gaia_edr3_zeropt.dat'))
        transmission_curve[transmission_curve == 99.99] = 0.
        super().__init__(jnp.array([transmission_curve[0]*10., transmission_curve[3]]), name='Gaia BP', AB_zeropoint=zp[1, 2], Vega_zeropoint=zp[0, 2])


class GaiaRP(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'gaia_edr3_passband.dat')).T
        zp = np.loadtxt((impresources.files(filter_data) / 'gaia_edr3_zeropt.dat'))
        transmission_curve[transmission_curve == 99.99] = 0.
        super().__init__(jnp.array([transmission_curve[0]*10., transmission_curve[5]]), name='Gaia RP', AB_zeropoint=zp[1, 4], Vega_zeropoint=zp[0, 4])
        
        
# Sartoretti et al. (2022)
class GaiaRVS(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) /'grvsfilter.csv'), skiprows=1, delimiter=',')
        super().__init__(jnp.array([transmission_curve[0]*10., transmission_curve[1]]), name='Gaia RVS', AB_zeropoint=21.317)

# Doi et al. (2010)

sdss = np.loadtxt((impresources.files(filter_data) / 'SDSS.dat'))

sdss_u = sdss[:, [0, 1]]
sdss_u[:, 1] *= sdss[:, -1]

sdss_g = sdss[:, [0, 2]]
sdss_g[:, 1] *= sdss[:, -1]

sdss_r = sdss[:, [0, 3]]
sdss_r[:, 1] *= sdss[:, -1]

sdss_i = sdss[:, [0, 4]]
sdss_i[:, 1] *= sdss[:, -1]

sdss_z = sdss[:, [0, 5]]
sdss_z[:, 1] *= sdss[:, -1]

class SDSSu(Filter):
    def __init__(self):
        super().__init__(sdss_u.T, name='SDSS u')


class SDSSg(Filter):
    def __init__(self):
        super().__init__(sdss_g.T, name='SDSS g')


class SDSSr(Filter):
    def __init__(self):
        super().__init__(sdss_r.T, name='SDSS r')


class SDSSi(Filter):
    def __init__(self):
        super().__init__(sdss_i.T, name='SDSS i')


class SDSSz(Filter):
    def __init__(self):
        super().__init__(sdss_z.T, name='SDSS z')


class TWOMASSJ(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / '2MASSJ.dat')).T
        transmission_curve[0] *= 1e4
        super().__init__(transmission_curve, name='2MASS J', non_photonic=True, Vega_zeropoint=-23.76248609213452 - 0.017)


class TWOMASSH(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / '2MASSH.dat')).T
        transmission_curve[0] *= 1e4
        super().__init__(transmission_curve, name='2MASS H', non_photonic=True, Vega_zeropoint=-24.84542522534151 + 0.016)


class TWOMASSK(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / '2MASSK.dat')).T
        transmission_curve[0] *= 1e4
        super().__init__(transmission_curve, name='2MASS Ks', non_photonic=True, Vega_zeropoint=-25.937629814008577 + 0.003)


# Morrissey et. al. (2004)
# https://cosmos.astro.caltech.edu/page/filterset


class GALEXFUV(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'GALEXFUV.dat')).T
        super().__init__(transmission_curve, name='GALEX FUV')


class GALEXNUV(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'GALEXNUV.dat')).T
        super().__init__(transmission_curve, name='GALEX NUV')


# https://github.com/lsst/throughputs/blob/main/baseline/README.md
# Just assume photonic with AB definition

class LSSTu(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'LSSTu.dat')).T
        super().__init__(transmission_curve, name='LSST u')


class LSSTg(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'LSSTg.dat')).T
        super().__init__(transmission_curve, name='LSST g')


class LSSTr(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'LSSTr.dat')).T
        super().__init__(transmission_curve, name='LSST r')


class LSSTi(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'LSSTi.dat')).T
        super().__init__(transmission_curve, name='LSST i')


class LSSTz(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'LSSTz.dat')).T
        super().__init__(transmission_curve, name='LSST z')


class LSSTy(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'LSSTy.dat')).T
        super().__init__(transmission_curve, name='LSST y')


# http://ipp.ifa.hawaii.edu/ps1.filters/
# Tonry et al. (2012)

panstarrs = np.loadtxt((impresources.files(filter_data) / 'panstarrs.dat'))
panstarrs[:, 0] *= 10.

panstarrs_open = panstarrs[:, [0, 1]]
panstarrs_g = panstarrs[:, [0, 2]]
panstarrs_r = panstarrs[:, [0, 3]]
panstarrs_i = panstarrs[:, [0, 4]]
panstarrs_z = panstarrs[:, [0, 5]]
panstarrs_y = panstarrs[:, [0, 6]]
panstarrs_w = panstarrs[:, [0, 7]]

class PANSTARRS_PS1_g(Filter):
    def __init__(self):
        super().__init__(panstarrs_g.T, name='PANSTARRS PS1 g', non_photonic=True)


class PANSTARRS_PS1_i(Filter):
    def __init__(self):
        super().__init__(panstarrs_i.T, name='PANSTARRS PS1 i', non_photonic=True)


class PANSTARRS_PS1_r(Filter):
    def __init__(self):
        super().__init__(panstarrs_r.T, name='PANSTARRS PS1 r', non_photonic=True)


class PANSTARRS_PS1_w(Filter):
    def __init__(self):
        super().__init__(panstarrs_w.T, name='PANSTARRS PS1 w', non_photonic=True)


class PANSTARRS_PS1_y(Filter):
    def __init__(self):
        super().__init__(panstarrs_y.T, name='PANSTARRS PS1 y', non_photonic=True)


class PANSTARRS_PS1_z(Filter):
    def __init__(self):
        super().__init__(panstarrs_z.T, name='PANSTARRS PS1 z', non_photonic=True)


class PANSTARRS_PS1_open(Filter):
    def __init__(self):
        super().__init__(panstarrs_open.T, name='PANSTARRS PS1 open', non_photonic=True)

# Source: Bessel (2011)

class Stromgrenu(Filter):
    def __init__(self):
        transmission_curve = jnp.array([
            [3150., 3175., 3200., 3225., 3250., 3275., 3300., 3325., 3350., 3375., 3400., 3425., 3450., 3475., 3500., 3525., 3550., 3575., 3600., 3625., 3650., 3675., 3700., 3725., 3750., 3775., 3800., 3825., 3850.],
            [0.000, 0.004, 0.050, 0.122, 0.219, 0.341, 0.479, 0.604, 0.710, 0.809, 0.886, 0.939, 0.976, 1.000, 0.995, 0.981, 0.943, 0.880, 0.782, 0.659, 0.525, 0.370, 0.246, 0.151, 0.071, 0.030, 0.014, 0.000, 0.000]
        ])
        super().__init__(transmission_curve, name='Stromgren u', AB_zeropoint=-22.40788262039795+0.308)


class Stromgrenv(Filter):
    def __init__(self):
        transmission_curve = jnp.array([
            [3750., 3775., 3800., 3825., 3850., 3875., 3900., 3925., 3950., 3975., 4000., 4025., 4050., 4075., 4100., 4125., 4150., 4175., 4200., 4225., 4250., 4275., 4300., 4325., 4350., 4375., 4400., 4425., 4450.],
            [0.000, 0.003, 0.006, 0.016, 0.029, 0.044, 0.060, 0.096, 0.157, 0.262, 0.404, 0.605, 0.810, 0.958, 1.000, 0.973, 0.882, 0.755, 0.571, 0.366, 0.224, 0.134, 0.079, 0.053, 0.039, 0.027, 0.014, 0.006, 0.000]
        ])
        super().__init__(transmission_curve, name='Stromgren v', AB_zeropoint=-22.40788262039795+0.327)


class Stromgrenb(Filter):
    def __init__(self):
        transmission_curve = jnp.array([
            [4350., 4375., 4400., 4425., 4450., 4475., 4500., 4525., 4550., 4575., 4600., 4625., 4650., 4675., 4700., 4725., 4750., 4775., 4800., 4825., 4850., 4875., 4900., 4925., 4950., 4975., 5000., 5025., 5050.],
            [0.000, 0.010, 0.023, 0.039, 0.056, 0.086, 0.118, 0.188, 0.287, 0.457, 0.681, 0.896, 0.998, 1.000, 0.942, 0.783, 0.558, 0.342, 0.211, 0.130, 0.072, 0.045, 0.027, 0.021, 0.015, 0.011, 0.007, 0.003, 0.000]
        ])
        super().__init__(transmission_curve, name='Stromgren b', AB_zeropoint=-22.40788262039795+0.187)


class Stromgreny(Filter):
    def __init__(self):
        transmission_curve = jnp.array([
            [5150., 5175., 5200., 5225., 5250., 5275., 5300., 5325., 5350., 5375., 5400., 5425., 5450., 5475., 5500., 5525., 5550., 5575., 5600., 5625., 5650., 5675., 5700., 5725., 5750., 5775., 5800., 5825., 5850.],
            [0.000, 0.022, 0.053, 0.082, 0.116, 0.194, 0.274, 0.393, 0.579, 0.782, 0.928, 0.985, 0.999, 1.000, 0.997, 0.938, 0.789, 0.574, 0.388, 0.232, 0.143, 0.090, 0.054, 0.031, 0.016, 0.010, 0.009, 0.004, 0.000]
        ])
        super().__init__(transmission_curve, name='Stromgren y', AB_zeropoint=-22.40788262039795+0.027)


class Bolometric(Filter):
    def __init__(self):
        transmission_curve = jnp.array([
            [1., 30000.],
            [1., 1.]
        ])
        super().__init__(transmission_curve, name='Bolometric')


FILTERS = [
    JohnsonCousinsU, JohnsonCousinsB, JohnsonCousinsV, JohnsonCousinsR, JohnsonCousinsI,
    TychoBT, TychoVT,
    GaiaG, GaiaRP, GaiaBP,
    SDSSu, SDSSg, SDSSr, SDSSi, SDSSz,
    TWOMASSJ, TWOMASSH, TWOMASSK,
    GALEXFUV, GALEXNUV,
    LSSTu, LSSTg, LSSTr, LSSTi, LSSTz, LSSTy,
    PANSTARRS_PS1_g, PANSTARRS_PS1_i, PANSTARRS_PS1_r, PANSTARRS_PS1_w, PANSTARRS_PS1_y, PANSTARRS_PS1_z, PANSTARRS_PS1_open,
    HipparcosHp,
    Stromgrenu, Stromgrenv, Stromgrenb, Stromgreny,
    Bolometric
]


FILTER_NAMES = [f.__name__ for f in FILTERS]
