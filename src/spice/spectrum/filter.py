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
                 ST_zeropoint: float = -21.10):
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
        super().__init__(transmission_curve, name='Johnson-Cousins U', AB_zeropoint=-22.40788262039795+0.771-0.01, ST_zeropoint=-21.10-0.142)


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
        super().__init__(transmission_curve, name='Johnson-Cousins B', AB_zeropoint=-22.40788262039795-0.138+0.008, ST_zeropoint=-21.10-0.625)


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
        super().__init__(transmission_curve, name='Johnson-Cousins V', AB_zeropoint=-22.40788262039795-0.023+0.003, ST_zeropoint=-21.10-0.019)


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
        super().__init__(transmission_curve, name='Johnson-Cousins R', AB_zeropoint=-22.40788262039795+0.16+0.003, ST_zeropoint=-21.10+0.538)


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
        super().__init__(transmission_curve, name='Johnson-Cousins I', AB_zeropoint=-22.40788262039795+0.402+0.002, ST_zeropoint=-21.10+1.220)


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
        super().__init__(transmission_curve, name='Hipparcos Hp', AB_zeropoint=-22.40788262039795-0.022-0.008, ST_zeropoint=-21.10-0.074)


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
        super().__init__(transmission_curve, name='Tycho BT', AB_zeropoint=-22.40788262039795-0.09-0.01, ST_zeropoint=-21.10-0.672)
        

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
        super().__init__(transmission_curve, name='Tycho VT', AB_zeropoint=-22.40788262039795-0.044+0.007, ST_zeropoint=-21.10-0.115)


# # J/A+A/649/A3     Gaia Early Data Release 3 photometric passbands (Riello+, 2021)

class GaiaG(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'gaia_edr3_passband.dat')).T
        zp = np.loadtxt((impresources.files(filter_data) / 'gaia_edr3_zeropt.dat'))
        transmission_curve[transmission_curve == 99.99] = 0.
        super().__init__(jnp.array([transmission_curve[0]*10., transmission_curve[1]]), name='Gaia G', AB_zeropoint=zp[0])


class GaiaBP(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'gaia_edr3_passband.dat')).T
        zp = np.loadtxt((impresources.files(filter_data) / 'gaia_edr3_zeropt.dat'))
        transmission_curve[transmission_curve == 99.99] = 0.
        super().__init__(jnp.array([transmission_curve[0]*10., transmission_curve[3]]), name='Gaia BP', AB_zeropoint=zp[2])


class GaiaRP(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'gaia_edr3_passband.dat')).T
        zp = np.loadtxt((impresources.files(filter_data) / 'gaia_edr3_zeropt.dat'))
        transmission_curve[transmission_curve == 99.99] = 0.
        super().__init__(jnp.array([transmission_curve[0]*10., transmission_curve[5]]), name='Gaia RP', AB_zeropoint=zp[4])
        
        
# Sartoretti et al. (2022)
class GaiaRVS(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) /'grvsfilter.csv'), skiprows=1, delimiter=',')
        super().__init__(jnp.array([transmission_curve[0]*10., transmission_curve[1]]), name='Gaia RVS', AB_zeropoint=21.317)

# Doi et al. (2010)

# Doi et al. (2010)
sdss_u = jnp.array([
    [2940., 2960., 2980., 3000., 3020., 3040., 3060., 3080., 3100., 3120., 3140., 3160., 3180., 3200., 3220., 3240., 3260., 3280., 3300., 3320., 3340., 3360., 3380., 3400., 3420., 3440., 3460., 3480., 3500., 3520., 3540., 3560., 3580., 3600., 3620., 3640., 3660., 3680., 3700., 3720., 3740., 3760., 3780., 3800., 3820., 3840., 3860., 3880., 3900., 3920., 3940., 3960., 3980., 4000., 4020., 4040., 4060., 4080., 4100., 4120., 4140., 4160., 4180.],
    [0.0001, 0.0003, 0.0005, 0.0009, 0.0014, 0.0029, 0.0050, 0.0077, 0.0136, 0.0194, 0.0276, 0.0367, 0.0460, 0.0558, 0.0656, 0.0743, 0.0826, 0.0905, 0.0969, 0.1033, 0.1098, 0.1162, 0.1223, 0.1268, 0.1314, 0.1353, 0.1390, 0.1427, 0.1460, 0.1494, 0.1518, 0.1537, 0.1555, 0.1570, 0.1586, 0.1604, 0.1625, 0.1638, 0.1626, 0.1614, 0.1563, 0.1497, 0.1416, 0.1282, 0.1148, 0.0956, 0.0744, 0.0549, 0.0407, 0.0265, 0.0177, 0.0107, 0.0050, 0.0032, 0.0015, 0.0008, 0.0005, 0.0003, 0.0003, 0.0003, 0.0002, 0.0001, 0.]
])

sdss_g = jnp.array([
    [3620., 3640., 3660., 3680., 3700., 3720., 3740., 3760., 3780., 3800., 3820., 3840., 3860., 3880., 3900., 3920., 3940., 3960., 3980., 4000., 4020., 4040., 4060., 4080., 4100., 4120., 4140., 4160., 4180., 4200., 4220., 4240., 4260., 4280., 4300., 4320., 4340., 4360., 4380., 4400., 4420., 4440., 4460., 4480., 4500., 4520., 4540., 4560., 4580., 4600., 4620., 4640., 4660., 4680., 4700., 4720., 4740., 4760., 4780., 4800., 4820., 4840., 4860., 4880., 4900., 4920., 4940., 4960., 4980., 5000., 5020., 5040., 5060., 5080., 5100., 5120., 5140., 5160., 5180., 5200., 5220., 5240., 5260., 5280., 5300., 5320., 5340., 5360., 5380., 5400., 5420., 5440., 5460., 5480., 5500., 5520., 5540., 5560., 5580., 5600., 5620.],
    [0.0002, 0.0015, 0.0027, 0.0038, 0.0035, 0.0032, 0.0031, 0.0037, 0.0048, 0.0067, 0.0115, 0.0220, 0.0353, 0.0507, 0.0740, 0.0973, 0.1224, 0.1484, 0.1757, 0.2081, 0.2404, 0.2617, 0.2785, 0.2954, 0.3122, 0.3290, 0.3411, 0.3512, 0.3603, 0.3660, 0.3717, 0.3773, 0.3830, 0.3886, 0.3943, 0.3999, 0.4043, 0.4083, 0.4122, 0.4161, 0.4200, 0.4240, 0.4279, 0.4314, 0.4337, 0.4359, 0.4381, 0.4404, 0.4426, 0.4448, 0.4470, 0.4488, 0.4504, 0.4521, 0.4537, 0.4553, 0.4569, 0.4586, 0.4601, 0.4611, 0.4622, 0.4633, 0.4644, 0.4655, 0.4666, 0.4677, 0.4687, 0.4698, 0.4709, 0.4719, 0.4730, 0.4741, 0.4752, 0.4762, 0.4770, 0.4765, 0.4753, 0.4731, 0.4704, 0.4672, 0.4625, 0.4512, 0.4326, 0.3996, 0.3429, 0.2768, 0.2013, 0.1397, 0.0899, 0.0585, 0.0398, 0.0269, 0.0189, 0.0136, 0.0096, 0.0068, 0.0051, 0.0037, 0.0024, 0.0013, 0.0002]
])

sdss_r = jnp.array([
    [5380., 5400., 5420., 5440., 5460., 5480., 5500., 5520., 5540., 5560., 5580., 5600., 5620., 5640., 5660., 5680., 5700., 5720., 5740., 5760., 5780., 5800., 5820., 5840., 5860., 5880., 5900., 5920., 5940., 5960., 5980., 6000., 6020., 6040., 6060., 6080., 6100., 6120., 6140., 6160., 6180., 6200., 6220., 6240., 6260., 6280., 6300., 6320., 6340., 6360., 6380., 6400., 6420., 6440., 6460., 6480., 6500., 6520., 6540., 6560., 6580., 6600., 6620., 6640., 6660., 6680., 6700., 6720., 6740., 6760., 6780., 6800., 6820., 6840., 6860., 6880., 6900., 6920., 6940., 6960., 6980., 7000., 7020., 7040.],
    [0.0002, 0.0018, 0.0050, 0.0105, 0.0225, 0.0452, 0.0751, 0.1175, 0.1641, 0.2118, 0.2567, 0.2979, 0.3368, 0.3724, 0.4042, 0.4327, 0.4531, 0.4667, 0.4774, 0.4868, 0.4949, 0.5019, 0.5068, 0.5118, 0.5162, 0.5187, 0.5212, 0.5237, 0.5262, 0.5287, 0.5309, 0.5323, 0.5336, 0.5349, 0.5362, 0.5370, 0.5361, 0.5352, 0.5342, 0.5333, 0.5333, 0.5358, 0.5383, 0.5409, 0.5434, 0.5454, 0.5462, 0.5470, 0.5477, 0.5485, 0.5488, 0.5480, 0.5472, 0.5464, 0.5455, 0.5449, 0.5450, 0.5450, 0.5447, 0.5435, 0.5423, 0.5394, 0.5324, 0.5190, 0.4992, 0.4683, 0.4230, 0.3685, 0.3030, 0.2344, 0.1724, 0.1212, 0.0842, 0.0556, 0.0370, 0.0273, 0.0201, 0.0130, 0.0097, 0.0076, 0.0054, 0.0036, 0.0019, 0.0003]
])

sdss_i = jnp.array([
    [6600., 6620., 6640., 6660., 6680., 6700., 6720., 6740., 6760., 6780., 6800., 6820., 6840., 6860., 6880., 6900., 6920., 6940., 6960., 6980., 7000., 7020., 7040., 7060., 7080., 7100., 7120., 7140., 7160., 7180., 7200., 7220., 7240., 7260., 7280., 7300., 7320., 7340., 7360., 7380., 7400., 7420., 7440., 7460., 7480., 7500., 7520., 7540., 7560., 7580., 7600., 7620., 7640., 7660., 7680., 7700., 7720., 7740., 7760., 7780., 7800., 7820., 7840., 7860., 7880., 7900., 7920., 7940., 7960., 7980., 8000., 8020., 8040., 8060., 8080., 8100., 8120., 8140., 8160., 8180., 8200., 8220., 8240., 8260., 8280., 8300., 8320., 8340., 8360., 8380.],
    [0.0002, 0.0008, 0.0012, 0.0017, 0.0026, 0.0046, 0.0080, 0.0131, 0.0226, 0.0365, 0.0560, 0.0834, 0.1162, 0.1553, 0.1952, 0.2377, 0.2839, 0.3222, 0.3565, 0.3869, 0.4104, 0.4301, 0.4458, 0.4565, 0.4648, 0.4706, 0.4764, 0.4791, 0.4814, 0.4823, 0.4815, 0.4806, 0.4771, 0.4732, 0.4694, 0.4655, 0.4617, 0.4578, 0.4539, 0.4505, 0.4477, 0.4449, 0.4421, 0.4393, 0.4364, 0.4335, 0.4306, 0.4264, 0.4220, 0.4176, 0.4132, 0.4088, 0.4042, 0.3996, 0.3951, 0.3905, 0.3860, 0.3815, 0.3770, 0.3725, 0.3680, 0.3636, 0.3610, 0.3586, 0.3562, 0.3539, 0.3515, 0.3492, 0.3469, 0.3449, 0.3432, 0.3411, 0.3388, 0.3362, 0.3328, 0.3279, 0.3215, 0.3043, 0.2763, 0.2379, 0.1857, 0.1355, 0.0874, 0.0578, 0.0360, 0.0212, 0.0144, 0.0094, 0.0061, 0.0020]
])

sdss_z = jnp.array([
    [7680., 7700., 7720., 7740., 7760., 7780., 7800., 7820., 7840., 7860., 7880., 7900., 7920., 7940., 7960., 7980., 8000., 8020., 8040., 8060., 8080., 8100., 8120., 8140., 8160., 8180., 8200., 8220., 8240., 8260., 8280., 8300., 8320., 8340., 8360., 8380., 8400., 8420., 8440., 8460., 8480., 8500., 8520., 8540., 8560., 8580., 8600., 8620., 8640., 8660., 8680., 8700., 8720., 8740., 8760., 8780., 8800., 8820., 8840., 8860., 8880., 8900., 8920., 8940., 8960., 8980., 9000., 9020., 9040., 9060., 9080., 9100., 9120., 9140., 9160., 9180., 9200., 9220., 9240., 9260., 9280., 9300., 9320., 9340., 9360., 9380., 9400., 9420., 9440., 9460., 9480., 9500., 9520., 9540., 9560., 9580., 9600., 9620., 9640., 9660., 9680., 9700., 9720., 9740., 9760., 9780., 9800., 9820., 9840., 9860., 9880., 9900., 9920., 9940., 9960., 9980., 10000., 10020., 10040., 10060., 10080., 10100., 10120., 10140., 10160., 10180., 10200., 10220., 10240., 10260., 10280., 10300., 10320., 10340., 10360., 10380., 10400., 10420., 10440., 10460., 10480., 10500., 10520., 10540., 10560., 10580., 10600., 10620., 10640., 10660., 10680., 10700., 10720., 10740., 10760., 10780., 10800., 10820., 10840., 10860., 10880., 10900., 10920., 10940., 10960., 10980., 11000., 11020., 11040., 11060., 11080., 11100., 11120., 11140., 11160.],
    [0.0000, 0.0000, 0.0001, 0.0001, 0.0001, 0.0001, 0.0002, 0.0002, 0.0002, 0.0003, 0.0004, 0.0006, 0.0008, 0.0010, 0.0012, 0.0016, 0.0023, 0.0030, 0.0044, 0.0059, 0.0078, 0.0105, 0.0132, 0.0171, 0.0212, 0.0257, 0.0309, 0.0362, 0.0415, 0.0467, 0.0519, 0.0570, 0.0621, 0.0664, 0.0705, 0.0742, 0.0773, 0.0803, 0.0824, 0.0845, 0.0861, 0.0871, 0.0882, 0.0893, 0.0904, 0.0911, 0.0912, 0.0913, 0.0915, 0.0917, 0.0914, 0.0906, 0.0898, 0.0889, 0.0881, 0.0869, 0.0854, 0.0838, 0.0822, 0.0806, 0.0790, 0.0772, 0.0755, 0.0738, 0.0720, 0.0704, 0.0688, 0.0672, 0.0656, 0.0640, 0.0625, 0.0612, 0.0598, 0.0585, 0.0571, 0.0559, 0.0547, 0.0535, 0.0523, 0.0511, 0.0499, 0.0487, 0.0475, 0.0463, 0.0451, 0.0440, 0.0430, 0.0420, 0.0410, 0.0400, 0.0390, 0.0379, 0.0369, 0.0358, 0.0347, 0.0337, 0.0327, 0.0317, 0.0307, 0.0297, 0.0287, 0.0276, 0.0266, 0.0256, 0.0245, 0.0235, 0.0226, 0.0216, 0.0206, 0.0196, 0.0186, 0.0176, 0.0166, 0.0156, 0.0147, 0.0138, 0.0132, 0.0125, 0.0119, 0.0113, 0.0106, 0.0099, 0.0093, 0.0086, 0.0080, 0.0074, 0.0070, 0.0065, 0.0061, 0.0056, 0.0052, 0.0047, 0.0042, 0.0038, 0.0033, 0.0030, 0.0029, 0.0027, 0.0026, 0.0024, 0.0022, 0.0021, 0.0019, 0.0018, 0.0016, 0.0015, 0.0014, 0.0013, 0.0012, 0.0012, 0.0011, 0.0010, 0.0009, 0.0008, 0.0007, 0.0007, 0.0007, 0.0006, 0.0006, 0.0006, 0.0005, 0.0005, 0.0005, 0.0004, 0.0004, 0.0004, 0.0003, 0.0003, 0.0002, 0.0002, 0.0002, 0.0001, 0.0001, 0.0001, 0.0000]
])

class SDSSu(Filter):
    def __init__(self):
        super().__init__(sdss_u, name='SDSS u')


class SDSSg(Filter):
    def __init__(self):
        super().__init__(sdss_g, name='SDSS g')


class SDSSr(Filter):
    def __init__(self):
        super().__init__(sdss_r, name='SDSS r')


class SDSSi(Filter):
    def __init__(self):
        super().__init__(sdss_i, name='SDSS i')


class SDSSz(Filter):
    def __init__(self):
        super().__init__(sdss_z, name='SDSS z')


# class TWOMASSJ(Filter):
#     def __init__(self):
#         transmission_curve = np.loadtxt((impresources.files(filter_data) / '2MASSJ.dat')).T
#         super().__init__(transmission_curve, name='2MASS J')


# class TWOMASSH(Filter):
#     def __init__(self):
#         transmission_curve = np.loadtxt((impresources.files(filter_data) / '2MASSH.dat')).T
#         super().__init__(transmission_curve, name='2MASS H')


# class TWOMASSKs(Filter):
#     def __init__(self):
#         transmission_curve = np.loadtxt((impresources.files(filter_data) / '2MASSKs.dat')).T
#         super().__init__(transmission_curve, name='2MASS Ks')


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

class PANSTARRS_PS1_g(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'PAN-STARRS_PS1.g.dat')).T
        super().__init__(transmission_curve, name='PANSTARRS PS1 g', non_photonic=True)


class PANSTARRS_PS1_i(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'PAN-STARRS_PS1.i.dat')).T
        super().__init__(transmission_curve, name='PANSTARRS PS1 i', non_photonic=True)


class PANSTARRS_PS1_r(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'PAN-STARRS_PS1.r.dat')).T
        super().__init__(transmission_curve, name='PANSTARRS PS1 r', non_photonic=True)


class PANSTARRS_PS1_w(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'PAN-STARRS_PS1.w.dat')).T
        super().__init__(transmission_curve, name='PANSTARRS PS1 w', non_photonic=True)


class PANSTARRS_PS1_y(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'PAN-STARRS_PS1.y.dat')).T
        super().__init__(transmission_curve, name='PANSTARRS PS1 y', non_photonic=True)


class PANSTARRS_PS1_z(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'PAN-STARRS_PS1.z.dat')).T
        super().__init__(transmission_curve, name='PANSTARRS PS1 z', non_photonic=True)


class PANSTARRS_PS1_open(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'PAN-STARRS_PS1.open.dat')).T
        super().__init__(transmission_curve, name='PANSTARRS PS1 open', non_photonic=True)

# Source: Bessel (2011)

class Stromgrenu(Filter):
    def __init__(self):
        transmission_curve = jnp.array([
            [3150., 3175., 3200., 3225., 3250., 3275., 3300., 3325., 3350., 3375., 3400., 3425., 3450., 3475., 3500., 3525., 3550., 3575., 3600., 3625., 3650., 3675., 3700., 3725., 3750., 3775., 3800., 3825., 3850.],
            [0.000, 0.004, 0.050, 0.122, 0.219, 0.341, 0.479, 0.604, 0.710, 0.809, 0.886, 0.939, 0.976, 1.000, 0.995, 0.981, 0.943, 0.880, 0.782, 0.659, 0.525, 0.370, 0.246, 0.151, 0.071, 0.030, 0.014, 0.000, 0.000]
        ])
        super().__init__(transmission_curve, name='Stromgren u')


class Stromgrenv(Filter):
    def __init__(self):
        transmission_curve = jnp.array([
            [3750., 3775., 3800., 3825., 3850., 3875., 3900., 3925., 3950., 3975., 4000., 4025., 4050., 4075., 4100., 4125., 4150., 4175., 4200., 4225., 4250., 4275., 4300., 4325., 4350., 4375., 4400., 4425., 4450.],
            [0.000, 0.003, 0.006, 0.016, 0.029, 0.044, 0.060, 0.096, 0.157, 0.262, 0.404, 0.605, 0.810, 0.958, 1.000, 0.973, 0.882, 0.755, 0.571, 0.366, 0.224, 0.134, 0.079, 0.053, 0.039, 0.027, 0.014, 0.006, 0.000]
        ])
        super().__init__(transmission_curve, name='Stromgren v')


class Stromgrenb(Filter):
    def __init__(self):
        transmission_curve = jnp.array([
            [4350., 4375., 4400., 4425., 4450., 4475., 4500., 4525., 4550., 4575., 4600., 4625., 4650., 4675., 4700., 4725., 4750., 4775., 4800., 4825., 4850., 4875., 4900., 4925., 4950., 4975., 5000., 5025., 5050.],
            [0.000, 0.010, 0.023, 0.039, 0.056, 0.086, 0.118, 0.188, 0.287, 0.457, 0.681, 0.896, 0.998, 1.000, 0.942, 0.783, 0.558, 0.342, 0.211, 0.130, 0.072, 0.045, 0.027, 0.021, 0.015, 0.011, 0.007, 0.003, 0.000]
        ])
        super().__init__(transmission_curve, name='Stromgren b')


class Stromgreny(Filter):
    def __init__(self):
        transmission_curve = jnp.array([
            [5150., 5175., 5200., 5225., 5250., 5275., 5300., 5325., 5350., 5375., 5400., 5425., 5450., 5475., 5500., 5525., 5550., 5575., 5600., 5625., 5650., 5675., 5700., 5725., 5750., 5775., 5800., 5825., 5850.],
            [0.000, 0.022, 0.053, 0.082, 0.116, 0.194, 0.274, 0.393, 0.579, 0.782, 0.928, 0.985, 0.999, 1.000, 0.997, 0.938, 0.789, 0.574, 0.388, 0.232, 0.143, 0.090, 0.054, 0.031, 0.016, 0.010, 0.009, 0.004, 0.000]
        ])
        super().__init__(transmission_curve, name='Stromgren y')


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
    # TWOMASSJ, TWOMASSH, TWOMASSKs,
    GALEXFUV, GALEXNUV,
    LSSTu, LSSTg, LSSTr, LSSTi, LSSTz, LSSTy,
    PANSTARRS_PS1_g, PANSTARRS_PS1_i, PANSTARRS_PS1_r, PANSTARRS_PS1_w, PANSTARRS_PS1_y, PANSTARRS_PS1_z, PANSTARRS_PS1_open,
    HipparcosHp,
    Stromgrenu, Stromgrenv, Stromgrenb, Stromgreny,
    Bolometric
]


FILTER_NAMES = [f.__name__ for f in FILTERS]
