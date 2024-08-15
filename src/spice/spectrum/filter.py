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
                 ab_zeropoint: float = 3.631 * 1e-20,
                 name: Optional[str] = None):
        self.__transmission_curve = transmission_curve

        tc_freq = wavelengths_to_frequencies(transmission_curve[0])
        tc_freq_mask = jnp.argsort(tc_freq)
        self.__transmission_curve_freq = jnp.vstack([tc_freq[tc_freq_mask],
                                                     transmission_curve[1][tc_freq_mask]])

        self.__ab_zeropoint = ab_zeropoint
        self.__name = name or re.sub(r"([A-Z])", r" \1", type(self).__name__).split()

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
        return _filter_responses(wavelengths, self.transmission_curve_wavelengths[0],
                                 self.transmission_curve_wavelengths[1])

    def filter_responses_for_frequencies(self, wavelengths: ArrayLike) -> ArrayLike:
        return _filter_responses(wavelengths, self.transmission_curve_frequencies[0],
                                 self.transmission_curve_frequencies[1])

    def plot_filter_responses_for_wavelengths(self,
                                              wavelengths: ArrayLike,
                                              intensities: ArrayLike,
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
                                              frequencies: ArrayLike,
                                              intensities: ArrayLike,
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


# Filter information: http://dx.doi.org/10.1086/132749

class BesselU(Filter):
    def __init__(self):
        transmission_curve = jnp.array([
            [3000., 3050., 3100., 3150., 3200., 3250., 3300., 3350., 3400.,
             3450., 3500., 3550., 3600., 3650., 3700., 3750., 3800., 3850.,
             3900., 3950., 4000., 4050., 4100., 4150., 4200.],
            [0., 0.016, 0.068, 0.167, 0.287, 0.423, 0.56, 0.673, 0.772,
             0.841, 0.905, 0.943, 0.981, 0.993, 1., 0.989, 0.916, 0.804,
             0.625, 0.423, 0.238, 0.114, 0.051, 0.019, 0.]
        ])
        super().__init__(transmission_curve, name='Bessel U')


class BesselB(Filter):
    def __init__(self):
        transmission_curve = jnp.array([
            [3600., 3700., 3800., 3900., 4000., 4100., 4200., 4300., 4400.,
             4500., 4600., 4700., 4800., 4900., 5000., 5100., 5200., 5300.,
             5400., 5500., 5600.],
            [0., 0.03, 0.134, 0.567, 0.92, 0.978, 1., 0.978, 0.935,
             0.853, 0.74, 0.64, 0.536, 0.424, 0.325, 0.235, 0.15, 0.095,
             0.043, 0.009, 0.]
        ])
        super().__init__(transmission_curve, name='Bessel B')


class BesselV(Filter):
    def __init__(self):
        transmission_curve = jnp.array([
            [4700., 4800., 4900., 5000., 5100., 5200., 5300., 5400., 5500.,
             5600., 5700., 5800., 5900., 6000., 6100., 6200., 6300., 6400.,
             6500., 6600., 6700., 6800., 6900., 7000.],
            [0., 0.03, 0.163, 0.458, 0.78, 0.967, 1., 0.973, 0.898,
             0.792, 0.684, 0.574, 0.461, 0.359, 0.27, 0.197, 0.135, 0.081,
             0.045, 0.025, 0.017, 0.013, 0.009, 0.]
        ])
        super().__init__(transmission_curve, name='Bessel V')


class BesselR(Filter):
    def __init__(self):
        transmission_curve = jnp.array([
            [5500., 5600., 5700., 5800., 5900., 6000., 6100., 6200., 6300.,
             6400., 6500., 6600., 6700., 6800., 6900., 7000., 7100., 7200.,
             7300., 7400., 7500., 8000., 8500., 9000.],
            [0., 0.23, 0.74, 0.91, 0.98, 1., 0.98, 0.96, 0.93, 0.9, 0.86,
             0.81, 0.78, 0.72, 0.67, 0.61, 0.56, 0.51, 0.46, 0.4, 0.35, 0.14,
             0.03, 0.]
        ])
        super().__init__(transmission_curve, name='Bessel R')


class BesselI(Filter):
    def __init__(self):
        transmission_curve = jnp.array([
            [7000., 7100., 7200., 7300., 7400., 7500., 7600., 7700., 7800.,
             7900., 8000., 8100., 8200., 8300., 8400., 8500., 8600., 8700.,
             8800., 8900., 9000., 9100., 9200.],
            [0., 0.024, 0.232, 0.555, 0.785, 0.91, 0.965, 0.985, 0.99,
             0.995, 1., 1., 0.99, 0.98, 0.95, 0.91, 0.86, 0.75,
             0.56, 0.33, 0.15, 0.03, 0.]
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


# https://cosmos.astro.caltech.edu/page/filterset


class SDSSu(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'SDSSu.dat')).T
        super().__init__(transmission_curve, name='SDSS u')


class SDSSg(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'SDSSg.dat')).T
        super().__init__(transmission_curve, name='SDSS g')


class SDSSr(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'SDSSr.dat')).T
        super().__init__(transmission_curve, name='SDSS r')


class SDSSi(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'SDSSi.dat')).T
        super().__init__(transmission_curve, name='SDSS i')


class SDSSz(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'SDSSz.dat')).T
        super().__init__(transmission_curve, name='SDSS z')


# https://www.ipac.caltech.edu/2mass/releases/allsky/doc/sec3_1b1.html


class TWOMASSJ(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / '2MASSJ.dat')).T
        super().__init__(transmission_curve, name='2MASS J')


class TWOMASSH(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / '2MASSH.dat')).T
        super().__init__(transmission_curve, name='2MASS H')


class TWOMASSKs(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / '2MASSKs.dat')).T
        super().__init__(transmission_curve, name='2MASS Ks')


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

class PANSTARRS_PS1_g(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'PAN-STARRS_PS1.g.dat')).T
        super().__init__(transmission_curve, name='PANSTARRS PS1 g')


class PANSTARRS_PS1_i(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'PAN-STARRS_PS1.i.dat')).T
        super().__init__(transmission_curve, name='PANSTARRS PS1 i')


class PANSTARRS_PS1_r(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'PAN-STARRS_PS1.r.dat')).T
        super().__init__(transmission_curve, name='PANSTARRS PS1 r')


class PANSTARRS_PS1_w(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'PAN-STARRS_PS1.w.dat')).T
        super().__init__(transmission_curve, name='PANSTARRS PS1 w')


class PANSTARRS_PS1_y(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'PAN-STARRS_PS1.y.dat')).T
        super().__init__(transmission_curve, name='PANSTARRS PS1 y')


class PANSTARRS_PS1_z(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'PAN-STARRS_PS1.z.dat')).T
        super().__init__(transmission_curve, name='PANSTARRS PS1 z')


class PANSTARRS_PS1_open(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'PAN-STARRS_PS1.open.dat')).T
        super().__init__(transmission_curve, name='PANSTARRS PS1 open')

# http://ulisse.pd.astro.it/Astro/ADPS/Systems/Sys_135/index_135.html

class HipparcosHp(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'Hipparcos_Hipparcos.Hp.dat')).T
        super().__init__(transmission_curve, name='Hipparcos Hp')

# http://ulisse.pd.astro.it/Astro/ADPS/Systems/Sys_093/index_093.html


class CousinsI(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'Generic_Cousins.I.dat')).T
        super().__init__(transmission_curve, name='Cousins I')


class CousinsR(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'Generic_Cousins.R.dat')).T
        super().__init__(transmission_curve, name='Cousins R')


# http://svo2.cab.inta-csic.es/theory/fps/index.php?mode=browse&gname=Generic&gname2=Johnson&asttype=


class JohnsonU(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'Generic_Johnson.U.dat')).T
        super().__init__(transmission_curve, name='Johnson U')


class JohnsonB(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'Generic_Johnson.B.dat')).T
        super().__init__(transmission_curve, name='Johnson B')


class JohnsonV(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'Generic_Johnson.V.dat')).T
        super().__init__(transmission_curve, name='Johnson V')


class JohnsonR(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'Generic_Johnson.R.dat')).T
        super().__init__(transmission_curve, name='Johnson R')


class JohnsonI(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'Generic_Johnson.I.dat')).T
        super().__init__(transmission_curve, name='Johnson I')


class JohnsonJ(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'Generic_Johnson.J.dat')).T
        super().__init__(transmission_curve, name='Johnson J')


class JohnsonM(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'Generic_Johnson.M.dat')).T
        super().__init__(transmission_curve, name='Johnson M')


# http://svo2.cab.inta-csic.es/theory/fps/index.php?mode=browse&gname=Generic&gname2=Stromgren&asttype=


class Stromgrenu(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'Generic_Stromgren.u.dat')).T
        super().__init__(transmission_curve, name='Stromgren u')


class Stromgrenv(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'Generic_Stromgren.v.dat')).T
        super().__init__(transmission_curve, name='Stromgren v')


class Stromgrenb(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'Generic_Stromgren.b.dat')).T
        super().__init__(transmission_curve, name='Stromgren b')


class Stromgreny(Filter):
    def __init__(self):
        transmission_curve = np.loadtxt((impresources.files(filter_data) / 'Generic_Stromgren.y.dat')).T
        super().__init__(transmission_curve, name='Stromgren y')


FILTERS = [
    BesselU, BesselB, BesselV, BesselR, BesselI,
    GaiaG, GaiaRP, GaiaBP,
    SDSSu, SDSSg, SDSSr, SDSSi, SDSSz,
    TWOMASSJ, TWOMASSH, TWOMASSKs,
    GALEXFUV, GALEXNUV,
    LSSTu, LSSTg, LSSTr, LSSTi, LSSTz, LSSTy,
    PANSTARRS_PS1_g, PANSTARRS_PS1_i, PANSTARRS_PS1_r, PANSTARRS_PS1_w, PANSTARRS_PS1_y, PANSTARRS_PS1_z, PANSTARRS_PS1_open,
    HipparcosHp,
    CousinsI, CousinsR,
    JohnsonU, JohnsonB, JohnsonV, JohnsonR, JohnsonI, JohnsonJ, JohnsonM,
    Stromgrenu, Stromgrenv, Stromgrenb, Stromgreny
]


FILTER_NAMES = [f.__name__ for f in FILTERS]
