from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from spice.spectrum.solar_parameters import SOLAR_PARAMETERS


T = TypeVar("T")


class SpectrumEmulator(ABC, Generic[T]):
    """Abstract interface for spectrum emulators.

    Concrete emulators must provide the parameter contract
    (:meth:`stellar_parameter_names` and :meth:`to_parameters`). The
    :meth:`flux` and :meth:`intensity` methods are *optional capabilities*:
    they raise :class:`NotImplementedError` by default so an emulator that only
    supports one of the two (e.g. an intensity-only grid interpolator) does not
    have to implement the other.
    """

    @property
    @abstractmethod
    def stellar_parameter_names(self) -> T:
        """Labels of the stellar parameters (no geometry parameters such as mu).

        Returns:
            T: the ordered parameter names.
        """
        raise NotImplementedError

    @property
    def solar_parameters(self) -> T:
        """The solar parameter vector, derived from the emulator's parameter set."""
        return self.to_parameters(SOLAR_PARAMETERS)

    @abstractmethod
    def to_parameters(self, parameters: T) -> T:
        """Convert a parameter mapping/array into the emulator's parameter vector."""
        raise NotImplementedError

    def flux(self, log_wavelengths: T, parameters: T) -> T:
        """Disc-integrated flux for the given wavelengths and parameters.

        Optional capability — override in emulators that support it.

        Args:
            log_wavelengths (T): [log(angstrom)]
            parameters (T): an array of predefined stellar parameters

        Returns:
            T: flux corresponding to the passed wavelengths [erg/cm2/s/angstrom]
        """
        raise NotImplementedError

    def intensity(self, log_wavelengths: T, mu: float, parameters: T) -> T:
        """Specific intensity for the given wavelengths, mu and parameters.

        Optional capability — override in emulators that support it.

        Args:
            log_wavelengths (T): [log(angstrom)]
            mu (float): cosine of the angle between the star's radius and the line of sight
            parameters (T): an array of predefined stellar parameters

        Returns:
            T: intensities corresponding to passed wavelengths [erg/cm2/s/angstrom]
        """
        raise NotImplementedError
