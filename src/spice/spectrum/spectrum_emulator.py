from abc import ABC, abstractmethod
from typing import Generic, TypeVar


T = TypeVar("T")


class SpectrumEmulator(Generic[T]):
    @property
    @abstractmethod
    def stellar_parameter_names(self) -> T:
        """Get labels of stellar parameters (no geometry-related parameters, e.g. mu)

        Returns:
            T:
        """
        raise NotImplementedError
    
    @abstractmethod
    def to_parameters(self, parameters: T) -> T:
        raise NotImplementedError

    @abstractmethod
    def flux(self, log_wavelengths: T, parameters: T) -> T:
        raise NotImplementedError

    @abstractmethod
    def intensity(self, log_wavelengths: T, mu: float, parameters: T) -> T:
        """Calculate the intensity for given wavelengths and mus

        Args:
            log_wavelengths (T): [log(angstrom)]
            mu (float): cosine of the angle between the star's radius and the line of sight
            spectral_parameters (T): an array of predefined stellar parameters

        Returns:
            T: intensities corresponding to passed wavelengths [erg/cm2/s/angstrom]
        """
        raise NotImplementedError
