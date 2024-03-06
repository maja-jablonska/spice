from enum import auto, Enum
import jax.numpy as jnp


class Abundance(Enum):
    FeH = auto()
    C = auto()
    N = auto()
    O = auto()
    Li = auto()
    Mg = auto()
    Si = auto()
    Ca = auto()
    Ti = auto()
    Na = auto()
    Mn = auto()
    Co = auto()
    Ni = auto()
    Ba = auto()
    Sr = auto()
    Eu = auto()
    Fe = auto()


SOLAR_TEFF = jnp.log10(5780)
SOLAR_LOGG = jnp.log10(4.43)
SOLAR_VTURB = 2.0 # TODO: check the solar Vrad
# TODO: check the FeH
# FeH - logarytm

# TODO: add various solar abundances
SOLAR_ABUNDANCES = jnp.array(
    [0., 8.46, 7.83, 8.69, 0.96, 7.55, 7.51, 6.30,
     4.97, 6.22, 5.42, 4.94, 6.20, 2.27, 2.83, 0.52, 7.46])