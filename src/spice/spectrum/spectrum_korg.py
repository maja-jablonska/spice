from .spectrum import BaseSpectrum
from overrides import override
from typing import Callable, Dict, List, Union
from jax import numpy as jnp
from jax.typing import ArrayLike
import numpy as np
from flax import linen as nn
import os
import pickle

from juliacall import Main as jl

jl.seval("using Korg"); Korg = jl.Korg
from matplotlib import pyplot as plt

def read_from_pickle(fn):
    with open(fn, "rb") as f:
        return pickle.load(f)

labels_names = ['Teff', 'logg']

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
labels_names += elements_90
