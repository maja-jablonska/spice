from .spectrum import BaseSpectrum
from overrides import override
from typing import Callable, Dict, List, Union
from jax import numpy as jnp
from jax.typing import ArrayLike
import numpy as np
from flax import linen as nn
import os
import pickle


def read_from_pickle(fn):
    with open(fn, "rb") as f:
        return pickle.load(f)
    
    
restored_params = read_from_pickle(os.path.join(os.path.dirname(__file__), "spectrum_korg.pickle"))

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

min_parameters = np.array([ 10**3.6020677e+00,  6.7894318e-05, -3.0967069e+00,
                   -3.0676999e+00, -3.0745316e+00, -3.9427969e+00, -3.0743036e+00,
                   -3.9395518e+00, -3.0945952e+00, -3.9913447e+00, -3.0912573e+00,
                   -3.9436648e+00, -3.0687692e+00, -3.9603372e+00, -3.0501244e+00,
                   -3.9276848e+00, -3.0535388e+00, -3.9437594e+00, -3.0742633e+00,
                   -3.9935064e+00, -3.0404315e+00, -3.9243042e+00, -3.0448878e+00,
                   -3.0718670e+00, -3.0910945e+00, -3.0706975e+00, -3.0922050e+00,
                   -3.0688632e+00, -3.0594923e+00, -3.0894337e+00, -3.0827107e+00,
                   -3.0674851e+00, -3.0640681e+00, -3.0954568e+00, -3.0848448e+00,
                   -3.0905144e+00, -3.0564837e+00, -3.0913301e+00, -3.0624452e+00,
                   -3.0834172e+00, -3.0891290e+00, -3.0757093e+00, -3.0745728e+00,
                   -3.0874336e+00, -3.0687723e+00, -3.0653961e+00, -3.0539694e+00,
                   -3.0937476e+00, -3.0898323e+00, -3.0889518e+00, -3.0755427e+00,
                   -3.0582242e+00, -3.0872426e+00, -3.0816150e+00, -3.0554674e+00,
                   -3.0856342e+00, -3.0835328e+00, -3.0897248e+00, -3.0617685e+00,
                   -3.0575101e+00, -3.0585477e+00, -3.0800762e+00, -3.0684376e+00,
                   -3.0913398e+00, -3.0836990e+00, -3.0840561e+00, -3.0606158e+00,
                   -3.0590074e+00, -3.0943856e+00, -3.0542331e+00, -3.0707209e+00,
                   -3.0604641e+00, -3.0693922e+00, -3.0811100e+00, -3.0765646e+00,
                   -3.0718412e+00, -3.0700235e+00, -3.0863514e+00, -3.0796990e+00,
                   -3.0911481e+00, -3.0982413e+00, -3.0959229e+00, -3.0847242e+00,
                   -3.0752430e+00, -3.0774732e+00, -3.0616822e+00, -3.0831549e+00,
                   -3.0878015e+00, -3.0649395e+00, -3.0819504e+00, -3.0861907e+00,
                   -3.0948832e+00], dtype=np.float32)

max_parameters = np.array([10**3.9030874, 4.9994946, 1.5791478, 1.5938333, 1.556247 ,
                   2.4546688, 1.5835627, 2.3390923, 1.5827703, 2.474998 , 1.5719196,
                   2.4787211, 1.5915307, 2.4861922, 1.5669659, 2.3987606, 1.5724797,
                   2.3464055, 1.5723362, 2.417851 , 1.5701865, 2.4026892, 1.5782307,
                   1.5555463, 1.5652467, 1.5771998, 1.5877018, 1.5428551, 1.5558861,
                   1.585077 , 1.5889889, 1.5746806, 1.5721723, 1.5954087, 1.5470334,
                   1.5964056, 1.5271162, 1.5534036, 1.5895414, 1.5685614, 1.5850676,
                   1.5744147, 1.5856276, 1.5695841, 1.5675495, 1.5714014, 1.5820699,
                   1.565108 , 1.5696615, 1.5757725, 1.5908864, 1.5696541, 1.5982723,
                   1.5729141, 1.5913082, 1.5881267, 1.5856838, 1.5859636, 1.5782737,
                   1.5861198, 1.5552615, 1.5601346, 1.5701807, 1.5661758, 1.5905483,
                   1.5745987, 1.5879754, 1.5771769, 1.5518768, 1.5926497, 1.5798193,
                   1.5753173, 1.56341  , 1.584925 , 1.5742487, 1.589338 , 1.5780828,
                   1.5803379, 1.573083 , 1.5763483, 1.5877959, 1.563212 , 1.5299966,
                   1.5825379, 1.5524042, 1.5794249, 1.5907439, 1.5698735, 1.5870272,
                   1.5856891, 1.5592823, 1.5569359], dtype=np.float32)

default_params = (min_parameters+max_parameters)/2
default_abundances = default_params[2:]


D_ATT              = 256
D_FF               = 4*D_ATT
NO_LAYER           = 16
NO_HEAD            = 8
NO_TOKEN           = 16
OUT_DIMENSIONALITY = 2

def frequency_encoding(x, min_period, max_period, dimension):
    periods = jnp.logspace(jnp.log10(min_period), jnp.log10(max_period), num=dimension)
    
    y = jnp.sin(2*jnp.pi/periods*x)
    return y

class MLP_single_wavelength_att(nn.Module):
    @nn.compact
    def __call__(self, x):
        p, w = x
        enc_w = frequency_encoding(w, min_period=1e-6, max_period=10, dimension=D_ATT)
        enc_w = enc_w[None, ...]
        p = nn.gelu(nn.Dense(4*D_ATT)(p))
        p = nn.Dense(NO_TOKEN*D_ATT)(p)
        enc_p = jnp.reshape(p, (NO_TOKEN, D_ATT))
        
        x_pre = enc_w
        x_post = enc_w
        for _ in range(NO_LAYER):
            # MHA
            _x = x_post + nn.MultiHeadDotProductAttention(num_heads=NO_HEAD)(inputs_q=x_post,
                                                                            inputs_kv=nn.LayerNorm()(enc_p))
            x_pre = x_pre + _x
            x_post = nn.LayerNorm()(_x)
            # MLP
            _x = x_post + nn.Dense(D_ATT)(nn.gelu(nn.Dense(D_FF)(x_post)))
            
            x_pre = x_pre + _x
            x_post = nn.LayerNorm()(_x)
        
        x_pre = nn.LayerNorm()(x_pre)
        x = x_pre + x_post
        x = nn.gelu(nn.Dense(256)(x[0]))
        x = nn.Dense(OUT_DIMENSIONALITY)(x)
        return x
    
class MLP_wavelength_att(nn.Module):
    
    @nn.compact
    def __call__(self, inputs, train):
        p = inputs["parameters"]
        log_waves = inputs["logwave"]
        DecManyWave = nn.vmap(
                    MLP_single_wavelength_att, 
                    in_axes=((None, 0),),out_axes=0,
                    variable_axes={'params': None}, 
                    split_rngs={'params': False})
        
        x = DecManyWave(name="decoder")((p, log_waves))
        return x

print("Models defined.")

m = MLP_wavelength_att()

def flux(log_wave: ArrayLike, mu: float, parameters: ArrayLike) -> ArrayLike:
    """Calculates the flux using the Korg-trained transformer-based model.

    Args:
        log_wave (ArrayLike): Array of logarithmic wavelengths (log10 of wavelength in Angstroms).
        mu (float): Cosine of the angle between the line of sight and the surface normal. It ranges from -1 to 1.
        parameters (ArrayLike): Array of parameters - logteff, logg, "Li", "Be", ... , "U"

    Returns:
        ArrayLike: Array of intensities in []
    """
    # parameters: logteff, logg, "Li", "Be", ... , "U"

    mu_array = jnp.array([mu])
    log_wave = jnp.atleast_2d(log_wave)
    
    # The temperature should be in log10(log10(teff))
    parameters = parameters.at[0].set(jnp.log10(jnp.log10(parameters[0])))
    p = jnp.concatenate((parameters[:2], mu_array, parameters[2:]), axis=0)
    
    x = m.apply({'params': restored_params},
                   {"logwave":log_wave, "parameters": p},
                   train=False)
    return jnp.power(10, x).T

class KorgSpectrum(BaseSpectrum):
    @override
    @staticmethod
    def get_label_names() -> List[str]:
        return labels_names
    
    @override
    @staticmethod
    def is_in_bounds(parameters: ArrayLike) -> bool:
        return jnp.all(parameters>=min_parameters) and jnp.all(parameters<=max_parameters)
    
    @override
    @staticmethod
    def get_default_parameters() -> ArrayLike:
        return default_params
    
    @override(check_signature=False)
    @staticmethod
    def to_parameters(Teff: float = default_params[0],
                      logg: float = default_params[1],
                      abundances: Union[ArrayLike, Dict[str, float]] = None):
        if isinstance(abundances, dict):
            abundance_values = jnp.array([abundances.get(element, 0.) for element in elements_90])
        else:
            abundance_values = abundances or jnp.array(default_abundances)
        
        return jnp.concatenate([jnp.array([Teff, logg]), abundance_values])
    
    @override
    @staticmethod
    def flux_method() -> Callable[..., ArrayLike]:
        return flux
