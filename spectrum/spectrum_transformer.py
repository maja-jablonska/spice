from .spectrum import BaseSpectrum
from overrides import override
from typing import Callable, Dict, List, Union
from jax.typing import ArrayLike
from jax import numpy as jnp
import numpy as np
from flax import linen as nn
from flax.training import checkpoints

import os 


dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')

model_name = "MLP_w_att_69099_st1000000_1000000"
prefix = f"checkpoint_{model_name}"
CHECKPOINTS_DIR = os.path.join(dir_path, "compute_grid/ckpts_maja_T3")

restored_state = checkpoints.restore_checkpoint(ckpt_dir=CHECKPOINTS_DIR, target=None, prefix=prefix)
restored_params = restored_state["params"]["VmapMLP_wavelength_att_0"]

log_wave = np.linspace(np.log10(4000), np.log10(4100), 10000)

labels_names = ['Teff', 'logg', 'Vturb', 'FeH', 
          'C', 'N', 'O', 'Li', 'Mg', 'Si', 
          'Ca', 'Ti', 'Na', 'Mn', 'Co', 
          'Ni', 'Ba', 'Sr', 'Eu', 'Fe']
element_names = labels_names[3:]

max_params = np.array([7999.0, 5.5, 2.0, 0.5, 10.0512, 
                       9.4641, 10.2513, 3.9998, 8.8423, 
                       8.886, 7.4469, 6.022, 7.7607, 
                       6.2947, 6.4165, 7.2375, 4.7222, 
                       5.2535, 2.9325, 8.0])

min_params = np.array([2500.0, -0.49, 0.5, -4.998, 2.5942, 
                       2.0183, 3.5896, -0.9999, 2.2702, 
                       2.3443, 1.1272, -0.3188, 0.3513, 
                       -0.4472, -0.2082, 0.7536, -3.6691, 
                       -3.0606, -5.4177, 2.5016])

default_params = (min_params+max_params)/2.0
default_abundances = default_params[3:]


def frequency_encoding(x, min_period, max_period, dimension):
    periods = jnp.logspace(jnp.log10(min_period), jnp.log10(max_period), num=dimension)
    
    y = jnp.sin(2*jnp.pi/periods*x)
    return y

class MLP_single_wavelength_att(nn.Module):
    @nn.compact
    def __call__(self, x):
        p, w = x
        enc_w = frequency_encoding(w, min_period=1e-5, max_period=1.0, dimension=128)
        enc_w = enc_w[None, ...]
        p = nn.gelu(nn.Dense(4*128)(p))
        p = nn.Dense(16*128)(p)
        enc_p = jnp.reshape(p, (16, 128))
        
        # print(enc_p.shape, enc_w.shape)
        # [batch_sizes…, length, features]
        x = enc_w
        for _ in range(10):
            x = x + nn.MultiHeadDotProductAttention(num_heads=8)(inputs_q=x, inputs_kv=nn.LayerNorm()(enc_p))
            x = x + nn.Dense(128)(nn.gelu(nn.Dense(256)(nn.LayerNorm()(x))))
        
        x = nn.gelu(nn.Dense(256)(x[0]))
        x = nn.gelu(nn.Dense(256)(x))
        x = nn.Dense(2)(x)
        return x
    
class MLP_wavelength_att_mu(nn.Module):
    
    @nn.compact
    def __call__(self, inputs, train):
        log_waves, mu, p = inputs
        
        A = jnp.log10(0.6*(mu + 2/3)) # Gray atmosphere
        A -= 6.
        
        DecManyWave = nn.vmap(
                    MLP_single_wavelength_att, 
                    in_axes=((None, 0),), out_axes=0,
                    variable_axes={'params': None}, 
                    split_rngs={'params': False})
        
        x = DecManyWave(name="decoder")((p, log_waves))
        x = x.at[:,1].add(A)
        return x

print("Models defined.")

m = MLP_wavelength_att_mu()

def flux(log_wave: ArrayLike, mu: float, parameters: ArrayLike) -> ArrayLike:
    """Calculates the flux using the transformer-based model trained on a spectra grid.

    Args:
        log_wave (ArrayLike): Array of logarithmic wavelengths (log10 of wavelength in Angstroms).
        mu (float): Cosine of the angle between the line of sight and the surface normal. It ranges from -1 to 1.
        parameters (ArrayLike): Array of parameters - logteff, logg, "Li", "Be", ... , "U"

    Returns:
        ArrayLike: Array of intensities in []
    """
    # The temperature should be log10(log10(teff))
    parameters = parameters.at[0].set(jnp.log10(jnp.log10(parameters[0])))
    x = m.apply({'params': restored_params},
                   (jnp.atleast_2d(log_wave), mu, parameters),
                   train=False)
    return jnp.power(10, x).T


class TransformerSpectrum(BaseSpectrum):
    @override
    @staticmethod
    def get_label_names(self) -> List[str]:
        return labels_names
    
    @override
    @staticmethod
    def is_in_bounds(parameters: ArrayLike) -> bool:
        return jnp.all(parameters>=min_params) and jnp.all(parameters<=max_params)
    
    @override
    @staticmethod
    def get_default_parameters() -> ArrayLike:
        return default_params
    
    @override(check_signature=False)
    @staticmethod
    def to_parameters(Teff: float = default_params[0],
                      logg: float = default_params[1],
                      Vturb: float = default_params[2],
                      abundances: Union[ArrayLike, Dict[str, float]] = None):
        if isinstance(abundances, dict):
            abundance_values = jnp.array([abundances.get(element, 0.) for element in element_names])
        else:
            abundance_values = abundances or jnp.array(default_abundances)
        
        return jnp.concatenate([jnp.array([Teff, logg, Vturb]), abundance_values])
    
    @override
    @staticmethod
    def flux_method() -> Callable[..., ArrayLike]:
        return flux
