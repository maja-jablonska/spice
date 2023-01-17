import jax
from jax import lax, random, numpy as jnp
import flax
import numpy as np
from flax import linen as nn
from flax.training import train_state, checkpoints

from typing import Callable, Tuple


model_name = "MLP_sine_2"
CHECKPOINTS_DIR = "compute_grid/ckpts"
prefix = f"checkpoint_{model_name}_"
restored_state = checkpoints.restore_checkpoint(ckpt_dir=CHECKPOINTS_DIR, target=None, prefix=prefix)
restored_params = restored_state["params"]


def frequency_encoding(x, min_period, max_period, dimension):
    periods = jnp.logspace(jnp.log10(min_period), jnp.log10(max_period), num=dimension)
    
    y = jnp.sin(2*jnp.pi/periods*x)
    return y

class MLP_single_wavelength_sine(nn.Module):
    architecture: tuple = (256, 256, 256, 256)
    @nn.compact
    def __call__(self, x):
        w = x
        enc_w = frequency_encoding(w, min_period=1e-5, max_period=1.0, dimension=128)
        _x = enc_w
        for features in self.architecture:
            _x = nn.gelu(nn.Dense(features)(_x))
        x = nn.Dense(2, bias_init=nn.initializers.ones)(_x)
        return x
    
    
class MLP_wavelength_sine_mu(nn.Module):
    @nn.compact
    def __call__(self, inputs, train):
        log_waves, mu = inputs
        
        A = jnp.log10(0.6*(mu + 2/3)) # Gray atmosphere
        
        DecManyWave = nn.vmap(
                    MLP_single_wavelength_sine, 
                    in_axes=0, out_axes=0,
                    variable_axes={'params': None}, 
                    split_rngs={'params': False})
        
        x = DecManyWave(name="decoder")(log_waves)
        x = x.at[:,1].add(A) # mnożenie razy pewna stała kontinuum
        return x

m = MLP_wavelength_sine_mu()
    
def flux(log_wave: jnp.ndarray, mu: float) -> jnp.ndarray:
    x = m.apply({'params': restored_params},
                   (log_wave, mu),
                   train=False)
    y = jnp.power(10, x[:, 1])
    return jnp.array([jnp.multiply(x[:, 0], y), y])
