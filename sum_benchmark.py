import phoebe
import numpy as np
from phoebe import u
import os
import matplotlib.pyplot as plt

# TODO: how to save?
import pickle

from spectrum import *
from spectrum_nn import flux
import jax.numpy as jnp
import jax

from phoebe_utils import PhoebeConfig
import timeit
import inspect
import click
from jax import lax
import math
from functools import partial

from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)

with open('test_single_sun.pickle', 'rb') as f:
    b = pickle.load(f)

config = PhoebeConfig(b, 'mesh01')
time = config.times[0]
coords = config.get_mesh_coordinates(time)
VW = config.get_radial_velocities(time)

mus = config.get_mus(time)
pos_mus = np.argwhere(mus>0)

projected_areas = config.get_projected_areas(time)

mus = jnp.array(mus[pos_mus])
areas = jnp.array(projected_areas[pos_mus])
vrads = jnp.array(VW[pos_mus])


def spectrum_flash_sum_no_checkpoint(log_wavelengths,
                                       areas,
                                       mus,
                                       vrads,
                                       chunk_size: int = DEFAULT_CHUNK_SIZE):
    # Z każdym elementem powierzchni przekazujemy
    # wektor jego wartości (mu, przyspieszenie, itd.)
    # Część wartości będzie przekazywana do modelu
    
    # Just the 1D case for now
    n_areas, n_samples = areas.shape
    mus_flattened = mus.reshape(areas.shape)
    points = log_wavelengths.shape[0]

    def chunk_scanner(carries, _):
        chunk_idx, atmo_sum, areas_sum = carries
        k_chunk_sizes = min(chunk_size, n_areas)

        a_chunk = lax.dynamic_slice(areas,
                                    (chunk_idx, 0),
                                    (k_chunk_sizes, n_samples))
        m_chunk = lax.dynamic_slice(mus_flattened,
                                    (chunk_idx, 0),
                                    (k_chunk_sizes, n_samples))
        vrad_chunk = lax.dynamic_slice(vrads,
                                        (chunk_idx, 0),
                                        (k_chunk_sizes, n_samples))

        
        # atmosphere_mul is the spectrum simulated for the corresponding wavelengths and optionally given parameters of mu, logg, and T.
        # It is then multiplied by the observed area to scale the contributions of spectra chunks
        # Shape: (n_vertices, 2, n_wavelengths)
        # 2 corresponds to the two components: continuum and full spectrum with lines
        atmosphere_mul = jnp.multiply(
            a_chunk.reshape((-1, 1, 1)),
            atmosphere_flux(log_wavelengths,
                            m_chunk))

        # (observed) radial velocity correction for both continuum and spectrum correction
        # Shape: (n_vertices, 2, n_wavelengths)
        vrad_atmosphere = jax.vmap(
            lambda a: v_interp(
                log_wavelengths,
                v_apply_vrad(
                    log_wavelengths,
                    vrad_chunk),
                a), in_axes=(1,))(atmosphere_mul)
        
        # Sum the atmosphere contributions and normalize by all areas' sum
        new_atmo_sum = atmo_sum + jnp.sum(vrad_atmosphere, axis=1)
        new_areas_sum = areas_sum + jnp.sum(a_chunk, axis=0)
        
        return (chunk_idx + k_chunk_sizes, new_atmo_sum, new_areas_sum), None

    # Return (2, n_vertices) for continuum and spectrum with lines
    (_, out, areas_sum), lse = lax.scan(
        chunk_scanner,
        init=(0, jnp.zeros((2, log_wavelengths.shape[-1])), jnp.zeros(1,)),
        xs=None,
        length=math.ceil(n_areas/chunk_size))
    return out/areas_sum


def newest(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)


def naive_sum_memory_benchmark(logdir, filename, points):
    LOG_WAVELENGTHS = jnp.linspace(3.65, 3.85, points)
    
    with jax.profiler.trace("/notebooks/tmp"):
        atmosphere_mul = jnp.multiply(
            areas.reshape((-1, 1, 1)),
            atmosphere_flux(LOG_WAVELENGTHS,
                            mus))

        # (observed) radial velocity correction for both continuum and spectrum correction
        # Shape: (n_vertices, 2, n_wavelengths)
        vrad_atmosphere = jax.vmap(
            lambda a: v_interp(
                LOG_WAVELENGTHS,
                v_apply_vrad(
                    LOG_WAVELENGTHS,
                    vrads),
                a), in_axes=(1,))(atmosphere_mul)

        atmosphere = jnp.sum(vrad_atmosphere, axis=1)/jnp.sum(areas, axis=0)
        atmosphere.block_until_ready()
        
    profiler_path = os.path.join(logdir, 'plugins/profile')
    os.rename(newest(profiler_path), os.path.join(profiler_path, filename))
    
    
def naive_grad_memory_benchmark(logdir, filename, points):
    LOG_WAVELENGTHS = jnp.linspace(3.65, 3.85, points)
    
    with jax.profiler.trace("/notebooks/tmp"):
        atmosphere_mul = jnp.multiply(
            areas.reshape((-1, 1, 1)),
            atmosphere_flux(LOG_WAVELENGTHS,
                            mus))

        # (observed) radial velocity correction for both continuum and spectrum correction
        # Shape: (n_vertices, 2, n_wavelengths)
        vrad_atmosphere = lambda v: jax.vmap(
            lambda a: v_interp(
                LOG_WAVELENGTHS,
                v_apply_vrad(
                    LOG_WAVELENGTHS,
                    v*vrads),
                a), in_axes=(1,))(atmosphere_mul)

        atmosphere = jax.jit(jax.grad(lambda v: (jnp.sum(vrad_atmosphere(v), axis=1)/jnp.sum(areas, axis=0))[0, 0]))(2.)
        atmosphere.block_until_ready()
        
    profiler_path = os.path.join(logdir, 'plugins/profile')
    os.rename(newest(profiler_path), os.path.join(profiler_path, filename))
    
    
def flash_w_checkpoints_sum_memory_benchmark(logdir, filename, points, chunk_size):
    LOG_WAVELENGTHS = jnp.linspace(3.65, 3.85, points)
    
    with jax.profiler.trace("/notebooks/tmp"):
        atmosphere = spectrum_flash_sum_no_checkpoint(LOG_WAVELENGTHS, areas, mus, vrads, chunk_size)
        atmosphere.block_until_ready()
        
    profiler_path = os.path.join(logdir, 'plugins/profile')
    os.rename(newest(profiler_path), os.path.join(profiler_path, filename))
    

def flash_w_checkpoints_grad_memory_benchmark(logdir, filename, points, chunk_size):
    LOG_WAVELENGTHS = jnp.linspace(3.65, 3.85, points)
    
    with jax.profiler.trace("/notebooks/tmp"):
        afg = jax.jit(jax.grad(lambda v: spectrum_flash_sum_no_checkpoint(LOG_WAVELENGTHS, areas, mus, vrads*v, 171)[0, 83549]))(2.)
        afg.block_until_ready()
        
    profiler_path = os.path.join(logdir, 'plugins/profile')
    os.rename(newest(profiler_path), os.path.join(profiler_path, filename))
    

def flash_grad_memory_benchmark(logdir, filename, points, chunk_size):
    LOG_WAVELENGTHS = jnp.linspace(3.65, 3.85, points)
    
    with jax.profiler.trace("/notebooks/tmp"):
        afg = jax.jit(jax.grad(lambda v: spectrum_flash_sum(LOG_WAVELENGTHS, areas, mus, vrads*v, 171)[0, 83549]))(2.)
        afg.block_until_ready()
        
    profiler_path = os.path.join(logdir, 'plugins/profile')
    os.rename(newest(profiler_path), os.path.join(profiler_path, filename))
    

def flash_sum_memory_benchmark(logdir, filename, points, chunk_size):
    
    LOG_WAVELENGTHS = jnp.linspace(3.65, 3.85, points)
    
    click.echo(f'Chunk size: {chunk_size}')
    
    with jax.profiler.trace("/notebooks/tmp"):
        atmosphere = spectrum_flash_sum(LOG_WAVELENGTHS, areas, mus, vrads, chunk_size)
        atmosphere.block_until_ready()
        
    profiler_path = os.path.join(logdir, 'plugins/profile')
    os.rename(newest(profiler_path), os.path.join(profiler_path, filename))
    
    
@click.command()
@click.option('-l', '--logdir',
              type=str, default='/notebooks/tmp')
@click.option('-n', '--machine-name',
              type=str)
@click.option('-t', '--method-type',
              type=click.Choice(['naive', 'flash_w_checkpoint', 'flash'], case_sensitive=False))
@click.option('-m', '--method',
              type=click.Choice(['sum', 'grad'], case_sensitive=False))
@click.option('-c', '--chunk-size',
              type=int, default=256)
@click.option('-p', '--points',
              type=int, default=100000)
def benchmark(logdir,
              machine_name, method_type,
              method, chunk_size, points):
    
    LOG_WAVELENGTHS = jnp.linspace(3.65, 3.85, points)
    
    click.echo(f'Preparing a benchmark of {method_type} {method} function...')
    
    filename_chunk_str = f'_{chunk_size}' if method_type != 'naive' else ''
    filename = f'{machine_name}_{method_type}_{method}_{points}{filename_chunk_str}'
    
    if method == 'sum':
        if method_type == 'naive':
            naive_sum_memory_benchmark(logdir, filename, points)
            click.echo(f'Complete! Saved to {os.path.join(logdir, "plugins/profile", filename)}.')
        elif method_type == 'flash_w_checkpoint':
            flash_w_checkpoints_sum_memory_benchmark(logdir, filename, points, chunk_size)
            click.echo(f'Complete! Saved to {os.path.join(logdir, "plugins/profile", filename)}.')
        elif method_type == 'flash':
            flash_sum_memory_benchmark(logdir, filename, points, chunk_size)
            click.echo(f'Complete! Saved to {os.path.join(logdir, "plugins/profile", filename)}.')
            
    elif method == 'grad':
        if method_type == 'naive':
            naive_grad_memory_benchmark(logdir, filename, points)
            click.echo(f'Complete! Saved to {os.path.join(logdir, "plugins/profile", filename)}.')
        elif method_type == 'flash_w_checkpoint':
            flash_w_checkpoints_grad_memory_benchmark(logdir, filename, points, chunk_size)
            click.echo(f'Complete! Saved to {os.path.join(logdir, "plugins/profile", filename)}.')
        elif method_type == 'flash':
            flash_grad_memory_benchmark(logdir, filename, points, chunk_size)
            click.echo(f'Complete! Saved to {os.path.join(logdir, "plugins/profile", filename)}.')

if __name__ == '__main__':
    benchmark()