import phoebe
import numpy as np
from phoebe import u
import os
import matplotlib.pyplot as plt

# TODO: how to save?
import pickle

with open('test_binary.pickle', 'rb') as f:
    b = pickle.load(f)
    
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

config = PhoebeConfig(b, 'bigmesh')
time = config.times[0]

coords = config.get_mesh_coordinates(time)
VW = config.get_radial_velocities(time)

mus = config.get_mus(time)

# Get only the visible vertices
pos_mus = np.argwhere(mus>0)

projected_areas = config.get_projected_areas(time)

mus = jnp.array(mus[pos_mus])
areas = jnp.array(projected_areas[pos_mus])

atmosphere_flux = jax.jit(jax.vmap(flux, in_axes=(None, 0)))


def _flash_sum_w_checkpoints(areas, mus, chunk_size, precision=10000):
    # Just the 1D case for now
    n_areas, n_samples = areas.shape
    mus_flattened = mus.reshape(areas.shape)

    def chunk_scanner(carries, _):
        chunk_idx, atmo_sun = carries
        k_chunk_sizes = min(chunk_size, n_areas)

        a_chunk = lax.dynamic_slice(areas,
                                    (chunk_idx, 0),
                                    (k_chunk_sizes, n_samples))
        m_chunk = lax.dynamic_slice(mus_flattened,
                                    (chunk_idx, 0),
                                    (k_chunk_sizes, n_samples))

        new_atmo_sum = atmo_sun + jnp.sum(
            jnp.multiply(
                a_chunk.reshape((-1, 1, 1)),
                atmosphere_flux(jnp.linspace(0, 1, precision),
                                jnp.ones_like(m_chunk).flatten())
            ), axis=0)
        return (chunk_idx + k_chunk_sizes, new_atmo_sum), None

    (_, out), lse = lax.scan(chunk_scanner, init=(0, jnp.zeros((precision, 2))), xs=None, length=math.ceil(n_areas/chunk_size))
    return out

def _flash_sum(areas, mus, chunk_size, precision=10000):
    # Just the 1D case for now
    n_areas, n_samples = areas.shape
    mus_flattened = mus.reshape(areas.shape)

    @partial(jax.checkpoint, prevent_cse=False)
    def chunk_scanner(carries, _):
        chunk_idx, atmo_sun = carries
        k_chunk_sizes = min(chunk_size, n_areas)

        a_chunk = lax.dynamic_slice(areas,
                                    (chunk_idx, 0),
                                    (k_chunk_sizes, n_samples))
        m_chunk = lax.dynamic_slice(mus_flattened,
                                    (chunk_idx, 0),
                                    (k_chunk_sizes, n_samples))

        new_atmo_sum = atmo_sun + jnp.sum(
            jnp.multiply(
                a_chunk.reshape((-1, 1, 1)),
                atmosphere_flux(jnp.linspace(0, 1, precision),
                                jnp.ones_like(m_chunk).flatten())
            ), axis=0)
        return (chunk_idx + k_chunk_sizes, new_atmo_sum), None

    (_, out), lse = lax.scan(chunk_scanner, init=(0, jnp.zeros((precision, 2))), xs=None, length=math.ceil(n_areas/chunk_size))
    return out


def newest(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)


def naive_sum_memory_benchmark(logdir, filename, points):
    atmosphere = jnp.sum(
        jnp.multiply(
            areas.reshape((-1, 1, 1)),
            atmosphere_flux(jnp.linspace(0, 1, points),
                            jnp.ones_like(mus).flatten())), axis=0)/jnp.sum(areas)
    atmosphere.block_until_ready()
    
    with jax.profiler.trace("/notebooks/tmp"):
        atmosphere = jnp.sum(
            jnp.multiply(
                areas.reshape((-1, 1, 1)),
                atmosphere_flux(jnp.linspace(0, 1, points),
                                jnp.ones_like(mus).flatten())), axis=0)/jnp.sum(areas)
        atmosphere.block_until_ready()
        
    profiler_path = os.path.join(logdir, 'plugins/profile')
    os.rename(newest(profiler_path), os.path.join(profiler_path, filename))
    
    
def naive_grad_memory_benchmark(logdir, filename, points):
    atmosphere = jax.random.normal(jax.random.PRNGKey(1), shape=(points, 2))
    atmosphere_flux_grad = jax.jit(
    jax.grad(
        lambda a, m: 
        jnp.sum(jnp.abs(jnp.sum(jnp.multiply(
        a.reshape((-1, 1, 1)), jax.vmap(flux, in_axes=(None, 0))(
            jnp.linspace(0, 1, points),
            jnp.ones_like(m).flatten())), axis=0)/jnp.sum(a)-atmosphere)[:, 0])
    ))
    afg = atmosphere_flux_grad(mus, areas)
    afg.block_until_ready()
    
    with jax.profiler.trace("/notebooks/tmp"):
        afg = atmosphere_flux_grad(mus, areas)
        afg.block_until_ready()
        
    profiler_path = os.path.join(logdir, 'plugins/profile')
    os.rename(newest(profiler_path), os.path.join(profiler_path, filename))
    
    
def flash_w_checkpoints_sum_memory_benchmark(logdir, filename, points, chunk_size):
    atmosphere = _flash_sum_w_checkpoints(areas, mus, chunk_size, points)/jnp.sum(areas)
    atmosphere.block_until_ready()
    
    with jax.profiler.trace("/notebooks/tmp"):
        atmosphere = _flash_sum_w_checkpoints(areas, mus, chunk_size, points)/jnp.sum(areas)
        atmosphere.block_until_ready()
        
    profiler_path = os.path.join(logdir, 'plugins/profile')
    os.rename(newest(profiler_path), os.path.join(profiler_path, filename))
    

def flash_w_checkpoints_grad_memory_benchmark(logdir, filename, points, chunk_size):
    atmosphere = jax.random.normal(jax.random.PRNGKey(1), shape=(points, 2))
    atmosphere_flux_grad = jax.jit(
    jax.grad(
        lambda a, m: 
        (jnp.sum(
            jnp.power(_flash_sum_w_checkpoints(a, m, chunk_size, points)/jnp.sum(a)
                    -atmosphere, 2)))
        )
    )
    afg = atmosphere_flux_grad(mus, areas)
    afg.block_until_ready()
    
    with jax.profiler.trace("/notebooks/tmp"):
        afg = atmosphere_flux_grad(mus, areas)
        afg.block_until_ready()
        
    profiler_path = os.path.join(logdir, 'plugins/profile')
    os.rename(newest(profiler_path), os.path.join(profiler_path, filename))
    

def flash_grad_memory_benchmark(logdir, filename, points, chunk_size):
    atmosphere = jax.random.normal(jax.random.PRNGKey(1), shape=(points, 2))
    atmosphere_flux_grad = jax.jit(
    jax.grad(
        lambda a, m: 
        (jnp.sum(
            jnp.power(_flash_sum(a, m, chunk_size, points)/jnp.sum(a)
                    -atmosphere, 2)))
        )
    )
    afg = atmosphere_flux_grad(mus, areas)
    afg.block_until_ready()
    
    with jax.profiler.trace("/notebooks/tmp"):
        afg = atmosphere_flux_grad(mus, areas)
        afg.block_until_ready()
        
    profiler_path = os.path.join(logdir, 'plugins/profile')
    os.rename(newest(profiler_path), os.path.join(profiler_path, filename))
    

def flash_sum_memory_benchmark(logdir, filename, points, chunk_size):
    click.echo(f'Chunk size: {chunk_size}')
    atmosphere = _flash_sum(areas, mus, chunk_size, points)/jnp.sum(areas)
    atmosphere.block_until_ready()
    
    with jax.profiler.trace("/notebooks/tmp"):
        atmosphere = _flash_sum(areas, mus, chunk_size, points)/jnp.sum(areas)
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