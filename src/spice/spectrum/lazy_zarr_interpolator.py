from itertools import product
from typing import Any, Dict, List
import warnings
import jax
from jax._src.typing import ArrayLike
import jax.numpy as jnp
import numpy as np
from jax import lax
from functools import partial
import os
from tqdm.auto import tqdm
import zarr
import pandas as pd
from overrides import override

from spice.spectrum.spectrum_emulator import SpectrumEmulator

DEVICE = jax.devices()[0]

@jax.jit
def _axis_linear_base(values, value, atol=1e-6, rtol=1e-6):
    n = values.shape[0]

    # --- match detection (no dynamic shapes) ---
    mask = jnp.isclose(values, value, atol=atol, rtol=rtol)
    has_match = jnp.any(mask)

    # first True index (safe because argmax returns 0 if all False)
    idx = jnp.argmax(mask)

    # --- match case ---
    def match_case(_):
        def single(_):
            return idx, idx

        def multiple(_):
            return lax.cond(
                idx == 0,
                lambda _: (0, 1),
                lambda _: lax.cond(
                    idx == n - 1,
                    lambda _: (n - 2, n - 1),
                    lambda _: (idx - 1, idx + 1),
                    operand=None,
                ),
                operand=None,
            )

        return lax.cond(n == 1, single, multiple, operand=None)

    # --- no match case ---
    def no_match_case(_):
        hi = jnp.searchsorted(values, value, side="right")
        hi = jnp.clip(hi, 1, n - 1)
        return hi - 1, hi

    return lax.cond(has_match, match_case, no_match_case, operand=None)

@jax.jit
def _axis_linear_bracket(values, value, radius=0, atol=1e-6, rtol=1e-6):
    n = values.shape[0]

    base_lo, base_hi = _axis_linear_base(
        values,
        value,
        atol=atol,
        rtol=rtol,
    )

    # clamp indices (JAX-safe)
    lo = jnp.maximum(0, base_lo - radius)
    hi = jnp.minimum(n - 1, base_hi + radius)

    same = lo == hi

    lo_value = values[lo]
    hi_value = values[hi]

    def compute_t(_):
        return (value - lo_value) / (hi_value - lo_value)

    t = lax.cond(
        same,
        lambda _: jnp.array(0.0, dtype=values.dtype),
        compute_t,
        operand=None,
    )

    return lo, hi, t

def _compute_brackets(axis_values, query_values, atol=1e-6, rtol=1e-6):
    d = len(axis_values)

    def body(i):
        return jnp.array(
            _axis_linear_bracket(axis_values[i], query_values[i], 0, atol, rtol)
        )

    return jnp.stack([body(i) for i in range(d)])

@jax.jit
def compute_brackets_batched(axis_values, queries, atol=1e-6, rtol=1e-6):
    """
    queries: (B, d)
    returns: (B, d, 3)
    """

    def per_query(q):
        return jnp.stack([
            jnp.array(_axis_linear_bracket(axis_values[i], q[i], 0, atol, rtol))
            for i in range(len(axis_values))
        ])

    return jax.vmap(per_query)(queries)


def make_corners(d):
    return jnp.array(np.array(list(np.ndindex(*(2,) * d)), dtype=np.int32))

@partial(jax.jit, static_argnums=(2, 3))
def weighted_rows_from_brackets(grid, brackets, corners_dimension, num_rows):
    """
    grid: (N1, N2, ..., Nd) int32 with row indices (or -1)
    brackets: (d, 3) array -> [lo, hi, t] per axis
    corners_dimension: int, number of dimensions in the grid

    returns:
        weights_per_row: (num_rows,) dense weight vector
    """

    brackets = jnp.asarray(brackets)
    lo = brackets[:, 0].astype(jnp.int32)
    hi = brackets[:, 1].astype(jnp.int32)
    t  = brackets[:, 2]
    
    corners = make_corners(corners_dimension)

    # --- indices for each corner ---
    indices = jnp.where(corners == 0, lo, hi)  # (2^d, d)

    # --- weights ---
    weights = jnp.where(corners == 0, 1.0 - t, t)
    weights = jnp.prod(weights, axis=1)  # (2^d,)

    # --- gather rows ---
    rows = grid[tuple(indices.T)]  # (2^d,)

    # --- mask invalid ---
    valid = (weights > 0.0) & (rows >= 0)

    rows = jnp.where(valid, rows, 0)
    weights = jnp.where(valid, weights, 0.0)

    # --- accumulate ---
    out = jnp.zeros((num_rows,), dtype=weights.dtype)
    out = out.at[rows].add(weights)

    total_weight = jnp.sum(out)

    # --- normalize ---
    out = jnp.where(
        total_weight > 0,
        out / total_weight,
        out
    )

    return out

@partial(jax.jit, static_argnums=(2, 3))
def weighted_rows_from_brackets_batched(grid, brackets, d, num_rows):
    """
    brackets: (B, d, 3)
    returns: (B, num_rows)
    """

    def per_query(b):
        return weighted_rows_from_brackets(grid, b, d, num_rows)

    return jax.vmap(per_query)(brackets)

class GridIndex:
    def __init__(self, grid, axes, columns, device=DEVICE):
        """
        axes: list/tuple of 1D arrays (one per dimension)
        """
        self.axes_np = [np.asarray(ax) for ax in axes]
        self.grid_np = np.asarray(grid)
        self.columns = columns

        self.axes = tuple(
            jax.device_put(ax, device=device) for ax in self.axes_np
        )
        self.grid = jax.device_put(self.grid_np, device=device)

        self.d = len(self.axes)
        self.num_rows = int(self.grid.max() + 1)

        self.grid.block_until_ready()

    @classmethod
    def from_dataframe(cls, df, columns, device=DEVICE):
        """
        columns: list of column names defining the grid axes
        """
        print('Building grid index...', flush=True)

        axes = [np.unique(df[c].to_numpy()) for c in columns]
        shape = tuple(len(ax) for ax in axes)

        grid = np.full(shape, -1, dtype=np.int32)

        indices = [
            np.searchsorted(axes[i], df[columns[i]].to_numpy())
            for i in range(len(columns))
        ]

        row_idx = df.row_idx.to_numpy(dtype=np.int32, copy=False)

        grid[tuple(indices)] = row_idx

        print('Grid index ready.', flush=True)
        return cls(grid, axes, columns, device=device)

    def _grid_row(self, i, j, k, m):
        row = int(self.grid_np[i, j, k, m])
        if row < 0:
            raise KeyError(f'No spectrum exists for grid cell {(i, j, k, m)}.')
        return row
    
    @partial(jax.jit, static_argnums=(0,))
    def lookup(self, query_values, atol=1e-6, rtol=1e-6):
        """
        query_values: (d,) array
        """

        brackets = _compute_brackets(self.axes, query_values, atol, rtol)
        # (d, 3)

        weights_dense = weighted_rows_from_brackets(
            self.grid,
            brackets,
            self.d,
            self.num_rows,
        )

        max_corners = 2 ** self.d

        rows = jnp.nonzero(weights_dense > 0, size=max_corners, fill_value=-1)[0]
        weights = weights_dense[rows]
        weights = jnp.where(rows >= 0, weights, 0.0)

        return rows, weights
    
    def batched_lookup(self, queries, atol=1e-6, rtol=1e-6):
        brackets = compute_brackets_batched(self.axes, queries, atol, rtol)

        weights_dense = weighted_rows_from_brackets_batched(
            self.grid,
            brackets,
            self.d,
            self.num_rows,
        )

        max_corners = 2 ** self.d

        def extract(w):
            rows = jnp.nonzero(w > 0, size=max_corners, fill_value=-1)[0]
            weights = w[rows]
            weights = jnp.where(rows >= 0, weights, 0.0)
            return rows, weights

        rows, weights = jax.vmap(extract)(weights_dense)
        return rows, weights
    


@jax.jit
def combine_rows_jax(row_indices, weights, data):
    """
    row_indices: (K,) int32   (e.g. 16 corners)
    weights:     (K,) float32
    data:        (N, D) float32  (e.g. spectra)

    returns:
        combined: (D,)
    """

    # mask invalid rows (-1)
    valid = row_indices >= 0

    row_indices = jnp.where(valid, row_indices, 0)
    weights = jnp.where(valid, weights, 0.0)

    # gather
    selected = data[row_indices]           # (K, D)

    # weighted sum
    weighted = selected * weights[:, None]
    combined = jnp.sum(weighted, axis=0)

    # normalize
    total = jnp.sum(weights)
    combined = jnp.where(total > 0, combined / total, combined)

    return combined

def _prepare_global_rows(rows, weights):
    """
    rows: (B, K)
    weights: (B, K)

    returns:
        unique_rows: (U,)
        inverse: (B, K) → indices into unique_rows
        weights_masked: (B, K)
    """
    B, K = rows.shape

    flat_rows = rows.reshape(-1)

    valid = flat_rows >= 0
    flat_rows = jnp.where(valid, flat_rows, 0.0)

    unique_rows, inverse = jnp.unique(flat_rows, return_inverse=True, size=B*K, fill_value=-1)

    inverse = jnp.where(valid, inverse, 0.0)
    
    # rebuild inverse with padding
    full_inverse = jnp.full((B * K,), -1, dtype=jnp.int32)
    full_inverse = full_inverse = jnp.where(valid, inverse, -1)
    full_inverse = full_inverse.reshape(B, K)

    weights_masked = jnp.where(rows >= 0, weights, 0.0)

    return unique_rows, full_inverse, weights_masked

@jax.jit
def batch_kernel(row_indices, weights, flux, continuum):
    flux_out = combine_rows_jax(row_indices, weights, flux)
    cont_out = combine_rows_jax(row_indices, weights, continuum)
    return flux_out, cont_out

@jax.jit
def _accumulate_batch(inverse, weights, flux_chunk, cont_chunk):
    """
    inverse: (B, K) → indices into chunk
    weights: (B, K)
    flux_chunk: (U, L)
    cont_chunk: (U, L)

    returns:
        flux_out: (B, L)
        cont_out: (B, L)
    """

    B, K = inverse.shape
    L = flux_chunk.shape[1]

    # mask invalid
    valid = inverse >= 0

    safe_idx = jnp.where(valid, inverse, 0)

    gathered_flux = flux_chunk[safe_idx]   # (B, K, L)
    gathered_cont = cont_chunk[safe_idx]

    w = jnp.where(valid, weights, 0.0)     # (B, K)
    w = w[..., None]                       # (B, K, 1)

    flux_out = jnp.sum(gathered_flux * w, axis=1)
    cont_out = jnp.sum(gathered_cont * w, axis=1)

    norm = jnp.sum(weights, axis=1, keepdims=True)
    norm = jnp.where(norm > 0, norm, 1.0)

    flux_out = flux_out / norm
    cont_out = cont_out / norm

    return flux_out, cont_out


def _parameter_helper(interpolator, parameter_values: Dict[str, Any] = None) -> ArrayLike:
    """Convert passed values to the accepted parameters format

    Args:
        parameter_values (Dict[str, Any], optional): parameter values in the format of {'parameter_name': value}. Unset parameters will be set to solar values.
        relative (bool, optional): if True, the values are treated as relative to the solar values for the abundaces 

    Returns:
        ArrayLike:
    """
    
    if not parameter_values:
        return interpolator.solar_parameters
    
    # Initialize parameters with solar values
    parameters = jnp.array(interpolator.solar_parameters)

    if parameter_values:
        if isinstance(parameter_values, dict):
            # Convert parameter names to indices for direct access
            parameter_indices = {label: i for i, label in enumerate(interpolator.stellar_parameter_names)}
            
            for label, value in parameter_values.items():
                # Get the index of the parameter
                idx = parameter_indices[label]
                parameters = parameters.at[idx].set(value)
        else:
            parameters = jnp.array(parameter_values)
    
    if not (jnp.all(parameters >= interpolator.min_stellar_parameters) and jnp.all(parameters <= interpolator.max_stellar_parameters)):
        warnings.warn("Possible exceeding parameter bonds - extrapolating.")
        
    return parameters


def _load_chunk(rowwise_data, start, end):
    flux = rowwise_data['flux'][start:end]
    cont = rowwise_data['continuum'][start:end]

    return (
        jax.device_put(np.asarray(flux)),
        jax.device_put(np.asarray(cont)),
    )
    
@jax.jit
def _accumulate_from_chunk(rows, weights, flux_chunk, cont_chunk, offset):
    """
    rows: (B, K) global indices
    weights: (B, K)
    flux_chunk: (C, L)
    offset: int (start index of chunk)
    """

    B, K = rows.shape
    C, L = flux_chunk.shape

    # map global → local indices
    local_idx = rows - offset

    # valid mask
    valid = (local_idx >= 0) & (local_idx < C)

    # clamp to safe range
    local_idx_safe = jnp.clip(local_idx, 0, C - 1)

    # gather
    gathered_flux = flux_chunk[local_idx_safe]      # (B, K, L)
    gathered_cont = cont_chunk[local_idx_safe]

    # mask invalid rows
    weights_masked = jnp.where(valid, weights, 0.0)

    # weighted sum
    flux_out = jnp.sum(gathered_flux * weights_masked[..., None], axis=1)
    cont_out = jnp.sum(gathered_cont * weights_masked[..., None], axis=1)

    return flux_out, cont_out


class LazyZarrInterpolator(SpectrumEmulator[ArrayLike]):
    def __init__(self, zarr_path, solar_parameters, params=None, device=DEVICE):
        arrays = ['wavelength', 'flux', 'continuum']
        self.store = zarr.open(zarr_path, mode='r')
        self.length = int(self.store['flux'].shape[0])
        if params is None:
            params = self.store['param_names']

        self.grid_index = GridIndex.from_dataframe(pd.read_parquet(f'{zarr_path}/index.parquet'), columns=params, device=device)
        
        self.solar_parameters = solar_parameters
        self.stellar_parameter_names = params
        # Read min/max parameters from GridIndex
        self.min_stellar_parameters = jnp.array([self.grid_index.axes[self.grid_index.columns.index(param)].min() for param in params])
        self.max_stellar_parameters = jnp.array([self.grid_index.axes[self.grid_index.columns.index(param)].max() for param in params])

        self.rowwise_data = {}
        self.static_data = {}
        
        for k in tqdm(arrays, desc='Loading data'):
            arr = self.store[k]
            if arr.ndim > 0 and arr.shape[0] == self.length:
                self.rowwise_data[k] = arr
            else:
                self.static_data[k] = jax.device_put(np.asarray(arr[:]), device=device)
        self._row_cache = {k: {} for k in self.rowwise_data}
    
    def _accumulate_all_chunks(self, rows, weights, chunk_size=1024):
        B = rows.shape[0]

        # get wavelength length
        L = self.static_data['wavelength'].shape[0]

        flux_total = jnp.zeros((B, L))
        cont_total = jnp.zeros((B, L))

        n_chunks = (self.length + chunk_size - 1) // chunk_size

        for i in range(n_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, self.length)

            flux_chunk, cont_chunk = _load_chunk(self.rowwise_data, start, end)

            flux_part, cont_part = _accumulate_from_chunk(
                rows,
                weights,
                flux_chunk,
                cont_chunk,
                start,
            )

            flux_total = flux_total + flux_part
            cont_total = cont_total + cont_part

        return flux_total, cont_total
        
    def solar_parameters(self) -> ArrayLike:
        return self.solar_parameters

    def is_in_bounds(self, parameters: ArrayLike) -> bool:
        return jnp.all(parameters >= self.min_stellar_parameters) and jnp.all(parameters <= self.max_stellar_parameters)
    
    def stellar_parameter_names(self) -> List:
        return self.stellar_parameter_names
        
    @override
    def to_parameters(self, parameters=None):
        if parameters is None:
            return self.solar_parameters
        return _parameter_helper(self, parameters)
        
    def _load_rows(self, unique_rows):
        rows_host = jax.device_get(unique_rows)

        flux = self.rowwise_data['flux'][rows_host]
        cont = self.rowwise_data['continuum'][rows_host]

        return (
            jax.device_put(flux),
            jax.device_put(cont),
        )

    def get_weighted_batch(self, queries):
        """
        queries: (B, d)
        """

        rows, weights = self.grid_index.batched_lookup(queries)

        flux_out, cont_out = self._accumulate_all_chunks(rows, weights)

        return {
            **self.static_data,
            "flux": flux_out,
            "continuum": cont_out,
        }

    def _resample_weighted_batch(self, log_wavelengths, weighted_batch):
        native_wavelengths = self.static_data['wavelength']
        target_wavelengths = jnp.power(10.0, log_wavelengths)

        interp_batch = jax.vmap(
            lambda spectrum: jnp.interp(target_wavelengths, native_wavelengths, spectrum)
        )

        return (
            interp_batch(weighted_batch['flux']),
            interp_batch(weighted_batch['continuum']),
        )


class IntensityLazyZarrInterpolator(LazyZarrInterpolator):
    def __init__(self, zarr_path, solar_parameters, params=None, device=DEVICE):
        if 'mu' not in params:
            raise ValueError("mu must be included in params")
        super().__init__(zarr_path, solar_parameters, params, device)
        self.stellar_parameter_names = [p for p in params if p != 'mu']
    
    @override
    def intensity(self, log_wavelengths, mu, parameters):
        query = jnp.concatenate([jnp.atleast_1d(parameters), jnp.atleast_1d(jnp.squeeze(mu))])
        weighted_batch = self.get_weighted_batch(jnp.atleast_2d(query))
        flux, continuum = self._resample_weighted_batch(log_wavelengths, weighted_batch)
        return jnp.stack([flux[0], continuum[0]], axis=-1)
    
    @override
    def flux(self, log_wavelengths, parameters, mus_number=20):
        roots, weights = np.polynomial.legendre.leggauss(mus_number)
        roots = (roots + 1) / 2
        weights /= 2
        return 2*jnp.pi*jnp.sum(
            jax.vmap(self.intensity, in_axes=(None, 0, None))(log_wavelengths, roots, parameters)*
            roots[:, jnp.newaxis, jnp.newaxis]*weights[:, jnp.newaxis, jnp.newaxis],
            axis=0
        )
        
        
import jax
import jax.numpy as jnp
from jax import lax


def _linear(mu, coeffs):
    u = coeffs[0]
    jax.debug.print("u: {u}, mu: {mu}, 1.0 - u * (1.0 - mu): {val}", u=u, mu=mu, val=1.0 - u * (1.0 - mu))
    return 1.0 - u * (1.0 - mu)


def _quadratic(mu, coeffs):
    u1, u2 = coeffs[0], coeffs[1]
    return 1.0 - (
        u1 * (1.0 - mu) +
        u2 * (1.0 - mu) ** 2
    )


def _nonlinear_4(mu, coeffs):
    c1, c2, c3, c4 = coeffs
    return 1.0 - (
        c1 * (1 - jnp.sqrt(mu)) +
        c2 * (1 - mu) +
        c3 * (1 - mu ** 1.5) +
        c4 * (1 - mu ** 2)
    )


# pack functions into tuple (static structure)
_LD_FUNCS = (_linear, _quadratic, _nonlinear_4)
LAW_IDS = {
    'linear': 0,
    'quadratic': 1,
    'nonlinear_4': 2,
}


def limb_darkening(mu, law_id, coeffs):
    """
    mu: [..., n_cells]
    law_id: int (0=linear, 1=quadratic, 2=nonlinear_4)
    coeffs: array-like (max size, unused entries ignored)
    """
    return lax.switch(law_id, _LD_FUNCS, mu, coeffs)
        
def get_limb_darkening_law_id(law: str) -> int:
    return LAW_IDS[law]


class FluxLazyZarrInterpolator(LazyZarrInterpolator):
    def __init__(self, zarr_path, solar_parameters, params=None, device=DEVICE, limb_darkening_law=None, limb_darkening_coeffs=None):
        super().__init__(zarr_path, solar_parameters, params, device)
        if limb_darkening_law is None:
            self.limb_darkening_law = get_limb_darkening_law_id('linear')
        else:
            self.limb_darkening_law = get_limb_darkening_law_id(limb_darkening_law)
        if limb_darkening_coeffs is None:
            self.limb_darkening_coeffs = jnp.array([0.6, 0.6, 0., 0.])
        else:
            coeffs = jnp.atleast_1d(limb_darkening_coeffs)
            # Pad to length 4 with zeros
            if coeffs.shape[0] < 4:
                coeffs = jnp.pad(coeffs, (0, 4 - coeffs.shape[0]), constant_values=0.0)
            self.limb_darkening_coeffs = coeffs
        
    def apply_limb_darkening(self, flux, mu):
        return flux * limb_darkening(mu, self.limb_darkening_law, self.limb_darkening_coeffs)
    
    @override
    def intensity(self, log_wavelengths, mu, parameters):
        weighted_batch = self.get_weighted_batch(jnp.atleast_2d(parameters))
        flux, continuum = self._resample_weighted_batch(log_wavelengths, weighted_batch)
        mu_value = jnp.squeeze(mu)
        return jnp.stack([self.apply_limb_darkening(flux[0], mu_value),
                          self.apply_limb_darkening(continuum[0], mu_value)], axis=-1)
    
    @override
    def flux(self, log_wavelengths, parameters, mus_number=20):
        weighted_batch = self.get_weighted_batch(jnp.atleast_2d(parameters))
        flux, continuum = self._resample_weighted_batch(log_wavelengths, weighted_batch)
        return jnp.stack([flux[0], continuum[0]], axis=-1)
