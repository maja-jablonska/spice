from typing import Any, Dict, List
import inspect as _inspect
import time as _time
import sys as _sys
import warnings

from spice.spectrum.parameters import parameter_helper

_jax_already_loaded = "jax" in _sys.modules
_t0 = _time.perf_counter()
import jax
from jax.typing import ArrayLike
import jax.numpy as jnp
from jax import lax
if not _jax_already_loaded:
    from spice.utils import log as _log
    _log.info(f"JAX loaded in {_time.perf_counter() - _t0:.1f} s")

# `vmap_method` was added to jax.pure_callback in JAX 0.4.32 (replacing the
# older `vectorized` kwarg). Pin the right hint at import time so we don't
# blow up on environments stuck on an older JAX (the kwarg would otherwise
# get treated as a callback argument and JAX rejects the str at trace time).
_PURE_CALLBACK_PARAMS = _inspect.signature(jax.pure_callback).parameters
if "vmap_method" in _PURE_CALLBACK_PARAMS:
    _BROADCAST_ALL_KW: Dict[str, Any] = {"vmap_method": "broadcast_all"}
elif "vectorized" in _PURE_CALLBACK_PARAMS:
    _BROADCAST_ALL_KW = {"vectorized": True}
else:
    _BROADCAST_ALL_KW = {}

import numpy as np
from functools import partial, lru_cache
from overrides import override

from spice.spectrum.spectrum_emulator import SpectrumEmulator


def _default_device():
    return jax.devices()[0]


def _float_dtype():
    return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


def _np_float_dtype():
    return np.float64 if jax.config.jax_enable_x64 else np.float32

@jax.jit
def _axis_linear_base(values, value, atol=1e-6, rtol=1e-6):
    n = jnp.int32(values.shape[0])

    # --- match detection (no dynamic shapes) ---
    mask = jnp.isclose(values, value, atol=atol, rtol=rtol)
    has_match = jnp.any(mask)

    # first True index (safe because argmax returns 0 if all False)
    idx = jnp.int32(jnp.argmax(mask))

    # --- match case ---
    def match_case(_):
        def single(_):
            return idx, idx

        def multiple(_):
            return lax.cond(
                idx == 0,
                lambda _: (jnp.int32(0), jnp.int32(1)),
                lambda _: lax.cond(
                    idx == n - 1,
                    lambda _: (n - jnp.int32(2), n - jnp.int32(1)),
                    lambda _: (idx - jnp.int32(1), idx + jnp.int32(1)),
                    operand=None,
                ),
                operand=None,
            )

        return lax.cond(n == 1, single, multiple, operand=None)

    # --- no match case ---
    def no_match_case(_):
        hi = jnp.int32(jnp.searchsorted(values, value, side="right"))
        hi = jnp.clip(hi, jnp.int32(1), n - jnp.int32(1))
        return hi - jnp.int32(1), hi

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


@lru_cache(maxsize=16)
def _make_corners_np(d):
    return np.array(list(np.ndindex(*(2,) * d)), dtype=np.int32)


def make_corners(d):
    return jnp.array(_make_corners_np(d))


@jax.jit
def _sparse_lookup_single(keys, values, query_key):
    """Lookup a single key in the sorted keys array. Returns row index or -1."""
    pos = jnp.searchsorted(keys, query_key)
    pos = jnp.clip(pos, 0, keys.shape[0] - 1)
    found = keys[pos] == query_key
    return jnp.where(found, values[pos], jnp.int32(-1))


@partial(jax.jit, static_argnums=(4,))
def sparse_weighted_rows_from_brackets(keys, values, strides, brackets, corners_dimension):
    """
    keys: (S,) int64, sorted encoded multi-dim indices
    values: (S,) int32, row indices
    strides: (d,) int64, mixed-radix strides
    brackets: (d, 3) array -> [lo, hi, t] per axis
    corners_dimension: int

    returns:
        rows: (2^d,) int32
        weights: (2^d,) float
    """
    brackets = jnp.asarray(brackets)
    lo = brackets[:, 0].astype(jnp.int32)
    hi = brackets[:, 1].astype(jnp.int32)
    t = brackets[:, 2]

    corners = make_corners(corners_dimension)

    # grid indices for each corner
    indices = jnp.where(corners == 0, lo, hi)  # (2^d, d)

    # encode each corner to a flat key
    corner_keys = jnp.dot(indices.astype(jnp.int64), strides)  # (2^d,)

    # lookup each corner
    rows = jax.vmap(lambda k: _sparse_lookup_single(keys, values, k))(corner_keys)  # (2^d,)

    # multilinear weights
    weights = jnp.where(corners == 0, 1.0 - t, t)
    weights = jnp.prod(weights, axis=1)  # (2^d,)

    # mask invalid
    valid = (weights > 0.0) & (rows >= 0)
    rows = jnp.where(valid, rows, -1)
    weights = jnp.where(valid, weights, 0.0)

    # normalize
    total = jnp.sum(weights)
    weights = jnp.where(total > 0, weights / total, weights)

    return rows, weights


@partial(jax.jit, static_argnums=(4,))
def sparse_weighted_rows_from_brackets_batched(keys, values, strides, brackets, d):
    """
    brackets: (B, d, 3)
    returns: rows (B, 2^d), weights (B, 2^d)
    """
    def per_query(b):
        return sparse_weighted_rows_from_brackets(keys, values, strides, b, d)

    return jax.vmap(per_query)(brackets)


class SparseGridIndex:
    """Grid index using sorted flat keys instead of a dense N-d array.

    Works with grids of many parameters where a dense array would be
    too large to allocate.
    """

    def __init__(self, keys, values, axes, columns, strides, device=None):
        device = device or _default_device()
        self.axes_np = [np.asarray(ax) for ax in axes]
        self.columns = columns

        self.axes = tuple(
            jax.device_put(ax, device=device) for ax in self.axes_np
        )
        self.keys = jax.device_put(keys, device=device)
        self.values = jax.device_put(values, device=device)
        self.strides = jax.device_put(strides, device=device)

        self.d = len(self.axes)
        self._num_corners = 2 ** self.d
        self.num_rows = int(np.max(values) + 1) if len(values) > 0 else 0

        self.keys.block_until_ready()

    @classmethod
    def from_dataframe(cls, df, columns, device=None):
        device = device or _default_device()
        print('Building sparse grid index...', flush=True)

        axes = [np.unique(df[c].to_numpy()) for c in columns]

        # compute mixed-radix strides
        axis_lengths = [len(ax) for ax in axes]
        strides = np.ones(len(columns), dtype=np.int64)
        for i in range(len(columns) - 2, -1, -1):
            strides[i] = strides[i + 1] * axis_lengths[i + 1]

        # check for int64 overflow
        total_size = int(strides[0]) * axis_lengths[0] if axis_lengths else 0
        assert total_size >= 0, (
            f"Grid too large for int64 encoding: strides would overflow. "
            f"Axis lengths: {axis_lengths}"
        )

        # compute multi-dim indices for each row
        indices = np.stack([
            np.searchsorted(axes[i], df[columns[i]].to_numpy())
            for i in range(len(columns))
        ], axis=1)  # (S, d)

        # encode to flat keys
        keys = indices.astype(np.int64) @ strides  # (S,)
        row_idx = df.row_idx.to_numpy(dtype=np.int32, copy=False)

        # sort by key
        order = np.argsort(keys)
        keys = keys[order]
        row_idx = row_idx[order]

        print('Sparse grid index ready.', flush=True)
        return cls(keys, row_idx, axes, columns, strides, device=device)

    @partial(jax.jit, static_argnums=(0,))
    def lookup(self, query_values, atol=1e-6, rtol=1e-6):
        brackets = _compute_brackets(self.axes, query_values, atol, rtol)

        rows, weights = sparse_weighted_rows_from_brackets(
            self.keys, self.values, self.strides,
            brackets, self.d,
        )

        return rows, weights

    def batched_lookup(self, queries, atol=1e-6, rtol=1e-6):
        brackets = compute_brackets_batched(self.axes, queries, atol, rtol)

        rows, weights = sparse_weighted_rows_from_brackets_batched(
            self.keys, self.values, self.strides,
            brackets, self.d,
        )

        return rows, weights


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
    def __init__(self, grid, axes, columns, device=None):
        """
        axes: list/tuple of 1D arrays (one per dimension)
        """
        device = device or _default_device()
        self.axes_np = [np.asarray(ax) for ax in axes]
        self.grid_np = np.asarray(grid)
        self.columns = columns

        self.axes = tuple(
            jax.device_put(ax, device=device) for ax in self.axes_np
        )
        self.grid = jax.device_put(self.grid_np, device=device)

        self.d = len(self.axes)
        self._num_corners = 2 ** self.d
        self.num_rows = int(self.grid.max() + 1)

        self.grid.block_until_ready()

    @classmethod
    def from_dataframe(cls, df, columns, device=None):
        """
        columns: list of column names defining the grid axes
        """
        device = device or _default_device()
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

        rows = jnp.nonzero(weights_dense > 0, size=self._num_corners, fill_value=-1)[0]
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

        max_corners = self._num_corners

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


class LazyZarrInterpolator(SpectrumEmulator[ArrayLike]):
    def __init__(self, zarr_path, solar_parameters, params=None, device=None, sparse=True,
                 in_memory=False, in_memory_threshold_bytes=2 * 1024 ** 3):
        """
        in_memory: True forces rowwise arrays (flux, continuum) onto the JAX device;
            False keeps them lazy on the zarr store (default); "auto" loads them
            eagerly only if their combined size is <= in_memory_threshold_bytes.
        """
        from spice.utils import log

        _zarr_loaded = "zarr" in _sys.modules
        _t = _time.perf_counter()
        import zarr
        if not _zarr_loaded:
            log.info(f"zarr loaded in {_time.perf_counter() - _t:.1f} s")

        _pd_loaded = "pandas" in _sys.modules
        _t = _time.perf_counter()
        import pandas as pd
        if not _pd_loaded:
            log.info(f"pandas loaded in {_time.perf_counter() - _t:.1f} s")

        from tqdm.auto import tqdm

        if in_memory not in (True, False, "auto"):
            raise ValueError(f"in_memory must be True, False, or 'auto'; got {in_memory!r}")

        device = device or _default_device()
        # A tqdm bar runs inside this block, so don't try to substitute the
        # start line in place -- print the completion on a new line instead.
        with log.timed(
            f"Loading spectral grid from {zarr_path}",
            "Spectral grid loaded in {elapsed:.1f} s",
            inplace=False,
        ):
            arrays = ['wavelength', 'flux', 'continuum']
            self.store = zarr.open(zarr_path, mode='r')
            self.length = int(self.store['flux'].shape[0])
            if params is None:
                params = self.store['param_names']

            df = pd.read_parquet(f'{zarr_path}/index.parquet')
            if sparse:
                self.grid_index = SparseGridIndex.from_dataframe(df, columns=params, device=device)
            else:
                axis_lengths = [int(df[c].nunique()) for c in params]
                dense_bytes = int(np.prod(axis_lengths)) * 4  # int32 per cell
                dense_threshold = 1 * 1024 ** 3
                if dense_bytes > dense_threshold:
                    warnings.warn(
                        f"Dense grid index requires {dense_bytes / 1e9:.2f} GB "
                        f"(axis lengths: {axis_lengths}). Consider sparse=True to "
                        f"avoid allocating a large N-d array."
                    )
                self.grid_index = GridIndex.from_dataframe(df, columns=params, device=device)

            solar_arr = np.asarray(solar_parameters)
            if solar_arr.shape[-1] != len(params):
                raise ValueError(
                    f"solar_parameters must have length {len(params)} to match params "
                    f"{list(params)}; got {solar_arr.shape[-1]}"
                )
            self._solar_parameters = solar_arr
            self._stellar_parameter_names = params
            # Read min/max parameters from GridIndex
            self.min_stellar_parameters = jnp.array([self.grid_index.axes[self.grid_index.columns.index(param)].min() for param in params])
            self.max_stellar_parameters = jnp.array([self.grid_index.axes[self.grid_index.columns.index(param)].max() for param in params])

            rowwise_arrays = {}
            static_arrays = {}
            for k in arrays:
                arr = self.store[k]
                if arr.ndim > 0 and arr.shape[0] == self.length:
                    rowwise_arrays[k] = arr
                else:
                    static_arrays[k] = arr

            rowwise_bytes = sum(
                int(np.prod(a.shape)) * np.dtype(a.dtype).itemsize
                for a in rowwise_arrays.values()
            )
            if in_memory is True:
                load_rowwise_eagerly = True
                if rowwise_bytes > in_memory_threshold_bytes:
                    warnings.warn(
                        f"Eager in_memory load requested for {rowwise_bytes / 1e9:.2f} GB "
                        f"of rowwise data, which exceeds the {in_memory_threshold_bytes / 1e9:.2f} GB "
                        f"threshold and may cause an out-of-memory crash. Consider "
                        f"in_memory='auto' or in_memory=False."
                    )
            elif in_memory == "auto":
                load_rowwise_eagerly = rowwise_bytes <= in_memory_threshold_bytes
                log.info(
                    f"Rowwise grid size {rowwise_bytes / 1e6:.1f} MB; "
                    f"threshold {in_memory_threshold_bytes / 1e6:.1f} MB; "
                    f"{'eager' if load_rowwise_eagerly else 'lazy'} load."
                )
            else:
                load_rowwise_eagerly = False

            self._rowwise_in_memory = load_rowwise_eagerly
            self.rowwise_data = {}
            self.static_data = {}

            for k in tqdm(arrays, desc='[spice] Loading data'):
                if k in static_arrays:
                    self.static_data[k] = jax.device_put(
                        np.asarray(static_arrays[k][:]), device=device
                    )
                else:
                    arr = rowwise_arrays[k]
                    if load_rowwise_eagerly:
                        self.rowwise_data[k] = jax.device_put(
                            np.asarray(arr[:]), device=device
                        )
                    else:
                        self.rowwise_data[k] = arr

    @property
    def stellar_parameter_names(self):
        return self._stellar_parameter_names

    @property
    def parameter_names(self):
        return self._stellar_parameter_names

    @property
    def solar_parameters(self):
        return self._solar_parameters

    def _accumulate_in_memory(self, rows, weights):
        # Peak memory is (K, L) per step instead of (B, K, L) — the gather +
        # reduce is mapped over the batch axis with lax.map. For B=1000, K=16,
        # L=100k that's ~6 GB → ~6 MB.
        flux_data = self.rowwise_data['flux']
        cont_data = self.rowwise_data['continuum']
        out_dtype = _float_dtype()

        def per_query(rw):
            r, w = rw
            valid = r >= 0
            safe_rows = jnp.where(valid, r, 0)
            safe_weights = jnp.where(valid, w, 0.0).astype(flux_data.dtype)
            w_col = safe_weights[:, None]
            flux_sel = flux_data[safe_rows]
            cont_sel = cont_data[safe_rows]
            flux_acc = jnp.sum(flux_sel * w_col, axis=0).astype(out_dtype)
            cont_acc = jnp.sum(cont_sel * w_col, axis=0).astype(out_dtype)
            return flux_acc, cont_acc

        return jax.lax.map(per_query, (rows, weights))

    def _accumulate_direct(self, rows, weights):
        L = self.static_data['wavelength'].shape[0]
        rowwise_data = self.rowwise_data
        np_dtype = _np_float_dtype()
        jax_dtype = _float_dtype()

        def _host_fn(rows_np, weights_np):
            # broadcast_all: vmap adds a leading batch dim → (vmap_B, B_inner, K)
            # non-vmapped: (B_inner, K)
            orig_shape = rows_np.shape
            K = orig_shape[-1]
            if rows_np.ndim == 3:
                vmap_B, B_inner = orig_shape[0], orig_shape[1]
                rows_flat = rows_np.reshape(-1, K)
                weights_flat = weights_np.reshape(-1, K)
            else:
                vmap_B = None
                B_inner = orig_shape[0]
                rows_flat = rows_np
                weights_flat = weights_np

            total_B = rows_flat.shape[0]
            valid_mask = rows_flat >= 0
            unique_rows = np.unique(rows_flat[valid_mask])

            if len(unique_rows) == 0:
                flux_out = np.zeros((total_B, L), dtype=np_dtype)
                cont_out = np.zeros((total_B, L), dtype=np_dtype)
            else:
                flux_subset = np.asarray(rowwise_data['flux'][unique_rows])  # (U, L)
                cont_subset = np.asarray(rowwise_data['continuum'][unique_rows])  # (U, L)

                local_indices = np.searchsorted(unique_rows, np.where(valid_mask, rows_flat, 0))
                safe_idx = np.where(valid_mask, local_indices, 0)
                w = np.where(valid_mask, weights_flat, weights_flat.dtype.type(0))

                # Scatter weights into (total_B, U) then matmul with (U, L).
                # Avoids materialising the (total_B, K, L) intermediate which is
                # total_B * 2^d * L in size and causes OOM for large chunk_size/d.
                U = len(unique_rows)
                weights_per_row = np.zeros((total_B, U), dtype=np_dtype)
                b_idx = np.repeat(np.arange(total_B), K)
                u_idx = safe_idx.ravel()
                w_flat = w.ravel().astype(np_dtype)
                valid_flat = valid_mask.ravel()
                np.add.at(weights_per_row, (b_idx[valid_flat], u_idx[valid_flat]), w_flat[valid_flat])

                flux_out = (weights_per_row @ flux_subset).astype(np_dtype)   # (total_B, L)
                cont_out = (weights_per_row @ cont_subset).astype(np_dtype)   # (total_B, L)

            if vmap_B is not None:
                return (flux_out.reshape(vmap_B, B_inner, L),
                        cont_out.reshape(vmap_B, B_inner, L))
            return flux_out, cont_out

        B = rows.shape[0]
        result_shape = (
            jax.ShapeDtypeStruct((B, L), jax_dtype),
            jax.ShapeDtypeStruct((B, L), jax_dtype),
        )
        return jax.pure_callback(_host_fn, result_shape, rows, weights, **_BROADCAST_ALL_KW)

    def is_in_bounds(self, parameters: ArrayLike) -> bool:
        return jnp.all(parameters >= self.min_stellar_parameters) and jnp.all(parameters <= self.max_stellar_parameters)
    
    @override
    def to_parameters(self, parameters=None):
        if parameters is None:
            return self.solar_parameters
        return parameter_helper(self, parameters)
        
    def get_weighted_batch(self, queries):
        """
        queries: (B, d)
        """

        rows, weights = self.grid_index.batched_lookup(queries)

        if self._rowwise_in_memory:
            flux_out, cont_out = self._accumulate_in_memory(rows, weights)
        else:
            flux_out, cont_out = self._accumulate_direct(rows, weights)

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
    def __init__(self, zarr_path, solar_parameters, params=None, device=None, sparse=True,
                 in_memory=False, in_memory_threshold_bytes=2 * 1024 ** 3):
        if 'mu' not in params:
            raise ValueError("mu must be included in params")
        super().__init__(zarr_path, solar_parameters, params, device, sparse=sparse,
                         in_memory=in_memory, in_memory_threshold_bytes=in_memory_threshold_bytes)
        self._stellar_parameter_names = [p for p in params if p != 'mu']
        self.min_stellar_parameters = jnp.array([self.grid_index.axes[self.grid_index.columns.index(p)].min() for p in self._stellar_parameter_names])
        self.max_stellar_parameters = jnp.array([self.grid_index.axes[self.grid_index.columns.index(p)].max() for p in self._stellar_parameter_names])

        mu_idx = params.index('mu')
        self._solar_parameters = np.delete(np.asarray(self._solar_parameters), mu_idx)
        
    

    @override
    def to_parameters(self, parameters=None):
        if parameters is None:
            return self.solar_parameters
        if isinstance(parameters, dict):
            parameters = {k: v for k, v in parameters.items() if k != 'mu'}
        return parameter_helper(self, parameters)
    
    @override
    def parameter_names(self):
        return self._stellar_parameter_names + ['mu']

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



from spice.spectrum.flux_limb_darkening import apply_flux_limb_darkening


class FluxLazyZarrInterpolator(LazyZarrInterpolator):
    def __init__(self, zarr_path, solar_parameters, params=None, device=None, sparse=True,
                 in_memory=False, in_memory_threshold_bytes=2 * 1024 ** 3):
        super().__init__(zarr_path, solar_parameters, params, device, sparse=sparse,
                         in_memory=in_memory, in_memory_threshold_bytes=in_memory_threshold_bytes)

    @override
    def intensity(self, log_wavelengths, mu, parameters, ld_law="linear", ld_coeffs=None):
        """Specific intensity at angle ``mu`` derived from the disc-integrated
        flux via flux-conservation limb darkening.

        Args:
            log_wavelengths: log10(Angstrom) sample points.
            mu: cosine of the angle from disc centre.
            parameters: stellar parameters in training order.
            ld_law: name of the limb-darkening law (``linear``, ``quadratic``,
                ``nonlinear_4``).
            ld_coeffs: coefficients for the chosen law. Defaults to
                ``[0.6, 0.6, 0, 0]`` (typical solar-type linear LD).
        """
        weighted_batch = self.get_weighted_batch(jnp.atleast_2d(parameters))
        flux, continuum = self._resample_weighted_batch(log_wavelengths, weighted_batch)
        mu_value = jnp.squeeze(mu)
        return jnp.stack([apply_flux_limb_darkening(flux[0], mu_value, ld_law, ld_coeffs),
                          apply_flux_limb_darkening(continuum[0], mu_value, ld_law, ld_coeffs)],
                         axis=-1)

    @override
    def flux(self, log_wavelengths, parameters, mus_number=20):
        weighted_batch = self.get_weighted_batch(jnp.atleast_2d(parameters))
        flux, continuum = self._resample_weighted_batch(log_wavelengths, weighted_batch)
        return jnp.stack([flux[0], continuum[0]], axis=-1)
