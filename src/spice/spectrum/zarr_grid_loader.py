import logging
import time

import jax
import jax.numpy as jnp
import numpy as np
import zarr

logger = logging.getLogger(__name__)


def _zarr_row_chunk_size(arr):
    """Best-effort extraction of the first-dimension chunk size."""
    chunks = getattr(arr, "chunks", None)
    if not chunks:
        return None
    try:
        row_chunk = int(chunks[0])
    except (TypeError, ValueError):
        return None
    return row_chunk if row_chunk > 0 else None


def _effective_chunk_rows(requested_chunk_rows, n_rows_total, row_chunk):
    """Choose a row block size aligned with zarr chunks when possible."""
    if requested_chunk_rows <= 0:
        raise ValueError(
            f"chunk_rows must be a positive integer, got {requested_chunk_rows}."
        )
    if n_rows_total <= 0:
        return 0
    if row_chunk is None:
        return min(requested_chunk_rows, n_rows_total)

    # Keep read windows chunk-aligned to avoid re-decoding partial chunks.
    n_chunks = (requested_chunk_rows + row_chunk - 1) // row_chunk
    aligned_rows = max(row_chunk, n_chunks * row_chunk)
    return min(aligned_rows, n_rows_total)


def _load_dense_2d_array(arr, name, chunk_rows):
    """Load a 2D zarr array into a dense numpy array in chunk-aligned row blocks."""
    if len(arr.shape) != 2:
        raise ValueError(f"Expected 2D array for '{name}', got shape {arr.shape}.")

    n_rows_total, n_cols = int(arr.shape[0]), int(arr.shape[1])
    if n_rows_total == 0:
        return np.empty((0, n_cols), dtype=arr.dtype)

    zarr_row_chunk = _zarr_row_chunk_size(arr)
    effective_rows = _effective_chunk_rows(
        requested_chunk_rows=int(chunk_rows),
        n_rows_total=n_rows_total,
        row_chunk=zarr_row_chunk,
    )

    logger.info(
        "Loading %s: shape=%s, dtype=%s, requested_chunk_rows=%d, zarr_row_chunk=%s, effective_chunk_rows=%d",
        name,
        arr.shape,
        arr.dtype,
        chunk_rows,
        zarr_row_chunk if zarr_row_chunk is not None else "unknown",
        effective_rows,
    )

    out = np.empty((n_rows_total, n_cols), dtype=arr.dtype)
    t0 = time.perf_counter()
    for start in range(0, n_rows_total, effective_rows):
        end = min(start + effective_rows, n_rows_total)
        logger.info(
            "Loading %s rows %d-%d of %d (%.0f%%)...",
            name,
            start,
            end,
            n_rows_total,
            100 * end / n_rows_total,
        )
        out[start:end, :] = np.asarray(arr[start:end, :])

    elapsed = max(time.perf_counter() - t0, 1e-9)
    size_mb = out.nbytes / 1e6
    logger.info(
        "%s loaded: shape=%s, dtype=%s, size=%.2f MB in %.2fs (%.2f MB/s)",
        name.capitalize(),
        out.shape,
        out.dtype,
        size_mb,
        elapsed,
        size_mb / elapsed,
    )
    return out


def validate_query_params(query_params, axes, axis_names=None):
    """Check that query_params have correct shape and are within grid bounds."""
    n_dims = len(axes)
    axis_names = axis_names or [f"axis_{i}" for i in range(n_dims)]

    # Handle both single (n_dims,) and batch (batch, n_dims) queries
    if query_params.ndim == 1:
        if query_params.shape[0] != n_dims:
            raise ValueError(
                f"query_params has {query_params.shape[0]} values but grid has {n_dims} varying axes "
                f"({axis_names}). Expected shape ({n_dims},)."
            )
        to_check = np.asarray(query_params).reshape(1, -1)
    else:
        if query_params.shape[1] != n_dims:
            raise ValueError(
                f"query_params has {query_params.shape[1]} columns but grid has {n_dims} varying axes "
                f"({axis_names}). Expected shape (batch, {n_dims})."
            )
        to_check = np.asarray(query_params)

    for d in range(n_dims):
        ax = np.asarray(axes[d])
        lo, hi = float(ax[0]), float(ax[-1])
        vals = to_check[:, d]
        if np.any(vals < lo) or np.any(vals > hi):
            raise ValueError(
                f"Parameter '{axis_names[d]}' out of bounds: valid range [{lo}, {hi}], "
                "got values outside this range."
            )


def _cell_weights_and_start_indices(query_params, axes):
    n_dims = len(axes)
    weights = []
    start_indices = []

    for d in range(n_dims):
        ax = axes[d]
        # searchsorted can be backend/version sensitive; this is a jit-safe equivalent.
        idx = jnp.sum(query_params[d] >= ax) - 1
        idx = jnp.clip(idx, 0, ax.shape[0] - 2)
        lo = jax.lax.dynamic_index_in_dim(ax, idx, keepdims=False)
        hi = jax.lax.dynamic_index_in_dim(ax, idx + 1, keepdims=False)
        t = (query_params[d] - lo) / (hi - lo)
        t = jnp.clip(t, 0.0, 1.0)
        weights.append(t)
        start_indices.append(idx.astype(jnp.int32))

    return weights, start_indices


def _interpolate_from_subcube(subcube, weights):
    for t in weights:
        subcube = subcube[0] * (1.0 - t) + subcube[1] * t
    return subcube


def _interpolate_spectrum_impl(query_params, axes, flux_cube):
    n_dims = len(axes)
    weights, start_indices = _cell_weights_and_start_indices(query_params, axes)

    start = jnp.empty((n_dims + 1,), dtype=jnp.int32)
    for d in range(n_dims):
        start = start.at[d].set(start_indices[d])
    start = start.at[n_dims].set(0)

    sizes = tuple([2] * n_dims + [flux_cube.shape[-1]])
    subcube = jax.lax.dynamic_slice(flux_cube, start, sizes)
    return _interpolate_from_subcube(subcube, weights)


def _interpolate_spectrum_and_continuum_impl(query_params, axes, flux_cube, continuum_cube):
    n_dims = len(axes)
    weights, start_indices = _cell_weights_and_start_indices(query_params, axes)

    start = jnp.empty((n_dims + 1,), dtype=jnp.int32)
    for d in range(n_dims):
        start = start.at[d].set(start_indices[d])
    start = start.at[n_dims].set(0)

    sizes = tuple([2] * n_dims + [flux_cube.shape[-1]])
    flux_subcube = jax.lax.dynamic_slice(flux_cube, start, sizes)
    continuum_subcube = jax.lax.dynamic_slice(continuum_cube, start, sizes)

    return (
        _interpolate_from_subcube(flux_subcube, weights),
        _interpolate_from_subcube(continuum_subcube, weights),
    )


_interpolate_spectrum_core = jax.jit(_interpolate_spectrum_impl)
batch_interpolate_core = jax.jit(jax.vmap(_interpolate_spectrum_impl, in_axes=(0, None, None)))
_interpolate_spectrum_and_continuum_core = jax.jit(_interpolate_spectrum_and_continuum_impl)
batch_interpolate_spectrum_and_continuum_core = jax.jit(
    jax.vmap(_interpolate_spectrum_and_continuum_impl, in_axes=(0, None, None, None))
)


def interpolate_spectrum(query_params, axes, flux_cube, axis_names=None, validate=True):
    """Interpolate spectrum at query_params. Validates shape and bounds by default."""
    query_params = jnp.asarray(query_params)
    if validate:
        validate_query_params(query_params, axes, axis_names)
    return _interpolate_spectrum_core(query_params, axes, flux_cube)


def batch_interpolate(queries, axes, flux_cube, axis_names=None, validate=True):
    """Batch interpolate spectra. Validates shape and bounds by default."""
    queries = jnp.asarray(queries)
    if validate:
        validate_query_params(queries, axes, axis_names)
    return batch_interpolate_core(queries, axes, flux_cube)


def interpolate_spectrum_and_continuum(
    query_params, axes, flux_cube, continuum_cube, axis_names=None, validate=True
):
    """Interpolate both flux and continuum at query_params. Returns (spectrum, continuum)."""
    query_params = jnp.asarray(query_params)
    if validate:
        validate_query_params(query_params, axes, axis_names)
    return _interpolate_spectrum_and_continuum_core(query_params, axes, flux_cube, continuum_cube)


def batch_interpolate_spectrum_and_continuum(
    queries, axes, flux_cube, continuum_cube, axis_names=None, validate=True
):
    """Batch interpolate flux and continuum. Returns (spectra, continua)."""
    queries = jnp.asarray(queries)
    if validate:
        validate_query_params(queries, axes, axis_names)
    return batch_interpolate_spectrum_and_continuum_core(
        queries, axes, flux_cube, continuum_cube
    )


def _build_axis_fixed_interpolators(axes):
    fixed_axes = tuple(jnp.asarray(ax) for ax in axes)

    def _single(query_params, flux_cube):
        return _interpolate_spectrum_impl(query_params, fixed_axes, flux_cube)

    def _single_pair(query_params, flux_cube, continuum_cube):
        return _interpolate_spectrum_and_continuum_impl(
            query_params, fixed_axes, flux_cube, continuum_cube
        )

    single = jax.jit(_single)
    batch = jax.jit(jax.vmap(_single, in_axes=(0, None)))
    single_pair = jax.jit(_single_pair)
    batch_pair = jax.jit(jax.vmap(_single_pair, in_axes=(0, None, None)))
    return single, batch, single_pair, batch_pair


class ZarrGridLoader:
    def __init__(self, store_path, chunk_rows=500):
        """
        Args:
            store_path: Path to the zarr store.
            chunk_rows: Rows per chunk when loading flux/continuum. Smaller values
                reduce risk of "Operation canceled" on slow/network storage. Default 500.
        """
        chunk_rows = int(chunk_rows)
        if chunk_rows <= 0:
            raise ValueError(f"chunk_rows must be a positive integer, got {chunk_rows}.")

        logger.info("Opening zarr store at %s", store_path)
        self.store = zarr.open_group(store_path)
        logger.info("Store opened successfully")

        logger.info("Loading param_names...")
        param_names = list(self.store["param_names"][:])
        param_names = [pn.decode() if hasattr(pn, "decode") else str(pn) for pn in param_names]
        logger.info("Loaded %d param names: %s", len(param_names), param_names)

        flux_arr = self.store["flux"]
        n_rows_total = flux_arr.shape[0]
        flux = _load_dense_2d_array(flux_arr, name="flux", chunk_rows=chunk_rows)

        continuum_arr = self.store["continuum"]
        if continuum_arr.shape[0] != n_rows_total:
            raise ValueError(
                "Flux and continuum row counts must match, got "
                f"{n_rows_total} and {continuum_arr.shape[0]}."
            )
        continuum = _load_dense_2d_array(
            continuum_arr, name="continuum", chunk_rows=chunk_rows
        )

        logger.info("Loading wavelength array...")
        wavelength = np.asarray(self.store["wavelength"][:])
        logger.info("Wavelength loaded: shape=%s, dtype=%s", wavelength.shape, wavelength.dtype)

        logger.info("Loading params array...")
        params = np.asarray(self.store["params"][:])
        logger.info("Params loaded: shape=%s, dtype=%s", params.shape, params.dtype)

        mu_selected = None
        if "mu_selected" in self.store:
            logger.info("Loading mu_selected array...")
            mu_selected = np.asarray(self.store["mu_selected"][:], dtype=np.float32)
            logger.info(
                "mu_selected loaded: shape=%s, dtype=%s",
                mu_selected.shape,
                mu_selected.dtype,
            )

        n_rows = flux.shape[0]
        logger.info("Total grid rows: %d", n_rows)

        logger.info("Discovering varying axes...")
        varying = {}
        for i, name in enumerate(param_names):
            unique = np.unique(params[:, i])
            if len(unique) > 1:
                varying[name] = (i, unique)
        logger.info("Found %d varying parameters: %s", len(varying), list(varying.keys()))

        varying_sorted = sorted(varying.items(), key=lambda x: -len(x[1][1]))
        axis_names = []
        axes = []
        col_indices = []
        product = 1
        for name, (col_idx, unique_vals) in varying_sorted:
            n_ax = len(unique_vals)
            if product * n_ax <= n_rows:
                axis_names.append(name)
                axes.append(unique_vals)
                col_indices.append(col_idx)
                product *= n_ax
                logger.debug("Added axis '%s': %d values, product so far=%d", name, n_ax, product)
            else:
                logger.debug("Skipping axis '%s' (would exceed n_rows)", name)
                break

        if product != n_rows:
            raise ValueError(
                f"Params do not form a dense rectilinear grid: {n_rows} rows but "
                f"product of varying axis lengths = {product}. "
                "Use LazyZarrSpectrumEmulator for sparse/unstructured parameter grids."
            )

        logger.info("Sorting grid by axis order...")
        sort_keys = [params[:, c] for c in reversed(col_indices)]
        order = np.lexsort(sort_keys)
        logger.info("Sort order computed")

        grid_shape = tuple(len(ax) for ax in axes)
        logger.info("Reshaping flux to grid shape %s...", grid_shape)
        flux_cube = flux[order].reshape(*grid_shape, -1)
        logger.info("Reshaping continuum to grid shape %s...", grid_shape)
        continuum_cube = continuum[order].reshape(*grid_shape, -1)

        mu_selected_cube = None
        if mu_selected is not None:
            if mu_selected.shape[0] != n_rows_total:
                raise ValueError(
                    "mu_selected row count must match flux/params rows, got "
                    f"{mu_selected.shape[0]} and {n_rows_total}."
                )
            logger.info("Reshaping mu_selected to grid shape %s...", grid_shape)
            mu_selected_cube = mu_selected[order].reshape(*grid_shape, 1)

        logger.info("Grid axes: %s", axis_names)
        logger.info("Grid shape: %s", grid_shape)
        logger.info("Flux cube shape: %s", flux_cube.shape)
        logger.info("Continuum cube shape: %s", continuum_cube.shape)

        logger.info("Converting to JAX arrays and building interpolators...")
        self.axis_names = axis_names
        self.axes_jnp = tuple(jnp.asarray(ax) for ax in axes)
        self.flux_cube_jnp = jnp.asarray(flux_cube)
        self.continuum_cube_jnp = jnp.asarray(continuum_cube)
        self.wavelength_jnp = jnp.asarray(wavelength)
        self.mu_selected_cube_jnp = (
            jnp.asarray(mu_selected_cube) if mu_selected_cube is not None else None
        )

        (
            self._interpolate_spectrum_jit,
            self._batch_interpolate_spectrum_jit,
            self._interpolate_spectrum_and_continuum_jit,
            self._batch_interpolate_spectrum_and_continuum_jit,
        ) = _build_axis_fixed_interpolators(self.axes_jnp)
        logger.info("ZarrGridLoader initialization complete")

    def sample_point(self, indices=None):
        """Return a valid query point. indices: per-axis index (default: midpoint of each axis)."""
        if indices is None:
            indices = [len(ax) // 2 for ax in self.axes_jnp]
        return jnp.asarray(
            [float(self.axes_jnp[d][indices[d]]) for d in range(len(self.axis_names))]
        )

    def sample_points(self, n=3, along_axis=0):
        """Return n valid query points, varying along the given axis (others at midpoint)."""
        axes = [np.asarray(ax) for ax in self.axes_jnp]
        mid = [len(ax) // 2 for ax in axes]
        idx = np.linspace(0, len(axes[along_axis]) - 1, min(n, len(axes[along_axis])), dtype=int)
        points = []
        for i in idx:
            row = mid.copy()
            row[along_axis] = int(i)
            points.append([float(axes[d][row[d]]) for d in range(len(axes))])
        return jnp.asarray(points)

    def interpolate_spectrum(self, query_params, validate=True):
        query_params = jnp.asarray(query_params)
        if validate:
            validate_query_params(query_params, self.axes_jnp, self.axis_names)
        return self._interpolate_spectrum_jit(query_params, self.flux_cube_jnp)

    def batch_interpolate(self, queries, validate=True):
        queries = jnp.asarray(queries)
        if validate:
            validate_query_params(queries, self.axes_jnp, self.axis_names)
        return self._batch_interpolate_spectrum_jit(queries, self.flux_cube_jnp)

    def interpolate_spectrum_and_continuum(self, query_params, validate=True):
        query_params = jnp.asarray(query_params)
        if validate:
            validate_query_params(query_params, self.axes_jnp, self.axis_names)
        return self._interpolate_spectrum_and_continuum_jit(
            query_params, self.flux_cube_jnp, self.continuum_cube_jnp
        )

    def batch_interpolate_spectrum_and_continuum(self, queries, validate=True):
        queries = jnp.asarray(queries)
        if validate:
            validate_query_params(queries, self.axes_jnp, self.axis_names)
        return self._batch_interpolate_spectrum_and_continuum_jit(
            queries, self.flux_cube_jnp, self.continuum_cube_jnp
        )
