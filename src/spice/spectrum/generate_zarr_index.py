"""Build the ``index.parquet`` that a zarr spectral grid needs for axis lookup.

Besides the stellar-parameter columns (and an optional ``mu`` column), the index
can carry a ``geometry`` provenance column recording whether each grid node was
synthesised in ``spherical`` or ``plane_parallel`` radiative-transfer geometry.
MARCS/Korg compute giants (low ``logg``) in spherical symmetry and dwarfs (high
``logg``) plane-parallel, so this records *which* was used per node rather than
silently re-deriving it later. Geometry is provenance only -- it is never an
interpolation axis (a categorical cannot be interpolated), so it rides along in
the parquet, is ignored by the grid index, and is exposed as
``LazyZarrInterpolator.geometry`` for callers that want to detect when a query
straddles the plane-parallel/spherical boundary.

The source of truth is a ``geometry`` array in the store (written by the grid
producer, one label per row). When absent it can be derived from ``logg`` via
``--geometry-from-logg`` as a heuristic fallback.
"""

import argparse
from pathlib import Path

import numpy as np
import zarr


REQUIRED_GRID_ARRAYS = frozenset({"flux", "continuum", "wavelength", "params"})

# Canonical labels for a grid node's radiative-transfer geometry, stored as
# provenance in the index (and optionally as a ``geometry`` array in the store).
PLANE_PARALLEL = "plane_parallel"
SPHERICAL = "spherical"

# Integer codes tolerated in a numeric ``geometry`` array (0 -> PP, 1 -> sph).
_GEOMETRY_CODES = {0: PLANE_PARALLEL, 1: SPHERICAL}

# Column aliases recognised when deriving geometry from surface gravity.
_LOGG_ALIASES = ("logg", "log_g", "log g", "log_gs", "surface_gravity")

# Default logg below which a node is labelled spherical when deriving geometry
# from gravity. The targeted grids use logg < 3 -> spherical; MARCS' own
# standard boundary is ~3.5. Overridable on the CLI.
DEFAULT_SPHERICAL_LOGG_THRESHOLD = 3.0


def resolve_grid_group(store, store_path):
    """Descend through wrapper groups until the expected grid arrays are present."""
    current = store

    while True:
        array_keys = set(current.array_keys())
        if REQUIRED_GRID_ARRAYS.issubset(array_keys):
            return current

        child_groups = list(current.group_keys())
        if len(child_groups) != 1 or array_keys:
            missing = sorted(REQUIRED_GRID_ARRAYS - array_keys)
            raise KeyError(
                "Could not locate grid arrays "
                f"{missing} in zarr store '{store_path}'. "
                "Expected them at the root or inside a single nested wrapper group."
            )

        current = current[child_groups[0]]


def load_param_names(group):
    names = []
    for value in group["param_names"][:]:
        if hasattr(value, "decode"):
            names.append(value.decode())
        else:
            names.append(str(value))
    return names


def _get_polars():
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError(
            "Generating parquet indexes requires polars. Install it with `pip install polars`."
        ) from exc
    return pl


def build_index_frame(params, param_names, mu_values=None, geometry=None):
    """Build a polars DataFrame index from a params array.

    Parameters
    ----------
    params : array-like, shape (N_rows, N_params)
        The parameter grid. Columns corresponding to names other than ``"mu"``
        are included directly.
    param_names : list[str]
        Name for each column of *params*.
    mu_values : array-like or None
        Optional 1-D array of mu values.  When provided it is included as the
        ``"mu"`` column regardless of whether ``"mu"`` appears in *param_names*.
    geometry : sequence[str] or None
        Optional per-row radiative-transfer geometry labels (e.g.
        ``"spherical"`` / ``"plane_parallel"``). When provided it is added as a
        ``"geometry"`` provenance column. It is not a grid axis.
    """
    params = np.asarray(params)
    if params.ndim != 2:
        raise ValueError(f"Expected 'params' to be 2D, got shape {params.shape}.")

    if len(param_names) != params.shape[1]:
        raise ValueError(
            f"'param_names' length ({len(param_names)}) does not match params width ({params.shape[1]})."
        )
    if len(set(param_names)) != len(param_names):
        raise ValueError(f"Parameter names must be unique, got {param_names}.")

    columns = {"row_idx": np.arange(params.shape[0], dtype=np.int64)}
    for idx, name in enumerate(param_names):
        if name == "mu":
            continue
        columns[name] = params[:, idx]

    if mu_values is not None:
        mu_values = np.asarray(mu_values).squeeze()
        if mu_values.ndim != 1:
            raise ValueError(
                f"Expected 'mu_values' to be 1D, got shape {mu_values.shape}."
            )
        if mu_values.shape[0] != params.shape[0]:
            raise ValueError(
                f"'mu_values' length ({mu_values.shape[0]}) does not match params rows ({params.shape[0]})."
            )
        columns["mu"] = mu_values

    if geometry is not None:
        geometry = list(geometry)
        if len(geometry) != params.shape[0]:
            raise ValueError(
                f"'geometry' length ({len(geometry)}) does not match params rows ({params.shape[0]})."
            )
        columns["geometry"] = geometry

    return _get_polars().DataFrame(columns)


def _load_mu_from_zarr(group, params, param_names, store_path):
    if "mu_selected" in group:
        mu_values = np.asarray(group["mu_selected"][:])
        if mu_values.ndim == 2 and mu_values.shape[1] == 1:
            mu_values = mu_values[:, 0]
        if mu_values.ndim != 1:
            raise ValueError(
                f"Expected 'mu_selected' to be 1D, got shape {mu_values.shape}."
            )
        if mu_values.shape[0] != params.shape[0]:
            raise ValueError(
                "'mu_selected' length "
                f"({mu_values.shape[0]}) does not match params rows ({params.shape[0]})."
            )
        return mu_values

    if "mu" in param_names:
        return params[:, param_names.index("mu")]

    raise KeyError(
        f"Missing 'mu_selected' array in zarr store '{store_path}', and 'mu' is not present in 'params'."
    )


def geometry_labels_from_logg(logg, threshold=DEFAULT_SPHERICAL_LOGG_THRESHOLD):
    """Heuristically label nodes ``spherical`` below ``threshold`` else
    ``plane_parallel``, from surface gravity. Exposed so grid producers can
    build a ``geometry`` array consistent with the index convention."""
    return [SPHERICAL if float(g) < threshold else PLANE_PARALLEL
            for g in np.asarray(logg).reshape(-1)]


def _decode_geometry_array(values):
    """Normalise a raw ``geometry`` array into a list of string labels.

    String/bytes entries are passed through verbatim (only decoded), preserving
    whatever the producer wrote; integer codes are mapped via
    :data:`_GEOMETRY_CODES` (0 -> plane_parallel, 1 -> spherical).
    """
    labels = []
    for value in np.asarray(values).reshape(-1):
        if hasattr(value, "decode"):
            labels.append(value.decode())
        elif isinstance(value, str):
            labels.append(str(value))
        else:
            code = int(value)
            if code not in _GEOMETRY_CODES:
                raise ValueError(
                    f"Unrecognised integer geometry code {code!r}; expected one "
                    f"of {sorted(_GEOMETRY_CODES)} ({_GEOMETRY_CODES})."
                )
            labels.append(_GEOMETRY_CODES[code])
    return labels


def _derive_geometry_from_logg(params, param_names, threshold):
    lowered = {name.lower(): idx for idx, name in enumerate(param_names)}
    logg_idx = next((lowered[a] for a in _LOGG_ALIASES if a in lowered), None)
    if logg_idx is None:
        raise KeyError(
            "Cannot derive geometry from gravity: no logg-like column in "
            f"param_names {param_names} (looked for {list(_LOGG_ALIASES)})."
        )
    return geometry_labels_from_logg(np.asarray(params)[:, logg_idx], threshold)


def _load_geometry_from_zarr(group, params, param_names, store_path,
                             geometry_from_logg=None):
    """Return per-row geometry labels, or ``None`` when unavailable.

    Priority: an explicit ``geometry`` array in the store (the producer's source
    of truth) over derivation from ``logg`` (a heuristic, only when
    ``geometry_from_logg`` is a threshold) over ``None`` (geometry is optional
    provenance, so its absence is not an error).
    """
    n_rows = np.asarray(params).shape[0]
    if "geometry" in group:
        raw = np.asarray(group["geometry"][:])
        if raw.ndim == 2 and raw.shape[1] == 1:
            raw = raw[:, 0]
        if raw.ndim != 1:
            raise ValueError(f"Expected 'geometry' to be 1D, got shape {raw.shape}.")
        if raw.shape[0] != n_rows:
            raise ValueError(
                f"'geometry' length ({raw.shape[0]}) does not match params rows ({n_rows})."
            )
        return _decode_geometry_array(raw)

    if geometry_from_logg is not None:
        return _derive_geometry_from_logg(params, param_names, geometry_from_logg)

    return None


def build_index_frame_from_zarr(store_path, include_mu=True,
                                include_geometry=True, geometry_from_logg=None):
    store_path = Path(store_path).expanduser().resolve()
    root = zarr.open(str(store_path), mode="r")
    if not hasattr(root, "array_keys"):
        raise KeyError(
            f"Expected a zarr group at '{store_path}', got a {type(root).__name__}."
        )
    group = resolve_grid_group(root, str(store_path))

    if "param_names" not in group:
        raise KeyError(f"Missing 'param_names' array in zarr store '{store_path}'.")

    params = np.asarray(group["params"][:])
    param_names = load_param_names(group)

    mu_values = None
    if include_mu:
        mu_values = _load_mu_from_zarr(group, params, param_names, store_path)

    geometry = None
    if include_geometry:
        geometry = _load_geometry_from_zarr(
            group, params, param_names, store_path,
            geometry_from_logg=geometry_from_logg,
        )

    return build_index_frame(params, param_names, mu_values=mu_values,
                             geometry=geometry)


def write_index_parquet(
    store_path,
    output_path=None,
    compression="zstd",
    include_mu=True,
    include_geometry=True,
    geometry_from_logg=None,
):
    store_path = Path(store_path).expanduser().resolve()
    if output_path is None:
        output_path = store_path / "index.parquet"
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = build_index_frame_from_zarr(
        store_path, include_mu=include_mu,
        include_geometry=include_geometry, geometry_from_logg=geometry_from_logg,
    )
    frame.write_parquet(output_path, compression=compression)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an index.parquet file for a spectrum zarr store."
    )
    parser.add_argument("store_path", help="Path to the spectrum zarr store.")
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        help="Output parquet path. Defaults to <store_path>/index.parquet.",
    )
    parser.add_argument(
        "--compression",
        default="zstd",
        help="Parquet compression codec to use. Defaults to zstd.",
    )
    mu_group = parser.add_mutually_exclusive_group()
    mu_group.add_argument(
        "--include-mu",
        dest="include_mu",
        action="store_true",
        help="Include the `mu` column in the generated index, using `mu_selected` when present.",
    )
    mu_group.add_argument(
        "--exclude-mu",
        dest="include_mu",
        action="store_false",
        help="Exclude the `mu` column from the generated index.",
    )
    parser.set_defaults(include_mu=True)

    geom_group = parser.add_mutually_exclusive_group()
    geom_group.add_argument(
        "--include-geometry",
        dest="include_geometry",
        action="store_true",
        help="Record per-node radiative-transfer geometry (spherical / "
             "plane_parallel) from the store's `geometry` array when present. "
             "Provenance only; never used as an interpolation axis.",
    )
    geom_group.add_argument(
        "--exclude-geometry",
        dest="include_geometry",
        action="store_false",
        help="Do not write a `geometry` provenance column.",
    )
    parser.set_defaults(include_geometry=True)
    parser.add_argument(
        "--geometry-from-logg",
        dest="geometry_from_logg",
        nargs="?",
        type=float,
        const=DEFAULT_SPHERICAL_LOGG_THRESHOLD,
        default=None,
        metavar="THRESHOLD",
        help="When the store has no `geometry` array, derive it from surface "
             "gravity: nodes with logg below THRESHOLD (default "
             f"{DEFAULT_SPHERICAL_LOGG_THRESHOLD} when the flag is given) are "
             "labelled spherical. Heuristic; prefer a stored `geometry` array.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = write_index_parquet(
        store_path=args.store_path,
        output_path=args.output_path,
        compression=args.compression,
        include_mu=args.include_mu,
        include_geometry=args.include_geometry,
        geometry_from_logg=args.geometry_from_logg,
    )
    print(output_path)


if __name__ == "__main__":
    main()
