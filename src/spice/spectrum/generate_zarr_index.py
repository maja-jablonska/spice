import argparse
from pathlib import Path

import numpy as np
import zarr


REQUIRED_GRID_ARRAYS = frozenset({"flux", "continuum", "wavelength", "params"})


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


def _iter_index_columns(params, param_names):
    for idx, name in enumerate(param_names):
        if name == "mu":
            continue
        yield name, params[:, idx]


def _load_mu_column(group, params, param_names, store_path):
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


def build_index_frame(store_path, include_mu=True):
    store_path = Path(store_path).expanduser().resolve()
    group = resolve_grid_group(zarr.open_group(str(store_path), mode="r"), str(store_path))

    if "param_names" not in group:
        raise KeyError(f"Missing 'param_names' array in zarr store '{store_path}'.")

    params = np.asarray(group["params"][:])
    if params.ndim != 2:
        raise ValueError(f"Expected 'params' to be 2D, got shape {params.shape}.")

    param_names = load_param_names(group)
    if len(param_names) != params.shape[1]:
        raise ValueError(
            f"'param_names' length ({len(param_names)}) does not match params width ({params.shape[1]})."
        )
    if len(set(param_names)) != len(param_names):
        raise ValueError(f"Parameter names must be unique, got {param_names}.")

    columns = {"row_idx": np.arange(params.shape[0], dtype=np.int64)}
    for name, values in _iter_index_columns(params, param_names):
        columns[name] = values
    if include_mu:
        columns["mu"] = _load_mu_column(group, params, param_names, store_path)

    return _get_polars().DataFrame(columns)


def write_index_parquet(
    store_path,
    output_path=None,
    compression="zstd",
    include_mu=True,
):
    store_path = Path(store_path).expanduser().resolve()
    if output_path is None:
        output_path = store_path / "index.parquet"
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = build_index_frame(store_path, include_mu=include_mu)
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
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = write_index_parquet(
        store_path=args.store_path,
        output_path=args.output_path,
        compression=args.compression,
        include_mu=args.include_mu,
    )
    print(output_path)


if __name__ == "__main__":
    main()
