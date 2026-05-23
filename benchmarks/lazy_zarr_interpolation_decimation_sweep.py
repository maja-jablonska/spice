"""Decimation sweep for LazyZarrInterpolator leave-out reconstruction error.

Mirrors the protocol in ``lazy_zarr_linear_interpolation_errors.ipynb``:

  * **Kept** catalogue rows define the sparse interpolation grid (written to a
    temporary decimated zarr).
  * **Held-out** rows are reconstructed with ``get_weighted_batch`` and compared
    to the full-grid truth spectrum at the same parameters.

Presets (Teff / log g axis indexing only; other axes use every catalogue row
at each thermal vertex):

  ``leave_out_half`` — keep every 2nd, reconstruct every other 2nd (offset 1).
  ``keep_4_interp_2`` — keep every 4th, reconstruct every 2nd at offset 2
  (midpoints between kept thermal vertices; mimics a coarser underlying grid).

For each configuration the script writes:

  * ``<name>_per_spectrum.csv`` — one row per held-out spectrum,
  * ``<name>_by_parameter.csv`` — mean / median / max RMS % vs each grid axis,
  * ``<name>_errors.png`` — histogram, Teff–log g heatmaps, parameter panels,
  * ``summary.csv`` / ``summary_comparison.png`` — cross-config min / median / max.

Run (full grid; slow):

    python lazy_zarr_interpolation_decimation_sweep.py \\
        --store-path /path/to/regular_synthesized_spectra.zarr

Quick smoke test on a random subset:

    python lazy_zarr_interpolation_decimation_sweep.py \\
        --store-path /path/to/regular_synthesized_spectra.zarr \\
        --max-held 500 --configs leave_out_half
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("JAX_PLATFORMS", "cpu")

_REPO_SRC = Path(__file__).resolve().parent.parent / "src"
if _REPO_SRC.is_dir() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr

from spice.spectrum.lazy_zarr_interpolator import LazyZarrInterpolator

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else range(kwargs.get("total", 0))


@dataclass(frozen=True)
class DecimationSpec:
    """Which Teff / log g axis indices are kept vs held for reconstruction."""

    name: str
    keep_stride: int
    keep_offset: int = 0
    held_stride: int = 2
    held_offset: int = 1

    def describe(self) -> str:
        return (
            f"keep i ≡ {self.keep_offset} (mod {self.keep_stride}), "
            f"held i ≡ {self.held_offset} (mod {self.held_stride}), "
            f"held ∩ keep excluded"
        )


PRESETS: dict[str, DecimationSpec] = {
    "leave_out_half": DecimationSpec(
        "leave_out_half",
        keep_stride=2,
        keep_offset=0,
        held_stride=2,
        held_offset=1,
    ),
    "keep_4_interp_2": DecimationSpec(
        "keep_4_interp_2",
        keep_stride=4,
        keep_offset=0,
        held_stride=2,
        held_offset=2,
    ),
    "keep_3_interp_2": DecimationSpec(
        "keep_3_interp_2",
        keep_stride=3,
        keep_offset=0,
        held_stride=2,
        held_offset=2,
    ),
}


def _axis_index_positions(n: int, stride: int, offset: int) -> np.ndarray:
    return np.arange(offset, n, stride, dtype=np.int32)


def build_thermal_masks(
    teff_axis: np.ndarray,
    logg_axis: np.ndarray,
    teff_idx_all: np.ndarray,
    logg_idx_all: np.ndarray,
    spec: DecimationSpec,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    keep_teff = _axis_index_positions(len(teff_axis), spec.keep_stride, spec.keep_offset)
    keep_logg = _axis_index_positions(len(logg_axis), spec.keep_stride, spec.keep_offset)
    held_teff = _axis_index_positions(len(teff_axis), spec.held_stride, spec.held_offset)
    held_logg = _axis_index_positions(len(logg_axis), spec.held_stride, spec.held_offset)

    keep_mask = np.isin(teff_idx_all, keep_teff) & np.isin(logg_idx_all, keep_logg)
    held_mask = np.isin(teff_idx_all, held_teff) & np.isin(logg_idx_all, held_logg)
    held_mask &= ~keep_mask

    meta = {
        "keep_teff_vals": teff_axis[keep_teff],
        "keep_logg_vals": logg_axis[keep_logg],
        "held_teff_vals": teff_axis[held_teff],
        "held_logg_vals": logg_axis[held_logg],
    }
    return keep_mask, held_mask, meta


def write_decimated_zarr(
    dec_path: Path,
    wave_np: np.ndarray,
    flux_kept: np.ndarray,
    cont_kept: np.ndarray,
    index_kept: pd.DataFrame,
    param_cols: Sequence[str],
) -> None:
    if dec_path.exists():
        shutil.rmtree(dec_path)
    dec_group = zarr.open_group(str(dec_path), mode="w")
    dec_group.create_array("wavelength", data=wave_np)
    chunk_rows = min(64, flux_kept.shape[0])
    dec_group.create_array(
        "flux", data=flux_kept, chunks=(chunk_rows, flux_kept.shape[1])
    )
    dec_group.create_array(
        "continuum", data=cont_kept, chunks=(chunk_rows, cont_kept.shape[1])
    )
    out_index = index_kept[param_cols].copy()
    out_index.insert(0, "row_idx", np.arange(len(out_index), dtype=np.int32))
    out_index.to_parquet(str(dec_path / "index.parquet"))


def reconstruct_chunked(
    interp_dec: LazyZarrInterpolator,
    queries: np.ndarray,
    n_wave: int,
    chunk: int,
) -> np.ndarray:
    out = np.empty((queries.shape[0], n_wave), dtype=np.float64)
    n_chunks = (queries.shape[0] + chunk - 1) // chunk
    for i in tqdm(range(0, queries.shape[0], chunk), desc="Reconstructing", total=n_chunks):
        sl = slice(i, min(i + chunk, queries.shape[0]))
        out[sl] = np.asarray(
            interp_dec.get_weighted_batch(jnp.asarray(queries[sl], dtype=jnp.float64))[
                "flux"
            ]
        )
    return out


def rms_stats(rms_per_spec_pct: np.ndarray) -> dict[str, float]:
    return {
        "n_held": int(rms_per_spec_pct.size),
        "rms_pct_min": float(np.min(rms_per_spec_pct)),
        "rms_pct_median": float(np.median(rms_per_spec_pct)),
        "rms_pct_mean": float(np.mean(rms_per_spec_pct)),
        "rms_pct_p95": float(np.percentile(rms_per_spec_pct, 95)),
        "rms_pct_max": float(np.max(rms_per_spec_pct)),
    }


def aggregate_by_parameter(
    held_df: pd.DataFrame,
    rms_per_spec_pct: np.ndarray,
    param_cols: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for col in param_cols:
        values = held_df[col].to_numpy()
        for v in np.unique(values):
            m = values == v
            if not np.any(m):
                continue
            r = rms_per_spec_pct[m]
            rows.append(
                {
                    "parameter": col,
                    "value": float(v),
                    "count": int(m.sum()),
                    "rms_pct_mean": float(np.mean(r)),
                    "rms_pct_median": float(np.median(r)),
                    "rms_pct_max": float(np.max(r)),
                }
            )
    return pd.DataFrame(rows)


def aggregate_thermal_grid(
    held_df: pd.DataFrame,
    rms_per_spec_pct: np.ndarray,
    held_teff_vals: np.ndarray,
    held_logg_vals: np.ndarray,
    reducer,
) -> np.ndarray:
    teff_col = held_df["teff"].to_numpy(dtype=np.float32)
    logg_col = held_df["logg"].to_numpy(dtype=np.float32)
    out = np.full((len(held_teff_vals), len(held_logg_vals)), np.nan)
    for i, t in enumerate(held_teff_vals):
        for j, g in enumerate(held_logg_vals):
            m = (teff_col == t) & (logg_col == g)
            if np.any(m):
                out[i, j] = reducer(rms_per_spec_pct[m])
    return out


def run_decimation_case(
    spec: DecimationSpec,
    store_path: Path,
    df: pd.DataFrame,
    param_cols: list[str],
    teff_axis: np.ndarray,
    logg_axis: np.ndarray,
    flux_full_np: np.ndarray,
    cont_full_np: np.ndarray,
    wave_np: np.ndarray,
    *,
    chunk: int,
    max_held: int | None,
    in_memory: bool,
    rng: np.random.Generator,
) -> dict:
    teff_idx_all = np.searchsorted(teff_axis, df["teff"].to_numpy(dtype=np.float32))
    logg_idx_all = np.searchsorted(logg_axis, df["logg"].to_numpy(dtype=np.float32))

    keep_mask, held_mask, meta = build_thermal_masks(
        teff_axis, logg_axis, teff_idx_all, logg_idx_all, spec
    )

    held_df = df.loc[held_mask].reset_index(drop=True)
    if max_held is not None and len(held_df) > max_held:
        pick = rng.choice(len(held_df), size=max_held, replace=False)
        held_df = held_df.iloc[np.sort(pick)].reset_index(drop=True)

    held_queries = held_df[param_cols].to_numpy(dtype=np.float64)
    held_truth = flux_full_np[held_df["row_idx"].to_numpy(dtype=int)].astype(np.float64)

    orig_kept_rowidx = df.loc[keep_mask, "row_idx"].to_numpy(dtype=int)
    new_flux = flux_full_np[orig_kept_rowidx]
    new_cont = cont_full_np[orig_kept_rowidx]
    index_kept = df.loc[keep_mask, param_cols].copy()

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"spice_dec_{spec.name}_"))
    dec_path = tmp_dir / "decimated.zarr"
    try:
        write_decimated_zarr(
            dec_path, wave_np, new_flux, new_cont, index_kept, param_cols
        )
        interp_dec = LazyZarrInterpolator(
            str(dec_path),
            params=param_cols,
            in_memory=in_memory,
            sparse=True,
            accumulate_chunk_size=8,
        )
        held_pred = reconstruct_chunked(
            interp_dec, held_queries, wave_np.shape[0], chunk=chunk
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    denom = np.maximum(np.maximum(np.abs(held_pred), np.abs(held_truth)), 1e-30)
    pct_spec = 100.0 * (held_pred - held_truth) / denom
    rms_per_spec_pct = np.sqrt(np.mean(pct_spec**2, axis=1))
    maxabs_per_spec = np.max(np.abs(pct_spec), axis=1)

    per_spec = held_df[param_cols].copy()
    per_spec["row_idx"] = held_df["row_idx"].to_numpy()
    per_spec["rms_pct"] = rms_per_spec_pct
    per_spec["mean_pct"] = np.mean(pct_spec, axis=1)
    per_spec["max_abs_pct"] = maxabs_per_spec

    by_param = aggregate_by_parameter(held_df, rms_per_spec_pct, param_cols)
    mean_rms_grid = aggregate_thermal_grid(
        held_df,
        rms_per_spec_pct,
        meta["held_teff_vals"],
        meta["held_logg_vals"],
        np.mean,
    )
    max_rms_grid = aggregate_thermal_grid(
        held_df,
        rms_per_spec_pct,
        meta["held_teff_vals"],
        meta["held_logg_vals"],
        np.max,
    )

    summary = {
        "config": spec.name,
        "description": spec.describe(),
        "n_kept_rows": int(keep_mask.sum()),
        "n_held_rows_full_mask": int(held_mask.sum()),
        "kept_teff": len(meta["keep_teff_vals"]),
        "kept_logg": len(meta["keep_logg_vals"]),
        "held_teff": len(meta["held_teff_vals"]),
        "held_logg": len(meta["held_logg_vals"]),
        **rms_stats(rms_per_spec_pct),
    }

    return {
        "summary": summary,
        "per_spec": per_spec,
        "by_param": by_param,
        "rms_per_spec_pct": rms_per_spec_pct,
        "held_teff_vals": meta["held_teff_vals"],
        "held_logg_vals": meta["held_logg_vals"],
        "mean_rms_grid": mean_rms_grid,
        "max_rms_grid": max_rms_grid,
    }


def plot_case(
    result: dict,
    output_path: Path,
    param_cols: Sequence[str],
    focus_params: Sequence[str],
) -> None:
    rms = result["rms_per_spec_pct"]
    by_param = result["by_param"]
    held_teff = result["held_teff_vals"]
    held_logg = result["held_logg_vals"]
    mean_grid = result["mean_rms_grid"]
    max_grid = result["max_rms_grid"]
    name = result["summary"]["config"]

    n_param_panels = len(focus_params)
    fig = plt.figure(figsize=(14, 3.2 * (2 + (n_param_panels + 2) // 3)))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.1, 1.2], hspace=0.35, wspace=0.32)

    ax_hist = fig.add_subplot(gs[0, :])
    ax_hist.hist(rms, bins=60, color="steelblue", edgecolor="k", linewidth=0.3)
    ax_hist.set_xlabel("per-spectrum RMS % (reconstructed − truth)")
    ax_hist.set_ylabel("count")
    ax_hist.set_title(
        f"{name}: N = {len(rms)} held-out  |  "
        f"min = {rms.min():.3f}%  median = {np.median(rms):.3f}%  "
        f"max = {rms.max():.3f}%"
    )

    extent = (
        float(held_logg.min()),
        float(held_logg.max()),
        float(held_teff.min()),
        float(held_teff.max()),
    )
    kw_h = dict(origin="lower", aspect="auto", extent=extent)

    ax_m = fig.add_subplot(gs[1, 0])
    im_m = ax_m.imshow(mean_grid, **kw_h, cmap="magma")
    ax_m.set_xlabel("log g (held)")
    ax_m.set_ylabel("Teff (held)")
    ax_m.set_title("Mean RMS %")
    plt.colorbar(im_m, ax=ax_m, fraction=0.046)

    ax_x = fig.add_subplot(gs[1, 1])
    im_x = ax_x.imshow(max_grid, **kw_h, cmap="magma")
    ax_x.set_xlabel("log g (held)")
    ax_x.set_ylabel("Teff (held)")
    ax_x.set_title("Max RMS %")
    plt.colorbar(im_x, ax=ax_x, fraction=0.046)

    ax_s = fig.add_subplot(gs[1, 2])
    ax_s.axis("off")
    lines = [result["summary"]["description"], ""]
    for k, v in result["summary"].items():
        if k in ("config", "description"):
            continue
        lines.append(f"{k}: {v}")
    ax_s.text(0.0, 1.0, "\n".join(lines), va="top", fontsize=9, family="monospace")

    for k, pname in enumerate(focus_params):
        ax = fig.add_subplot(gs[2, k % 3])
        sub = by_param[by_param["parameter"] == pname].sort_values("value")
        if sub.empty:
            ax.set_visible(False)
            continue
        ax.plot(sub["value"], sub["rms_pct_mean"], "o-", color="C0", ms=4, label="mean")
        ax.plot(
            sub["value"],
            sub["rms_pct_median"],
            "s--",
            color="C1",
            ms=3,
            alpha=0.85,
            label="median",
        )
        ax.set_xlabel(pname)
        ax.set_ylabel("RMS %")
        ax.set_title(f"RMS vs {pname}")
        if k == 0:
            ax.legend(fontsize=7, loc="best")

    fig.suptitle(f"LazyZarr leave-out reconstruction — {name}", y=0.995)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_path}")


def plot_summary_comparison(summaries: pd.DataFrame, output_path: Path) -> None:
    configs = summaries["config"].tolist()
    x = np.arange(len(configs))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(configs)), 4))
    for i, (col, label, color) in enumerate(
        [
            ("rms_pct_min", "min", "C2"),
            ("rms_pct_median", "median", "C0"),
            ("rms_pct_max", "max", "C3"),
        ]
    ):
        ax.bar(x + (i - 1) * width, summaries[col], width, label=label, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=15, ha="right")
    ax.set_ylabel("per-spectrum RMS %")
    ax.set_title("Leave-out reconstruction error by decimation preset")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison figure to {output_path}")


def load_catalogue(store_path: Path) -> tuple[pd.DataFrame, list[str]]:
    index_path = store_path / "index.parquet"
    if not index_path.is_file():
        raise FileNotFoundError(f"Expected {index_path}")
    df = pd.read_parquet(index_path)
    param_cols = [c for c in df.columns if c != "row_idx"]
    return df, param_cols


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--store-path",
        type=Path,
        required=True,
        help="Path to full synthesized-spectra zarr (must contain index.parquet).",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(PRESETS.keys()),
        choices=list(PRESETS.keys()),
        help="Decimation presets to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here / "lazy_zarr_decimation_sweep",
        help="Directory for CSV tables and PNG figures.",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=64,
        help="Batch size for get_weighted_batch during reconstruction.",
    )
    parser.add_argument(
        "--max-held",
        type=int,
        default=None,
        help="Randomly subsample held-out spectra (for quick tests).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed when --max-held is set.",
    )
    parser.add_argument(
        "--in-memory",
        action="store_true",
        help="Load decimated rowwise flux into RAM (faster, large memory).",
    )
    parser.add_argument(
        "--focus-params",
        nargs="+",
        default=["teff", "logg", "[Fe/H]", "mu"],
        help="Parameters for dependence line panels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"Store: {args.store_path.resolve()}")
    df, param_cols = load_catalogue(args.store_path)

    print("Loading full grid (truth lookup only) ...")
    interp_full = LazyZarrInterpolator(
        str(args.store_path),
        params=param_cols,
        in_memory=False,
        sparse=True,
        accumulate_chunk_size=None,
    )
    cols = list(interp_full.grid_index.columns)
    teff_axis = np.asarray(interp_full.grid_index.axes[cols.index("teff")])
    logg_axis = np.asarray(interp_full.grid_index.axes[cols.index("logg")])
    flux_full_np = np.asarray(interp_full.rowwise_data["flux"])
    cont_full_np = np.asarray(interp_full.rowwise_data["continuum"])
    wave_np = np.asarray(interp_full.static_data["wavelength"])
    print(f"  Teff nodes: {len(teff_axis)}, log g nodes: {len(logg_axis)}")
    print(f"  index rows: {len(df)}, flux shape: {flux_full_np.shape}")

    summaries = []
    for cfg_name in args.configs:
        spec = PRESETS[cfg_name]
        print(f"\n=== {spec.name} ===")
        print(f"  {spec.describe()}")
        result = run_decimation_case(
            spec,
            args.store_path,
            df,
            param_cols,
            teff_axis,
            logg_axis,
            flux_full_np,
            cont_full_np,
            wave_np,
            chunk=args.chunk,
            max_held=args.max_held,
            in_memory=args.in_memory,
            rng=rng,
        )
        summaries.append(result["summary"])

        per_path = args.output_dir / f"{spec.name}_per_spectrum.csv"
        param_path = args.output_dir / f"{spec.name}_by_parameter.csv"
        fig_path = args.output_dir / f"{spec.name}_errors.png"
        result["per_spec"].to_csv(per_path, index=False)
        result["by_param"].to_csv(param_path, index=False)
        plot_case(
            {**result, "summary": result["summary"]},
            fig_path,
            param_cols,
            args.focus_params,
        )
        print(f"  wrote {per_path.name}, {param_path.name}")

        st = result["summary"]
        print(
            f"  RMS %: min={st['rms_pct_min']:.3f}  "
            f"median={st['rms_pct_median']:.3f}  "
            f"max={st['rms_pct_max']:.3f}"
        )

    summary_df = pd.DataFrame(summaries)
    summary_path = args.output_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved {summary_path}")
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print(summary_df.to_string(index=False))

    plot_summary_comparison(
        summary_df, args.output_dir / "summary_comparison.png"
    )


if __name__ == "__main__":
    main()
