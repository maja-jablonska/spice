#!/usr/bin/env python
"""Run SUPPNet continuum normalization on harps_data.zarr (notebook-free).

Usage (from this directory, suppnet-env activated):

    python harps_suppnet_normalize.py

Requires tensorflow-macos >= 2.9 on Apple Silicon (see setup_suppnet_env.sh).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import zarr
from tqdm import tqdm

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

_tf_major, _tf_minor = (int(x) for x in tf.__version__.split(".")[:2])
if (_tf_major, _tf_minor) < (2, 9):
    raise SystemExit(
        f"TensorFlow {tf.__version__} aborts on Apple Silicon during SUPPNet predict.\n"
        'Run: pip install "tensorflow-macos>=2.9.2,<2.10" "tensorflow-metal>=0.5.1,<0.6"'
    )
for _gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_gpu, True)

SUPPNET_REPO = Path(os.environ.get("SUPPNET_REPO", Path.home() / "code/suppnet")).expanduser()
if str(SUPPNET_REPO) not in sys.path and SUPPNET_REPO.is_dir():
    sys.path.insert(0, str(SUPPNET_REPO))

from suppnet.NN_utility import get_smoothed_continuum, get_suppnet  # noqa: E402


def finite_spectrum(wave_row, flux_row):
    w = np.asarray(wave_row, dtype=float)
    f = np.asarray(flux_row, dtype=float)
    mask = np.isfinite(w) & np.isfinite(f)
    return w[mask], f[mask]


def mask_zero_only_gaps(wave, flux, normed_flux, continuum, **extra):
    df = pd.DataFrame(
        {"wave": wave, "flux": flux, "normed_flux": normed_flux, "continuum": continuum, **extra}
    )
    zero_gap = df[["flux", "normed_flux", "continuum"]].eq(0).all(axis=1)
    df.loc[zero_gap, ["normed_flux", "continuum"]] = np.nan
    return df


def suppnet_normalize_row(wave, flux, nn):
    if wave.size < 100:
        raise ValueError(f"spectrum too short ({wave.size} points)")
    flux = np.asarray(flux, dtype=float).copy()
    median = np.nanmedian(flux)
    if not np.isfinite(median) or median == 0:
        raise ValueError("non-finite or zero flux median")
    flux /= median

    cont, cont_err, seg, seg_err = nn.normalize(wave, flux)
    cont_smo = get_smoothed_continuum(wave, cont, cont_err)
    normed = flux / cont_smo
    normed_err = cont_err / cont_smo

    return mask_zero_only_gaps(
        wave,
        flux,
        normed,
        cont_smo,
        normed_error=normed_err,
        continuum_raw=cont,
        continuum_err=cont_err,
        segmentation=seg,
        segmentation_err=seg_err,
    )


def is_zarr_v3_store(path: Path) -> bool:
    meta = path / "zarr.json"
    if not meta.is_file():
        return False
    return json.loads(meta.read_text()).get("zarr_format") == 3


def ensure_zarr2_copy(src: Path, dst: Path) -> None:
    if dst.is_dir() and dst.stat().st_mtime >= src.stat().st_mtime:
        return
    print(f"Converting {src.name} (Zarr v3) → {dst.name} (Zarr v2)…")
    code = textwrap.dedent(
        f"""
        import shutil
        from pathlib import Path
        import zarr
        import numpy as np
        src, dst = Path({str(src)!r}), Path({str(dst)!r})
        if dst.exists():
            shutil.rmtree(dst)
        g3 = zarr.open_group(src, mode='r')
        g2 = zarr.open_group(dst, mode='w', zarr_format=2)
        for key in g3.array_keys():
            g2.create_array(key, data=np.asarray(g3[key]))
        for key, val in g3.attrs.asdict().items():
            g2.attrs[key] = val
        print('converted', len(list(g3.array_keys())), 'arrays')
        """
    )
    subprocess.check_call(
        ["conda", "run", "-n", "astro", "python", "-c", code],
        cwd=str(src.parent),
    )


def open_harps_store(path: Path, mode: str = "r+"):
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"{path} not found — run harps_read.ipynb first.")

    zarr_major = int(zarr.__version__.split(".")[0])
    if is_zarr_v3_store(path) and zarr_major < 3:
        v2_path = path.with_name("harps_data_zarr2.zarr")
        ensure_zarr2_copy(path, v2_path)
        path = v2_path
        print(f"Using Zarr v2 mirror: {path}")

    try:
        return zarr.open_group(path, mode=mode), path
    except zarr.errors.GroupNotFoundError as exc:
        raise RuntimeError(
            f"Cannot open {path} with zarr {zarr.__version__}."
        ) from exc


def write_zarr_array(group, name: str, data: np.ndarray, fill_value=np.nan) -> None:
    if hasattr(group, "create_array"):
        group.create_array(name, data=data, overwrite=True, fill_value=fill_value)
    else:
        if name in group:
            del group[name]
        group.create_dataset(name, data=data, fill_value=fill_value)


def save_all_file(df, path):
    out = pd.DataFrame(
        {
            "wave": df["wave"],
            "flux": df["flux"],
            "normed_flux": df["normed_flux"],
            "normed_error": df["normed_error"],
            "smoothed_continuum": df["continuum"],
            "continuum": df["continuum_raw"],
            "continuum_err": df["continuum_err"],
            "segmentation": df["segmentation"],
            "segmentation_err": df["segmentation_err"],
        }
    )
    gap = out["flux"].eq(0)
    out.loc[gap, out.columns != "wave"] = 0.0
    out.to_csv(path, sep=" ", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zarr",
        type=Path,
        default=None,
        help="Input zarr (default: harps_data_zarr2.zarr if present else harps_data.zarr)",
    )
    parser.add_argument(
        "--resampling-step",
        type=float,
        default=0.05,
        help="SUPPNet internal resampling step [Å]",
    )
    parser.add_argument(
        "--weights",
        choices=("active", "synth", "emission"),
        default="active",
    )
    parser.add_argument(
        "--export-txt-dir",
        type=Path,
        default=None,
        help="Optional directory for spectrum<i>.all QC files",
    )
    args = parser.parse_args()

    workdir = Path(__file__).resolve().parent
    if args.zarr is not None:
        zarr_in = args.zarr.expanduser().resolve()
    else:
        v2 = workdir / "harps_data_zarr2.zarr"
        zarr_in = v2 if v2.is_dir() else workdir / "harps_data.zarr"

    print(f"zarr input:  {zarr_in}")
    print(f"tensorflow:  {tf.__version__}")
    print(f"SUPPNet repo: {SUPPNET_REPO}")

    store, zarr_path = open_harps_store(zarr_in, mode="r+")
    raw_source_files = list(store.attrs.get("raw_source_files", []))
    if not raw_source_files:
        raise RuntimeError("zarr has no raw_source_files — re-run harps_read.ipynb.")

    nrows = store["wavelengths"].shape[0]
    ncols = store["wavelengths"].shape[1]
    print(f"{nrows} spectra; first file: {raw_source_files[0]}")

    print("Loading SUPPNet model…")
    nn = get_suppnet(
        resampling_step=args.resampling_step,
        step_size=256,
        norm_only=False,
        which_weights=args.weights,
    )
    print("Model ready.")

    export_dir = args.export_txt_dir
    if export_dir is not None:
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

    normalized_wave_rows = []
    normalized_flux_rows = []
    continuum_rows = []
    failed = []

    def append_nan_row() -> None:
        normalized_wave_rows.append(np.full(ncols, np.nan))
        normalized_flux_rows.append(np.full(ncols, np.nan))
        continuum_rows.append(np.full(ncols, np.nan))

    for i in tqdm(range(nrows), desc="SUPPNet"):
        wave, flux = finite_spectrum(store["wavelengths"][i], store["fluxes"][i])
        if wave.size == 0:
            failed.append((i, "no finite pixels"))
            append_nan_row()
            continue
        try:
            df = suppnet_normalize_row(wave, flux, nn)
        except Exception as exc:
            failed.append((i, repr(exc)))
            append_nan_row()
            continue

        if export_dir is not None:
            save_all_file(df, export_dir / f"spectrum{i}.all")

        w_pad = np.full(ncols, np.nan)
        f_pad = np.full(ncols, np.nan)
        c_pad = np.full(ncols, np.nan)
        n = df["wave"].size
        w_pad[:n] = df["wave"].values
        f_pad[:n] = df["normed_flux"].values
        c_pad[:n] = df["continuum"].values
        normalized_wave_rows.append(w_pad)
        normalized_flux_rows.append(f_pad)
        continuum_rows.append(c_pad)
        gc.collect()

    print(f"done: {nrows - len(failed)}/{nrows} OK, {len(failed)} failed")
    for row, msg in failed[:10]:
        print(f"  row {row}: {msg}")

    write_zarr_array(store, "normalized_wave", np.asarray(normalized_wave_rows), fill_value=np.nan)
    write_zarr_array(store, "normalized_flux", np.asarray(normalized_flux_rows), fill_value=np.nan)
    write_zarr_array(store, "continuum", np.asarray(continuum_rows), fill_value=np.nan)
    store.attrs["normalized_flux_source_files"] = [f"spectrum{i}.all" for i in range(nrows)]
    store.attrs["normalized_rows_aligned_to_raw"] = True
    store.attrs["suppnet_resampling_step_A"] = args.resampling_step
    store.attrs["suppnet_weights"] = args.weights

    print(f"Wrote normalized_wave, normalized_flux, continuum to {zarr_path}")


if __name__ == "__main__":
    main()
