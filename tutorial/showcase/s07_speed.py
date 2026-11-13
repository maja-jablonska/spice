"""Showcase 7 - speed slide.

Wall time per disc-integrated spectrum on Gadi GPUs, emulator v1.2.0 (x64), from the
2026-11-07 benchmark runs (jobs 183397925 A100, 183397926 V100):
benchmarks/benchmark_results/spice_components_1833979{25,26}.gadi-pbs.csv.
Two small multiples (mesh size, wavelength count), log-log, neural emulator vs a
precomputed static spectrum, solid = A100, dashed = V100, direct labels.
No compute: runs locally in seconds.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

RES = HERE.parent.parent / "benchmarks" / "benchmark_results"
RUNS = {"A100": "spice_components_183397925.gadi-pbs.csv", "V100": "spice_components_183397926.gadi-pbs.csv"}


def load():
    import pandas as pd
    out = {}
    for gpu, f in RUNS.items():
        df = pd.read_csv(RES / f)
        out[gpu] = df[df.precision == "x64"]
    return out


def render(out, theme):
    import matplotlib.pyplot as plt
    t = C.apply_theme(theme)
    data = load()
    col = {"aemu_harps_interp": t["accent"], "static": t["accent2"]}
    name = {"aemu_harps_interp": "emulator", "static": "static"}
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw=dict(wspace=0.45, left=0.08, right=0.88,
                                                                     top=0.72, bottom=0.14))
    for ax, sweep, xcol, xlabel, fixed in [
        (axs[0], "mesh", "n_mesh_elements", "Surface elements", "2,000 wavelengths"),
        (axs[1], "wavelengths", "n_wavelengths", "Wavelengths", "5,120 elements")]:
        for src in ("aemu_harps_interp", "static"):
            for gpu, ls in (("A100", "-"), ("V100", "--")):
                df = data[gpu]
                s = df[(df.sweep == sweep) & (df.source == src)].sort_values(xcol)
                x, y = s[xcol].values, 1e3 * s.mean_time_s.values
                ax.loglog(x, y, ls, color=col[src], lw=2, marker="o", ms=8,
                          markeredgecolor=t["bg"], markeredgewidth=2)
                ax.annotate(f"{name[src]} {gpu}", (x[-1], y[-1]), xytext=(8, (6 if gpu == "V100" else -6) if src == "static" else 0), textcoords="offset points",
                            va="center", fontsize=10, color=t["muted"], annotation_clip=False)
        ax.set_xlabel(xlabel); ax.set_title(f"{xlabel} sweep ({fixed})", loc="left", color=t["muted"], fontsize=12)
        ax.grid(True, which="major", color=t["grid"], lw=0.8); ax.set_axisbelow(True)
        xs = sorted(data["A100"][data["A100"].sweep == sweep][xcol].unique())
        ax.set_xticks(xs); ax.set_xticklabels([f"{v:,}" for v in xs]); ax.minorticks_off()
    axs[0].set_ylabel("Time per spectrum [ms]")
    a100 = data["A100"]
    hero = a100[(a100.sweep == "wavelengths") & (a100.source == "aemu_harps_interp") & (a100.n_wavelengths == 1000)]
    ms = 1e3 * float(hero.mean_time_s.iloc[0])
    per = float(hero.mean_time_s.iloc[0]) / (5120 * 1000) * 1e9
    fig.text(0.08, 0.95, f"{ms:.0f} ms", fontsize=30, color=t["fg"], weight="bold", va="top")
    fig.text(0.08, 0.84, "per full neural-emulator spectrum: 5,120 surface elements × 1,000 wavelengths on one A100\n"
             f"≈ {per:.0f} ns per element-wavelength; mu-dependent intensities, Doppler-shifted, every element",
             fontsize=11.5, color=t["muted"], va="center")
    C.save_still(fig, out / f"speed_{theme}")
    plt.close(fig)


if __name__ == "__main__":
    a = C.parser(__doc__).parse_args()
    o = C.outdir(a)
    for th in C.themes(a.theme):
        render(o, th)
