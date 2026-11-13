"""Showcase 5 - gradients through an eclipse.

Exact (forward-mode autodiff) derivatives of the Stromgren-y magnitude with respect to R1, R2, inclination and the
secondary's Teff, phase by phase through both eclipses of the showcase F + K binary.
The curves show *where in the light curve each parameter carries information* --
something a finite-difference code has to buy with 2N extra light curves.
A few phases are checked against central finite differences (printed + saved).
Outputs: a still figure (both eclipses) and a GIF sweeping through the primary
eclipse with the sky view.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

UNITS = [("R1", 0.01, "per 0.01 R$_\\odot$"), ("R2", 0.01, "per 0.01 R$_\\odot$"),
         ("incl", 0.1, "per 0.1°"), ("Teff2", 10.0, "per 10 K")]
LABELS = {"R1": "$R_1$", "R2": "$R_2$", "incl": "$i$", "Teff2": "$T_{\\mathrm{eff},2}$"}


def compute(args, out):
    C.setup_jax()
    import jax
    import jax.numpy as jnp
    import difflc as D

    emu = C.load_emulator()
    lc = D.LightCurve(emu, n_vert=400 if args.quick else 2000, n_wl=150 if args.quick else 500)
    th0 = jnp.asarray(D.THETA0)
    wins = D.eclipse_windows(lc, th0)
    print("eclipse windows (phase):", wins, flush=True)
    n_ph = 6 if args.quick else 90
    res = {}
    for kind, a, b in wins:
        phs = np.linspace(a, b, n_ph)
        vals, grads = [], []
        for ph in phs:
            v, g = lc.mag_jac(th0, ph)          # forward mode (see difflc.mag_jac)
            vals.append(v); grads.append(g)
        res[kind] = (phs, np.array(vals), np.array(grads))
        print(f"{kind}: {n_ph} gradients done", flush=True)
    # finite-difference spot check at the deepest phase of each eclipse
    steps = np.array([0.005, 0.005, 0.02, 5.0])
    fd_rows = []
    for kind, (phs, vals, grads) in res.items():
        j = int(np.argmax(vals))
        for p in range(4):
            e = np.zeros(4); e[p] = steps[p]
            fd = (float(lc.mag(th0 + e, phs[j])) - float(lc.mag(th0 - e, phs[j]))) / (2 * steps[p])
            fd_rows.append((kind, D.NAMES[p], grads[j, p], fd))
            print(f"FD check [{kind}] phase {phs[j]:.4f} d/d{D.NAMES[p]}: AD {grads[j, p]:+.5e}  FD {fd:+.5e}", flush=True)
    # sky snapshots for the primary-eclipse GIF
    kind0 = sorted(res)[0]                       # the primary eclipse
    snaps = []
    for ph in res[kind0][0]:
        m1, m2 = lc.meshes(th0, ph)
        snaps += [C.snapshot(m, {"teff": np.asarray(m.parameters)[:, 0]}, up=(0.0, 1.0, 0.0)) for m in (m1, m2)]
    save = {"kinds": np.array(list(res)), "gif_kind": kind0, "theta0": np.asarray(th0),
            "fd": np.array([(r[2], r[3]) for r in fd_rows]), "fd_labels": np.array([f"{r[0]}:{r[1]}" for r in fd_rows])}
    for kind, (phs, vals, grads) in res.items():
        save[f"{kind}_phase"], save[f"{kind}_mag"], save[f"{kind}_grad"] = phs, vals, grads
    np.savez_compressed(out / "data.npz", **save, **C.pack_snapshots(snaps))


def _panels(fig, axes, d, kind, t, cursor=None):
    colors = [t["accent2"], t["accent"], t["accent4"], t["accent3"]]
    phs, mag, g = d[f"{kind}_phase"], d[f"{kind}_mag"], d[f"{kind}_grad"]
    axes[0].plot(phs, mag, color=t["fg"], lw=2)
    axes[0].set_ylim(mag.max() + 0.05 * np.ptp(mag), mag.min() - 0.05 * np.ptp(mag))
    axes[0].set_ylabel("y [mag]")
    axes[0].set_title(kind, loc="left", color=t["muted"])
    for p, (ax, (nm, scale, unit)) in enumerate(zip(axes[1:], UNITS)):
        ax.axhline(0, color=t["grid"], lw=1)
        ax.plot(phs, 1e3 * g[:, p] * scale, color=colors[p], lw=2)
        ax.set_ylabel(f"∂m/∂{LABELS[nm]}\n[mmag {unit.split('per ')[1]}]", fontsize=10)
    for ax in axes:
        ax.set_xlim(phs[0], phs[-1])
        if cursor is not None:
            ax.axvline(cursor, color=t["muted"], lw=1, ls="--")
    for ax in axes[:-1]:
        ax.set_xticklabels([])
    axes[-1].set_xlabel("Orbital phase")


def render(out, theme):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    t = C.apply_theme(theme)
    d = np.load(out / "data.npz")
    kinds = list(d["kinds"])
    fig, axs = plt.subplots(5, len(kinds), figsize=(6 * len(kinds), 9), sharex="col", squeeze=False,
                            gridspec_kw=dict(hspace=0.12, wspace=0.35, left=0.1, right=0.97, top=0.95, bottom=0.07))
    for c, kind in enumerate(kinds):
        _panels(fig, axs[:, c], d, kind, t)
    C.save_still(fig, out / f"gradients_{theme}")
    plt.close(fig)

    kind = str(d["gif_kind"])
    snaps = C.unpack_snapshots(d)
    phs = d[f"{kind}_phase"]
    allpoly = np.concatenate([s["poly"] for s in snaps])
    cx, cy = np.median(allpoly[..., 0]), np.median(allpoly[..., 1])
    lim = 1.25 * (np.abs(allpoly[..., 0] - cx).max())
    teff = np.concatenate([s["teff"] for s in snaps])
    norm = Normalize(teff.min() - 300, teff.max() + 300)

    def frame(k):
        fig = plt.figure(figsize=(12, 7))
        ax0 = fig.add_axes([0.02, 0.08, 0.4, 0.84])
        axes = [fig.add_axes([0.55, 0.79 - 0.175 * j, 0.42, 0.15]) for j in range(5)]
        C.draw_disk(ax0, [snaps[2 * k], snaps[2 * k + 1]], "teff", "inferno", norm)
        ax0.set_xlim(cx - lim, cx + lim); ax0.set_ylim(cy - lim, cy + lim); ax0.set_aspect("equal"); ax0.axis("off")
        ax0.set_title("JAX autodiff through the occlusion", color=t["fg"])
        _panels(fig, axes, d, kind, t, cursor=phs[k])
        return fig

    C.save_animation(frame, len(phs), out / f"gradients_sweep_{theme}", fps=15, still_index=len(phs) // 2)


if __name__ == "__main__":
    a = C.parser(__doc__).parse_args()
    o = C.outdir(a)
    if not a.render_only:
        compute(a, o)
    for th in C.themes(a.theme):
        render(o, th)
