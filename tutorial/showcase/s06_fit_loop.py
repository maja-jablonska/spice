"""Showcase 6 - gradient-based fit, step by step.

Synthetic Stromgren-y photometry of the showcase F + K binary (truth = the s03
system, 3 mmag Gaussian noise, both eclipses) is fitted from a deliberately wrong
start with Adam on exact SPICE gradients (R1, R2, inclination, Teff2; the zero point
is solved analytically each step). The animation shows the model light curve
snapping onto the data while the parameters converge to the truth.
Synthetic data: this demonstrates the machinery, not a published fit.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

START = np.array([1.45, 0.75, 84.6, 5300.0])
SCALE = np.array([0.05, 0.05, 1.0, 200.0])
NOISE = 0.003


def compute(args, out):
    C.setup_jax()
    import jax
    import jax.numpy as jnp
    import difflc as D

    emu = C.load_emulator()
    lc = D.LightCurve(emu, n_vert=300 if args.quick else 1000, n_wl=120 if args.quick else 400)
    truth = jnp.asarray(D.THETA0)
    wins = D.eclipse_windows(lc, truth, margin=0.3)
    n_per = 6 if args.quick else 28
    phases = np.concatenate([np.linspace(a, b, n_per) for _, a, b in wins])
    win_id = np.repeat(np.arange(len(wins)), n_per)
    rng = np.random.default_rng(42)
    data = np.asarray(lc.curve(truth, phases)) + rng.normal(0, NOISE, len(phases))
    print("data done", flush=True)

    n_steps = 3 if args.quick else 80
    z = np.zeros(4)
    m_, v_ = np.zeros(4), np.zeros(4)
    lr, b1, b2, eps = 0.12, 0.9, 0.999, 1e-8
    thetas, losses, models = [], [], []
    for k in range(n_steps):
        th = START + z * SCALE
        mj = [lc.mag_jac(th, ph) for ph in phases]      # forward-mode, phase by phase (memory)
        model = np.array([m for m, _ in mj]); J = np.array([j for _, j in mj])
        r = data - model
        rc = r - r.mean()                                 # analytic zero point
        L = 0.5 * np.sum((rc / NOISE) ** 2) / len(phases)
        g_theta = -(rc / NOISE ** 2) @ J / len(phases)    # sum(rc) = 0, so the zero point drops out
        g = g_theta * SCALE
        thetas.append(th.copy()); losses.append(float(L)); models.append(model + r.mean())
        print(f"step {k:3d}  chi2/N {2 * L:10.3f}  theta {th}", flush=True)
        m_ = b1 * m_ + (1 - b1) * g
        v_ = b2 * v_ + (1 - b2) * g ** 2
        step_lr = lr * (0.25 + 0.75 * 0.5 * (1 + np.cos(np.pi * k / n_steps)))   # cosine decay to lr/4
        z = z - step_lr * (m_ / (1 - b1 ** (k + 1))) / (np.sqrt(v_ / (1 - b2 ** (k + 1))) + eps)
        if k % 5 == 4:   # checkpoint: a walltime kill still leaves a renderable fit (--render-only)
            np.savez_compressed(out / "data.npz", phases=phases, win_id=win_id, data=data, truth=np.asarray(truth),
                                thetas=np.array(thetas), losses=np.array(losses), models=np.array(models),
                                kinds=np.array([w[0] for w in wins]))
    np.savez_compressed(out / "data.npz", phases=phases, win_id=win_id, data=data, truth=np.asarray(truth),
                        thetas=np.array(thetas), losses=np.array(losses), models=np.array(models),
                        kinds=np.array([w[0] for w in wins]))


def render(out, theme):
    import matplotlib.pyplot as plt
    import difflc as D
    t = C.apply_theme(theme)
    d = np.load(out / "data.npz")
    ph, wid, y, truth = d["phases"], d["win_id"], d["data"], d["truth"]
    th, L, models, kinds = d["thetas"], d["losses"], d["models"], d["kinds"]
    n = len(L)
    colors = [t["accent2"], t["accent"], t["accent4"], t["accent3"]]
    labels = ["$R_1$", "$R_2$", "$i$", "$T_{\\mathrm{eff},2}$"]
    # hold the final state for a second
    order = list(range(n)) + [n - 1] * 15

    def frame(f):
        k = order[f]
        fig = plt.figure(figsize=(13, 6))
        axl = [fig.add_axes([0.07 + 0.3 * j, 0.14, 0.26, 0.74]) for j in range(len(kinds))]
        axp = fig.add_axes([0.70, 0.52, 0.28, 0.36])
        axL = fig.add_axes([0.70, 0.14, 0.28, 0.26])
        for j, ax in enumerate(axl):
            s = wid == j
            ax.plot(ph[s], y[s], "o", ms=4, color=t["obs"], label="synthetic data")
            ax.plot(ph[s], models[k][s], color=t["accent"], lw=2.2, label="SPICE model")
            lo, hi = y[s].min(), y[s].max(); pad = 0.12 * (hi - lo)
            ax.set_ylim(max(hi, models[k][s].max()) + pad, min(lo, models[k][s].min()) - pad)
            ax.set_xlabel("Orbital phase"); ax.set_title(kinds[j].split(" (")[0], loc="left", color=t["muted"])
        axl[0].set_ylabel("Strömgren y [mag]"); axl[0].legend(loc="lower left")
        for p in range(4):
            dev = (th[:k + 1, p] - truth[p]) / SCALE[p]
            axp.plot(np.arange(k + 1), dev, color=colors[p], lw=2, label=labels[p])
        axp.axhline(0, color=t["muted"], lw=1, ls="--")
        axp.set_xlim(0, n); axp.set_ylim(-6, 6)
        axp.set_ylabel("(θ − truth) / scale"); axp.set_xticklabels([])
        axp.legend(ncol=4, loc="upper right", fontsize=10, handlelength=1.2)
        axL.semilogy(np.arange(k + 1), 2 * L[:k + 1], color=t["fg"], lw=2)
        axL.set_xlim(0, n); axL.set_ylim(0.5, 2 * L.max() * 1.5)
        axL.set_xlabel("Adam step"); axL.set_ylabel("χ²/N")
        fig.text(0.70, 0.93, f"step {k}   χ²/N = {2 * L[k]:.2f}", color=t["fg"], fontsize=14)
        return fig

    C.save_animation(frame, len(order), out / f"fit_loop_{theme}", fps=12, still_index=len(order) - 1)


if __name__ == "__main__":
    a = C.parser(__doc__).parse_args()
    o = C.outdir(a)
    if not a.render_only:
        compute(a, o)
    for th in C.themes(a.theme):
        render(o, th)
