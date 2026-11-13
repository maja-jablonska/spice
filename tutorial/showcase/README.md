# SPICE showcase (website / talks)

Animations, figures and two interactive pages that show off what SPICE does. Spectra come from
the TPayne-spice-harps-interp **v1.2.0** emulator (11 inputs). Do not use the laptop's old
`RozanskiT/TPayne-spice-harps` bundle: it predates the C/N/O fix. `common.load_emulator` refuses it.

| Script | What it shows | Output |
|---|---|---|
| `s01_rotation.py` | star spun up 0 → 100 km/s, Mg b broadening in step | GIF + PNG |
| `s02_doppler_imaging.py` | two cool spots, bump travelling through Fe I 6173, dynamic spectrum | GIF + PNG |
| `s03_eclipse.py` | F + K eclipsing binary: sky view, Strömgren y light curve, composite spectrum | GIF + PNG |
| `s04_cepheid.py` | δ Cep breathing on the SPIPS model, line asymmetry and RV | GIF + PNG |
| `s05_gradients.py` | `jax.grad` of the light curve w.r.t. R1, R2, i, Teff2 through both eclipses (+ FD check) | PNG/PDF + GIF |
| `s06_fit_loop.py` | Adam fit of synthetic photometry from a wrong start | GIF + PNG |
| `s07_speed.py` | timing chart from the v1.2.0 Gadi benchmarks (local, no compute) | PNG/PDF |
| `s08_star_builder_grid.py` + `build_pages.py` | interactive star builder (Teff, v_eq, i, spot, phase) | HTML |
| `s09_photometry_grid.py` + `build_pages.py` | interactive passband lab, page integral checked against SPICE | HTML |
| `s10_rossiter_mclaughlin.py` | hot-Jupiter transit, aligned vs 60° misaligned: Doppler shadow + RM RV | GIF + PNG |
| `s11_solar_oscillations.py` | Sun with l = 0–5 p modes (all power in 7 modes, so the RV is ~50× solar), line residuals, RV | GIF + PNG |

Every compute script takes `--out DIR [--quick] [--render-only] [--theme dark|light|both]`.
It writes `data.npz` and then renders both themes. `--render-only` restyles the figures without JAX.

Geometry convention: the spin axis is +z and the line of sight is `inclination_to_los_axis(i)`, so the
inclinations and spot latitudes are the true ones. Binary and transit spins are set prograde along the
orbit normal, and `common.add_aligned_rotation` checks the velocity field it actually gets.

## Running on Gadi

The jobs use `gadi/showcase.pbs` (dg97/gpursaa, aemu2 env, v1.2.0 bundle under `emu_train/.../ft_pen1z`,
checked by sha256). `gadi/submit_showcase.sh smoke` runs every compute script with `--quick` in one job.
`gadi/submit_showcase.sh full` then submits the four full jobs together, through `qsub_gpu.sh`
(MAX_GPU=5). Outputs go to `/scratch/y89/mj8805/showcase/out/<script>/`. Copy back only the GIF/PNG/PDF files
and the small `data.npz` of s08/s09.

Then build the pages locally:

    python build_pages.py --star out/s08_star_builder_grid/data.npz \
                          --phot out/s09_photometry_grid/data.npz --out pages_built

## Docs gallery

`docs/source/gallery.rst` shows the animations as animated WebP (800 px wide, quality 55, 0.4-1.1 MB each)
in `docs/img/showcase/`, converted from the GIFs with Pillow:

    from PIL import Image, ImageSequence
    im = Image.open("eclipse_dark.gif"); fr = [f.convert("RGB") for f in ImageSequence.Iterator(im)]
    s = [f.resize((800, round(f.height * 800 / f.width)), Image.LANCZOS) for f in fr]
    s[0].save("eclipse_dark.webp", "WEBP", save_all=True, append_images=s[1:],
              duration=im.info["duration"], loop=0, quality=55, method=6)

Regenerate them whenever a script is re-run.
