# `kochukhov2017_fe_lte.zarr` — LTE specific-intensity grid for the spotted-star test

The grid is published in the Hugging Face dataset
[mjablonska/spice-paper-data](https://huggingface.co/datasets/mjablonska/spice-paper-data) (`kochukhov2017_fe_lte/`);
`python tutorial/paper_results/fetch_paper_data.py --only kochukhov` downloads it to `data/kochukhov2017_fe_lte.zarr`.

Dedicated grid for Section 3.1 of the SPICE paper (the Doppler-imaging test case of
Kochukhov 2017, A&A 597, A58). It exists because that test needs iron abundances up to
A(Fe) = 9.5, two dex above solar, which neither the paper's small regular NLTE grid
(stops at [Fe/H] = 0) nor the tabulated Fe NLTE departure coefficients (scaled-solar iron
only, A(Fe) <= 8.5) cover.

## Contents

| | |
|---|---|
| Rows | 340 = 17 iron abundances x 20 viewing angles, one model atmosphere |
| Wavelengths | 5000.00-5020.00 A, step 0.01 A (2001 points), air wavelengths, sampling R ~ 500,000 |
| Arrays | `flux` and `continuum` (specific intensities, erg s^-1 cm^-2 A^-1 sr^-1, Turbospectrum `Intensity` mode), `params`, `param_names`, `wavelength`, `global_index`, `provenance/*` |
| SPICE index | `index.parquet`, built with `python -m spice.spectrum.generate_zarr_index` |

## Axes

| Parameter | Values |
|---|---|
| `teff` | 8000 K (fixed) |
| `logg` | 4.0 (fixed) |
| `feh` | 0.0 (atmosphere metallicity and scaling of all elements except Fe; fixed) |
| `vmicro` / `turbvel` | 2.0 km/s (fixed) |
| `fe` | A(Fe) = 12 + log(N_Fe/N_H) = 8.0, 8.1, ..., 9.6 (17 nodes) |
| `mu` | 0.05, 0.10, ..., 1.00 (20 angles) |

The `marcs_*` columns record the atmosphere actually used (the same model for every row).
Load in SPICE with

```python
IntensityLazyZarrInterpolator('data/kochukhov2017_fe_lte.zarr',
                              params=['teff', 'logg', 'fe', 'mu', 'vmicro'], sparse=True)
```

and set `fe` per surface element. For Kochukhov's abundances, log(N_Fe/N_tot) = -4.0
(photosphere) and -2.5 (spot centres) correspond to A(Fe) = 8.035 and 9.535 on the Magg et
al. (2022) solar scale used by the synthesis (A(Fe)_sun = 7.46, log N_H/N_tot = -0.035),
i.e. [Fe/H] = +0.58 and +2.08.

## Physics

* Turbospectrum v20 (NLTE fork, https://github.com/maja-jablonska/Turbospectrum_NLTE,
  pipeline commit `f2c8afd`), gfortran build, `Intensity` mode with explicit mu points.
* **LTE** line formation (`calculation_mode: LTE`, `nlte: false`), matching Kochukhov (2017).
* Model atmosphere: MARCS standard composition, plane-parallel,
  `p8000_g+4.0_m0.0_t02_st_z+0.00_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.mod`
  (sha256 of the MARCS set `9a62d76f6abba50f...`).
* Abundances: Magg et al. (2022) solar scale (`ABUND_SOURCE magg`), metallicity 0.0,
  [alpha/Fe] = 0; iron overridden per row via Turbospectrum `INDIVIDUAL ABUNDANCES`
  (absolute A(Fe)).
* Line list: `nlte_ges_linelist_jmg16jan2026_I_II` (GES atomic list, January 2026 version;
  sha256 `4c9bde82...`), the same list as the paper's small regular grid. No molecules.
* Microturbulence 2 km/s in both the atmosphere label (`t02`) and the synthesis.

Full provenance (configs, manifests, hashes, environment) is stored inside the zarr under
`provenance/`.

## How it was made

```bash
# in a checkout of https://github.com/maja-jablonska/Turbospectrum_NLTE (our fork), with the config copied to configs/pipeline/
cd Turbospectrum_NLTE
python scripts/synthesize_regular_grid.py \
    --config-json configs/pipeline/config_kochukhov2017_fe_lte.json --no-scratch
cp -R runs/kochukhov2017-fe-lte/outputs/zarr/regular_synthesized_spectra.zarr \
      <spice>/data/kochukhov2017_fe_lte.zarr
python -m spice.spectrum.generate_zarr_index <spice>/data/kochukhov2017_fe_lte.zarr
```

The config is kept next to this file as `config_kochukhov2017_fe_lte.json`. The run takes
about two minutes on a laptop (17 Turbospectrum calls).

## Used by

`kochukhov2017_fe_lte.ipynb` (this directory), which produces the paper's Figs. 7-9 and the
spectra pickle `kochukhov2017_fe_lte_spectra.pkl`.
