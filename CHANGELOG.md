# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Documentation page *Paper data* and `tutorial/paper_results/fetch_paper_data.py`,
  which downloads the paper's grids and model outputs from the Hugging Face dataset
  `mjablonska/spice-paper-data` (tag v1.0) to the paths the paper scripts read.
- Wavelength-coverage warnings (`spice.spectrum.WavelengthCoverageWarning`).
  The passband functions warn when more than 0.5% of a filter's response lies
  outside the supplied wavelength grid, because the magnitude is then the band
  average over the covered part only. The synthesis entry points
  (`simulate_observed_flux`, `simulate_monochromatic_luminosity`,
  `luminosity`) warn when an emulator is
  asked for wavelengths outside its training range. New
  `Filter.wavelength_coverage(wavelengths)` and
  `SpectrumEmulator.wavelength_range`, read from the bundle extras for aemu
  bundles (`None`, i.e. unrestricted, by default). Checks run on concrete
  arrays only and are skipped under `jax.jit`.
- Active binary and legacy KD-tree regression tests covering orbital units,
  eclipse areas, scalar/list/stacked evaluation, and real PHOEBE integration,
  replacing the permanently skipped tests and missing pickle fixture.
- `find_binary_eclipses`: locates eclipse windows directly from a `Binary` or
  `PhoebeBinary` and returns contact times, mid-eclipse times, and eclipse
  kinds — no manual orbit evaluation needed. For a Keplerian `Binary` it
  samples a low-resolution orbit itself from the stored orbital elements; for
  a `PhoebeBinary` it scans PHOEBE's own precomputed uvw orbit samples, so
  the events match the PHOEBE meshes exactly (verified against
  `phoebe.default_binary()`: mid-eclipse at phases 0.0/0.5).
- `LICENSE` (MIT), `CONTRIBUTING.md`, and `docs/requirements.txt` (the latter
  unblocks the Read the Docs build, which referenced a missing file).
- Shared `spice.constants` module collecting the physical constants that were
  previously duplicated across `spectrum/` and `models/`.
- Docstrings for the computed properties of `MeshModel`.

### Changed
- The `aemu` extra now requires `astro-emulators-toolkit>=0.2,<0.3`; bundles trained
  with toolkit 0.2 (such as `mjablonska/TPayne-spice-harps-interp`) cannot be loaded by
  0.1. TransformerPayne bundles need 64-bit JAX (`JAX_ENABLE_X64=1`).
- **Breaking**: the default line-of-sight vector of `MeshModel` is now
  `[0., 0., -1.]` (observer → star), homogenizing the previously mesh-specific
  `[0., 1., 0.]` default with the convention used by `get_orbit_jax` and
  `eclipse_timestamps_kepler`. PHOEBE's velocity adapter uses `[0., 0., +1.]`
  to preserve radial-velocity signs in its imported uvw frame. The default rotation axis
  changed from `[0., 0., 1.]` to `[0., 1., 0.]` so that it stays in the default
  sky plane and a rotating star is still viewed equator-on by default (with the
  old z-axis it would have become pole-on, silently zeroing `los_velocities`).
  Pass explicit `los_vector` / `rotation_axis` values to recover the old
  behavior.
- **Breaking**: `override_log_g` in `IcosphereModel.construct` has inverted
  semantics. SPICE now computes per-element log g from mass and per-element
  radius by default (previously this required `override_log_g=True`);
  passing `override_log_g=True` now means "keep the explicitly supplied
  log g values" and issues a warning. `log_g_index` is resolved from
  `parameter_names` in all cases and stored on the model.
- `SpectrumEmulator` now inherits from `abc.ABC`, so its abstract methods are
  enforced.
- Limb-darkening law-id maps and helpers are consolidated to avoid the
  conflicting definitions between `limb_darkening.py` and
  `flux_limb_darkening.py`.
- Progress/diagnostic `print()` calls in library code now go through
  `spice.utils.log`.

### Fixed
- Keep the eclipse helper's grid size static under JAX compilation so legacy
  automatic neighbour estimation and explicit grid sizes work without tracer errors.
- Compute pulsation geometry from the combined vertex displacement, using
  consistently normalized area differences and preserving reference centers.
  Inactive modes no longer change gravity or produce negative/inflated
  luminosity. Reset also clears pulsation velocities, and evaluation reuses
  stored harmonic indices under JIT.
- Preserve wavelength coordinates in user-spectrum grids without a mu axis,
  reject dimension mismatches, and recognize the canonical `m/h` dictionary key.
- Respect arbitrary mu-axis order in Zarr intensity grids and keep wavelength
  sampling shared even when its length equals the number of model rows.
- Repair native mesh parameter updates by name and unknown-name errors.
- Apply default batch spot smoothness, compare angular spot radii consistently
  with chord distances, and keep harmonic spot patterns invariant to translation.
- Allow the packaged Gaia RVS CSV and attribution file through Git ignore rules
  so ordinary staging includes the restored data.
- Keep automatically computed atmosphere log g synchronized after rotation,
  pulsation, and reset, while retaining explicit `override_log_g=True` values.
  Rotation gradients remain finite for padded vertices and zero speed.
- Use projected triangle bounds for eclipse candidate rejection and mask
  rejected neighbours explicitly, fixing missed total eclipses for large
  foreground triangles and dependence on face storage order.
- Respect dictionary limb-darkening coefficients in `UserSpectrumInterpolator`
  and accept list, tuple, NumPy, and JAX parameter arrays, including JIT inputs.
- Repair variable-resolution convolution under JIT, including segment
  boundaries and short final segments.
- Package the official Gaia DR3 RVS response with provenance, read its
  wavelength/response columns correctly, and use separate AB and Vega
  zero points. Custom filters without explicit names now support photometry.
- Correct per-Angstrom flux conversions in AB, ST, and Vega photometry,
  including Gaia and energy-response filters. Absolute magnitudes previously
  affected by the spectral-density mismatch were offset by +20 mag.
- Integrate monochromatic luminosity over Angstroms, correcting the factor
  of 1e-8 error in bolometric luminosity.
- Align the installation with JAX 0.10: Python >=3.11 and NumPy >=2,<3;
  Read the Docs now uses Python 3.11.
- Accept scalar, list, and array parameters in mesh construction with current JAX.
- Preserve the configured rotation axis when changing the observer direction.
- Resolve occlusion for every face, including meshes with more than 60% of
  their triangles facing the observer.
- Convert rotation speeds from km/s to m/s in the centrifugal gravity term.
- Replace missing read-only PHOEBE test pickles with deterministic fixtures;
  optional-dependency smoke tests now raise on failures instead of returning False.
- The lazy-zarr interpolator tests no longer leak a stubbed `spice.utils`
  module into `sys.modules`, which broke any later test (or same-process
  import) that used `spice.utils.log`.
- `find_eclipses` no longer opens events at egress crossings, which recorded a
  bogus "eclipse" spanning the out-of-eclipse gap between two real events plus
  a degenerate zero-width event per period. It also drops events whose egress
  root collapses back onto the ingress time (scan-window-edge artifacts).
- `IcosphereModel.construct` no longer ignores `log_g_index=0` (a truthiness
  bug silently skipped the log g computation and emitted a spurious warning
  for index 0).
- `IntensityLazyZarrInterpolator.parameter_names` is now a property, matching the
  base class (it was previously a plain method).
- Array-truthiness hazard (`if not parameter_values`) in the parameter helpers,
  which raised on array inputs.
- PHOEBE/optional-dependency import guards narrowed from `except Exception` to
  `except ImportError` so genuine import errors are no longer masked.
- Removed dead code (`src/speed_test.py`), unused imports, and stale
  commented-out blocks; fixed typos ("obseved", "bonds") and an invalid string
  escape (`'$\AA$'`) that emitted a `SyntaxWarning`.

## [1.7.0]

- Baseline release prior to the changes recorded under *Unreleased*. See the
  Git history for details of earlier versions.
