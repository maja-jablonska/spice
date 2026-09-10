# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
- **Breaking**: the default line-of-sight vector of `MeshModel` is now
  `[0., 0., -1.]` (observer → star), homogenizing the previously mesh-specific
  `[0., 1., 0.]` default with the convention already used by `PhoebeModel`,
  `get_orbit_jax`, and `eclipse_timestamps_kepler`. The default rotation axis
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
