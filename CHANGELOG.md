# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `LICENSE` (MIT), `CONTRIBUTING.md`, and `docs/requirements.txt` (the latter
  unblocks the Read the Docs build, which referenced a missing file).
- Shared `spice.constants` module collecting the physical constants that were
  previously duplicated across `spectrum/` and `models/`.
- Docstrings for the computed properties of `MeshModel`.

### Changed
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
