"""Regression tests for the bundle-metadata schema tolerance in
``spice.spectrum.aemu_spectrum_emulator``.

The module supports two ``astro_emulators_toolkit`` bundle layouts:

* the *old* split layout, where the ``Emulator`` exposes
  ``reference_scaling_inputs`` / ``reference_scaling_outputs`` as separate
  top-level dicts, and ``input_domain`` ``min/max_tree`` carries ``parameters``
  at the top level;
* the *new* combined layout, where a single ``reference_scaling`` dict carries
  ``min_tree`` / ``max_tree`` nested by section
  (``{"inputs": {...}, "outputs": {...}}``), and ``input_domain`` is likewise
  nested by section.

These tests lock that tolerance down by building fake emulators (plain
``SimpleNamespace`` objects -- the helpers under test only do attribute /
dict lookups, they never call into the toolkit runtime) for both layouts and
asserting they resolve to equivalent affine blocks / parameter bounds.

The tests run without the ``[aemu]`` optional extra installed: a synthetic
``astro_emulators_toolkit`` stand-in is injected into ``sys.modules`` before
the module under test is imported, satisfying its eager
``_lazy_import_astro_emulators_toolkit`` call.
"""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest


def _install_fake_aemu_modules() -> None:
    """Inject minimal stand-ins for ``astro_emulators_toolkit`` and
    ``astro_emulators_toolkit.emulator_runtime`` into ``sys.modules`` so the
    module under test can be imported without the optional extra installed.

    Only attributes referenced at module import time / by the helpers under
    test need to exist; the schema helpers themselves never touch the
    toolkit runtime."""
    if "astro_emulators_toolkit" not in sys.modules:
        fake_pkg = types.ModuleType("astro_emulators_toolkit")
        fake_pkg.Emulator = type("Emulator", (), {})
        fake_pkg.normalize_tree = lambda tree, *a, **kw: tree
        fake_pkg.denormalize_tree = lambda tree, *a, **kw: tree
        sys.modules["astro_emulators_toolkit"] = fake_pkg
    if "astro_emulators_toolkit.emulator_runtime" not in sys.modules:
        fake_runtime = types.ModuleType("astro_emulators_toolkit.emulator_runtime")
        fake_runtime.apply_jax_runtime = lambda *a, **kw: None
        fake_runtime.make_frozen_apply_runtime = lambda *a, **kw: (lambda x: x)
        sys.modules["astro_emulators_toolkit.emulator_runtime"] = fake_runtime


_install_fake_aemu_modules()

# Import after the shim is in place. If the real toolkit is installed, the
# shim no-ops and the real module is used -- either way the helpers under
# test are pure-Python and behave identically.
aemu_module = importlib.import_module("spice.spectrum.aemu_spectrum_emulator")

_affine_section_of_combined = aemu_module._affine_section_of_combined
_affine_reference_scaling = aemu_module._affine_reference_scaling
_input_domain_param_bounds = aemu_module._input_domain_param_bounds


def _split_layout_emulator(inputs_block, outputs_block, *, domain=None):
    """Build a fake emulator exposing the *old* split top-level attributes.

    ``domain`` is the ``input_domain`` dict to attach (flat-or-nested layout
    chosen by the caller); ``None`` omits the attribute entirely."""
    attrs = {
        "reference_scaling_inputs": inputs_block,
        "reference_scaling_outputs": outputs_block,
        "reference_scaling": None,
        "spec": None,
    }
    if domain is not None:
        attrs["input_domain"] = domain
    return SimpleNamespace(**attrs)


def _combined_layout_emulator(combined_block, *, domain=None):
    """Build a fake emulator exposing the *new* combined top-level attribute.

    The split attributes are deliberately set to ``None`` so the helper
    must fall through to the combined block."""
    attrs = {
        "reference_scaling_inputs": None,
        "reference_scaling_outputs": None,
        "reference_scaling": combined_block,
        "spec": None,
    }
    if domain is not None:
        attrs["input_domain"] = domain
    return SimpleNamespace(**attrs)


def _spec_only_emulator(spec):
    """Build a fake emulator that exposes scaling only via ``emulator.spec``,
    not as a direct attribute -- the deepest fallback path."""
    return SimpleNamespace(
        reference_scaling_inputs=None,
        reference_scaling_outputs=None,
        reference_scaling=None,
        spec=spec,
    )


# Canonical numeric content shared across layout variants so we can prove the
# helpers return *equivalent* affine blocks regardless of declared layout.
_INPUTS_MIN = {"parameters": [3000.0, -2.0], "wavelengths": [3.5]}
_INPUTS_MAX = {"parameters": [12000.0, 0.5], "wavelengths": [4.2]}
_OUTPUTS_MIN = {"flux": [-25.0]}
_OUTPUTS_MAX = {"flux": [-5.0]}


class TestAffineSectionOfCombined:
    def test_extracts_inputs_section(self):
        combined = {
            "min_tree": {"inputs": _INPUTS_MIN, "outputs": _OUTPUTS_MIN},
            "max_tree": {"inputs": _INPUTS_MAX, "outputs": _OUTPUTS_MAX},
        }
        block = _affine_section_of_combined(combined, "inputs")
        assert block == {"min_tree": _INPUTS_MIN, "max_tree": _INPUTS_MAX}

    def test_extracts_outputs_section(self):
        combined = {
            "min_tree": {"inputs": _INPUTS_MIN, "outputs": _OUTPUTS_MIN},
            "max_tree": {"inputs": _INPUTS_MAX, "outputs": _OUTPUTS_MAX},
        }
        block = _affine_section_of_combined(combined, "outputs")
        assert block == {"min_tree": _OUTPUTS_MIN, "max_tree": _OUTPUTS_MAX}

    def test_forwards_kind_and_parametrization_metadata(self):
        combined = {
            "kind": "affine",
            "parametrization": "min_max",
            "min_tree": {"inputs": _INPUTS_MIN, "outputs": _OUTPUTS_MIN},
            "max_tree": {"inputs": _INPUTS_MAX, "outputs": _OUTPUTS_MAX},
        }
        block = _affine_section_of_combined(combined, "inputs")
        assert block["kind"] == "affine"
        assert block["parametrization"] == "min_max"

    def test_returns_none_when_min_or_max_tree_not_section_nested(self):
        # A combined block whose min/max_tree are leaf-dicts (the *old* split
        # layout shape, accidentally placed under a combined wrapper) must
        # not be misinterpreted as a section-nested combined block.
        not_section_nested = {
            "min_tree": _INPUTS_MIN,
            "max_tree": _INPUTS_MAX,
        }
        assert _affine_section_of_combined(not_section_nested, "inputs") is None

    @pytest.mark.parametrize("bad", [None, "not-a-dict", 7, []])
    def test_returns_none_for_non_dict_input(self, bad):
        assert _affine_section_of_combined(bad, "inputs") is None


class TestAffineReferenceScaling:
    def test_old_split_layout_returns_block_unchanged(self):
        inputs_block = {"min_tree": _INPUTS_MIN, "max_tree": _INPUTS_MAX}
        outputs_block = {"min_tree": _OUTPUTS_MIN, "max_tree": _OUTPUTS_MAX}
        emu = _split_layout_emulator(inputs_block, outputs_block)
        assert _affine_reference_scaling(emu, "inputs") is inputs_block
        assert _affine_reference_scaling(emu, "outputs") is outputs_block

    def test_new_combined_layout_resolves_per_side_block(self):
        combined = {
            "min_tree": {"inputs": _INPUTS_MIN, "outputs": _OUTPUTS_MIN},
            "max_tree": {"inputs": _INPUTS_MAX, "outputs": _OUTPUTS_MAX},
        }
        emu = _combined_layout_emulator(combined)
        assert _affine_reference_scaling(emu, "inputs") == {
            "min_tree": _INPUTS_MIN,
            "max_tree": _INPUTS_MAX,
        }
        assert _affine_reference_scaling(emu, "outputs") == {
            "min_tree": _OUTPUTS_MIN,
            "max_tree": _OUTPUTS_MAX,
        }

    def test_falls_back_to_split_layout_in_spec(self):
        # The top-level attributes are missing/None; the only declaration
        # is in ``emulator.spec`` under the old split keys.
        inputs_block = {"min_tree": _INPUTS_MIN, "max_tree": _INPUTS_MAX}
        outputs_block = {"min_tree": _OUTPUTS_MIN, "max_tree": _OUTPUTS_MAX}
        emu = _spec_only_emulator(
            {
                "reference_scaling_inputs": inputs_block,
                "reference_scaling_outputs": outputs_block,
            }
        )
        assert _affine_reference_scaling(emu, "inputs") is inputs_block
        assert _affine_reference_scaling(emu, "outputs") is outputs_block

    def test_falls_back_to_combined_layout_in_spec(self):
        combined = {
            "min_tree": {"inputs": _INPUTS_MIN, "outputs": _OUTPUTS_MIN},
            "max_tree": {"inputs": _INPUTS_MAX, "outputs": _OUTPUTS_MAX},
        }
        emu = _spec_only_emulator({"reference_scaling": combined})
        assert _affine_reference_scaling(emu, "inputs") == {
            "min_tree": _INPUTS_MIN,
            "max_tree": _INPUTS_MAX,
        }
        assert _affine_reference_scaling(emu, "outputs") == {
            "min_tree": _OUTPUTS_MIN,
            "max_tree": _OUTPUTS_MAX,
        }

    def test_returns_none_when_nothing_declared(self):
        emu = SimpleNamespace(
            reference_scaling_inputs=None,
            reference_scaling_outputs=None,
            reference_scaling=None,
            spec=None,
        )
        assert _affine_reference_scaling(emu, "inputs") is None
        assert _affine_reference_scaling(emu, "outputs") is None

    def test_top_level_split_layout_wins_over_combined(self):
        # When both layouts are present, the eager (split) attribute should
        # take precedence -- that's the actual order in the helper and the
        # documented priority. We assert it explicitly so a future refactor
        # cannot silently reorder the lookup.
        split_block = {"min_tree": _INPUTS_MIN, "max_tree": _INPUTS_MAX}
        decoy_combined = {
            "min_tree": {"inputs": {"parameters": [0.0]}, "outputs": {"flux": [0.0]}},
            "max_tree": {"inputs": {"parameters": [1.0]}, "outputs": {"flux": [1.0]}},
        }
        emu = SimpleNamespace(
            reference_scaling_inputs=split_block,
            reference_scaling_outputs=None,
            reference_scaling=decoy_combined,
            spec=None,
        )
        assert _affine_reference_scaling(emu, "inputs") is split_block

    def test_layouts_resolve_to_equivalent_blocks(self):
        # The strongest parity check: identical declared scaling under
        # either layout must yield identical (per-side) affine blocks, so
        # downstream code (normalize_tree, etc.) sees the same content.
        split = _split_layout_emulator(
            {"min_tree": _INPUTS_MIN, "max_tree": _INPUTS_MAX},
            {"min_tree": _OUTPUTS_MIN, "max_tree": _OUTPUTS_MAX},
        )
        combined = _combined_layout_emulator(
            {
                "min_tree": {"inputs": _INPUTS_MIN, "outputs": _OUTPUTS_MIN},
                "max_tree": {"inputs": _INPUTS_MAX, "outputs": _OUTPUTS_MAX},
            }
        )
        for side in ("inputs", "outputs"):
            a = _affine_reference_scaling(split, side)
            b = _affine_reference_scaling(combined, side)
            assert a is not None and b is not None
            assert a["min_tree"] == b["min_tree"]
            assert a["max_tree"] == b["max_tree"]


class TestInputDomainParamBounds:
    def test_old_flat_layout(self):
        bounds = np.array([3000.0, -2.0])
        domain = {
            "min_tree": {"parameters": bounds, "wavelengths": np.array([3.5])},
            "max_tree": {"parameters": np.array([12000.0, 0.5]), "wavelengths": np.array([4.2])},
        }
        emu = SimpleNamespace(input_domain=domain)
        np.testing.assert_array_equal(
            _input_domain_param_bounds(emu, "min"), bounds
        )
        np.testing.assert_array_equal(
            _input_domain_param_bounds(emu, "max"), domain["max_tree"]["parameters"]
        )

    def test_new_section_nested_layout(self):
        bounds_min = np.array([3000.0, -2.0])
        bounds_max = np.array([12000.0, 0.5])
        domain = {
            "min_tree": {"inputs": {"parameters": bounds_min, "wavelengths": np.array([3.5])}},
            "max_tree": {"inputs": {"parameters": bounds_max, "wavelengths": np.array([4.2])}},
        }
        emu = SimpleNamespace(input_domain=domain)
        np.testing.assert_array_equal(
            _input_domain_param_bounds(emu, "min"), bounds_min
        )
        np.testing.assert_array_equal(
            _input_domain_param_bounds(emu, "max"), bounds_max
        )

    def test_both_layouts_resolve_equally(self):
        # Parity: equivalent declared bounds under either layout must yield
        # the same returned array.
        bounds = np.array([3000.0, -2.0])
        flat = SimpleNamespace(
            input_domain={
                "min_tree": {"parameters": bounds},
                "max_tree": {"parameters": np.array([12000.0, 0.5])},
            }
        )
        nested = SimpleNamespace(
            input_domain={
                "min_tree": {"inputs": {"parameters": bounds}},
                "max_tree": {"inputs": {"parameters": np.array([12000.0, 0.5])}},
            }
        )
        np.testing.assert_array_equal(
            _input_domain_param_bounds(flat, "min"),
            _input_domain_param_bounds(nested, "min"),
        )
        np.testing.assert_array_equal(
            _input_domain_param_bounds(flat, "max"),
            _input_domain_param_bounds(nested, "max"),
        )

    def test_returns_none_when_input_domain_missing(self):
        emu = SimpleNamespace(input_domain=None)
        assert _input_domain_param_bounds(emu, "min") is None
        emu_no_attr = SimpleNamespace()
        assert _input_domain_param_bounds(emu_no_attr, "min") is None

    def test_returns_none_when_parameters_leaf_missing(self):
        # Tree shape present but ``parameters`` leaf absent in either layout.
        domain = {
            "min_tree": {"wavelengths": np.array([3.5])},
            "max_tree": {"wavelengths": np.array([4.2])},
        }
        emu = SimpleNamespace(input_domain=domain)
        assert _input_domain_param_bounds(emu, "min") is None
        # Section-nested but ``inputs`` lacks ``parameters``.
        domain_nested = {
            "min_tree": {"inputs": {"wavelengths": np.array([3.5])}},
            "max_tree": {"inputs": {"wavelengths": np.array([4.2])}},
        }
        emu_nested = SimpleNamespace(input_domain=domain_nested)
        assert _input_domain_param_bounds(emu_nested, "min") is None

    def test_flat_layout_takes_precedence_over_section_nested(self):
        # A pathological tree carrying both layouts simultaneously should
        # resolve via the flat path -- that's the documented priority.
        flat_bounds = np.array([3000.0, -2.0])
        nested_decoy = np.array([0.0, 0.0])
        domain = {
            "min_tree": {
                "parameters": flat_bounds,
                "inputs": {"parameters": nested_decoy},
            },
            "max_tree": {
                "parameters": np.array([12000.0, 0.5]),
                "inputs": {"parameters": np.array([1.0, 1.0])},
            },
        }
        emu = SimpleNamespace(input_domain=domain)
        np.testing.assert_array_equal(
            _input_domain_param_bounds(emu, "min"), flat_bounds
        )
