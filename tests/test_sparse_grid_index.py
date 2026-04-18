import importlib.util
import os
import sys
import types
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pandas as pd
import jax.numpy as jnp
import jax
import pytest


@lru_cache(maxsize=1)
def _load_module():
    root = Path(__file__).resolve().parents[1]
    emulator_path = root / "src" / "spice" / "spectrum" / "spectrum_emulator.py"
    interp_path = root / "src" / "spice" / "spectrum" / "lazy_zarr_interpolator.py"

    target_module_names = [
        "spice",
        "spice.spectrum",
        "spice.spectrum.spectrum_emulator",
        "spice.spectrum.lazy_zarr_interpolator",
    ]
    missing = object()
    saved_modules = {name: sys.modules.get(name, missing) for name in target_module_names}

    try:
        spice_pkg = types.ModuleType("spice")
        spice_pkg.__path__ = []
        spectrum_pkg = types.ModuleType("spice.spectrum")
        spectrum_pkg.__path__ = []
        spice_pkg.spectrum = spectrum_pkg

        sys.modules["spice"] = spice_pkg
        sys.modules["spice.spectrum"] = spectrum_pkg

        emulator_spec = importlib.util.spec_from_file_location(
            "spice.spectrum.spectrum_emulator", emulator_path
        )
        emulator_module = importlib.util.module_from_spec(emulator_spec)
        sys.modules["spice.spectrum.spectrum_emulator"] = emulator_module
        emulator_spec.loader.exec_module(emulator_module)

        interp_spec = importlib.util.spec_from_file_location(
            "spice.spectrum.lazy_zarr_interpolator", interp_path
        )
        interp_module = importlib.util.module_from_spec(interp_spec)
        sys.modules["spice.spectrum.lazy_zarr_interpolator"] = interp_module
        interp_spec.loader.exec_module(interp_module)
        return interp_module
    finally:
        for name in target_module_names:
            old_module = saved_modules[name]
            if old_module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


_mod = _load_module()
GridIndex = _mod.GridIndex
SparseGridIndex = _mod.SparseGridIndex
_compute_brackets = _mod._compute_brackets
make_corners = _mod.make_corners


def _make_test_df(axis_values_list, columns):
    """Create a test dataframe with all combinations of axis values."""
    from itertools import product as iterproduct

    combos = list(iterproduct(*axis_values_list))
    data = {col: [c[i] for c in combos] for i, col in enumerate(columns)}
    data["row_idx"] = list(range(len(combos)))
    return pd.DataFrame(data)


class TestSparseGridIndex:
    def test_construction_from_dataframe(self):
        columns = ["teff", "logg"]
        df = _make_test_df(
            [[5000.0, 6000.0], [4.0, 4.5, 5.0]],
            columns,
        )
        idx = SparseGridIndex.from_dataframe(df, columns)

        assert idx.d == 2
        assert idx.num_rows == 6
        assert len(idx.keys) == 6
        assert len(idx.values) == 6

    def test_lookup_exact_match(self):
        columns = ["teff", "logg"]
        df = _make_test_df(
            [[5000.0, 6000.0], [4.0, 4.5, 5.0]],
            columns,
        )
        idx = SparseGridIndex.from_dataframe(df, columns)

        query = jnp.array([5000.0, 4.0])
        rows, weights = idx.lookup(query)

        # Should find exactly one row with weight 1.0
        valid = weights > 0
        assert jnp.sum(valid) >= 1
        assert jnp.isclose(jnp.sum(weights), 1.0)

    def test_lookup_interpolation(self):
        columns = ["teff", "logg"]
        df = _make_test_df(
            [[5000.0, 6000.0], [4.0, 5.0]],
            columns,
        )
        idx = SparseGridIndex.from_dataframe(df, columns)

        query = jnp.array([5500.0, 4.5])
        rows, weights = idx.lookup(query)

        # Should interpolate between 4 corners with equal weights
        valid = weights > 0
        assert jnp.sum(valid) == 4
        np.testing.assert_allclose(float(jnp.sum(weights)), 1.0, atol=1e-6)
        # Equal interpolation at midpoint: each corner should have weight 0.25
        np.testing.assert_allclose(
            np.sort(np.asarray(weights[weights > 0])),
            [0.25, 0.25, 0.25, 0.25],
            atol=1e-6,
        )

    def test_equivalence_with_dense_grid_index(self):
        columns = ["teff", "logg", "feh"]
        df = _make_test_df(
            [[5000.0, 5500.0, 6000.0], [4.0, 4.5], [-0.5, 0.0, 0.5]],
            columns,
        )
        dense = GridIndex.from_dataframe(df, columns)
        sparse = SparseGridIndex.from_dataframe(df, columns)

        # Test several query points
        queries = jnp.array([
            [5250.0, 4.25, 0.0],
            [5000.0, 4.0, -0.5],
            [5750.0, 4.3, 0.25],
        ])

        for i in range(queries.shape[0]):
            d_rows, d_weights = dense.lookup(queries[i])
            s_rows, s_weights = sparse.lookup(queries[i])

            # Both should produce the same weighted combination
            # Build dense weight vectors and compare
            d_out = jnp.zeros(dense.num_rows)
            d_valid = d_rows >= 0
            d_out = d_out.at[jnp.where(d_valid, d_rows, 0)].add(
                jnp.where(d_valid, d_weights, 0.0)
            )

            s_out = jnp.zeros(sparse.num_rows)
            s_valid = s_rows >= 0
            s_out = s_out.at[jnp.where(s_valid, s_rows, 0)].add(
                jnp.where(s_valid, s_weights, 0.0)
            )

            np.testing.assert_allclose(
                np.asarray(d_out), np.asarray(s_out), atol=1e-5,
                err_msg=f"Mismatch at query {i}: {queries[i]}",
            )

    def test_batched_lookup(self):
        columns = ["teff", "logg"]
        df = _make_test_df(
            [[5000.0, 6000.0], [4.0, 5.0]],
            columns,
        )
        idx = SparseGridIndex.from_dataframe(df, columns)

        queries = jnp.array([
            [5500.0, 4.5],
            [5000.0, 4.0],
        ])

        rows, weights = idx.batched_lookup(queries)

        assert rows.shape[0] == 2
        assert weights.shape[0] == 2

        # Verify each query individually
        for i in range(2):
            single_rows, single_weights = idx.lookup(queries[i])
            np.testing.assert_allclose(
                np.sort(np.asarray(single_rows)),
                np.sort(np.asarray(rows[i])),
                atol=1e-6,
            )

    def test_missing_corners_sparse_grid(self):
        """Test with a grid that has missing combinations."""
        columns = ["teff", "logg"]
        # Only 3 out of 4 combinations exist
        df = pd.DataFrame({
            "teff": [5000.0, 5000.0, 6000.0],
            "logg": [4.0, 5.0, 4.0],
            "row_idx": [0, 1, 2],
        })
        idx = SparseGridIndex.from_dataframe(df, columns)

        # Query at midpoint - one corner (6000, 5.0) is missing
        query = jnp.array([5500.0, 4.5])
        rows, weights = idx.lookup(query)

        # Should still produce valid output (normalized over existing corners)
        total = float(jnp.sum(weights))
        np.testing.assert_allclose(total, 1.0, atol=1e-6)

        # The missing corner should have weight 0
        n_valid = int(jnp.sum(weights > 0))
        assert n_valid == 3

    def test_high_dimensional_grid(self):
        """Test that SparseGridIndex works with many parameters where GridIndex would fail."""
        d = 8
        columns = [f"param_{i}" for i in range(d)]
        # 3 values per axis: 3^8 = 6561 total entries (dense grid would be fine,
        # but this verifies the sparse version works for higher d)
        axis_values = [[0.0, 0.5, 1.0] for _ in range(d)]
        df = _make_test_df(axis_values, columns)

        idx = SparseGridIndex.from_dataframe(df, columns)
        assert idx.d == 8
        assert idx.num_rows == 3**8

        # Query at midpoint
        query = jnp.array([0.25] * d)
        rows, weights = idx.lookup(query)

        # 2^8 = 256 corners
        assert rows.shape[0] == 256
        assert weights.shape[0] == 256
        np.testing.assert_allclose(float(jnp.sum(weights)), 1.0, atol=1e-5)

    def test_jit_compatible(self):
        columns = ["teff", "logg"]
        df = _make_test_df(
            [[5000.0, 6000.0], [4.0, 5.0]],
            columns,
        )
        idx = SparseGridIndex.from_dataframe(df, columns)

        query = jnp.array([5500.0, 4.5])

        # lookup is already jitted via @partial(jax.jit, static_argnums=(0,))
        rows1, weights1 = idx.lookup(query)
        rows2, weights2 = idx.lookup(query)

        np.testing.assert_array_equal(np.asarray(rows1), np.asarray(rows2))
        np.testing.assert_array_equal(np.asarray(weights1), np.asarray(weights2))
