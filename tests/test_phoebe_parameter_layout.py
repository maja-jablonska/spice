"""Each mesh element must get its OWN parameter vector.

``PhoebeModel.construct`` collects one array per parameter -- shape
``(n_parameters, n_elements)`` -- and must transpose it into
``(n_elements, n_parameters)`` rows, the layout ``simulate_observed_flux``
passes to the emulator.

It used to call ``np.array(params).reshape((n_elements, -1))``. Reshape does
not transpose: it reinterprets the flat buffer, so row 0 came out as the first
``n_parameters`` *teff* values instead of element 0's
``(teff, logg, feh, ...)``. Every element was then emulated at meaningless
parameters -- a mixture of temperatures, gravities and abundances -- while
still producing a plausible-looking spectrum. The single-parameter case is
unaffected (both layouts coincide), which is how it survived.
"""

import numpy as np
import pytest

from spice.models.phoebe_model import _stack_per_element


def test_multi_parameter_rows_are_per_element():
    n = 5
    teff = np.linspace(4900.0, 4950.0, n)
    logg = np.linspace(2.90, 2.95, n)
    feh = np.full(n, -0.30)
    out = _stack_per_element([teff, logg, feh], n)

    assert out.shape == (n, 3)
    for i in range(n):
        # Element i's row must be exactly its own three values.
        assert out[i, 0] == pytest.approx(teff[i])
        assert out[i, 1] == pytest.approx(logg[i])
        assert out[i, 2] == pytest.approx(feh[i])


def test_reshape_layout_would_scramble():
    """Pin the specific corruption, so the old formula cannot come back."""
    n = 5
    teff = np.linspace(4900.0, 4950.0, n)
    logg = np.linspace(2.90, 2.95, n)
    feh = np.full(n, -0.30)

    correct = _stack_per_element([teff, logg, feh], n)
    scrambled = np.array([teff, logg, feh]).reshape((n, -1))

    assert not np.allclose(correct, scrambled)
    # The old layout put three temperatures in row 0 instead of teff/logg/feh.
    assert scrambled[0, 1] > 4000.0
    assert correct[0, 1] == pytest.approx(logg[0])


def test_single_parameter_is_a_column():
    n = 4
    teff = np.linspace(4900.0, 4950.0, n)
    out = _stack_per_element(teff, n)
    assert out.shape == (n, 1)
    assert np.allclose(out[:, 0], teff)


def test_single_parameter_list_is_also_a_column():
    n = 4
    teff = np.linspace(4900.0, 4950.0, n)
    out = _stack_per_element([teff], n)
    assert out.shape == (n, 1)
    assert np.allclose(out[:, 0], teff)
