"""Named updates use the caller's parameter layout for native meshes."""
import jax.numpy as jnp
import numpy as np
import pytest
from spice.models import IcosphereModel
from spice.models.mesh_transform import update_parameter, update_parameters


@pytest.mark.parametrize('per_face', [False, True])
def test_named_update_matches_indexed_update(per_face):
    model = IcosphereModel.construct(100, 1., 1., [5777., -1.], ['teff', 'm/h'])
    values = jnp.linspace(5000., 6000., len(model.faces)) if per_face else 6000.
    actual = update_parameter(model, 'teff', values, ['teff', 'm/h'])
    expected = update_parameter(model, 0, values)
    np.testing.assert_array_equal(actual.parameters, expected.parameters)
    np.testing.assert_array_equal(actual.parameters[:, 1], -1.)
    np.testing.assert_array_equal(actual.parameters[:, 0], values)


@pytest.mark.parametrize('plural', [False, True])
def test_unknown_parameter_name_has_a_clear_error(plural):
    model = IcosphereModel.construct(100, 1., 1., [5777.], ['teff'])
    with pytest.raises(ValueError, match='missing not found'):
        if plural:
            update_parameters(model, ['missing'], [1.], ['teff'])
        else:
            update_parameter(model, 'missing', 1., ['teff'])
