import numpy as np
import jax.numpy as jnp

from jax import config

from spice.models import IcosphereModel, Binary, find_binary_eclipses
from spice.models.binary import add_orbit

config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)

LOS = jnp.array([0., 0., -1.])


def _binary(i, ecc=0.0, omega=0.0, T=0.0):
    primary = IcosphereModel.construct(100, 1.0, 1.0, jnp.array([5772.]), ['teff'])
    secondary = IcosphereModel.construct(100, 0.8, 0.8, jnp.array([5772.]), ['teff'])
    binary = Binary.from_bodies(primary, secondary)
    return add_orbit(binary, P=0.01, ecc=ecc, T=T, i=i, omega=omega, Omega=0.0,
                     mean_anomaly=0.0, reference_time=0.0, vgamma=0.0,
                     orbit_resolution_points=50)


class TestFindBinaryEclipses:
    def test_edge_on_circular_two_total_eclipses(self):
        binary = _binary(i=np.pi / 2)
        eclipses = find_binary_eclipses(binary, los_vector=LOS)

        assert len(eclipses) == 2
        assert all(e['kind'] == 'total' for e in eclipses)
        # Conjunctions at quarter phases for this convention
        mids = sorted(e['mid'] for e in eclipses)
        assert np.isclose(mids[0], 0.25 * binary.P, rtol=1e-3)
        assert np.isclose(mids[1], 0.75 * binary.P, rtol=1e-3)
        for e in eclipses:
            assert e['T1'] < e['T2'] < e['mid'] < e['T3'] < e['T4']
            # No event may span the out-of-eclipse gap between conjunctions
            assert e['T4'] - e['T1'] < 0.25 * binary.P

    def test_edge_on_durations_match_chord_geometry(self):
        binary = _binary(i=np.pi / 2)
        eclipse = find_binary_eclipses(binary, los_vector=LOS)[0]

        # Circular orbit: T4-T1 = 2(R1+R2)/v_rel with v_rel = 2*pi*a/P
        import spice.constants as const
        a_m = (const.G_SI * 1.8 * const.SOLAR_MASS_KG * (binary.P * 3.15576e7) ** 2
               / (4 * np.pi ** 2)) ** (1 / 3)
        a_solrad = a_m / const.SOLAR_RAD_M
        expected = 2 * 1.8 / (2 * np.pi * a_solrad) * binary.P
        assert np.isclose(eclipse['T4'] - eclipse['T1'], expected, rtol=1e-2)

    def test_face_on_no_eclipses(self):
        binary = _binary(i=0.0)
        assert find_binary_eclipses(binary, los_vector=LOS) == []

    def test_custom_scan_window(self):
        binary = _binary(i=np.pi / 2)
        eclipses = find_binary_eclipses(
            binary, times=np.linspace(0.5 * binary.P, binary.P, 60), los_vector=LOS)

        assert len(eclipses) == 1
        assert np.isclose(eclipses[0]['mid'], 0.75 * binary.P, rtol=1e-3)

    def test_los_perpendicular_to_orbital_los_sees_nothing(self):
        binary = _binary(i=np.pi / 2)
        assert find_binary_eclipses(binary, los_vector=jnp.array([0., 1., 0.])) == []
