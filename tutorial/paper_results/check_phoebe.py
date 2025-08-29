import argparse
import phoebe
import numpy as np
import matplotlib.pyplot as plt
from phoebe.parameters.dataset import _mesh_columns
import jax.numpy as jnp
from transformer_payne import Blackbody
from spice.models.binary import PhoebeBinary
from spice.models.phoebe_utils import PhoebeConfig

from spice.models.mesh_model import IcosphereModel
from spice.models.binary import Binary, add_orbit, evaluate_orbit_at_times
from spice.models.mesh_view import get_mesh_view

def main(inclination, period, q, ecc, times, primary_mass):
    """
    Compares PHOEBE and SPICE velocities and positions for a binary system.

    Args:
        inclination (float): Inclination of the binary system in degrees.
        period (float): Period of the binary system in days.
    """
    print(f"Running comparison for inclination={inclination} deg, period={period} days")

    # Time array
    times = np.linspace(0, period, times)

    # Create PHOEBE bundle
    b = phoebe.default_binary()

    COLUMNS = _mesh_columns

    # Set parameters from command line arguments
    b.add_dataset('mesh', times=times, columns=COLUMNS, dataset='mesh01')
    b.add_dataset('orb', compute_times=times, dataset='orb01')
    b.set_value('period@binary@component', period)
    b.set_value('q@binary@component', q)
    b.set_value('ecc@binary@component', ecc)
    b.flip_constraint('mass@primary', solve_for='sma')
    b.set_value('mass@primary@component', primary_mass)
    b.set_value_all('incl@binary', inclination)
    print(f"PHOEBE parameters set: inclination={b['incl@binary']}, period={b['period@binary']}, q={b['q@binary']}, ecc={b['ecc@binary']}")
    b.run_compute(irrad_method='none', coordinates='uvw', ltte=False)

    def default_icosphere(mass=1.):
        return get_mesh_view(IcosphereModel.construct(100, 1., mass, [5700, 0.], ['teff', 'abun']), jnp.array([0., 1., 0.]))

    times_yr = times * 0.0027378507871321013

    body1 = default_icosphere(b.get_parameter('mass@primary@component').value)
    body2 = default_icosphere(b.get_parameter('mass@secondary@component').value)
    binary = Binary.from_bodies(body1, body2)

    binary = add_orbit(binary, P = b.get_parameter('period@binary@component').value*0.0027378507871321013,
                       ecc = b.get_parameter('ecc@binary@component').value, T = 0.,
                        i = jnp.deg2rad(b.get_parameter('incl@binary@component').value),
                        omega = b.get_parameter('per0@binary@component').value*0.017453292519943295,
                        Omega = b.get_parameter('long_an@binary@component').value*0.0027378507871321013,
                        vgamma = b.get_parameter('vgamma').value,
                        reference_time = b.get_parameter('t0_ref@binary@component').value*0.0027378507871321013,
                        mean_anomaly = b.get_parameter('mean_anom@binary@component').value*0.017453292519943295,
                        orbit_resolution_points = len(times_yr))

    print(f"Evaluating orbit at times: {times_yr}")
    pb1, pb2 = evaluate_orbit_at_times(binary, times_yr)
    
    bb = Blackbody()

    p1 = PhoebeConfig(b, 'mesh01', 'orb01')
    pb = PhoebeBinary.construct(p1, bb.parameter_names, parameter_values={pn: sp for pn, sp in zip(bb.parameter_names, bb.solar_parameters)})

    # Get SPICE velocities
    spice_vels1 = np.array([_pb1.orbital_velocity for _pb1 in pb1])
    spice_vels2 = np.array([_pb2.orbital_velocity for _pb2 in pb2])
    
    phoebe_vels1 = pb.body1_velocities
    phoebe_vels2 = pb.body2_velocities

    phoebe_pos1 = pb.body1_centers
    phoebe_pos2 = pb.body2_centers

    # Get SPICE positions
    spice_pos1 = [_pb1.center for _pb1 in pb1]
    spice_pos2 = [_pb2.center for _pb2 in pb2]

    filename = f'phoebe_binaries_check/position_velocity_offsets_incl_{inclination}_period_{period}_q_{q}_ecc_{ecc}_primary_mass_{primary_mass}.npz'
    print(f"Saving position and velocity offsets to {filename}")
    np.savez(filename,
             phoebe_pos1=phoebe_pos1,
             phoebe_pos2=phoebe_pos2,
             phoebe_vels1=phoebe_vels1,
             phoebe_vels2=phoebe_vels2,
             spice_pos1=spice_pos1,
             spice_pos2=spice_pos2,
             spice_vels1=spice_vels1,
             spice_vels2=spice_vels2,
             times=times,
             sma=b.get_parameter('sma@binary@component').value,
             primary_mass=primary_mass,
             secondary_mass=b.get_parameter('mass@secondary@component').value,
             inclination=inclination,
             period=period,
             q=q,
             ecc=ecc)
    print(f"Position and velocity offsets saved to {filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare PHOEBE and SPICE binary models.')
    parser.add_argument('inclination', type=float, help='Inclination in degrees')
    parser.add_argument('period', type=float, help='Period in days')    
    parser.add_argument('--times', type=int, default=10, help='Number of time points to evaluate')
    parser.add_argument('--q', type=float, default=1.0, help='Mass ratio')
    parser.add_argument('--ecc', type=float, default=0.0, help='Eccentricity')
    parser.add_argument('--primary_mass', type=float, default=1.0, help='Primary mass')
    args = parser.parse_args()

    main(args.inclination, args.period, args.q, args.ecc, args.times, args.primary_mass)
