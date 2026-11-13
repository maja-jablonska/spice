"""Differentiable Stromgren-y light curve of the showcase F + K binary (used by s05, s06).

theta = (R1 [R_sun], R2 [R_sun], inclination [deg], Teff2 [K]). The icospheres are
rebuilt inside the function so the radii are traced (no outer jit: the icosphere
needs a concrete vertex count). Spins are omitted -- photometry does not need them.
"""
import numpy as np

import common as C
from s03_eclipse import STARS, P_DAYS, INCL_DEG

THETA0 = np.array([STARS[0]["radius"], STARS[1]["radius"], INCL_DEG, STARS[1]["teff"]])
NAMES = ["R1", "R2", "incl", "Teff2"]
WL_PHOT = (5150.0, 5850.0)          # covers Stromgren y


class LightCurve:
    def __init__(self, emu, n_vert=1500, n_wl=500, chunk_size=1024):
        import jax.numpy as jnp
        from spice.spectrum.filter import Stromgreny
        self.emu, self.n_vert, self.chunk = emu, n_vert, chunk_size
        self.wl, self.lw = C.wl_grid(*WL_PHOT, n_wl)
        self.wl_j, self.lw_j = jnp.asarray(self.wl), jnp.asarray(self.lw)
        self.filt = Stromgreny()
        self.P = P_DAYS / 365.25
        self.names = list(emu.stellar_parameter_names)
        self._nn = None

    def binary(self, theta):
        import jax.numpy as jnp
        from spice.models import IcosphereModel, Binary
        from spice.models.binary import add_orbit
        R1, R2, incl, T2 = theta[0], theta[1], theta[2], theta[3]
        i_t = self.names.index("teff")
        p1 = jnp.asarray(C.stellar_params(self.emu, STARS[0]["teff"], STARS[0]["logg"], 0.0, 1.2))
        p2 = jnp.asarray(C.stellar_params(self.emu, 5000.0, STARS[1]["logg"], 0.0, 1.2)).at[i_t].set(T2)
        b1 = IcosphereModel.construct(self.n_vert, R1, STARS[0]["mass"], p1, self.names)
        b2 = IcosphereModel.construct(self.n_vert, R2, STARS[1]["mass"], p2, self.names)
        if self._nn is None:
            # from_bodies sizes the occlusion neighbour search from the (here traced) radii; size it
            # once at concrete values, with headroom for the radii the fit explores
            c1 = IcosphereModel.construct(self.n_vert, float(THETA0[0]), STARS[0]["mass"], p1, self.names)
            c2 = IcosphereModel.construct(self.n_vert, float(THETA0[1]), STARS[1]["mass"], p1, self.names)  # geometry only
            ref = Binary.from_bodies(c1, c2)
            self._nn = (int(1.5 * ref.n_neighbours1) + 1, int(1.5 * ref.n_neighbours2) + 1)
            print("occlusion neighbours", self._nn, flush=True)
        return add_orbit(Binary.from_bodies(b1, b2, *self._nn), self.P, 0.0, 0.0, jnp.deg2rad(incl), 0.0, 0.0, 0.0,
                         0.0, 0.0, 400)

    def mag(self, theta, phase, binary=None):
        from spice.models.binary import evaluate_orbit
        from spice.spectrum import simulate_observed_flux, AB_passband_luminosity
        b = self.binary(theta) if binary is None else binary
        m1, m2 = evaluate_orbit(b, phase * self.P)
        f = (simulate_observed_flux(self.emu.intensity, m1, self.lw_j, chunk_size=self.chunk, disable_doppler_shift=True)[:, 0]
             + simulate_observed_flux(self.emu.intensity, m2, self.lw_j, chunk_size=self.chunk, disable_doppler_shift=True)[:, 0])
        return AB_passband_luminosity(self.filt, self.wl_j, f)

    def mag_jac(self, theta, phase):
        """(magnitude, d magnitude / d theta) at one phase, by forward mode: one jvp per
        parameter. Reverse mode through the occlusion stores the whole tape and runs out
        of GPU memory on fine meshes; with four parameters forward mode is cheap."""
        import jax
        import jax.numpy as jnp
        th = jnp.asarray(theta)
        f = lambda t: self.mag(t, phase)
        jac, val = [], None
        for k in range(len(th)):
            v, dv = jax.jvp(f, (th,), (jnp.zeros_like(th).at[k].set(1.0),))
            val = v if val is None else val
            jac.append(dv)
        return float(val), np.array([float(x) for x in jac])

    def curve(self, theta, phases):
        import jax.numpy as jnp
        b = self.binary(theta)
        return jnp.stack([self.mag(theta, ph, b) for ph in phases])

    def meshes(self, theta, phase):
        from spice.models.binary import evaluate_orbit
        return evaluate_orbit(self.binary(theta), phase * self.P)


def eclipse_windows(lc: LightCurve, theta, margin=0.15):
    """[(label, phase_start, phase_end)] for both eclipses, labelled by which star is
    hidden (find_binary_eclipses calls both of these 'total', so its kind is no key)."""
    from spice.models import find_binary_eclipses
    from spice.models.binary import evaluate_orbit
    b = lc.binary(theta)
    out = []
    for e in find_binary_eclipses(b):
        m1, m2 = evaluate_orbit(b, e["mid"])
        los = np.asarray(m1.los_vector)
        hot_hidden = float(np.asarray(m2.center) @ los) < float(np.asarray(m1.center) @ los)
        label = "primary eclipse (F star hidden)" if hot_hidden else "secondary eclipse (K star hidden)"
        a, bb = float(e["T1"]) / lc.P, float(e["T4"]) / lc.P
        w = bb - a
        out.append((label, a - margin * w, bb + margin * w))
    assert len({o[0] for o in out}) == len(out), f"eclipse labels not unique: {out}"
    return sorted(out)
