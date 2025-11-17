# Ring spot scaling guide

This note shows how to tune `RingSpotConfig` so that the umbral core and the plage ring grow or shrink in a predictable way before you call [`add_ring_spot`](../src/spice/models/spots.py) or the lower-level helpers in `spice.utils.ring_spot`.

## 1. Geometry knobs inside `RingSpotConfig`

| Parameter | Effect | How to scale |
| --- | --- | --- |
| `sigma_umb_deg` | Gaussian width (1σ) of the cool core around the spot axis. | Multiply by a constant factor to make the core wider or narrower. A doubling makes the e-folding radius twice as large; halve it for a tighter core. |
| `theta0_deg` | Polar angle that places the bright ring away from the axis. | Larger values push the ring closer to the stellar limb, smaller values keep it near the pole. |
| `sigma_plage_deg` | Gaussian width (1σ) of the plage ring. | Controls the ring thickness; scale it just like `sigma_umb_deg`. |
| `deltaT_umb` / `deltaT_plage` | Temperature contrast between the perturbed and base map. | Keep these proportional to maintain the same fractional contrast after resizing. |
| `dCa_umb` / `dCa_plage` and `A_plage` / `B_umb` | Abundance or spectral depth adjustments. | Use the same scale factor as the temperature contrasts if you want the Ca response to remain visually consistent with the temperature map. |

Because the weights are Gaussian, the radius that encloses ~76% of the umbral power is `r ≈ sqrt(2) * sigma` (measured in the same units you use for `sigma`). If you prefer to specify a full-width at half-maximum (FWHM), convert it to `sigma` via `sigma = fwhm / 2.355` before writing the config.

## 2. Scaling recipe

1. **Start from the base config**: `cfg = RingSpotConfig()`.
2. **Decide on a size multiplier** `s`. For example, `s = 0.5` for a smaller spot or `s = 1.8` for a much larger ring.
3. **Apply the multiplier** to every angular width in degrees:
   ```python
   from dataclasses import replace
   scaled = replace(
       cfg,
       sigma_umb_deg=cfg.sigma_umb_deg * s,
       theta0_deg=cfg.theta0_deg * s,
       sigma_plage_deg=cfg.sigma_plage_deg * s,
   )
   ```
4. **Optionally scale the contrasts** so that the larger ring does not overshoot the base field:
   ```python
   scaled = replace(
       scaled,
       deltaT_umb=cfg.deltaT_umb * s,
       deltaT_plage=cfg.deltaT_plage * s,
       dCa_plage=cfg.dCa_plage * s,
       A_plage=cfg.A_plage * s,
   )
   ```
   Scaling the contrasts is not required—skip this block if you prefer to keep the original amplitudes.
5. **Send the config to `add_ring_spot`** and continue with your pipeline.

## 3. Example presets

```python
from dataclasses import replace
from spice.models.spots import RingSpotConfig, add_ring_spot

compact_core = replace(
    RingSpotConfig(),
    sigma_umb_deg=10.0,
    theta0_deg=30.0,
    sigma_plage_deg=4.0,
    deltaT_umb=800.0,
    deltaT_plage=100.0,
)

wide_ring = replace(
    RingSpotConfig(),
    sigma_umb_deg=28.0,
    theta0_deg=65.0,
    sigma_plage_deg=15.0,
    deltaT_umb=1500.0,
    deltaT_plage=250.0,
    A_plage=1.0,
)
```

Apply either preset to a mesh column (parameter index `0` is commonly the effective temperature):

```python
mesh = IcosphereModel.construct(
    ntri=1000,
    radius=1.0,
    mass=1.0,
    parameters=bb.to_parameters(dict(teff=5700)),
    parameter_names=bb.parameter_names,
)

compact_mesh = add_ring_spot(mesh, param_index=0, config=compact_core)
wide_mesh = add_ring_spot(mesh, param_index=0, config=wide_ring)
```

## 4. Quick diagnostics

You can verify that the scaling behaves as expected by measuring the angular radius that contains most of the umbral weight:

```python
import jax.numpy as jnp
from spice.utils.ring_spot import ring_spot_weights

n_hat = mesh.d_centers / jnp.linalg.norm(mesh.d_centers, axis=1, keepdims=True)
spot_axis = jnp.array([0.0, 0.0, 1.0])

w_umb_small, w_plage_small = ring_spot_weights(n_hat, spot_axis, compact_core)
w_umb_wide, w_plage_wide = ring_spot_weights(n_hat, spot_axis, wide_ring)

print("compact core effective radius", jnp.sqrt(2.0) * compact_core.sigma_umb_deg)
print("wide core effective radius", jnp.sqrt(2.0) * wide_ring.sigma_umb_deg)
```

The printed radii should follow the scaling multiplier you chose in step 2. Repeat the same measurement for `sigma_plage_deg` if you want to compare the plage thickness.

## 5. Putting it all together

1. Choose a multiplier `s` and update the config using section 2.
2. Optionally dial in custom contrasts like the presets shown above.
3. Apply `add_ring_spot` on your mesh (temperature column plus optional Ca column) to bake the perturbation into the model parameters.
4. Forward the spotted mesh to your SPICE emulator, or pull the weights directly through `ring_spot_weights` for bespoke workflows.

Following these steps lets you quickly generate both compact polar rings and extended equatorial belts without rewriting your spot machinery—just rescale the configuration parameters and reuse the same mesh transform.
