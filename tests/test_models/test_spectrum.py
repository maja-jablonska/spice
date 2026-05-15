import pytest
import jax.numpy as jnp
from spice.spectrum.spectrum import (__flux_flash_sum, __spectrum_flash_sum, simulate_observed_flux, simulate_monochromatic_luminosity,
                                   luminosity, AB_passband_luminosity, ST_passband_luminosity,
                                   absolute_bol_luminosity)
from spice.spectrum.filter import Filter
from .utils import default_icosphere
import chex


class TestSpectrumFunctions:
    def test_simulate_observed_flux(self):
        """Test simulation of observed flux"""
        mesh = default_icosphere()
        wavelengths = jnp.logspace(3, 4, 100)  # 1000-10000 Angstroms
        log_wavelengths = jnp.log10(wavelengths)
        chunk_size = int(mesh.parameters.shape[0])
        wavelengths_chunk_size = int(log_wavelengths.shape[0])
        
        # Mock intensity function that returns constant values
        def mock_intensity(wavelengths, mu, params):
            return jnp.ones((wavelengths.shape[0], 2))
        
        flux = simulate_observed_flux(
            mock_intensity,
            mesh,
            log_wavelengths,
            chunk_size=chunk_size,
            wavelengths_chunk_size=wavelengths_chunk_size
        )
        
        assert flux.shape == (100, 2)  # Check output shape
        assert jnp.all(jnp.isfinite(flux))  # Check for non-finite values
        assert jnp.all(flux >= 0)  # Flux should be non-negative

    def test_simulate_observed_flux_matches_wavelength_chunking(self):
        """Test that wavelength chunking preserves spectral ordering and values."""
        mesh = default_icosphere()
        log_wavelengths = jnp.linspace(3.0, 4.0, 37)
        chunk_size = int(mesh.parameters.shape[0])

        def mock_intensity(wavelengths, mu, params):
            return jnp.stack([wavelengths, wavelengths + 1.0], axis=-1)

        flux_unchunked = simulate_observed_flux(
            mock_intensity,
            mesh,
            log_wavelengths,
            chunk_size=chunk_size,
            wavelengths_chunk_size=int(log_wavelengths.shape[0])
        )
        flux_chunked = simulate_observed_flux(
            mock_intensity,
            mesh,
            log_wavelengths,
            chunk_size=chunk_size,
            wavelengths_chunk_size=8
        )

        chex.assert_shape(flux_chunked, (37, 2))
        assert jnp.allclose(flux_chunked, flux_unchunked)

    def test_simulate_monochromatic_luminosity(self):
        """Test simulation of monochromatic luminosity"""
        mesh = default_icosphere()
        wavelengths = jnp.logspace(3, 4, 100)
        log_wavelengths = jnp.log10(wavelengths)
        chunk_size = int(mesh.parameters.shape[0])
        wavelengths_chunk_size = int(log_wavelengths.shape[0])
        
        # Mock flux function
        def mock_flux(wavelengths, params):
            return jnp.ones((wavelengths.shape[0], 2))
        
        lum = simulate_monochromatic_luminosity(
            mock_flux,
            mesh,
            log_wavelengths,
            chunk_size=chunk_size,
            wavelengths_chunk_size=wavelengths_chunk_size
        )
        
        assert lum.shape == (100, 2)
        assert jnp.all(jnp.isfinite(lum))
        assert jnp.all(lum >= 0)

    def test_luminosity(self):
        """Test bolometric luminosity calculation"""
        mesh = default_icosphere()
        wavelengths = jnp.logspace(3, 4, 100)
        chunk_size = int(mesh.parameters.shape[0])
        wavelengths_chunk_size = int(wavelengths.shape[0])
        
        # Mock flux function
        def mock_flux(wavelengths, params):
            return jnp.ones((wavelengths.shape[0], 2))
        
        bol_lum = luminosity(
            mock_flux,
            mesh,
            wavelengths,
            chunk_size=chunk_size,
            wavelengths_chunk_size=wavelengths_chunk_size
        )
        
        assert jnp.isscalar(bol_lum)
        assert jnp.isfinite(bol_lum)
        assert bol_lum > 0

    def test_passband_luminosities(self):
        """Test AB and ST magnitude calculations"""
        wavelengths = jnp.logspace(3, 4, 100)
        observed_flux = jnp.ones((100, 2))  # Mock constant flux
        
        # Create mock filter
        filter_wavelengths = jnp.linspace(1000, 10000, 50)
        filter_responses = jnp.ones_like(filter_wavelengths)
        mock_filter = Filter(name="test_filter", transmission_curve=jnp.stack((filter_wavelengths, filter_responses)))
        
        # Test AB magnitude
        ab_mag = AB_passband_luminosity(mock_filter, wavelengths, observed_flux[:, 0])
        assert jnp.isscalar(ab_mag)
        assert jnp.isfinite(ab_mag)
        
        # # Test ST magnitude
        st_mag = ST_passband_luminosity(mock_filter, wavelengths, observed_flux[:, 0])
        assert jnp.isscalar(st_mag)
        assert jnp.isfinite(st_mag)

    def test_absolute_bol_luminosity(self):
        """Test absolute bolometric magnitude calculation"""
        test_luminosity = 3.828e33  # Solar luminosity in erg/s
        abs_mag = absolute_bol_luminosity(test_luminosity)
        
        assert jnp.isscalar(abs_mag)
        assert jnp.isfinite(abs_mag)
        # Solar absolute bolometric magnitude should be close to 4.75
        assert jnp.abs(abs_mag - 4.75) < 0.1

    def test_doppler_shift(self):
        """Test Doppler shift in flux calculation"""
        mesh = default_icosphere()
        wavelengths = jnp.logspace(3, 4, 100)
        log_wavelengths = jnp.log10(wavelengths)
        chunk_size = int(mesh.parameters.shape[0])
        wavelengths_chunk_size = int(log_wavelengths.shape[0])
        
        def mock_intensity(wavelengths, mu, params):
            return jnp.ones((wavelengths.shape[0], 2))
        
        # Compare with and without Doppler shift
        flux_with_doppler = simulate_observed_flux(
            mock_intensity,
            mesh,
            log_wavelengths,
            chunk_size=chunk_size,
            wavelengths_chunk_size=wavelengths_chunk_size,
            disable_doppler_shift=False
        )
        flux_without_doppler = simulate_observed_flux(
            mock_intensity,
            mesh,
            log_wavelengths,
            chunk_size=chunk_size,
            wavelengths_chunk_size=wavelengths_chunk_size,
            disable_doppler_shift=True
        )
        
        # Fluxes should be different if mesh has non-zero velocities
        if jnp.any(mesh.los_velocities != 0):
            assert not jnp.allclose(flux_with_doppler, flux_without_doppler)

            # Test dimensions of flux arrays
            chex.assert_shape(flux_with_doppler, (100, 2))
            chex.assert_shape(flux_without_doppler, (100, 2))

            # Test dimensions of intermediate arrays
            test_wavelengths = jnp.logspace(3, 4, 50)
            test_log_wavelengths = jnp.log10(test_wavelengths)
            test_areas = jnp.ones((42,))
            test_vrads = jnp.zeros((42,))
            test_params = jnp.ones((42, 3))

            # Test __spectrum_flash_sum output dimensions
            flash_sum = __spectrum_flash_sum(mock_intensity,
                                           test_log_wavelengths,
                                           test_areas,
                                           jnp.ones_like(test_areas),
                                           test_vrads,
                                           test_params,
                                           chunk_size=16)
            chex.assert_shape(flash_sum, (50, 2))

            # Test __flux_flash_sum output dimensions 
            flux_sum = __flux_flash_sum(mock_intensity,
                                      test_log_wavelengths,
                                      test_areas,
                                      test_vrads,
                                      test_params,
                                      chunk_size=16)
            chex.assert_shape(flux_sum, (50, 2))

            # Test simulate_monochromatic_luminosity output dimensions
            test_mesh = default_icosphere()
            mono_lum = simulate_monochromatic_luminosity(mock_intensity,
                                                       test_mesh,
                                                       test_log_wavelengths,
                                                       chunk_size=int(test_mesh.parameters.shape[0]),
                                                       wavelengths_chunk_size=int(test_log_wavelengths.shape[0]))
            chex.assert_shape(mono_lum, (50, 2))

    def test_simulate_observed_flux_scales_as_r_squared_over_d_squared(self):
        """Observed flux must scale as (R / d)^2.

        Regression for the spurious ``m.radius**2`` factor that scaled the
        output as R^4 instead of R^2 (fixed in the commit that introduced
        this test). Also defends against any future R^N-or-d^M scaling
        regression: comparing the *ratio* of two configs at matching mesh
        resolution removes the icosphere discretisation error, leaving only
        a small per-radius float artifact from the ``volume_scale``
        correction in ``IcosphereModel.construct`` (~3e-7 between R=1 and
        R=5). Tolerance is set to ``1e-6`` to absorb that while still being
        7+ orders of magnitude tighter than what any R^N!=2 regression
        would produce (the original bug gave R=5 ratio of 625 vs 25).

        The previous tests in this module all use ``default_icosphere()`` at
        ``R = 1`` where the buggy R^2 prefactor was identically 1, which is
        why the original bug was invisible to CI.
        """
        log_wavelengths = jnp.log10(jnp.logspace(3, 4, 32))

        def mock_intensity(wavelengths, mu, params):
            return jnp.ones((wavelengths.shape[0], 2))

        # --- Radius scaling at fixed distance ---
        flux_by_r = {}
        for radius in [1.0, 2.0, 5.0]:
            mesh = default_icosphere(radius=radius)
            flux_by_r[radius] = simulate_observed_flux(
                mock_intensity, mesh, log_wavelengths, distance=10.0,
                disable_doppler_shift=True,
            )

        baseline_r = flux_by_r[1.0]
        assert jnp.all(baseline_r > 0), "baseline flux at R=1 must be positive"
        for radius, flux in flux_by_r.items():
            ratio = flux / baseline_r
            expected = radius ** 2
            assert jnp.allclose(ratio, expected, rtol=1e-6), (
                f"R={radius} R_sun: observed-flux ratio vs R=1 was "
                f"{float(jnp.mean(ratio)):.6e}, expected R^2={expected}"
            )

        # --- Distance scaling at fixed radius. Same mesh across all d, so
        # the volume_scale artifact cancels exactly and we can tighten this
        # one to round-off precision.
        mesh = default_icosphere(radius=1.0)
        flux_by_d = {}
        for distance in [10.0, 50.0, 100.0]:
            flux_by_d[distance] = simulate_observed_flux(
                mock_intensity, mesh, log_wavelengths, distance=distance,
                disable_doppler_shift=True,
            )

        baseline_d = flux_by_d[10.0]
        for distance, flux in flux_by_d.items():
            ratio = flux / baseline_d
            expected = (10.0 / distance) ** 2
            assert jnp.allclose(ratio, expected, rtol=1e-12), (
                f"d={distance} pc: observed-flux ratio vs d=10 was "
                f"{float(jnp.mean(ratio)):.6e}, expected (10/d)^2={expected}"
            )

    def test_simulate_monochromatic_luminosity_scales_as_r_squared(self):
        """Monochromatic luminosity must scale as R^2.

        Defensive regression for the parallel ``m.radius**2`` prefactor in
        ``simulate_monochromatic_luminosity``: that factor is *correct*
        because the integrand uses ``m.areas`` (unit-sphere normalised),
        unlike ``simulate_observed_flux`` which uses ``m.visible_cast_areas``
        (already in R_sun^2). If a future change normalises ``m.areas`` to
        physical units without dropping the prefactor, this test will catch
        the resulting R^4 scaling.
        """
        log_wavelengths = jnp.log10(jnp.logspace(3, 4, 32))

        def mock_flux(wavelengths, params):
            return jnp.ones((wavelengths.shape[0], 2))

        lum_by_r = {}
        for radius in [1.0, 2.0, 5.0]:
            mesh = default_icosphere(radius=radius)
            lum_by_r[radius] = simulate_monochromatic_luminosity(
                mock_flux, mesh, log_wavelengths, disable_doppler_shift=True,
            )

        baseline = lum_by_r[1.0]
        assert jnp.all(baseline > 0), "baseline luminosity at R=1 must be positive"
        for radius, lum in lum_by_r.items():
            ratio = lum / baseline
            expected = radius ** 2
            # Slightly looser tolerance than the observed-flux test: m.areas
            # carries a per-radius `volume_scale` correction in
            # IcosphereModel.construct that varies at the ~1e-7 level across
            # radii, unlike cast_areas which scales exactly as R^2.
            assert jnp.allclose(ratio, expected, rtol=1e-6), (
                f"R={radius} R_sun: monochromatic luminosity ratio vs R=1 "
                f"was {float(jnp.mean(ratio)):.6e}, expected R^2={expected}"
            )
