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
        
        # Mock intensity function that returns constant values
        def mock_intensity(wavelengths, mu, params):
            return jnp.ones((wavelengths.shape[0], 2))
        
        flux = simulate_observed_flux(mock_intensity, mesh, log_wavelengths)
        
        assert flux.shape == (100, 2)  # Check output shape
        assert jnp.all(jnp.isfinite(flux))  # Check for non-finite values
        assert jnp.all(flux >= 0)  # Flux should be non-negative

    def test_simulate_monochromatic_luminosity(self):
        """Test simulation of monochromatic luminosity"""
        mesh = default_icosphere()
        wavelengths = jnp.logspace(3, 4, 100)
        log_wavelengths = jnp.log10(wavelengths)
        
        # Mock flux function
        def mock_flux(wavelengths, params):
            return jnp.ones((wavelengths.shape[0], 2))
        
        lum = simulate_monochromatic_luminosity(mock_flux, mesh, log_wavelengths)
        
        assert lum.shape == (100, 2)
        assert jnp.all(jnp.isfinite(lum))
        assert jnp.all(lum >= 0)

    def test_luminosity(self):
        """Test bolometric luminosity calculation"""
        mesh = default_icosphere()
        wavelengths = jnp.logspace(3, 4, 100)
        
        # Mock flux function
        def mock_flux(wavelengths, params):
            return jnp.ones((wavelengths.shape[0], 2))
        
        bol_lum = luminosity(mock_flux, mesh, wavelengths)
        
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
        mock_filter = Filter(jnp.array([filter_wavelengths, filter_responses]))
        
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
        
        def mock_intensity(wavelengths, mu, params):
            return jnp.ones((wavelengths.shape[0], 2))
        
        # Compare with and without Doppler shift
        flux_with_doppler = simulate_observed_flux(mock_intensity, mesh, log_wavelengths, 
                                                disable_doppler_shift=False)
        flux_without_doppler = simulate_observed_flux(mock_intensity, mesh, log_wavelengths,
                                                    disable_doppler_shift=True)
        
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
                                                       test_log_wavelengths)
            chex.assert_shape(mono_lum, (50, 2))
