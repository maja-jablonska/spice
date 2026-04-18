#!/usr/bin/env python3

def test_spectrum_functionality():
    """Test spectrum-related functionality"""
    try:
        # Test importing spectrum-related classes
        try:
            from spice.spectrum.spectrum import (
                simulate_observed_flux,
                simulate_monochromatic_luminosity,
                luminosity,
                AB_passband_luminosity,
                absolute_bol_luminosity
            )
            from spice.spectrum.utils import (
                ERG_S_TO_W,
                SPHERE_STERADIAN,
                ZERO_POINT_LUM_W,
                apply_spectral_resolution
            )
            from spice.spectrum.spectrum_emulator import SpectrumEmulator
            from spice.spectrum.atlas9_interpolator import ATLAS9Interpolator
            print("✓ Successfully imported spectrum-related classes")
            return True
        except ImportError as e:
            print(f"✗ Failed to import spectrum-related classes: {e}")
            print("  Make sure stellar-spice is installed: pip install stellar-spice")
            return False
    except Exception as e:
        print(f"✗ Unexpected error testing spectrum functionality: {e}")
        return False

def test_basic_functionality():
    """Test basic stellar-spice functionality that doesn't require PHOEBE"""
    try:
        # First test if we can import numpy
        import numpy as np
        print("✓ Successfully imported numpy")
        
        # Test importing JAX and related packages
        try:
            import jax
            import jax.numpy as jnp
            from jax.typing import ArrayLike
            from jaxtyping import Array, Float
            print("✓ Successfully imported JAX and related packages")
        except ImportError as e:
            print(f"✗ Failed to import JAX: {e}")
            print("  Make sure JAX is installed: pip install jax jaxlib")
            return False
            
        # Test importing basic model classes
        try:
            from spice.models.model import Model
            from spice.models.mesh_model import MeshModel, IcosphereModel
            from spice.models.mesh_generation import icosphere
            from spice.models.utils import lat_to_theta, lon_to_phi, theta_to_lat, phi_to_lon
            from spice.models.mesh_view_kdtree import resolve_occlusion, get_optimal_search_radius
            print("✓ Successfully imported basic model classes")
        except ImportError as e:
            print(f"✗ Failed to import basic model classes: {e}")
            print("  Make sure stellar-spice is installed: pip install stellar-spice")
            return False
            
        # Test importing geometry utilities
        try:
            from spice.geometry.utils import polygon_area
            from spice.geometry.sutherland_hodgman import clip
            print("✓ Successfully imported geometry utilities")
        except ImportError as e:
            print(f"✗ Failed to import geometry utilities: {e}")
            print("  Make sure stellar-spice is installed: pip install stellar-spice")
            return False
            
        # Test importing warning utilities
        try:
            from spice.utils.warnings import ExperimentalWarning, JAXWarning
            print("✓ Successfully imported warning utilities")
        except ImportError as e:
            print(f"✗ Failed to import warning utilities: {e}")
            print("  Make sure stellar-spice is installed: pip install stellar-spice")
            return False
        
        # Test creating a simple mesh model
        try:
            model = IcosphereModel.construct(
                n_vertices=42,  # Small icosphere for testing
                radius=1.0,
                mass=1.0,
                parameters=5000.0,  # Temperature
                parameter_names=['teff']
            )
            print("✓ Successfully created a basic mesh model")
            return True
        except Exception as e:
            print(f"✗ Failed to create mesh model: {e}")
            return False
            
    except ImportError as e:
        print(f"✗ Failed to import numpy: {e}")
        print("  Make sure numpy is installed: pip install numpy")
        return False

def test_phoebe_functionality():
    """Test PHOEBE-specific functionality"""
    try:
        # First check if we can import the PHOEBE availability flag
        try:
            from spice.models.phoebe_utils import PHOEBE_AVAILABLE
            print(f"\nPHOEBE_AVAILABLE: {PHOEBE_AVAILABLE}")
        except ImportError as e:
            print(f"✗ Failed to import PHOEBE utilities: {e}")
            print("  Make sure stellar-spice is installed: pip install stellar-spice")
            return False
        
        if not PHOEBE_AVAILABLE:
            print("✗ PHOEBE is not installed")
            print("  To install PHOEBE support, run: pip install stellar-spice[phoebe]")
            return False
            
        print("✓ PHOEBE is installed and available")
        
        # Try to create a simple PHOEBE bundle
        try:
            import phoebe
            b = phoebe.default_binary()
            print("✓ Successfully created a PHOEBE bundle")
            return True
        except Exception as e:
            print(f"✗ Failed to create PHOEBE bundle: {e}")
            return False
            
    except ImportError as e:
        print(f"✗ Error importing PHOEBE: {e}")
        print("  To install PHOEBE support, run: pip install stellar-spice[phoebe]")
        return False

def main():
    """Run all tests"""
    print("Testing basic stellar-spice functionality...")
    basic_ok = test_basic_functionality()
    
    print("\nTesting spectrum functionality...")
    spectrum_ok = test_spectrum_functionality()
    
    print("\nTesting PHOEBE functionality...")
    phoebe_ok = test_phoebe_functionality()
    
    print("\nTest Summary:")
    print(f"Basic functionality: {'✓' if basic_ok else '✗'}")
    print(f"Spectrum functionality: {'✓' if spectrum_ok else '✗'}")
    print(f"PHOEBE functionality: {'✓' if phoebe_ok else '✗'}")
    
    if not basic_ok:
        print("\nError: Basic functionality failed. Please check your stellar-spice installation.")
        print("Try running: pip install stellar-spice")
    elif not spectrum_ok:
        print("\nNote: Basic functionality works, but spectrum functionality failed.")
        print("Check your stellar-spice installation for spectrum-related dependencies.")
    elif not phoebe_ok:
        print("\nNote: Basic and spectrum functionality work, but PHOEBE support is not available.")
        print("To enable PHOEBE support, run: pip install stellar-spice[phoebe]")
    else:
        print("\nAll functionality is working correctly!")

if __name__ == "__main__":
    main() 