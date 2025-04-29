import os
import tempfile
from typing import Optional, List
from pathlib import Path

from huggingface_hub import hf_hub_download
from spice.spectrum.user_spectrum_interpolator import UserSpectrumInterpolator

class ATLAS9Interpolator(UserSpectrumInterpolator):
    """
    A specialized spectrum interpolator for ATLAS9 model atmospheres.
    
    This class extends UserSpectrumInterpolator with specific functionality
    for ATLAS9 model grids, including automatic download from Hugging Face.
    
    The grid includes parameters:
    - teff: Effective temperature (4000K to 9000K)
    - logg: Surface gravity (0.0 to 5.0)
    - m/h: Metallicity (-1.0 to 0.5)
    - mu: Angle parameter (0.0 to 1.0)
    """
    
    def __init__(self, 
                 grid_path: Optional[str] = None,
                 use_cache: bool = True,
                 cache_dir: Optional[str] = None,
                 parameter_names: Optional[List[str]] = None):
        """
        Initialize the ATLAS9 interpolator.
        
        Args:
            grid_path: Path to the grid file. If None, downloads from Hugging Face.
            use_cache: Whether to use cached files when downloading from Hugging Face.
            cache_dir: Directory to cache downloaded files. If None, uses the default.
            parameter_names: Custom parameter names (default: ["teff", "logg", "m/h"])
        """
        
        warnings.warn(" ⚠️ The ATLAS9 interpolator is highly experimental and not fully validated. Use with caution!")
        
        # Define default parameter names for ATLAS9 models
        self._default_parameter_names = parameter_names or ["teff", "logg", "m/h"]
        
        # If grid_path is not provided, download from Hugging Face
        if grid_path is None:
            grid_path = self._download_atlas9_grid(use_cache=use_cache, cache_dir=cache_dir)
        
        # Initialize the parent class
        super().__init__(
            grid_path=grid_path,
            parameter_names=self._default_parameter_names,
            has_continuum=True,
            has_mu=True
        )
        
        # Print grid parameter ranges
        self._print_parameter_ranges()
    
    def _download_atlas9_grid(self, use_cache: bool = True, cache_dir: Optional[str] = None) -> str:
        """
        Download the ATLAS9 grid from Hugging Face if not already cached.
        
        Args:
            use_cache: Whether to use cached files
            cache_dir: Directory to cache downloaded files
            
        Returns:
            Path to the downloaded grid file
        """
        try:
            filename = "atlas9_models.h5"
            repo_id = "mjablonska/atlas9_interpolator"
            
            # Use the provided cache_dir or the default
            if cache_dir is None:
                # Use a common cache directory that persists between sessions
                cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "atlas9_models")
                os.makedirs(cache_dir, exist_ok=True)
            
            # Check if the file already exists in the cache
            cached_file = os.path.join(cache_dir, filename)
            if os.path.exists(cached_file) and use_cache:
                print(f"Using cached ATLAS9 grid from {cached_file}")
                return cached_file
            
            print(f"Downloading ATLAS9 grid from Hugging Face ({repo_id})...")
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir if use_cache else None,
                force_download=not use_cache
            )
            
            print(f"Downloaded ATLAS9 grid to {downloaded_path}")
            return downloaded_path
            
        except Exception as e:
            error_msg = f"Failed to download ATLAS9 grid: {str(e)}"
            raise RuntimeError(error_msg)
    
    def _print_parameter_ranges(self):
        """Print the parameter ranges available in the grid."""
        # Extract parameter ranges
        teff_range = (float(self.parameters[:, 0].min()), float(self.parameters[:, 0].max()))
        logg_range = (float(self.parameters[:, 1].min()), float(self.parameters[:, 1].max()))
        mh_range = (float(self.parameters[:, 2].min()), float(self.parameters[:, 2].max()))
        mu_values = sorted(list(set(float(mu) for mu in self.parameters[:, 3])))
        
        print("Parameter ranges in the grid:")
        print(f"Temperature (Teff): {teff_range[0]} - {teff_range[1]} K")
        print(f"Surface gravity (log g): {logg_range[0]} - {logg_range[1]}")
        print(f"Metallicity [M/H]: {mh_range[0]} - {mh_range[1]}")
        print(f"Mu values: {mu_values}")
