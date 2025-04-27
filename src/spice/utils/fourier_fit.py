import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

def fit_fourier_series(radius, phases, period=None, n_terms=4, plot=False, m_order=None, n_degree=None):
    """
    Fit a Fourier series to radius variation data and return results compatible with add_pulsation.
    
    Parameters:
    -----------
    radius : array-like
        The radius measurements
    phases : array-like
        The phase values corresponding to each radius measurement
    period : float, optional
        The pulsation period in days. If provided, will be converted to seconds for SPICE.
    n_terms : int, optional
        Number of terms in the Fourier series (default: 4)
    plot : bool, optional
        Whether to display a plot of the original data and the fit (default: False)
    m_order : int, optional
        The order (m) of the spherical harmonics for pulsation
    n_degree : int, optional
        The degree (n) of the spherical harmonics for pulsation
        
    Returns:
    --------
    dict
        A dictionary containing:
        - 'mean': mean radius (a0)
        - 'coefficients': list of [a_i, b_i] coefficients
        - 'amplitudes_phases': numpy array of shape (n_terms, 2) with [amplitude, phase] pairs
        - 'fitted_function': function that can evaluate the fit at any phase
        - 'pulsation_params': dict with parameters ready for add_pulsation:
            - 'm_order': order of spherical harmonics (if provided)
            - 'n_degree': degree of spherical harmonics (if provided)
            - 'period': period in seconds (if provided)
            - 'fourier_series_parameters': numpy array of shape (n_terms, 2)
    """
    # Ensure arrays
    radius = np.array(radius)
    phases = np.array(phases)
    
    # Sort data by phase
    sort_idx = np.argsort(phases)
    phases = phases[sort_idx]
    radius = radius[sort_idx]
    
    # Define the Fourier series function
    def fourier_series(x, a0, *args):
        """
        Fourier series function with a0 as the mean and args as pairs of (a_n, b_n) coefficients.
        """
        n_terms = len(args) // 2
        result = a0
        for i in range(n_terms):
            a_i = args[2*i]
            b_i = args[2*i+1]
            result += a_i * np.cos(2 * np.pi * (i+1) * x) + b_i * np.sin(2 * np.pi * (i+1) * x)
        return result
    
    # Fit the Fourier series to the radius data
    initial_guess = [np.mean(radius)] + [0.0] * (2 * n_terms)
    params, _ = optimize.curve_fit(fourier_series, phases, radius, p0=initial_guess)
    
    # Extract the fitted parameters
    a0 = params[0]
    fourier_coeffs = params[1:]
    
    # Convert to amplitude and phase format
    coefficients = []
    amplitudes_phases = []
    for i in range(n_terms):
        a_i = fourier_coeffs[2*i]
        b_i = fourier_coeffs[2*i+1]
        coefficients.append([a_i, b_i])
        
        amplitude = np.sqrt(a_i**2 + b_i**2)
        phase = np.arctan2(b_i, a_i)
        amplitudes_phases.append([amplitude, phase])
    
    # Convert to numpy array for use with add_pulsation
    amplitudes_phases_array = np.array(amplitudes_phases)
    
    # Create a function for evaluating the fit
    def fitted_function(x):
        return fourier_series(x, *params)
    
    # Visualization if requested
    if plot:
        phase_fine = np.linspace(0, 1, 1000)
        radius_fit = fitted_function(phase_fine)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(phases, radius, color='skyblue', label='Original data', zorder=2)
        plt.plot(phase_fine, radius_fit, color='crimson', linewidth=2, label=f'Fourier fit ({n_terms} terms)', zorder=3)
        plt.xlabel('Phase')
        plt.ylabel('Radius')
        plt.title('Fourier Series Decomposition of Radius Variation')
        plt.legend()
        plt.grid(True, linestyle='--', zorder=1)
        plt.show()
    
    return amplitudes_phases_array