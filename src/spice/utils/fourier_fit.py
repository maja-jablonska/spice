import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

def fit_fourier_series(radius, phases, period=None, n_terms=4, plot=False, m_order=None, n_degree=None):
    """
    Fit a Fourier series to radius variation data and return [amplitude, phase] pairs.

    The fitted model is ``a0 + sum_k a_k cos(2 pi k x) + b_k sin(2 pi k x)`` with
    ``x`` the phase in [0, 1); each term is returned as ``A_k cos(2 pi k x - phi_k)``
    with ``A_k = hypot(a_k, b_k)`` and ``phi_k = arctan2(b_k, a_k)``, the convention
    used by ``add_pulsation``'s Fourier parameters.

    Parameters:
    -----------
    radius : array-like
        The radius measurements.
    phases : array-like
        The phase values (in [0, 1)) corresponding to each radius measurement.
    period : float, optional
        Unused; kept for backward compatibility.
    n_terms : int, optional
        Number of terms in the Fourier series (default: 4)
    plot : bool, optional
        Whether to display a plot of the original data and the fit (default: False)
    m_order, n_degree : int, optional
        Unused; kept for backward compatibility.

    Returns:
    --------
    numpy.ndarray
        Array of shape (n_terms, 2) with [amplitude, phase] per harmonic. The mean
        ``a0`` is not returned. Amplitudes are in the units of ``radius``;
        ``add_pulsation`` expects them as a fraction of the stellar radius, so pass
        ``radius / R`` (or divide the amplitudes by ``R``).
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