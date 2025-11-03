import numpy as np
from scipy.interpolate import CubicSpline

def find_extrema(signal):
    """
    Find local maxima and minima in the signal.
    
    Parameters:
    -----------
    signal : array-like
        Input signal
        
    Returns:
    --------
    max_indices : array
        Indices of local maxima
    min_indices : array
        Indices of local minima
    """
    max_indices = []
    min_indices = []
    
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            max_indices.append(i)
        elif signal[i] < signal[i-1] and signal[i] < signal[i+1]:
            min_indices.append(i)
    
    return np.array(max_indices), np.array(min_indices)

def get_envelopes(signal):
    """
    Calculate upper and lower envelopes using cubic spline interpolation.
    
    Parameters:
    -----------
    signal : array-like
        Input signal
        
    Returns:
    --------
    upper_env : array
        Upper envelope
    lower_env : array
        Lower envelope
    mean_env : array
        Mean of upper and lower envelopes
    """
    max_indices, min_indices = find_extrema(signal)
    
    # Need at least 2 extrema for interpolation
    if len(max_indices) < 2 or len(min_indices) < 2:
        return None, None, None
    
    # Add boundary points
    t = np.arange(len(signal))
    
    # Extend maxima to boundaries
    max_ext = np.concatenate(([0], max_indices, [len(signal)-1]))
    max_vals = np.concatenate(([signal[0]], signal[max_indices], [signal[-1]]))
    
    # Extend minima to boundaries
    min_ext = np.concatenate(([0], min_indices, [len(signal)-1]))
    min_vals = np.concatenate(([signal[0]], signal[min_indices], [signal[-1]]))
    
    # Create cubic spline interpolations
    try:
        upper_spline = CubicSpline(max_ext, max_vals)
        lower_spline = CubicSpline(min_ext, min_vals)
        
        upper_env = upper_spline(t)
        lower_env = lower_spline(t)
        mean_env = (upper_env + lower_env) / 2.0
        
        return upper_env, lower_env, mean_env
    except:
        return None, None, None

def is_imf(signal, tolerance=0.05):
    """
    Check if a signal satisfies IMF conditions:
    1. Number of extrema and zero crossings differ by at most 1
    2. Mean of envelopes is close to zero
    
    Parameters:
    -----------
    signal : array-like
        Signal to check
    tolerance : float
        Tolerance for mean envelope
        
    Returns:
    --------
    bool : True if signal is an IMF
    """
    max_indices, min_indices = find_extrema(signal)
    
    # Count zero crossings
    zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
    
    # Total number of extrema
    n_extrema = len(max_indices) + len(min_indices)
    
    # Condition 1: extrema and zero crossings differ by at most 1
    if abs(n_extrema - zero_crossings) > 1:
        return False
    
    # Condition 2: mean of envelopes is close to zero
    upper_env, lower_env, mean_env = get_envelopes(signal)
    if mean_env is None:
        return False
    
    mean_amplitude = np.mean(np.abs(signal))
    if mean_amplitude == 0:
        return True
    
    mean_env_normalized = np.mean(np.abs(mean_env)) / mean_amplitude
    
    return mean_env_normalized < tolerance

def sift(signal, max_iterations=100):
    """
    Perform sifting process to extract one IMF.
    
    Parameters:
    -----------
    signal : array-like
        Input signal
    max_iterations : int
        Maximum number of sifting iterations
        
    Returns:
    --------
    imf : array
        Extracted IMF
    """
    h = signal.copy()
    
    for iteration in range(max_iterations):
        upper_env, lower_env, mean_env = get_envelopes(h)
        
        if mean_env is None:
            # Can't continue sifting
            break
        
        h_new = h - mean_env
        
        # Check if we have an IMF
        if is_imf(h_new):
            return h_new
        
        # Check convergence
        sd = np.sum((h - h_new) ** 2) / (np.sum(h ** 2) + 1e-10)
        if sd < 0.3:
            return h_new
        
        h = h_new
    
    return h

def emd(signal, max_imfs=10):
    """
    Perform Empirical Mode Decomposition on a signal.
    
    Parameters:
    -----------
    signal : array-like
        Input signal
    max_imfs : int
        Maximum number of IMFs to extract
        
    Returns:
    --------
    imfs : list of arrays
        List of extracted IMFs
    residue : array
        Final residue
    """
    imfs = []
    residue = signal.copy()
    
    for i in range(max_imfs):
        # Extract IMF
        imf = sift(residue)
        
        # Check if we can extract more IMFs
        max_indices, min_indices = find_extrema(residue)
        if len(max_indices) < 2 or len(min_indices) < 2:
            break
        
        imfs.append(imf)
        residue = residue - imf
        
        # Stop if residue is monotonic or too small
        if len(find_extrema(residue)[0]) < 2:
            break
        
        if np.std(residue) < 0.01 * np.std(signal):
            break
    
    return imfs, residue

def compute_imf_frequency(imf, fs):
    """
    Compute the dominant frequency of an IMF using FFT.
    
    Parameters:
    -----------
    imf : array-like
        IMF signal
    fs : float
        Sampling frequency in Hz
        
    Returns:
    --------
    freq : float
        Dominant frequency in Hz
    """
    # Compute FFT
    N = len(imf)
    fft_result = np.fft.fft(imf)
    
    # Get positive frequencies only
    freqs = np.fft.fftfreq(N, d=1/fs)
    positive_freq_idx = freqs > 0
    freqs_positive = freqs[positive_freq_idx]
    magnitude = np.abs(fft_result[positive_freq_idx])
    
    if len(magnitude) == 0:
        return 0.0
    
    # Find the frequency with maximum magnitude
    peak_idx = np.argmax(magnitude)
    dominant_freq = freqs_positive[peak_idx]
    
    return dominant_freq

def select_respiratory_imf(imfs, fs, freq_range=(0.15, 0.4)):
    """
    Select the IMF that corresponds to respiratory frequency range.
    
    Parameters:
    -----------
    imfs : list of arrays
        List of IMFs from EMD
    fs : float
        Sampling frequency in Hz
    freq_range : tuple
        Frequency range for respiratory signal (default: 0.15-0.4 Hz)
        
    Returns:
    --------
    respiratory_imf : array or None
        IMF in respiratory frequency range
    imf_index : int or None
        Index of selected IMF
    imf_freq : float or None
        Dominant frequency of selected IMF
    """
    min_freq, max_freq = freq_range
    
    for i, imf in enumerate(imfs):
        freq = compute_imf_frequency(imf, fs)
        
        if min_freq <= freq <= max_freq:
            return imf, i, freq
    
    return None, None, None
