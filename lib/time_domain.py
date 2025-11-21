import numpy as np
from scipy.stats import skew

def sdnn(nn_intervals):
    """Standard deviation of NN intervals"""
    return np.std(nn_intervals, ddof=1)

def sdann(nn_intervals, segment_length=300):
    """Standard deviation of the averages of NN intervals in 5-min segments (default 300s)"""
    n_segments = int(np.floor(len(nn_intervals) / segment_length))
    if n_segments < 1:
        return np.nan
    segment_means = [np.mean(nn_intervals[i*segment_length:(i+1)*segment_length]) for i in range(n_segments)]
    return np.std(segment_means, ddof=1)

def sdnn_index(nn_intervals, segment_length=300):
    """Mean of the standard deviations of NN intervals in 5-min segments"""
    n_segments = int(np.floor(len(nn_intervals) / segment_length))
    if n_segments < 1:
        return np.nan
    segment_stds = [np.std(nn_intervals[i*segment_length:(i+1)*segment_length], ddof=1) for i in range(n_segments)]
    return np.mean(segment_stds)

def rmssd(nn_intervals):
    """Root mean square of successive differences"""
    diff = np.diff(nn_intervals)
    return np.sqrt(np.mean(diff**2))

def sdsd(nn_intervals):
    """Standard deviation of successive differences"""
    diff = np.diff(nn_intervals)
    return np.std(diff, ddof=1)

def nn50(nn_intervals, threshold=0.05):
    """Number of pairs of successive NN intervals differing by more than 50 ms (default 0.05s)"""
    diff = np.abs(np.diff(nn_intervals))
    return np.sum(diff > threshold)

def pnn50(nn_intervals, threshold=0.05):
    """Proportion of NN50 divided by total number of successive differences"""
    n_diffs = len(nn_intervals) - 1
    if n_diffs == 0:
        return 0.0
    return nn50(nn_intervals, threshold) / n_diffs * 100

def mean_hr(nn_intervals):
    """Mean heart rate (bpm) from NN intervals (seconds)"""
    mean_nn = np.mean(nn_intervals)
    if mean_nn == 0:
        return np.nan
    return 60.0 / mean_nn

def hti(nn_intervals, bin_size=0.01):
    """HRV Triangular Index: total number of NN intervals divided by the height of the histogram of NN intervals"""
    counts, _ = np.histogram(nn_intervals, bins=np.arange(np.min(nn_intervals), np.max(nn_intervals)+bin_size, bin_size))
    return len(nn_intervals) / np.max(counts)

def tinn(nn_intervals, bin_size=0.01):
    """TINN: Triangular interpolation of NN interval histogram (HRV Task Force standard)"""
    # Create histogram
    counts, bin_edges = np.histogram(nn_intervals, bins=np.arange(np.min(nn_intervals), np.max(nn_intervals)+bin_size, bin_size))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Find mode (peak)
    max_idx = np.argmax(counts)
    mode = bin_centers[max_idx]
    max_count = counts[max_idx]
    
    # Find left intercept (N): where line from peak to left edge crosses X-axis
    # Linear interpolation from peak going left
    left_idx = 0
    for i in range(max_idx, 0, -1):
        if counts[i] == 0 or i == 0:
            left_idx = i
            break
    
    # Find right intercept (M): where line from peak to right edge crosses X-axis
    right_idx = len(counts) - 1
    for i in range(max_idx, len(counts)):
        if counts[i] == 0 or i == len(counts) - 1:
            right_idx = i
            break
    
    # TINN = M - N (baseline width of the triangle)
    tinn_value = bin_centers[right_idx] - bin_centers[left_idx]
    
    return tinn_value

def cvnn(nn_intervals):
    """Coefficient of variation of NN intervals"""
    return np.std(nn_intervals, ddof=1) / np.mean(nn_intervals)

def cvsd(nn_intervals):
    """Coefficient of variation of successive differences: RMSSD / MeanNN"""
    mean_nn = np.mean(nn_intervals)
    if mean_nn == 0:
        return np.nan
    return rmssd(nn_intervals) / mean_nn

def skewness(nn_intervals):
    """Skewness of NN interval distribution"""
    return skew(nn_intervals)

# Convenience function to compute all features
def compute_time_domain_features(nn_intervals):
    return {
        'SDNN (ms)': sdnn(nn_intervals) * 1000,  # Convert to ms
        'SDANN (ms)': sdann(nn_intervals) * 1000,
        'SDNN index (ms)': sdnn_index(nn_intervals) * 1000,
        'RMSSD (ms)': rmssd(nn_intervals) * 1000,
        'SDSD (ms)': sdsd(nn_intervals) * 1000,
        'NN50': nn50(nn_intervals),
        'pNN50 (%)': pnn50(nn_intervals),
        'HR (bpm)': mean_hr(nn_intervals),
        'HTI': hti(nn_intervals, bin_size=(1/128)),
        'TINN (ms)': tinn(nn_intervals) * 1000,
        'CVNN': cvnn(nn_intervals),
        'CVSD': cvsd(nn_intervals),
        'Skewness': skewness(nn_intervals)
    }
