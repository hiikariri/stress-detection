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
    """Proportion of NN50 divided by total number of intervals"""
    return nn50(nn_intervals, threshold) / len(nn_intervals) * 100

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
    """TINN: Triangular interpolation of NN interval histogram (approximate)"""
    counts, bin_edges = np.histogram(nn_intervals, bins=np.arange(np.min(nn_intervals), np.max(nn_intervals)+bin_size, bin_size))
    max_idx = np.argmax(counts)
    left = max_idx
    right = max_idx
    while left > 0 and counts[left] > 0:
        left -= 1
    while right < len(counts)-1 and counts[right] > 0:
        right += 1
    return bin_edges[right] - bin_edges[left]

def cvnn(nn_intervals):
    """Coefficient of variation of NN intervals"""
    return np.std(nn_intervals, ddof=1) / np.mean(nn_intervals)

def cvsd(nn_intervals):
    """Coefficient of variation of successive differences"""
    diff = np.diff(nn_intervals)
    return np.std(diff, ddof=1) / np.mean(nn_intervals)

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
        'HTI': hti(nn_intervals),
        'TINN (ms)': tinn(nn_intervals) * 1000,
        'CVNN': cvnn(nn_intervals),
        'CVSD': cvsd(nn_intervals),
        'Skewness': skewness(nn_intervals)
    }
