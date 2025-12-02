import numpy as np
from scipy.stats import skew

# All functions expect nn_intervals in seconds
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

def hti(nn_intervals, bin_size=0.0078125):
    """The total number of NN intervals divided by the height of the NN interval histogram constructed with a fixed bin width."""
    min_val = np.min(nn_intervals)
    max_val = np.max(nn_intervals)
    bins = np.arange(min_val, max_val + bin_size, bin_size)

    counts, _ = np.histogram(nn_intervals, bins=bins)
    return len(nn_intervals) / np.max(counts)

def tinn(nn_intervals, bin_size=0.0078125):
    """The baseline width of the triangle that best approximates the NN interval histogram using least-squares fitting"""
    # Histogram
    hist, bin_edges = np.histogram(nn_intervals, bins='auto')
    x = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Peak
    m_idx = np.argmax(hist)
    Xm = x[m_idx]
    Hm = hist[m_idx]

    best_error = np.inf
    best_N = None
    best_M = None

    # Try all N < Xm and M > Xm combinations
    for i in range(0, m_idx):
        for j in range(m_idx + 1, len(x)):
            N = x[i]
            M = x[j]

            # Build triangular model
            T = np.zeros_like(hist)

            # Increasing line from N to Xm
            left_mask = (x >= N) & (x <= Xm)
            T[left_mask] = Hm * (x[left_mask] - N) / (Xm - N)

            # Decreasing line from Xm to M
            right_mask = (x >= Xm) & (x <= M)
            T[right_mask] = Hm * (M - x[right_mask]) / (M - Xm)

            # Compute error
            error = np.sum((hist - T) ** 2)

            if error < best_error:
                best_error = error
                best_N = N
                best_M = M

    return best_M - best_N

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

# Compute all features and return as a dictionary, units in ms where applicable
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
        'TINN (ms)': tinn(nn_intervals, bin_size=(1/128)) * 1000,
        'CVNN': cvnn(nn_intervals),
        'CVSD': cvsd(nn_intervals),
        'Skewness': skewness(nn_intervals)
    }
