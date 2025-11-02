"""
Poincaré Plot Analysis Library
Provides functions for computing SD1, SD2, and visualizing Poincaré plots
for heart rate variability (HRV) analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def compute_poincare_features(rr_intervals):
    """
    Compute Poincaré plot features (SD1, SD2, SD1/SD2 ratio) from RR intervals.
    
    Parameters:
    -----------
    rr_intervals : array-like
        RR intervals in milliseconds (or any consistent unit)
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'SD1': Standard deviation perpendicular to identity line (short-term variability)
        - 'SD2': Standard deviation along identity line (long-term variability)
        - 'SD1_SD2_ratio': Ratio of SD1 to SD2
        - 'rr_intervals': The input RR intervals as numpy array
    
    Notes:
    ------
    SD1 represents short-term HRV (parasympathetic activity)
    SD2 represents long-term HRV (sympathetic + parasympathetic activity)
    """
    rr_intervals = np.array(rr_intervals)
    
    if len(rr_intervals) < 2:
        return {
            'SD1': None,
            'SD2': None,
            'SD1_SD2_ratio': None,
            'rr_intervals': rr_intervals
        }
    
    # Calculate successive differences
    diffs = np.diff(rr_intervals)
    
    # Standard deviation of differences
    std_diffs = np.std(diffs, ddof=1) if len(diffs) > 1 else np.std(diffs)
    
    # Standard deviation of RR intervals
    std_rr = np.std(rr_intervals, ddof=1) if len(rr_intervals) > 1 else np.std(rr_intervals)
    
    # SD1: short-term variability (dispersion perpendicular to identity line)
    SD1 = np.sqrt(0.5 * (std_diffs ** 2))
    
    # SD2: long-term variability (dispersion along identity line)
    sd2_inner = 2 * (std_rr ** 2) - 0.5 * (std_diffs ** 2)
    SD2 = np.sqrt(sd2_inner) if sd2_inner > 0 else 0.0
    
    # SD1/SD2 Ratio
    SD_ratio = SD1 / SD2 if SD2 != 0 else np.nan
    
    return {
        'SD1': float(SD1),
        'SD2': float(SD2),
        'SD1_SD2_ratio': float(SD_ratio) if not np.isnan(SD_ratio) else None,
        'rr_intervals': rr_intervals
    }


def plot_poincare(rr_intervals, title='Poincaré Plot', show_ellipse=True, figsize=(7, 7), 
                  margin_factor=0.3, use_percentile=True, percentile_range=(2, 98)):
    """
    Create a Poincaré plot with optional SD1/SD2 ellipse.
    
    Parameters:
    -----------
    rr_intervals : array-like
        RR intervals in milliseconds
    title : str, optional
        Title for the plot
    show_ellipse : bool, optional
        Whether to show the SD1/SD2 ellipse (default: True)
    figsize : tuple, optional
        Figure size (width, height) in inches
    margin_factor : float, optional
        Margin around data points as fraction of data range (default: 0.05 = 5%)
    use_percentile : bool, optional
        Use percentile-based axis limits to ignore extreme outliers (default: True)
    percentile_range : tuple, optional
        Percentile range for axis limits (default: (2, 98) = middle 96% of data)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The matplotlib figure object
    ax : matplotlib.axes.Axes
        The matplotlib axes object
    metrics : dict
        Dictionary containing SD1, SD2, and SD1/SD2 ratio
    """
    rr_intervals = np.array(rr_intervals)
    
    if len(rr_intervals) < 2:
        raise ValueError("Need at least 2 RR intervals to create Poincaré plot")
    
    # Compute Poincaré features
    metrics = compute_poincare_features(rr_intervals)
    
    # Prepare scatter data
    x = rr_intervals[:-1]   # RR_n
    y = rr_intervals[1:]    # RR_n+1
    mean_rr = np.mean(rr_intervals)
    
    # Calculate dynamic axis limits
    if use_percentile:
        # Use percentile to exclude extreme outliers
        min_x = np.percentile(x, percentile_range[0])
        max_x = np.percentile(x, percentile_range[1])
        min_y = np.percentile(y, percentile_range[0])
        max_y = np.percentile(y, percentile_range[1])
    else:
        # Use actual min/max
        min_x, max_x = np.min(x), np.max(x)
        min_y, max_y = np.min(y), np.max(y)
    
    # Overall data range
    data_min = min(min_x, min_y)
    data_max = max(max_x, max_y)
    data_range = data_max - data_min
    
    # Add margin around data
    margin = data_range * margin_factor
    axis_min = 300
    axis_max = 1200
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot
    # ax.scatter(x, y, color='blue', alpha=0.6, s=50, label='RRₙ vs RRₙ₊₁', 
    #            zorder=3, edgecolors='navy', linewidths=0.5)
    ax.scatter(x, y, color='blue', alpha=0.6, s=10, label='RRₙ vs RRₙ₊₁')
    
    # Identity line (adjusted to axis limits)
    ax.plot([axis_min, axis_max], [axis_min, axis_max], 'r--', 
            linewidth=2, label='Identity Line (y = x)', zorder=2, alpha=0.7)
    
    # Add SD1/SD2 ellipse if requested and metrics are valid
    if show_ellipse and metrics['SD1'] is not None and metrics['SD2'] is not None:
        SD1 = metrics['SD1']
        SD2 = metrics['SD2']
        
        if SD1 > 0 and SD2 > 0:
            ellipse = Ellipse(
                (mean_rr, mean_rr),      # Center at mean RR
                width=2 * SD2,           # Major axis (long-term variability)
                height=2 * SD1,          # Minor axis (short-term variability)
                angle=45,                # Rotated 45° from x-axis
                edgecolor='darkorange',
                facecolor='none',
                linewidth=3,
                label=f'SD1={SD1:.2f}, SD2={SD2:.2f}',
                zorder=4
            )
            ax.add_patch(ellipse)
    
    # Set axis limits
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    
    # Formatting
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('RRₙ [ms]', fontsize=12)
    ax.set_ylabel('RRₙ₊₁ [ms]', fontsize=12)
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    
    # Add text box with metrics
    if metrics['SD1'] is not None:
        textstr = f"SD1 = {metrics['SD1']:.2f} ms\n"
        textstr += f"SD2 = {metrics['SD2']:.2f} ms\n"
        if metrics['SD1_SD2_ratio'] is not None:
            textstr += f"SD1/SD2 = {metrics['SD1_SD2_ratio']:.3f}"
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.85, 
                    edgecolor='gray', linewidth=1.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props, zorder=5)
    
    plt.tight_layout()
    
    return fig, ax, metrics


def interpret_poincare_ratio(ratio):
    """
    Provide interpretation of SD1/SD2 ratio.
    
    Parameters:
    -----------
    ratio : float
        SD1/SD2 ratio value
    
    Returns:
    --------
    str
        Interpretation string
    """
    if ratio is None or np.isnan(ratio):
        return "Cannot interpret: Invalid ratio"
    
    if ratio < 0.3:
        return "Very low ratio: Possible indication of reduced short-term variability (high sympathetic activity)"
    elif ratio < 0.5:
        return "Low ratio: Suggests sympathetic dominance"
    elif ratio < 1.0:
        return "Moderate ratio: Balanced autonomic activity"
    elif ratio < 1.5:
        return "High ratio: Suggests parasympathetic dominance"
    else:
        return "Very high ratio: Strong parasympathetic activity"


def convert_intervals_to_ms(intervals, unit='seconds'):
    """
    Convert RR intervals to milliseconds if needed.
    
    Parameters:
    -----------
    intervals : array-like
        RR intervals
    unit : str, optional
        Current unit of intervals ('seconds' or 'ms')
    
    Returns:
    --------
    numpy.ndarray
        RR intervals in milliseconds
    """
    intervals = np.array(intervals)
    
    if unit.lower() in ['s', 'sec', 'seconds', 'second']:
        return intervals * 1000.0
    elif unit.lower() in ['ms', 'milliseconds', 'millisecond']:
        return intervals
    else:
        raise ValueError(f"Unknown unit: {unit}. Use 'seconds' or 'ms'")
