"""
Poincaré Plot Analysis Library
Modified to accept external SDNN/RMSSD values passed from the main code.
With Density Coloring and SD1/SD2 Axis Visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import gaussian_kde 

def compute_poincare_from_stats(sdnn, rmssd):
    """
    Hitung SD1 dan SD2 menggunakan nilai SDNN dan RMSSD yang sudah ada.
    Rumus hubungan:
    SD1^2 = 0.5 * RMSSD^2
    SD2^2 = 2 * SDNN^2 - SD1^2
    """
    if sdnn is None or rmssd is None:
        return None, None, None
        
    # SD1
    sd1 = np.sqrt(0.5 * (rmssd ** 2))
    
    # SD2
    sd2_sq = 2 * (sdnn ** 2) - (sd1 ** 2)
    sd2 = np.sqrt(sd2_sq) if sd2_sq > 0 else 0
    
    ratio = sd1 / sd2 if sd2 != 0 else None # Fixed division by zero check
    
    return sd1, sd2, ratio

def compute_poincare_features(rr_intervals):
    """
    Hitung manual dari raw data (cadangan jika SDNN/RMSSD tidak diberikan).
    """
    rr_intervals = np.array(rr_intervals)
    if len(rr_intervals) < 2:
        return {'SD1': None, 'SD2': None, 'SD1_SD2_ratio': None}
    
    diffs = np.diff(rr_intervals)
    std_diffs = np.std(diffs, ddof=1) 
    std_rr = np.std(rr_intervals, ddof=1) 
    
    SD1 = np.sqrt(0.5 * (std_diffs ** 2))
    val_sd2 = 2 * (std_rr ** 2) - 0.5 * (std_diffs ** 2)
    SD2 = np.sqrt(val_sd2) if val_sd2 > 0 else 0.0
    SD_ratio = SD1 / SD2 if SD2 != 0 else np.nan
    
    return {'SD1': SD1, 'SD2': SD2, 'SD1_SD2_ratio': SD_ratio}

def plot_poincare(rr_intervals, title='Poincaré Plot', show_ellipse=True, figsize=(7, 7), 
                  margin_factor=0.1, use_percentile=True, percentile_range=(1, 99),
                  external_sdnn=None, external_rmssd=None): 
    """
    Membuat plot Poincaré dengan Density Color dan Sumbu SD1/SD2.
    """
    rr_intervals = np.array(rr_intervals)
    
    if len(rr_intervals) < 2:
        raise ValueError("Butuh minimal 2 interval RR")

    # --- LOGIKA PEMILIHAN DATA ---
    if external_sdnn is not None and external_rmssd is not None:
        SD1, SD2, ratio = compute_poincare_from_stats(external_sdnn, external_rmssd)
        metrics = {'SD1': SD1, 'SD2': SD2, 'SD1_SD2_ratio': ratio}
    else:
        metrics = compute_poincare_features(rr_intervals)
        SD1 = metrics['SD1']
        SD2 = metrics['SD2']

    # --- VISUALISASI ---
    x = rr_intervals[:-1]
    y = rr_intervals[1:]
    mean_rr = np.mean(rr_intervals)
    
    # Scaling otomatis (Saya kembalikan ke dinamis agar tidak 'kaku' / kosong)
    if use_percentile:
        min_val = np.percentile(rr_intervals, percentile_range[0])
        max_val = np.percentile(rr_intervals, percentile_range[1])
    else:
        min_val = np.min(rr_intervals)
        max_val = np.max(rr_intervals)
        
    if max_val == min_val: 
        max_val += 50
        min_val -= 50
    
    data_range = max_val - min_val
    margin = max(data_range * margin_factor, 20)
    
    # Gunakan limit dinamis agar zoom pas ke data (0-1000 sering kejauhan)
    plot_min = 0 #min_val - margin
    plot_max = 1000 #max_val + margin
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # --- BAGIAN WARNA WARNI (Density Plot) ---
    try:
        # Hitung density
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        
        # Urutkan agar titik paling merah ada di atas
        idx = z.argsort()
        x_plot, y_plot, z_plot = x[idx], y[idx], z[idx]
        
        # Plot dengan colormap 'turbo' (seperti rainbow)
        scatter = ax.scatter(x_plot, y_plot, c=z_plot, s=25, cmap='turbo', 
                             edgecolors='none', alpha=0.8)
        
        # Tambahkan colorbar kecil
        plt.colorbar(scatter, ax=ax, label='Density')
        
    except Exception:
        # Fallback jika data terlalu sedikit/error
        ax.scatter(x, y, color='#1f77b4', alpha=0.6, s=25, label='RRₙ vs RRₙ₊₁')

    # Garis diagonal
    ax.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', linewidth=1, alpha=0.3, zorder=1)
    
    # --- BAGIAN ELIPS & SUMBU ---
    if show_ellipse and metrics['SD1'] is not None and metrics['SD2'] is not None:
        if SD1 > 0 and SD2 > 0:
            # 1. Gambar Elips Luar
            ellipse = Ellipse(
                (mean_rr, mean_rr),
                width=2 * SD2,
                height=2 * SD1,
                angle=45,
                edgecolor='darkorange',
                facecolor='none',
                linewidth=2.5,
                label=f'SD1={SD1:.2f}, SD2={SD2:.2f}',
                zorder=4
            )
            ax.add_patch(ellipse)

            # 2. Gambar Sumbu SD2 (Garis Merah Panjang - Variabilitas Jangka Panjang)
            # Proyeksi sudut 45 derajat
            dx2 = SD2 * 0.7071 # cos(45)
            dy2 = SD2 * 0.7071 # sin(45)
            ax.plot([mean_rr - dx2, mean_rr + dx2], 
                    [mean_rr - dy2, mean_rr + dy2], 
                    color='red', linewidth=2, zorder=5, label='SD2 Axis')

            # 3. Gambar Sumbu SD1 (Garis Hijau Pendek - Variabilitas Jangka Pendek)
            # Proyeksi sudut 135 derajat (-cos, sin)
            dx1 = SD1 * 0.7071 
            dy1 = SD1 * 0.7071
            ax.plot([mean_rr + dx1, mean_rr - dx1], 
                    [mean_rr - dy1, mean_rr + dy1], 
                    color='green', linewidth=2, zorder=5, label='SD1 Axis')
    
    # Formatting Axis
    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('RRₙ [ms]', fontsize=12)
    ax.set_ylabel('RRₙ₊₁ [ms]', fontsize=12)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.legend(loc='upper left', fontsize=9)
    
    # Text Box Info
    if metrics['SD1'] is not None:
        textstr = '\n'.join((
            f"SD1 = {metrics['SD1']:.2f} ms",
            f"SD2 = {metrics['SD2']:.2f} ms",
            f"Ratio = {metrics['SD1_SD2_ratio']:.3f}"
        ))
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='lightgray')
        ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    return fig, ax, metrics

def interpret_poincare_ratio(ratio):
    if ratio is None or np.isnan(ratio): return "Invalid ratio"
    if ratio < 0.3: return "Very low ratio (High Sympathetic)"
    elif ratio < 0.5: return "Low ratio (Sympathetic Dominance)"
    elif ratio < 1.0: return "Moderate ratio (Balanced)"
    elif ratio < 1.5: return "High ratio (Parasympathetic Dominance)"
    else: return "Very high ratio (Strong Parasympathetic)"

def convert_intervals_to_ms(intervals, unit='seconds'):
    intervals = np.array(intervals)
    if unit.lower() in ['s', 'sec', 'seconds']: return intervals * 1000.0
    return intervals
