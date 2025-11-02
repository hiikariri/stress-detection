import numpy as np
import pandas as pd
import streamlit as st
from scipy.interpolate import interp1d
import lib.bmepy as bmepy
import math

# --- Fungsi Manual dari Notebook Pengguna ---
def _hamming(N):
    """Membuat window Hamming dengan panjang N."""
    if N == 1:
        return np.array([1.0])
    n = np.arange(N)
    return 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))

# --- Fungsi Interpolasi (Robust - Tidak Berubah) ---

def _interpolate_rr(rr_intervals_seconds, fs=4):
    """
    Menginterpolasi interval RR (dalam detik) yang tidak merata ke frekuensi sampling yang teratur.
    Mengembalikan sinyal terinterpolasi dalam MILIDETIK (ms) untuk perhitungan PSD.
    """
    num_intervals = len(rr_intervals_seconds)

    if num_intervals < 2:
        raise ValueError(f"Sinyal PI intervals terlalu pendek ({num_intervals} sampel). Butuh minimal 2.")

    interp_kind = 'cubic' if num_intervals >= 4 else 'linear'

    rr_intervals_ms = rr_intervals_seconds * 1000
    rr_times_seconds = np.cumsum(rr_intervals_seconds)
    rr_times_seconds = rr_times_seconds - rr_times_seconds[0]
    
    max_time = rr_times_seconds[-1]
    if max_time <= 0:
        raise ValueError("Durasi total sinyal PI intervals adalah nol.")
        
    interp_times = np.arange(0, max_time, 1/fs)
    
    f = interp1d(rr_times_seconds, rr_intervals_ms, kind=interp_kind, fill_value="extrapolate")
    interp_rr_signal_ms = f(interp_times)
    
    return interp_rr_signal_ms, fs


def _calculate_welch_manual_scipy_scaled(signal_ms, fs_rr):
    """
    Menghitung PSD menggunakan metode Welch manual,
    tetapi dengan SKALA dan KOREKSI yang sama dengan `scipy.signal.welch`.
    """
    window_size = 256
    overlap = 128
    pad_to = 256
    
    if len(signal_ms) < window_size:
        st.warning(f"Sinyal terinterpolasi ({len(signal_ms)} sampel) lebih pendek dari segmen window ({window_size}). Menggunakan window yang lebih kecil.")
        window_size = len(signal_ms)
        if overlap >= window_size:
            overlap = window_size // 2
        if window_size == 0:
             return np.array([0]), np.array([0]) 

    step = window_size - overlap
    segments_psd = []
    
    window = _hamming(window_size)
    
    scale = 1.0 / (fs_rr * (window**2).sum())
    
    for start in range(0, len(signal_ms) - window_size + 1, step):
        segment = signal_ms[start : start + window_size]
        
        # 2. Terapkan window
        windowed = segment * window
        
        if pad_to > window_size:
            padded = np.concatenate([windowed, np.zeros(pad_to - window_size)])
        else:
            padded = windowed
        
        fft_result = bmepy.fft(padded) 
        
        fft_half = np.array(fft_result[:(pad_to // 2) + 1])
        
        psd = (np.abs(fft_half)**2)
 
        psd *= scale
        
        if pad_to % 2: # Jika nfft ganjil
            psd[1:] *= 2
        else: # Jika nfft genap
            psd[1:-1] *= 2
        
        segments_psd.append(psd)
    
    if not segments_psd:
        st.error("Tidak ada segmen yang bisa diproses. Sinyal mungkin terlalu pendek.")
        return np.array([0]), np.array([0])

    
    psd_avg = np.mean(segments_psd, axis=0)
    freqs = np.linspace(0, fs_rr / 2, len(psd_avg))
    
    return freqs, psd_avg

def _get_band_power_trapz(freqs, psd, f_low, f_high):
    """
    Menghitung power band menggunakan integrasi trapezoid (np.trapz).
    Ini adalah cara yang benar untuk menghitung power (ms²) dari density (ms²/Hz).
    """
    indices = np.where((freqs >= f_low) & (freqs < f_high))[0]
    if len(indices) < 2:
        return 0
    

    return np.trapz(psd[indices], freqs[indices])



@st.cache_data
def compute_frequency_domain_features(pi_intervals, fs_interp=4):
    """
    Menghitung semua fitur domain frekuensi dari PI intervals (diharapkan dalam detik).
    """
    
    try:

        rr_signal_ms, fs_rr = _interpolate_rr(pi_intervals, fs=fs_interp)
    except ValueError as e:
        st.warning(f"Gagal melakukan interpolasi sinyal: {e}")
        return None, None, {}


    rr_signal_detrended = rr_signal_ms - np.mean(rr_signal_ms)
    freqs, psd = _calculate_welch_manual_scipy_scaled(rr_signal_detrended, fs_rr)

    
    if freqs is None or len(freqs) <= 1:
        st.warning("Gagal menghitung PSD, hasil frekuensi kosong.")
        return None, None, {}

    vlf_band = (0.003, 0.04)
    lf_band = (0.04, 0.15)
    hf_band = (0.15, 0.4)


    vlf_power = _get_band_power_trapz(freqs, psd, *vlf_band)
    lf_power = _get_band_power_trapz(freqs, psd, *lf_band)
    hf_power = _get_band_power_trapz(freqs, psd, *hf_band)
    
    total_power = vlf_power + lf_power + hf_power
    
    lf_plus_hf = lf_power + hf_power
    
    lf_nu = (lf_power / lf_plus_hf) * 100 if lf_plus_hf > 0 else 0
    hf_nu = (hf_power / lf_plus_hf) * 100 if lf_plus_hf > 0 else 0
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else np.inf

    def get_peak_freq(f_low, f_high):
        idx_band = np.where((freqs >= f_low) & (freqs < f_high))[0]
        if len(idx_band) == 0:
            return None
        try:
            peak_idx = idx_band[np.argmax(psd[idx_band])]
            return freqs[peak_idx]
        except ValueError:
            return None

    lf_peak_freq = get_peak_freq(*lf_band)
    hf_peak_freq = get_peak_freq(*hf_band)

    features = {
        'Total Power (TP) (ms²)': total_power,
        'VLF Power (ms²)': vlf_power,
        'LF Power (ms²)': lf_power,
        'HF Power (ms²)': hf_power,
        'LF/HF Ratio': lf_hf_ratio,
        'LF (n.u.)': lf_nu,
        'HF (n.u.)': hf_nu,
        'LF Peak Frequency (Hz)': lf_peak_freq,
        'HF Peak Frequency (Hz)': hf_peak_freq
    }
    
    return freqs, psd, features

