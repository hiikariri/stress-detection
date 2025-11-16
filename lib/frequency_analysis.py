import numpy as np
import pandas as pd
import streamlit as st
from scipy.interpolate import interp1d
import lib.bmepy as bmepy

def _hamming(N):
    if N == 1:
        return np.array([1.0])
    n = np.arange(N)
    return 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))


def _interpolate_rr(rr_intervals_seconds, fs=4):
    num_intervals = len(rr_intervals_seconds)

    if num_intervals < 2:
        raise ValueError(f"PI intervals terlalu pendek ({num_intervals}). Butuh minimal 2.")

    interp_kind = 'cubic' if num_intervals >= 4 else 'linear'

    rr_intervals = np.array(rr_intervals_seconds)
    rr_times_seconds = np.cumsum(rr_intervals)
    rr_times_seconds = rr_times_seconds - rr_times_seconds[0]

    max_time = rr_times_seconds[-1]
    if max_time <= 0:
        raise ValueError("Durasi total PI intervals adalah nol.")

    interp_times = np.arange(0, max_time, 1 / fs)

    f = interp1d(rr_times_seconds, rr_intervals, kind=interp_kind, fill_value="extrapolate")
    interp_rr_signal = f(interp_times)

    return interp_rr_signal, fs


def _calculate_welch_manual_scipy_scaled(signal_ms, fs_rr):
    x = np.asarray(signal_ms)
    if x.size == 0:
        return np.array([]), np.array([])

    nperseg = 256
    noverlap = 128
    nfft = 256

    if x.size < nperseg:
        nperseg = x.size
        noverlap = max(0, nperseg // 2)

    step = nperseg - noverlap
    if step <= 0:
        step = 1

    win = _hamming(nperseg)
    U = (win ** 2).sum()

    segments_psd = []

    for start in range(0, x.size - nperseg + 1, step):
        seg = x[start:start + nperseg].copy()
        seg = seg - np.mean(seg)
        seg_win = seg * win

    
        if seg_win.size < nfft:
            seg_padded = np.zeros(nfft, dtype=seg_win.dtype)
            seg_padded[:seg_win.size] = seg_win
        else:
            seg_padded = seg_win[:nfft]

        fft_full = bmepy.fft(seg_padded)
        fft_res = np.asarray(fft_full[: (nfft // 2) + 1 ])
        psd_seg = (np.abs(fft_res) ** 2) / (fs_rr * U)

        # One-sided correction: double bins except DC and (if present) Nyquist
        if nfft % 2 == 0:
            if psd_seg.size > 1:
                psd_seg[1:-1] *= 2
        else:
            if psd_seg.size > 1:
                psd_seg[1:] *= 2

        segments_psd.append(psd_seg)

    if len(segments_psd) == 0:
        seg = x.copy()
        seg = seg - np.mean(seg)
        win_short = _hamming(seg.size)
        U_short = (win_short ** 2).sum()
        seg_win = seg * win_short
        # Fallback: compute full FFT via bmepy and take positive frequencies
        if seg_win.size < nfft:
            seg_padded = np.zeros(nfft, dtype=seg_win.dtype)
            seg_padded[:seg_win.size] = seg_win
        else:
            seg_padded = seg_win[:nfft]

        fft_full = bmepy.fft(seg_padded)
        fft_res = np.asarray(fft_full[: (nfft // 2) + 1 ])
        psd_seg = (np.abs(fft_res) ** 2) / (fs_rr * U_short)
        if nfft % 2 == 0:
            if psd_seg.size > 1:
                psd_seg[1:-1] *= 2
        else:
            if psd_seg.size > 1:
                psd_seg[1:] *= 2
        segments_psd.append(psd_seg)

    psd_avg = np.mean(np.vstack(segments_psd), axis=0)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs_rr)

    return freqs, psd_avg


def _get_band_power_trapz(freqs, psd, f_low, f_high):
    indices = np.where((freqs >= f_low) & (freqs < f_high))[0]
    if len(indices) < 2:
        return 0

    return np.trapz(psd[indices], freqs[indices])


@st.cache_data
def compute_frequency_domain_features(pi_intervals, fs_interp=4):
    try:
        rr_signal, fs_rr = _interpolate_rr(pi_intervals, fs=fs_interp)
    except ValueError as e:
        st.warning(f"Gagal interpolasi sinyal: {e}")
        return None, None, {}

    rr_signal_detrended = rr_signal - np.mean(rr_signal)
    freqs, psd = _calculate_welch_manual_scipy_scaled(rr_signal_detrended, fs_rr)

    # Convert PSD from s^2/Hz to ms^2/Hz
    psd = psd * 1e6

    if freqs is None or len(freqs) <= 1:
        st.warning("Gagal menghitung PSD: frekuensi kosong.")
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
        'Total Power (TP) (ms²)': total_power * 10e-6,
        'VLF Power (ms²)': vlf_power * 10e-6,
        'LF Power (ms²)': lf_power * 10e-6,
        'HF Power (ms²)': hf_power * 10e-6,
        'LF/HF Ratio': lf_hf_ratio,
        'LF (n.u.)': lf_nu,
        'HF (n.u.)': hf_nu,
        'LF Peak Frequency (Hz)': lf_peak_freq,
        'HF Peak Frequency (Hz)': hf_peak_freq
    }

    return freqs, psd, features

