import numpy as np
import streamlit as st
from scipy.interpolate import interp1d
from scipy.signal import welch

def _interpolate_rr(rr_intervals_seconds, fs=4):
    """
    Menginterpolasi interval RR (dalam detik) yang tidak merata ke frekuensi sampling yang teratur.
    Mengembalikan sinyal terinterpolasi dalam MILIDETIK (ms) untuk perhitungan PSD.
    """
    # Ubah ke milidetik
    rr_intervals_ms = rr_intervals_seconds * 1000
    
    # Buat vektor waktu (dalam detik) untuk interval RR asli yang tidak merata
    rr_times_seconds = np.cumsum(rr_intervals_seconds)
    rr_times_seconds = rr_times_seconds - rr_times_seconds[0] # Mulai dari 0
    
    # Buat vektor waktu baru yang disampel secara merata
    max_time = rr_times_seconds[-1]
    interp_times = np.arange(0, max_time, 1/fs)
    
    # Interpolasi nilai RR (dalam ms) pada titik waktu baru
    f = interp1d(rr_times_seconds, rr_intervals_ms, kind='cubic', fill_value="extrapolate")
    interp_rr_signal_ms = f(interp_times)
    
    # Kembalikan sinyal (dalam ms) dan sampling rate barunya
    return interp_rr_signal_ms, fs

@st.cache_data # Meng-cache hasil agar perpindahan halaman cepat
def compute_frequency_domain_features(pi_intervals, fs_interp=4):
    """
    Menghitung semua fitur domain frekuensi dari PI intervals (diharapkan dalam detik).
    
    Mengembalikan:
    - freqs (array): Vektor frekuensi dari PSD
    - psd (array): Vektor Power Spectral Density (dalam ms^2/Hz)
    - features (dict): Kamus berisi semua fitur yang dihitung
    """
    
    # --- Langkah 1: Interpolasi PI intervals ---
    try:
        # Ubah PI intervals (detik) menjadi sinyal 4Hz (dalam milidetik)
        rr_signal_ms, fs_rr = _interpolate_rr(pi_intervals, fs=fs_interp)
    except ValueError:
         # Tidak cukup data untuk diinterpolasi
        st.warning("Gagal melakukan analisis frekuensi: Sinyal terlalu pendek.")
        return None, None, {}

    # --- Langkah 2: Detrend sinyal (hapus mean) ---
    rr_signal_detrended = rr_signal_ms - np.mean(rr_signal_ms)
    
    # --- Langkah 3: Hitung PSD menggunakan metode Welch ---
    # Menggunakan jendela 256 sampel (64 detik pada 4Hz) adalah standar yang baik untuk HRV
    freqs, psd = welch(
        rr_signal_detrended, 
        fs=fs_rr, 
        nperseg=256, 
        noverlap=128,  # 50% overlap
        nfft=256, 
        scaling='density' # Menghasilkan unit ms^2/Hz
    )
    
    # --- Langkah 4: Tentukan pita frekuensi (berdasarkan materi Anda) [cite: 1116, 1117, 1118] ---
    vlf_band = (0.003, 0.04)
    lf_band = (0.04, 0.15)
    hf_band = (0.15, 0.4)

    # --- Langkah 5: Hitung power di setiap band (integrasi PSD) ---
    def get_band_power(f_low, f_high):
        idx_band = np.where((freqs >= f_low) & (freqs < f_high))[0]
        if len(idx_band) == 0:
            return 0
        # Integrasi PSD menggunakan trapezoidal rule
        return np.trapz(psd[idx_band], freqs[idx_band])

    vlf_power = get_band_power(*vlf_band)
    lf_power = get_band_power(*lf_band)
    hf_power = get_band_power(*hf_band)
    
    # Total Power (TP) adalah jumlah dari semua komponen [cite: 1260]
    total_power = vlf_power + lf_power + hf_power
    
    # --- Langkah 6: Hitung LF/HF ratio dan unit normalisasi (n.u.) [cite: 1128, 1129, 1132] ---
    # n.u. dihitung dengan mengecualikan VLF
    lf_plus_hf = lf_power + hf_power
    
    lf_nu = (lf_power / lf_plus_hf) * 100 if lf_plus_hf > 0 else 0
    hf_nu = (hf_power / lf_plus_hf) * 100 if lf_plus_hf > 0 else 0
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else np.inf

    # --- Langkah 7: Temukan Frekuensi Puncak (Peak Frequency) [cite: 1260] ---
    def get_peak_freq(f_low, f_high):
        idx_band = np.where((freqs >= f_low) & (freqs < f_high))[0]
        if len(idx_band) == 0:
            return None
        try:
            # Temukan indeks dari magnitudo PSD terbesar di dalam band
            peak_idx = idx_band[np.argmax(psd[idx_band])]
            return freqs[peak_idx]
        except ValueError:
            return None

    lf_peak_freq = get_peak_freq(*lf_band)
    hf_peak_freq = get_peak_freq(*hf_band)

    # --- Langkah 8: Susun kamus fitur ---
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