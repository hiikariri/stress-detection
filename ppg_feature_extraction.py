import lib.dwt as dwt
import lib.bmepy as bmepy
import lib.time_domain as bmetm
import lib.frequency_analysis as bmefq 
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from scipy.signal import butter, filtfilt, resample, find_peaks


def load_force(csv_path):
    time = []
    force = []
    with open(csv_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2 and parts[1]:
                try:
                    # Try to get time from first column, force from second
                    t = float(parts[0])
                    f_val = float(parts[1])
                    time.append(t)
                    force.append(f_val)
                except ValueError:
                    continue
    return np.array(time), np.array(force)

def z_score(signal):
    return (signal - np.mean(signal)) / np.std(signal)

def min_max_scaling(signal):
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

@st.cache_data # <-- DITAMBAHKAN CACHE
def load_data(file):
    df = pd.read_csv(file, sep=',')
    df.columns = [col.lower() for col in df.columns]
    signal = df['ir'] if 'ir' in df.columns else df.iloc[:,0]
    return signal

@st.cache_data # <-- DITAMBAHKAN CACHE
def preprocess_signal(signal, sampling_rate, target_sampling_rate):
    new_length = int(len(signal) * target_sampling_rate / sampling_rate)
    signal = resample(signal, new_length) # Downsample
    signal = np.mean(signal) - signal  # Invert signal
    signal = z_score(signal) # Z-score normalization
    b, a = butter(4, [0.5, 10], fs=target_sampling_rate, btype='band')
    filtered_signal = filtfilt(b, a, signal) # Bandpass filter
    time = np.arange(len(signal)) / target_sampling_rate
    total_time = len(signal) / target_sampling_rate
    return signal, filtered_signal, time, total_time

def plot_signal(time, signal, title='Signal', x_label='Time (s)', y_label='Amplitude'):
    fig = go.Figure()
    # Support single or dual signal plotting
    if isinstance(signal, tuple) or isinstance(signal, list):
        sig1 = signal[0]
        sig2 = signal[1]
        fig.add_trace(go.Scatter(x=time, y=sig1, mode='lines', name='Signal 1'))
        fig.add_trace(go.Scatter(x=time, y=sig2, mode='lines', name='Signal 2'))
    else:
        fig.add_trace(go.Scatter(x=time, y=signal, mode='lines', name='Signal'))
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)
    st.plotly_chart(fig)

def plot_bar(x, y, title='Bar Plot', x_label='Index', y_label='Amplitude'):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=y, name='bar'))
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)
    st.plotly_chart(fig)

st.markdown("""
<div style='display: flex; align-items: center;'>
    <h1 style='display: inline; margin: 0;'>PPG Feature Extraction</h1>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Stress Detection")
    st.markdown("Analyze PPG signals and extract features for stress detection. Select a page below:")
    page = st.radio(
        "",
        [
            "Respiratory & Vasometric Extraction",
            "Time Domain Analysis",
            "Frequency Domain Analysis",
            "Non-Linear Analysis"
        ]
    )
    st.markdown("---")
    st.markdown("Made by hiikariri")

uploaded_file = st.file_uploader('Upload PPG CSV file', type=['csv'])
original_sampling_rate = st.number_input('Sampling Rate (Hz)', min_value=1, value=50)
target_sampling_rate = st.number_input('Target Sampling Rate (Hz)', min_value=1, value=40)
ppg_features = {}
pi_intervals = None     # <-- 2. INISIALISASI VARIABEL GLOBAL
peak_indices = None   # <-- 2. INISIALISASI VARIABEL GLOBAL

if uploaded_file:
    signal = load_data(uploaded_file)
    processed_signal, filtered_signal, time, total_time = preprocess_signal(signal, original_sampling_rate, target_sampling_rate)
    st.write(f'Total Time: {total_time:.2f} seconds')
    plot_signal(time, signal, title='Raw PPG Signal', x_label='Time (s)', y_label='Amplitude')
    plot_signal(time, processed_signal, title='Processed PPG Signal', x_label='Time (s)', y_label='Amplitude')
    plot_signal(time, filtered_signal, title='Filtered PPG Signal', x_label='Time (s)', y_label='Amplitude')

    # --- 3. PERHITUNGAN GLOBAL HRV ---
    # Dihitung di sini agar 'pi_intervals' dan 'peak_indices' tersedia untuk SEMUA halaman.
    peak_indices = bmepy.detect_peaks(filtered_signal, height=0.1, distance=1, prominence=0.1)
    
    if len(peak_indices) > 1:
        pi_intervals = np.diff(time[peak_indices]) # Interval dalam detik
    else:
        st.warning("Not enough peaks detected in the signal to perform full HRV analysis.")
    # --- AKHIR PERHITUNGAN GLOBAL ---


if page.startswith("Respiratory & Vasometric Extraction"):
    st.markdown("---")
    st.header("Respiratory & Vasometric Extraction")
    st.info("Respiratory Rate and Vasometric Activity Extraction using DWT")
    st.markdown("---")
    if uploaded_file:
        # DWT Decomposition
        h = []
        g = []
        qj = [[] for _ in range(9)]
        delay = []

        h, g, qj, delay = dwt.discrete_wavelet_transform()
        plot_bar(np.arange(len(h)), h, title='h(n)', x_label='n', y_label='h(n)')
        plot_bar(np.arange(len(g)), g, title='g(n)', x_label='n', y_label='g(n)')
        for j in range(1, 9):
            plot_bar(None, qj[j][0:len(qj[j])], title=f'q{j}(k)', x_label='k', y_label=f'q{j}(k)')

        st.write(f'Delay values: {delay}')

        Q = np.zeros((9, target_sampling_rate // 2 + 1))
        Q = dwt.dwt_freq_response(target_sampling_rate)
        fig = go.Figure()

        freqs = np.arange(target_sampling_rate // 2 + 1)
        for j in range(1, 9):
            fig.add_trace(go.Scatter(
                x=freqs,
                y=Q[j],
                mode='lines',
                name=f'Scale {j}'
            ))

        fig.update_layout(
            title='DWT Frequency Responses',
            xaxis_title='Frequency (Hz)',
            yaxis_title='Magnitude',
            legend=dict(title='Scale (j)', orientation='h', y=-0.2),
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # DWT Bandwidths
        df_bands = dwt.compute_dwt_bandwidths(target_sampling_rate)
        st.subheader("DWT Frequency Bandwidths")
        st.dataframe(df_bands.style.format({
            "Min Frequency (Hz)": "{:.4f}",
            "Max Frequency (Hz)": "{:.4f}",
            "Bandwidth (Hz)": "{:.4f}"
        }))

        # DWT Application
        signal_length = len(processed_signal)
        total_time = signal_length / target_sampling_rate
        st.subheader('Select Segment for DWT Analysis')
        segment_time_range = st.slider(
            'Select start and end time (seconds) for DWT segment',
            min_value=0.0,
            max_value=float(f'{total_time:.2f}'),
            value=(0.0, float(f'{total_time:.2f}')),
            step=1/target_sampling_rate
        )
        start_time, end_time = segment_time_range
        start_sample = int(start_time * target_sampling_rate)
        end_sample = int(end_time * target_sampling_rate)
        segment = processed_signal[start_sample:end_sample]
        segment_total_time = (end_sample - start_sample) / target_sampling_rate
        st.write(f"Selected segment duration: {segment_total_time:.2f} seconds")
        
        x_axis = np.arange(len(segment)) / target_sampling_rate + start_time
        plot_signal(x_axis, segment, title='Selected PPG Segment', x_label='Time (s)', y_label='Amplitude')
        dwt_signal = dwt.apply_dwt(delay, qj, segment_total_time, target_sampling_rate, segment)
        for j in range(1, 9):
            x_axis = np.arange(len(dwt_signal[j])) / target_sampling_rate + start_time
            plot_signal(x_axis, np.array(dwt_signal[j]), title=f'DWT Coefficient Scale {j}', x_label='Time (s)', y_label='Amplitude')

        # Respiratory Rate Extraction
        st.subheader('Respiratory Rate Extraction')
        rr_signal = np.array(dwt_signal[7])
        signal_length = len(rr_signal)
        time_array = np.arange(len(rr_signal)) / target_sampling_rate

        rr_signal = rr_signal - np.mean(rr_signal)

        detected_peaks = {}
        peaks, threshold = bmepy.adaptive_threshold_peaks(rr_signal, window_size=20, factor=-1, distance=50)
        respiration_rate = len(peaks)/(segment_total_time/60)
        ppg_features['respiratory_rate (brpm)'] = round(respiration_rate, 4)
        
        csv_path = 'data\data_davis_50hz_rr.csv'
        rr_time, rr_force = load_force(csv_path)
        rr_force = rr_force - np.mean(rr_force)
        rr_peaks = find_peaks(rr_force, height=2, distance=1)[0]
        real_respiration_rate = len(rr_peaks)/(rr_time[-1]/60)
        st.write("Ground Truth Respiration Signal")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rr_time,
            y=rr_force,
            mode='lines',
            name='Respiration Force',
            line=dict(color='grey', width=1, dash='dot')
        ))

        if len(rr_peaks) > 0:
            fig.add_trace(go.Scatter(
                x=rr_time[rr_peaks],
                y=rr_force[rr_peaks],
                mode='markers',
                name=f'Detected Peaks ({len(rr_peaks)})',
                marker=dict(
                    color='green',
                    size=8,
                    symbol='circle',
                    line=dict(width=2, color='white')
                )
            ))

        fig.update_layout(
            title='Respiration Force Peak Detection',
            xaxis_title='Time (seconds)',
            yaxis_title='Force (N)',
            template='plotly_white',
            height=500,
            showlegend=True
        )
        st.plotly_chart(fig)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_array,
            y=rr_signal,
            mode='lines',
            name='Scale 7 Signal',
            line=dict(color='blue', width=1),
            opacity=0.7
        ))

        if len(peaks) > 0:
            fig.add_trace(go.Scatter(
                x=time_array[peaks],
                y=rr_signal[peaks],
                mode='markers',
                name=f'Detected Peaks ({len(peaks)})',
                marker=dict(
                    color='green',
                    size=8,
                    symbol='circle',
                    line=dict(width=2, color='white')
                )
            ))

        fig_detailed = go.Figure()
        fig_detailed.add_trace(go.Scatter(
            x=time_array,
            y=rr_signal,
            mode='lines',
            name='Scale 7 Signal',
            line=dict(color='blue', width=1.5)
        ))

        # Detected peaks
        if len(peaks) > 0:
            fig_detailed.add_trace(go.Scatter(
                x=time_array[peaks],
                y=rr_signal[peaks],
                mode='markers',
                name=f'Detected Peaks ({len(peaks)})',
                marker=dict(
                    color='green',
                    size=10,
                    symbol='circle',
                    line=dict(width=2, color='white')
                )
            ))
            
            for i, peak_idx in enumerate(peaks):
                fig_detailed.add_vline(
                    x=time_array[peak_idx],
                    line_dash="dash",
                    line_color="green",
                    opacity=0.4,
                    annotation_text=f"P{i+1}",
                    annotation_position="top"
                )

        fig_detailed.add_trace(go.Scatter(
            x=time_array,
            y=threshold,
            mode='lines',
            name='Adaptive Threshold',
            line=dict(color='red', width=1, dash='dot')
        ))

        fig_detailed.update_layout(
            title='Peak Detection Results',
            xaxis_title='Time (seconds)',
            yaxis_title='Amplitude',
            template='plotly_white',
            height=500,
            showlegend=True
        )
        st.plotly_chart(fig_detailed)
        st.write(f"Estimated Respiration Rate: {respiration_rate} breaths per minute")
        st.write(f"Label Respiration Rate: {real_respiration_rate} breaths per minute")

        # Vasometric Activity Extraction
        st.subheader("Vasometric Activity Extraction")
        scale_8 = np.array(dwt_signal[8])
        plot_signal(x_axis, scale_8, title='Vasometric Activity Signal (Scale 8)', x_label='Sample Index', y_label='Amplitude')
        
        N = len(scale_8)
        dft_result = bmepy.dft(scale_8)
        # fft_result = scipy.fft.fft(scale_8)
        
        T = N/target_sampling_rate
        freqs = np.arange(N)/T
        freqs_pos = freqs[:N//2]
        mag_pos = np.abs(dft_result[:N//2])

        # Find highest peak magnitude between 0.04 and 0.15 Hz
        freq_mask = (freqs_pos >= 0.04) & (freqs_pos <= 0.15)
        peak_freq = None
        peak_mag = None
        if np.any(freq_mask):
            mag_band = mag_pos[freq_mask]
            freq_band = freqs_pos[freq_mask]
            peak_idx = np.argmax(mag_band)
            peak_freq = freq_band[peak_idx]
            peak_mag = mag_band[peak_idx]
            st.write(f"Highest peak magnitude in 0.04-0.15 Hz: {peak_mag:.4f} at {peak_freq:.4f} Hz")
            ppg_features['vasometric_activity (Hz)'] = round(peak_freq, 4)
        else:
            st.write("No frequency components found in 0.04-0.15 Hz range.")
            ppg_features['vasometric_activity (Hz)'] = None

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=freqs_pos,
            y=mag_pos,
            mode='lines',
            name='FFT Magnitude'
        ))
        if peak_freq is not None and peak_mag is not None:
            fig.add_trace(go.Scatter(
                x=[peak_freq],
                y=[peak_mag],
                mode='markers',
                name='Highest Peak (0.04-0.15 Hz)',
                marker=dict(color='red', size=10, symbol='x')
            ))
        fig.update_layout(
            title='FFT of Scale 8 Signal',
            xaxis_title='Frequency (Hz)',
            yaxis_title='Magnitude'
        )
        st.plotly_chart(fig)

        st.write("Extracted PPG Features")
        data = pd.DataFrame.from_dict(ppg_features, orient='index', columns=['Value'])
        data = data.reset_index()
        data.columns = ['Feature', 'Value']
        st.dataframe(data)

elif page.startswith("Time Domain Analysis"):
    st.markdown("---")
    st.header("Time Domain Analysis")
    st.info("HRV Analysis of PPG Signal")
    st.markdown("---")

    if uploaded_file and pi_intervals is not None and peak_indices is not None:
        st.subheader("Filtered Signal with Detected Peaks (Local Maxima)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time, y=filtered_signal, mode='lines', name='Filtered Signal'))
        fig.add_trace(go.Scatter(
            x=time[peak_indices],
            y=filtered_signal[peak_indices],
            mode='markers',
            name=f'Detected Peaks ({len(peak_indices)})',
            marker=dict(color='red', size=8, symbol='circle')
        ))
        fig.update_layout(
            title='Filtered Signal with Detected Peaks',
            xaxis_title='Time (s)',
            yaxis_title='Amplitude',
            xaxis=dict(range=[np.min(time), np.max(time)]),
            yaxis=dict(range=[np.min(filtered_signal), np.max(filtered_signal)])
        )
        st.plotly_chart(fig)

        # PI Interval calculation
        st.subheader("PI Intervals (seconds)")
        st.write(pi_intervals)
        
        # Tachogram visualization
        st.subheader("Tachogram (PI Intervals vs Beat Number)")
        fig_tachogram = go.Figure()
        fig_tachogram.add_trace(go.Scatter(
            x=np.arange(1, len(pi_intervals)+1),
            y=pi_intervals,
            mode='lines+markers',
            name='PI Interval'
        ))
        fig_tachogram.update_layout(
            title='Tachogram',
            xaxis_title='Beat Number',
            yaxis_title='Pulse Interval (s)',
            template='plotly_white',
            height=350
        )
        st.plotly_chart(fig_tachogram)

        st.write(f"Mean Pulse Interval: {np.mean(pi_intervals):.4f} s")
        st.write(f"Std Pulse Interval: {np.std(pi_intervals):.4f} s")

        # Hitung fitur time domain
        time_domain_features = bmetm.compute_time_domain_features(pi_intervals)
        st.subheader("Time Domain Features")
        data = pd.DataFrame.from_dict(time_domain_features, orient='index', columns=['Value'])
        data = data.reset_index()
        data.columns = ['Feature', 'Value']
        st.dataframe(data.style.format({"Value": "{:.4f}"}))

    elif uploaded_file:
        st.warning("Could not perform time domain analysis. Check if enough peaks were detected on the main page.")
    else:
        st.info("Please upload a PPG file to begin analysis.")


elif page.startswith("Frequency Domain Analysis"):
    st.markdown("---")
    st.header("Frequency Domain Analysis")
    st.info("HRV Frequency Domain Analysis using Welch's Periodogram, based on your notebook code and lecture slides.")
    st.markdown("---")

    if uploaded_file and pi_intervals is not None:
    
        freqs, psd, freq_features = bmefq.compute_frequency_domain_features(pi_intervals, fs_interp=4)

        if freqs is not None and psd is not None and freq_features:
            
            # band  plotting
            vlf_band = (0.003, 0.04)
            lf_band = (0.04, 0.15)
            hf_band = (0.15, 0.4)
            
            st.subheader("HRV Power Spectral Density (PSD)")
            fig_psd = go.Figure()
            fig_psd.add_trace(go.Scatter(x=freqs, y=psd, mode='lines', name='PSD', line=dict(color='blue')))
            
            fig_psd.add_vrect(x0=vlf_band[0], x1=vlf_band[1], fillcolor="green", opacity=0.2, layer="below", line_width=0, name="VLF (0.003-0.04 Hz)")
            fig_psd.add_vrect(x0=lf_band[0], x1=lf_band[1], fillcolor="orange", opacity=0.2, layer="below", line_width=0, name="LF (0.04-0.15 Hz)")
            fig_psd.add_vrect(x0=hf_band[0], x1=hf_band[1], fillcolor="red", opacity=0.2, layer="below", line_width=0, name="HF (0.15-0.4 Hz)")
            
        
            lf_peak = freq_features.get('LF Peak Frequency (Hz)')
            hf_peak = freq_features.get('HF Peak Frequency (Hz)')
            
            if lf_peak is not None:
                peak_lf_power = psd[np.argmin(np.abs(freqs - lf_peak))]
                fig_psd.add_annotation(
                    x=lf_peak, y=peak_lf_power,
                    text=f"LF Peak: {lf_peak:.3f} Hz", showarrow=True, arrowhead=1, yshift=10
                )
            if hf_peak is not None:
                peak_hf_power = psd[np.argmin(np.abs(freqs - hf_peak))]
                fig_psd.add_annotation(
                    x=hf_peak, y=peak_hf_power,
                    text=f"HF Peak: {hf_peak:.3f} Hz", showarrow=True, arrowhead=1, yshift=10
                )

            fig_psd.update_layout(
                title="Welch's Power Spectral Density",
                xaxis_title="Frequency (Hz)",
                yaxis_title="Power Spectral Density (ms²/Hz)",
                xaxis_range=[0, 0.5] # Fokus pada area di bawah 0.5 Hz
            )
            st.plotly_chart(fig_psd)

            
            st.subheader("Power Band Analysis")

            # Ambil semua nilai fitur
            lf_power = freq_features.get('LF Power (ms²)')
            hf_power = freq_features.get('HF Power (ms²)')
            lf_nu = freq_features.get('LF (n.u.)')
            hf_nu = freq_features.get('HF (n.u.)')
            lf_hf_ratio = freq_features.get('LF/HF Ratio')

            
            col1, col2 = st.columns(2)

            with col1:
                # Plot untuk Absolute Power (LF dan HF)
                fig_bar_abs = go.Figure()
                fig_bar_abs.add_trace(go.Bar(
                    x=['LF Power', 'HF Power'],
                    y=[lf_power, hf_power],
                    text=[f'{lf_power:.2f} ms²', f'{hf_power:.2f} ms²'],
                    textposition='auto',
                    marker_color=['#FF8C00', '#DC143C'] # Oranye dan Merah
                ))
                fig_bar_abs.update_layout(
                    title=f'Absolute Power',
                    yaxis_title='Power (ms²)'
                )
                st.plotly_chart(fig_bar_abs)

            with col2:
                # Plot untuk Normalized Units (LF n.u. dan HF n.u.)
                fig_bar_nu = go.Figure()
                fig_bar_nu.add_trace(go.Bar(
                    x=['LF (n.u.)', 'HF (n.u.)'],
                    y=[lf_nu, hf_nu],
                    text=[f'{lf_nu:.1f}%', f'{hf_nu:.1f}%'],
                    textposition='auto',
                    marker_color=['#FF8C00', '#DC143C'] # Oranye dan Merah
                ))
                fig_bar_nu.update_layout(
                    title=f'Normalized Power (LF/HF Ratio: {lf_hf_ratio:.2f})',
                    yaxis_title='Normalized Units (n.u.)'
                )
                st.plotly_chart(fig_bar_nu)


            # --- Plot 4: Tabel Fitur ---
            st.subheader("All Frequency Domain Features")
            data = pd.DataFrame.from_dict(freq_features, orient='index', columns=['Value'])
            data = data.reset_index()
            data.columns = ['Feature', 'Value']
            st.dataframe(data.style.format({"Value": "{:.4f}"}))

    elif uploaded_file:
        st.warning("Could not perform frequency domain analysis. Check if enough peaks were detected on the main page.")
    else:
        st.info("Please upload a PPG file to begin analysis.")


elif page.startswith("Non-Linear Analysis"):
    st.markdown("---")
    st.header("Non-Linear Analysis")
    st.info("This page will show non-linear features (to be implemented)")
    st.markdown("---")