import lib.dwt as dwt
import lib.bmepy as bmepy
import lib.time_domain as bmetm
import lib.frequency_analysis as bmefq
import lib.poincare as poincare
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from scipy.signal import butter, filtfilt, resample, find_peaks


def load_force(csv_path):
    """
    Load a two-column CSV (time,force) from a path or an uploaded file-like object.
    Accepts either a filesystem path (str) or a file-like object (Streamlit UploadedFile).
    """
    time = []
    force = []
    # If csv_path is a file-like (has read), read its contents
    if hasattr(csv_path, 'read'):
        raw = csv_path.read()
        # raw may be bytes or str
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8')
        lines = raw.splitlines()
        iterator = lines
    else:
        with open(csv_path, 'r', encoding='utf-8') as f:
            iterator = f.readlines()

    for line in iterator:
        parts = line.strip().split(',')
        if len(parts) >= 2 and parts[1]:
            try:
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

@st.cache_data
def load_data(file):
    df = pd.read_csv(file, sep=',')
    df.columns = [col.lower() for col in df.columns]
    signal = df['ir'] if 'ir' in df.columns else df.iloc[:,0]
    return signal

@st.cache_data
def preprocess_signal(signal, sampling_rate, target_sampling_rate):
    new_length = int(len(signal) * target_sampling_rate / sampling_rate)
    signal = resample(signal, new_length)
    signal = np.mean(signal) - signal
    signal = z_score(signal)
    b, a = butter(4, [0.5, 10], fs=target_sampling_rate, btype='band')
    filtered_signal = filtfilt(b, a, signal)
    time = np.arange(len(signal)) / target_sampling_rate
    total_time = len(signal) / target_sampling_rate
    return signal, filtered_signal, time, total_time

@st.cache_data
def cached_discrete_wavelet_transform():
    return dwt.discrete_wavelet_transform()

@st.cache_data
def cached_dwt_freq_response(fs):
    return dwt.dwt_freq_response(fs)

@st.cache_data
def cached_compute_dwt_bandwidths(fs):
    return dwt.compute_dwt_bandwidths(fs)

def plot_signal(time, signal, title='Signal', x_label='Time (s)', y_label='Amplitude'):
    fig = go.Figure()
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
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label) # Sebelumnya 'y_label'
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
            "Non-Linear Analysis",
            "All Features & Export"
        ]
    )
    st.markdown("---")
    st.markdown("Made by hiikariri")
    st.markdown("Made by hatt04")
    st.markdown("Made by sinterchlas")

uploaded_file = st.file_uploader('Upload PPG CSV file', type=['csv'])
original_sampling_rate = st.number_input('Sampling Rate (Hz)', min_value=1, value=50)
target_sampling_rate = st.number_input('Target Sampling Rate (Hz)', min_value=1, value=40)
rr_file = st.file_uploader('Upload ground-truth RR CSV (optional -> two columns: time,force)', type=['csv'], key='rr_gt')
ppg_features = {}
pi_intervals = None     
peak_indices = None   

if uploaded_file:
    signal = load_data(uploaded_file)
    processed_signal, filtered_signal, time, total_time = preprocess_signal(signal, original_sampling_rate, target_sampling_rate)
    st.write(f'Total Time: {total_time:.2f} seconds')
    plot_signal(time, signal, title='Raw PPG Signal', x_label='Time (s)', y_label='Amplitude')
    plot_signal(time, processed_signal, title='Processed PPG Signal', x_label='Time (s)', y_label='Amplitude')
    plot_signal(time, filtered_signal, title='Filtered PPG Signal', x_label='Time (s)', y_label='Amplitude')

    st.subheader("Parameter Deteksi Puncak (HRV)")
    col1, col2, col3 = st.columns(3)
    peak_height = col1.number_input("Peak Height", min_value=0.0, value=0.5, step=0.1, help="Ketinggian minimum puncak (lihat sumbu Y pada 'Filtered PPG Signal')")
    peak_distance = col2.number_input("Peak Distance (samples)", min_value=1, value=20, step=1, help="Jarak minimum antar puncak (mis. 20 sampel pada 40Hz = 0.5 detik)")
    peak_prominence = col3.number_input("Peak Prominence", min_value=0.0, value=0.5, step=0.1, help="Seberapa 'menonjol' puncak dari sinyal di sekitarnya")

    peak_indices = bmepy.detect_peaks(filtered_signal, height=peak_height, distance=peak_distance, prominence=peak_prominence)
    
    if len(peak_indices) > 1:
        pi_intervals = np.diff(time[peak_indices])
    else:
        st.warning("Not enough peaks detected in the signal to perform full HRV analysis. Please adjust peak parameters above.")

if page.startswith("Respiratory & Vasometric Extraction"):
    st.markdown("---")
    st.header("Respiratory & Vasometric Extraction")
    st.markdown("---")
    if uploaded_file:
        # DWT Decomposition
        h = []
        g = []
        qj = [[] for _ in range(9)]
        delay = []

        h, g, qj, delay = cached_discrete_wavelet_transform()
        plot_bar(np.arange(len(h)), h, title='h(n)', x_label='n', y_label='h(n)')
        plot_bar(np.arange(len(g)), g, title='g(n)', x_label='n', y_label='g(n)')
        for j in range(1, 9):
            plot_bar(None, qj[j][0:len(qj[j])], title=f'q{j}(k)', x_label='k', y_label=f'q{j}(k)')

        st.write(f'Delay values: {delay}')

        Q = cached_dwt_freq_response(target_sampling_rate)
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
        df_bands = cached_compute_dwt_bandwidths(target_sampling_rate)
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

        real_respiration_rate = None
        if rr_file is not None:
            try:
                rr_time, rr_force = load_force(rr_file)
                if len(rr_time) > 0 and len(rr_force) > 0:
                    rr_force = rr_force - np.mean(rr_force)
                    rr_peaks = find_peaks(rr_force, height=2, distance=1)[0]
                    real_respiration_rate = len(rr_peaks)/(rr_time[-1]/60) if rr_time[-1] > 0 else None
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

                    # show comparison of estimated vs ground truth if available
                    if real_respiration_rate is not None:
                        st.write(f"Estimated Respiration Rate: {respiration_rate} breaths per minute")
                        st.write(f"Label Respiration Rate: {real_respiration_rate} breaths per minute")
                    else:
                        st.write(f"Estimated Respiration Rate: {respiration_rate} breaths per minute")
                else:
                    st.warning("Uploaded RR file seems empty or incorrectly formatted.")
            except Exception as e:
                st.error(f"Failed to load RR ground truth file: {e}")
        else:
            st.info("No ground-truth RR file uploaded.")

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
        if real_respiration_rate is not None:
            st.write(f"Label Respiration Rate: {real_respiration_rate} breaths per minute")
        else:
            st.write(f"Label Respiration Rate: N/A")

        # Vasometric Activity Extraction
        st.subheader("Vasometric Activity Extraction")
        scale_8 = np.array(dwt_signal[8])
        plot_signal(x_axis, scale_8, title='Vasometric Activity Signal (Scale 8)', x_label='Sample Index', y_label='Amplitude')
        
        N = len(scale_8)
        dft_result = bmepy.dft(scale_8) # from scratch
        # fft_result = scipy.fft.fft(scale_8) # library
        
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
    st.markdown("---")

    if uploaded_file and peak_indices is not None:
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

        if pi_intervals is not None:
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

            time_domain_features = bmetm.compute_time_domain_features(pi_intervals)
            st.subheader("Time Domain Features")
            data = pd.DataFrame.from_dict(time_domain_features, orient='index', columns=['Value'])
            data = data.reset_index()
            data.columns = ['Feature', 'Value']
            st.dataframe(data.style.format({"Value": "{:.4f}"}))
            # Merge time-domain features into global ppg_features (make keys unique if needed)
            for k, v in time_domain_features.items():
                try:
                    ppg_features[str(k)] = float(v) if v is not None else None
                except Exception:
                    ppg_features[str(k)] = v
        
        else:
            st.error("Gagal menghitung fitur Time Domain. Hanya 1 atau 0 puncak terdeteksi. Silakan sesuaikan parameter di halaman utama.")

    elif uploaded_file:
        st.warning("Could not perform time domain analysis. Check if enough peaks were detected on the main page.")
    else:
        st.info("Please upload a PPG file to begin analysis.")

elif page.startswith("Frequency Domain Analysis"):
    st.markdown("---")
    st.header("Frequency Domain Analysis")
    st.markdown("---")

    if uploaded_file and pi_intervals is not None:
        freqs, psd, freq_features = bmefq.compute_frequency_domain_features(pi_intervals, fs_interp=4)
        if freqs is not None and psd is not None and freq_features:
            # Tentukan band untuk plotting
            vlf_band = (0.003, 0.04)
            lf_band = (0.04, 0.15)
            hf_band = (0.15, 0.4)
            
            # PSD (Welch Plot)
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
                xaxis_range=[0, 0.5] # limit under 0.5 Hz
            )
            st.plotly_chart(fig_psd)

            # Bar Plot (Absolute dan Normalized)
            st.subheader("Power Band Analysis")

            # Ambil semua nilai fitur
            lf_power = freq_features.get('LF Power (ms²)')
            hf_power = freq_features.get('HF Power (ms²)')
            lf_nu = freq_features.get('LF (n.u.)')
            hf_nu = freq_features.get('HF (n.u.)')
            lf_hf_ratio = freq_features.get('LF/HF Ratio')

            col1, col2 = st.columns(2)

            with col1:
                fig_bar_abs = go.Figure()
                fig_bar_abs.add_trace(go.Bar(
                    x=['LF Power', 'HF Power'],
                    y=[lf_power, hf_power],
                    text=[f'{lf_power:.2f} ms²', f'{hf_power:.2f} ms²'],
                    textposition='auto',
                    marker_color=['#FF8C00', '#DC143C']
                ))
                fig_bar_abs.update_layout(
                    title=f'Absolute Power',
                    yaxis_title='Power (ms²)'
                )
                st.plotly_chart(fig_bar_abs)

            with col2:
                # Plot normalized units (LF n.u. dan HF n.u.)
                fig_bar_nu = go.Figure()
                fig_bar_nu.add_trace(go.Bar(
                    x=['LF (n.u.)', 'HF (n.u.)'],
                    y=[lf_nu, hf_nu],
                    text=[f'{lf_nu:.1f}%', f'{hf_nu:.1f}%'],
                    textposition='auto',
                    marker_color=['#FF8C00', '#DC143C']
                ))
                fig_bar_nu.update_layout(
                    title=f'Normalized Power (LF/HF Ratio: {lf_hf_ratio:.2f})',
                    yaxis_title='Normalized Units (n.u.)'
                )
                st.plotly_chart(fig_bar_nu)


            # Frequency domain feature result
            st.subheader("All Frequency Domain Features")
            data = pd.DataFrame.from_dict(freq_features, orient='index', columns=['Value'])
            data = data.reset_index()
            data.columns = ['Feature', 'Value']
            st.dataframe(data.style.format({"Value": "{:.4f}"}))
            # Merge frequency-domain features into global ppg_features
            for k, v in freq_features.items():
                try:
                    ppg_features[str(k)] = float(v) if v is not None else None
                except Exception:
                    ppg_features[str(k)] = v

    elif uploaded_file:
        st.warning("Could not perform frequency domain analysis. Check if enough peaks were detected on the main page.")
    else:
        st.info("Please upload a PPG file to begin analysis.")

elif page.startswith("Non-Linear Analysis"):
    st.markdown("---")
    st.header("Non-Linear Analysis")
    st.info("Poincaré Plot (SD1 / SD2) and other non-linear features")
    st.markdown("---")

    if not uploaded_file:
        st.info("Upload a PPG CSV file to compute non-linear features.")
    else:
        # Detect peaks (same parameters as Time Domain) and compute PI intervals
        peak_indices = bmepy.detect_peaks(filtered_signal, height=0.5, distance=5, prominence=0.5)

        if len(peak_indices) > 1:
            pi_intervals = np.diff(time[peak_indices])
            
            # Convert to ms for Poincaré analysis using library function
            rr_ms = poincare.convert_intervals_to_ms(pi_intervals, unit='seconds')

            if len(rr_ms) < 2:
                st.write("Not enough RR intervals to compute Poincaré metrics.")
            else:
                # Compute Poincaré metrics using library
                poincare_metrics = poincare.compute_poincare_features(rr_ms)
                
                SD1 = poincare_metrics['SD1']
                SD2 = poincare_metrics['SD2']
                SD_ratio = poincare_metrics['SD1_SD2_ratio']

                # Store in ppg_features dictionary
                ppg_features['poincare_SD1 (ms)'] = round(SD1, 4) if SD1 is not None else None
                ppg_features['poincare_SD2 (ms)'] = round(SD2, 4) if SD2 is not None else None
                ppg_features['poincare_SD1/SD2_ratio'] = round(SD_ratio, 4) if SD_ratio is not None else None

                st.subheader("Poincaré Analysis Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="SD1 (Short-term HRV)",
                        value=f"{SD1:.2f} ms" if SD1 is not None else "N/A",
                        help="Standard deviation perpendicular to identity line. Reflects parasympathetic (vagal) activity."
                    )
                
                with col2:
                    st.metric(
                        label="SD2 (Long-term HRV)",
                        value=f"{SD2:.2f} ms" if SD2 is not None else "N/A",
                        help="Standard deviation along identity line. Reflects overall HRV (sympathetic + parasympathetic)."
                    )
                
                with col3:
                    st.metric(
                        label="SD1/SD2 Ratio",
                        value=f"{SD_ratio:.3f}" if SD_ratio is not None else "N/A",
                        help="Ratio indicates balance between short-term and long-term variability."
                    )
                
                st.markdown("---")
                if SD_ratio is not None:
                    interpretation = poincare.interpret_poincare_ratio(SD_ratio)
                    if SD_ratio < 0.3:
                        color = "🔴"
                        status = "Stres Tinggi / Dominasi Simpatik"
                    elif SD_ratio < 0.5:
                        color = "🟠"
                        status = "Stres Sedang / Aktivitas Simpatik"
                    elif SD_ratio < 1.0:
                        color = "🟢"
                        status = "Aktivitas Otonom Seimbang"
                    elif SD_ratio < 1.5:
                        color = "🔵"
                        status = "Dominasi Parasimpatik"
                    else:
                        color = "🟣"
                        status = "Aktivitas Parasimpatik Kuat"
                    
                    st.markdown(f"""
                    <div style='padding: 20px; background-color: #13263b; border-radius: 10px; border-left: 5px solid #4CAF50;'>
                        <h3 style='margin-top: 0;'>{color} Autonomic Balance Status</h3>
                        <p style='font-size: 18px; font-weight: bold; color: #d11e15;'>{status}</p>
                        <p style='font-size: 14px; line-height: 1.6;'>{interpretation}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    with st.expander("📖 Penjelasan Detail Metrik Poincaré", expanded=False):
                        st.markdown(f"""
                        ### Statistik Interval RR
                        - **Total Interval RR**: {len(rr_ms)} interval
                        - **Rata-rata Interval RR**: {np.mean(rr_ms):.2f} ms ({60000/np.mean(rr_ms):.1f} bpm)
                        - **Standar Deviasi Interval RR**: {np.std(rr_ms):.2f} ms
                        - **Interval RR Minimum**: {np.min(rr_ms):.2f} ms
                        - **Interval RR Maksimum**: {np.max(rr_ms):.2f} ms
                        - **Rentang**: {np.max(rr_ms) - np.min(rr_ms):.2f} ms
                        
                        ---
                        
                        ### Metrik Poincaré Plot
                        
                        **SD1 (Variabilitas Jangka Pendek)**: {SD1:.2f} ms
                        - Merepresentasikan **variabilitas beat-to-beat** (antar detak)
                        - Mencerminkan aktivitas sistem saraf **parasimpatik** (vagal)
                        - SD1 Tinggi → Aktivitas parasimpatik lebih tinggi → Stres lebih rendah
                        - SD1 Rendah → Tonus vagal berkurang → Stres/kelelahan lebih tinggi
                        
                        **SD2 (Variabilitas Jangka Panjang)**: {SD2:.2f} ms
                        - Merepresentasikan **variabilitas interval RR secara keseluruhan**
                        - Mencerminkan aktivitas simpatik dan parasimpatik
                        - Menunjukkan fluktuasi detak jantung jangka panjang
                        
                        **Rasio SD1/SD2**: {SD_ratio:.3f}
                        - **< 0.3**: Aktivitas simpatik sangat tinggi (stres tinggi)
                        - **0.3 - 0.5**: Dominasi simpatik (stres sedang)
                        - **0.5 - 1.0**: Fungsi otonom seimbang (sehat)
                        - **1.0 - 1.5**: Dominasi parasimpatik (kondisi rileks)
                        - **> 1.5**: Aktivitas parasimpatik sangat tinggi (sangat rileks)
                        
                        ---
                        
                        ### Signifikansi Klinis
                        
                        - **Individu sehat**: Umumnya memiliki rasio SD1/SD2 antara 0.5 - 1.0
                        - **Stres/Kecemasan**: Sering menunjukkan rasio < 0.5 (SD1 berkurang)
                        - **Atlet/Kondisi rileks**: Dapat menunjukkan rasio > 1.0 (tonus vagal tinggi)
                        - **Penyakit kardiovaskular**: Sering menunjukkan SD1 dan SD2 yang berkurang
                        
                        ---
                        
                        ### Referensi
                        - Brennan, M., et al. (2001). "Do existing measures of Poincaré plot geometry reflect nonlinear features of heart rate variability?" *IEEE Trans Biomed Eng*, 48(11), 1342-1347.
                        - Karmakar, C. K., et al. (2009). "Complex correlation measure: a novel descriptor for Poincaré plot." *Biomed Eng Online*, 8, 17.
                        """)

                    st.markdown("---")
                    export_data = {
                        "Metrik": [
                            "Total Interval RR",
                            "Rata-rata RR (ms)",
                            "Rata-rata HR (bpm)",
                            "Std RR (ms)",
                            "Min RR (ms)",
                            "Maks RR (ms)",
                            "Rentang (ms)",
                            "SD1 (ms)",
                            "SD2 (ms)",
                            "Rasio SD1/SD2",
                            "Status Otonom"
                        ],
                        "Nilai": [
                            len(rr_ms),
                            f"{np.mean(rr_ms):.2f}",
                            f"{60000/np.mean(rr_ms):.1f}",
                            f"{np.std(rr_ms):.2f}",
                            f"{np.min(rr_ms):.2f}",
                            f"{np.max(rr_ms):.2f}",
                            f"{np.max(rr_ms) - np.min(rr_ms):.2f}",
                            f"{SD1:.2f}",
                            f"{SD2:.2f}",
                            f"{SD_ratio:.3f}",
                            status
                        ]
                    }
                    
                    export_df = pd.DataFrame(export_data)
                    
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        csv = export_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Analysis (CSV)",
                            data=csv,
                            file_name=f"poincare_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )

                st.markdown("---")
                st.subheader("Poincaré Plot Visualization")
                
                try:
                    fig, ax, metrics = poincare.plot_poincare(
                        rr_ms, 
                        title='Poincaré Plot (SD1/SD2 Analysis)',
                        show_ellipse=True,
                        figsize=(8, 8),
                        margin_factor=0.05,        # 5% margin (lebih ketat)
                        use_percentile=True,       # Gunakan percentile untuk ignore outlier
                        percentile_range=(2, 1)    # Fokus ke 96% data di tengah
                    )
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error creating Poincaré plot: {str(e)}")

                st.markdown("---")
                st.subheader("Extracted PPG Features (including Poincaré)")
                if ppg_features:
                    data = pd.DataFrame.from_dict(ppg_features, orient='index', columns=['Value'])
                    data = data.reset_index()
                    data.columns = ['Feature', 'Value']
                    st.dataframe(data, use_container_width=True)
        else:
            st.write("Not enough detected beats to compute Poincaré plot (need at least 2 peaks).")

elif page.startswith("All Features & Export"):
    st.markdown("---")
    st.header("All Extracted Features & Export")
    st.markdown("---")

    if not ppg_features:
        st.info("No features have been extracted yet. Run analyses on the other pages first.")
    else:
        # Build a dataframe from ppg_features and display
        all_features_df = pd.DataFrame.from_dict(ppg_features, orient='index', columns=['Value'])
        all_features_df = all_features_df.reset_index()
        all_features_df.columns = ['Feature', 'Value']
        # Try formatting numeric values
        def try_float(x):
            try:
                return float(x)
            except Exception:
                return x

        all_features_df['Value'] = all_features_df['Value'].apply(try_float)

        st.subheader('Extracted Features')
        st.dataframe(all_features_df, use_container_width=True)

        # Export button
        csv_data = all_features_df.to_csv(index=False)
        st.download_button(
            label='📥 Download All Features (CSV)',
            data=csv_data,
            file_name=f'all_extracted_features_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mime='text/csv'
        )
