# ================== IMPORTS ==================
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import tempfile
import endaq

from scipy.fft import fft
from scipy.signal import welch, get_window, find_peaks
from sklearn.ensemble import IsolationForest

# ================== PAGE ==================
st.set_page_config(layout="wide")
st.title("📊 enDAQ SHM Dashboard (Channel 80)")

# ================== SIDEBAR ==================

# -------- SAMPLING FREQUENCY --------
st.sidebar.header("Sampling Frequency")
fs_auto_placeholder = st.sidebar.empty()

use_manual_fs = st.sidebar.checkbox("Use Manual fs", False)
manual_fs = st.sidebar.number_input("Manual fs (Hz)", value=0.0)

# -------- TIME RANGE --------
st.sidebar.header("Time Selection")
duration_placeholder = st.sidebar.empty()

# Create placeholder for slider (important)
time_slider_placeholder = st.sidebar.empty()

# -------- RESAMPLING --------
st.sidebar.header("Resampling")
resample_ratio = st.sidebar.slider(
    "Resample Ratio (target fs / current fs)",
    0.1, 2.0, 1.0, step=0.1
)

# -------- OTHER SETTINGS --------
ma_window = st.sidebar.slider("Moving Average Window", 1, 50, 5)

# FFT
st.sidebar.header("FFT Settings")
use_windowing = st.sidebar.checkbox("Windowing", False)
use_averaging = st.sidebar.checkbox("Averaging", False)
use_overlap = st.sidebar.checkbox("Overlap", False)

window_type = "hann"
window_size = 1024
overlap_factor = 0.5

if use_windowing:
    window_type = st.sidebar.selectbox("Window Type", ["hann", "hamming", "blackman", "bartlett"])

if use_averaging or use_overlap:
    window_size = st.sidebar.slider("Window Size", 128, 4096, 1024)

if use_overlap:
    overlap_factor = st.sidebar.slider("Overlap Factor", 0.1, 0.9, 0.5)

# Frequency range
st.sidebar.subheader("Frequency Range")
fmin = st.sidebar.number_input("Min Frequency", value=0.0)
fmax = st.sidebar.number_input("Max Frequency", value=2000.0)

# Peaks
st.sidebar.header("Peak Detection")
peak_height = st.sidebar.number_input("Min Height", value=0.0)
peak_distance = st.sidebar.slider("Min Distance", 1, 200, 20)
peak_prominence = st.sidebar.number_input("Prominence", value=0.0)

# PSD
st.sidebar.subheader("PSD")
nperseg = st.sidebar.slider("nperseg", 128, 4096, 1024)

# Anomaly
st.sidebar.subheader("Anomaly Detection")
contamination = st.sidebar.slider("Contamination", 0.001, 0.1, 0.01)

# ================== FILE ==================
uploaded_file = st.file_uploader("Upload IDE file", type=["ide"])

# ================== FUNCTIONS ==================

def estimate_fs(time_index):
    dt = np.diff(time_index.astype('int64') / 1e9)
    dt = dt[dt > 0]

    if len(dt) == 0:
        return None, True

    median_dt = np.median(dt)
    std_dt = np.std(dt)

    fs_est = 1.0 / median_dt
    irregular = (std_dt / median_dt) > 0.01

    return fs_est, irregular


def apply_fft(signal_data, fs):
    n = len(signal_data)

    if use_averaging or use_overlap:
        step = int(window_size * (1 - overlap_factor)) if use_overlap else window_size
        freqs = np.fft.fftfreq(window_size, 1/fs)[:window_size//2]
        fft_all = []

        for start in range(0, n - window_size + 1, step):
            seg = signal_data[start:start+window_size]

            if use_windowing:
                seg = seg * get_window(window_type, window_size)

            fft_all.append(np.abs(fft(seg)[:window_size//2]))

        fft_all = np.array(fft_all)

        if use_averaging:
            return freqs, np.mean(fft_all, axis=0)
        else:
            return freqs, fft_all[0]

    else:
        if use_windowing:
            signal_data = signal_data * get_window(window_type, n)

        fft_vals = np.abs(fft(signal_data))
        freqs = np.fft.fftfreq(n, 1/fs)

        return freqs[:n//2], fft_vals[:n//2]


def compute_psd(signal_data, fs):
    return welch(signal_data, fs=fs, nperseg=nperseg)


def detect_peaks(freqs, values):
    peaks, _ = find_peaks(values,
                          height=peak_height,
                          distance=peak_distance,
                          prominence=peak_prominence)
    return peaks


def plot_histograms(df):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for i, col in enumerate(["X (40g)", "Y (40g)", "Z (40g)"]):
        sns.histplot(df[col], bins=50, kde=True, ax=ax[i])
        ax[i].set_title(col)
    st.pyplot(fig)


def run_anomaly(df):
    model = IsolationForest(contamination=contamination, random_state=42)
    df["anomaly"] = model.fit_predict(df[["X (40g)", "Y (40g)", "Z (40g)"]])
    return df


# ================== MAIN ==================
if uploaded_file:

    uploaded_file.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ide") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    doc = endaq.ide.get_doc(path)

    # ---- Load data ----
    acc_df = endaq.ide.get_primary_sensor_data(
        doc=doc.channels[80],
        measurement_type="acceleration",
        time_mode="datetime"
    )
    acc_df = acc_df - acc_df.median()

    df = acc_df.copy()

    # ---- Estimate fs ----
    fs_auto, irregular = estimate_fs(df.index)

    if fs_auto is None:
        st.error("❌ Could not estimate sampling frequency")
        st.stop()

    # ---- Duration ----
    n_samples = len(df)
    duration_sec = n_samples / fs_auto
    duration_placeholder.write(f"⏱️ Duration: {duration_sec:.2f} seconds")

    # ---- Proper time range slider ----
    
    time_range = time_slider_placeholder.slider(
    "Time Range (s)",
    0.0,
    duration_sec,
    (0.0, duration_sec)
    )
    
    start_time, end_time = time_range
    
    st.sidebar.write(f"Selected window: {end_time - start_time:.2f} sec")

    # ---- Manual fs default ----
    if manual_fs == 0.0:
        manual_fs = fs_auto

    fs_auto_placeholder.write(f"📌 Auto fs: {fs_auto:.2f} Hz")

    # ---- Handle irregular sampling ----
    if irregular:
        st.warning("⚠️ Irregular sampling → resampling")

        dt = 1 / fs_auto
        freq_str = f"{int(dt*1000)}L"
        df = df.resample(freq_str).mean().interpolate()

    # ---- Time slicing ----
    start_idx = int(start_time * fs_auto)
    end_idx = int(end_time * fs_auto)

    start_idx = max(0, min(start_idx, n_samples - 1))
    end_idx = max(start_idx + 1, min(end_idx, n_samples))

    df = df.iloc[start_idx:end_idx]

    # ---- Ratio resampling ----
    if resample_ratio != 1.0:
        target_fs = fs_auto * resample_ratio
        dt_new = 1 / target_fs
        freq_str = f"{int(dt_new * 1000)}L"

        df = df.resample(freq_str).mean().interpolate()
        fs_auto = target_fs

    # ---- Final fs ----
    fs = manual_fs if use_manual_fs else fs_auto

    if use_manual_fs:
        st.sidebar.warning(f"Using Manual fs: {fs:.2f} Hz")
    else:
        st.sidebar.success(f"Using Auto fs: {fs:.2f} Hz")

    # ================== TABS ==================
    tab1, tab2, tab3, tab4 = st.tabs(["Time", "Stats", "FFT/PSD", "Anomaly"])

    # -------- TIME --------
    with tab1:
        filt = df.copy()
        for c in ["X (40g)", "Y (40g)", "Z (40g)"]:
            filt[c] = filt[c].rolling(ma_window, min_periods=1).mean()

        fig = go.Figure()
        for c in ["X (40g)", "Y (40g)", "Z (40g)"]:
            fig.add_trace(go.Scatter(x=df.index, y=df[c], name=f"{c} Raw"))
            fig.add_trace(go.Scatter(x=filt.index, y=filt[c], name=f"{c} MA"))

        st.plotly_chart(fig, use_container_width=True)

    # -------- STATS --------
    with tab2:
        st.dataframe(df.describe())
        st.subheader("Histograms")
        plot_histograms(df)

    # -------- FFT + PSD --------
    with tab3:
        fig = go.Figure()
        peak_table = []

        for c in ["X (40g)", "Y (40g)", "Z (40g)"]:
            f, vals = apply_fft(df[c].values, fs)
            mask = (f >= fmin) & (f <= fmax)

            f, vals = f[mask], vals[mask]
            peaks = detect_peaks(f, vals)

            fig.add_trace(go.Scatter(x=f, y=vals, name=c))
            fig.add_trace(go.Scatter(x=f[peaks], y=vals[peaks],
                                     mode="markers",
                                     marker=dict(color="red"),
                                     name=f"{c} Peaks"))

            for p in peaks:
                peak_table.append({"Axis": c, "Freq": f[p], "Mag": vals[p]})

        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pd.DataFrame(peak_table))

        fig2 = go.Figure()
        for c in ["X (40g)", "Y (40g)", "Z (40g)"]:
            f, Pxx = compute_psd(df[c].values, fs)
            mask = (f >= fmin) & (f <= fmax)

            fig2.add_trace(go.Scatter(x=f[mask], y=Pxx[mask], name=c))

        fig2.update_layout(yaxis_type="log")
        st.plotly_chart(fig2, use_container_width=True)

    # -------- ANOMALY --------
    with tab4:
        df_an = run_anomaly(df.copy())

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_an.index, y=df_an["X (40g)"], name="X"))

        anomalies = df_an[df_an["anomaly"] == -1]

        fig.add_trace(go.Scatter(
            x=anomalies.index,
            y=anomalies["X (40g)"],
            mode="markers",
            marker=dict(color="red"),
            name="Anomaly"
        ))

        st.plotly_chart(fig, use_container_width=True)
        st.write("Anomalies detected:", len(anomalies))

else:
    st.info("Upload an IDE file to start")
