
############################################################ V1
# ================== STREAMLIT DASHBOARD ==================
# import streamlit as st
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.graph_objects as go
# import tempfile
# import endaq
#
# from scipy.fft import fft
# from scipy.signal import welch, get_window
# from sklearn.ensemble import IsolationForest
#
# # ================== PAGE CONFIG ==================
# st.set_page_config(layout="wide")
# st.title("📊 enDAQ Vibration Dashboard (Advanced FFT Controls)")
#
# # ================== SIDEBAR ==================
# st.sidebar.header("General Settings")
#
# start_idx = st.sidebar.slider("Start Index", 0, 200000, 50000)
# end_idx = st.sidebar.slider("End Index", 10000, 300000, 60000)
#
# ma_window = st.sidebar.slider("Moving Average Window", 1, 50, 5)
#
# fs = st.sidebar.number_input("Sampling Frequency (Hz)", value=4052.6)
#
# resample_rate = st.sidebar.selectbox(
#     "Resample Rate", ["None", "1S", "5S", "10S"]
# )
#
# # -------- FFT SETTINGS --------
# st.sidebar.header("FFT Settings")
#
# use_windowing = st.sidebar.checkbox("Apply Windowing", False)
# use_averaging = st.sidebar.checkbox("Apply Averaging", False)
# use_overlap = st.sidebar.checkbox("Apply Overlap", False)
#
# window_type = "hann"
# window_size = 1024
# overlap_factor = 0.5
#
# if use_windowing:
#     window_type = st.sidebar.selectbox(
#         "Window Type",
#         ["hann", "hamming", "blackman", "bartlett"]
#     )
#
# if use_averaging or use_overlap:
#     window_size = st.sidebar.slider("Window Size", 128, 4096, 1024)
#
# if use_overlap:
#     overlap_factor = st.sidebar.slider("Overlap Factor", 0.1, 0.9, 0.5)
#
# # Frequency range
# st.sidebar.subheader("Frequency Range")
# fmin = st.sidebar.number_input("Min Frequency", value=0.0)
# fmax = st.sidebar.number_input("Max Frequency", value=2000.0)
#
# # -------- PSD --------
# st.sidebar.subheader("PSD Settings")
# nperseg = st.sidebar.slider("nperseg (PSD)", 128, 4096, 1024)
#
# # -------- Anomaly --------
# st.sidebar.subheader("Anomaly Detection")
# contamination = st.sidebar.slider("Contamination", 0.001, 0.1, 0.01)
#
# # ================== FILE UPLOAD ==================
# uploaded_file = st.file_uploader("Upload IDE file", type=["ide"])
#
# # ================== FUNCTIONS ==================
# @st.cache_data
# def load_data(uploaded_file):
#     uploaded_file.seek(0)
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".ide") as tmp_file:
#         tmp_file.write(uploaded_file.read())
#         temp_path = tmp_file.name
#
#     doc = endaq.ide.get_doc(temp_path)
#
#     acc_df = endaq.ide.get_primary_sensor_data(
#         doc=doc.channels[80],
#         measurement_type="acceleration",
#         time_mode="datetime"
#     )
#
#     temp_df = endaq.ide.get_primary_sensor_data(
#         doc=doc.channels[59],
#         measurement_type="temperature",
#         time_mode="datetime"
#     )
#
#     temp_resampled = temp_df.reindex(acc_df.index, method='nearest')
#
#     df = acc_df.copy()
#     df['Temperature'] = temp_resampled.iloc[:, 0]
#
#     return df
#
#
# def apply_fft_advanced(df, column, fs):
#     signal_data = df[column].values
#     n = len(signal_data)
#
#     if use_averaging or use_overlap:
#         step = int(window_size * (1 - overlap_factor)) if use_overlap else window_size
#         freqs = np.fft.fftfreq(window_size, 1/fs)[:window_size // 2]
#         fft_all = []
#
#         for start in range(0, n - window_size + 1, step):
#             segment = signal_data[start:start + window_size]
#
#             if use_windowing:
#                 window = get_window(window_type, window_size)
#                 segment = segment * window
#
#             fft_vals = np.abs(fft(segment)[:window_size // 2])
#             fft_all.append(fft_vals)
#
#         fft_all = np.array(fft_all)
#
#         if use_averaging:
#             fft_mean = np.mean(fft_all, axis=0)
#             return freqs, fft_mean
#         else:
#             return freqs, fft_all[0]
#
#     else:
#         if use_windowing:
#             window = get_window(window_type, n)
#             signal_data = signal_data * window
#
#         fft_vals = np.abs(fft(signal_data))
#         freqs = np.fft.fftfreq(n, 1/fs)
#
#         return freqs[:n//2], fft_vals[:n//2]
#
#
# def compute_psd(df, column, fs, nperseg):
#     f, Pxx = welch(df[column].values, fs=fs, nperseg=nperseg)
#     return f, Pxx
#
#
# def run_anomaly_detection(df):
#     features = df[["X (40g)", "Y (40g)", "Z (40g)"]]
#     model = IsolationForest(contamination=contamination, random_state=42)
#     df["anomaly"] = model.fit_predict(features)
#     return df
#
#
# # ================== MAIN ==================
# if uploaded_file is not None:
#
#     df = load_data(uploaded_file)
#
#     selected_df = df.iloc[start_idx:end_idx]
#
#     if resample_rate != "None":
#         selected_df = selected_df.resample(resample_rate).mean()
#
#     tab1, tab2, tab3, tab4 = st.tabs(
#         ["📈 Time Series", "📊 Statistics", "⚡ FFT / PSD", "🤖 Anomaly"]
#     )
#
#     # -------- TIME SERIES --------
#     with tab1:
#         filtered_df = selected_df.copy()
#         for col in ["X (40g)", "Y (40g)", "Z (40g)"]:
#             filtered_df[col] = filtered_df[col].rolling(ma_window, min_periods=1).mean()
#
#         fig = go.Figure()
#         for col in ["X (40g)", "Y (40g)", "Z (40g)"]:
#             fig.add_trace(go.Scatter(x=selected_df.index, y=selected_df[col], name=f"{col} Raw"))
#             fig.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df[col], name=f"{col} MA"))
#
#         st.plotly_chart(fig, use_container_width=True)
#
#     # -------- STATS --------
#     with tab2:
#         st.dataframe(selected_df.describe())
#
#     # -------- FFT + PSD --------
#     with tab3:
#         fig_fft = go.Figure()
#
#         for col in ["X (40g)", "Y (40g)", "Z (40g)"]:
#             f, fft_vals = apply_fft_advanced(selected_df, col, fs)
#
#             mask = (f >= fmin) & (f <= fmax)
#
#             fig_fft.add_trace(go.Scatter(
#                 x=f[mask],
#                 y=fft_vals[mask],
#                 name=f"{col} FFT"
#             ))
#
#         fig_fft.update_layout(title="FFT", height=500)
#         st.plotly_chart(fig_fft, use_container_width=True)
#
#         # PSD
#         fig_psd = go.Figure()
#         for col in ["X (40g)", "Y (40g)", "Z (40g)"]:
#             f, Pxx = compute_psd(selected_df, col, fs, nperseg)
#             mask = (f >= fmin) & (f <= fmax)
#
#             fig_psd.add_trace(go.Scatter(
#                 x=f[mask],
#                 y=Pxx[mask],
#                 name=f"{col} PSD"
#             ))
#
#         fig_psd.update_layout(title="PSD", yaxis_type="log", height=500)
#         st.plotly_chart(fig_psd, use_container_width=True)
#
#     # -------- ANOMALY --------
#     with tab4:
#         anomaly_df = run_anomaly_detection(selected_df.copy())
#
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=anomaly_df.index, y=anomaly_df["X (40g)"], name="X"))
#
#         anomalies = anomaly_df[anomaly_df["anomaly"] == -1]
#
#         fig.add_trace(go.Scatter(
#             x=anomalies.index,
#             y=anomalies["X (40g)"],
#             mode="markers",
#             marker=dict(color="red"),
#             name="Anomaly"
#         ))
#
#         st.plotly_chart(fig, use_container_width=True)
#         st.write("Anomalies detected:", len(anomalies))
#
# else:
#     st.info("Upload an IDE file to start")


############################################################# Rev 2
# ================== STREAMLIT DASHBOARD (Channel 80) ==================
# Author: Hamed Dabiri + M365 Copilot
# Description: enDAQ SHM Dashboard forcing channel index 80

# ================== IMPORTS ==================
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import tempfile
import re
import endaq

from scipy.fft import fft
from scipy.signal import welch, get_window, find_peaks
from sklearn.ensemble import IsolationForest
from pandas.api.types import is_numeric_dtype

# ================== PAGE ==================
st.set_page_config(layout="wide")
st.title("📊 enDAQ SHM Dashboard (Channel 80)")

DEBUG = False  # set True to print diagnostics in the app

# ================== SIDEBAR ==================
st.sidebar.header("General Settings")

start_idx = st.sidebar.slider("Start Index", 0, 200000, 0)
end_idx = st.sidebar.slider("End Index", 10000, 300000, 60000)

ma_window = st.sidebar.slider("Moving Average Window", 1, 50, 5)
fs = st.sidebar.number_input("Sampling Frequency (Hz)", value=4052.6)

resample_rate = st.sidebar.selectbox("Resample", ["None", "1S", "5S", "10S"])

# -------- FFT SETTINGS --------
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

# -------- PEAK DETECTION --------
st.sidebar.header("Peak Detection")
peak_height = st.sidebar.number_input("Min Height", value=0.0)
peak_distance = st.sidebar.slider("Min Distance", 1, 200, 20)
peak_prominence = st.sidebar.number_input("Prominence", value=0.0)

# -------- PSD --------
st.sidebar.subheader("PSD")
nperseg = st.sidebar.slider("nperseg", 128, 4096, 1024)

# -------- ANOMALY --------
st.sidebar.subheader("Anomaly Detection")
contamination = st.sidebar.slider("Contamination", 0.001, 0.1, 0.01)

# ================== FILE ==================
uploaded_file = st.file_uploader("Upload IDE file", type=["ide"])

# ================== HELPERS ==================
def normalize_table(raw_table):
    """Normalize any object returned by endaq.ide.get_channel_table into a DataFrame."""
    if raw_table is None:
        return pd.DataFrame()

    try:
        if isinstance(raw_table, pd.DataFrame):
            table = raw_table.copy()
        elif isinstance(raw_table, (list, tuple)):
            if len(raw_table) == 0:
                table = pd.DataFrame()
            elif isinstance(raw_table[0], dict):
                table = pd.DataFrame.from_records(raw_table)
            else:
                table = pd.DataFrame({"value": raw_table})
        elif isinstance(raw_table, dict):
            vals = list(raw_table.values())
            if all(isinstance(v, (list, tuple, np.ndarray)) for v in vals):
                table = pd.DataFrame(raw_table)
            else:
                table = pd.DataFrame([raw_table])
        else:
            table = pd.DataFrame([{"value": raw_table}])
    except Exception as e:
        st.error(f"Failed to create channel table: {e}")
        return pd.DataFrame()

    # normalize column names
    safe_cols = []
    for c in table.columns:
        try:
            safe_cols.append(str(c).lower())
        except Exception:
            safe_cols.append("unnamed")
    table.columns = safe_cols
    return table

def find_col(table, names):
    for n in names:
        n = str(n).lower()
        for c in table.columns:
            if n in c:
                return c
    return None

def detect_acc_axes(df):
    """Detect axis columns, prefer x/y/z numeric columns; fallback to first 3 numeric."""
    cols = list(df.columns)
    numeric_cols = [c for c in cols if is_numeric_dtype(df[c])]
    xyz_accel = []
    for c in numeric_cols:
        name = str(c).lower()
        has_xyz = re.search(r'\b(x|y|z)\b', name) is not None or any(k in name for k in [" x", " y", " z"])
        has_acc = any(k in name for k in ["acc", "accel", "g"])
        if has_xyz and has_acc:
            xyz_accel.append(c)

    if len(xyz_accel) < 3:
        xyz_any = [c for c in numeric_cols if re.search(r'\b(x|y|z)\b', str(c), re.I)]
        merged = []
        for c in xyz_accel + xyz_any:
            if c not in merged:
                merged.append(c)
        xyz_accel = merged

    if len(xyz_accel) == 0:
        xyz_accel = numeric_cols[:3]

    return xyz_accel[:3]

def apply_fft_advanced(signal_data):
    arr = np.array(signal_data, dtype=float)  # copy
    n = len(arr)

    if use_averaging or use_overlap:
        step = int(window_size * (1 - overlap_factor)) if use_overlap else window_size
        if step <= 0:
            step = 1
        freqs = np.fft.fftfreq(window_size, 1/fs)[:window_size // 2]
        fft_all = []

        for start in range(0, n - window_size + 1, step):
            seg = arr[start:start + window_size].copy()
            if use_windowing:
                seg *= get_window(window_type, window_size)
            fft_all.append(np.abs(fft(seg)[:window_size // 2]))

        if len(fft_all) == 0:
            seg = arr.copy()
            if use_windowing:
                seg *= get_window(window_type, n)
            fft_vals = np.abs(fft(seg))
            freqs_full = np.fft.fftfreq(n, 1/fs)
            return freqs_full[:n // 2], fft_vals[:n // 2]

        fft_all = np.array(fft_all)
        return (freqs, np.mean(fft_all, axis=0)) if use_averaging else (freqs, fft_all[0])

    else:
        seg = arr.copy()
        if use_windowing:
            seg *= get_window(window_type, n)
        fft_vals = np.abs(fft(seg))
        freqs = np.fft.fftfreq(n, 1/fs)
        return freqs[:n // 2], fft_vals[:n // 2]

def compute_psd(signal_data):
    nseg = min(nperseg, len(signal_data)) if len(signal_data) > 0 else nperseg
    return welch(signal_data, fs=fs, nperseg=nseg)

def detect_peaks(freqs, values):
    peaks, _ = find_peaks(values,
                          height=peak_height,
                          distance=peak_distance,
                          prominence=peak_prominence)
    return peaks

def plot_histograms(df, axes):
    if len(axes) == 0:
        st.info("No axis columns detected for histograms.")
        return
    cols_to_plot = axes[:3]
    fig, ax = plt.subplots(1, len(cols_to_plot), figsize=(5 * len(cols_to_plot), 4))
    if len(cols_to_plot) == 1:
        ax = [ax]
    for i, col in enumerate(cols_to_plot):
        sns.histplot(df[col], bins=50, kde=True, ax=ax[i])
        ax[i].set_title(str(col))
    st.pyplot(fig)

# ================== MAIN ==================
if uploaded_file:
    uploaded_file.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ide") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    # Load doc
    try:
        doc = endaq.ide.get_doc(path)
    except Exception as e:
        st.error(f"Failed to load IDE document: {e}")
        st.stop()

    # Force channel index 80
    acc_id = 80  # <-- your requested channel index
    total_channels = len(doc.channels)
    st.write("Available Channels")
    st.write(f"Total channels detected: {total_channels}")

    if acc_id < 0 or acc_id >= total_channels:
        st.error(f"Requested channel index {acc_id} is out of range (0..{total_channels-1}).")
        st.stop()

    # Build a simple channel table for display & measurement lookup
    raw_table = endaq.ide.get_channel_table(doc)
    table = normalize_table(raw_table)

    # Show a small preview
    if not table.empty:
        st.dataframe(table.head(20))
    else:
        st.warning("Channel table is empty or unreadable; will still try to load channel 80.")

    # Try to infer measurement type from the table at index 80
    meas_col = find_col(table, ["measurement", "type", "sensor", "quantity"])
    measurement_hint = None
    if (meas_col is not None) and (acc_id in table.index):
        measurement_hint = str(table.loc[acc_id, meas_col]).lower()

    if DEBUG:
        st.write("🔎 Debug: measurement_hint for channel 80 =", measurement_hint)

    # Load data for channel 80 with a robust approach:
    # Try acceleration first if hinted, else try generic loading.
    acc_df = None
    load_errors = []

    # Attempt 1: If hint suggests acceleration, try with measurement_type="acceleration"
    try:
        if measurement_hint and any(k in measurement_hint for k in ["acc", "accel", "acceleration", "g"]):
            acc_df = endaq.ide.get_primary_sensor_data(
                doc=doc.channels[acc_id],
                measurement_type="acceleration",
                time_mode="datetime"
            )
    except Exception as e:
        load_errors.append(f"Acceleration load attempt failed: {e}")

    # Attempt 2: If not loaded yet, try temperature (in case this channel is a temp sensor)
    if acc_df is None:
        try:
            if measurement_hint and any(k in measurement_hint for k in ["temp", "temperature", "therm"]):
                acc_df = endaq.ide.get_primary_sensor_data(
                    doc=doc.channels[acc_id],
                    measurement_type="temperature",
                    time_mode="datetime"
                )
        except Exception as e:
            load_errors.append(f"Temperature load attempt failed: {e}")

    # Attempt 3: Try generic primary sensor data without specifying measurement_type
    if acc_df is None:
        try:
            # Some endaq versions allow calling without measurement_type if the channel has a primary type
            acc_df = endaq.ide.get_primary_sensor_data(
                doc=doc.channels[acc_id],
                time_mode="datetime"
            )
        except Exception as e:
            load_errors.append(f"Generic primary sensor load attempt failed: {e}")

    # Attempt 4: As a final fallback, try get_channel_data if available
    if acc_df is None:
        try:
            # If your installed endaq version has get_channel_data, uncomment the next lines:
            # acc_df = endaq.ide.get_channel_data(
            #     doc.channels[acc_id],
            #     time_mode="datetime"
            # )
            pass
        except Exception as e:
            load_errors.append(f"get_channel_data attempt failed: {e}")

    if acc_df is None or acc_df.empty:
        st.error(f"Failed to load data for channel {acc_id}. " +
                 (" | ".join(load_errors) if load_errors else "No detail available."))
        st.stop()

    # Work on a copy
    df = acc_df.copy()

    # Slice
    end_idx_safe = min(end_idx, len(df))
    start_idx_safe = min(start_idx, end_idx_safe)
    df = df.iloc[start_idx_safe:end_idx_safe]

    # Resample if datetime index
    if resample_rate != "None":
        if isinstance(df.index, pd.DatetimeIndex):
            try:
                df = df.resample(resample_rate).mean()
            except Exception as e:
                st.warning(f"Resampling failed: {e}")
        else:
            st.warning("Resampling requires a DatetimeIndex; skipping resample.")

    # Detect axis columns (or fallback to numeric columns)
    axes = detect_acc_axes(df)
    if DEBUG:
        st.write("🔎 Debug: Detected axes:", axes)

    if len(axes) == 0:
        st.warning("No numeric axis columns detected. Showing numeric columns preview.")
        st.dataframe(df.select_dtypes(include=[np.number]).head())

    # ================== TABS ==================
    tab1, tab2, tab3, tab4 = st.tabs(["Time", "Stats", "FFT/PSD", "Anomaly"])

    # -------- TIME --------
    with tab1:
        if len(axes) == 0:
            st.info("No axis columns detected to plot.")
        else:
            filt = df.copy()
            for c in axes:
                filt[c] = filt[c].rolling(ma_window, min_periods=1).mean()

            fig = go.Figure()
            for c in axes:
                fig.add_trace(go.Scatter(x=df.index, y=df[c], name=f"{c} Raw"))
                fig.add_trace(go.Scatter(x=filt.index, y=filt[c], name=f"{c} MA"))

            fig.update_layout(title=f"Channel {acc_id} — Raw vs Moving Average",
                              xaxis_title="Time", yaxis_title="Value")
            st.plotly_chart(fig, use_container_width=True)

    # -------- STATS --------
    with tab2:
        st.subheader("Descriptive Statistics")
        st.dataframe(df.select_dtypes(include=[np.number]).describe())

        st.subheader("Histograms")
        plot_histograms(df, axes)

    # -------- FFT + PSD --------
    with tab3:
        if len(axes) == 0:
            st.info("No axis columns detected to compute FFT/PSD.")
        else:
            # FFT + peak detection
            fig = go.Figure()
            peak_table = []

            for c in axes:
                f, vals = apply_fft_advanced(df[c].values)
                mask = (f >= fmin) & (f <= fmax)
                f_sel, vals_sel = f[mask], vals[mask]
                peaks = detect_peaks(f_sel, vals_sel)

                fig.add_trace(go.Scatter(x=f_sel, y=vals_sel, name=str(c)))
                fig.add_trace(go.Scatter(x=f_sel[peaks], y=vals_sel[peaks],
                                         mode="markers", marker=dict(color="red"),
                                         name=f"{c} Peaks"))

                for p in peaks:
                    peak_table.append({"Axis": str(c), "Freq": float(f_sel[p]), "Mag": float(vals_sel[p])})

            fig.update_layout(title=f"Channel {acc_id} — FFT Spectrum with Peaks",
                              xaxis_title="Frequency (Hz)", yaxis_title="Magnitude")
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(pd.DataFrame(peak_table))

            # PSD (log y-axis)
            fig2 = go.Figure()
            for c in axes:
                f_psd, Pxx = compute_psd(df[c].values)
                mask = (f_psd >= fmin) & (f_psd <= fmax)
                fig2.add_trace(go.Scatter(x=f_psd[mask], y=Pxx[mask], name=str(c)))

            fig2.update_layout(title=f"Channel {acc_id} — Power Spectral Density (Welch)",
                               xaxis_title="Frequency (Hz)", yaxis_title="PSD", yaxis_type="log")
            st.plotly_chart(fig2, use_container_width=True)

    # -------- ANOMALY --------
    with tab4:
        if len(axes) == 0:
            st.info("No axis columns detected to run anomaly detection.")
        else:
            model = IsolationForest(contamination=float(contamination), random_state=42)
            X = df[axes].select_dtypes(include=[np.number])
            df_an = df.copy()
            df_an["anomaly"] = model.fit_predict(X)

            fig = go.Figure()
            first_axis = axes[0]
            fig.add_trace(go.Scatter(x=df_an.index, y=df_an[first_axis], name=str(first_axis)))

            an = df_an[df_an["anomaly"] == -1]
            fig.add_trace(go.Scatter(x=an.index, y=an[first_axis],
                                     mode="markers", marker=dict(color="red"),
                                     name="Anomaly"))

            fig.update_layout(title=f"Channel {acc_id} — Anomaly Detection ({first_axis})",
                              xaxis_title="Time", yaxis_title="Value")
            st.plotly_chart(fig, use_container_width=True)
            st.write("Anomalies:", int((df_an["anomaly"] == -1).sum()))

else:
    st.info("Upload an IDE file to start")

else:
    st.info("Upload an IDE file to start")
