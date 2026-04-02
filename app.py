
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
st.sidebar.header("General Settings")

start_idx = st.sidebar.slider("Start Index", 0, 200000, 50000)
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

# ================== FUNCTIONS ==================

def apply_fft(signal_data):
    n = len(signal_data)

    if use_averaging or use_overlap:
        step = int(window_size * (1 - overlap_factor)) if use_overlap else window_size
        freqs = np.fft.fftfreq(window_size, 1/fs)[:window_size//2]
        fft_all = []

        for start in range(0, n - window_size + 1, step):
            seg = signal_data[start:start+window_size]

            if use_windowing:
                seg *= get_window(window_type, window_size)

            fft_all.append(np.abs(fft(seg)[:window_size//2]))

        fft_all = np.array(fft_all)

        if use_averaging:
            return freqs, np.mean(fft_all, axis=0)
        else:
            return freqs, fft_all[0]

    else:
        if use_windowing:
            signal_data *= get_window(window_type, n)

        fft_vals = np.abs(fft(signal_data))
        freqs = np.fft.fftfreq(n, 1/fs)

        return freqs[:n//2], fft_vals[:n//2]


def compute_psd(signal_data):
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

    # ---- Save file ----
    uploaded_file.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ide") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    doc = endaq.ide.get_doc(path)

    # ---- DEBUG (optional) ----
    # st.write(endaq.ide.get_channel_table(doc))

    # ================== LOAD CHANNEL 80 ==================
    try:
        acc_df = endaq.ide.get_primary_sensor_data(
            doc=doc.channels[80],
            measurement_type="acceleration",
            time_mode="datetime"
        )
    except Exception as e:
        st.error(f"❌ Channel 80 not found: {e}")
        st.stop()

    # ---- Optional temperature ----
    try:
        temp_df = endaq.ide.get_primary_sensor_data(
            doc=doc.channels[59],
            measurement_type="temperature",
            time_mode="datetime"
        )

        temp_resampled = temp_df.reindex(acc_df.index, method='nearest')
        df = acc_df.copy()
        df["Temperature"] = temp_resampled.iloc[:, 0]

    except:
        st.warning("⚠️ No temperature channel found")
        df = acc_df.copy()

    # ---- Select data ----
    df = df.iloc[start_idx:end_idx]

    if resample_rate != "None":
        df = df.resample(resample_rate).mean()

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
            f, vals = apply_fft(df[c].values)
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

        # PSD
        fig2 = go.Figure()
        for c in ["X (40g)", "Y (40g)", "Z (40g)"]:
            f, Pxx = compute_psd(df[c].values)
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
