# ################################# Rev 1
# # ================== IMPORTS ==================
# import streamlit as st
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.graph_objects as go
# import tempfile
# import endaq

# from scipy.fft import fft
# from scipy.signal import welch, get_window, find_peaks
# from sklearn.ensemble import IsolationForest
# from sklearn.decomposition import PCA

# # Try importing PyWavelets
# try:
#     import pywt
#     wavelet_available = True
# except ModuleNotFoundError:
#     wavelet_available = False
#     st.warning("PyWavelets not installed → Wavelet analysis disabled")

# # ================== PAGE ==================
# st.set_page_config(layout="wide")
# st.title("enDAQ Vibration Signal Dashboard")

# uploaded_file = st.file_uploader("Upload IDE file", type=["ide"])

# # ================== FUNCTIONS ==================
# def estimate_fs(time_index):
#     dt = np.diff(time_index.astype('int64') / 1e9)
#     dt = dt[dt > 0]
#     if len(dt) == 0:
#         return None, True
#     median_dt = np.median(dt)
#     std_dt = np.std(dt)
#     fs_est = 1.0 / median_dt
#     irregular = (std_dt / median_dt) > 0.01
#     return fs_est, irregular

# def apply_fft(signal_data, fs):
#     n = len(signal_data)
#     if use_averaging or use_overlap:
#         step = int(window_size * (1 - overlap_factor)) if use_overlap else window_size
#         freqs = np.fft.fftfreq(window_size, 1/fs)[:window_size//2]
#         fft_all = []
#         for start in range(0, n - window_size + 1, step):
#             seg = signal_data[start:start+window_size]
#             if use_windowing:
#                 seg = seg * get_window(window_type, window_size)
#             fft_all.append(np.abs(fft(seg)[:window_size//2]))
#         fft_all = np.array(fft_all)
#         if use_averaging:
#             return freqs, np.mean(fft_all, axis=0)
#         else:
#             return freqs, fft_all[0]
#     else:
#         if use_windowing:
#             signal_data = signal_data * get_window(window_type, n)
#         fft_vals = np.abs(fft(signal_data))
#         freqs = np.fft.fftfreq(n, 1/fs)
#         return freqs[:n//2], fft_vals[:n//2]

# def compute_psd(signal_data, fs):
#     return welch(signal_data, fs=fs, nperseg=nperseg)

# def detect_peaks(freqs, values, height=0.0, distance=20, prominence=0.0):
#     peaks, _ = find_peaks(values, height=height, distance=distance, prominence=prominence)
#     return peaks

# def plot_histograms(df):
#     fig, ax = plt.subplots(1, 3, figsize=(15, 4))
#     for i, col in enumerate(["X (40g)", "Y (40g)", "Z (40g)"]):
#         sns.histplot(df[col], bins=50, kde=True, ax=ax[i])
#         ax[i].set_title(col)
#     st.pyplot(fig)

# def compute_rms(df):
#     return np.sqrt((df[["X (40g)", "Y (40g)", "Z (40g)"]]**2).mean(axis=1))

# # ML models
# def run_if(df, contamination):
#     model = IsolationForest(contamination=contamination, random_state=42)

#     X = df[["X (40g)", "Y (40g)", "Z (40g)"]]

#     model.fit(X)
#     scores = model.decision_function(X)

#     # -------- CHANGE 2: score-based threshold --------
#     threshold = np.percentile(scores, contamination * 100)

#     df["score"] = scores
#     df["anomaly"] = np.where(scores < threshold, -1, 1)

#     return df

# def run_pca(df, n_components, contamination):
#     X = df[["X (40g)", "Y (40g)", "Z (40g)"]]
#     pca = PCA(n_components=n_components)
#     X_pca = pca.fit_transform(X)
#     X_rec = pca.inverse_transform(X_pca)
#     error = np.mean((X - X_rec)**2, axis=1)
#     threshold = np.percentile(error, 100*(1-contamination))
#     df["anomaly"] = np.where(error > threshold, -1, 1)
#     df["score"] = error
#     explained = np.sum(pca.explained_variance_ratio_)
#     return df, explained

# # STFT
# def compute_stft(signal, fs, window_size, overlap):
#     step = int(window_size * (1 - overlap/100))
#     segments = []
#     for start in range(0, len(signal) - window_size, step):
#         seg = signal[start:start+window_size]
#         seg = seg * np.hanning(window_size)
#         fft_vals = np.abs(np.fft.fft(seg))[:window_size//2]
#         segments.append(fft_vals)
#     segments = np.array(segments)
#     freqs = np.fft.fftfreq(window_size, 1/fs)[:window_size//2]
#     times = np.arange(segments.shape[0]) * (step/fs)
#     return times, freqs, segments

# # CWT
# def compute_cwt(signal, fs, wavelet, max_scale):
#     if not wavelet_available:
#         return None, None
#     scales = np.arange(1, max_scale)
#     coeffs, freqs = pywt.cwt(signal, scales, wavelet, sampling_period=1/fs)
#     power = np.abs(coeffs)
#     return freqs, power

# # ================== MAIN ==================
# if uploaded_file:
#     uploaded_file.seek(0)
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".ide") as tmp:
#         tmp.write(uploaded_file.read())
#         path = tmp.name
#     doc = endaq.ide.get_doc(path)
#     acc_df = endaq.ide.get_primary_sensor_data(
#         doc=doc.channels[80],
#         measurement_type="acceleration",
#         time_mode="datetime"
#     )
#     acc_df = acc_df - acc_df.median()
#     df_full = acc_df.copy()
#     fs_auto, irregular = estimate_fs(df_full.index)
#     if irregular:
#         dt = 1 / fs_auto
#         df_full = df_full.resample(f"{int(dt*1000)}L").mean().interpolate()
#         fs_auto = 1.0 / dt

#     # ================= SIDEBAR =================
#     st.sidebar.header("File Info & Time Selection")
#     duration_sec = len(df_full) / fs_auto
#     st.sidebar.write(f"Total duration: {duration_sec:.2f} s")
#     st.sidebar.markdown("---")

#     start_time = st.sidebar.number_input("Start Time (s)", 0.0, duration_sec, 0.0)
#     end_time = st.sidebar.number_input("End Time (s)", 0.0, duration_sec, min(40.0, duration_sec))
#     if end_time - start_time > 40:
#         end_time = start_time + 40
#     st.sidebar.markdown("---")

#     df = df_full.iloc[int(start_time*fs_auto):int(end_time*fs_auto)]

#     # -------- FS --------
#     st.sidebar.header("Sampling Frequency")
#     st.sidebar.write(f"Auto fs: {fs_auto:.2f} Hz")
#     override_fs = st.sidebar.checkbox("Override FS")
#     manual_fs = st.sidebar.number_input("Manual fs", value=fs_auto)
#     fs = manual_fs if override_fs else fs_auto
#     st.sidebar.markdown("---")

#     # -------- MOVING AVG --------
#     st.sidebar.header("Moving Average")
#     ma_window = st.sidebar.slider("Window", 1, 50, 5)
#     st.sidebar.markdown("---")

#     # -------- FFT --------
#     st.sidebar.header("FFT Settings")
#     use_windowing = st.sidebar.checkbox("Windowing")
#     use_averaging = st.sidebar.checkbox("Averaging")
#     use_overlap = st.sidebar.checkbox("Overlap")
#     window_type = "hann"
#     window_size = 1024
#     overlap_factor = 0.5
#     if use_windowing:
#         window_type = st.sidebar.selectbox("Window", ["hann","hamming","blackman"])
#     if use_averaging or use_overlap:
#         window_size = st.sidebar.slider("Window Size", 128, 4096, 1024)
#     if use_overlap:
#         overlap_factor = st.sidebar.slider("Overlap", 0.1, 0.9, 0.5)
#     fmin = st.sidebar.number_input("Min Freq", value=0.0)
#     fmax = st.sidebar.number_input("Max Freq", value=50.0)
#     peak_height = st.sidebar.number_input("Peak Height", value=0.0)
#     peak_distance = st.sidebar.slider("Peak Distance", 1, 200, 20)
#     peak_prominence = st.sidebar.number_input("Peak Prominence", value=0.0)
#     st.sidebar.markdown("---")

#     # -------- PSD --------
#     st.sidebar.header("PSD")
#     nperseg = st.sidebar.slider("nperseg", 128, 4096, 1024)
#     st.sidebar.markdown("---")

#     # -------- TIME-FREQ --------
#     st.sidebar.header("Time-Frequency")
#     stft_window = st.sidebar.slider("STFT Window", 128, 4096, 512)
#     stft_overlap = st.sidebar.slider("STFT Overlap %", 0, 90, 50)
#     if wavelet_available:
#         wavelet_type = st.sidebar.selectbox("Wavelet", ["morl","cmor","mexh"])
#         max_scale = st.sidebar.slider("Max Scale", 10, 200, 100)
#     st.sidebar.markdown("---")

#     # -------- MODEL --------
#     st.sidebar.header("Anomaly Detection")
#     model_choice = st.sidebar.selectbox("Model", ["Isolation Forest","PCA"], index=1)
#     if model_choice == "Isolation Forest":
#         # -------- CHANGE 1: smaller contamination --------
#         contamination = st.sidebar.slider(
#             "Contamination",
#             min_value=0.001,
#             max_value=0.1,
#             value=0.001,
#             step=0.001,
#             format="%.3f"
#         )
#     else:
#         n_components = st.sidebar.slider("Components", 1, 3, 2)
#         contamination = st.sidebar.slider(
#             "Contamination",
#             min_value=0.001,
#             max_value=0.1,
#             value=0.001,
#             step=0.001,
#             format="%.3f"
#         )

#     # -------- AXIS SELECTION --------
#     axis_choice = st.sidebar.selectbox("Axis for Anomaly", ["X (40g)","Y (40g)","Z (40g)","All"])
#     run_button = st.sidebar.button("Run Model")
#     st.sidebar.markdown("---")

#     # ================= TABS =================
#     tab1, tab2, tab3, tab4, tab5 = st.tabs(["Time","Stats","FFT/PSD","Time-Frequency","Anomaly"])

#     # -------- ANOMALY --------
#     with tab5:
#         if run_button:
#             df_an = df.copy()

#             # -------- CHANGE 3: RMS deadband --------
#             rms = compute_rms(df_an)
#             rms_threshold = np.percentile(rms, 10)
#             df_an = df_an[rms > rms_threshold]

#             axes_to_plot = ["X (40g)","Y (40g)","Z (40g)"] if axis_choice=="All" else [axis_choice]

#             if model_choice == "Isolation Forest":
#                 df_an = run_if(df_an, contamination)
#             else:
#                 df_an, explained = run_pca(df_an, n_components, contamination)
#                 st.write(f"Explained variance: {explained:.3f}")

#             anomalies = df_an[df_an["anomaly"]==-1]

#             fig = go.Figure()
#             for ax in axes_to_plot:
#                 fig.add_trace(go.Scatter(x=df_an.index, y=df_an[ax], name=ax))
#                 fig.add_trace(go.Scatter(
#                     x=anomalies.index,
#                     y=anomalies[ax],
#                     mode="markers",
#                     marker=dict(color="red"),
#                     name=f"{ax} Anomaly"
#                 ))
#             st.plotly_chart(fig, width="stretch")
#             st.write("Anomalies detected:", len(anomalies))
#         else:
#             st.info("Run model")
# else:
#     st.info("Upload file to start")

# ################################# Rev 2
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
from sklearn.decomposition import PCA

# Try importing PyWavelets
try:
    import pywt
    wavelet_available = True
except ModuleNotFoundError:
    wavelet_available = False
    st.warning("PyWavelets not installed → Wavelet analysis disabled")

# ================== PAGE ==================
st.set_page_config(layout="wide")
st.title("enDAQ Vibration Signal Dashboard")

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

def detect_peaks(freqs, values, height=0.0, distance=20, prominence=0.0):
    peaks, _ = find_peaks(values, height=height, distance=distance, prominence=prominence)
    return peaks

def plot_histograms(df):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for i, col in enumerate(["X (40g)", "Y (40g)", "Z (40g)"]):
        sns.histplot(df[col], bins=50, kde=True, ax=ax[i])
        ax[i].set_title(col)
    st.pyplot(fig)

def compute_rms(df):
    return np.sqrt((df[["X (40g)", "Y (40g)", "Z (40g)"]]**2).mean(axis=1))

# ML models
def run_if(df, contamination):
    model = IsolationForest(contamination=contamination, random_state=42)

    X = df[["X (40g)", "Y (40g)", "Z (40g)"]]

    model.fit(X)
    scores = model.decision_function(X)

    # -------- CHANGE 2: score-based threshold --------
    threshold = np.percentile(scores, contamination * 100)

    df["score"] = scores
    df["anomaly"] = np.where(scores < threshold, -1, 1)

    return df

def run_pca(df, n_components, contamination):
    X = df[["X (40g)", "Y (40g)", "Z (40g)"]]
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    X_rec = pca.inverse_transform(X_pca)
    error = np.mean((X - X_rec)**2, axis=1)
    threshold = np.percentile(error, 100*(1-contamination))
    df["anomaly"] = np.where(error > threshold, -1, 1)
    df["score"] = error
    explained = np.sum(pca.explained_variance_ratio_)
    return df, explained

# STFT
def compute_stft(signal, fs, window_size, overlap):
    step = int(window_size * (1 - overlap/100))
    segments = []
    for start in range(0, len(signal) - window_size, step):
        seg = signal[start:start+window_size]
        seg = seg * np.hanning(window_size)
        fft_vals = np.abs(np.fft.fft(seg))[:window_size//2]
        segments.append(fft_vals)
    segments = np.array(segments)
    freqs = np.fft.fftfreq(window_size, 1/fs)[:window_size//2]
    times = np.arange(segments.shape[0]) * (step/fs)
    return times, freqs, segments

# CWT
def compute_cwt(signal, fs, wavelet, max_scale):
    if not wavelet_available:
        return None, None
    scales = np.arange(1, max_scale)
    coeffs, freqs = pywt.cwt(signal, scales, wavelet, sampling_period=1/fs)
    power = np.abs(coeffs)
    return freqs, power

# ================== MAIN ==================
if uploaded_file:
    uploaded_file.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ide") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name
    doc = endaq.ide.get_doc(path)
    acc_df = endaq.ide.get_primary_sensor_data(
        doc=doc.channels[80],
        measurement_type="acceleration",
        time_mode="datetime"
    )
    acc_df = acc_df - acc_df.median()
    df_full = acc_df.copy()
    fs_auto, irregular = estimate_fs(df_full.index)
    if irregular:
        dt = 1 / fs_auto
        df_full = df_full.resample(f"{int(dt*1000)}L").mean().interpolate()
        fs_auto = 1.0 / dt

    # ================= SIDEBAR =================
    st.sidebar.header("File Info & Time Selection")
    duration_sec = len(df_full) / fs_auto
    st.sidebar.write(f"Total duration: {duration_sec:.2f} s")
    st.sidebar.markdown("---")

    start_time = st.sidebar.number_input("Start Time (s)", 0.0, duration_sec, 0.0)
    end_time = st.sidebar.number_input("End Time (s)", 0.0, duration_sec, min(40.0, duration_sec))
    if end_time - start_time > 40:
        end_time = start_time + 40
    st.sidebar.markdown("---")

    df = df_full.iloc[int(start_time*fs_auto):int(end_time*fs_auto)]

    # -------- FS --------
    st.sidebar.header("Sampling Frequency")
    st.sidebar.write(f"Auto fs: {fs_auto:.2f} Hz")
    override_fs = st.sidebar.checkbox("Override FS")
    manual_fs = st.sidebar.number_input("Manual fs", value=fs_auto)
    fs = manual_fs if override_fs else fs_auto
    st.sidebar.markdown("---")

    # -------- MOVING AVG --------
    st.sidebar.header("Moving Average")
    ma_window = st.sidebar.slider("Window", 1, 50, 5)
    st.sidebar.markdown("---")

    # -------- FFT --------
    st.sidebar.header("FFT Settings")
    use_windowing = st.sidebar.checkbox("Windowing")
    use_averaging = st.sidebar.checkbox("Averaging")
    use_overlap = st.sidebar.checkbox("Overlap")
    window_type = "hann"
    window_size = 1024
    overlap_factor = 0.5
    if use_windowing:
        window_type = st.sidebar.selectbox("Window", ["hann","hamming","blackman"])
    if use_averaging or use_overlap:
        window_size = st.sidebar.slider("Window Size", 128, 4096, 1024)
    if use_overlap:
        overlap_factor = st.sidebar.slider("Overlap", 0.1, 0.9, 0.5)
    fmin = st.sidebar.number_input("Min Freq", value=0.0)
    fmax = st.sidebar.number_input("Max Freq", value=50.0)
    peak_height = st.sidebar.number_input("Peak Height", value=0.0)
    peak_distance = st.sidebar.slider("Peak Distance", 1, 200, 20)
    peak_prominence = st.sidebar.number_input("Peak Prominence", value=0.0)
    st.sidebar.markdown("---")

    # -------- PSD --------
    st.sidebar.header("PSD")
    nperseg = st.sidebar.slider("nperseg", 128, 4096, 1024)
    st.sidebar.markdown("---")

    # -------- TIME-FREQ --------
    st.sidebar.header("Time-Frequency")
    stft_window = st.sidebar.slider("STFT Window", 128, 4096, 512)
    stft_overlap = st.sidebar.slider("STFT Overlap %", 0, 90, 50)
    if wavelet_available:
        wavelet_type = st.sidebar.selectbox("Wavelet", ["morl","cmor","mexh"])
        max_scale = st.sidebar.slider("Max Scale", 10, 200, 100)
    st.sidebar.markdown("---")

    # -------- MODEL --------
    st.sidebar.header("Anomaly Detection")
    model_choice = st.sidebar.selectbox("Model", ["Isolation Forest","PCA"], index=1)
    if model_choice == "Isolation Forest":
        # -------- CHANGE 1: smaller contamination --------
        contamination = st.sidebar.slider(
            "Contamination",
            min_value=0.001,
            max_value=0.1,
            value=0.001,
            step=0.001,
            format="%.3f"
        )
    else:
        n_components = st.sidebar.slider("Components", 1, 3, 2)
        contamination = st.sidebar.slider(
            "Contamination",
            min_value=0.001,
            max_value=0.1,
            value=0.001,
            step=0.001,
            format="%.3f"
        )

    # -------- RMS FILTER --------
    st.sidebar.header("RMS Deadband")
    rms_percentile = st.sidebar.slider(
        "RMS Threshold Percentile",
        min_value=0,
        max_value=50,
        value=10,
        step=1
    )
    st.sidebar.markdown("---")

    # -------- AXIS SELECTION --------
    axis_choice = st.sidebar.selectbox("Axis for Anomaly", ["X (40g)","Y (40g)","Z (40g)","All"])
    run_button = st.sidebar.button("Run Model")
    st.sidebar.markdown("---")

    # ================= TABS =================
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Time","Stats","FFT/PSD","Time-Frequency","Anomaly"])

    # -------- ANOMALY --------
    with tab5:
        if run_button:
            df_an = df.copy()

            # -------- CHANGE 3: RMS deadband --------
            rms = compute_rms(df_an)
            # rms_threshold = np.percentile(rms, 10)
            rms_threshold = np.percentile(rms, rms_percentile)
            df_an = df_an[rms > rms_threshold]

            axes_to_plot = ["X (40g)","Y (40g)","Z (40g)"] if axis_choice=="All" else [axis_choice]

            if model_choice == "Isolation Forest":
                df_an = run_if(df_an, contamination)
            else:
                df_an, explained = run_pca(df_an, n_components, contamination)
                st.write(f"Explained variance: {explained:.3f}")

            anomalies = df_an[df_an["anomaly"]==-1]

            fig = go.Figure()
            for ax in axes_to_plot:
                fig.add_trace(go.Scatter(x=df_an.index, y=df_an[ax], name=ax))
                fig.add_trace(go.Scatter(
                    x=anomalies.index,
                    y=anomalies[ax],
                    mode="markers",
                    marker=dict(color="red"),
                    name=f"{ax} Anomaly"
                ))
            st.plotly_chart(fig, width="stretch")
            st.write("Anomalies detected:", len(anomalies))
        else:
            st.info("Run model")
else:
    st.info("Upload file to start")
