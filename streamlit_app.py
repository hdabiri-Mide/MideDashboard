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
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed

# ================== PAGE CONFIG ==================
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
st.sidebar.subheader("Isolation Forest")
contamination = st.sidebar.slider("Contamination", 0.001, 0.1, 0.01)

# -------- LSTM PARAMETERS --------
st.sidebar.subheader("LSTM Autoencoder")
epochs_u = st.sidebar.slider("Epochs", 10, 100, 80)
batch_size_u = st.sidebar.slider("Batch Size", 8, 128, 64)
lstm_units = st.sidebar.slider("LSTM Units", 10, 200, 50)
selected_columns = ['X (40g)']

# ================== FILE UPLOAD ==================
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

def run_if_anomaly(df):
    model = IsolationForest(contamination=contamination, random_state=42)
    df["anomaly"] = model.fit_predict(df[["X (40g)", "Y (40g)", "Z (40g)"]])
    return df

def run_lstm_anomaly(df, selected_columns, epochs=80, batch_size=64, lstm_units=50):
    data = df[selected_columns].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    seq_len = 50
    X = []
    for i in range(len(data_scaled) - seq_len):
        X.append(data_scaled[i:i+seq_len])
    X = np.array(X)
    model = Sequential([
        LSTM(lstm_units, activation='relu', input_shape=(X.shape[1], X.shape[2]), return_sequences=False),
        RepeatVector(X.shape[1]),
        LSTM(lstm_units, activation='relu', return_sequences=True),
        TimeDistributed(Dense(X.shape[2]))
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, X, epochs=epochs, batch_size=batch_size, verbose=0)
    X_pred = model.predict(X, verbose=0)
    mse = np.mean(np.power(X - X_pred, 2), axis=(1,2))
    threshold = np.mean(mse) + 3*np.std(mse)
    anomalies = mse > threshold
    anomaly_index = df.index[seq_len:]
    result_df = pd.DataFrame({"reconstruction_error": mse, "anomaly": anomalies}, index=anomaly_index)
    return result_df, threshold

# ================== MAIN ==================
if uploaded_file:
    uploaded_file.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ide") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    doc = endaq.ide.get_doc(path)

    try:
        acc_df = endaq.ide.get_primary_sensor_data(
            doc=doc.channels[80],
            measurement_type="acceleration",
            time_mode="datetime"
        )
    except Exception as e:
        st.error(f"❌ Channel 80 not found: {e}")
        st.stop()

    try:
        temp_df = endaq.ide.get_primary_sensor_data(
            doc=doc.channels[59],
            measurement_type="temperature",
            time_mode="datetime"
        )
        temp_resampled = temp_df.reindex(acc_df.index, method='nearest')
        df = acc_df.copy()
        df["Temperature"] = temp_resampled.iloc[:,0]
    except:
        st.warning("⚠️ No temperature channel found")
        df = acc_df.copy()

    df = df.iloc[start_idx:end_idx]
    if resample_rate != "None":
        df = df.resample(resample_rate).mean()

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
        st.subheader("Isolation Forest")
        df_if = run_if_anomaly(df.copy())
        fig_if = go.Figure()
        fig_if.add_trace(go.Scatter(x=df_if.index, y=df_if["X (40g)"], name="X"))
        an_if = df_if[df_if["anomaly"]==-1]
        fig_if.add_trace(go.Scatter(x=an_if.index, y=an_if["X (40g)"],
                                    mode="markers",
                                    marker=dict(color="red"),
                                    name="IF Anomaly"))
        st.plotly_chart(fig_if, use_container_width=True)
        st.write("IF Anomalies:", len(an_if))

        # ---- LSTM Autoencoder ----
        st.subheader("LSTM Autoencoder")
        lstm_result, threshold = run_lstm_anomaly(df, selected_columns, epochs_u, batch_size_u, lstm_units)
        fig_lstm = go.Figure()
        fig_lstm.add_trace(go.Scatter(x=lstm_result.index, y=lstm_result["reconstruction_error"], name="Reconstruction Error"))
        fig_lstm.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Threshold")
        st.plotly_chart(fig_lstm, use_container_width=True)

        fig_signal = go.Figure()
        fig_signal.add_trace(go.Scatter(x=df.index, y=df["X (40g)"], name="Signal"))
        lstm_an = lstm_result[lstm_result["anomaly"]]
        fig_signal.add_trace(go.Scatter(x=lstm_an.index,
                                        y=df.loc[lstm_an.index, "X (40g)"],
                                        mode="markers",
                                        marker=dict(color="red"),
                                        name="LSTM Anomaly"))
        st.plotly_chart(fig_signal, use_container_width=True)
        st.write("LSTM Anomalies:", len(lstm_an))

else:
    st.info("Upload an IDE file to start")