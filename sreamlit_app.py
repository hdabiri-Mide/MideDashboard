
# # ################################# Rev 1
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
#     # st.warning("PyWavelets not installed → Wavelet analysis disabled")

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

# def run_if(df, contamination):
#     model = IsolationForest(contamination=contamination, random_state=42)
#     X = df[["X (40g)", "Y (40g)", "Z (40g)"]]
#     model.fit(X)
#     scores = model.decision_function(X)
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

#     start_time = st.sidebar.number_input("Start Time (s)", 0.0, duration_sec, 0.0)
#     end_time = st.sidebar.number_input("End Time (s)", 0.0, duration_sec, min(40.0, duration_sec))
#     if end_time - start_time > 40:
#         end_time = start_time + 40

#     df = df_full.iloc[int(start_time*fs_auto):int(end_time*fs_auto)]

#     st.sidebar.header("Sampling Frequency")
#     override_fs = st.sidebar.checkbox("Override FS")
#     manual_fs = st.sidebar.number_input("Manual fs", value=fs_auto)
#     fs = manual_fs if override_fs else fs_auto

#     ma_window = st.sidebar.slider("Window", 1, 50, 5)

#     st.sidebar.header("FFT Settings")
#     use_windowing = st.sidebar.checkbox("Windowing")
#     use_averaging = st.sidebar.checkbox("Averaging")
#     use_overlap = st.sidebar.checkbox("Overlap")
#     window_type = "hann"
#     window_size = 1024
#     overlap_factor = 0.5

#     fmin = st.sidebar.number_input("Min Freq", value=0.0)
#     fmax = st.sidebar.number_input("Max Freq", value=50.0)

#     st.sidebar.header("FFT Peak Detection")
#     peak_height = st.sidebar.number_input("Peak Height", value=0.0)
#     peak_distance = st.sidebar.slider("Peak Distance", 1, 200, 20)
#     peak_prominence = st.sidebar.number_input("Peak Prominence", value=0.0)

#     st.sidebar.header("PSD")
#     nperseg = st.sidebar.slider("nperseg", 128, 4096, 1024)

#     st.sidebar.header("Time-Frequency")
#     stft_window = st.sidebar.slider("STFT Window", 128, 4096, 512)
#     stft_overlap = st.sidebar.slider("STFT Overlap %", 0, 90, 50)

#     st.sidebar.header("Anomaly Detection")
#     model_choice = st.sidebar.selectbox("Model", ["Isolation Forest","PCA"], index=1)

#     contamination = st.sidebar.slider(
#         "Contamination",
#         min_value=0.001,
#         max_value=0.1,
#         value=0.001,
#         step=0.001,
#         format="%.3f"
#     )

#     rms_percentile = st.sidebar.slider("RMS Threshold Percentile", 0, 80, 10)

#     axis_choice = st.sidebar.selectbox("Axis for Anomaly", ["X (40g)","Y (40g)","Z (40g)","All"])
#     run_button = st.sidebar.button("Run Model")

#     # ================= TABS =================
#     tab1, tab2, tab3, tab4, tab5 = st.tabs(["Time","Stats","FFT/PSD","Time-Frequency","Anomaly"])

#     # ================= TAB 1: TIME =================
#     with tab1:
#         fig = go.Figure()
#         for col in ["X (40g)", "Y (40g)", "Z (40g)"]:
#             fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col))
#         fig.update_layout(title="Time Series", height=500)
#         st.plotly_chart(fig, use_container_width=True)

#     # ================= TAB 2: STATS =================
#     with tab2:
#         # -------- Histograms --------
#         fig = go.Figure()
#         for col in ["X (40g)", "Y (40g)", "Z (40g)"]:
#             fig.add_trace(go.Histogram(x=df[col], name=col, opacity=0.6))
    
#         fig.update_layout(barmode="overlay", title="Histograms")
#         st.plotly_chart(fig, use_container_width=True)
    
#         # -------- STATISTICAL METRICS TABLE --------
#         stats = pd.DataFrame({
#             "Axis": ["X (40g)", "Y (40g)", "Z (40g)"],
#             "Mean": [
#                 df["X (40g)"].mean(),
#                 df["Y (40g)"].mean(),
#                 df["Z (40g)"].mean()
#             ],
#             "Std": [
#                 df["X (40g)"].std(),
#                 df["Y (40g)"].std(),
#                 df["Z (40g)"].std()
#             ],
#             "RMS": [
#                 np.sqrt(np.mean(df["X (40g)"]**2)),
#                 np.sqrt(np.mean(df["Y (40g)"]**2)),
#                 np.sqrt(np.mean(df["Z (40g)"]**2))
#             ],
#             "Min": [
#                 df["X (40g)"].min(),
#                 df["Y (40g)"].min(),
#                 df["Z (40g)"].min()
#             ],
#             "Max": [
#                 df["X (40g)"].max(),
#                 df["Y (40g)"].max(),
#                 df["Z (40g)"].max()
#             ],
#             "Peak-to-Peak": [
#                 df["X (40g)"].max() - df["X (40g)"].min(),
#                 df["Y (40g)"].max() - df["Y (40g)"].min(),
#                 df["Z (40g)"].max() - df["Z (40g)"].min()
#             ]
#         })
    
#         st.subheader("Statistical Metrics")
#         st.dataframe(stats, use_container_width=True)
    
#     # ================= TAB 3: FFT / PSD =================
#     with tab3:
        
#         # -------- FFT (3-axis) --------
#         fig = go.Figure()

#         for col in ["X (40g)", "Y (40g)", "Z (40g)"]:
#             sig = df[col].values
#             freqs, fft_vals = apply_fft(sig, fs)

#             # ---- FFT line ----
#             fig.add_trace(go.Scatter(
#                 x=freqs,
#                 y=fft_vals,
#                 mode="lines",
#                 name=f"FFT {col}"
#             ))

#             # ---- PEAK DETECTION (NEW) ----
#             peaks = detect_peaks(
#                 freqs,
#                 fft_vals,
#                 height=peak_height,
#                 distance=peak_distance,
#                 prominence=peak_prominence
#             )

#             fig.add_trace(go.Scatter(
#                 x=freqs[peaks],
#                 y=fft_vals[peaks],
#                 mode="markers",
#                 marker=dict(color="red", size=8),
#                 name=f"Peaks {col}"
#             ))
            
#         st.plotly_chart(fig, use_container_width=True)   
        
#         # -------- PSD (3-axis) --------
#         fig2 = go.Figure()
    
#         for col in ["X (40g)", "Y (40g)", "Z (40g)"]:
#             sig = df[col].values
#             psd_freq, psd_vals = compute_psd(sig, fs)
    
#             fig2.add_trace(go.Scatter(
#                 x=psd_freq,
#                 y=psd_vals,
#                 mode="lines",
#                 name=f"PSD {col}"
#             ))
    
#         fig2.update_layout(
#             title="Power Spectral Density (X, Y, Z)",
#             xaxis_title="Frequency (Hz)",
#             yaxis_title="Power",
#             height=500
#         )
    
#         st.plotly_chart(fig2, use_container_width=True)
#     # ================= TAB 4: TIME-FREQUENCY =================
#     with tab4:
#         sig = df["X (40g)"].values
#         times, freqs, stft = compute_stft(sig, fs, stft_window, stft_overlap)

#         fig = go.Figure(data=go.Heatmap(
#             z=stft.T,
#             x=times,
#             y=freqs
#         ))
#         fig.update_layout(title="STFT Spectrogram")
#         st.plotly_chart(fig, use_container_width=True)

#     # ================= TAB 5: ANOMALY =================
#     with tab5:
#         if run_button:
#             df_an = df.copy()
#             rms = compute_rms(df_an)
#             rms_threshold = np.percentile(rms, rms_percentile)
#             df_an = df_an[rms > rms_threshold]

#             axes_to_plot = ["X (40g)","Y (40g)","Z (40g)"] if axis_choice=="All" else [axis_choice]

#             if model_choice == "Isolation Forest":
#                 df_an = run_if(df_an, contamination)
#             else:
#                 df_an, explained = run_pca(df_an, 2, contamination)
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
#                     name=f"{ax} anomaly"
#                 ))

#             st.plotly_chart(fig, use_container_width=True)
#             st.write("Anomalies detected:", len(anomalies))
#         else:
#             st.info("Run model")
# else:
#     st.info("Upload file to start")

# # ################################# Rev 2
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

try:
    import pywt
    wavelet_available = True
except ModuleNotFoundError:
    wavelet_available = False

# ================== PAGE CONFIG ==================
st.set_page_config(
    layout="wide",
    page_title="enDAQ Dashboard",
    page_icon="📡"
)

# ================== THEME ==================
ACCENT   = "#00C9A7"   # teal-green
ACCENT2  = "#845EF7"   # soft violet for secondary traces
BG_CARD  = "#1A1D2E"
BG_DARK  = "#0F1117"
TEXT     = "#E8ECF4"
MUTED    = "#6B7280"
RED      = "#FF6B6B"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="'DM Mono', monospace", color=TEXT, size=12),
    xaxis=dict(gridcolor="#1F2437", linecolor="#2A2F45", zerolinecolor="#2A2F45",
               tickfont=dict(color=TEXT), title_font=dict(color=TEXT)),
    yaxis=dict(gridcolor="#1F2437", linecolor="#2A2F45", zerolinecolor="#2A2F45",
               tickfont=dict(color=TEXT), title_font=dict(color=TEXT)),
    legend=dict(
        bgcolor="#1A1D2E",
        bordercolor="#2A2F45",
        borderwidth=1,
        font=dict(color=TEXT, size=11),
    ),
    title_font=dict(color=TEXT),
    margin=dict(l=12, r=12, t=40, b=12),
    height=420,
)

TRACE_COLORS = [ACCENT, ACCENT2, "#FFD43B"]

# ================== GLOBAL CSS ==================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@600;700;800&display=swap');

/* ---- Root & body ---- */
html, body, [data-testid="stAppViewContainer"] {{
    background: {BG_DARK};
    color: {TEXT};
    font-family: 'DM Mono', monospace;
}}

[data-testid="stAppViewContainer"] > .main {{
    background: {BG_DARK};
}}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {{
    background: {BG_CARD} !important;
    border-right: 1px solid #1F2437;
}}
[data-testid="stSidebar"] * {{
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
    color: {TEXT} !important;
}}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{
    color: {ACCENT} !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase;
    border-bottom: 1px solid #1F2437;
    padding-bottom: 4px;
    margin-top: 1rem !important;
}}

/* ---- Sliders & inputs ---- */
[data-testid="stSlider"] > div > div > div > div {{
    background: {ACCENT} !important;
}}
[data-baseweb="input"] input,
[data-baseweb="select"] div {{
    background: #0F1117 !important;
    border-color: #2A2F45 !important;
    color: {TEXT} !important;
}}

/* ---- Buttons ---- */
[data-testid="stSidebar"] button,
.stButton > button {{
    background: {ACCENT} !important;
    color: #0F1117 !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    padding: 0.45rem 1.2rem !important;
    transition: opacity 0.15s ease !important;
    width: 100%;
}}
.stButton > button:hover {{
    opacity: 0.85 !important;
}}

/* ---- File uploader ---- */
[data-testid="stFileUploadDropzone"] {{
    background: {BG_CARD} !important;
    border: 1.5px dashed {ACCENT} !important;
    border-radius: 10px !important;
    color: {MUTED} !important;
}}

/* ---- Tabs ---- */
[data-baseweb="tab-list"] {{
    background: {BG_CARD} !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 2px;
    border: 1px solid #1F2437;
}}
[data-baseweb="tab"] {{
    background: transparent !important;
    color: {MUTED} !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.04em;
    border-radius: 7px !important;
    padding: 6px 16px !important;
    border: none !important;
    transition: all 0.15s ease;
}}
[aria-selected="true"][data-baseweb="tab"] {{
    background: {ACCENT} !important;
    color: #0F1117 !important;
    font-weight: 700 !important;
}}

/* ---- Dataframe / table ---- */
[data-testid="stDataFrame"] {{
    border: 1px solid #1F2437;
    border-radius: 8px;
    overflow: hidden;
}}
/* Force dark background + light text on ALL dataframe cells */
[data-testid="stDataFrame"] iframe {{
    color-scheme: dark;
}}
.dvn-scroller {{
    background: {BG_CARD} !important;
}}
/* Glide data-grid (Streamlit's internal table renderer) */
.gdg-cell, .gdg-header-cell, .dvn-scroller * {{
    background: {BG_CARD} !important;
    color: {TEXT} !important;
}}

/* ---- Tooltip (?) badge ---- */
.param-row {{
    display: flex;
    align-items: center;
    gap: 5px;
    margin-bottom: 2px;
}}
.param-label {{
    font-size: 0.82rem;
    color: {TEXT};
}}
.tooltip-wrap {{
    position: relative;
    display: inline-flex;
    align-items: center;
}}
.tooltip-icon {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: #2A2F45;
    color: {MUTED};
    font-size: 0.6rem;
    font-weight: 700;
    cursor: default;
    line-height: 1;
    border: 1px solid #3A3F55;
    flex-shrink: 0;
}}
.tooltip-icon:hover + .tooltip-box,
.tooltip-wrap:hover .tooltip-box {{
    opacity: 1;
    pointer-events: auto;
    transform: translateY(0px);
}}
.tooltip-box {{
    opacity: 0;
    pointer-events: none;
    position: absolute;
    left: 20px;
    top: -4px;
    z-index: 9999;
    background: #2A2F45;
    color: {TEXT};
    font-size: 0.72rem;
    line-height: 1.45;
    padding: 7px 10px;
    border-radius: 6px;
    width: 200px;
    border: 1px solid #3A3F55;
    box-shadow: 0 4px 16px rgba(0,0,0,0.5);
    transform: translateY(4px);
    transition: opacity 0.15s ease, transform 0.15s ease;
}}

/* ---- Info / warning boxes ---- */
[data-testid="stAlert"] {{
    background: {BG_CARD} !important;
    border-left: 3px solid {ACCENT} !important;
    border-radius: 6px !important;
    color: {TEXT} !important;
}}

/* ---- Metric cards ---- */
.metric-card {{
    background: {BG_CARD};
    border: 1px solid #1F2437;
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
}}
.metric-card .label {{
    font-size: 0.68rem;
    color: {MUTED};
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 4px;
}}
.metric-card .value {{
    font-size: 1.35rem;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    color: {ACCENT};
}}
.metric-card .unit {{
    font-size: 0.72rem;
    color: {MUTED};
}}

/* ---- Section headings ---- */
.section-label {{
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {MUTED};
    margin-bottom: 8px;
    padding-bottom: 4px;
    border-bottom: 1px solid #1F2437;
}}

/* ---- Main title ---- */
h1 {{
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    color: {TEXT} !important;
    letter-spacing: -0.02em !important;
}}
</style>
""", unsafe_allow_html=True)


# ================== HELPERS ==================
def metric_card(label, value, unit=""):
    return f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        <div class="unit">{unit}</div>
    </div>"""

def section_label(text):
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)

def tip(label, tooltip):
    """Render a label with a hoverable ? tooltip badge."""
    st.markdown(f"""
    <div class="param-row">
        <span class="param-label">{label}</span>
        <span class="tooltip-wrap">
            <span class="tooltip-icon">?</span>
            <span class="tooltip-box">{tooltip}</span>
        </span>
    </div>""", unsafe_allow_html=True)

def plotly_chart(fig, **kwargs):
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, **kwargs)


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
        return freqs, np.mean(fft_all, axis=0) if use_averaging else fft_all[0]
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

def compute_rms(df):
    return np.sqrt((df[["X (40g)", "Y (40g)", "Z (40g)"]]**2).mean(axis=1))

def run_if(df, contamination):
    model = IsolationForest(contamination=contamination, random_state=42)
    X = df[["X (40g)", "Y (40g)", "Z (40g)"]]
    model.fit(X)
    scores = model.decision_function(X)
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


# ================== HEADER ==================
col_title, col_upload = st.columns([2, 3], gap="large")
with col_title:
    st.markdown("# enDAQ Vibration Dashboard")
    st.markdown(
        f'<span style="color:{MUTED};font-size:0.78rem;letter-spacing:0.06em;">'
        'enDAQ · Signal Processing · Anomaly Detection</span>',
        unsafe_allow_html=True
    )
with col_upload:
    uploaded_file = st.file_uploader("Upload IDE file", type=["ide"], label_visibility="collapsed")

st.markdown("<hr style='border:none;border-top:1px solid #1F2437;margin:0.5rem 0 1.2rem 0'>",
            unsafe_allow_html=True)

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
    with st.sidebar:
        st.markdown("## ⏱ Time Window")
        duration_sec = len(df_full) / fs_auto
        st.caption(f"Total duration: **{duration_sec:.2f} s**")

        tip("Start Time (s)", "Beginning of the analysis window. Signals before this point are ignored.")
        start_time = st.number_input("Start (s)", 0.0, duration_sec, 0.0, label_visibility="collapsed")

        tip("End Time (s)", "End of the analysis window. Maximum span is 40 s to keep processing fast.")
        end_time = st.number_input("End (s)", 0.0, duration_sec, min(40.0, duration_sec), label_visibility="collapsed")
        if end_time - start_time > 40:
            end_time = start_time + 40

        st.markdown("## 📶 Sampling")
        tip("Override fs", "Check to manually set the sampling frequency instead of auto-detecting it from the file timestamps.")
        override_fs = st.checkbox("Override fs", label_visibility="collapsed")
        tip("Manual fs (Hz)", "Sampling frequency in Hz. Only used when Override fs is checked.")
        manual_fs = st.number_input("Manual fs (Hz)", value=fs_auto, label_visibility="collapsed")
        fs = manual_fs if override_fs else fs_auto
        st.caption(f"Active fs: **{fs:.1f} Hz**")

        tip("MA Window", "Moving average window length (samples). Larger values smooth the time-series more but reduce temporal resolution.")
        ma_window = st.slider("MA Window", 1, 50, 5, label_visibility="collapsed")

        st.markdown("## 〰 FFT")
        tip("Windowing", "Apply a Hann window to each FFT segment to reduce spectral leakage caused by signal discontinuities at segment edges.")
        use_windowing = st.checkbox("Windowing", label_visibility="collapsed")
        tip("Averaging", "Average the FFT magnitude across multiple overlapping segments (Welch-style). Reduces variance at the cost of frequency resolution.")
        use_averaging = st.checkbox("Averaging", label_visibility="collapsed")
        tip("Overlap", "Enable 50% overlap between consecutive FFT segments. Increases the number of averages and smooths the spectrum.")
        use_overlap = st.checkbox("Overlap", label_visibility="collapsed")
        window_type    = "hann"
        window_size    = 1024
        overlap_factor = 0.5

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            tip("f min (Hz)", "Lower bound of the frequency axis shown in the FFT plot.")
            fmin = st.number_input("f min", value=0.0, label_visibility="collapsed")
        with col_f2:
            tip("f max (Hz)", "Upper bound of the frequency axis shown in the FFT plot.")
            fmax = st.number_input("f max", value=50.0, label_visibility="collapsed")

        st.markdown("## 🔺 Peaks")
        tip("Min Height", "Minimum FFT amplitude for a point to be considered a peak. Raise this to suppress low-amplitude noise peaks.")
        peak_height = st.number_input("Min Height", value=0.0, label_visibility="collapsed")
        tip("Min Distance", "Minimum number of frequency bins between two detected peaks. Prevents multiple markers on a single broad peak.")
        peak_distance = st.slider("Min Distance", 1, 200, 20, label_visibility="collapsed")
        tip("Min Prominence", "How much a peak must stand out from surrounding baseline. Higher values keep only the most dominant peaks.")
        peak_prominence = st.number_input("Min Prominence", value=0.0, label_visibility="collapsed")

        st.markdown("## 📊 PSD")
        tip("nperseg", "Length of each segment used in Welch's PSD estimate. Larger values give finer frequency resolution but more variance.")
        nperseg = st.slider("nperseg", 128, 4096, 1024, label_visibility="collapsed")

        st.markdown("## 🌀 STFT")
        tip("STFT Window size", "Number of samples per STFT frame. Larger → better frequency resolution, coarser time resolution.")
        stft_window = st.slider("Window size", 128, 4096, 512, label_visibility="collapsed")
        tip("STFT Overlap %", "Percentage of overlap between consecutive STFT frames. Higher overlap gives smoother time axis in the spectrogram.")
        stft_overlap = st.slider("Overlap %", 0, 90, 50, label_visibility="collapsed")

        st.markdown("## 🔍 Anomaly")
        tip("Model", "Isolation Forest isolates anomalies by randomly partitioning data — efficient for high-dimensional data. PCA reconstructs data and flags points with high reconstruction error.")
        model_choice = st.selectbox("Model", ["Isolation Forest", "PCA"], index=1, label_visibility="collapsed")
        tip("Contamination", "Expected fraction of anomalies in the data. A higher value flags more points as anomalous. Typical range: 0.001–0.05.")
        contamination = st.slider("Contamination", 0.001, 0.1, 0.001, 0.001, format="%.3f", label_visibility="collapsed")
        tip("RMS Threshold %ile", "Pre-filter: only keep time-windows whose RMS exceeds this percentile of the overall RMS distribution. Removes near-zero (idle) segments before anomaly detection.")
        rms_percentile = st.slider("RMS Threshold %ile", 0, 80, 10, label_visibility="collapsed")
        tip("Axis", "Which acceleration axis to display in the anomaly plot. 'All' overlays all three axes.")
        axis_choice = st.selectbox("Axis", ["X (40g)", "Y (40g)", "Z (40g)", "All"], label_visibility="collapsed")

        st.markdown("")
        run_button = st.button("▶  Run Anomaly Model")

    # ================= SLICE DATA =================
    df = df_full.iloc[int(start_time*fs_auto):int(end_time*fs_auto)]

    # ================= QUICK METRICS ROW =================
    rms_vals = [
        np.sqrt(np.mean(df[c]**2))
        for c in ["X (40g)", "Y (40g)", "Z (40g)"]
    ]
    cols_m = st.columns(5, gap="small")
    with cols_m[0]:
        st.markdown(metric_card("Duration", f"{end_time - start_time:.1f}", "s"),
                    unsafe_allow_html=True)
    with cols_m[1]:
        st.markdown(metric_card("Samples", f"{len(df):,}", "pts"), unsafe_allow_html=True)
    with cols_m[2]:
        st.markdown(metric_card("RMS X", f"{rms_vals[0]:.3f}", "g"), unsafe_allow_html=True)
    with cols_m[3]:
        st.markdown(metric_card("RMS Y", f"{rms_vals[1]:.3f}", "g"), unsafe_allow_html=True)
    with cols_m[4]:
        st.markdown(metric_card("RMS Z", f"{rms_vals[2]:.3f}", "g"), unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ================= TABS =================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Time Series", "Statistics", "FFT / PSD", "Time–Frequency", "Anomaly"
    ])

    # ── TAB 1: TIME ──────────────────────────────────────────────
    with tab1:
        fig = go.Figure()
        for i, col in enumerate(["X (40g)", "Y (40g)", "Z (40g)"]):
            fig.add_trace(go.Scatter(
                # x=df.index, y=df[col], name=col,
                y=df[col], name=col,
                line=dict(color=TRACE_COLORS[i], width=1),
                mode="lines"
            ))
        fig.update_layout(title="Acceleration — Time Series",
                          xaxis_title="Time index", yaxis_title="Acceleration (g)")
        plotly_chart(fig)

    # ── TAB 2: STATS ─────────────────────────────────────────────
    with tab2:
        fig = go.Figure()
        for i, col in enumerate(["X (40g)", "Y (40g)", "Z (40g)"]):
            fig.add_trace(go.Histogram(
                x=df[col], name=col, opacity=0.7,
                marker_color=TRACE_COLORS[i]
            ))
        fig.update_layout(barmode="overlay", title="Amplitude Distributions",
                          xaxis_title="Acceleration (g)", yaxis_title="Count")
        plotly_chart(fig)

        section_label("Statistical Metrics")
        stats = pd.DataFrame({
            "Axis":          ["X (40g)", "Y (40g)", "Z (40g)"],
            "Mean (g)":      [df[c].mean()  for c in ["X (40g)", "Y (40g)", "Z (40g)"]],
            "Std (g)":       [df[c].std()   for c in ["X (40g)", "Y (40g)", "Z (40g)"]],
            "RMS (g)":       [np.sqrt(np.mean(df[c]**2)) for c in ["X (40g)", "Y (40g)", "Z (40g)"]],
            "Min (g)":       [df[c].min()   for c in ["X (40g)", "Y (40g)", "Z (40g)"]],
            "Max (g)":       [df[c].max()   for c in ["X (40g)", "Y (40g)", "Z (40g)"]],
            "Peak-to-Peak":  [df[c].max() - df[c].min() for c in ["X (40g)", "Y (40g)", "Z (40g)"]],
        })

        # Render as dark-themed HTML table (st.dataframe ignores custom CSS)
        num_cols = [c for c in stats.columns if c != "Axis"]
        rows_html = ""
        row_colors = [BG_CARD, "#151828"]
        for r_idx, row in stats.iterrows():
            bg = row_colors[r_idx % 2]
            cells = f'<td style="padding:9px 14px;color:{TRACE_COLORS[r_idx]};font-weight:600">{row["Axis"]}</td>'
            for col in num_cols:
                cells += f'<td style="padding:9px 14px;text-align:right">{row[col]:.5f}</td>'
            rows_html += f'<tr style="background:{bg}">{cells}</tr>'

        header_cells = f'<th style="padding:8px 14px;text-align:left">Axis</th>'
        for col in num_cols:
            header_cells += f'<th style="padding:8px 14px;text-align:right">{col}</th>'

        table_html = f"""
        <div style="overflow-x:auto;border-radius:8px;border:1px solid #1F2437;margin-top:8px">
        <table style="width:100%;border-collapse:collapse;font-family:'DM Mono',monospace;
                      font-size:0.8rem;color:{TEXT};background:{BG_CARD}">
            <thead>
                <tr style="background:#0F1117;border-bottom:1px solid #2A2F45;
                           color:{MUTED};text-transform:uppercase;font-size:0.7rem;letter-spacing:0.07em">
                    {header_cells}
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
        </div>"""
        st.markdown(table_html, unsafe_allow_html=True)

    # ── TAB 3: FFT / PSD ─────────────────────────────────────────
    with tab3:
        section_label("FFT Spectrum")
        fig = go.Figure()
        for i, col in enumerate(["X (40g)", "Y (40g)", "Z (40g)"]):
            sig = df[col].values
            freqs, fft_vals = apply_fft(sig, fs)
            mask = (freqs >= fmin) & (freqs <= fmax)

            fig.add_trace(go.Scatter(
                x=freqs[mask], y=fft_vals[mask],
                name=col, mode="lines",
                line=dict(color=TRACE_COLORS[i], width=1.2)
            ))

            peaks = detect_peaks(freqs[mask], fft_vals[mask],
                                 height=peak_height, distance=peak_distance,
                                 prominence=peak_prominence)
            if len(peaks):
                fig.add_trace(go.Scatter(
                    x=freqs[mask][peaks], y=fft_vals[mask][peaks],
                    mode="markers", showlegend=False,
                    marker=dict(color=RED, size=7, symbol="circle",
                                line=dict(color="white", width=1))
                ))

        fig.update_layout(xaxis_title="Frequency (Hz)", yaxis_title="Amplitude")
        plotly_chart(fig)

        section_label("Power Spectral Density")
        fig2 = go.Figure()
        for i, col in enumerate(["X (40g)", "Y (40g)", "Z (40g)"]):
            psd_freq, psd_vals = compute_psd(df[col].values, fs)
            fig2.add_trace(go.Scatter(
                x=psd_freq, y=psd_vals,
                name=col, mode="lines",
                line=dict(color=TRACE_COLORS[i], width=1.2)
            ))
        fig2.update_layout(xaxis_title="Frequency (Hz)", yaxis_title="Power (g²/Hz)")
        plotly_chart(fig2)

    # ── TAB 4: TIME-FREQUENCY ────────────────────────────────────
    with tab4:
        section_label("STFT Spectrogram — X axis")
        sig = df["X (40g)"].values
        times, freqs_stft, stft_data = compute_stft(sig, fs, stft_window, stft_overlap)

        fig = go.Figure(data=go.Heatmap(
            z=stft_data.T,
            x=times,
            y=freqs_stft,
            colorscale="Viridis",
            colorbar=dict(title="Amplitude", tickfont=dict(color=TEXT))
        ))
        fig.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="Frequency (Hz)",
        )
        plotly_chart(fig)

    # ── TAB 5: ANOMALY ───────────────────────────────────────────
    with tab5:
        if run_button:
            df_an = df.copy()
            rms   = compute_rms(df_an)
            rms_threshold = np.percentile(rms, rms_percentile)
            df_an = df_an[rms > rms_threshold]

            axes_to_plot = (["X (40g)", "Y (40g)", "Z (40g)"]
                            if axis_choice == "All" else [axis_choice])

            if model_choice == "Isolation Forest":
                df_an = run_if(df_an, contamination)
            else:
                df_an, explained = run_pca(df_an, 2, contamination)
                st.markdown(
                    metric_card("Explained Variance", f"{explained:.3f}", ""),
                    unsafe_allow_html=True
                )

            df_an = df_an.reset_index(drop=True)    

            anomalies = df_an[df_an["anomaly"] == -1]

            # Anomaly count metric
            pct = len(anomalies) / max(len(df_an), 1) * 100
            col_a, col_b = st.columns([1, 4], gap="medium")
            with col_a:
                st.markdown(
                    metric_card("Anomalies", f"{len(anomalies):,}", f"{pct:.1f}% of window"),
                    unsafe_allow_html=True
                )

            with col_b:
                fig = go.Figure()
                for i, ax in enumerate(axes_to_plot):
                    fig.add_trace(go.Scatter(
                        x=df_an.index, y=df_an[ax],
                        name=ax, mode="lines",
                        line=dict(color=TRACE_COLORS[i], width=1)
                    ))
                    if len(anomalies):
                        fig.add_trace(go.Scatter(
                            x=anomalies.index, y=anomalies[ax],
                            mode="markers", name=f"{ax} anomaly",
                            marker=dict(color=RED, size=5,
                                        line=dict(color="white", width=0.8))
                        ))
                fig.update_layout(xaxis_title="Time index", yaxis_title="Acceleration (g)")
                plotly_chart(fig)

        else:
            st.info("Configure settings in the sidebar, then press **▶ Run Anomaly Model**.")

else:
    # ── EMPTY STATE ──────────────────────────────────────────────
    st.markdown(f"""
    <div style="
        text-align:center;
        padding: 5rem 2rem;
        color: {MUTED};
        border: 1.5px dashed #1F2437;
        border-radius: 14px;
        margin-top: 2rem;
    ">
        <div style="font-size:3rem;margin-bottom:1rem">📡</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.1rem;color:{TEXT};margin-bottom:0.5rem">
            Drop an IDE file to begin
        </div>
        <div style="font-size:0.8rem;letter-spacing:0.05em">
            Supports enDAQ .ide acceleration recordings
        </div>
    </div>
    """, unsafe_allow_html=True)
