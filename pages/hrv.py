import streamlit as st
import neurokit2 as nk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# Streamlit page settings
# --------------------------------------------------------------
st.set_page_config(page_title="HRV Explorer – Dual ECG", page_icon="📊", layout="wide")

# --------------------------------------------------------------
# 🛠️ Sidebar – simulation parameters
# --------------------------------------------------------------
st.sidebar.header("Simulation Parameters")

# Recording settings
duration_min = st.sidebar.slider("Duration (minutes)", 1, 15, 5)
sampling_rate = st.sidebar.slider("Sampling Rate (Hz)", 100, 1000, 500, step=50)
heart_rate = st.sidebar.slider("Heart Rate (BPM)", 40, 200, 80)

st.sidebar.markdown("---")

# HRV of the two comparison signals
hrv_std1 = st.sidebar.slider("HRV Std 1 (BPM variation)", 0, 20, 5, key="hrv1")
hrv_std2 = st.sidebar.slider("HRV Std 2 (BPM variation)", 0, 20, 15, key="hrv2")

st.sidebar.markdown("---")
zoom_toggle = st.sidebar.checkbox("Zoom into first N seconds", value=False)
zoom_seconds = st.sidebar.slider("Zoom window (s)", 1.0, 30.0, 10.0, 0.5) if zoom_toggle else None

# Convert minutes to seconds
duration = duration_min * 60

# --------------------------------------------------------------
# ECG generation helpers (cached for performance)
# --------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_ecg(duration_s: int, fs: int, hr: int, hrv_std: int):
    """Return synthetic ECG signal (numpy array) and its R‑peak indices."""
    ecg = nk.ecg_simulate(
        duration=duration_s,
        sampling_rate=fs,
        heart_rate=hr,
        heart_rate_std=hrv_std,
        method="ecgsyn",
    )
    _, info = nk.ecg_process(ecg, sampling_rate=fs)
    return ecg, info["ECG_R_Peaks"]

# Generate both signals
ecg1, rpeaks1 = generate_ecg(duration, sampling_rate, heart_rate, hrv_std1)
ecg2, rpeaks2 = generate_ecg(duration, sampling_rate, heart_rate, hrv_std2)

time = np.linspace(0, duration, len(ecg1))  # identical length & fs for both

# --------------------------------------------------------------
# Optional zoom
# --------------------------------------------------------------
if zoom_toggle and zoom_seconds:
    mask_zoom = time <= zoom_seconds
    plot_time = time[mask_zoom]
    ecg1_plot = ecg1[mask_zoom]
    ecg2_plot = ecg2[mask_zoom]

    # translate R‑peaks to new index space
    rpeaks1_plot = [int(i) for i in rpeaks1 if i < len(plot_time)]
    rpeaks2_plot = [int(i) for i in rpeaks2 if i < len(plot_time)]
else:
    plot_time = time
    ecg1_plot = ecg1
    ecg2_plot = ecg2
    rpeaks1_plot = rpeaks1
    rpeaks2_plot = rpeaks2

# --------------------------------------------------------------
# 📈 Time‑domain comparison – side‑by‑side
# --------------------------------------------------------------
col_a, col_b = st.columns(2, gap="medium")

with col_a:
    st.subheader(f"ECG 1 (HRV ±{hrv_std1} BPM)")
    fig1, ax1 = plt.subplots(figsize=(6, 3))
    ax1.plot(plot_time, ecg1_plot, lw=1)
    ax1.plot(plot_time[rpeaks1_plot], ecg1_plot[rpeaks1_plot], "ro", markersize=5, fillstyle="none")
    ax1.set(xlabel="Time (s)", ylabel="Amplitude")
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)

with col_b:
    st.subheader(f"ECG 2 (HRV ±{hrv_std2} BPM)")
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    ax2.plot(plot_time, ecg2_plot, lw=1, color="orange")
    ax2.plot(plot_time[rpeaks2_plot], ecg2_plot[rpeaks2_plot], "ro", markersize=5, fillstyle="none")
    ax2.set(xlabel="Time (s)", ylabel="Amplitude")
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

# --------------------------------------------------------------
# RR intervals & Poincaré plot
# --------------------------------------------------------------
rr1 = np.diff(rpeaks1) / sampling_rate
rr2 = np.diff(rpeaks2) / sampling_rate

fig_poinc, axes = plt.subplots(1, 2, figsize=(12, 5))
axes = axes if isinstance(axes, np.ndarray) else [axes]

axes[0].scatter(rr1[:-1], rr1[1:], alpha=0.7, c="blue", label="ECG 1")
axes[0].set(xlabel="RR(n) [s]", ylabel="RR(n+1) [s]", title="Poincaré – ECG 1")
axes[0].grid(True)

axes[1].scatter(rr2[:-1], rr2[1:], alpha=0.7, c="orange", label="ECG 2")
axes[1].set(xlabel="RR(n) [s]", ylabel="RR(n+1) [s]", title="Poincaré – ECG 2")
axes[1].grid(True)

st.pyplot(fig_poinc)

# --------------------------------------------------------------
# HRV Stats Summary
# --------------------------------------------------------------

def compute_hrv_metrics(rr_seconds: np.ndarray):
    rr_ms = rr_seconds * 1000.0
    sdnn = np.std(rr_ms, ddof=1)
    diff_ms = np.diff(rr_ms)
    rmssd = np.sqrt(np.mean(diff_ms ** 2)) if diff_ms.size else np.nan
    pnn50 = (np.sum(np.abs(diff_ms) > 50) / diff_ms.size * 100) if diff_ms.size else np.nan
    sd1 = np.sqrt(np.var(diff_ms) / 2) if diff_ms.size else np.nan
    sd2 = np.sqrt(2 * sdnn ** 2 - 0.5 * rmssd ** 2) if not np.isnan(rmssd) else np.nan
    mean_rr = np.mean(rr_ms)
    return {
        "Mean RR (ms)": round(mean_rr, 2),
        "SDNN (ms)": round(sdnn, 2),
        "RMSSD (ms)": round(rmssd, 2),
        "pNN50 (%)": round(pnn50, 2),
        "SD1 (ms)": round(sd1, 2),
        "SD2 (ms)": round(sd2, 2),
    }

summary_df = pd.DataFrame(
    [
        {"Signal": "ECG 1", **compute_hrv_metrics(rr1)},
        {"Signal": "ECG 2", **compute_hrv_metrics(rr2)},
    ]
).set_index("Signal")

st.subheader("HRV Stats Summary")
st.dataframe(summary_df, use_container_width=True)

# --------------------------------------------------------------
# Interpretations (unchanged)
# --------------------------------------------------------------
st.markdown(
    """
### How to interpret these HRV metrics

* **Mean RR** – average time between successive heart beats. Shorter means higher heart rate and often reflects sympathetic activation (stress, exercise).
* **SDNN** – standard deviation of all NN intervals; global HRV influenced by both branches of the autonomic nervous system. At rest, values **≥ 50 ms** are typical for healthy young adults.
* **RMSSD** – root mean square of successive differences; captures short‑term (beat‑to‑beat) variability driven predominantly by the parasympathetic (vagal) system.
* **pNN50** – percentage of adjacent NN intervals that differ by **> 50 ms**; another parasympathetic marker conceptually similar to RMSSD.
* **SD1 / SD2** – axes of the Poincaré ellipse. **SD1** quantifies short‑term variability (width), **SD2** represents long‑term variability (length). The SD1 : SD2 ratio can hint at autonomic balance (higher ratios → stronger vagal dominance).

> **Keep in mind**: HRV is highly context‑dependent – age, posture, respiration pattern, circadian timing, and recording length all modulate these values. Interpret them within the broader physiological and experimental context.
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------
# Educational Section (kept)
# --------------------------------------------------------------
st.subheader("What is a QRS in an ECG waveform? Visualization")
st.markdown(
    """
<div style="background-color: white; padding: 10px; border-radius: 4px;">
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/9e/SinusRhythmLabels.svg" alt="ECG waveform" style="display:block; margin:auto; width:512px;">
    <p style="text-align:center; margin-top: 8px;">Annotated ECG waveform showing P, QRS, and T waves</p>
</div>
    """,
    unsafe_allow_html=True,
)

