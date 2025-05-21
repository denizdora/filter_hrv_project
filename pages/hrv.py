import streamlit as st
import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Title and Sliders
# -------------------------------------------------------------------
st.title("ECG Simulation with QRS Detection")

duration_min = st.slider("Duration (minutes)", 5, 15, 10)
sampling_rate = st.slider("Sampling Rate (Hz)", 100, 1000, 500)
heart_rate = st.slider("Heart Rate (BPM)", 40, 200, 80)
hrv_std = st.slider("HRV Std (BPM Variation)", 0, 20, 5)

# Convert minutes to seconds
duration = duration_min * 60

# -------------------------------------------------------------------
# 1. Regular ECG Generation (Full Signal)
# -------------------------------------------------------------------
@st.cache_data
def generate_ecg(duration, sampling_rate, heart_rate, hrv_std):
    """
    Generate the full baseline ECG using ecg_simulate(), then detect R-peaks.
    Return both the raw ECG & the R-peaks for the entire simulation.
    """
    ecg = nk.ecg_simulate(
        duration=duration,
        sampling_rate=sampling_rate,
        heart_rate=heart_rate,
        heart_rate_std=hrv_std,
        method="ecgsyn"
    )
    signals, info = nk.ecg_process(ecg, sampling_rate=sampling_rate)
    return ecg, signals, info

# -------------------------------------------------------------------
# 2. Irregular ECG Generation (Full Signal)
# -------------------------------------------------------------------
@st.cache_data
def generate_irregular_ecg(duration, sampling_rate):
    """
    Generate an ECG with higher HRV => more "irregular" (not a real arrhythmia).
    Return the full signal & full R-peaks.
    """
    ecg = nk.ecg_simulate(
        duration=duration,
        sampling_rate=sampling_rate,
        heart_rate=80,
        heart_rate_std=15,
        method="ecgsyn"
    )
    signals, info = nk.ecg_process(ecg, sampling_rate=sampling_rate)
    return ecg, signals, info

# -------------------------------------------------------------------
# Main: Generate the full "Regular" ECG & store full R-peaks
# -------------------------------------------------------------------
full_ecg, full_signals, full_info = generate_ecg(
    duration, sampling_rate, heart_rate, hrv_std
)
full_rpeaks = full_info["ECG_R_Peaks"]  # For entire signal
full_time = np.linspace(0, duration, len(full_ecg))

# We'll only modify the *plot* version of the signal if user zooms
plot_ecg = full_ecg.copy()
plot_time = full_time.copy()
plot_rpeaks = full_rpeaks.copy()  # for the main plot

# -------------------------------------------------------------------
# Zoom / Full Checkbox (Main ECG)
# -------------------------------------------------------------------
show_full_main = st.checkbox("Show entire (main) ECG signal (may be unreadable)", value=False)
if not show_full_main:
    zoom_start, zoom_end = st.slider(
        "Zoom into time window (seconds) for main ECG",
        min_value=0.0,
        max_value=float(duration),
        value=(0.0, min(10.0, float(duration))),
        step=1.0
    )
    # Mask only for plotting
    mask_main = (plot_time >= zoom_start) & (plot_time <= zoom_end)
    plot_time = plot_time[mask_main]
    plot_ecg = plot_ecg[mask_main]

    # Adjust rpeaks so they match the new time window for the main plot
    plot_rpeaks = [
        i for i in plot_rpeaks if zoom_start <= i/sampling_rate <= zoom_end
    ]
    plot_rpeaks = [
        int((i/sampling_rate - zoom_start)*sampling_rate) for i in plot_rpeaks
    ]

# Plot main ECG
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(plot_time, plot_ecg, label="Regular ECG", color="blue")
ax.plot([plot_time[i] for i in plot_rpeaks if i < len(plot_time)],
        [plot_ecg[i] for i in plot_rpeaks if i < len(plot_ecg)],
        "ro", label="Detected R-peaks", markersize=8, fillstyle="none")
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Amplitude")
ax.set_title("Regular ECG (with or without mild HRV)")
ax.legend()
st.pyplot(fig)

# Compute Normal RR intervals from the FULL signal, not the zoomed
normal_rr = np.diff(full_rpeaks) / sampling_rate

# -------------------------------------------------------------------
# Abnormal Rhythm + Poincaré Plots
# -------------------------------------------------------------------
with st.expander("Abnormal Rhythm Simulations"):

    # Toggle only for Irregular ECG
    show_irregular = st.checkbox("Generate 'Irregular' ECG")

    # Prepare placeholders for Irregular R-peaks
    irreg_ecg_full = None
    irreg_rpeaks_full = None

    # ---------- Irregular Rhythm ----------
    if show_irregular:
        irreg_ecg_full, irreg_signals_full, irreg_info_full = generate_irregular_ecg(duration, sampling_rate)
        irreg_rpeaks_full = irreg_info_full["ECG_R_Peaks"]
        irreg_time_full = np.linspace(0, len(irreg_ecg_full)/sampling_rate, len(irreg_ecg_full))

        # For plotting
        plot_ecg_irreg = irreg_ecg_full.copy()
        plot_time_irreg = irreg_time_full.copy()
        plot_rpeaks_irreg = irreg_rpeaks_full.copy()

        show_full_irreg = st.checkbox("Show entire Irregular signal", value=False)
        if not show_full_irreg:
            irreg_zoom_start, irreg_zoom_end = st.slider(
                "Zoom (seconds) for Irregular ECG",
                min_value=0.0,
                max_value=float(len(irreg_ecg_full)/sampling_rate),
                value=(0.0, min(10.0, float(len(irreg_ecg_full)/sampling_rate))),
                step=0.5
            )
            mask_irreg = (plot_time_irreg >= irreg_zoom_start) & (plot_time_irreg <= irreg_zoom_end)
            plot_time_irreg = plot_time_irreg[mask_irreg]
            plot_ecg_irreg = plot_ecg_irreg[mask_irreg]

            # Adjust only the plotted R-peaks
            kept_peaks = [
                i for i in plot_rpeaks_irreg
                if irreg_zoom_start <= i/sampling_rate <= irreg_zoom_end
            ]
            kept_peaks = [
                int((i/sampling_rate - irreg_zoom_start)*sampling_rate)
                for i in kept_peaks
            ]
            plot_rpeaks_irreg = kept_peaks

        # Plot the Irregular ECG
        fig_irreg, ax_irreg = plt.subplots(figsize=(10, 3))
        ax_irreg.plot(plot_time_irreg, plot_ecg_irreg, label="Irregular ECG", color="orange")
        ax_irreg.plot(
            [plot_time_irreg[i] for i in plot_rpeaks_irreg if i < len(plot_time_irreg)],
            [plot_ecg_irreg[i]  for i in plot_rpeaks_irreg if i < len(plot_ecg_irreg)],
            "ro", label="Detected R-peaks", markersize=8, fillstyle="none"
        )
        ax_irreg.set_title("ECG with Large HRV (Simulated Irregularity)")
        ax_irreg.legend()
        st.pyplot(fig_irreg)

    # ---------------------------
    # Poincaré Plot(s)
    # ---------------------------
    poincare_rr_data = []
    poincare_labels = []
    poincare_colors = []

    # Normal always from the full R-peaks (no zoom)
    normal_rr = np.diff(full_rpeaks) / sampling_rate
    poincare_rr_data.append(normal_rr)
    poincare_labels.append("Normal")
    poincare_colors.append("blue")

    # If Irregular is active, compute its FULL RR intervals
    if irreg_rpeaks_full is not None:
        irr_rr_full = np.diff(irreg_rpeaks_full) / sampling_rate
        poincare_rr_data.append(irr_rr_full)
        poincare_labels.append("Irregular")
        poincare_colors.append("orange")

    ncols = len(poincare_rr_data)
    fig_poinc, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(6*ncols, 5))

    # If there's only one subplot, wrap it in a list so we can iterate
    if ncols == 1:
        axes = [axes]

    for i, rr_array in enumerate(poincare_rr_data):
        ax_pc = axes[i]
        x = rr_array[:-1]
        y = rr_array[1:]
        ax_pc.scatter(x, y, c=poincare_colors[i], alpha=0.7, label=poincare_labels[i])
        ax_pc.set_xlabel("RR(n) [s]")
        ax_pc.set_ylabel("RR(n+1) [s]")
        ax_pc.set_title(f"Poincaré Plot - {poincare_labels[i]}")
        ax_pc.grid(True)
        ax_pc.legend()

    st.pyplot(fig_poinc)

# -------------------------------------------------------------------
# Educational Section
# -------------------------------------------------------------------
with st.expander("What is a QRS in an ECG waveform? (Image)"):
    st.markdown(
        """
        <div style="background-color: white; padding: 10px; border-radius: 4px;">
            <img src="https://upload.wikimedia.org/wikipedia/commons/9/9e/SinusRhythmLabels.svg"
                 alt="ECG waveform"
                 style="display:block; margin:auto; width:512px;">
            <p style="text-align:center; margin-top: 8px;">
                Annotated ECG waveform showing P, QRS, and T waves
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------------------------------------------------
# Remaining TO-DO / NOTES
# -------------------------------------------------------------------
## - standard deviation and HRV difference?
## - longer signal?
## - abnormal rhythms, show and detect them
## - Poincaré plot, longer sequences
## - real data vs. synthesized data
## - noise injection, quantization, etc.
## - neurokit ecg_clean + compare
## - streamlit deployment, streamlit cloud
