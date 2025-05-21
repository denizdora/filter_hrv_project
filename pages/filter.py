import streamlit as st
import neurokit2 as nk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
import matplotlib.ticker as ticker # For frequency plot formatting
from scipy.signal import welch


# Set a fixed random state for reproducibility of simulation and noise
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def mirror_spectrum(f, Pxx):
    f_mirrored = np.concatenate([-f[::-1], f[1:]])
    Pxx_mirrored = np.concatenate([Pxx[::-1], Pxx[1:]])
    return f_mirrored, Pxx_mirrored

# --- Wiener Filter Implementation ---

# Cache the results of the filter calculation based on noisy signal, order, and symmetric flag
# Note: This implementation requires the *clean* signal for optimal coefficient calculation (Ideal Wiener Filter)
@st.cache_data # Cache the results of the filter calculation
def apply_wiener_filter(noisy_signal: np.ndarray, clean_signal: np.ndarray, order: int, symmetric: bool = False):
    """
    Applies an 'ideal' Wiener filter using known clean signal.

    Args:
        noisy_signal (np.ndarray): The input signal corrupted by noise (x[n]).
        clean_signal (np.ndarray): The desired clean signal (s[n]).
        order (int): The order parameter 'p' of the Wiener filter.
                     For causal, filter length = p.
                     For symmetric, filter length = 2*p + 1.
        symmetric (bool): If True, compute a non-causal symmetric filter.
                          If False (default), compute a causal filter.

    Returns:
        tuple: (filtered_signal, dict_of_internals)
               filtered_signal (np.ndarray): The filtered signal.
               dict_of_internals (dict): Contains filter coefficients (h),
                                         regularized auto-correlation matrix (Rxx),
                                         and cross-correlation vector (Rxs).
        Returns (None, None) if filtering fails.
    """
    if order < 1:
        st.warning("Filter order (p) must be at least 1.")
        return None, None # Return None to indicate failure clearly

    N = len(noisy_signal)
    filter_length = order if not symmetric else 2 * order + 1

    if N <= filter_length:
         st.warning(f"Signal length ({N}) must be greater than filter length ({filter_length}). Adjust order or signal duration.")
         return None, None

    # --- Calculate Correlations (Biased estimates: divide by N) ---
    # Rxx: Autocorrelation of the noisy signal (x[n])
    # Required lags for causal: 0 to p-1
    # Required lags for symmetric: 0 to 2p (filter length - 1)
    phi_xx_full = np.correlate(noisy_signal, noisy_signal, mode='full')
    center_idx = N - 1 # Index corresponding to lag 0

    # Rxs: Cross-correlation between noisy signal (x[n]) and clean signal (s[n])
    # Required lags for causal: 0 to p-1
    # Required lags for symmetric: -p to +p
    phi_xs_full = np.correlate(noisy_signal, clean_signal, mode='full')

    try:
        if not symmetric:
            # --- Causal Wiener Filter ---
            # First column of Rxx (Toeplitz matrix) needs lags 0 to p-1
            autocorr_for_toeplitz = phi_xx_full[center_idx : center_idx + order] / N
            if len(autocorr_for_toeplitz) != order:
                 raise ValueError(f"Incorrect autocorrelation length for causal Rxx. Expected {order}, got {len(autocorr_for_toeplitz)}")
            
            Rxx = toeplitz(autocorr_for_toeplitz)
            Rxx_raw = Rxx.copy()

            # Rxs vector needs lags 0 to p-1
            Rxs = phi_xs_full[center_idx : center_idx + order] / N
            if len(Rxs) != order:
                 raise ValueError(f"Incorrect cross-correlation length for causal Rxs. Expected {order}, got {len(Rxs)}")

            filter_size = order

        else:
            # --- Symmetric (Non-Causal) Wiener Filter ---
            # Filter length L = 2p + 1
            L = filter_length
            # First column of Rxx (Toeplitz matrix) needs lags 0 to 2p (L-1)
            autocorr_for_toeplitz = phi_xx_full[center_idx : center_idx + L] / N
            if len(autocorr_for_toeplitz) != L:
                 raise ValueError(f"Incorrect autocorrelation length for symmetric Rxx. Expected {L}, got {len(autocorr_for_toeplitz)}")
            
            Rxx = toeplitz(autocorr_for_toeplitz)
            Rxx_raw = Rxx.copy()

            # Rxs vector needs lags -p to +p
            # Indices: center_idx - p to center_idx + p
            start_idx = center_idx - order
            end_idx = center_idx + order + 1
            if start_idx < 0 or end_idx > len(phi_xs_full):
                # This happens if signal is too short for the requested symmetric lag range
                raise ValueError(f"Cannot extract required cross-correlation lags (-{order} to +{order}) for symmetric filter. Signal length may be too short for this order.")
            Rxs = phi_xs_full[start_idx : end_idx] / N
            if len(Rxs) != L:
                 raise ValueError(f"Incorrect cross-correlation length for symmetric Rxs. Expected {L}, got {len(Rxs)}")

            filter_size = L

        # --- Solve for Filter Coefficients ---
        # Add regularization (small value to diagonal) to prevent singularity in Rxx
        # This is often necessary with real-world data or numerical precision issues
        epsilon = 1e-6 # Slightly larger epsilon can sometimes help stability
        Rxx_reg = Rxx + epsilon * np.identity(filter_size)

        # Solve the Wiener-Hopf equation: Rxx * h = Rxs
        h_wiener = np.linalg.solve(Rxx_reg, Rxs)

    except np.linalg.LinAlgError:
         st.error(f"Singular matrix encountered for Wiener filter (order={order}, symmetric={symmetric}). Cannot solve Wiener-Hopf equations. Try a different order, signal duration, or noise level.")
         return None, None # Indicate failure
    except ValueError as e:
         st.error(f"Error constructing Wiener filter components: {e}")
         return None, None
    except Exception as e:
        st.error(f"An unexpected error occurred during Wiener filter calculation: {e}")
        return None, None


    # --- Apply the Filter ---
    # Convolve the *noisy* signal with the filter coefficients
    # 'same' mode keeps the output length equal to the input length and handles centering
    # For the symmetric filter, 'same' mode implies the output corresponds to the
    # input sequence, appropriately shifted for the non-causal
    filtered_signal = np.convolve(noisy_signal, h_wiener, mode='same')

    # Return filtered signal and internal details
    return filtered_signal, dict(
        h=h_wiener,        # impulse response
        Rxx=Rxx_reg,       # regularised autocorr matrix
        Rxx_raw=Rxx_raw,   # unregularised autocorr matrix
        Rxs=Rxs,           # cross‑corr vector
        symmetric=symmetric, # Store type for display
        order=order # Store order for display
    )

@st.cache_data
def apply_nonideal_wiener_filter(noisy_signal: np.ndarray, order: int, symmetric: bool = False):
    """
    Applies a non-ideal Wiener filter (no clean signal).
    Approximates r_xs ≈ r_xx using only the noisy input.

    Args:
        noisy_signal (np.ndarray): Noisy input x[n]
        order (int): Filter order p
        symmetric (bool): Use symmetric (non-causal) filter?

    Returns:
        tuple: (filtered_signal, internals_dict) or (None, None) on failure
    """
    N = len(noisy_signal)
    filter_length = order if not symmetric else 2 * order + 1

    if N <= filter_length:
        st.warning("Signal too short for the selected filter order.")
        return None, None

    phi_xx_full = np.correlate(noisy_signal, noisy_signal, mode='full')
    center_idx = N - 1

    try:
        if not symmetric:
            phi = phi_xx_full[center_idx : center_idx + order] / N
            Rxx = toeplitz(phi)
            Rxx_raw = Rxx.copy()
            Rxs = phi.copy()  # approximate r_xs ≈ r_xx

        else:
            L = filter_length
            phi = phi_xx_full[center_idx : center_idx + L] / N
            Rxx = toeplitz(phi)
            Rxx_raw = Rxx.copy()
            start_idx = center_idx - order
            end_idx = center_idx + order + 1
            Rxs = phi_xx_full[start_idx:end_idx] / N  # also from autocorr

        Rxx_reg = Rxx + 1e-6 * np.identity(Rxx.shape[0])
        h = np.linalg.solve(Rxx_reg, Rxs)
        filtered_signal = np.convolve(noisy_signal, h, mode='same')

        return filtered_signal, dict(
            h=h, Rxx=Rxx_reg, Rxx_raw=Rxx_raw, Rxs=Rxs,
            symmetric=symmetric, order=order
        )

    except Exception as e:
        st.error(f"Non-ideal filter error: {e}")
        return None, None

@st.cache_data
def apply_smoothed_wiener_filter(
    noisy_signal: np.ndarray,
    order: int,
    symmetric: bool = False,
    ma_window: int = 15):
    """
    Applies a non-ideal Wiener filter using a smoothed version of the noisy signal
    to estimate the cross-correlation vector.

    Args:
        noisy_signal (np.ndarray): Noisy input x[n]
        order (int): Filter order p
        symmetric (bool): Use symmetric (non-causal) filter?
        ma_window (int): Moving average window size for smoothing (default = 15)

    Returns:
        tuple: (filtered_signal, internals_dict) or (None, None) on failure
    """
    N = len(noisy_signal)
    filter_length = order if not symmetric else 2 * order + 1

    if N <= filter_length or ma_window < 1:
        st.warning("Signal too short or MA window too small.")
        return None, None

    # --- Step 1: Smoothed signal via moving average ---
    smoothed_signal = np.convolve(noisy_signal, np.ones(ma_window) / ma_window, mode='same')

    # --- Step 2: Compute autocorrelation matrix from noisy signal ---
    phi_xx_full = np.correlate(noisy_signal, noisy_signal, mode='full')
    center_idx = N - 1

    # --- Step 3: Estimate cross-correlation using smoothed signal ---
    phi_xs_full = np.correlate(noisy_signal, smoothed_signal, mode='full')

    try:
        if not symmetric:
            phi = phi_xx_full[center_idx : center_idx + order] / N
            Rxx = toeplitz(phi)
            Rxx_raw = Rxx.copy()
            Rxs = phi_xs_full[center_idx : center_idx + order] / N

        else:
            L = filter_length
            phi = phi_xx_full[center_idx : center_idx + L] / N
            Rxx = toeplitz(phi)
            Rxx_raw = Rxx.copy()
            start_idx = center_idx - order
            end_idx = center_idx + order + 1
            Rxs = phi_xs_full[start_idx:end_idx] / N

        # Regularization and filter solve
        Rxx_reg = Rxx + 1e-6 * np.identity(Rxx.shape[0])
        h = np.linalg.solve(Rxx_reg, Rxs)
        filtered_signal = np.convolve(noisy_signal, h, mode='same')

        return filtered_signal, dict(
            h=h, Rxx=Rxx_reg, Rxx_raw=Rxx_raw, Rxs=Rxs,
            symmetric=symmetric, order=order
        )

    except Exception as e:
        st.error(f"Smoothed non-ideal Wiener error: {e}")
        return None, None


# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="ECG Wiener Filter Demo", page_icon="📈")

st.title("ECG Denoising Demo: Wiener Filter")

st.markdown("""
This demo illustrates the concept of an **ideal Wiener filter** for ECG denoising.
An ideal Wiener filter requires access to the *clean* signal to compute the optimal filter coefficients.
In this simulation, we generate a synthetic clean ECG, add noise, and then apply the Wiener filter using the clean signal information to achieve maximum possible noise reduction in the Mean Squared Error sense.
""")

# --- Sidebar Controls ---
st.sidebar.header("Simulation Parameters")
# Using unique keys for sliders for better stability in Streamlit
duration = st.sidebar.select_slider("Signal Duration (s)", options=[15, 20, 25, 30, 35], value=20)
fs = st.sidebar.select_slider("Sampling Frequency (Hz)",
                              options=[200, 300, 400, 500, 600,
                               700, 800, 900, 1000], value=200)
st.sidebar.write(f"Total Samples: **{duration * fs}**")
bpm = st.sidebar.select_slider("Heart Rate (bpm)", options=list(range(40, 121)), value=70)
hrstd = st.sidebar.select_slider(
    "HRV Std (Heart Rate Variability)",
    options=list(range(1, 21)),
    value=5,
    help="Controls the beat-to-beat variation in heart rate (in BPM). Higher values create more realistic variability in the synthetic ECG.",
    key="hrstd"
)

st.sidebar.caption(
    "**Note:** A higher HRV Std introduces more beat-to-beat irregularity. The default value of 5 represents a healthy resting rhythm."
)

noise_std = st.sidebar.select_slider("Noise Standard Deviation", options=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40], value=0.20)

st.sidebar.markdown("---") # Separator

st.sidebar.header("Wiener Filter Parameters")

wiener_order = st.sidebar.select_slider(
    "Wiener Filter Order (p)",
    options=list(range(1, 10)),
    value=4,
    help="Sets the order parameter 'p'.\n"
         "*   For a **causal** filter (symmetric unchecked), the filter will have **'p' taps**.\n"
         "*   For a **symmetric** (non-causal) filter (checkbox checked), the filter will have **'2p+1' taps**.\n"
         "A higher 'p' allows for a more complex filter, potentially improving noise reduction "
         "but increasing computational cost and risk of overfitting.",
    key="wiener_order"
)

st.sidebar.caption(
    "Filter order 'p' determines the number of taps.(more explanations in detailed explanations part)\n"
    "- Causal: p taps\n"
    "- Symmetric: 2p+1 taps\n"
    "- Higher order improves denoising but may overfit."
)

wiener_symmetric = st.sidebar.checkbox(
    "Use Symmetric (Non-Causal) Filter",
    value=False, # Or True, if you want symmetric by default
    help="If checked, a symmetric (non-causal) filter is used. It considers past, present, "
         "and future samples, resulting in **'2p+1' total taps** (where 'p' is the 'Wiener Filter Order' "
         "selected above). Typically offers better performance but requires access to future data "
         "(buffering in real-time).",
    key="wiener_symmetric"
)

use_nonideal = st.sidebar.checkbox(
    "Use Non-Ideal Wiener Filter (No Clean Signal)",
    value=False,
    help="In non-ideal mode, the filter is estimated without knowledge of the clean signal. "
         "We approximate the cross-correlation between noisy and clean signals using autocorrelation."
)

st.sidebar.markdown("---") # Separator

detailed = st.sidebar.checkbox("Show Detailed Explanations & Internals", value=False, key="show_detailed")

st.sidebar.markdown("---")
st.sidebar.info("Built with NeuroKit2 & Streamlit")

# --- Generate Signals ---
# Cached ECG simulation - Uses the fixed RANDOM_STATE
@st.cache_data
def generate_clean_ecg(duration, fs, bpm, hrstd):
    try:
      # Specify heart rate variability for a more realistic signal
      return nk.ecg_simulate(
          duration=duration,
          sampling_rate=fs,
          method="ecgsyn",
          heart_rate=bpm,
          heart_rate_std=hrstd,
          random_state=RANDOM_STATE)
    except Exception as e:
      st.error(f"Error generating clean ECG: {e}. Try different parameters.")
      # Fallback to a simple sine wave if generation fails
      st.warning("Falling back to a simple sine wave.")
      t = np.linspace(0, duration, int(fs * duration), endpoint=False)
      return 0.5 * np.sin(2 * np.pi * 1 * t) # 1 Hz sine wave (very basic)

@st.cache_data
def extract_rr_intervals(signal: np.ndarray, fs: int):
    """
    Detects R-peaks and returns RR intervals in milliseconds.
    Uses NeuroKit2's default method.
    """
    try:
        ecg_proc, rpeaks = nk.ecg_peaks(signal, sampling_rate=fs)
        rr_intervals = np.diff(rpeaks['ECG_R_Peaks']) / fs * 1000  # Convert to ms
        return rr_intervals, rpeaks
    except Exception as e:
        st.warning(f"R-peak detection failed: {e}")
        return None, None


ecg_clean = generate_clean_ecg(duration, fs, bpm, hrstd)
time = np.linspace(0, duration, len(ecg_clean), endpoint=False)

# Add white Gaussian noise
np.random.seed(RANDOM_STATE)
noise = np.random.normal(loc=0, scale=noise_std, size=len(ecg_clean))
ecg_noisy = ecg_clean + noise

# --- Apply Ideal Wiener Filter ---
ecg_filtered, wiener_info = apply_wiener_filter(ecg_noisy, ecg_clean, wiener_order, wiener_symmetric)
filter_params_str = f"p={wiener_order}, symmetric={wiener_symmetric}"

# # --- R-Peak Detection for Clean, Noisy, Filtered ---
# rr_clean, rpeaks_clean = extract_rr_intervals(ecg_clean, fs)
# rr_noisy, rpeaks_noisy = extract_rr_intervals(ecg_noisy, fs)
# rr_filtered, rpeaks_filtered = extract_rr_intervals(ecg_filtered, fs) if ecg_filtered is not None else (None, None)

# --- If Using Non-Ideal Filters ---
if use_nonideal:
    ecg_nonideal_raw, _ = apply_nonideal_wiener_filter(ecg_noisy, wiener_order, wiener_symmetric)
    ecg_nonideal_smooth, _ = apply_smoothed_wiener_filter(ecg_noisy, wiener_order, wiener_symmetric)

#     rr_nonideal_raw, rpeaks_nonideal_raw = extract_rr_intervals(ecg_nonideal_raw, fs) if ecg_nonideal_raw is not None else (None, None)
#     rr_nonideal_smooth, rpeaks_nonideal_smooth = extract_rr_intervals(ecg_nonideal_smooth, fs) if ecg_nonideal_smooth is not None else (None, None)

#     rr_dict = {
#         "clean": rr_clean,
#         "noisy": rr_noisy,
#         "filtered": rr_filtered,
#         "nonideal_raw": rr_nonideal_raw,
#         "nonideal_smooth": rr_nonideal_smooth
#     }

#     rpeaks_dict = {
#         "clean": rpeaks_clean,
#         "noisy": rpeaks_noisy,
#         "filtered": rpeaks_filtered,
#         "nonideal_raw": rpeaks_nonideal_raw,
#         "nonideal_smooth": rpeaks_nonideal_smooth
#     }
# else:
#     rr_dict = {
#         "clean": rr_clean,
#         "noisy": rr_noisy,
#         "filtered": rr_filtered
#     }

#     rpeaks_dict = {
#         "clean": rpeaks_clean,
#         "noisy": rpeaks_noisy,
#         "filtered": rpeaks_filtered
#     }

# --- Display Stats ---
st.subheader("Signal Statistics & Filter Performance")

col_stats1, col_stats2, col_stats3 = st.columns(3)

with col_stats1:
    st.metric("Clean ECG Std Dev", f"{np.std(ecg_clean):.4f}")
with col_stats2:
    st.metric("Noisy ECG Std Dev", f"{np.std(ecg_noisy):.4f}")
with col_stats3:
    if ecg_filtered is not None:
        st.metric("Filtered ECG Std Dev", f"{np.std(ecg_filtered):.4f}")
    else:
        st.metric("Filtered ECG Std Dev", "N/A")

# Calculate MSE
col_mse1, col_mse2 = st.columns(2)
mse_noisy = np.mean((ecg_noisy - ecg_clean)**2)
col_mse1.write(f"**MSE (Noisy vs Clean):** {mse_noisy:.6f}")

if ecg_filtered is not None:
    mse_filtered = np.mean((ecg_filtered - ecg_clean)**2)
    col_mse2.write(f"**MSE (Filtered vs Clean):** {mse_filtered:.6f}")
    # Avoid division by zero or near-zero MSE values
    if mse_noisy > 1e-9 and mse_filtered > 1e-9:
        improvement = mse_noisy / mse_filtered
        col_mse2.write(f"**MSE Improvement Factor:** {improvement:.2f}x")
    else:
         col_mse2.write(f"**MSE Improvement Factor:** N/A (MSE values too small)")
else:
    col_mse2.write("**MSE (Filtered vs Clean):** N/A (Filtering Failed)")
    col_mse2.write("**MSE Improvement Factor:** N/A")

# # --- Hrv Analysis ---
# st.subheader("Heart Rate Variability (HRV) Analysis")

# st.markdown("### Extended HRV Metrics (via NeuroKit2)")

# hrv_extended = []

# for label, rpeaks in rpeaks_dict.items():
#     if rpeaks is None or "ECG_R_Peaks" not in rpeaks:
#         continue
#     try:
#         hrv_result = nk.hrv(rpeaks, sampling_rate=fs, show=False)
#         def get_metric(df, col):
#             try:
#                 return float(df[col].iloc[0])
#             except:
#                 return np.nan
            
#         hrv_extended.append({
#             "Signal": label,
#             "Mean RR (ms)": get_metric(hrv_result, "HRV_MeanNN"),
#             "SDNN (ms)": get_metric(hrv_result, "HRV_SDNN"),
#             "RMSSD (ms)": get_metric(hrv_result, "HRV_RMSSD"),
#             "pNN50 (%)": get_metric(hrv_result, "HRV_pNN50"),
#             "LF (ms²)": get_metric(hrv_result, "HRV_LF"),
#             "HF (ms²)": get_metric(hrv_result, "HRV_HF"),
#             "LF/HF": get_metric(hrv_result, "HRV_LFHF"),
#             "SD1": get_metric(hrv_result, "HRV_SD1"),
#             "SD2": get_metric(hrv_result, "HRV_SD2"),
#         })

#     except Exception as e:
#         st.warning(f"Could not compute HRV for {label}: {e}")

# if hrv_extended:
#     df_extended = pd.DataFrame(hrv_extended).set_index("Signal")
#     st.dataframe(df_extended.style.format("{:.2f}", na_rep="N/A"), use_container_width=True)
# else:
#     st.info("No HRV data available to display.")


# st.markdown("### HRV Time-Domain Metrics")
# st.dataframe(df_extended.style.format(precision=2), use_container_width=True)


# ---------- DETAILED EXPLANATION & INTERNALS SECTION ---------------------------------
if detailed:
    st.markdown("---") # Separator
    st.markdown("## Detailed Explanation & Internals")

    if wiener_info is not None:
        st.markdown("""
        The ideal Wiener filter coefficients $h[n]$ are calculated by solving the **Wiener-Hopf Equation**:
        $$ \\mathbf R_{xx} \\mathbf h = \\mathbf r_{xs} $$
        where:
        *   $\\mathbf R_{xx}$ is the autocorrelation matrix of the noisy input signal $x[n]$.
        *   $\\mathbf r_{xs}$ is the cross-correlation vector between the noisy input signal $x[n]$ and the desired clean signal $s[n]$.
        *   $\\mathbf h$ is the vector of filter coefficients we solve for.

        The filter order $p$ (or $2p+1$ for symmetric) determines the size of the matrices/vectors ($p$ times $p$ or $(2p+1)$ times $(2p+1)$).
        A small regularization value ($\\epsilon = 10^{-6}$) is added to the diagonal of $\\mathbf R_{xx}$ to improve numerical stability and prevent singularity.
        """)

        col_detailed1, col_detailed2 = st.columns(2)

        # -- Impulse response ---------------------------------
        with col_detailed1:
            st.markdown("#### Wiener filter impulse response $h[n]$")
            st.markdown(f"Filter Order: {len(wiener_info['h'])}")

            fig_h, ax_h = plt.subplots(figsize=(6, 3))
            ax_h.stem(np.arange(len(wiener_info['h'])), wiener_info["h"]) # Use np.arange for tap index
            ax_h.set_xlabel("Index")
            ax_h.set_ylabel("Amplitude")
            ax_h.set_title("Wiener Filter Impulse Response")
            ax_h.grid(True, alpha=0.4)
            st.pyplot(fig_h)

    else:
        st.warning("Detailed internals are not available because the Wiener filter could not be computed.")

if detailed:
    st.markdown("---")
    st.markdown("## Step-by-Step Internals & Visualizations")
    st.markdown("#### Step 1: Autocorrelation of Noisy Signal (Zoomed)")
    st.markdown("This is the zoomed-in autocorrelation of the noisy ECG signal near lag $k=0$, which is most relevant to Wiener filter design.")

    full_autocorr = np.correlate(ecg_noisy, ecg_noisy, mode='full') / len(ecg_noisy)
    lags = np.arange(-len(ecg_noisy) + 1, len(ecg_noisy))

    lag_limit = 500
    center_idx = len(full_autocorr) // 2
    zoom_autocorr = full_autocorr[center_idx - lag_limit : center_idx + lag_limit + 1]
    zoom_lags = lags[center_idx - lag_limit : center_idx + lag_limit + 1]

    fig_ac, ax_ac = plt.subplots(figsize=(8, 3))
    ax_ac.plot(zoom_lags, zoom_autocorr)
    ax_ac.set_title("Autocorrelation of Noisy Signal $\\phi_{xx}[k]$ (Zoomed)")
    ax_ac.set_xlabel("Lag $k$")
    ax_ac.set_ylabel("Amplitude")
    ax_ac.grid(True)


    st.pyplot(fig_ac)

    st.markdown("#### Step 2: Cross-Correlation Between Noisy and Clean Signal (Zoomed)")
    st.markdown("""
    This is the zoomed-in cross-correlation between the noisy input $x[n]$ and clean signal $s[n]$ around lag $k=0$.
    This range is especially important in computing $\\mathbf{r}_{xs}$ in the Wiener-Hopf equation.
    """)

    crosscorr = np.correlate(ecg_noisy, ecg_clean, mode="full") / len(ecg_noisy)
    lags_xs = np.arange(-len(ecg_noisy) + 1, len(ecg_noisy))

    lag_limit = 500
    center_idx = len(crosscorr) // 2
    zoom_crosscorr = crosscorr[center_idx - lag_limit : center_idx + lag_limit + 1]
    zoom_lags_xs = lags_xs[center_idx - lag_limit : center_idx + lag_limit + 1]

    fig_cc, ax_cc = plt.subplots(figsize=(8, 3))
    ax_cc.plot(zoom_lags_xs, zoom_crosscorr)
    ax_cc.set_title("Cross-Correlation $\\phi_{xs}[k]$ Between $x[n]$ and $s[n]$ (Zoomed)")
    ax_cc.set_xlabel("Lag $k$")
    ax_cc.set_ylabel("Amplitude")
    ax_cc.grid(True)

    st.pyplot(fig_cc)

    with st.expander("#### Note on Lag $k$"):
        st.markdown("""
        **Note on Lag $k$:**  
        The x-axis shows how much we shift one signal relative to the other.  
        - $k = 0$ means both signals are aligned.  
        - $k > 0$ means $s[n]$ is shifted right (future values of $s$).  
        - $k < 0$ means $s[n]$ is shifted left (past values of $s$).  

        This tells us how well the noisy and clean signals correlate when offset in time.
        """)

    st.markdown("#### Step 3: Regularization of $\\mathbf{R}_{xx}$ Matrix")
    st.markdown("""
    The autocorrelation matrix $\\mathbf{R}_{xx}$ is regularized by adding a small constant $\\epsilon = 10^{-6}$ to its diagonal.  
    This ensures the matrix is invertible and improves numerical stability.

    $$
    \\mathbf{R}_{xx}^{\\text{reg}} = \\mathbf{R}_{xx} + \\epsilon \\cdot \\mathbf{I}
    $$
    """)

    if "Rxx_raw" in wiener_info:
        col_raw, col_reg = st.columns(2)
        diff = wiener_info["Rxx"][:10, :10] - wiener_info["Rxx_raw"][:10, :10]
        with col_raw:
            st.markdown("**Unregularized $\\mathbf{R}_{xx}$** (Top 10×10 Block)")
            st.dataframe(wiener_info["Rxx_raw"][:10, :10].round(6))
            st.write("Difference Matrix (should only show non-zero on diagonal):")
            st.dataframe(diff.round(6))

        with col_reg:
            st.markdown("**Regularized $\\mathbf{R}_{xx}^{\\text{reg}}$** (Top 10×10 Block)")
            st.dataframe(wiener_info["Rxx"][:10, :10].round(6))
    else:
        st.warning("Unregularized matrix not available.")

    st.markdown("#### Step 4: Filtered Output as Convolution")
    st.markdown("""
    The final filtered signal $y[n]$ is produced by **convolving** the noisy ECG $x[n]$ with the Wiener filter's impulse response $h[n]$:
    $$
    y[n] = x[n] * h[n]
    $$
    This means that each point in the output is a weighted sum of nearby noisy samples, where the weights are given by $h[n]$.
    """)

    fig_conv, ax_conv = plt.subplots(figsize=(14, 4))
    ax_conv.plot(time, ecg_clean, label="Clean ECG $s[n]$", linestyle="--", color="blue", alpha=0.8)
    ax_conv.plot(time, ecg_noisy, label="Noisy ECG $x[n]$", color="red", alpha=0.5)
    ax_conv.plot(time, ecg_filtered, label="Filtered Output $y[n] = x[n] * h[n]$", color="green", linewidth=1.5)
    ax_conv.set_title("Convolution Output of Wiener Filter")
    ax_conv.set_xlabel("Time (s)")
    ax_conv.set_ylabel("Amplitude")
    ax_conv.grid(True, linestyle="--", alpha=0.5)
    ax_conv.legend()
    st.pyplot(fig_conv)

    st.markdown("---")
        # --- Non-Ideal Wiener Explanation ---
    st.markdown("### Non-Ideal Wiener Filtering (No Clean Signal)")
    st.markdown(r"""
    In practical scenarios, the clean signal $s[n]$ is usually unavailable. To overcome this, we tested two alternative methods to estimate the Wiener filter **without access to the clean signal**.

    #### 1. Raw Autocorrelation Approximation
    We assume:
    $$
    \mathbf{r}_{xs} \approx \mathbf{r}_{xx}
    $$

    This simplifies the Wiener-Hopf equation to:
    $$
    \mathbf{R}_{xx} \mathbf{h} \approx \mathbf{r}_{xx}
    $$

    Under this assumption, both the autocorrelation matrix $\mathbf{R}_{xx}$ and the cross-correlation vector $\mathbf{r}_{xs}$ are estimated from the noisy input signal $x[n]$ alone.

    **Limitation:** This method often produces weak denoising, as the filter essentially learns to reproduce the noisy signal.

    #### 2. Smoothed Estimate (Moving Average Proxy for $s[n]$)
    We generate a pseudo-clean estimate $\hat{s}[n]$ by applying a simple **moving average** to $x[n]$, and use it to compute the cross-correlation:
    $$
    \mathbf{r}_{xs} \approx \text{corr}(x[n], \hat{s}[n])
    $$

    This gives a more realistic structure for $\mathbf{r}_{xs}$ while still not relying on the true clean signal.

    **Advantage:** Often leads to noticeably better filtering, reducing high-frequency noise more effectively than the raw method.

    Both methods still use the same autocorrelation matrix $\mathbf{R}_{xx}$ estimated from $x[n]$, but differ in how $\mathbf{r}_{xs}$ is constructed. By comparing their outputs in the time and frequency domains, we demonstrate the impact of improved cross-correlation estimation on denoising performance.
    """)




st.markdown("---")
st.subheader("Time Domain Signal Plots")

# Checkboxes to select which signals to show
show_clean = st.checkbox("Show Clean ECG", value=True)
show_noisy = st.checkbox("Show Noisy ECG", value=True)
show_filtered = st.checkbox("Show Filtered ECG", value=True if ecg_filtered is not None else False)
if use_nonideal:
    st.markdown("### Non-Ideal Filtered Options")
    show_nonideal_raw = st.checkbox("Show Non-Ideal Filtered ECG (Raw Autocorr)", value=True)
    show_nonideal_smooth = st.checkbox("Show Non-Ideal Filtered ECG (Smoothed Estimate)", value=True)
else:
    show_nonideal_raw = False
    show_nonideal_smooth = False

# Main full-length signal plot
fig_dynamic, ax_dynamic = plt.subplots(figsize=(14, 4))
plotted_any_main = False

if show_noisy:
    ax_dynamic.plot(time, ecg_noisy, label="Noisy ECG", color="red", alpha=0.7)
    plotted_any_main = True

if show_filtered and ecg_filtered is not None:
    ax_dynamic.plot(time, ecg_filtered, label=f"Wiener Filtered ({filter_params_str})", color="green", linewidth=1.5)
    plotted_any_main = True

if show_clean:
    ax_dynamic.plot(time, ecg_clean, label="Clean ECG", color="blue", linestyle="--", alpha=0.8)
    plotted_any_main = True

# if use_nonideal:
#     ecg_nonideal_raw, _ = apply_nonideal_wiener_filter(ecg_noisy, wiener_order, wiener_symmetric)
#     ecg_nonideal_smooth, _ = apply_smoothed_wiener_filter(ecg_noisy, wiener_order, wiener_symmetric)

if show_nonideal_raw and ecg_nonideal_raw is not None:
        ax_dynamic.plot(time, ecg_nonideal_raw, label="Non-Ideal (Raw)", color="orange", linestyle="--", linewidth=1.5)
        plotted_any_main = True

if show_nonideal_smooth and ecg_nonideal_smooth is not None:
        ax_dynamic.plot(time, ecg_nonideal_smooth, label="Non-Ideal (Smoothed)", color="purple", linestyle="--", linewidth=1.5)
        plotted_any_main = True


if not plotted_any_main:
    st.warning("Select at least one signal to display.")
else:
    ax_dynamic.set_title("ECG Signals in Time Domain (Selected Signals)")
    ax_dynamic.set_xlabel("Time (s)")
    ax_dynamic.set_ylabel("Amplitude")
    ax_dynamic.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax_dynamic.set_xlim(0, duration)
    ax_dynamic.legend()
    st.pyplot(fig_dynamic)


# Optional zoom-in toggle
zoom_plot = st.checkbox("Show Zoomed-in Signal Plot", value=True, key="show_zoom_plot")

if zoom_plot:
    max_zoom = min(duration, 10.0)
    zoom_duration = st.slider("Zoom Duration (s)", min_value=1.0, max_value=max_zoom,
                              value=min(3.0, max_zoom), step=0.5, key="signal_zoom_slider")
    idx_zoom = int(zoom_duration * fs)

    fig_zoom, ax_zoom = plt.subplots(figsize=(14, 4))

    plotted_any = False

    if show_noisy:
        ax_zoom.plot(time[:idx_zoom], ecg_noisy[:idx_zoom], label="Noisy ECG", color="red", alpha=0.7)
        plotted_any = True

    if show_filtered and ecg_filtered is not None:
        ax_zoom.plot(time[:idx_zoom], ecg_filtered[:idx_zoom],
                     label=f"Wiener Filtered ({filter_params_str})", color="green", linewidth=1.5)
        plotted_any = True

    if show_clean:
        ax_zoom.plot(time[:idx_zoom], ecg_clean[:idx_zoom], label="Clean ECG", color="blue",
                     linestyle="--", alpha=0.8)
        plotted_any = True

    if use_nonideal:
        if show_nonideal_raw and ecg_nonideal_raw is not None:
            ax_zoom.plot(time[:idx_zoom], ecg_nonideal_raw[:idx_zoom],
                         label="Non-Ideal (Raw)", color="orange", linestyle="--", linewidth=1.5)
            plotted_any = True

        if show_nonideal_smooth and ecg_nonideal_smooth is not None:
            ax_zoom.plot(time[:idx_zoom], ecg_nonideal_smooth[:idx_zoom],
                         label="Non-Ideal (Smoothed)", color="purple", linestyle="--", linewidth=1.5)
            plotted_any = True

    if not plotted_any:
        st.warning("Select at least one signal to display in the zoom view.")
    else:
        ax_zoom.set_title(f"Zoomed Signal View: First {zoom_duration:.2f} Seconds")
        ax_zoom.set_xlabel("Time (s)")
        ax_zoom.set_ylabel("Amplitude")
        ax_zoom.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax_zoom.set_xlim(0, zoom_duration)
        ax_zoom.legend()
        st.pyplot(fig_zoom)

# Welch PSD for each signal
f_clean, Pxx_clean = welch(ecg_clean, fs=fs, nperseg=1024)
f_noisy, Pxx_noisy = welch(ecg_noisy, fs=fs, nperseg=1024)
if ecg_filtered is not None:
    f_filtered, Pxx_filtered = welch(ecg_filtered, fs=fs, nperseg=1024)

f_clean_sym, Pxx_clean_sym = mirror_spectrum(f_clean, Pxx_clean)
f_noisy_sym, Pxx_noisy_sym = mirror_spectrum(f_noisy, Pxx_noisy)
if ecg_filtered is not None:
    f_filtered_sym, Pxx_filtered_sym = mirror_spectrum(f_filtered, Pxx_filtered)

# Y-axis range for consistent comparison
ymin = -0.7
ymax = 1.7

st.markdown("---")
st.subheader("Time & Frequency Domain Comparison")

T_VIS = 10  # seconds to show in time domain comparison
idx_vis = int(T_VIS * fs)

# Row 1: Clean ECG
col1, col2 = st.columns(2)

with col1:
    fig_clean_time, ax_clean_time = plt.subplots(figsize=(6, 3))
    ax_clean_time.plot(time[:idx_vis], ecg_clean[:idx_vis], color="blue")
    ax_clean_time.set_title("Clean ECG (Time Domain, First 10s)")
    ax_clean_time.set_xlabel("Time (s)")
    ax_clean_time.set_ylabel("Amplitude")
    ax_clean_time.set_ylim(ymin, ymax)
    ax_clean_time.axhline(0, color='black', linewidth=0.5)
    ax_clean_time.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig_clean_time)

with col2:
    fig_clean_psd, ax_clean_psd = plt.subplots(figsize=(6, 3))
    ax_clean_psd.semilogy(f_clean_sym, Pxx_clean_sym, color="blue")
    ax_clean_psd.set_title("Clean ECG (Power Spectral Density)")
    ax_clean_psd.set_xlabel("Frequency (Hz)")
    ax_clean_psd.set_ylabel("Power/Frequency")
    ax_clean_time.set_xlim(0, T_VIS)
    ax_clean_psd.set_xlim(-100, 100)
    ax_clean_psd.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig_clean_psd)


# Row 2: Noisy ECG
col3, col4 = st.columns(2)

with col3:
    fig_noisy_time, ax_noisy_time = plt.subplots(figsize=(6, 3))
    ax_noisy_time.plot(time[:idx_vis], ecg_noisy[:idx_vis], color="red", alpha=0.7)
    ax_noisy_time.set_title("Noisy ECG (Time Domain, First 10s)")
    ax_noisy_time.set_xlabel("Time (s)")
    ax_noisy_time.set_ylabel("Amplitude")
    ax_noisy_time.set_ylim(ymin, ymax)
    ax_noisy_time.axhline(0, color='black', linewidth=0.5)
    ax_noisy_time.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig_noisy_time)

with col4:
    fig_noisy_psd, ax_noisy_psd = plt.subplots(figsize=(6, 3))
    ax_noisy_psd.semilogy(f_noisy_sym, Pxx_noisy_sym, color="red")
    ax_noisy_psd.set_title("Noisy ECG (Power Spectral Density)")
    ax_noisy_psd.set_xlabel("Frequency (Hz)")
    ax_noisy_psd.set_ylabel("Power/Frequency")
    ax_noisy_time.set_xlim(0, T_VIS)
    ax_noisy_psd.set_xlim(-100, 100)
    ax_noisy_psd.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig_noisy_psd)

# Row 3: Filtered ECG
col5, col6 = st.columns(2)

with col5:
    fig_filt_time, ax_filt_time = plt.subplots(figsize=(6, 3))
    if ecg_filtered is not None:
        ax_filt_time.plot(time[:idx_vis], ecg_filtered[:idx_vis], color="green")
        ax_filt_time.set_title("Filtered ECG (Time Domain, First 10s)")
    else:
        ax_filt_time.set_title("Filtered ECG (Unavailable)")
    ax_filt_time.set_xlabel("Time (s)")
    ax_filt_time.set_ylabel("Amplitude")
    ax_filt_time.set_xlim(0, T_VIS)
    ax_filt_time.set_ylim(ymin, ymax)
    ax_filt_time.axhline(0, color='black', linewidth=0.5)
    ax_filt_time.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig_filt_time)

with col6:
    fig_filt_psd, ax_filt_psd = plt.subplots(figsize=(6, 3))
    if ecg_filtered is not None:
        ax_filt_psd.semilogy(f_filtered_sym, Pxx_filtered_sym, color="green")
        ax_filt_psd.set_title("Filtered ECG (Power Spectral Density)")
    else:
        ax_filt_psd.set_title("Filtered ECG PSD (Unavailable)")
    ax_filt_psd.set_xlabel("Frequency (Hz)")
    ax_filt_psd.set_ylabel("Power/Frequency")
    ax_filt_psd.set_xlim(-100, 100)
    ax_filt_psd.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig_filt_psd)

if use_nonideal:
        # Row 4: Non-Ideal (Raw Autocorr)
    f_nonideal_raw, Pxx_nonideal_raw = welch(ecg_nonideal_raw, fs=fs, nperseg=1024)
    f_raw_sym, Pxx_raw_sym = mirror_spectrum(f_nonideal_raw, Pxx_nonideal_raw)

    col7, col8 = st.columns(2)
    with col7:
        fig_raw_time, ax_raw_time = plt.subplots(figsize=(6, 3))
        ax_raw_time.plot(time[:idx_vis], ecg_nonideal_raw[:idx_vis], color="orange")
        ax_raw_time.set_title("Non-Ideal (Raw) - Time Domain")
        ax_raw_time.set_xlabel("Time (s)")
        ax_raw_time.set_ylabel("Amplitude")
        ax_raw_time.set_xlim(0, T_VIS)
        ax_raw_time.set_ylim(ymin, ymax)
        ax_raw_time.axhline(0, color='black', linewidth=0.5)
        ax_raw_time.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig_raw_time)

    with col8:
        fig_raw_psd, ax_raw_psd = plt.subplots(figsize=(6, 3))
        ax_raw_psd.semilogy(f_raw_sym, Pxx_raw_sym, color="orange")
        ax_raw_psd.set_title("Non-Ideal (Raw) - PSD")
        ax_raw_psd.set_xlabel("Frequency (Hz)")
        ax_raw_psd.set_ylabel("Power/Frequency")
        ax_raw_psd.set_xlim(-100, 100)
        ax_raw_psd.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig_raw_psd)

    # Row 5: Non-Ideal (Smoothed Estimate)
    f_nonideal_smooth, Pxx_nonideal_smooth = welch(ecg_nonideal_smooth, fs=fs, nperseg=1024)
    f_smooth_sym, Pxx_smooth_sym = mirror_spectrum(f_nonideal_smooth, Pxx_nonideal_smooth)

    col9, col10 = st.columns(2)
    with col9:
        fig_smooth_time, ax_smooth_time = plt.subplots(figsize=(6, 3))
        ax_smooth_time.plot(time[:idx_vis], ecg_nonideal_smooth[:idx_vis], color="purple")
        ax_smooth_time.set_title("Non-Ideal (Smoothed) - Time Domain")
        ax_smooth_time.set_xlabel("Time (s)")
        ax_smooth_time.set_ylabel("Amplitude")
        ax_smooth_time.set_xlim(0, T_VIS)
        ax_smooth_time.set_ylim(ymin, ymax)
        ax_smooth_time.axhline(0, color='black', linewidth=0.5)
        ax_smooth_time.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig_smooth_time)

    with col10:
        fig_smooth_psd, ax_smooth_psd = plt.subplots(figsize=(6, 3))
        ax_smooth_psd.semilogy(f_smooth_sym, Pxx_smooth_sym, color="purple")
        ax_smooth_psd.set_title("Non-Ideal (Smoothed) - PSD")
        ax_smooth_psd.set_xlabel("Frequency (Hz)")
        ax_smooth_psd.set_ylabel("Power/Frequency")
        ax_smooth_psd.set_xlim(-100, 100)
        ax_smooth_psd.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig_smooth_psd)
