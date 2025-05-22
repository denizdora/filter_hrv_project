import streamlit as st

# --------------------------------------------------------------
# Streamlit App — Home / Landing Page
# --------------------------------------------------------------
# This file lives in the **root** of the project so that a simple
# ``streamlit run Home.py`` (or just ``streamlit run .``) will
# launch the multipage app.  All feature‑specific pages reside
# inside the *pages/* folder and are automatically discovered by
# Streamlit.
# --------------------------------------------------------------

st.set_page_config(
    page_title="ECG HRV & Filtering — Home",
    page_icon="🏠",
    layout="centered",
)

st.title("ECG HRV & Filtering Demo")

st.markdown(
    """
Welcome! This mini-suite lets you explore two complementary ECG/HRV tools:

1. **HRV Explorer**  Simulate ECG signals, detect R-peaks, generate Poincaré plots, and compute a rich panel of time-domain HRV metrics (MeanRR, SDNN, RMSSD, pNN50, SD1/SD2…). Ideal for learning how heart-rate variability changes under different physiological conditions.
2. **Wiener Filter Denoiser** - Dive into signal-processing theory with an *ideal* vs *non-ideal* Wiener filter. Tune filter order, symmetry, noise level, and inspect internal matrices & impulse responses while watching ECG noise disappear in both time and frequency domains.

Use the quick-links below **or** the Streamlit sidebar to jump straight into either module.
    """
)

# --------------------------------------------------------------
# Navigation helpers (Streamlit ≥ 1.22 supports st.page_link)
# If the current runtime is older, we gracefully fall back to a
# note so the user can still navigate via the sidebar.
# --------------------------------------------------------------

if hasattr(st, "page_link"):
    cols = st.columns(2)

    with cols[0]:
        st.page_link("pages/hrv.py", label="Open HRV Explorer", icon="📊")
        st.caption("Simulate ECG & inspect HRV metrics →")

    with cols[1]:
        st.page_link("pages/filter.py", label="Open Wiener Filter Demo", icon="⚙️")
        st.caption("Interactive ECG denoising →")
else:
    st.info(
        "**Tip**: If you don't see buttons above, update Streamlit to ≥ 1.22 ``pip install --upgrade streamlit`` \n"
        "   or simply select the desired page from the left-hand sidebar (**View all pages**)."
    )

st.markdown("---")

st.markdown(
    """
### About this project
*Built with* **Streamlit**, **NeuroKit2**, **NumPy/Pandas**

Feel free to fork, tweak parameters, and extend with frequency-domain HRV, real-world ECG uploads, or alternative denoising techniques.
    """
)
