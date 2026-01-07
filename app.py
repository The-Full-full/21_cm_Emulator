"""
Application Name: 21cm Global Signal Emulator
---------------------------------------------
Description:
    This is an interactive Streamlit web application designed to emulate the Global 21-cm Signal
    from the Cosmic Dawn and Epoch of Reionization.

    It utilizes a pre-trained Deep Neural Network (FCemu) to predict the differential brightness
    temperature (Tb) as a function of redshift (z), based on various astrophysical and cosmological
    parameters.

Key Features:
    - Interactive Sidebar: Allows users to vary specific physical parameters (e.g., f_star, L_X).
    - Real-time Inference: Runs the neural network prediction instantly upon parameter change.
    - Scientific Visualization: Plots the resulting global signal and provides physical context.
    - Optimized Performance: Uses a 'lite' dataset for fast initialization.

Author: [ron + roy / Team Name]
"""

import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.ndimage import gaussian_filter1d

# --- 1. CONFIGURATION ---
# Mandatory: Define Legacy Keras compatibility for the emulator model
os.environ["TF_USE_LEGACY_KERAS"] = "1"
from build_NN import FCemu

# --- 2. PATH CONFIGURATION ---
# Use relative paths to ensure the app runs on any machine (local or cloud)
# regardless of the user directory structure.

# Get the absolute path of the directory where this script (app.py) is located
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the model directory relative to the script location
MODEL_DIR = os.path.join(CURRENT_DIR, '100b_tr_set_model')

MODEL_NAME = '100b_model'
LITE_DATA_FILE = 'emulator_data_lite.pk'

# --- PARAMETER DESCRIPTIONS (Scientific) ---
# Dictionary mapping parameter names to their physical descriptions
PARAM_DESCRIPTIONS = {
    # Star Formation
    'F_STAR10': "Star Formation Efficiency ($f_{*,10}$): The fraction of gas converting to stars in halos of mass $10^{10} M_{\odot}$. Controls the intensity of the UV signal.",
    'ALPHA_STAR': "Star Formation Slope ($\\alpha_*$): Determines how star formation efficiency changes with halo mass. Positive values mean efficient formation in massive halos.",
    't_STAR': "Star Formation Timescale ($t_*$): The duration of star formation bursts as a fraction of the Hubble time. Affects how quickly galaxies evolve.",

    # Escape Fraction (Reionization)
    'F_ESC10': "Escape Fraction ($f_{esc,10}$): The fraction of ionizing UV photons escaping from halos of mass $10^{10} M_{\odot}$. This is the main driver of when Reionization happens.",
    'ALPHA_ESC': "Escape Fraction Slope ($\\alpha_{esc}$): How the escape fraction scales with halo mass. Critical for understanding which galaxies drive reionization.",
    'M_TURN': "Turnover Mass ($M_{turn}$): The halo mass threshold below which star formation is suppressed (due to feedback).",

    # X-rays (Heating)
    'L_X': "X-ray Luminosity ($L_X/SFR$): The energy output in X-rays per unit of star formation. Responsible for heating the gas (IGM) and creating the absorption trough.",
    'NU_X_THRESH': "X-ray Threshold ($E_0$): The minimum energy of X-ray photons capable of escaping the galaxy. Lower values mean softer X-rays that heat the gas locally.",
    'X_RAY_SPEC_INDEX': "X-ray Spectral Index ($\\alpha_X$): The slope of the X-ray power-law spectrum. Harder spectra (lower values) penetrate deeper into the universe.",

    # Cosmology
    'R_MFP': "Mean Free Path ($R_{mfp}$): The maximum distance ionizing photons can travel through the neutral gas.",
    'TAU_E': "Optical Depth ($\\tau_e$): The integrated electron scattering optical depth, a key cosmological constraint."
}

# --- 3. STYLING ---
# --- CSS Styling (Space Background + Navigation Bar) ---
page_bg_img = """
<style>
/* Define the main application background */
[data-testid="stAppViewContainer"] {
    background-color: black; /* Base color */
    /* Load transparent star pattern */
    background-image: url("https://www.transparenttextures.com/patterns/stardust.png");
    /* Repeat image to tile the screen */
    background-repeat: repeat;
    color: white; /* Keep text white */
}

[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}

/* (Optional) Header centering logic */
h1, h2, h3 {
    text-align: center;
}

/* --- NAVIGATION BAR STYLING --- */
/* Target the radio button container */
[data-testid="stRadio"] > div {
    display: flex;
    justify-content: center; /* Center the buttons */
    background-color: white; /* White background for the bar */
    padding: 10px;
    border-radius: 10px;
    width: 100%;
}

/* Target the text labels inside the radio buttons */
[data-testid="stRadio"] p {
    font-size: 18px !important;
    font-weight: bold;
    color: #007BFF !important; /* Blue Text */
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)


# --- 5. EMULATOR LOADER FUNCTION ---
# Note: Renamed to _v4 to force cache clearing if logic changes
@st.cache_resource(show_spinner=False)
def load_emulator_system_v4(model_dir, name):
    lite_path = os.path.join(model_dir, LITE_DATA_FILE)
    full_path = os.path.join(model_dir, 'training_files.pk')

    # Logic: Prioritize LITE file for speed
    if os.path.exists(lite_path):
        data_path = lite_path
    elif os.path.exists(full_path):
        data_path = full_path
    else:
        return None, None, None, None

    try:
        # Load Data
        if 'lite' in data_path:
            X_test = pickle.load(open(data_path, 'rb'))
        else:
            data = pickle.load(open(data_path, 'rb'))
            X_test = data[4]

        # Load Neural Network Emulator
        emulator = FCemu(restore=True, files_dir=model_dir, name=name)
        Z_BINS = emulator.z_glob

        # Return raw parameter names (bytes or strings), handled outside
        raw_names = emulator.param_names

        return emulator, X_test, Z_BINS, raw_names

    except Exception as e:
        print(f"Error loading system: {e}")
        return None, None, None, None

# --- NAVIGATION ---
# Updated list based on user request
nav_options = ["Home", "Cosmological Parameters", "Relevant Degeneracies", "About Us", "Credits"]

selected_page = st.radio(
    "Navigation", 
    nav_options,
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("---")

# --- PAGE CONTENT ---

if selected_page == "Home":
    # --- MOVED EMULATOR CONTENT HERE ---
    
    # Header
    st.markdown("<h1 style='text-align: center;'>The Global 21 cm Signal </h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Probing the Cosmic Dawn and Epoch of Reionization</h3>", unsafe_allow_html=True)

    st.write("""
    The 21-cm spectral line, corresponding to a rest-frame frequency of 1420 MHz, arises from the hyperfine transition of the ground state of neutral hydrogen. 
    This signal serves as a critical probe of the Early Universe, tracing the thermal history and ionization state of the Intergalactic Medium (IGM) from the Dark Ages through the Cosmic Dawn to the Epoch of Reionization (EoR).
    """)

    st.markdown("<h3 style='text-align: center;'>Theoretical Framework</h2>", unsafe_allow_html=True)
    st.write("""
    The observable quantity is the differential brightness temperature, $\delta T_b$, defined relative to the Cosmic Microwave Background (CMB). 
    The physics of the signal is governed by the contrast between the hydrogen spin temperature ($T_S$) and the background CMB temperature ($T_{CMB}$):
    """)

    # Scientific Equation
    st.latex(r"""
    \delta T_b \approx 27 \, x_{HI} \, (1 + \delta_b) \left( 1 - \frac{T_{CMB}}{T_S} \right) \left( \frac{1+z}{10} \right)^{1/2} \, [\text{mK}]
    """)

    st.write("""
    Where:
    - $x_{HI}$ is the neutral hydrogen fraction.
    - $\delta_b$ is the baryon overdensity.
    - $z$ is the redshift.
    - The ratio between  $ T_S $ and  $ T_{CMB} $ determines the signal regime:
        - **Absorption ($T_S < T_{CMB}$):** Negative signal (Deep trough).
        - **Emission ($T_S > T_{CMB}$):** Positive signal.
    """)

    st.markdown("---")

    # --- LOADER ---
    with st.spinner('Initializing Emulator System...'):
        emulator, X_test, Z_BINS, raw_param_names = load_emulator_system_v4(MODEL_DIR, MODEL_NAME)

    if emulator is None:
        st.error("System Error: Could not load emulator files. Please check paths and data files.")
        st.stop()

    # --- STRING CONVERSION ---
    param_names = []
    for p in raw_param_names:
        if isinstance(p, bytes):
            param_names.append(p.decode('utf-8'))
        else:
            param_names.append(str(p))

    # --- INTERACTIVE CONTROL ---
    st.subheader("Interactive Parameter Exploration")

    # Metrics
    min_vals = np.min(X_test, axis=0)
    max_vals = np.max(X_test, axis=0)
    mean_vals = np.mean(X_test, axis=0)
    num_params = X_test.shape[1]
    input_vector = mean_vals.copy()

    # Reset Button
    if st.button("Reset Parameters to Defaults"):
        for i in range(num_params):
            st.session_state[f"slider_{i}"] = float(mean_vals[i])

    # Sliders
    cols = st.columns(2)
    for i in range(num_params):
        p_name = param_names[i]
        
        # Friendly description mapping
        desc_key = p_name.strip()
        if desc_key not in PARAM_DESCRIPTIONS:
            for key in PARAM_DESCRIPTIONS:
                if key in desc_key:
                    desc_key = key
                    break
        p_desc = PARAM_DESCRIPTIONS.get(desc_key, f"Adjust {p_name}")

        current_min = float(min_vals[i])
        current_max = float(max_vals[i])
        current_default = float(mean_vals[i])

        with cols[i % 2]:
            val = st.slider(
                label=f"{p_name}",
                min_value=current_min,
                max_value=current_max,
                value=current_default,
                step=(current_max - current_min) / 100.0,
                help=p_desc,
                key=f"slider_{i}"
            )
            input_vector[i] = val
    
    # --- PREDICTION ---
    input_vector_batch = input_vector.reshape(1, -1)
    try:
        predictions = emulator.predict(input_vector_batch)
    except Exception:
        st.error("Emulator Prediction Failed")
        st.stop()

    # --- PLOTTING ---
    st.subheader(f"Global Signal Prediction")

    Tb_index = 3
    xHI_index = 1
    Tk_index = 4
    Ts_index = 5
    sample_idx = 0

    if len(predictions) > Ts_index:
        Tb_data = predictions[Tb_index][sample_idx]
        xHI_data = predictions[xHI_index][sample_idx]
        Tk_data = predictions[Tk_index][sample_idx]
        Ts_data = predictions[Ts_index][sample_idx]

        # Gaussian Smoothing
        Tb_data = gaussian_filter1d(Tb_data, sigma=1)

        if len(Z_BINS) == len(Tb_data):
            x_axis = Z_BINS
        else:
            x_axis = range(len(Tb_data))

        Tcmb_data = 2.725 * (1 + x_axis)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1]})

        # Plot 1: Tb
        ax1.plot(x_axis, Tb_data, color='#00ff00', linewidth=2.5, label=r'Brightness Temperature ($\delta T_b$)')
        ax1.set_ylabel(r'$\delta T_b$ [mK]', fontsize=12)
        ax1.set_title("Brightness Temperature ($\delta T_b$)", fontsize=14, color='white')
        ax1.set_xlim(5, 35)
        ax1.set_ylim(-200, 20)
        ax1.axhline(y=0, color='white', linestyle='--', alpha=0.5)
        ax1.grid(True, which='both', linestyle='--', alpha=0.3)
        ax1.legend(loc='lower right')

        # Plot 2: xHI
        ax2.plot(x_axis, xHI_data, color='cyan', linewidth=2.5, label='Neutral Fraction ($x_{HI}$)')
        ax2.set_ylabel(r'$x_{HI}$', fontsize=12)
        ax2.set_title("Neutral Hydrogen Fraction ($x_{HI}$)", fontsize=14, color='white')
        ax2.set_ylim(-0.1, 1.1)
        ax2.grid(True, which='both', linestyle='--', alpha=0.3)
        ax2.legend(loc='lower right')

        # Plot 3: Thermal History
        ax3.semilogy(x_axis, Tk_data, color='red', linewidth=2, label='$T_k$ (Gas Temp)')
        ax3.semilogy(x_axis, Ts_data, color='orange', linewidth=2, label='$T_s$ (Spin Temp)')
        ax3.semilogy(x_axis, Tcmb_data, color='white', linestyle='--', linewidth=2, label='$T_{cmb}$')

        ax3.set_ylabel('Temperature [K]', fontsize=12)
        ax3.set_xlabel('Redshift ($z$)', fontsize=12)
        ax3.set_title("Thermal History", fontsize=14, color='white')
        ax3.grid(True, which='major', linestyle='--', alpha=0.3)  # Major ticks only
        ax3.legend(loc='lower right')

        # Dark Theme Styling
        fig.patch.set_alpha(0.0)
        for ax in [ax1, ax2, ax3]:
            ax.set_facecolor((0, 0, 0, 0.2))
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('white')

        plt.subplots_adjust(hspace=0.3)
        st.pyplot(fig)
    else:
        st.error("Model output structure mismatch.")

elif selected_page == "Cosmological Parameters":
    st.title("Cosmological Parameters")
    st.write("Detailed explanation of the cosmological parameters used in this emulator will appear here.")
    st.write("Current Placeholders: F_STAR10, ALPHA_STAR, t_STAR, F_ESC10, ALPHA_ESC, M_TURN, L_X, NU_X_THRESH, X_RAY_SPEC_INDEX, R_MFP, TAU_E")

elif selected_page == "Relevant Degeneracies":
    st.title("Relevant Degeneracies")
    st.write("This section will discuss the physical degeneracies between different parameters that affect the 21-cm signal.")
    st.write("(e.g., Degeneracy between star formation efficiency and X-ray heating intensity during Cosmic Dawn)")

elif selected_page == "About Us":
    st.title("About Us")
    st.write("We are a research team dedicated to exploring the Epoch of Reionization.")

elif selected_page == "Credits":
    st.title("Credits & Acknowledgements")
    st.write("Special thanks to our supervisor and the open-source community.")
    st.write("Powered by Streamlit, TensorFlow, and Python.")
