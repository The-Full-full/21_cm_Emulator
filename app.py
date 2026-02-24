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

st.set_page_config(layout="wide", page_title="21cm Emulator")

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
MODEL_DIR = os.path.join(CURRENT_DIR, 'model_files')

MODEL_NAME = 'globals_model'

# --- PARAMETER DESCRIPTIONS (Scientific) ---
# Dictionary mapping parameter names to their physical descriptions
PARAM_DESCRIPTIONS = {
    # Star Formation
    'F_STAR10': r"Star Formation Efficiency ($f_{*,10}$): The fraction of gas converting to stars in halos of mass $10^{10} M_{\odot}$. Controls the intensity of the UV signal.",
    'ALPHA_STAR': "Star Formation Slope ($\\alpha_*$): Determines how star formation efficiency changes with halo mass. Positive values mean efficient formation in massive halos.",
    't_STAR': "Star Formation Timescale ($t_*$): The duration of star formation bursts as a fraction of the Hubble time. Affects how quickly galaxies evolve.",

    # Escape Fraction (Reionization)
    'F_ESC10': r"Escape Fraction ($f_{esc,10}$): The fraction of ionizing UV photons escaping from halos of mass $10^{10} M_{\odot}$. This is the main driver of when Reionization happens.",
    'ALPHA_ESC': "Escape Fraction Slope ($\\alpha_{esc}$): How the escape fraction scales with halo mass. Critical for understanding which galaxies drive reionization.",
    'M_TURN': "Turnover Mass ($M_{turn}$): The halo mass threshold below which star formation is suppressed (due to feedback).",

    # X-rays (Heating)
    'L_X': "X-ray Luminosity ($L_X/SFR$): The energy output in X-rays per unit of star formation. Responsible for heating the gas (IGM) and creating the absorption trough.",
    'NU_X_THRESH': "X-ray Threshold ($E_0$): The minimum energy of X-ray photons capable of escaping the galaxy. Lower values mean softer X-rays that heat the gas locally.",
    'X_RAY_SPEC_INDEX': "X-ray Spectral Index ($\\alpha_X$): The slope of the X-ray power-law spectrum. Harder spectra (lower values) penetrate deeper into the universe.",

    # Cosmology
    # Parameters Mean Free Path and Optical Depth removed
}

# Dictionary mapping parameter names to their LaTeX display labels
PARAM_LABELS = {
    'F_STAR10': r'$f_{*,10}$',
    'ALPHA_STAR': r'$\alpha_*$',
    't_STAR': r'$t_*$',
    'F_ESC10': r'$f_{esc,10}$',
    'ALPHA_ESC': r'$\alpha_{esc}$',
    'M_TURN': r'$M_{turn}$',
    'L_X': r'$L_X/SFR$',
    'NU_X_THRESH': r'$E_0$',
    'X_RAY_SPEC_INDEX': r'$\alpha_X$'
}

# --- 3. STYLING ---
# --- CSS Styling (Space Background + Navigation Bar) ---

page_bg_img = """
<style>
/* Define the main application background */
[data-testid="stAppViewContainer"] {
    background-color: black;
    background-image: url("https://www.transparenttextures.com/patterns/stardust.png");
    background-repeat: repeat;
    color: white;
}

[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}

h1, h2, h3 {
    text-align: center;
}

/* --- NAVIGATION BAR STYLING (CENTERING FIX) --- */

/* Ensure the outermost wrapper is always centered, adapting to any screen size */
div[data-testid="stElementContainer"]:has([data-testid="stRadio"]) {
    display: flex !important;
    justify-content: center !important;
    width: 100% !important;
}

/* Center the main container of the radio widget */
[data-testid="stRadio"] {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    width: 100% !important;
    margin-top: 40px !important;
}

/* Form the actual styling of the menu background itself */
[data-testid="stRadio"] > div {
    background-color: rgba(255, 255, 255, 0.1) !important;
    padding: 10px 30px !important;
    border-radius: 20px !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    
    display: inline-flex !important;
    flex-direction: row !important;
    align-items: center !important;
    justify-content: center !important;
    width: max-content !important; 
    margin: 0 auto !important;
    flex-wrap: wrap !important; /* Prevents breaking on very small screens */
}

/* Style the text inside */
[data-testid="stRadio"] label p {
    font-size: 18px !important;
    color: white !important;
    font-weight: bold !important;
}

/* --- COMPACT SLIDERS & BUTTONS --- */
/* Reduce internal padding and margins for each individual slider widget */
div[data-testid="stSlider"] {
    padding-bottom: 0px !important;
    padding-top: 0px !important;
    margin-bottom: -15px !important;
}

/* Ensure the Reset button text stays on one line */
div[data-testid="stButton"] button p {
    white-space: nowrap !important;
    font-size: 14px !important;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)


# --- 5. EMULATOR LOADER FUNCTION ---
# Note: Renamed to _v5 to force cache clearing
@st.cache_resource(show_spinner=False)
def load_emulator_system_v5(model_dir, name):
    try:
        # Load Neural Network Emulator
        # The FCemu restore method automatically reads 'model_data.h5'
        emulator = FCemu(restore=True, files_dir=model_dir, name=name)
        return emulator

    except Exception as e:
        print(f"Error loading system: {e}")
        return None

# --- NAVIGATION ---

# Global Header
st.markdown("<div style='text-align: center; color: white; margin-bottom: -20px; font-size: 2.5rem; font-weight: bold;'>The Global 21 cm Signal</div>", unsafe_allow_html=True)

# Updated list based on user request
nav_options = ["Home", "Cosmological Parameters"]

selected_page = st.radio(
    "Navigation", 
    nav_options,
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("---")

# --- PAGE CONTENT ---

if selected_page == "Home":
    
    # Header (Removed the duplicate main title)
    st.markdown("<div style='text-align: center; font-size: 1.5rem; font-weight: 500; margin-top: 10px; margin-bottom: 20px;'>Probing the Cosmic Dawn and Epoch of Reionization</div>", unsafe_allow_html=True)

    st.write("""
    The 21-cm spectral line, corresponding to a rest-frame frequency of 1420 MHz, arises from the hyperfine transition of the ground state of neutral hydrogen. 
    This signal serves as a critical probe of the Early Universe, tracing the thermal history and ionization state of the Intergalactic Medium (IGM) from the Dark Ages through the Cosmic Dawn to the Epoch of Reionization (EoR).
    """)

    st.markdown("<div style='text-align: center; font-size: 1.5rem; font-weight: bold; margin-bottom: 10px;'>Theoretical Framework</div>", unsafe_allow_html=True)
    st.write(r"""
    The observable quantity is the differential brightness temperature, $\delta T_b$, defined relative to the Cosmic Microwave Background (CMB). 
    The physics of the signal is governed by the contrast between the hydrogen spin temperature ($T_S$) and the background CMB temperature ($T_{CMB}$):
    """)

    # Scientific Equation
    st.latex(r"""
    \delta T_b \approx 27 \, x_{HI} \, (1 + \delta_b) \left( 1 - \frac{T_{CMB}}{T_S} \right) \left( \frac{1+z}{10} \right)^{1/2} \, [\text{mK}]
    """)

    st.write(r"""
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
    with st.spinner('Initializing Emulator System (New Model)...'):
        if 'emulator_loaded' not in st.session_state:
            emulator = load_emulator_system_v5(MODEL_DIR, MODEL_NAME)
            st.session_state['emulator_loaded'] = emulator
        else:
            emulator = st.session_state['emulator_loaded']

    if emulator is None:
        st.error(f"System Error: Could not load emulator files from {MODEL_DIR}. Please check the files and try again.")
        st.stop()

    # --- METADATA EXTRACTION ---
    # Extract parameter info directly from the loaded emulator object
    raw_param_names = emulator.param_names
    min_vals = emulator.tr_params_min
    max_vals = emulator.tr_params_max
    z_bins = emulator.z_glob

    # Decode bytes if necessary
    param_names = []
    for p in raw_param_names:
        if isinstance(p, bytes):
            param_names.append(p.decode('utf-8'))
        else:
            param_names.append(str(p))

    # --- INTERACTIVE CONTROL ---
    st.subheader("Interactive Parameter Exploration" , anchor=False)

    num_params = len(param_names)
    input_vector = np.zeros(num_params)

    # Sliders

    # Create layout: Left (Controls) takes 1 part, Right (Graphs) takes 3 parts
    col_controls, col_graphs = st.columns([1, 3], gap="medium")

    # --- Left Side: Sliders ---
    with col_controls:
        # We put all controls in a container with a fixed height (~850px)
        # to match the approximate height of the 3 graphs on the right side.
        # This prevents the slider column from looking vastly different lengths on different screens,
        # by simply adding an internal scrollbar if it exceeds the height.
        with st.container(height=650):
            # הוספת anchor=False מבטלת את סמל הקישור שמופיע מתחת לכותרת
            st.subheader("Parameters", anchor=False)

            # Reset Button - קיצור הטקסט כדי לתפוס פחות מקום
            if st.button("Reset Parameters"):
                for i in range(num_params):
                    default_val = (min_vals[i] + max_vals[i]) / 2.0
                    st.session_state[f"slider_{i}"] = float(default_val)
            # Sliders Loop - Simpler, in a single column
            for i in range(num_params):
                p_name = param_names[i]
                
                # Pre-calculate defaults since we might skip the UI render
                current_min = float(min_vals[i])
                current_max = float(max_vals[i])
                current_default = (current_min + current_max) / 2.0

                # Hide unused parameters but keep them in the input vector with default values
                if p_name in ['R_MFP', 'TAU_E']:
                    input_vector[i] = current_default
                    continue

                # (Slider description logic remains the same...)
                desc_key = p_name.strip()
                if desc_key not in PARAM_DESCRIPTIONS:
                    for key in PARAM_DESCRIPTIONS:
                        if key in desc_key:
                            desc_key = key
                            break
                p_desc = PARAM_DESCRIPTIONS.get(desc_key, f"Adjust {p_name}")

                display_label = PARAM_LABELS.get(desc_key, p_name)

                # Initialize session state for this slider if it doesn't exist
                # This prevents Streamlit from warning about value conflicts
                if f"slider_{i}" not in st.session_state:
                    st.session_state[f"slider_{i}"] = float(current_default)

                # Create Slider (without explicit value= parameter)
                val = st.slider(
                    label=display_label,
                    min_value=current_min,
                    max_value=current_max,
                    step=(current_max - current_min) / 100.0,
                    help=p_desc,
                    key=f"slider_{i}"
                )
                input_vector[i] = val
    # --- Right Side: Graphs ---
    with col_graphs:
        # All Prediction and Plotting code goes here

        # --- PREDICTION ---
        input_vector_batch = input_vector.reshape(1, -1)
        try:
            predictions = emulator.predict(input_vector_batch)
        except Exception as e:
            st.error(f"Emulator Prediction Failed: {e}")
            st.stop()

        # --- PLOTTING ---
        st.subheader("Global Signal Prediction", anchor=False)

        # New Model Indices (Verified)
        xHI_index = 0
        Tb_index = 1
        Tk_index = 2
        Ts_index = 3

        sample_idx = 0

        if len(predictions) > Ts_index:
            xHI_data = predictions[xHI_index][sample_idx]
            Tb_data = predictions[Tb_index][sample_idx]
            Tk_data = predictions[Tk_index][sample_idx]
            Ts_data = predictions[Ts_index][sample_idx]

            # Gaussian Smoothing (Apply to Tb)
            #Tb_data = gaussian_filter1d(Tb_data, sigma=1)

            # X-Axis Logic
            if len(z_bins) == len(Tb_data):
                z_axis = np.array(z_bins)
            else:
                z_axis = np.arange(len(Tb_data))

            freq_axis = 1420.4 / (1 + z_axis)

            Tcmb_data = 2.725 * (1 + z_axis)

            # Wrap plots in a container to maintain a consistent height, matching the left column
            with st.container(height=590):
                # Reduced figsize height from 16 to 11 to help it fit on smaller screens without too much scrolling
                fig, (ax3, ax2, ax1) = plt.subplots(3, 1, figsize=(12, 14), sharex=False, gridspec_kw={'height_ratios': [1, 1, 1]})

                freq_min = 1420.4 / (1 + 35) # approx 39.45 MHz (z=35)
                freq_max = 1420.4 / (1 + 5)  # approx 236.73 MHz (z=5)

                # Plot 1: Tb
                ax1.plot(freq_axis, Tb_data, color='BlueViolet', linewidth=2.5, label=r'Brightness Temperature ($\delta T_b$)')
                ax1.set_ylabel(r"$\delta T_b$ [mK]", fontsize=12)
                ax1.set_xlim(freq_min, freq_max)
                if np.min(Tb_data) < -200:
                     ax1.set_ylim(np.min(Tb_data)*1.1, 20)
                else:
                     ax1.set_ylim(-250, 50)

                ax1.axhline(y=0, color='white', linestyle='--', alpha=0.5)
                ax1.grid(True, which='both', linestyle='--', alpha=0.3)
                ax1.legend(loc='lower right')

                # Plot 2: xHI
                ax2.plot(freq_axis, xHI_data, color='CornflowerBlue', linewidth=2.5, label='Neutral Fraction ($x_{HI}$)')
                ax2.set_ylabel(r"$x_{HI}$", fontsize=12)
                ax2.set_ylim(-0.1, 1.1)
                ax2.set_xlim(freq_min, freq_max)
                ax2.grid(True, which='both', linestyle='--', alpha=0.3)
                ax2.legend(loc='lower right')

                # Plot 3: Thermal History
                ax3.semilogy(freq_axis, Tk_data, color='red', linewidth=2, label='$T_k$ (Gas Temp)')
                ax3.semilogy(freq_axis, Ts_data, color='orange', linewidth=2, label='$T_s$ (Spin Temp)')
                ax3.semilogy(freq_axis, Tcmb_data, color='white', linestyle='--', linewidth=2, label='$T_{cmb}$')

                ax3.set_ylabel(r"$Temperature [K]$", fontsize=12)
                ax3.grid(True, which='major', linestyle='--', alpha=0.3)  # Major ticks only
                ax3.legend(loc='lower right')
                ax3.set_xlim(freq_min, freq_max)
                ax3.set_ylim(10**-2,10**4)
                
                # --- Primary X-Axis Frequency (Bottom, Linear) ---
                for ax in [ax1, ax2, ax3]:
                    ax.set_xlabel(r"Frequency (MHz)", fontsize=12)
                
                # --- Secondary X-Axis Redshift (Top, Non-Linear) ---
                # Conversion functions (Frequency <-> Redshift)
                def freq_to_z(f):
                    return (1420.4 / f) - 1
                
                def z_to_freq(z):
                    return 1420.4 / (1 + z)

                for ax in [ax1, ax2, ax3]:
                    secax = ax.secondary_xaxis('top', functions=(freq_to_z, z_to_freq))
                    # Only add the label to the top-most plot to avoid clutter
                    if ax == ax3:
                        secax.set_xlabel(r"Redshift ($z$)", fontsize=12, labelpad=10)
                    
                    # Style the secondary axis to match the dark theme
                    secax.tick_params(colors='white')
                    secax.xaxis.label.set_color('white')
                    for spine in secax.spines.values():
                        spine.set_color('white')

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

                plt.subplots_adjust(hspace=0.45) # Increased hspace to make room for the new top axes
                st.pyplot(fig)
        else:
            st.error("Model output structure mismatch. Check if the model is producing all 4 expected outputs.")

elif selected_page == "Cosmological Parameters":
    st.markdown("<div style='text-align: center; font-size: 2.5rem; font-weight: bold;'>Cosmological Parameters</div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; font-size: 1.2rem; margin-top: 10px; margin-bottom: 40px; color: #a9a9a9;'>Physical constants and variables driving the emulator</div>", unsafe_allow_html=True)

    # Group parameters by scientific category to create rows
    param_groups = {
        "Star Formation": {
            "keys": ['F_STAR10', 'ALPHA_STAR', 't_STAR']
        },
        "Reionization": {
            "keys": ['F_ESC10', 'ALPHA_ESC', 'M_TURN']
        },
        "Heating (X-rays)": {
            "keys": ['L_X', 'NU_X_THRESH', 'X_RAY_SPEC_INDEX']
        }
    }
    
    # The 3 chosen colors: Blue, Green, Purple
    card_colors = [
        'rgba(30, 64, 175, 0.4)',  # Indigo/Blue
        'rgba(6, 95, 70, 0.4)',    # Emerald/Green
        'rgba(76, 29, 149, 0.4)',  # Deep Purple
    ]
    
    # CSS for the custom HTML cards
    st.markdown("""
    <style>
    .param-card {
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 25px;
        border: 1px solid rgba(255,255,255,0.1);
        height: 250px; /* Fixed height so all boxes are exactly the same size */
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        overflow: hidden; /* Prevent text spilling if it's too long */
    }
    .param-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.4);
    }
    .param-card-title {
        font-size: 1.35rem;
        font-weight: bold;
        margin-bottom: 12px;
        border-bottom: 1px solid rgba(255,255,255,0.2);
        padding-bottom: 10px;
        color: white;
    }
    .param-card-desc {
        font-size: 0.95rem; /* Slightly smaller text to ensure it fits perfectly */
        color: #e5e7eb;
        line-height: 1.5;
    }
    .category-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: white;
        margin-top: 30px;
        margin-bottom: 15px;
        padding-bottom: 5px;
        border-bottom: 2px solid rgba(255,255,255,0.3);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Map LaTeX to HTML
    latex_to_html = {
        r"$f_{*,10}$": "<i>f</i><sub>*,10</sub>",
        r"$\alpha_*$": "&alpha;<sub>*</sub>",
        r"$t_*$": "<i>t</i><sub>*</sub>",
        r"$f_{esc,10}$": "<i>f</i><sub>esc,10</sub>",
        r"$\alpha_{esc}$": "&alpha;<sub>esc</sub>",
        r"$M_{turn}$": "<i>M</i><sub>turn</sub>",
        r"$L_X/SFR$": "<i>L<sub>X</sub></i> / SFR",
        r"$E_0$": "<i>E</i><sub>0</sub>",
        r"$\alpha_X$": "&alpha;<sub>X</sub>",
        r"$10^{10} M_{\odot}$": "10<sup>10</sup> M<sub>&#8857;</sub>"
    }

    # Iterate over the 3 categories to build 3 separate rows
    for row_idx, (category_name, group_info) in enumerate(param_groups.items()):
        st.markdown(f"<div class='category-header'>{category_name}</div>", unsafe_allow_html=True)
        cols = st.columns(3, gap="medium")
        
        for col_idx, key in enumerate(group_info["keys"]):
            if key not in PARAM_DESCRIPTIONS:
                continue
                
            val = PARAM_DESCRIPTIONS[key]
            
            # Translate LaTeX
            for tex, html in latex_to_html.items():
                val = val.replace(tex, html)
                
            # Split title and description
            if ": " in val:
                title, desc = val.split(": ", 1)
            else:
                title, desc = key, val
            
            # Stagger the colors across columns and rows dynamically
            # Row 0 starts at color 0 (Blue, Green, Purple)
            # Row 1 starts at color 2 (Purple, Blue, Green) 
            # Row 2 starts at color 1 (Green, Purple, Blue)
            color_offset = (row_idx * 2) % 3
            bg_color = card_colors[(col_idx + color_offset) % 3]
                
            with cols[col_idx]:
                st.markdown(f"""
                <div class="param-card" style="background-color: {bg_color};">
                    <div class="param-card-title">{title}</div>
                    <div class="param-card-desc">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

