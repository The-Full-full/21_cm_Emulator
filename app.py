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

# Force Matplotlib to use LaTeX-style formatting for math text and serif fonts
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'serif'
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
    display: none !important;
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

/* Prevent infinite stretching on ultrawide monitors to strictly maintain container ratios */
.main .block-container {
    max-width: 1400px !important;
}

/* --- COMPACT SLIDERS & BUTTONS --- */
/* Reduce internal padding and margins for each individual slider widget */
div[data-testid="stSlider"] {
    padding-bottom: 0px !important;
    padding-top: 0px !important;
    margin-bottom: -15px !important;
}

/* Force Plotly chart to be a perfect responsive square that fills the column */
.st-key-plotly_f_esc_vs_f_star_v6 iframe,
.st-key-plotly_f_star_vs_L_X_v6 iframe,
.st-key-plotly_f_esc_vs_L_X_v6 iframe {
    height: auto !important;
    aspect-ratio: 1 / 1 !important;
}

/* Custom Reset Button Styling */
div[data-testid="stButton"] button {
    background-color: rgba(255, 255, 255, 0.05) !important; /* אפור שקוף כהה */
    border: 1px solid rgba(255, 255, 255, 0.2) !important; /* מסגרת עדינה */
    color: white !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}

/* אפקט כשעוברים עם העכבר מעל הכפתור */
div[data-testid="stButton"] button:hover {
    background-color: rgba(255, 255, 255, 0.15) !important;
    border-color: rgba(255, 255, 255, 0.5) !important;
}

/* אפקט של לחיצה על הכפתור */
div[data-testid="stButton"] button:active {
    background-color: rgba(255, 255, 255, 0.2) !important;

}

/* Hide Streamlit Fullscreen Buttons */
button[title="View fullscreen"],
button[title="Fullscreen"],
[data-testid="StyledFullScreenButton"] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
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
nav_options = ["Home", "Astrophysical Parameters", "Relevant Degeneracies", "About Us & Credits"]

selected_page = st.radio(
    "Navigation", 
    nav_options,
    horizontal=True,
    label_visibility="collapsed"
)

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

num_params = len(param_names)
# --- PAGE CONTENT ---

if selected_page == "Home":
    
    # Header (Removed the duplicate main title)
    st.markdown("<div style='text-align: center; font-size: 1.5rem; font-weight: 500; margin-top: 10px; margin-bottom: 20px;'>Probing the Cosmic Dawn and Epoch of Reionization</div>", unsafe_allow_html=True)

    st.write("""
    The 21-cm spectral line, corresponding to a rest-frame frequency of 1420 MHz, arises from the hyperfine transition of the ground state of neutral hydrogen. 
    This signal serves as a critical probe of the Early Universe, tracing the thermal history and ionization state of the Intergalactic Medium (IGM) from the Dark Ages through the Cosmic Dawn to the Epoch of Reionization (EoR).
    """)

    # --- INTERACTIVE CONTROL ---
    st.subheader("Interactive Parameter Exploration" , anchor=False)

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
            Tb_data = gaussian_filter1d(Tb_data, sigma=2)

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
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), sharex=False, gridspec_kw={'height_ratios': [1, 1, 1]})

                freq_min = 1420.4 / (1 + 35) # approx 39.45 MHz (z=35)
                freq_max = 1420.4 / (1 + 5)  # approx 236.73 MHz (z=5)

                # Plot 1: Tb
                ax1.plot(freq_axis, Tb_data, color='BlueViolet', linewidth=2.5, label=r'$\rm{Brightness \,\, Temperature} \,\, (\delta T_b)$')
                ax1.set_ylabel(r"$\delta T_b \,\, [\rm{mK}]$", fontsize=12)
                ax1.set_xlim(freq_max, freq_min)
                if np.min(Tb_data) < -200:
                     ax1.set_ylim(np.min(Tb_data)*1.1, 20)
                else:
                     ax1.set_ylim(-250, 50)

                ax1.axhline(y=0, color='white', linestyle='--', alpha=0.5)
                ax1.grid(True, which='both', linestyle='--', alpha=0.3)
                ax1.legend(loc='lower right')

                # Plot 2: xHI
                ax2.plot(freq_axis, xHI_data, color='CornflowerBlue', linewidth=2.5, label=r'$\rm{Neutral \,\, Fraction} \,\, (x_{\rm{HI}})$')
                ax2.set_ylabel(r"$x_{\rm{HI}}$", fontsize=12)
                ax2.set_ylim(-0.1, 1.1)
                ax2.set_xlim(freq_max, freq_min)
                ax2.grid(True, which='both', linestyle='--', alpha=0.3)
                ax2.legend(loc='lower right')

                # Plot 3: Thermal History
                ax3.semilogy(freq_axis, Tk_data, color='red', linewidth=2, label=r'$T_k \,\, \rm{(Gas \,\, Temp)}$')
                ax3.semilogy(freq_axis, Ts_data, color='orange', linewidth=2, label=r'$T_s \,\, \rm{(Spin \,\, Temp)}$')
                ax3.semilogy(freq_axis, Tcmb_data, color='white', linestyle='--', linewidth=2, label=r'$T_{\rm{cmb}}$')

                ax3.set_ylabel(r"$\rm{Temperature \,\, [K]}$", fontsize=12)
                ax3.grid(True, which='major', linestyle='--', alpha=0.3)  # Major ticks only
                ax3.legend(loc='lower right')
                ax3.set_xlim(freq_max, freq_min)
                ax3.set_ylim(10**-2,10**4)
                
                # --- Primary X-Axis Frequency (Bottom, Linear) ---
                for ax in [ax1, ax2, ax3]:
                    ax.set_xlabel(r"$\rm{Frequency} \,\, (\rm{MHz})$", fontsize=12)
                
                # --- Secondary X-Axis Redshift (Top, Non-Linear) ---
                # Conversion functions (Frequency <-> Redshift)
                def freq_to_z(f):
                    return (1420.4 / f) - 1
                
                def z_to_freq(z):
                    return 1420.4 / (1 + z)

                for ax in [ax1, ax2, ax3]:
                    secax = ax.secondary_xaxis('top', functions=(freq_to_z, z_to_freq))
                    # Only add the label to the top-most plot to avoid clutter
                    if ax == ax1:
                        secax.set_xlabel(r"$\rm{Redshift} \,\, (z)$", fontsize=12, labelpad=10)
                    
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

elif selected_page == "Astrophysical Parameters":
    st.markdown("<div style='text-align: center; font-size: 1.5rem; font-weight: bold; margin-bottom: 10px; margin-top: 20px;'>Theoretical Framework</div>", unsafe_allow_html=True)
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

    st.markdown("<br>", unsafe_allow_html=True)

    st.write(r"""
        The spin temperature ($T_S$) itself is calculated based on its coupling to the CMB radiation, gas collisions, and the local Lyman-$\alpha$ radiation field:
        """)

    st.markdown("<br>", unsafe_allow_html=True)

    st.latex(r"""
        T_S^{-1} = \frac{T_{CMB}^{-1} + x_c T_k^{-1} + x_\alpha T_\alpha^{-1}}{1 + x_c + x_\alpha}
        """)

    st.write(r"""
        Where:
        - $x_c$ is the collisional coupling coefficient.
        - $x_\alpha$ is the Wouthuysen-Field (Lyman-$\alpha$) coupling coefficient.
        - $T_k$ is the kinetic temperature of the gas.
        - $T_\alpha$ is the color temperature of the radiation field (typically $T_\alpha \approx T_k$).
        """)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: left; font-size: 1.1rem; color: #a9a9a9; font-weight: bold;'>The astrophysical parameters defined below serve as the crucial inputs to these theoretical equations, directly governing the resulting 21-cm signal.<br>By adjusting these parameters in the emulator, you can explore how different physical conditions affect the predicted signal.</div>", unsafe_allow_html=True)

    st.markdown("---")
    
    st.markdown("<div style='text-align: center; font-size: 2.5rem; font-weight: bold; margin-bottom: 20px;'>Astrophysical Parameters</div>", unsafe_allow_html=True)

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
        height: 165px; /* Fixed height so all boxes are exactly the same size */
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        overflow: hidden; /* Prevent text spilling if it's too long */
    }
    .param-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.4);
    }
    .param-card-title {
        font-size: 1.15rem;
        font-weight: bold;
        margin-bottom: 12px;
        border-bottom: 1px solid rgba(255,255,255,0.2);
        padding-bottom: 10px;
        color: white;
    }
    .param-card-desc {
        font-size: 0.85rem; /* Slightly smaller text to ensure it fits perfectly */
        color: #e5e7eb;
        line-height: 1.5;
    }
    .category-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: white;
        margin-top: 10px;
        margin-bottom: 10px;
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

elif selected_page == "Relevant Degeneracies":
    st.markdown("""
        <style>
        div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div > div:first-child {
            background-color: #8b5cf6 !important;
        }
        div[data-testid="stSlider"] div[role="slider"] {
            background-color: #8b5cf6 !important;
            border-color: #8b5cf6 !important;
        }
        div[data-testid="stSlider"] * {
            --primary-color: #8b5cf6 !important;
        }
        div.element-container:has(#param-radio-marker) + div.element-container div[data-testid="stRadio"] {
            background-color: rgba(255, 255, 255, 0.05);
            padding: 12px 25px;
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.15);
            display: inline-block;
        }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; font-size: 2.5rem; font-weight: bold; margin-bottom: 20px;'>Relevant Degeneracies</div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; font-size: 1.2rem; margin-bottom: 40px; color: #a9a9a9;'>Explore astrophysical degeneracies and their impact on the global 21-cm signal in real-time.</div>", unsafe_allow_html=True)

    # --- 1. REAL MCMC DATA LOADER FOR FIGURE 7 ---
    @st.cache_data
    def load_real_mcmc_samples(pair_name):
        import pickle
        import os
        import glob
        
        # Try to load pre-calculated NPY files first (cloud-friendly)
        npy_path = None
        if pair_name == "f_esc vs f_star":
            npy_path = "f_esc_vs_f_star_filled.npy"
        elif pair_name == "f_star vs L_X":
            npy_path = "f_star_vs_L_X_filled.npy"
        else:  # f_esc vs L_X
            npy_path = "f_esc_vs_L_X_filled.npy"
            
        if npy_path and os.path.exists(npy_path):
            try:
                filled_data = np.load(npy_path)
                return filled_data[:, 0], filled_data[:, 1]
            except Exception as e:
                st.error(f"Error loading {npy_path}: {e}")
        else:
            # DEBUGGING: Print all npy files in the directory to see what actually exists on the server!
            all_npy_files = glob.glob("*.npy")
            st.error(f"Expected to find {npy_path}, but it doesn't exist on the server. Available NPY files: {all_npy_files}")
                
        # Fallback to raw .pk file
        mcmc_file_path = "MCMC_2023-08-09_all_fields_mini.pk"
        if not os.path.exists(mcmc_file_path):
            st.error(f"Data file {mcmc_file_path} not found.")
            return np.array([]), np.array([])
            
        with open(mcmc_file_path, "rb") as f:
            data = pickle.load(f)
            
        # Columns mapping based on MCMC analysis: 0: f_star, 4: f_esc, 7: L_X
        if pair_name == "f_esc vs f_star":
            return data[:, 0], data[:, 4]
        elif pair_name == "f_star vs L_X":
            return data[:, 0], data[:, 7]
        else:
            return data[:, 7], data[:, 4]

    # --- 2. INITIALIZE SESSION STATE ---
    # Default values are set to the medians of the real MCMC parameters
    if 'degen_val_f_star' not in st.session_state:
        st.session_state['degen_val_f_star'] = -1.253
    if 'degen_val_f_esc' not in st.session_state:
        st.session_state['degen_val_f_esc'] = -1.513
    if 'degen_val_l_x' not in st.session_state:
        st.session_state['degen_val_l_x'] = 40.491

    def format_pair(pair_name):
        mapping = {
            "f_esc vs f_star": r"$f_{\text{esc}}$ vs $f_*$",
            "f_star vs L_X": r"$f_*$ vs $L_X$",
            "f_esc vs L_X": r"$f_{\text{esc}}$ vs $L_X$"
        }
        return mapping.get(pair_name, pair_name)

    # --- 3. PARAMETER PAIR SELECTOR ---
    st.markdown('<span id="param-radio-marker"></span>', unsafe_allow_html=True)
    degen_pair = st.radio(
        "Select Parameter Pair to Analyze Degeneracy:",
        ["f_esc vs f_star", "f_star vs L_X", "f_esc vs L_X"],
        format_func=format_pair,
        horizontal=True
    )

    # Load data for the chosen pair
    x_data, y_data = load_real_mcmc_samples(degen_pair)

    # Map selected parameter pair to focused ranges (zooming in on the blue density region)
    if degen_pair == "f_esc vs f_star":
        x_label_pure = r"\log_{10}(f_{*,10})"
        y_label_pure = r"\log_{10}(f_{\text{esc},10})"
        x_label = "log<sub>10</sub>(<i>f</i><sub>*,10</sub>)"
        y_label = "log<sub>10</sub>(<i>f</i><sub>esc,10</sub>)"
        x_current = st.session_state['degen_val_f_star']
        y_current = st.session_state['degen_val_f_esc']
        x_key = 'degen_val_f_star'
        y_key = 'degen_val_f_esc'
    elif degen_pair == "f_star vs L_X":
        x_label_pure = r"\log_{10}(f_{*,10})"
        y_label_pure = r"\log_{10}(L_X / {\rm SFR})"
        x_label = "log<sub>10</sub>(<i>f</i><sub>*,10</sub>)"
        y_label = "log<sub>10</sub>(<i>L</i><sub>X</sub> / SFR)"
        x_current = st.session_state['degen_val_f_star']
        y_current = st.session_state['degen_val_l_x']
        x_key = 'degen_val_f_star'
        y_key = 'degen_val_l_x'
    else:  # f_esc vs L_X
        x_label_pure = r"\log_{10}(f_{\text{esc},10})"
        y_label_pure = r"\log_{10}(L_X / {\rm SFR})"
        x_label = "log<sub>10</sub>(<i>f</i><sub>esc,10</sub>)"
        y_label = "log<sub>10</sub>(<i>L</i><sub>X</sub> / SFR)"
        x_current = st.session_state['degen_val_f_esc']
        y_current = st.session_state['degen_val_l_x']
        x_key = 'degen_val_f_esc'
        y_key = 'degen_val_l_x'

    # Determine the original physical boundaries for data processing
    def get_data_limits(key):
        if key == 'degen_val_l_x': return (38.0, 42.0)
        return (-3.0, 0.0) # f_star and f_esc

    # Determine the visual boundaries for the axes (zoomed in)
    def get_axis_limits(key):
        if key == 'degen_val_l_x': return (38.0, 42.0)
        if key == 'degen_val_f_star': return (-2.25, -0.5)
        if key == 'degen_val_f_esc': return (-2.0, -0.25)
        return (-3.0, 0.0)

    data_x_min, data_x_max = get_data_limits(x_key)
    data_y_min, data_y_max = get_data_limits(y_key)

    x_min, x_max = get_axis_limits(x_key)
    y_min, y_max = get_axis_limits(y_key)

    key_degen_pair = degen_pair.replace(" ", "_")
    slider_x_key = f"degen_slider_{key_degen_pair}_x"
    slider_y_key = f"degen_slider_{key_degen_pair}_y"

    # Pre-emptively read widget state values from session_state if they exist.
    # When the user slides the controls, Streamlit puts the new values in
    # st.session_state[slider_x_key] and st.session_state[slider_y_key] BEFORE running the script.
    # By copying them into our source of truth coordinates here, the Plotly chart and Matplotlib
    # chart will both render with the new values in the current pass, with NO double-rerun needed.
    if slider_x_key in st.session_state and st.session_state[slider_x_key] is not None:
        st.session_state[x_key] = float(st.session_state[slider_x_key])
    if slider_y_key in st.session_state and st.session_state[slider_y_key] is not None:
        st.session_state[y_key] = float(st.session_state[slider_y_key])

    x_current = st.session_state[x_key]
    y_current = st.session_state[y_key]

    x_key = f"degen_val_{x_label_pure}" # Not used directly, overridden above

    @st.cache_data(show_spinner=False)
    def get_cached_plotly_base(degen_pair_name, min_x, max_x, min_y, max_y, data_x_min, data_x_max, data_y_min, data_y_max, xlabel, ylabel):
        import plotly.graph_objects as go
        from scipy.ndimage import gaussian_filter
        import copy
        
        # Calculate 2D density grid for the Heatmap using original data boundaries
        x_d, y_d = load_real_mcmc_samples(degen_pair_name)
        bins = 40
        h, x_edges, y_edges = np.histogram2d(x_d, y_d, bins=bins, range=[[data_x_min, data_x_max], [data_y_min, data_y_max]])
        h = gaussian_filter(h, sigma=1.0)
        
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
        z_data = h.T

        base_fig = go.Figure()

        custom_blues = [
            [0.0, 'rgba(0, 0, 0, 0.0)'],       
            [0.1, 'rgba(30, 58, 138, 0.15)'],   
            [0.3, 'rgba(29, 78, 216, 0.4)'],    
            [0.6, 'rgba(59, 130, 246, 0.7)'],   
            [1.0, 'rgba(147, 197, 253, 0.95)']  
        ]

        # Add 2D density Heatmap
        base_fig.add_trace(go.Heatmap(
            x=x_centers, y=y_centers, z=z_data,
            colorscale=custom_blues, showscale=False,
            hoverinfo='skip', zsmooth='best'
        ))

        # Add 2D density contours
        base_fig.add_trace(go.Contour(
            x=x_centers, y=y_centers, z=z_data,
            name='Contour Lines', contours=dict(coloring='none'),
            line=dict(width=1.5, color='rgba(56, 189, 248, 0.8)'),
            ncontours=8, hoverinfo='skip'
        ))

        # Generate a dense grid of invisible points to capture click events anywhere
        x_grid = np.linspace(min_x, max_x, 150)
        y_grid = np.linspace(min_y, max_y, 150)
        xx, yy = np.meshgrid(x_grid, y_grid)
        
        base_fig.add_trace(go.Scatter(
            x=xx.flatten(), y=yy.flatten(),
            mode='markers',
            marker=dict(color='rgba(0,0,0,0)', size=15, symbol='square'),
            hoverinfo='none', showlegend=False, name='click_grid'
        ))

        # Style Plotly figure
        base_fig.update_layout(
            autosize=True,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0.5)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=50, r=50, t=50, b=50), # Exact symmetric margins for perfect square
            showlegend=True,
            legend=dict(
                x=0.02, y=0.17,
                xanchor='left', yanchor='top',
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor='rgba(255,255,255,0.2)',
                borderwidth=1
            ),
            xaxis=dict(
                automargin=False, 
                gridcolor='rgba(255,255,255,0.1)', 
                zeroline=False, fixedrange=True, showline=True,
                linewidth=2, linecolor='white', mirror=True, range=[min_x, max_x]
            ),
            yaxis=dict(
                automargin=False, 
                gridcolor='rgba(255,255,255,0.1)', 
                zeroline=False, fixedrange=True, showline=True,
                linewidth=2, linecolor='white', mirror=True, range=[min_y, max_y],
                scaleanchor="x", scaleratio=1
            )
        )
        return base_fig
        
    import copy
    import plotly.graph_objects as go
    # Try to get from cache, or generate if not exists
    fig = copy.deepcopy(get_cached_plotly_base(
        degen_pair, 
        x_min, x_max, y_min, y_max,
        data_x_min, data_x_max, data_y_min, data_y_max,
        x_label, y_label
    ))

    # Add red marker for currently selected point
    fig.add_trace(go.Scatter(
        x=[x_current],
        y=[y_current],
        mode='markers',
        marker=dict(color='red', size=9, symbol='circle', line=dict(color='white', width=1.5)),
        name='Active Coordinate',
        hoverinfo='all'
    ))

    # --- 4. LAYOUT CREATION ---
    col_plot, col_predict = st.columns([1.0, 1.5], gap="medium")

    with col_plot:
        st.markdown("<div class='col-plot-anchor'></div>", unsafe_allow_html=True)
        with st.container(height=750, border=True):
            st.subheader("2D Posterior Degeneracy Map", anchor=False)

            # Active Parameter values readout
            st.latex(rf"\small \color{{#a78bfa}} {x_label_pure}: \,\, \color{{white}} {x_current:.2f} \quad | \quad \color{{#a78bfa}} {y_label_pure}: \,\, \color{{white}} {y_current:.2f}")

            # Interactive Map Instruction
            st.markdown("<div style='text-align: center; color: #a78bfa; font-size: 1.1em; margin-bottom: 5px; margin-top: 5px; font-weight: bold;'>Click anywhere on the map to set the parameters!</div>", unsafe_allow_html=True)

            # Render Chart and Capture Events
            event = st.plotly_chart(
                fig, 
                on_select="rerun", 
                selection_mode="points",
                use_container_width=True,
                key=f"plotly_{key_degen_pair}_v6", 
                config={'displayModeBar': False, 'scrollZoom': False}
            )

            # Check and handle click events (requires rerun to sync coordinates to slider defaults)
            if event:
                selection = None
                if hasattr(event, "selection"):
                    selection = event.selection
                elif isinstance(event, dict) and "selection" in event:
                    selection = event["selection"]
                
                if selection:
                    points = []
                    if hasattr(selection, "points"):
                        points = selection.points
                    elif isinstance(selection, dict) and "points" in selection:
                        points = selection["points"]
                    
                    if len(points) > 0:
                        x_coords = [p.get("x") for p in points if isinstance(p, dict) and p.get("x") is not None]
                        y_coords = [p.get("y") for p in points if isinstance(p, dict) and p.get("y") is not None]
                    
                        if len(x_coords) == 0:
                            x_coords = [p.x for p in points if hasattr(p, "x") and p.x is not None]
                        if len(y_coords) == 0:
                            y_coords = [p.y for p in points if hasattr(p, "y") and p.y is not None]
                        
                        if len(x_coords) > 0 and len(y_coords) > 0:
                            # Apply a tiny offset (-0.035) down and left to correct visual pointer hotspot illusion
                            click_x = np.mean(x_coords) - 0.035
                            click_y = np.mean(y_coords) - 0.035
                            st.session_state[x_key] = float(click_x)
                            st.session_state[y_key] = float(click_y)
                            st.session_state[slider_x_key] = float(click_x)
                            st.session_state[slider_y_key] = float(click_y)
                            st.rerun()

    with col_predict:
        st.markdown("<div class='col-plot-anchor'></div>", unsafe_allow_html=True)
        with st.container(height=750, border=True):
            st.subheader("Global Signal Prediction", anchor=False)
            # Spacer to align the top of the Matplotlib prediction plot exactly with the top of the Plotly plot
            st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)

            # Build input vector for emulator
            degen_input = np.zeros((1, num_params))
        
            active_2d_indices = []
            if degen_pair == "f_esc vs f_star":
                active_2d_indices = [2, 0]
            elif degen_pair == "f_star vs L_X":
                active_2d_indices = [0, 6]
            elif degen_pair == "f_esc vs L_X":
                active_2d_indices = [2, 6]
            
            for i in range(num_params):
                p_name = param_names[i].strip()
                if i in active_2d_indices:
                    if p_name == 'F_STAR10':
                        degen_input[0, i] = st.session_state['degen_val_f_star']
                    elif p_name == 'F_ESC10':
                        degen_input[0, i] = st.session_state['degen_val_f_esc']
                    elif p_name == 'L_X':
                        degen_input[0, i] = st.session_state['degen_val_l_x']
                elif p_name in ['R_MFP', 'TAU_E']:
                    degen_input[0, i] = (min_vals[i] + max_vals[i]) / 2.0
                else:
                    default_val = (min_vals[i] + max_vals[i]) / 2.0
                    degen_input[0, i] = st.session_state.get(f"degen_slider_{i}", default_val)

            # Clip values to emulator bounds
            for i in range(num_params):
                degen_input[0, i] = np.clip(degen_input[0, i], min_vals[i], max_vals[i])

            # Run Prediction
            try:
                degen_preds = emulator.predict(degen_input)
                degen_Tb = degen_preds[1][0]
            
                # Smooth
                degen_Tb = gaussian_filter1d(degen_Tb, sigma=2)

                # Redshift/Frequency Conversion Functions
                def freq_to_z_degen(f):
                    return (1420.4 / f) - 1
            
                def z_to_freq_degen(z):
                    return 1420.4 / (1 + z)

                if len(z_bins) == len(degen_Tb):
                    z_axis_degen = np.array(z_bins)
                else:
                    z_axis_degen = np.arange(len(degen_Tb))

                freq_axis_degen = 1420.4 / (1 + z_axis_degen)
                freq_min = 1420.4 / (1 + 35)
                freq_max = 1420.4 / (1 + 5)

                # Plot Tb using Matplotlib
                fig_pred, ax = plt.subplots(figsize=(7.5, 5.0), dpi=100)
                ax.plot(freq_axis_degen, degen_Tb, color='BlueViolet', linewidth=2.5, label=r'$\rm{Brightness \,\, Temp} \,\, (\delta T_b)$')
                ax.set_ylabel(r"$\delta T_b \,\, [\rm{mK}]$", fontsize=12)
                ax.set_xlabel(r"$\rm{Frequency} \,\, (\rm{MHz})$", fontsize=12)
                ax.set_xlim(freq_max, freq_min)
                if np.min(degen_Tb) < -200:
                    ax.set_ylim(np.min(degen_Tb)*1.1, 20)
                else:
                    ax.set_ylim(-250, 50)
                ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
                ax.grid(True, which='both', linestyle='--', alpha=0.3)
                ax.legend(loc='lower right')

                # Add Top Redshift Axis
                secax = ax.secondary_xaxis('top', functions=(freq_to_z_degen, z_to_freq_degen))
                secax.set_xlabel(r"$\rm{Redshift} \,\, (z)$", fontsize=12, labelpad=10)
                secax.tick_params(colors='white')
                secax.xaxis.label.set_color('white')
                for spine in secax.spines.values():
                    spine.set_color('white')

                # Dark theme formatting
                fig_pred.patch.set_alpha(0.0)
                ax.set_facecolor((0, 0, 0, 0.2))
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('white')
                for spine in ax.spines.values():
                    spine.set_color('white')

                st.pyplot(fig_pred, use_container_width=True)
            except Exception as e:
                st.error(f"Inference failed: {e}")


    st.markdown("<hr style='margin-top: -10px; margin-bottom: 15px;'>", unsafe_allow_html=True)
    st.subheader("Other Parameters", anchor=False)

    # We display sliders for the parameters not in the 2D plot
    slider_cols = st.columns(4)
    col_idx = 0

    for i in range(num_params):
        if i in active_2d_indices or param_names[i] in ['R_MFP', 'TAU_E']:
            continue
        
        p_name = param_names[i]
        desc_key = p_name.strip()
        if desc_key not in PARAM_DESCRIPTIONS:
            for key in PARAM_DESCRIPTIONS:
                if key in desc_key:
                    desc_key = key
                    break
                
        p_desc = PARAM_DESCRIPTIONS.get(desc_key, f"Adjust {p_name}")
        display_label = PARAM_LABELS.get(desc_key, p_name)
    
        current_min = float(min_vals[i])
        current_max = float(max_vals[i])
        current_default = (current_min + current_max) / 2.0
    
        if f"degen_slider_{i}" not in st.session_state:
            st.session_state[f"degen_slider_{i}"] = float(current_default)
        
        with slider_cols[col_idx % 4]:
            st.slider(
                label=display_label,
                min_value=current_min,
                max_value=current_max,
                step=(current_max - current_min) / 100.0,
                help=p_desc,
                key=f"degen_slider_{i}"
            )
        col_idx += 1

elif selected_page == "About Us & Credits":
    st.markdown("<div style='text-align: center; font-size: 2.5rem; font-weight: bold; margin-bottom: 30px;'>About Us & Credits</div>", unsafe_allow_html=True)
    
    combined_css = """
    <style>
    /* About Us & Credits CSS */
    .about-card {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.3), rgba(56, 189, 248, 0.15)); /* Light blue gradient */
        border: 1px solid rgba(56, 189, 248, 0.5);
        border-radius: 15px;
        padding: 30px;
        margin-bottom: 30px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .about-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 20px rgba(0,0,0,0.5);
    }
    .about-info {
        flex: 1;
        padding-right: 30px;
    }
    .about-name {
        font-size: 2.5rem;
        font-weight: bold;
        color: white;
        margin-bottom: 10px;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.5);
    }
    .about-desc {
        font-size: 1.15rem;
        color: rgba(255, 255, 255, 0.9);
        line-height: 1.6;
    }
    .about-img-container {
        width: 160px;
        height: 160px;
        flex-shrink: 0;
        border-radius: 12px;
        overflow: hidden;
        border: 3px solid rgba(255, 255, 255, 0.4);
        background-color: rgba(0, 0, 0, 0.2);
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .about-img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }

    /* Responsive logic for small screens */
    @media (max-width: 768px) {
        .about-card {
            flex-direction: column;
            text-align: center;
        }
        .about-info {
            padding-right: 0;
            margin-bottom: 20px;
        }
    }
    
    /* Credits CSS */
    .credit-section {
        background: rgba(255, 255, 255, 0.05);
        border-left: 5px solid rgba(167, 139, 250, 0.8); /* Purple accent */
        padding: 20px 25px;
        margin-bottom: 25px;
        border-radius: 0 10px 10px 0;
        transition: background 0.3s ease, transform 0.2s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .credit-section:hover {
        background: rgba(255, 255, 255, 0.1);
        transform: translateX(5px);
    }
    .credit-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #c4b5fd; /* Light purple */
        margin-bottom: 10px;
    }
    .credit-text {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.85);
        line-height: 1.6;
    }
    .credit-highlight {
        font-weight: bold;
        color: white;
    }
    </style>
    """
    st.markdown(combined_css, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    # 1st Card: Roy
    with col1:
        st.markdown("""
        <div class="about-card">
            <div class="about-info">
                <div class="about-name">Roy Badash</div>
                <div class="about-desc">
                    Roy is a second-year Physics undergraduate at Ben-Gurion University with a strong passion for astrophysics and cosmology. 
                </div>
            </div>
            <div class="about-img-container">
                <img class="about-img" src="https://api.dicebear.com/7.x/initials/svg?seed=Roy&backgroundColor=0ea5e9" alt="Roy">
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 2nd Card: Ron
    with col2:
        st.markdown("""
        <div class="about-card">
            <div class="about-info">
                <div class="about-name">Ron Rapoport</div>
                <div class="about-desc">
                    Ron is a second-year Physics undergraduate at Ben-Gurion University with a strong passion for astrophysics and cosmology. 
                </div>
            </div>
            <div class="about-img-container">
                <img class="about-img" src="https://api.dicebear.com/7.x/initials/svg?seed=Ron&backgroundColor=0ea5e9" alt="Ron">
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='text-align: center; font-size: 2rem; font-weight: bold; margin-top: 50px; margin-bottom: 30px;'>Credits & Acknowledgments</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="credit-section">
        <div class="credit-title">Academic Guidance & Mentorship</div>
        <div class="credit-text">
            We would like to express our deepest gratitude to our lecturer, <span class="credit-highlight">Ely Kovetz</span>, 
            and our doctoral instructor, <span class="credit-highlight">Hovav Lazare</span>. 
            Their guidance, expertise, and continuous feedback were invaluable to the success of this project.
        </div>
    </div>

    <div class="credit-section" style="border-left-color: rgba(96, 165, 250, 0.8);">
        <div class="credit-title">Powered By</div>
        <div class="credit-text">
            This interactive web application was brought to life using <span class="credit-highlight">Streamlit</span>. 
            The underlying Deep Neural Network, which enables real-time inferences of the 21cm Global Signal, 
            was built using <span class="credit-highlight">TensorFlow + Keras</span>, along with robust data processing 
            from <span class="credit-highlight">NumPy</span> and <span class="credit-highlight">SciPy</span>.
        </div>
    </div>

    <div class="credit-section" style="border-left-color: rgba(52, 211, 153, 0.8);">
        <div class="credit-title">Scientific Foundation</div>
        <div class="credit-text">
            The emulator is trained on advanced cosmological simulations representing the Cosmic Dawn 
            and Epoch of Reionization. A special thanks to the broader astrophysics community for the development 
            of semi-numerical simulation codes (such as 21cmFAST) that generate the vast datasets required for 
            training deep learning models in cosmology.<br><br>
            <i>For a comprehensive overview of the scientific foundation and deeper insights into the emulator, please refer to the <a href="https://arxiv.org/abs/2307.15577" target="_blank" style="color: #8b5cf6; text-decoration: underline;">published paper</a>.</i>
        </div>
    </div>
    """, unsafe_allow_html=True)
