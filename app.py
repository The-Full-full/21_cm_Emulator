import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pickle

# --- 1. CONFIGURATION ---
# חובה: הגדרת תאימות ל-Keras הישן
os.environ["TF_USE_LEGACY_KERAS"] = "1"
from build_NN import FCemu

# --- 2. PATH CONFIGURATION ---
# וודא שהנתיב הזה נכון אצלך במחשב!
MODEL_DIR = r'C:\Users\roy18\PycharmProjects\21_cm_Emulator\100b_tr_set_model'
MODEL_NAME = '100b_model'
LITE_DATA_FILE = 'emulator_data_lite.pk'

# --- PARAMETER DESCRIPTIONS (Scientific) ---
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
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: linear-gradient(to right bottom, #2e0000, #4b0082);
    color: white;
}
[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}
.stSpinner > div {
    border-top-color: #00ff00 !important;
}
.stInfo {
    background-color: rgba(20, 20, 50, 0.8) !important;
    border-left: 5px solid #00ff00 !important;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# --- 4. HEADER & INTRODUCTION (Scientific) ---
st.title('The Global 21-cm Signal 🌌')
st.markdown("### Probing the Cosmic Dawn and Epoch of Reionization")

st.write("""
The 21-cm spectral line, corresponding to a rest-frame frequency of 1420 MHz, arises from the hyperfine transition of the ground state of neutral hydrogen. 
This signal serves as a critical probe of the Early Universe, tracing the thermal history and ionization state of the Intergalactic Medium (IGM) from the Dark Ages through the Cosmic Dawn to the Epoch of Reionization (EoR).
""")

st.subheader("Theoretical Framework")
st.write("""
The observable quantity is the differential brightness temperature, $\delta T_b$, defined relative to the Cosmic Microwave Background (CMB). 
The physics of the signal is governed by the contrast between the hydrogen spin temperature ($T_S$) and the background CMB temperature ($T_{CMB}$):
""")

# משוואת ה-Tb המדעית
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


# --- 5. EMULATOR LOADER FUNCTION ---
# שיניתי את השם ל-_v4 כדי לכפות רענון מוחלט
@st.cache_resource(show_spinner=False)
def load_emulator_system_v4(model_dir, name):
    lite_path = os.path.join(model_dir, LITE_DATA_FILE)
    full_path = os.path.join(model_dir, 'training_files.pk')

    if os.path.exists(lite_path):
        data_path = lite_path
    elif os.path.exists(full_path):
        data_path = full_path
    else:
        return None, None, None, None

    try:
        if 'lite' in data_path:
            X_test = pickle.load(open(data_path, 'rb'))
        else:
            data = pickle.load(open(data_path, 'rb'))
            X_test = data[4]

        emulator = FCemu(restore=True, files_dir=model_dir, name=name)
        Z_BINS = emulator.z_glob

        # מחזירים את השמות כמו שהם (raw), נטפל בהם בחוץ
        raw_names = emulator.param_names

        return emulator, X_test, Z_BINS, raw_names

    except Exception as e:
        print(f"Error loading system: {e}")
        return None, None, None, None


# --- 6. LOAD SYSTEM ---
with st.spinner('Initializing Emulator System...'):
    emulator, X_test, Z_BINS, raw_param_names = load_emulator_system_v4(MODEL_DIR, MODEL_NAME)

if emulator is None:
    st.error("System Error: Could not load emulator files.")
    st.stop()

# --- 7. SAFETY NET: STRING CONVERSION ---
# המרה כפויה של השמות כאן ועכשיו, מחוץ לפונקציה
# זה פותר את השגיאה ב-100%
param_names = []
for p in raw_param_names:
    if isinstance(p, bytes):
        param_names.append(p.decode('utf-8'))
    else:
        param_names.append(str(p))

# --- 8. INTERACTIVE CONTROL ---
st.subheader("Interactive Parameter Exploration")

st.write("**Parameters included in this model:**")
# עכשיו זה בטוח יעבוד כי param_names עבר המרה הרגע
st.info(", ".join(param_names))

min_vals = np.min(X_test, axis=0)
max_vals = np.max(X_test, axis=0)
mean_vals = np.mean(X_test, axis=0)
num_params = X_test.shape[1]

col1, col2 = st.columns([1, 2])

with col1:
    selected_param_index = st.selectbox(
        "Select Parameter to Vary:",
        options=range(num_params),
        format_func=lambda i: param_names[i] if i < len(param_names) else f"Param {i}"
    )

current_param_name = param_names[selected_param_index]

# שליפת התיאור
desc_key = current_param_name.strip()
if desc_key not in PARAM_DESCRIPTIONS:
    # ניסיון למצוא מפתח דומה אם יש בעיות רווחים
    for key in PARAM_DESCRIPTIONS:
        if key in desc_key:
            desc_key = key
            break

param_desc = PARAM_DESCRIPTIONS.get(desc_key, "Control this parameter to see its effect on the 21cm signal.")

with col2:
    st.info(f"**{current_param_name}:** {param_desc}")

    selected_value = st.slider(
        f"Adjust Value",
        min_value=float(min_vals[selected_param_index]),
        max_value=float(max_vals[selected_param_index]),
        value=float(mean_vals[selected_param_index]),
        step=(float(max_vals[selected_param_index]) - float(min_vals[selected_param_index])) / 100.0
    )

# --- 9. PREDICTION ---
input_vector = mean_vals.copy()
input_vector[selected_param_index] = selected_value
input_vector_batch = input_vector.reshape(1, -1)

try:
    predictions = emulator.predict(input_vector_batch)
except Exception:
    st.error("Emulator Prediction Failed")
    st.stop()

# --- 10. PLOTTING ---
st.subheader(f"Global Signal Prediction: $\delta T_b$ vs Redshift")

Tb_index = 3
sample_idx = 0

if len(predictions) > Tb_index:
    Tb_data = predictions[Tb_index][sample_idx]

    if len(Z_BINS) == len(Tb_data):
        x_axis = Z_BINS
    else:
        x_axis = range(len(Tb_data))

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(x_axis, Tb_data, color='#00ff00', linewidth=2.5, label='Emulator Prediction')

    ax.set_xlabel('Redshift ($z$)', fontsize=12)
    ax.set_ylabel(r'Brightness Temperature $\delta T_b$ [mK]', fontsize=12)
    ax.set_title(f'Effect of varying {current_param_name}', fontsize=14)

    ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(loc='upper right')

    fig.patch.set_alpha(0.0)
    ax.set_facecolor((0, 0, 0, 0.2))  # RGBA tuple fixed

    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_color('white')

    st.pyplot(fig)

else:
    st.error("Model output structure mismatch (Tb not found).")