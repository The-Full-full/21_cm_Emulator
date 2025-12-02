import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# 🔴 שינוי 1: הגדרת תאימות ל-Keras הישן (חייב להיות ראשון!)
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# 🔴 שינוי 2: ייבוא המחלקה של האמולטור מהקובץ שלך
# וודא שהקובץ build_NN.py נמצא באותה תיקייה!
from build_NN import FCemu

# --- התחלת עיצוב (נשאר ללא שינוי) ---
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: linear-gradient(to right bottom, #2e0000, #4b0082);
    color: white;
}
[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
# --- סוף עיצוב ---

# כותרת
st.title('The 21 cm Signal 🌌')

# טקסט
st.header('Analysis of the Hydrogen Line')
st.write("""
The 21cm line (or hydrogen line) is a radio-frequency spectral line created by a change in the energy state of neutral hydrogen atoms. 
The wave has a frequency of 1420.40575 MHz and a corresponding wavelength of 21.106 cm.
This signal is crucial in astrophysics and cosmology as it allows us to "see" neutral hydrogen, the main component of matter in the universe. 
It's used to map the spiral arms of our galaxy, study other galaxies, and most importantly, to probe the "Cosmic Dawn" - the era when the first stars and galaxies formed and reionized the universe.
""")

# הסבר פרמטרים
st.subheader("Key Parameters")
st.write("""
- **Spin Temperature ($T_B$):** An effective temperature describing the population ratio of the two hydrogen energy levels.
- **Background Temperature ($T_{CMB}$):** The temperature of the Cosmic Microwave Background at that epoch.
- **Neutral Hydrogen Fraction (X):** The relative fraction of hydrogen that is neutral (not ionized).
- **Matter Density (P) & Redshift (Z):** These determine the overall density of hydrogen atoms.
""")

# פסקה עם קצת הסבר
st.write("""
The signal appears in emission ($T_S > T_{CMB}$) or absorption ($T_S < T_{CMB}$), 
and its magnitude is proportional to the temperature difference and the amount of neutral hydrogen.
""")


# 🔴 שינוי 3: פונקציה חכמה לטעינת המודל (Cache)
# הפונקציה הזו רצה רק פעם אחת כדי לחסוך זמן טעינה
@st.cache_resource
def load_emulator_system(model_dir, name):
    # נתיב לקובץ הנתונים
    data_path = os.path.join(model_dir, 'training_files.pk')

    if not os.path.exists(data_path):
        st.error(f"Error: Could not find data file at {data_path}")
        return None, None, None

    # טעינת הנתונים
    data = pickle.load(open(data_path, 'rb'))
    X_train, _, X_val, _, X_test, Y_test = data

    # טעינת המודל
    emulator = FCemu(restore=True, files_dir=model_dir, name=name)
    Z_BINS = emulator.z_glob  # ציר ה-Redshift

    return emulator, X_test, Z_BINS


# 🔴 שינוי 4: הגדרת נתיבים (החלפנו את הסליידרים הידניים בטעינת מודל)
# עדכן את הנתיב הזה לנתיב המדויק במחשב שלך!
MODEL_DIR = r'C:\Users\roy18\PycharmProjects\21_cm_Emulator\100b_tr_set_model'
MODEL_NAME = '100b_model'

# כפתור טעינה (כדי לא לתקוע את האתר מיד בהתחלה)
st.subheader("Neural Network Emulator")

# טעינת המודל בפועל
with st.spinner('Loading Neural Network Model...'):
    emulator, X_test, Z_BINS = load_emulator_system(MODEL_DIR, MODEL_NAME)

if emulator is None:
    st.warning("Please check the MODEL_DIR path in the code.")
    st.stop()

# 🔴 שינוי 5: החלפת הסליידרים והגרף המזויף בכפתור סימולציה אמיתי
if st.button('🎲 Run Random Simulation from Test Set'):

    # 1. בחירת דוגמה אקראית
    random_idx = np.random.randint(0, len(X_test))
    st.info(f"Running simulation on sample index: #{random_idx}")

    # 2. הרצת החיזוי (Prediction)
    sample_input = X_test[random_idx:random_idx + 1]
    predictions = emulator.predict(sample_input)

    # 3. הכנת הנתונים לגרפים
    sample_idx = 0

    # הגדרת ציר X (בדיקה אם יש ציר Z אמיתי או שנשתמש באינדקסים)
    if len(predictions) > 1 and len(Z_BINS) == len(predictions[1][sample_idx]):
        x_axis = Z_BINS
        xlabel_z = 'Redshift (z)'
    else:
        x_axis = range(len(predictions[1][sample_idx]))
        xlabel_z = 'Index'

    # --- יצירת הגרפים האמיתיים ---
    st.subheader("Simulation Results")

    col1, col2 = st.columns(2)

    # גרף 1: Power Spectrum
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ps_data = predictions[0][sample_idx, :, 0, 0]
        ax1.plot(ps_data, 'b-', linewidth=2)
        ax1.set_title('Power Spectrum (z=z_0)')
        ax1.set_xlabel('k bins')
        ax1.set_ylabel('Power')
        ax1.grid(True)
        st.pyplot(fig1)

    # גרף 2: Brightness Temperature (הכי חשוב!)
    with col2:
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        # Tb נמצא בדרך כלל באינדקס 3
        if len(predictions) > 3:
            Tb_data = predictions[3][sample_idx]
            ax3.plot(x_axis, Tb_data, 'g-', linewidth=2)
            ax3.set_title('Brightness Temperature (Tb)')
            ax3.set_xlabel(xlabel_z)
            ax3.set_ylabel('mK')
            ax3.grid(True)
            ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            st.pyplot(fig3)
        else:
            st.write("Tb data not found in model output")

    # גרף 3: Spin Temperature
    if len(predictions) > 5:
        st.subheader("Spin Temperature vs Kinetic Temperature")
        fig5, ax5 = plt.subplots(figsize=(10, 4))

        # Ts
        Ts_data = predictions[5][sample_idx]
        ax5.plot(x_axis, Ts_data, 'm-', linewidth=2, label='Spin Temp (Ts)')

        # Tk (אם קיים)
        if len(predictions) > 4:
            Tk_data = predictions[4][sample_idx]
            ax5.plot(x_axis, Tk_data, color='orange', linewidth=2, linestyle='--', label='Kinetic Temp (Tk)')

        ax5.set_title('Temperatures Evolution')
        ax5.set_xlabel(xlabel_z)
        ax5.set_ylabel('Temperature [K]')
        ax5.set_yscale('log')
        ax5.legend()
        ax5.grid(True, which="both", ls="-")
        st.pyplot(fig5)

# סיכום נחמד (נשאר ללא שינוי)
st.write("""
This signal is crucial in astrophysics and cosmology because:
* It allows us to "see" **neutral hydrogen**, the main component of matter in the universe.
* It is used to **map the spiral arms** of our galaxy.
* It helps us probe the **"Cosmic Dawn"** – the era when the first stars formed.
""")