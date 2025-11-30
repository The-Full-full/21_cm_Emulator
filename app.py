import streamlit as st
import numpy as np
import pandas as pd

# --- התחלת עיצוב ---
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

# --- אזור הסליידרים החדש (מסודר בעמודות) ---
st.subheader("Simulation Controls")

# יצירת שתי עמודות עבור TB ו-TCMB
col1, col2 = st.columns(2)

with col1:
    # סליידר עבור טמפרטורת הבהירות
    Tb = st.slider('Spin Temp ($T_B$) [mK]', -15.0, 100.0, 20.0)

with col2:
    # סליידר עבור טמפרטורת הרקע
    Tcmb = st.slider('Background ($T_{CMB}$) [mK]', 0.0, 100.0, 2.7)

# סליידר הסחה לאדום (מתחת לעמודות)
RedS = st.slider('Redshift ($Z$)', 0.0, 1100.0, 10.0)


# --- חישוב הגרף הפיזיקלי ---

# 1. חישוב האמפליטודה (הגובה של הגל) לפי היחס בין הטמפרטורות
# הוספנו את (1+RedS) למכנה כי האות נחלש ככל שההסחה לאדום גדולה יותר
amplitude = (Tb - Tcmb) / (1 + RedS/100)

# 2. יצירת ציר ה-X (תדרים סביב 1420)
x = np.linspace(1400, 1440, 200)

# 3. יצירת ציר ה-Y (צורת פעמון/גאוסיאן במקום סתם סינוס)
# זה מדמה קו ספקטרלי בודד ב-1420 מגה-הרץ
y = amplitude * np.exp(-0.5 * ((x - 1420)**2) / 2**2)

# יצירת ה-Dataframe לגרף עם שמות צירים ברורים
chart_data = pd.DataFrame({
    'Frequency (MHz)': x,
    'Brightness Temp (mK)': y
})

# כותרת דינמית שמשתנה לפי התוצאה
st.subheader(f'Signal Simulation (Amplitude: {amplitude:.2f} mK)')

# הצגת הגרף
# הגדרנו במפורש מה ציר X ומה ציר Y כדי שייראה טוב
st.line_chart(chart_data, x='Frequency (MHz)', y='Brightness Temp (mK)')


# סיכום נחמד (נשאר כמו שהיה)
st.write("""
This signal is crucial in astrophysics and cosmology because:
* It allows us to "see" **neutral hydrogen**, the main component of matter in the universe.
* It is used to **map the spiral arms** of our galaxy.
* It helps us probe the **"Cosmic Dawn"** – the era when the first stars formed.
""")