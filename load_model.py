import os

# --- תיקון תאימות (חייב להיות השורה הראשונה בקובץ) ---
os.environ["TF_USE_LEGACY_KERAS"] = "1"
# -----------------------------------------------------

import numpy as np
import pickle
import matplotlib.pyplot as plt
from build_NN import FCemu


def load_model_and_predict(model_dir, name,
                           testing_files_dir=None,
                           ):
    if testing_files_dir is None:
        testing_files_dir = model_dir

    # --- הקוד המקורי של טעינת הנתונים ---
    print("Loading data...")
    data = pickle.load(open(f'{testing_files_dir}/training_files.pk', 'rb'))
    X_train, _, X_val, _, X_test, Y_test = data

    # --- טעינת המודל ---
    print("Loading model...")
    emulator = FCemu(restore=True, files_dir=model_dir, name=name)
    Z_BINS = emulator.z_glob  # ציר ה-Redshift

    # --- ביצוע התחזית ---
    print("Running prediction (this might take a moment)...")
    predictions = emulator.predict(X_test)

    # חישובים מקוריים של הקוד (לא נוגעים בזה)
    upper_bounds = np.max(X_test, axis=0)
    lower_bounds = np.min(X_test, axis=0)
    x = 1

    # -------------------------------------------
    # --- חלק הויזואליזציה החדש (5 חלונות) ---
    # -------------------------------------------
    print("Visualizing results in 5 separate windows...")

    sample_idx = 0  # בחירת הדוגמה הראשונה

    # הגדרת ציר ה-X (Redshift) לגרפים שתלויים בו
    # בודקים שהמימדים תואמים
    if len(predictions) > 1 and len(Z_BINS) == len(predictions[1][sample_idx]):
        x_axis = Z_BINS
        xlabel_z = 'Redshift (z)'
    else:
        # גיבוי למקרה שמשהו במימדים לא מסתדר
        x_axis = range(len(predictions[1][sample_idx]))
        xlabel_z = 'Index'

    # --- גרף 1: Power Spectrum (PS) ---
    plt.figure(figsize=(10, 6))
    # לוקחים חתך עבור ה-Redshift הראשון
    ps_data = predictions[0][sample_idx, :, 0, 0]
    plt.plot(ps_data, 'b-', linewidth=2, label='Predicted PS (z=z_0)')
    plt.title(f'1. Power Spectrum (at first Redshift) - Sample #{sample_idx}')
    plt.xlabel('k bins (Index)')
    plt.ylabel('Power')
    plt.grid(True)
    plt.legend()

    # --- גרף 2: Neutral Hydrogen Fraction (xHI) ---
    if len(predictions) > 1:
        plt.figure(figsize=(10, 6))
        xHI_data = predictions[1][sample_idx]
        plt.plot(x_axis, xHI_data, 'r-', linewidth=2)
        plt.title(f'2. Neutral Hydrogen Fraction (xHI)')
        plt.xlabel(xlabel_z)
        plt.ylabel('Fraction (0 to 1)')
        plt.grid(True)

    # --- גרף 3: Brightness Temperature (Tb) ---
    # זה הגרף שביקשת ספציפית - Tb ביחס ל-Redshift
    if len(predictions) > 3:
        plt.figure(figsize=(10, 6))
        Tb_data = predictions[3][sample_idx]
        plt.plot(x_axis, Tb_data, 'g-', linewidth=2)
        plt.title(f'3. Brightness Temperature (Tb)')
        plt.xlabel(xlabel_z)
        plt.ylabel('Temperature [mK]')
        plt.grid(True)
        # הוספת קו אפס לנוחות
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)

    # --- גרף 4: Kinetic Temperature (Tk) - הגרף החדש! ---
    # Tk הוא בדרך כלל באינדקס 4
    if len(predictions) > 4:
        plt.figure(figsize=(10, 6))
        Tk_data = predictions[4][sample_idx]
        plt.plot(x_axis, Tk_data, color='orange', linewidth=2)
        plt.title(f'4. Kinetic Temperature (Tk)')
        plt.xlabel(xlabel_z)
        plt.ylabel('Temperature [K]')
        plt.yscale('log')  # סקאלה לוגריתמית
        plt.grid(True, which="both", ls="-")

    # --- גרף 5: Spin Temperature (Ts) ---
    if len(predictions) > 5:
        plt.figure(figsize=(10, 6))
        Ts_data = predictions[5][sample_idx]
        plt.plot(x_axis, Ts_data, 'm-', linewidth=2)
        plt.title(f'5. Spin Temperature (Ts)')
        plt.xlabel(xlabel_z)
        plt.ylabel('Temperature [K]')
        plt.yscale('log')  # סקאלה לוגריתמית
        plt.grid(True, which="both", ls="-")

    # פקודה שמציגה את כל החלונות שייצרנו
    print("Saving plots to project folder...")

    # טריק לשמירת כל הגרפים הפתוחים
    figs = [plt.figure(n) for n in plt.get_fignums()]
    for i, fig in enumerate(figs):
        fig.savefig(f'output_graph_{i + 1}.png', dpi=300)  # שומר באיכות גבוהה

    print("Graphs saved successfully!")
    plt.show()  # מציג אותם על המסך
    print("Done.")


# --- הפעלת הפונקציה ---
# וודא שהנתיב מעודכן למחשב שלך
load_model_and_predict(r'C:\Users\roy18\PycharmProjects\21_cm_Emulator\100b_tr_set_model',
                       name='100b_model',
                       )