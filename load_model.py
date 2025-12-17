import os

# --- Compatibility Fix (Must be the first line) ---
os.environ["TF_USE_LEGACY_KERAS"] = "1"
# -----------------------------------------------------

import numpy as np
import pickle
import matplotlib.pyplot as plt
from build_NN import FCemu


def load_model_and_predict(model_dir, name, testing_files_dir=None):
    """
    Loads the model, performs prediction on the test set, and visualizes 5 key metrics.
    """
    if testing_files_dir is None:
        testing_files_dir = model_dir

    # --- Load Data ---
    print("Loading data...")
    # Use os.path.join for cross-platform compatibility (Windows/Mac/Linux)
    data_path = os.path.join(testing_files_dir, 'training_files.pk')

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    data = pickle.load(open(data_path, 'rb'))
    X_train, _, X_val, _, X_test, Y_test = data

    # --- Load Model ---
    print("Loading model...")
    emulator = FCemu(restore=True, files_dir=model_dir, name=name)
    Z_BINS = emulator.z_glob  # Redshift axis

    # --- Run Prediction ---
    print("Running prediction (this might take a moment)...")
    predictions = emulator.predict(X_test)

    # -------------------------------------------
    # --- Visualization (5 Windows) ---
    # -------------------------------------------
    print("Visualizing results in 5 separate windows...")

    sample_idx = 0  # Plot the first sample in the test set

    # Define X-axis (Redshift)
    if len(predictions) > 1 and len(Z_BINS) == len(predictions[1][sample_idx]):
        x_axis = Z_BINS
        xlabel_z = 'Redshift (z)'
    else:
        # Fallback if dimensions don't match
        x_axis = range(len(predictions[1][sample_idx]))
        xlabel_z = 'Index'

    # --- Graph 1: Power Spectrum (PS) ---
    plt.figure(figsize=(10, 6))
    ps_data = predictions[0][sample_idx, :, 0, 0]
    plt.plot(ps_data, 'b-', linewidth=2, label='Predicted PS (z=z_0)')
    plt.title(f'1. Power Spectrum (at first Redshift) - Sample #{sample_idx}')
    plt.xlabel('k bins (Index)')
    plt.ylabel('Power')
    plt.grid(True)
    plt.legend()

    # --- Graph 2: Neutral Hydrogen Fraction (xHI) ---
    if len(predictions) > 1:
        plt.figure(figsize=(10, 6))
        xHI_data = predictions[1][sample_idx]
        plt.plot(x_axis, xHI_data, 'r-', linewidth=2)
        plt.title(f'2. Neutral Hydrogen Fraction (xHI)')
        plt.xlabel(xlabel_z)
        plt.ylabel('Fraction (0 to 1)')
        plt.grid(True)

    # --- Graph 3: Brightness Temperature (Tb) ---
    if len(predictions) > 3:
        plt.figure(figsize=(10, 6))
        Tb_data = predictions[3][sample_idx]
        plt.plot(x_axis, Tb_data, 'g-', linewidth=2)
        plt.title(f'3. Brightness Temperature (Tb)')
        plt.xlabel(xlabel_z)
        plt.ylabel('Temperature [mK]')
        plt.grid(True)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)

    # --- Graph 4: Kinetic Temperature (Tk) ---
    if len(predictions) > 4:
        plt.figure(figsize=(10, 6))
        Tk_data = predictions[4][sample_idx]
        plt.plot(x_axis, Tk_data, color='orange', linewidth=2)
        plt.title(f'4. Kinetic Temperature (Tk)')
        plt.xlabel(xlabel_z)
        plt.ylabel('Temperature [K]')
        plt.yscale('log')
        plt.grid(True, which="both", ls="-")

    # --- Graph 5: Spin Temperature (Ts) ---
    if len(predictions) > 5:
        plt.figure(figsize=(10, 6))
        Ts_data = predictions[5][sample_idx]
        plt.plot(x_axis, Ts_data, 'm-', linewidth=2)
        plt.title(f'5. Spin Temperature (Ts)')
        plt.xlabel(xlabel_z)
        plt.ylabel('Temperature [K]')
        plt.yscale('log')
        plt.grid(True, which="both", ls="-")

    # Save plots
    print("Saving plots to project folder...")
    try:
        figs = [plt.figure(n) for n in plt.get_fignums()]
        for i, fig in enumerate(figs):
            fig.savefig(f'output_graph_{i + 1}.png', dpi=300)
        print("Graphs saved successfully!")
    except Exception as e:
        print(f"Note: Could not auto-save graphs ({e})")

    plt.show()
    print("Done.")


# --- Main Execution Block (The Fix) ---
# This logic allows the file to run on ANY computer without changing paths manually.
if __name__ == "__main__":

    # 1. Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. Define the model directory relative to this script
    model_folder_path = os.path.join(current_dir, '100b_tr_set_model')
    model_name = '100b_model'

    # 3. Run the function safely
    if os.path.exists(model_folder_path):
        load_model_and_predict(model_folder_path, name=model_name)
    else:
        print(f"Error: Model folder not found at: {model_folder_path}")
        print("Please ensure the '100b_tr_set_model' folder is in the same directory as this script.")