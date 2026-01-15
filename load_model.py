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
    Loads the model, performs prediction on a random test sample, and visualizes outputs.
    """
    import h5py

    # --- Load Data Metadata ---
    print("Loading data metadata...")
    data_path = os.path.join(model_dir, 'model_data.h5')

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    # Read min/max for normalization/denormalization
    with h5py.File(data_path, 'r') as h5f:
        # Decode recursively if bytes
        def decode_if_bytes(x):
            return x.decode('utf-8') if isinstance(x, bytes) else x

        param_names = [decode_if_bytes(x) for x in h5f['params_names'][:]]
        tr_min = h5f['tr_params_min'][:]
        tr_max = h5f['tr_params_max'][:]
        z_glob = h5f['z_glob'][:]

    # Create a dummy test sample (mean of ranges)
    # in usage we normally have a test set, here we just simulate one for verification
    print("Generating synthetic test sample (Mean of parameters)...")
    X_test_sample = (tr_min + tr_max) / 2.0
    X_test = X_test_sample.reshape(1, -1)

    # --- Load Model ---
    print("Loading model...")
    emulator = FCemu(restore=True, files_dir=model_dir, name=name)
    Z_BINS = emulator.z_glob  # Redshift axis

    # --- Run Prediction ---
    print("Running prediction...")
    predictions = emulator.predict(X_test)

    # -------------------------------------------
    # --- Visualization ---
    # -------------------------------------------
    print("Visualizing results...")

    sample_idx = 0 
    x_axis = Z_BINS
    xlabel_z = 'Redshift (z)'

    # Output indices based on `predictions_to_dict` or standard FCemu output order
    # Typically: xHI, Tb, Tk, Ts (indices 0, 1, 2, 3) for the new model (no PS)
    
    # --- Graph 1: Neutral Hydrogen Fraction (xHI) ---
    if len(predictions) > 0:
        plt.figure(figsize=(10, 6))
        xHI_data = predictions[0][sample_idx]
        plt.plot(x_axis, xHI_data, 'r-', linewidth=2)
        plt.title(f'1. Neutral Hydrogen Fraction (xHI)')
        plt.xlabel(xlabel_z)
        plt.ylabel('Fraction (0 to 1)')
        plt.grid(True)
    
    # --- Graph 2: Brightness Temperature (Tb) ---
    if len(predictions) > 1:
        plt.figure(figsize=(10, 6))
        Tb_data = predictions[1][sample_idx]
        plt.plot(x_axis, Tb_data, 'g-', linewidth=2)
        plt.title(f'2. Brightness Temperature (Tb)')
        plt.xlabel(xlabel_z)
        plt.ylabel('Temperature [mK]')
        plt.grid(True)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)

    # --- Graph 3: Kinetic Temperature (Tk) ---
    if len(predictions) > 2:
        plt.figure(figsize=(10, 6))
        Tk_data = predictions[2][sample_idx]
        plt.plot(x_axis, Tk_data, color='orange', linewidth=2)
        plt.title(f'3. Kinetic Temperature (Tk)')
        plt.xlabel(xlabel_z)
        plt.ylabel('Temperature [K]')
        plt.yscale('log')
        plt.grid(True, which="both", ls="-")

    # --- Graph 4: Spin Temperature (Ts) ---
    if len(predictions) > 3:
        plt.figure(figsize=(10, 6))
        Ts_data = predictions[3][sample_idx]
        plt.plot(x_axis, Ts_data, 'm-', linewidth=2)
        plt.title(f'4. Spin Temperature (Ts)')
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
        print(f"Saved {len(figs)} graphs successfully!")
    except Exception as e:
        print(f"Note: Could not auto-save graphs ({e})")

    # plt.show() # blocking
    print("Done.")


# --- Main Execution Block (The Fix) ---
# This logic allows the file to run on ANY computer without changing paths manually.
if __name__ == "__main__":

    # 1. Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. Define the model directory relative to this script
    model_folder_path = os.path.join(current_dir, 'model_files')
    model_name = 'globals_model'

    # 3. Run the function safely
    if os.path.exists(model_folder_path):
        load_model_and_predict(model_folder_path, name=model_name)
    else:
        print(f"Error: Model folder not found at: {model_folder_path}")
        print("Please ensure the 'model_files' folder is in the same directory as this script.")