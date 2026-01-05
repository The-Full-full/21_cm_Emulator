"""
Script Name: Create Lite Data for Deployment
Description:
    This script optimizes the data loading process for the 21cm Emulator Streamlit App.

    The original training dataset ('training_files.pk') contains the entire training set (X_train),
    which is very large and unnecessary for the inference/testing phase of the web application.

    This script performs the following operations:
    1. Loads the massive 'training_files.pk' file.
    2. Extracts ONLY the Test Set (X_test), which is required for defining parameter ranges.
    3. Saves this smaller dataset to a new binary file ('emulator_data_lite.pk').

    Result:
    Drastically reduces the application's startup time by loading ~1MB instead of ~X00MB.

Usage:
    Run this script once before deploying the app or whenever the underlying model data changes.
"""

import pickle
import os
import numpy as np
import sys

# --- CONFIGURATION ---
# Define the directory containing the model and data files
# NOTE: Ensure this path is correct for your local machine
# Determine the absolute directory where this script is located
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the model directory relative to the script location
MODEL_DIR = os.path.join(CURRENT_DIR, '100b_tr_set_model')

# Define input (heavy) and output (lite) file paths
input_heavy_file = os.path.join(MODEL_DIR, 'training_files.pk')
output_lite_file = os.path.join(MODEL_DIR, 'emulator_data_lite.pk')


def create_lite_file():
    print("--- Starting Data Optimization Process ---")

    # 1. Check if source file exists
    if not os.path.exists(input_heavy_file):
        print(f"Error: Source file not found at: {input_heavy_file}")
        sys.exit(1)

    print(f"1. Loading heavy dataset from: {input_heavy_file}")
    print("   (This may take a moment due to file size...)")

    try:
        # Load the full pickle file
        data = pickle.load(open(input_heavy_file, 'rb'))

        # 2. Extract X_test
        # The data structure is known to be: [X_train, Y_train, X_val, Y_val, X_test, Y_test]
        # We only need X_test (index 4) for the slider limits in the app.
        X_test = data[4]

        print(f"2. Data loaded successfully. Extracted Test Set shape: {X_test.shape}")

        # 3. Save to new lite file
        print(f"3. Saving optimized data to: {output_lite_file}")
        with open(output_lite_file, 'wb') as f:
            pickle.dump(X_test, f)

        # Calculate file size for comparison (optional)
        file_size_mb = os.path.getsize(output_lite_file) / (1024 * 1024)

        print("\n--- SUCCESS! ---")
        print(f"Optimized file 'emulator_data_lite.pk' created successfully.")
        print(f"File Size: {file_size_mb:.2f} MB")
        print("The Streamlit app will now load significantly faster.")

    except Exception as e:
        print(f"\n--- ERROR ---")
        print(f"An error occurred during processing: {e}")


if __name__ == "__main__":
    create_lite_file()