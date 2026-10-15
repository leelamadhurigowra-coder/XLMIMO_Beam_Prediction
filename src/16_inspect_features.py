import numpy as np
from pathlib import Path
import sys

# ============================================================
# STEP 1: FIND THE PROJECT DIRECTORY
# ============================================================

# This file is inside the src folder.
# Its parent directory is the main project folder.

PROJECT_DIR = Path(__file__).resolve().parent.parent

print("\n========================================")
print("       XLMIMO DATASET INSPECTION")
print("========================================")

print("\nProject directory:")
print(PROJECT_DIR)


# ============================================================
# STEP 2: FIND THE ENGINEERED FEATURES FILE
# ============================================================

file_path = PROJECT_DIR / "data" / "engineered_features.npz"

# If the expected location is missing, search the project.
if not file_path.exists():
    print("\nDataset not found in the default data folder.")
    print("Searching the project directory...")

    matches = list(
        PROJECT_DIR.rglob("engineered_features.npz")
    )

    if matches:
        file_path = matches[0]
        print("\nDataset found at:")
        print(file_path)

    else:
        print("\nERROR: engineered_features.npz was not found.")
        print("\nPlease check whether the dataset exists in your project.")
        sys.exit(1)


# ============================================================
# STEP 3: LOAD THE ENGINEERED FEATURES
# ============================================================

try:
    data = np.load(file_path, allow_pickle=False)

except Exception as error:
    print("\nERROR while loading dataset:")
    print(error)
    sys.exit(1)


# ============================================================
# STEP 4: DISPLAY FEATURE NAMES
# ============================================================

print("\n========================================")
print("          AVAILABLE FEATURES")
print("========================================")

print("\nFeature names:")
print(data.files)


# ============================================================
# STEP 5: INSPECT EACH FEATURE
# ============================================================

for key in data.files:

    array = data[key]

    print("\n----------------------------------------")
    print("Feature:", key)
    print("----------------------------------------")

    print("Shape:", array.shape)
    print("Dimensions:", array.ndim)
    print("Data type:", array.dtype)
    print("Total elements:", array.size)

    if np.issubdtype(array.dtype, np.number):

        finite = np.isfinite(array)

        finite_count = np.sum(finite)
        nonfinite_count = np.sum(~finite)

        print("Finite values:", finite_count)
        print("Non-finite values:", nonfinite_count)

        if finite_count > 0:

            valid_values = array[finite]

            print("Minimum:", np.min(valid_values))
            print("Maximum:", np.max(valid_values))
            print("Mean:", np.mean(valid_values))
            print("Standard deviation:", np.std(valid_values))

    else:
        print("Numerical statistics not available.")


# ============================================================
# STEP 6: INSPECT OTHER DATA FILES
# ============================================================

print("\n========================================")
print("          OTHER DATA FILES")
print("========================================")

other_files = [
    "random_forest_input_features.npy",
    "beam_labels.npy",
    "beam_label_receiver_indices.npy"
]

for filename in other_files:

    path = file_path.parent / filename

    print("\nFile:", filename)

    if path.exists():

        try:
            array = np.load(path, allow_pickle=False)

            print("Status: FOUND")
            print("Shape:", array.shape)
            print("Data type:", array.dtype)
            print("Total elements:", array.size)

        except Exception as error:
            print("Error loading file:", error)

    else:
        print("Status: NOT FOUND")


# ============================================================
# STEP 7: FINAL SUMMARY
# ============================================================

print("\n========================================")
print("             INSPECTION COMPLETE")
print("========================================")

print("\nDataset location:")
print(file_path)

print("\nAll available features have been inspected.")

print("\nPlease share this terminal output for the")
print("next stage of Transformer dataset preparation.")

print("\n========================================")