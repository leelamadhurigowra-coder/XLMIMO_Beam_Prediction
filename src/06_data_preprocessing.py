
import os
import json
import numpy as np
import scipy.io as sio

# ==================================================
# STAGE 5: DATA PREPROCESSING
# ==================================================

DATA_PATH = "data"
RESULTS_PATH = "results"

os.makedirs(RESULTS_PATH, exist_ok=True)

print("=" * 65)
print("STAGE 5: DEEPMIMO DATA PREPROCESSING")
print("=" * 65)


# ==================================================
# STEP 1: FUNCTION TO LOAD MATLAB ARRAYS
# ==================================================

def load_mat_array(filename):

    filepath = os.path.join(DATA_PATH, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Missing dataset file: {filepath}"
        )

    mat_data = sio.loadmat(filepath)

    variables = [
        key for key in mat_data
        if not key.startswith("__")
    ]

    if len(variables) != 1:
        raise ValueError(
            f"Unexpected variables in {filename}: {variables}"
        )

    key = variables[0]

    array = np.asarray(mat_data[key])

    print(f"\nLoaded: {filename}")
    print("Variable:", key)
    print("Shape:", array.shape)
    print("Data type:", array.dtype)

    return array


# ==================================================
# STEP 2: LOAD PROPAGATION PARAMETERS
# ==================================================

power = load_mat_array(
    "power_t001_tx000_r000.mat"
)

phase = load_mat_array(
    "phase_t001_tx000_r000.mat"
)

delay = load_mat_array(
    "delay_t001_tx000_r000.mat"
)

aod_az = load_mat_array(
    "aod_az_t001_tx000_r000.mat"
)

aod_el = load_mat_array(
    "aod_el_t001_tx000_r000.mat"
)

aoa_az = load_mat_array(
    "aoa_az_t001_tx000_r000.mat"
)

aoa_el = load_mat_array(
    "aoa_el_t001_tx000_r000.mat"
)


# ==================================================
# STEP 3: VALIDATE ARRAY DIMENSIONS
# ==================================================

arrays = {
    "power": power,
    "phase": phase,
    "delay": delay,
    "aod_az": aod_az,
    "aod_el": aod_el,
    "aoa_az": aoa_az,
    "aoa_el": aoa_el
}

print("\n" + "=" * 65)
print("DIMENSION VALIDATION")
print("=" * 65)

reference_shape = power.shape

for name, array in arrays.items():

    if array.shape != reference_shape:

        raise ValueError(
            f"Dimension mismatch: {name} has "
            f"shape {array.shape}, expected {reference_shape}"
        )

    print(f"{name}: Shape verified")


# ==================================================
# STEP 4: CREATE COMMON VALIDITY MASK
# ==================================================

valid_mask = np.ones(
    reference_shape,
    dtype=bool
)

for array in arrays.values():

    valid_mask &= np.isfinite(array)

total_entries = valid_mask.size

valid_entries = np.count_nonzero(valid_mask)

invalid_entries = total_entries - valid_entries

print("\nCOMMON VALIDITY REPORT")

print("Total entries:", total_entries)
print("Valid entries:", valid_entries)
print("Invalid entries:", invalid_entries)

print(
    "Valid percentage:",
    round(valid_entries / total_entries * 100, 2),
    "%"
)


# ==================================================
# STEP 5: PREPARE ALIGNED ARRAYS
# ==================================================

processed = {}

for name, array in arrays.items():

    # Preserve original values where valid.
    # Use zero as a placeholder where invalid.
    # The validity mask must always accompany these arrays.

    processed[name] = np.where(
        valid_mask,
        array,
        0
    ).astype(np.float32)

    print(
        f"{name} processed:",
        processed[name].shape
    )


# ==================================================
# STEP 6: SAVE PREPROCESSED DATA
# ==================================================

output_path = os.path.join(
    RESULTS_PATH,
    "preprocessed_deepmimo.npz"
)

np.savez_compressed(
    output_path,
    **processed,
    valid_mask=valid_mask
)

print("\nPreprocessed dataset saved:")
print(output_path)


# ==================================================
# STEP 7: SAVE PREPROCESSING REPORT
# ==================================================

report = {
    "stage": "Data Preprocessing",
    "original_shape": list(reference_shape),
    "total_entries": int(total_entries),
    "valid_entries": int(valid_entries),
    "invalid_entries": int(invalid_entries),
    "valid_percentage": round(
        float(valid_entries / total_entries * 100),
        2
    ),
    "features": list(arrays.keys()),
    "output_file": output_path,
    "method": (
        "Common validity mask, dimension validation, "
        "float32 conversion and aligned array preparation."
    )
}

report_path = os.path.join(
    RESULTS_PATH,
    "preprocessing_report.json"
)

with open(report_path, "w") as file:

    json.dump(
        report,
        file,
        indent=4
    )

print("\nPreprocessing report saved:")
print(report_path)

print("\nPREPROCESSING COMPLETED!")