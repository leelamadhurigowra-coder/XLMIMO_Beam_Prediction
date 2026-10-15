
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# ==================================================
# STAGE 6: FEATURE ENGINEERING
# ==================================================

DATA_PATH = "results/preprocessed_deepmimo.npz"
RESULTS_PATH = "results"

os.makedirs(RESULTS_PATH, exist_ok=True)

print("=" * 65)
print("STAGE 6: FEATURE ENGINEERING")
print("=" * 65)

# ==================================================
# STEP 1: LOAD PREPROCESSED DATA
# ==================================================

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "Preprocessed data not found. "
        "Run Stage 5 first."
    )

data = np.load(DATA_PATH)

power = data["power"]
phase = data["phase"]
delay = data["delay"]
aod_az = data["aod_az"]
aod_el = data["aod_el"]
aoa_az = data["aoa_az"]
aoa_el = data["aoa_el"]

valid_mask = data["valid_mask"].astype(bool)

print("\nInput shape:", power.shape)
print("Valid entries:", np.sum(valid_mask))

# ==================================================
# STEP 2: COMPLEX PATH REPRESENTATION
# ==================================================

# NOTE:
# This conversion assumes power is in linear units.
# We preserve it as a provisional feature until
# the original DeepMIMO power convention is verified.

safe_power = np.maximum(power, 0)

path_amplitude = np.sqrt(safe_power)

complex_path = (
    path_amplitude * np.exp(1j * phase)
)

complex_path = np.where(
    valid_mask,
    complex_path,
    0
)

print("\nComplex path representation created.")

# ==================================================
# STEP 3: PHASE FEATURES
# ==================================================

phase_sin = np.where(
    valid_mask,
    np.sin(phase),
    0
)

phase_cos = np.where(
    valid_mask,
    np.cos(phase),
    0
)

# ==================================================
# STEP 4: ANGULAR FEATURES
# ==================================================

# Convert angles from degrees to radians.
# This assumes the stored angles are in degrees.

aod_az_rad = np.deg2rad(aod_az)
aod_el_rad = np.deg2rad(aod_el)
aoa_az_rad = np.deg2rad(aoa_az)
aoa_el_rad = np.deg2rad(aoa_el)

angular_features = {
    "aod_az_sin": np.where(
        valid_mask, np.sin(aod_az_rad), 0
    ),
    "aod_az_cos": np.where(
        valid_mask, np.cos(aod_az_rad), 0
    ),
    "aod_el_sin": np.where(
        valid_mask, np.sin(aod_el_rad), 0
    ),
    "aod_el_cos": np.where(
        valid_mask, np.cos(aod_el_rad), 0
    ),
    "aoa_az_sin": np.where(
        valid_mask, np.sin(aoa_az_rad), 0
    ),
    "aoa_az_cos": np.where(
        valid_mask, np.cos(aoa_az_rad), 0
    ),
    "aoa_el_sin": np.where(
        valid_mask, np.sin(aoa_el_rad), 0
    ),
    "aoa_el_cos": np.where(
        valid_mask, np.cos(aoa_el_rad), 0
    )
}

print("Angular features created.")

# ==================================================
# STEP 5: RECEIVER-LEVEL SUMMARY FEATURES
# ==================================================

# Count valid paths for each receiver.
path_count = np.sum(
    valid_mask,
    axis=1
)

# Mean path power for each receiver.
mean_power = np.divide(
    np.sum(
        np.where(valid_mask, power, 0),
        axis=1
    ),
    path_count,
    out=np.zeros(power.shape[0], dtype=float),
    where=path_count > 0
)

# Strongest valid path for each receiver.
max_power = np.max(
    np.where(valid_mask, power, -np.inf),
    axis=1
)

max_power[path_count == 0] = 0

# Mean delay for each receiver.
mean_delay = np.divide(
    np.sum(
        np.where(valid_mask, delay, 0),
        axis=1
    ),
    path_count,
    out=np.zeros(delay.shape[0], dtype=float),
    where=path_count > 0
)

print("Receiver-level summary features created.")

# ==================================================
# STEP 6: SAVE ENGINEERED FEATURES
# ==================================================

features = {
    "complex_path": complex_path,
    "phase_sin": phase_sin,
    "phase_cos": phase_cos,
    "path_count": path_count,
    "mean_power": mean_power,
    "max_power": max_power,
    "mean_delay": mean_delay,
    "valid_mask": valid_mask
}

features.update(angular_features)

output_file = os.path.join(
    RESULTS_PATH,
    "engineered_features.npz"
)

np.savez_compressed(
    output_file,
    **features
)

print("\nEngineered features saved:")
print(output_file)

# ==================================================
# STEP 7: VISUALIZE PATH COUNTS
# ==================================================

plt.figure(figsize=(10, 6))

plt.hist(
    path_count,
    bins=30,
    edgecolor="black"
)

plt.title("Distribution of Valid Paths per Receiver")
plt.xlabel("Number of Valid Propagation Paths")
plt.ylabel("Number of Receivers")
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_PATH,
        "path_count_distribution.png"
    ),
    dpi=300
)

plt.show()

# ==================================================
# STEP 8: SAVE FEATURE REPORT
# ==================================================

report = {
    "stage": "Feature Engineering",
    "input_shape": list(power.shape),
    "engineered_features": list(features.keys()),
    "number_of_features": len(features),
    "output_file": output_file,
    "status": "Path-level feature generation completed",
    "note": (
        "Frequency-dependent channel and beamforming "
        "features require antenna-array configuration "
        "and verified channel units."
    )
}

with open(
    os.path.join(
        RESULTS_PATH,
        "feature_engineering_report.json"
    ),
    "w"
) as file:

    json.dump(report, file, indent=4)

print("\nFeature engineering completed!")