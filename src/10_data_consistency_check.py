
import numpy as np
import json
from pathlib import Path
from scipy.io import loadmat

# ============================================================
# STAGE 8.2: DATA CONSISTENCY CHECK
# ============================================================

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Results directory
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("DEEPMIMO DATA CONSISTENCY CHECK")
print("=" * 60)

# ------------------------------------------------------------
# STEP 1: LOCATE ORIGINAL DATASET FILES AUTOMATICALLY
# ------------------------------------------------------------

print("\nSearching for original dataset files...")

power_matches = list(
    BASE_DIR.rglob("power_t001_tx000_r000.mat")
)

rx_matches = list(
    BASE_DIR.rglob("rx_pos_t001_tx000_r000.mat")
)

if not power_matches:
    raise FileNotFoundError(
        "Power file not found. Make sure "
        "power_t001_tx000_r000.mat exists inside your project folder."
    )

if not rx_matches:
    raise FileNotFoundError(
        "Receiver position file not found. Make sure "
        "rx_pos_t001_tx000_r000.mat exists inside your project folder."
    )

power_file = power_matches[0]
rx_file = rx_matches[0]

print("Power file found:", power_file)
print("Receiver file found:", rx_file)

# ------------------------------------------------------------
# STEP 2: LOAD ORIGINAL DATA
# ------------------------------------------------------------

print("\nLoading original dataset...")

power = loadmat(power_file)["power"]
rx_pos = loadmat(rx_file)["rx_pos"]

print("Original power shape:", power.shape)
print("Original receiver position shape:", rx_pos.shape)

# ------------------------------------------------------------
# STEP 3: LOAD ENGINEERED FEATURES
# ------------------------------------------------------------

feature_file = RESULTS_DIR / "engineered_features.npz"

if not feature_file.exists():
    raise FileNotFoundError(
        "engineered_features.npz not found in results folder. "
        "Run the feature engineering script first."
    )

features = np.load(feature_file, allow_pickle=True)

print("\nAvailable engineered arrays:")

for key in features.files:
    print(f"{key}: {features[key].shape}")

# Extract important arrays
path_count = features["path_count"]
valid_mask = features["valid_mask"]

# ------------------------------------------------------------
# STEP 4: CALCULATE DATASET STATISTICS
# ------------------------------------------------------------

original_rows = power.shape[0]
engineered_rows = path_count.shape[0]

finite_power_mask = np.isfinite(power)

rows_with_finite_power = np.any(
    finite_power_mask,
    axis=1
)

rows_with_valid_paths = np.any(
    valid_mask,
    axis=1
)

# Check whether receiver coordinates are unique
unique_rx_positions = np.unique(
    rx_pos,
    axis=0
)

# ------------------------------------------------------------
# STEP 5: CHECK DIMENSION CONSISTENCY
# ------------------------------------------------------------

dimension_checks = {
    "power_and_receiver_rows_match":
        power.shape[0] == rx_pos.shape[0],

    "engineered_features_and_mask_match":
        path_count.shape[0] == valid_mask.shape[0],

    "original_and_engineered_rows_match":
        original_rows == engineered_rows,

    "power_and_valid_mask_path_dimensions_match":
        power.shape[1] == valid_mask.shape[1]
}

# ------------------------------------------------------------
# STEP 6: CREATE REPORT
# ------------------------------------------------------------

report = {
    "dataset_information": {
        "original_power_shape": list(power.shape),
        "original_receiver_positions_shape": list(rx_pos.shape),
        "engineered_path_count_shape": list(path_count.shape),
        "engineered_valid_mask_shape": list(valid_mask.shape),
        "original_rows": int(original_rows),
        "engineered_rows": int(engineered_rows),
        "unique_receiver_positions": int(
            unique_rx_positions.shape[0]
        )
    },

    "power_statistics": {
        "total_power_values": int(power.size),
        "finite_power_values": int(
            finite_power_mask.sum()
        ),
        "non_finite_power_values": int(
            (~finite_power_mask).sum()
        ),
        "rows_with_at_least_one_finite_power": int(
            rows_with_finite_power.sum()
        ),
        "rows_with_no_finite_power": int(
            (~rows_with_finite_power).sum()
        )
    },

    "engineered_data_statistics": {
        "rows_with_at_least_one_valid_path": int(
            rows_with_valid_paths.sum()
        ),
        "rows_with_no_valid_paths": int(
            (~rows_with_valid_paths).sum()
        ),
        "total_valid_paths": int(
            valid_mask.sum()
        ),
        "total_invalid_paths": int(
            (~valid_mask).sum()
        )
    },

    "dimension_checks": dimension_checks,

    "overall_consistency": (
        "PASS"
        if all(dimension_checks.values())
        else "REVIEW REQUIRED"
    )
}

# ------------------------------------------------------------
# STEP 7: SAVE REPORT
# ------------------------------------------------------------

report_file = RESULTS_DIR / "data_consistency_report.json"

with open(report_file, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4)

# Also save a readable text report
text_report_file = RESULTS_DIR / "data_consistency_report.txt"

with open(text_report_file, "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("DEEPMIMO DATA CONSISTENCY REPORT\n")
    f.write("=" * 60 + "\n\n")

    for section, values in report.items():
        f.write(f"\n{section.upper().replace('_', ' ')}\n")
        f.write("-" * 50 + "\n")

        if isinstance(values, dict):
            for key, value in values.items():
                f.write(f"{key}: {value}\n")
        else:
            f.write(f"{values}\n")

# ------------------------------------------------------------
# STEP 8: DISPLAY RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DATA CONSISTENCY RESULTS")
print("=" * 60)

print("\nOriginal rows:", original_rows)
print("Engineered rows:", engineered_rows)

print(
    "Unique receiver positions:",
    unique_rx_positions.shape[0]
)

print(
    "Finite power values:",
    finite_power_mask.sum()
)

print(
    "Non-finite power values:",
    (~finite_power_mask).sum()
)

print(
    "Valid paths:",
    valid_mask.sum()
)

print("\nDimension checks:")

for check, result in dimension_checks.items():
    status = "PASS" if result else "REVIEW"
    print(f"{check}: {status}")

print("\nOverall status:", report["overall_consistency"])

print("\nReports saved successfully:")
print(report_file)
print(text_report_file)

print("\nDATA CONSISTENCY CHECK COMPLETED")