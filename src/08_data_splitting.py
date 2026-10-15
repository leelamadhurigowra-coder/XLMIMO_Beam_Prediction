
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

# ============================================================
# STAGE 7: DATA SPLITTING
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# STEP 1: LOAD ENGINEERED DATA
# ------------------------------------------------------------

feature_file = RESULTS_DIR / "engineered_features.npz"

if not feature_file.exists():
    raise FileNotFoundError(
        "engineered_features.npz not found!"
    )

data = np.load(feature_file, allow_pickle=True)

valid_mask = data["valid_mask"]

# ------------------------------------------------------------
# STEP 2: SELECT RECEIVERS WITH VALID PATHS
# ------------------------------------------------------------

valid_receiver_mask = np.any(valid_mask, axis=1)

valid_receiver_indices = np.where(
    valid_receiver_mask
)[0]

total_receivers = len(valid_receiver_mask)
usable_receivers = len(valid_receiver_indices)
excluded_receivers = total_receivers - usable_receivers

print("Total receivers:", total_receivers)
print("Usable receivers:", usable_receivers)
print("Excluded receivers:", excluded_receivers)

# ------------------------------------------------------------
# STEP 3: SPLIT INTO TRAIN, VALIDATION AND TEST
# ------------------------------------------------------------

train_idx, temp_idx = train_test_split(
    valid_receiver_indices,
    test_size=0.30,
    random_state=42,
    shuffle=True
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=42,
    shuffle=True
)

# ------------------------------------------------------------
# STEP 4: SAVE INDICES
# ------------------------------------------------------------

np.savez(
    RESULTS_DIR / "data_split_indices.npz",
    train_indices=train_idx,
    val_indices=val_idx,
    test_indices=test_idx,
    valid_receiver_indices=valid_receiver_indices
)

# ------------------------------------------------------------
# STEP 5: CREATE REPORT
# ------------------------------------------------------------

report = {
    "total_receivers": int(total_receivers),
    "usable_receivers": int(usable_receivers),
    "excluded_receivers": int(excluded_receivers),

    "training_samples": int(len(train_idx)),
    "validation_samples": int(len(val_idx)),
    "testing_samples": int(len(test_idx)),

    "training_percentage": round(
        len(train_idx) / usable_receivers * 100, 2
    ),

    "validation_percentage": round(
        len(val_idx) / usable_receivers * 100, 2
    ),

    "testing_percentage": round(
        len(test_idx) / usable_receivers * 100, 2
    ),

    "random_state": 42,
    "split_method": "Receiver-level random split",
    "validity_rule": "At least one valid path"
}

with open(
    RESULTS_DIR / "data_split_report.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(report, f, indent=4)

# ------------------------------------------------------------
# STEP 6: DISPLAY RESULTS
# ------------------------------------------------------------

print("\nDATA SPLITTING COMPLETED")
print("=" * 45)

for key, value in report.items():
    print(f"{key}: {value}")

print("\nFiles saved in results folder.")