
import os
import json
import numpy as np
import scipy.io as sio

# ==========================================
# STAGE 4: DATA CLEANING
# ==========================================

DATA_PATH = "data"
RESULTS_PATH = "results"

os.makedirs(RESULTS_PATH, exist_ok=True)

print("=" * 60)
print("STAGE 4: DATA CLEANING")
print("=" * 60)

# ==========================================
# STEP 1: Load power data
# ==========================================

power_file = os.path.join(
    DATA_PATH,
    "power_t001_tx000_r000.mat"
)

power_data = sio.loadmat(power_file)

power_key = [
    key for key in power_data
    if not key.startswith("__")
][0]

power = np.asarray(
    power_data[power_key],
    dtype=np.float64
)

print("\nOriginal power shape:", power.shape)

# ==========================================
# STEP 2: Identify invalid entries
# ==========================================

valid_mask = np.isfinite(power)

total_entries = power.size
valid_entries = np.count_nonzero(valid_mask)
invalid_entries = total_entries - valid_entries

valid_percentage = (
    valid_entries / total_entries
) * 100

invalid_percentage = (
    invalid_entries / total_entries
) * 100

print("\nDATA QUALITY REPORT")
print("-" * 40)

print("Total entries:", total_entries)
print("Valid entries:", valid_entries)
print("Invalid entries:", invalid_entries)

print(
    "Valid percentage:",
    round(valid_percentage, 2), "%"
)

print(
    "Invalid percentage:",
    round(invalid_percentage, 2), "%"
)

# ==========================================
# STEP 3: Check valid numerical values
# ==========================================

valid_power = power[valid_mask]

if valid_power.size > 0:

    print("\nValid power statistics")

    print("Minimum:", np.min(valid_power))
    print("Maximum:", np.max(valid_power))
    print("Mean:", np.mean(valid_power))
    print("Median:", np.median(valid_power))

else:
    raise ValueError("No valid power entries found.")

# ==========================================
# STEP 4: Check receiver positions
# ==========================================

position_file = os.path.join(
    DATA_PATH,
    "rx_pos_t001_tx000_r000.mat"
)

position_data = sio.loadmat(position_file)

position_key = [
    key for key in position_data
    if not key.startswith("__")
][0]

positions = np.asarray(position_data[position_key])

print("\nReceiver position shape:", positions.shape)

if positions.ndim == 2:

    unique_positions = np.unique(
        positions,
        axis=0
    )

    duplicate_positions = (
        len(positions) - len(unique_positions)
    )

    print("Unique receiver positions:", len(unique_positions))
    print("Duplicate positions:", duplicate_positions)

else:
    duplicate_positions = None
    print("Position array requires further inspection.")

# ==========================================
# STEP 5: Save validity mask
# ==========================================

mask_path = os.path.join(
    RESULTS_PATH,
    "valid_path_mask.npy"
)

np.save(mask_path, valid_mask)

print("\nValidity mask saved:", mask_path)

# ==========================================
# STEP 6: Save cleaned valid power entries
# ==========================================

cleaned_power = np.where(
    valid_mask,
    power,
    np.nan
)

cleaned_path = os.path.join(
    RESULTS_PATH,
    "cleaned_power.npy"
)

np.save(cleaned_path, cleaned_power)

print("Cleaned power array saved:", cleaned_path)

# ==========================================
# STEP 7: Generate cleaning report
# ==========================================

report = {
    "stage": "Data Cleaning",
    "total_entries": int(total_entries),
    "valid_entries": int(valid_entries),
    "invalid_entries": int(invalid_entries),
    "valid_percentage": round(float(valid_percentage), 2),
    "invalid_percentage": round(float(invalid_percentage), 2),
    "duplicate_receiver_positions": (
        int(duplicate_positions)
        if duplicate_positions is not None
        else None
    ),
    "original_shape": list(power.shape),
    "cleaning_method": (
        "Non-finite path entries identified and "
        "preserved as NaN for downstream masking."
    )
}

report_path = os.path.join(
    RESULTS_PATH,
    "data_cleaning_report.json"
)

with open(report_path, "w") as file:
    json.dump(report, file, indent=4)

print("\nCleaning report saved:", report_path)

print("\nDATA CLEANING AUDIT COMPLETED!")