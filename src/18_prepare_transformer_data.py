
import numpy as np
from pathlib import Path
import json
import sys

# ============================================================
# STEP 1: PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"

FEATURE_FILE = RESULTS_DIR / "engineered_features.npz"
LABEL_FILE = DATA_DIR / "beam_labels.npy"
INDEX_FILE = DATA_DIR / "beam_label_receiver_indices.npy"

OUTPUT_FILE = DATA_DIR / "transformer_input.npy"
OUTPUT_LABELS = DATA_DIR / "transformer_labels.npy"
OUTPUT_REPORT = RESULTS_DIR / "transformer_data_report.json"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# STEP 2: LOAD DATA
# ============================================================

print("\n========== LOADING DATA ==========\n")

for path in [FEATURE_FILE, LABEL_FILE, INDEX_FILE]:
    if not path.exists():
        print("Missing file:", path)
        sys.exit(1)

features = np.load(FEATURE_FILE)
labels = np.load(LABEL_FILE)
receiver_indices = np.load(INDEX_FILE)

print("Feature arrays:", features.files)
print("Labels shape:", labels.shape)
print("Receiver indices:", receiver_indices.shape)


# ============================================================
# STEP 3: EXTRACT ARRAYS
# ============================================================

complex_path = features["complex_path"]
valid_mask = features["valid_mask"]

angle_names = [
    "aod_az_sin",
    "aod_az_cos",
    "aod_el_sin",
    "aod_el_cos",
    "aoa_az_sin",
    "aoa_az_cos",
    "aoa_el_sin",
    "aoa_el_cos"
]

angles = np.stack(
    [features[name] for name in angle_names],
    axis=-1
)

# The eight angular components are retained here.
# Each path has a complex coefficient and a validity flag.

print("\nComplex path shape:", complex_path.shape)
print("Validity mask shape:", valid_mask.shape)
print("Angle tensor shape:", angles.shape)


# ============================================================
# STEP 4: DEFINE FREQUENCY GRID
# ============================================================

# Project parameters:
# Carrier frequency = 3.5 GHz
# Bandwidth = 100 MHz
# Number of subcarriers = 64

fc = 3.5e9
bandwidth = 100e6
num_subcarriers = 64

subcarrier_spacing = bandwidth / num_subcarriers

# Centered frequency offsets
frequency_offsets = (
    np.arange(num_subcarriers) - (num_subcarriers - 1) / 2
) * subcarrier_spacing

normalized_frequency = (
    frequency_offsets / (bandwidth / 2)
).astype(np.float32)

print("\nCarrier frequency:", fc)
print("Bandwidth:", bandwidth)
print("Subcarriers:", num_subcarriers)
print("Subcarrier spacing:", subcarrier_spacing)


# ============================================================
# STEP 5: ALIGN USABLE RECEIVERS
# ============================================================

X_complex = complex_path[receiver_indices]
X_valid = valid_mask[receiver_indices]
X_angles = angles[receiver_indices]

Y = labels.astype(np.int64)

num_receivers = len(receiver_indices)

print("\nUsable receivers:", num_receivers)
print("Aligned labels:", Y.shape)


# ============================================================
# STEP 6: CREATE FREQUENCY-DEPENDENT FEATURES
# ============================================================

# Each subcarrier has 17 features:
#
# 1-2: Real and imaginary parts of total channel
# 3-18: Real and imaginary parts of 8
#      direction-weighted channel components
# 19: Normalized frequency
#
# The final feature count is 19.

num_features = 19

X = np.lib.format.open_memmap(
    OUTPUT_FILE,
    mode="w+",
    dtype=np.float32,
    shape=(num_receivers, num_subcarriers, num_features)
)

print("\nCreating Transformer input...")
print("Output shape:", X.shape)

# Process batches to limit RAM use
batch_size = 256

for start in range(0, num_receivers, batch_size):

    end = min(start + batch_size, num_receivers)

    coeff = X_complex[start:end].astype(np.complex64)
    mask = X_valid[start:end]
    direction = X_angles[start:end]

    # Invalid paths contribute zero.
    coeff = np.where(mask, coeff, 0)

    delay = features["mean_delay"]

    # NOTE:
    # This script uses a temporary per-receiver delay approximation
    # only if individual path delays are not stored.
    #
    # A later validation step must replace this with individual
    # path delays if available in the original MATLAB data.

    receiver_delay = delay[receiver_indices[start:end]]

    # Construct a relative frequency-dependent phase.
    phase_rotation = np.exp(
        -1j * 2 * np.pi *
        frequency_offsets[None, :, None] *
        receiver_delay[:, None, None]
    )

    # Shape: batch x subcarriers x paths
    frequency_coeff = (
        coeff[:, None, :] * phase_rotation
    )

    # Aggregate complex path coefficients
    total_channel = np.sum(frequency_coeff, axis=2)

    X[start:end, :, 0] = total_channel.real
    X[start:end, :, 1] = total_channel.imag

    # Direction-weighted complex channel features
    for k in range(8):

        weighted = np.sum(
            frequency_coeff *
            direction[:, None, :, k],
            axis=2
        )

        X[start:end, :, 2 + 2*k] = weighted.real
        X[start:end, :, 3 + 2*k] = weighted.imag

    # Normalized frequency
    X[start:end, :, 18] = normalized_frequency[None, :]

    if start % (batch_size * 20) == 0:
        print(f"Processed {end}/{num_receivers} receivers")


# Flush data to disk
X.flush()

# Save aligned labels
np.save(OUTPUT_LABELS, Y)


# ============================================================
# STEP 7: VALIDATE OUTPUT
# ============================================================

print("\n========== OUTPUT VALIDATION ==========\n")

print("Input shape:", X.shape)
print("Label shape:", Y.shape)

print("Input finite:", np.isfinite(X).all())
print("Labels finite:", np.isfinite(Y).all())

print("Minimum label:", Y.min())
print("Maximum label:", Y.max())

print("Unique labels:", len(np.unique(Y)))

assert X.shape == (num_receivers, 64, 19)
assert Y.shape == (num_receivers, 64)
assert np.isfinite(X).all()


# ============================================================
# STEP 8: SAVE REPORT
# ============================================================

report = {
    "input_shape": list(X.shape),
    "label_shape": list(Y.shape),
    "num_receivers": int(num_receivers),
    "num_subcarriers": int(num_subcarriers),
    "num_features": int(num_features),
    "num_beams": int(len(np.unique(Y))),
    "carrier_frequency_hz": fc,
    "bandwidth_hz": bandwidth,
    "subcarrier_spacing_hz": subcarrier_spacing,
    "feature_names": [
        "channel_real",
        "channel_imag",
        "aod_az_sin_real",
        "aod_az_sin_imag",
        "aod_az_cos_real",
        "aod_az_cos_imag",
        "aod_el_sin_real",
        "aod_el_sin_imag",
        "aod_el_cos_real",
        "aod_el_cos_imag",
        "aoa_az_sin_real",
        "aoa_az_sin_imag",
        "aoa_az_cos_real",
        "aoa_az_cos_imag",
        "aoa_el_sin_real",
        "aoa_el_sin_imag",
        "aoa_el_cos_real",
        "aoa_el_cos_imag",
        "normalized_frequency"
    ],
    "note": (
        "First-pass frequency-dependent representation. "
        "Individual path delays and channel conventions "
        "require validation before final scientific use."
    )
}

with open(OUTPUT_REPORT, "w") as f:
    json.dump(report, f, indent=4)

print("\nSaved files:")
print(OUTPUT_FILE)
print(OUTPUT_LABELS)
print(OUTPUT_REPORT)

print("\n========== DATA PREPARATION COMPLETE ==========")