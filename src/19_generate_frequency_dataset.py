
import json
import numpy as np
from scipy.io import loadmat
from pathlib import Path

# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

FEATURE_FILE = RESULTS_DIR / "engineered_features.npz"
DELAY_FILE = DATA_DIR / "delay_t001_tx000_r000.mat"

# IMPORTANT: Both files are inside the results folder
LABEL_FILE = RESULTS_DIR / "beam_labels.npy"
INDEX_FILE = RESULTS_DIR / "beam_label_receiver_indices.npy"

OUTPUT_FILE = RESULTS_DIR / "frequency_dataset.npy"
METADATA_FILE = RESULTS_DIR / "frequency_dataset_metadata.json"


# ============================================================
# 2. COMMUNICATION PARAMETERS
# ============================================================

CARRIER_FREQUENCY = 3.5e9
BANDWIDTH = 100e6
NUM_SUBCARRIERS = 64

# Frequency offsets around the carrier frequency
frequency_offsets = np.linspace(
    -BANDWIDTH / 2,
    BANDWIDTH / 2,
    NUM_SUBCARRIERS,
    dtype=np.float64
)

# Normalize frequency between -1 and +1
normalized_frequency = (
    frequency_offsets / (BANDWIDTH / 2)
).astype(np.float32)


# ============================================================
# 3. CHECK WHETHER INPUT FILES EXIST
# ============================================================

print("\n========== CHECKING INPUT FILES ==========\n")

required_files = [
    FEATURE_FILE,
    DELAY_FILE,
    LABEL_FILE,
    INDEX_FILE
]

for file_path in required_files:
    print("Checking:", file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )

print("\nAll required files found!")


# ============================================================
# 4. LOAD DATA
# ============================================================

print("\n========== LOADING DATA ==========\n")

features = np.load(FEATURE_FILE)

complex_path = features["complex_path"]
valid_mask = features["valid_mask"]

direction_names = [
    "aod_az_sin",
    "aod_az_cos",
    "aod_el_sin",
    "aod_el_cos",
    "aoa_az_sin",
    "aoa_az_cos",
    "aoa_el_sin",
    "aoa_el_cos"
]

direction_features = [
    features[name] for name in direction_names
]

# Load original MATLAB delay data
delay_data = loadmat(DELAY_FILE)
delay = delay_data["delay"]

# Load beam labels and corresponding receiver indices
receiver_indices = np.load(INDEX_FILE)
beam_labels = np.load(LABEL_FILE)

print("Complex path shape:", complex_path.shape)
print("Delay shape:", delay.shape)
print("Validity mask shape:", valid_mask.shape)
print("Receiver indices shape:", receiver_indices.shape)
print("Beam labels shape:", beam_labels.shape)


# ============================================================
# 5. VALIDATE DATA DIMENSIONS
# ============================================================

print("\n========== VALIDATING DIMENSIONS ==========\n")

if complex_path.shape != delay.shape:
    raise ValueError(
        "Complex path and delay dimensions do not match."
    )

if valid_mask.shape != delay.shape:
    raise ValueError(
        "Validity mask and delay dimensions do not match."
    )

if len(receiver_indices) != len(beam_labels):
    raise ValueError(
        "Receiver indices and beam labels are not aligned."
    )

if np.any(receiver_indices < 0) or np.any(
    receiver_indices >= complex_path.shape[0]
):
    raise ValueError(
        "Some receiver indices are outside the dataset."
    )

print("All dimension checks passed!")


# ============================================================
# 6. DEFINE OUTPUT FEATURE STRUCTURE
# ============================================================

# Total channel response:
# Real + Imaginary = 2 features
#
# Eight direction-weighted responses:
# Each has Real + Imaginary = 16 features
#
# Normalized frequency = 1 feature
#
# Total = 19 features per subcarrier

NUM_FEATURES = 19

num_receivers = len(receiver_indices)

output_shape = (
    num_receivers,
    NUM_SUBCARRIERS,
    NUM_FEATURES
)

print("\n========== OUTPUT CONFIGURATION ==========\n")
print("Number of receivers:", num_receivers)
print("Number of subcarriers:", NUM_SUBCARRIERS)
print("Features per subcarrier:", NUM_FEATURES)
print("Expected output shape:", output_shape)


# ============================================================
# 7. CREATE DISK-BACKED OUTPUT ARRAY
# ============================================================

# This avoids loading the entire output tensor into RAM.
# The resulting file will be approximately 1.65 GB.

X = np.lib.format.open_memmap(
    OUTPUT_FILE,
    mode="w+",
    dtype=np.float32,
    shape=output_shape
)


# ============================================================
# 8. GENERATE FREQUENCY-DEPENDENT FEATURES
# ============================================================

BATCH_SIZE = 256

print("\n========== GENERATING DATASET ==========\n")

for start in range(0, num_receivers, BATCH_SIZE):

    end = min(start + BATCH_SIZE, num_receivers)

    # Original receiver row numbers
    original_rows = receiver_indices[start:end]

    # Select path coefficients and delays
    batch_coeff = complex_path[original_rows]
    batch_delay = delay[original_rows]
    batch_valid = valid_mask[original_rows]

    # Keep only valid paths
    batch_valid = (
        batch_valid
        & np.isfinite(batch_delay)
        & np.isfinite(batch_coeff.real)
        & np.isfinite(batch_coeff.imag)
    )

    # Replace invalid coefficients and delays with zero
    batch_coeff = np.where(
        batch_valid,
        batch_coeff,
        0.0
    )

    batch_delay = np.where(
        batch_valid,
        batch_delay,
        0.0
    )

    # Select direction features for this batch
    batch_directions = []

    for direction in direction_features:

        direction_batch = direction[original_rows]

        # Avoid invalid values affecting the weighted sums
        direction_batch = np.where(
            batch_valid,
            direction_batch,
            0.0
        )

        batch_directions.append(direction_batch)

    batch_size = end - start

    # --------------------------------------------------------
    # PROCESS ALL 64 SUBCARRIERS
    # --------------------------------------------------------

    for subcarrier in range(NUM_SUBCARRIERS):

        delta_f = frequency_offsets[subcarrier]

        # Frequency-dependent phase rotation:
        #
        # exp(-j * 2*pi * delta_f * delay)
        #
        frequency_rotation = np.exp(
            -1j * 2 * np.pi * delta_f * batch_delay
        )

        # Calculate frequency-dependent path response
        path_response = batch_coeff * frequency_rotation

        # Sum contributions from all paths
        total_channel = np.sum(
            path_response,
            axis=1
        )

        # First two features: total channel
        feature_columns = [
            total_channel.real,
            total_channel.imag
        ]

        # Direction-weighted channel responses
        for direction in batch_directions:

            weighted_channel = np.sum(
                path_response * direction,
                axis=1
            )

            feature_columns.append(
                weighted_channel.real
            )

            feature_columns.append(
                weighted_channel.imag
            )

        # Final feature: normalized frequency
        feature_columns.append(
            np.full(
                batch_size,
                normalized_frequency[subcarrier],
                dtype=np.float32
            )
        )

        # Store the 19 features
        X[
            start:end,
            subcarrier,
            :
        ] = np.column_stack(
            feature_columns
        ).astype(np.float32)

    # Save each completed batch
    X.flush()

    print(
        f"Processed receivers: {end}/{num_receivers}"
    )


# ============================================================
# 9. SAVE DATASET METADATA
# ============================================================

metadata = {
    "num_receivers": int(num_receivers),
    "num_subcarriers": int(NUM_SUBCARRIERS),
    "num_features_per_subcarrier": int(NUM_FEATURES),
    "carrier_frequency_hz": CARRIER_FREQUENCY,
    "bandwidth_hz": BANDWIDTH,
    "frequency_offsets_hz": frequency_offsets.tolist(),

    "feature_description": [
        "total_channel_real",
        "total_channel_imag",
        "aod_az_sin_weighted_real",
        "aod_az_sin_weighted_imag",
        "aod_az_cos_weighted_real",
        "aod_az_cos_weighted_imag",
        "aod_el_sin_weighted_real",
        "aod_el_sin_weighted_imag",
        "aod_el_cos_weighted_real",
        "aod_el_cos_weighted_imag",
        "aoa_az_sin_weighted_real",
        "aoa_az_sin_weighted_imag",
        "aoa_az_cos_weighted_real",
        "aoa_az_cos_weighted_imag",
        "aoa_el_sin_weighted_real",
        "aoa_el_sin_weighted_imag",
        "aoa_el_cos_weighted_real",
        "aoa_el_cos_weighted_imag",
        "normalized_frequency"
    ],

    "output_file": str(OUTPUT_FILE)
}

with open(METADATA_FILE, "w") as f:
    json.dump(metadata, f, indent=4)


# ============================================================
# 10. FINAL VERIFICATION
# ============================================================

print("\n========== DATASET COMPLETE ==========\n")

print("Saved dataset:", OUTPUT_FILE)
print("Dataset shape:", X.shape)
print("Dataset dtype:", X.dtype)
print("Metadata:", METADATA_FILE)

print("\nExpected shape:", output_shape)
print("Actual shape:", X.shape)

if X.shape != output_shape:
    raise ValueError("Final dataset shape is incorrect.")

print("\nAll checks passed!")
print("========== SUCCESS ==========")