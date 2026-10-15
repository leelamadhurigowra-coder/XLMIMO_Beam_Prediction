
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# ==========================================
# STEP 1: Define paths
# ==========================================

DATA_PATH = "data"
RESULTS_PATH = "results"

os.makedirs(RESULTS_PATH, exist_ok=True)

file_path = os.path.join(
    DATA_PATH,
    "power_t001_tx000_r000.mat"
)

# ==========================================
# STEP 2: Load power data
# ==========================================

mat_data = sio.loadmat(file_path)

print("=" * 60)
print("DEEPMIMO CHANNEL POWER ANALYSIS")
print("=" * 60)

# Identify the actual MATLAB variable
variable_names = [
    key for key in mat_data
    if not key.startswith("__")
]

print("\nVariables found:", variable_names)

power = mat_data[variable_names[0]]

power = np.asarray(power, dtype=float)

print("\nPower array shape:", power.shape)
print("Data type:", power.dtype)

# ==========================================
# STEP 3: Clean data for analysis
# ==========================================

valid_power = power[np.isfinite(power)]

print("\nTotal values:", power.size)
print("Valid values:", valid_power.size)
print("Non-finite values:", power.size - valid_power.size)

if valid_power.size == 0:
    raise ValueError("No valid power values found.")

# ==========================================
# STEP 4: Calculate statistics
# ==========================================

print("\nCHANNEL POWER STATISTICS")

print("Minimum:", np.min(valid_power))
print("Maximum:", np.max(valid_power))
print("Mean:", np.mean(valid_power))
print("Median:", np.median(valid_power))
print("Standard deviation:", np.std(valid_power))

# ==========================================
# STEP 5: Plot histogram
# ==========================================

plt.figure(figsize=(10, 6))

plt.hist(
    valid_power,
    bins=50,
    edgecolor="black",
    alpha=0.75
)

plt.title("Distribution of DeepMIMO Path Power")
plt.xlabel("Path Power (dataset units)")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_PATH,
        "channel_power_histogram.png"
    ),
    dpi=300
)

plt.show()

# ==========================================
# STEP 6: Plot power across samples
# ==========================================

plt.figure(figsize=(10, 6))

plt.plot(
    valid_power[:min(5000, len(valid_power))],
    linewidth=0.7
)

plt.title("Channel Power Across Dataset Entries")
plt.xlabel("Valid data entry index")
plt.ylabel("Path Power (dataset units)")
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_PATH,
        "channel_power_variation.png"
    ),
    dpi=300
)

plt.show()

print("\nAnalysis completed successfully!")
print("Graphs saved in the results folder.")