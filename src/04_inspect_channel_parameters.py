
import os
import numpy as np
import scipy.io as sio

# ==========================================
# STEP 1: Dataset path
# ==========================================

DATA_PATH = "data"

print("=" * 65)
print("DEEPMIMO CHANNEL PARAMETER INSPECTION")
print("=" * 65)

# ==========================================
# STEP 2: Files to inspect
# ==========================================

files_to_inspect = [
    "power_t001_tx000_r000.mat",
    "phase_t001_tx000_r000.mat",
    "delay_t001_tx000_r000.mat",
    "aod_az_t001_tx000_r000.mat",
    "aod_el_t001_tx000_r000.mat",
    "aoa_az_t001_tx000_r000.mat",
    "aoa_el_t001_tx000_r000.mat",
    "rx_pos_t001_tx000_r000.mat",
    "tx_pos_t001_tx000_r000.mat"
]

# ==========================================
# STEP 3: Inspect each file
# ==========================================

for filename in files_to_inspect:

    filepath = os.path.join(DATA_PATH, filename)

    print("\n" + "-" * 65)
    print("FILE:", filename)

    if not os.path.exists(filepath):
        print("File not found. Skipping.")
        continue

    try:
        data = sio.loadmat(filepath)

        variables = [
            key for key in data
            if not key.startswith("__")
        ]

        for key in variables:

            array = np.asarray(data[key])

            print("\nVariable:", key)
            print("Shape:", array.shape)
            print("Data type:", array.dtype)

            if np.issubdtype(array.dtype, np.number):

                finite = array[np.isfinite(array)]

                if finite.size > 0:
                    print("Minimum:", np.min(finite))
                    print("Maximum:", np.max(finite))
                    print("Mean:", np.mean(finite))

    except Exception as error:
        print("Error:", error)

print("\nInspection completed!")