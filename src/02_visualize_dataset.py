
import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# STEP 1: Dataset directory
# ==========================================

DATA_PATH = "data"

# ==========================================
# STEP 2: Load receiver positions
# ==========================================

file_path = os.path.join(
    DATA_PATH,
    "rx_pos_t001_tx000_r000.mat"
)

data = sio.loadmat(file_path)

print("Available variables:", list(data.keys()))

# ==========================================
# STEP 3: Extract position array
# ==========================================

for key in data:
    if not key.startswith("__"):
        positions = data[key]
        print("Variable:", key)
        print("Shape:", positions.shape)
        print("Data type:", positions.dtype)

# ==========================================
# STEP 4: Visualize user positions
# ==========================================

positions = np.asarray(positions)

plt.figure(figsize=(10, 7))

if positions.ndim == 2 and positions.shape[1] >= 2:
    plt.scatter(
        positions[:, 0],
        positions[:, 1],
        s=2,
        alpha=0.5
    )

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("DeepMIMO ASU Campus - Receiver Locations")
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)

    plt.savefig(
        "results/receiver_locations.png",
        dpi=300
    )

    plt.show()

    print("Visualization saved successfully!")

else:
    print("Unexpected position dimensions.")
    print("We need to inspect the array format.")