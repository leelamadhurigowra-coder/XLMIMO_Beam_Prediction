import numpy as np
from pathlib import Path

# Find the main project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Dataset location
file_path = PROJECT_DIR / "results" / "engineered_features.npz"

print("\n========== CHANNEL FEATURE CHECK ==========\n")

if not file_path.exists():
    print("ERROR: Dataset not found:")
    print(file_path)
    raise FileNotFoundError(file_path)

# Load dataset
data = np.load(file_path)

print("Dataset:", file_path)

print("\nAvailable arrays:")

for name in data.files:
    arr = data[name]
    print(f"{name:25s} Shape: {arr.shape}, Type: {arr.dtype}")

print("\n========== IMPORTANT CHANNEL ARRAYS ==========\n")

keywords = [
    "power",
    "phase",
    "delay",
    "complex",
    "valid",
    "aod",
    "aoa"
]

for name in data.files:

    if any(word in name.lower() for word in keywords):

        arr = data[name]

        print(f"\n{name}")
        print("Shape:", arr.shape)
        print("Data type:", arr.dtype)

        if np.issubdtype(arr.dtype, np.number):

            finite = np.isfinite(arr)

            print("Finite values:", np.sum(finite))
            print("Non-finite values:", np.sum(~finite))

print("\n========== CHECK COMPLETE ==========")