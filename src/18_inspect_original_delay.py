
import numpy as np
from scipy.io import loadmat
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
delay_file = PROJECT_DIR / "data" / "delay_t001_tx000_r000.mat"

print("\n========== ORIGINAL DELAY INSPECTION ==========\n")
print("File:", delay_file)

if not delay_file.exists():
    raise FileNotFoundError(f"Delay file not found: {delay_file}")

mat_data = loadmat(delay_file)

print("\nAvailable MATLAB variables:")

for key, value in mat_data.items():
    if not key.startswith("__"):
        print("\nVariable:", key)
        print("Shape:", value.shape)
        print("Data type:", value.dtype)

        if np.issubdtype(value.dtype, np.number):
            finite = np.isfinite(value)

            print("Finite values:", np.sum(finite))
            print("Non-finite values:", np.sum(~finite))

            if np.any(finite):
                valid = value[finite]

                print("Minimum:", np.min(valid))
                print("Maximum:", np.max(valid))
                print("Mean:", np.mean(valid))
                print("First 10 values:")
                print(valid[:10])

print("\n========== INSPECTION COMPLETE ==========")