from pathlib import Path
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "results" / "experiment_32x32"

OUT.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("32x32 XL-MIMO EXPERIMENT: CHANNEL DATA AUDIT")
print("=" * 60)
print("Output folder:", OUT)

mat_files = sorted(DATA.glob("*.mat"))

if not mat_files:
    print("ERROR: No .mat files found directly inside data/")
    print("Check your data folder and file locations.")
    raise SystemExit(1)

print("\nMATLAB files found:", len(mat_files))

for path in mat_files:
    print("\n" + "-" * 50)
    print("File:", path.name)

    try:
        content = loadmat(path)
        variables = [
            key for key in content.keys()
            if not key.startswith("__")
        ]

        for key in variables:
            value = content[key]

            if isinstance(value, np.ndarray):
                print(
                    f"  {key}: shape={value.shape}, "
                    f"dtype={value.dtype}"
                )
            else:
                print(f"  {key}: {type(value).__name__}")

    except Exception as exc:
        print("Could not read this file:", exc)

print("\n" + "=" * 60)
print("AUDIT COMPLETE")
print("No existing labels or models were changed.")
print("=" * 60)
