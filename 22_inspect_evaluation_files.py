
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

files = [
    "beam_labels.npy",
    "beam_label_receiver_indices.npy",
    "frequency_dataset.npy",
    "transformer_diagnostics.npz",
    "transformer_report.json",
]

print("\n=== XL-MIMO EVALUATION FILE INSPECTION ===")

for name in files:
    path = RESULTS / name
    print(f"\nFile: {name}")

    if not path.exists():
        print("NOT FOUND")
        continue

    print(f"Size: {path.stat().st_size / (1024**2):.2f} MB")

    if path.suffix == ".npy":
        arr = np.load(path, mmap_mode="r")
        print("Shape:", arr.shape)
        print("Data type:", arr.dtype)

    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                print(key, data[key].shape, data[key].dtype)

    elif path.suffix == ".json":
        print(path.read_text(encoding="utf-8")[:3000])

print("\n=== INSPECTION COMPLETE ===")
