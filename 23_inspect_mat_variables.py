from pathlib import Path
from scipy.io import loadmat

root = Path.cwd()
data = root / "data"

for path in sorted(data.rglob("*.mat")):
    print("\n" + "=" * 60)
    print("FILE:", path.relative_to(root))

    try:
        content = loadmat(path)
        for key, value in content.items():
            if not key.startswith("__"):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    except Exception as exc:
        print("ERROR:", exc)

print("\nMATLAB FILE INSPECTION COMPLETE")
