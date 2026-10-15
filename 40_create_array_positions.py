from pathlib import Path
import numpy as np

out = Path.cwd() / "results" / "experiment_32x32" / "real_channel"
out.mkdir(parents=True, exist_ok=True)

c = 299792458.0
fc = 3.5e9
d = c / fc / 2
n = 32

x = (np.arange(n) - (n - 1) / 2) * d
y = (np.arange(n) - (n - 1) / 2) * d
xx, yy = np.meshgrid(x, y, indexing="ij")

positions = np.column_stack([
    xx.ravel(),
    yy.ravel(),
    np.zeros(n * n)
]).astype(np.float32)

np.save(out / "array_positions_m.npy", positions)

print("Array positions recreated.")
print("Shape:", positions.shape)
print("Saved to:", out / "array_positions_m.npy")
