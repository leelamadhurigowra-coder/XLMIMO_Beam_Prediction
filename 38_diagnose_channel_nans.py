from pathlib import Path
import numpy as np
from scipy.io import loadmat

root = Path.cwd()
data = root / "data"
out = root / "results" / "experiment_32x32" / "real_channel"

print("=== ORIGINAL CHANNEL CHECK ===")

for filename, variable in [
    ("power_t001_tx000_r000.mat", "power"),
    ("phase_t001_tx000_r000.mat", "phase"),
    ("delay_t001_tx000_r000.mat", "delay"),
]:
    a = loadmat(data / filename)[variable]
    print(
        f"{variable}: shape={a.shape}, "
        f"finite={np.isfinite(a).mean()*100:.2f}%, "
        f"NaNs={np.isnan(a).sum()}, "
        f"min={np.nanmin(a):.6g}, max={np.nanmax(a):.6g}"
    )

print("\n=== RESPONSE CHECK ===")
r = np.load(out / "beam_power_response.npy", mmap_mode="r")
print("Response shape:", r.shape)

for i in [0, 1, 10, 100, 1000]:
    if i < len(r):
        a = np.asarray(r[i])
        print(
            f"Receiver {i}: finite={np.isfinite(a).mean()*100:.2f}%, "
            f"NaNs={np.isnan(a).sum()}, "
            f"label row={np.load(out / 'beam_labels.npy', mmap_mode='r')[i].tolist()[:8]}"
        )

print("\nDIAGNOSTIC COMPLETE")
