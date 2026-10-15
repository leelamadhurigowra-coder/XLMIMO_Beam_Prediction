from pathlib import Path
from scipy.io import loadmat
import numpy as np

root = Path.cwd()
data = root / "data"
out = root / "results" / "experiment_32x32"
out.mkdir(parents=True, exist_ok=True)

names = [
    "power_t001_tx000_r000.mat",
    "phase_t001_tx000_r000.mat",
    "delay_t001_tx000_r000.mat",
    "inter_t001_tx000_r000.mat",
    "rx_pos_t001_tx000_r000.mat",
    "tx_pos_t001_tx000_r000.mat",
]

lines = []

for name in names:
    path = data / name
    lines.append(f"\nFILE: {name}")

    if not path.exists():
        lines.append("STATUS: NOT FOUND")
        continue

    contents = loadmat(path)
    for key, arr in contents.items():
        if key.startswith("__") or not isinstance(arr, np.ndarray):
            continue

        lines.append(f"Variable: {key}")
        lines.append(f"Shape: {arr.shape}, dtype: {arr.dtype}")

        if np.issubdtype(arr.dtype, np.number):
            finite = arr[np.isfinite(arr)]
            lines.append(f"Finite: {finite.size}/{arr.size}")
            if finite.size:
                lines.append(f"Min: {finite.min():.6g}")
                lines.append(f"Max: {finite.max():.6g}")

report = "\n".join(lines)
report_path = out / "channel_summary.txt"
report_path.write_text(report, encoding="utf-8")

print(report)
print(f"\nSaved summary to: {report_path}")
