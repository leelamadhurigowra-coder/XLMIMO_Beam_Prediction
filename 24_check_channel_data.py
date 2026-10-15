from pathlib import Path
from scipy.io import loadmat
import numpy as np

root = Path.cwd()
data = root / "data"

names = [
    "power_t001_tx000_r000.mat",
    "phase_t001_tx000_r000.mat",
    "delay_t001_tx000_r000.mat",
    "inter_t001_tx000_r000.mat",
    "inter_pos_t001_tx000_r000.mat",
    "rx_pos_t001_tx000_r000.mat",
    "tx_pos_t001_tx000_r000.mat",
]

for name in names:
    path = data / name
    print("\n" + "=" * 55)
    print("FILE:", name)

    if not path.exists():
        print("NOT FOUND")
        continue

    contents = loadmat(path)

    for key, value in contents.items():
        if key.startswith("__") or not isinstance(value, np.ndarray):
            continue

        print("Variable:", key)
        print("Shape:", value.shape)
        print("Data type:", value.dtype)

        if np.issubdtype(value.dtype, np.number):
            finite = value[np.isfinite(value)]
            if finite.size:
                print("Finite values:", finite.size, "/", value.size)
                print("Min:", float(finite.min()))
                print("Max:", float(finite.max()))

print("\nCHANNEL CHECK COMPLETE")
