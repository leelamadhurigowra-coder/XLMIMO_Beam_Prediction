from pathlib import Path
import numpy as np
from scipy.io import loadmat

root = Path.cwd()
data = root / "data"

def read_mat(filename, key):
    return loadmat(data / filename)[key]

tx = read_mat("tx_pos_t001_tx000_r000.mat", "tx_pos")[0]
rx = read_mat("rx_pos_t001_tx000_r000.mat", "rx_pos")
inter_pos = read_mat(
    "inter_pos_t001_tx000_r000.mat", "inter_pos"
)
inter = read_mat("inter_t001_tx000_r000.mat", "inter")

print("TX position:", tx)
print("RX shape:", rx.shape)
print("Interaction positions shape:", inter_pos.shape)

# Inspect one receiver and the first few paths.
for r in [0, 1, 10, 100, 1000]:
    print("\nReceiver index:", r)
    print("RX position:", rx[r])

    for p in range(min(3, inter_pos.shape[1])):
        count = inter[r, p]
        print(f" Path {p}: interaction count={count}")

        if np.isfinite(count) and count > 0:
            points = inter_pos[r, p]
            points = points[np.all(np.isfinite(points), axis=1)]
            print("  Valid interaction points:", points[:int(count)])

print("\nGEOMETRY INSPECTION COMPLETE")
