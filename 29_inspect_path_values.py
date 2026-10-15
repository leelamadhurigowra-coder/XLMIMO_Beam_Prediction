from pathlib import Path
from scipy.io import loadmat
import numpy as np

data = Path.cwd() / "data"

def load(name, key):
    return loadmat(data / name)[key]

power = load("power_t001_tx000_r000.mat", "power")
phase = load("phase_t001_tx000_r000.mat", "phase")
delay = load("delay_t001_tx000_r000.mat", "delay")
inter = load("inter_t001_tx000_r000.mat", "inter")
positions = load("inter_pos_t001_tx000_r000.mat", "inter_pos")

r = 10

print("Receiver:", r)
print("Path | Power(dB) | Phase | Delay(s) | inter value")

for p in range(power.shape[1]):
    print(
        f"{p:4d} | {power[r,p]:9.3f} | "
        f"{phase[r,p]:8.3f} | {delay[r,p]:.6e} | "
        f"{inter[r,p]}"
    )

print("\nStored finite interaction coordinates:")
for p in range(power.shape[1]):
    points = positions[r, p]
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points):
        print(f"Path {p}: {len(points)} stored points")
        print(points)

print("\nCHECK COMPLETE")
