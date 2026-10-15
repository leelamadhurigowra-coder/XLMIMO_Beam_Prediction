from pathlib import Path
import json
import numpy as np
from scipy.io import loadmat

ROOT = Path.cwd()
DATA = ROOT / "data"
OUT = ROOT / "results" / "experiment_32x32"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------- CONFIGURATION ----------------
C = 299792458.0
FC = 3.5e9
BANDWIDTH = 100e6
N_SUBCARRIERS = 64
NX = 32
NY = 32
N_BEAMS = 64
D = C / FC / 2

def load_mat(filename, variable):
    path = DATA / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    values = loadmat(path)
    if variable not in values:
        raise KeyError(f"{variable} not found in {filename}")
    return np.asarray(values[variable])

print("=" * 60)
print("GENERATING 32x32 UPA EXPERIMENT")
print("=" * 60)

tx = load_mat("tx_pos_t001_tx000_r000.mat", "tx_pos").reshape(-1, 3)
rx = load_mat("rx_pos_t001_tx000_r000.mat", "rx_pos")

power = load_mat("power_t001_tx000_r000.mat", "power")
delay = load_mat("delay_t001_tx000_r000.mat", "delay")

if tx.shape[0] != 1:
    raise ValueError("Expected one transmitter in this experiment.")
if rx.ndim != 2 or rx.shape[1] != 3:
    raise ValueError(f"Unexpected receiver shape: {rx.shape}")
if power.shape != delay.shape or power.shape[0] != rx.shape[0]:
    raise ValueError("Power, delay and receiver dimensions do not match.")

# ---------------- ARRAY GEOMETRY ----------------
x = (np.arange(NX) - (NX - 1) / 2) * D
y = (np.arange(NY) - (NY - 1) / 2) * D
xx, yy = np.meshgrid(x, y, indexing="ij")

array_positions = np.column_stack([
    xx.ravel(), yy.ravel(), np.zeros(NX * NY)
])

# 64 distinct directions:
# 8 azimuths x 8 off-broadside angles.
# Angles are measured relative to array broadside (+z).
azimuths = np.linspace(-np.pi, np.pi, 8, endpoint=False)
off_broadside = np.deg2rad(np.linspace(5, 60, 8))

directions = []
for theta in off_broadside:
    for phi in azimuths:
        directions.append([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])

directions = np.asarray(directions, dtype=np.float64)

# Steering vector for each candidate direction.
k = 2 * np.pi / (C / FC)
steering = np.exp(
    1j * k * (
        directions[:, 0, None] * array_positions[None, :, 0]
        + directions[:, 1, None] * array_positions[None, :, 1]
    )
) / np.sqrt(NX * NY)

# ---------------- LOS PROXY LABELS ----------------
# Vector from transmitter to receiver.
# This is a LOS/geometric baseline, not a multipath channel solution.
vec = rx.astype(np.float64) - tx[0][None, :]
distance = np.linalg.norm(vec, axis=1)

valid_rx = (
    np.all(np.isfinite(rx), axis=1)
    & np.isfinite(distance)
    & (distance > 0)
    & np.any(np.isfinite(power) & np.isfinite(delay) & (delay > 0), axis=1)
)

unit = np.zeros_like(vec)
unit[valid_rx] = vec[valid_rx] / distance[valid_rx, None]

# Array is in the XY plane; z is its broadside axis.
# Map the LOS direction into the forward hemisphere for this proxy.
u = unit[:, 0]
v = unit[:, 1]
w = np.maximum(np.abs(unit[:, 2]), 1e-12)
los_dirs = np.column_stack([u, v, w])
los_dirs /= np.maximum(
    np.linalg.norm(los_dirs, axis=1, keepdims=True), 1e-12
)

# Select the codebook direction with the largest directional cosine.
beam_scores = los_dirs @ directions.T
labels = np.argmax(beam_scores, axis=1).astype(np.int16)
labels[~valid_rx] = -1

# Save all outputs in the separate experiment folder.
np.save(OUT / "array_positions_m.npy", array_positions.astype(np.float32))
np.save(OUT / "beam_directions_xyz.npy", directions.astype(np.float32))
np.save(OUT / "los_proxy_beam_labels.npy", labels)
np.save(OUT / "valid_receiver_mask.npy", valid_rx)
np.save(OUT / "receiver_distances_m.npy", distance.astype(np.float32))

config = {
    "array": "32x32 uniform planar array",
    "antenna_elements": int(NX * NY),
    "carrier_frequency_hz": FC,
    "bandwidth_hz": BANDWIDTH,
    "subcarriers": N_SUBCARRIERS,
    "element_spacing_m": float(D),
    "candidate_beams": N_BEAMS,
    "transmitter_position": tx[0].tolist(),
    "total_receivers": int(len(rx)),
    "valid_receivers": int(valid_rx.sum()),
    "label_method": "LOS geometric direction proxy",
    "important_limitation": (
        "These are not multipath-optimal beam labels. "
        "Validate coordinate axes, array orientation and codebook "
        "against the scenario before scientific claims."
    )
}

(OUT / "experiment_config.json").write_text(
    json.dumps(config, indent=2), encoding="utf-8"
)

print("Array elements:", NX * NY)
print("Element spacing (m):", round(D, 6))
print("Array aperture x/y (m):", round((NX - 1) * D, 4))
print("Candidate beams:", N_BEAMS)
print("Total receivers:", len(rx))
print("Valid receivers:", int(valid_rx.sum()))
print("Label shape:", labels.shape)
print("Unique assigned labels:", np.unique(labels[labels >= 0]).size)
print("\nSaved files to:", OUT)
print("Labels are a LOS proxy, not final multipath-optimal labels.")
print("=" * 60)
