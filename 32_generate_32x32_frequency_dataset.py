from pathlib import Path
import json
import numpy as np
from scipy.io import loadmat

ROOT = Path.cwd()
DATA = ROOT / "data"
OUT = ROOT / "results" / "experiment_32x32"
OUT.mkdir(parents=True, exist_ok=True)

# Configuration
C = 299792458.0
FC = 3.5e9
BANDWIDTH = 100e6
N_SC = 64
NX = NY = 32
N_BEAMS = 64
D = C / FC / 2
CHUNK = 256

def read_mat(filename, key):
    return loadmat(DATA / filename)[key]

print("Loading channel and geometry...")
tx = read_mat("tx_pos_t001_tx000_r000.mat", "tx_pos").reshape(-1, 3)[0]
rx = read_mat("rx_pos_t001_tx000_r000.mat", "rx_pos").astype(np.float64)
power_db = read_mat("power_t001_tx000_r000.mat", "power").astype(np.float64)
phase_deg = read_mat("phase_t001_tx000_r000.mat", "phase").astype(np.float64)
delay = read_mat("delay_t001_tx000_r000.mat", "delay").astype(np.float64)

n_rx, n_paths = power_db.shape
if phase_deg.shape != power_db.shape or delay.shape != power_db.shape:
    raise ValueError("Power, phase, and delay shapes do not match.")
if rx.shape != (n_rx, 3):
    raise ValueError("Receiver position shape does not match channel data.")

# Array positions in its assumed local XY plane.
x = (np.arange(NX) - (NX - 1) / 2) * D
y = (np.arange(NY) - (NY - 1) / 2) * D
xx, yy = np.meshgrid(x, y, indexing="ij")
array_pos = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(NX * NY)])

# 64 beam directions matching the provisional codebook.
azimuths = np.linspace(-np.pi, np.pi, 8, endpoint=False)
off_broadside = np.deg2rad(np.linspace(5, 60, 8))
dirs = np.array([
    [np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)]
    for t in off_broadside for p in azimuths
])

# Build a frequency grid with spacing bandwidth / N_SC.
freq_offsets = (np.arange(N_SC) - N_SC / 2) * (BANDWIDTH / N_SC)

# LOS geometry is used to define a direction for each receiver.
# Multipath angles are not inferred from interaction coordinates here.
vec = rx - tx[None, :]
dist = np.linalg.norm(vec, axis=1)
rx_ok = np.all(np.isfinite(rx), axis=1) & (dist > 0)

unit = np.zeros_like(vec)
unit[rx_ok] = vec[rx_ok] / dist[rx_ok, None]
los_dirs = np.column_stack([unit[:, 0], unit[:, 1], np.abs(unit[:, 2])])
norm = np.linalg.norm(los_dirs, axis=1)
rx_ok &= norm > 0
los_dirs[rx_ok] /= norm[rx_ok, None]

labels = np.full(n_rx, -1, dtype=np.int16)
labels[rx_ok] = np.argmax(los_dirs[rx_ok] @ dirs.T, axis=1)

# Select the strongest valid path for a simple frequency-dependent
# channel proxy. This is NOT a full multipath array channel.
valid_path = (
    np.isfinite(power_db) & np.isfinite(phase_deg)
    & np.isfinite(delay) & (delay > 0)
)
valid_path &= np.isfinite(power_db) & (power_db > -300)

strongest_idx = np.argmax(np.where(valid_path, power_db, -np.inf), axis=1)
has_path = valid_path.any(axis=1)
rows = np.arange(n_rx)
selected_power = power_db[rows, strongest_idx]
selected_phase = np.deg2rad(phase_deg[rows, strongest_idx])
selected_delay = delay[rows, strongest_idx]

usable = rx_ok & has_path
labels[~usable] = -1

# Create a compact dataset per receiver and subcarrier.
# Features: channel real/imag, normalized frequency, LOS direction xyz,
# normalized distance, and selected path delay.
X_path = OUT / "frequency_dataset_los_proxy.npy"
X = np.lib.format.open_memmap(
    X_path, mode="w+", dtype=np.float32, shape=(n_rx, N_SC, 8)
)

for start in range(0, n_rx, CHUNK):
    end = min(start + CHUNK, n_rx)
    idx = np.arange(start, end)
    valid = usable[idx]

    f = freq_offsets[None, :]
    tau = selected_delay[idx, None]
    amp = 10 ** (selected_power[idx, None] / 20.0)
    phase = selected_phase[idx, None]

    h = amp * np.exp(1j * (phase - 2 * np.pi * f * tau))
    h[~valid, :] = 0

    features = np.zeros((len(idx), N_SC, 8), dtype=np.float32)
    features[:, :, 0] = h.real
    features[:, :, 1] = h.imag
    features[:, :, 2] = np.broadcast_to(
        freq_offsets / (BANDWIDTH / 2), (len(idx), N_SC)
    )
    features[:, :, 3:6] = los_dirs[idx, None, :]
    features[:, :, 6] = np.log1p(np.maximum(dist[idx], 0))[:, None]
    features[:, :, 7] = np.log1p(
        np.maximum(selected_delay[idx], 0)
    )[:, None]

    X[idx] = features

    if end % 10000 == 0 or end == n_rx:
        print(f"Processed {end}/{n_rx} receivers")

X.flush()

np.save(OUT / "los_proxy_beam_labels.npy", labels)
np.save(OUT / "valid_receiver_mask.npy", usable)
np.save(OUT / "frequency_offsets_hz.npy", freq_offsets.astype(np.float32))
np.save(OUT / "beam_directions_xyz.npy", dirs.astype(np.float32))
np.save(OUT / "array_positions_m.npy", array_pos.astype(np.float32))

metadata = {
    "array": "32x32 UPA",
    "antenna_elements": NX * NY,
    "carrier_frequency_hz": FC,
    "bandwidth_hz": BANDWIDTH,
    "subcarriers": N_SC,
    "element_spacing_m": D,
    "candidate_beams": N_BEAMS,
    "receivers": n_rx,
    "usable_receivers": int(usable.sum()),
    "features_per_subcarrier": 8,
    "feature_names": [
        "channel_real", "channel_imag", "normalized_frequency",
        "los_direction_x", "los_direction_y", "los_direction_z",
        "log1p_distance_m", "log1p_selected_delay_s"
    ],
    "label_method": "LOS geometry proxy",
    "channel_method": "strongest-path frequency proxy; no array steering applied",
    "scientific_warning": (
        "This dataset is a preliminary proxy, not a validated "
        "32x32 multipath channel. Labels are derived from LOS geometry; "
        "the features do not yet contain a 1024-element channel response."
    )
}
(OUT / "frequency_dataset_metadata.json").write_text(
    json.dumps(metadata, indent=2), encoding="utf-8"
)

print("\nDATASET CREATED")
print("Feature shape:", X.shape)
print("Label shape:", labels.shape)
print("Usable receivers:", int(usable.sum()))
print("Unique labels:", np.unique(labels[labels >= 0]).size)
print("Saved to:", X_path)
print("NOTE: This is a LOS/strongest-path proxy, not final XL-MIMO data.")
