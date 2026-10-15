from pathlib import Path
import json
import numpy as np
from scipy.io import loadmat

ROOT = Path.cwd()
DATA = ROOT / "data"
OUT = ROOT / "results" / "experiment_32x32"
OUT.mkdir(parents=True, exist_ok=True)

# Simulation settings
CARRIER_HZ = 3.5e9
BANDWIDTH_HZ = 100e6
NUM_SUBCARRIERS = 64
NX = 32
NY = 32
NUM_BEAMS = 64

def load_variable(filename, variable):
    path = DATA / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    data = loadmat(path)
    if variable not in data:
        raise KeyError(f"{variable} not found in {filename}")
    return np.asarray(data[variable])

print("Loading original channel data...")

power_db = load_variable(
    "power_t001_tx000_r000.mat", "power"
).astype(np.float64)

phase_deg = load_variable(
    "phase_t001_tx000_r000.mat", "phase"
).astype(np.float64)

delay_s = load_variable(
    "delay_t001_tx000_r000.mat", "delay"
).astype(np.float64)

inter = load_variable(
    "inter_t001_tx000_r000.mat", "inter"
).astype(np.float64)

print("Power:", power_db.shape)
print("Phase:", phase_deg.shape)
print("Delay:", delay_s.shape)
print("Interactions:", inter.shape)

if not (
    power_db.shape == phase_deg.shape
    == delay_s.shape == inter.shape
):
    raise ValueError("Channel arrays do not have matching shapes.")

num_receivers, num_paths = power_db.shape

# Only paths with finite values and a valid interaction count are used.
valid = (
    np.isfinite(power_db)
    & np.isfinite(phase_deg)
    & np.isfinite(delay_s)
    & np.isfinite(inter)
    & (inter >= 0)
    & (delay_s > 0)
)

# Create a 64-direction codebook over the visible angular grid.
# The angles below are direction-cosine coordinates (u, v).
# They are not claimed to be a validated site-specific codebook.
side = int(np.sqrt(NUM_BEAMS))
if side * side != NUM_BEAMS:
    raise ValueError("NUM_BEAMS must be a perfect square in this version.")

u_grid = np.linspace(-0.95, 0.95, side)
v_grid = np.linspace(-0.95, 0.95, side)
beam_uv = np.array([(u, v) for v in v_grid for u in u_grid])

# Half-wavelength spacing at the carrier frequency.
wavelength = 299792458.0 / CARRIER_HZ
spacing = wavelength / 2

x = np.arange(NX) - (NX - 1) / 2
y = np.arange(NY) - (NY - 1) / 2
xx, yy = np.meshgrid(x, y, indexing="ij")

# Each codebook beam has 1,024 antenna weights.
beam_weights = np.exp(
    -1j * 2 * np.pi * spacing / wavelength
    * (
        beam_uv[:, 0, None] * xx.ravel()[None, :]
        + beam_uv[:, 1, None] * yy.ravel()[None, :]
    )
) / np.sqrt(NX * NY)

labels = np.full(num_receivers, -1, dtype=np.int16)
usable = np.any(valid, axis=1)

print("Receivers:", num_receivers)
print("Usable receivers:", int(usable.sum()))
print("Array elements:", NX * NY)
print("Candidate beams:", NUM_BEAMS)
print("Calculating labels...")

# This computes a first-pass angular beam label per receiver.
# The original MATLAB files do not contain 1,024-element channel
# vectors, so this is an angle-based proxy, not measured array gain.
for start in range(0, num_receivers, 1000):
    end = min(start + 1000, num_receivers)
    rows = np.arange(start, end)

    for r in rows:
        mask = valid[r]
        if not np.any(mask):
            continue

        # Treat phase as degrees and power as dB for this first pass.
        # Validate these conventions against scenario documentation.
        path_amp = 10.0 ** (power_db[r, mask] / 20.0)
        path_phase = np.deg2rad(phase_deg[r, mask])
        weights = path_amp * np.exp(1j * path_phase)

        # This placeholder cannot infer direction from interaction
        # coordinates reliably; don't fabricate angle labels.
        # Mark for review instead of inventing beam targets.
        labels[r] = -1

    if end % 10000 == 0 or end == num_receivers:
        print(f"Checked {end}/{num_receivers} receivers")

# Save geometry and codebook for the next validated channel step.
np.save(OUT / "beam_codebook_uv.npy", beam_uv.astype(np.float32))
np.save(OUT / "array_x_positions.npy", (xx.ravel() * spacing).astype(np.float32))
np.save(OUT / "array_y_positions.npy", (yy.ravel() * spacing).astype(np.float32))
np.save(OUT / "valid_path_mask.npy", valid)
np.save(OUT / "usable_receiver_mask.npy", usable)

config = {
    "array": "32x32 UPA",
    "antenna_elements": NX * NY,
    "carrier_frequency_hz": CARRIER_HZ,
    "bandwidth_hz": BANDWIDTH_HZ,
    "subcarriers": NUM_SUBCARRIERS,
    "element_spacing_m": float(spacing),
    "candidate_beams": NUM_BEAMS,
    "label_status": "NOT GENERATED: angular convention and path-to-beam mapping require validation",
    "usable_receivers": int(usable.sum()),
    "total_receivers": int(num_receivers),
}

(OUT / "experiment_config.json").write_text(
    json.dumps(config, indent=2), encoding="utf-8"
)

print("\nSaved codebook, array geometry, and masks to:")
print(OUT)
print("No final beam labels were generated.")
print("Validate angle conventions before training.")
