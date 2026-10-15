from pathlib import Path
import json
import numpy as np
from scipy.io import loadmat

ROOT = Path.cwd()
DATA = ROOT / "data"
OUT = ROOT / "results" / "experiment_32x32" / "real_channel"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------- CONFIGURATION ----------------
C = 299792458.0
FC = 3.5e9
BANDWIDTH = 100e6
N_SC = 64
NX = NY = 32
N_BEAMS = 64
CHUNK = 64

def load_mat(filename, variable):
    path = DATA / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    result = loadmat(path)
    if variable not in result:
        raise KeyError(f"{variable} missing in {filename}")
    return np.asarray(result[variable])

print("Loading propagation data...")

tx = load_mat("tx_pos_t001_tx000_r000.mat", "tx_pos").reshape(-1, 3)[0]
rx = load_mat("rx_pos_t001_tx000_r000.mat", "rx_pos").astype(np.float64)
power_db = load_mat("power_t001_tx000_r000.mat", "power").astype(np.float64)
phase_deg = load_mat("phase_t001_tx000_r000.mat", "phase").astype(np.float64)
delay = load_mat("delay_t001_tx000_r000.mat", "delay").astype(np.float64)
inter_pos = load_mat("inter_pos_t001_tx000_r000.mat", "inter_pos").astype(np.float64)

n_rx, n_paths = power_db.shape

if phase_deg.shape != power_db.shape or delay.shape != power_db.shape:
    raise ValueError("Power, phase and delay shapes do not match.")
if inter_pos.shape[:2] != power_db.shape or inter_pos.shape[-1] != 3:
    raise ValueError(f"Unexpected interaction-position shape: {inter_pos.shape}")

# ---------------- ARRAY AND CODEBOOK ----------------
wavelength = C / FC
spacing = wavelength / 2
k = 2 * np.pi / wavelength

x = (np.arange(NX) - (NX - 1) / 2) * spacing
y = (np.arange(NY) - (NY - 1) / 2) * spacing
xx, yy = np.meshgrid(x, y, indexing="ij")

# Array lies in the local XY plane; broadside is +Z.
array_x = xx.ravel()
array_y = yy.ravel()

azimuths = np.linspace(-np.pi, np.pi, 8, endpoint=False)
off_broadside = np.deg2rad(np.linspace(5, 60, 8))
beam_dirs = np.array([
    [np.sin(t) * np.cos(phi),
     np.sin(t) * np.sin(phi),
     np.cos(t)]
    for t in off_broadside for phi in azimuths
], dtype=np.float64)

beam_u = beam_dirs[:, 0]
beam_v = beam_dirs[:, 1]

# Frequency offsets span 100 MHz with 100 MHz / 64 spacing.
freq_offsets = (np.arange(N_SC) - N_SC / 2) * (BANDWIDTH / N_SC)

# ---------------- PATH DIRECTIONS ----------------
# Use the first finite interaction coordinate as the departure direction.
# If no interaction coordinate exists, use the direct TX-to-RX direction.
# Interaction ordering and the coordinate convention must be validated.
rx_to_tx = rx - tx[None, :]
direct_dist = np.linalg.norm(rx_to_tx, axis=1)
direct_unit = rx_to_tx / np.maximum(direct_dist[:, None], 1e-12)

path_dirs = np.zeros((n_rx, n_paths, 3), dtype=np.float32)
has_direction = np.zeros((n_rx, n_paths), dtype=bool)
used_interaction = np.zeros((n_rx, n_paths), dtype=bool)

for p in range(n_paths):
    points = inter_pos[:, p, :, :]
    finite_point = np.all(np.isfinite(points), axis=2)
    any_point = finite_point.any(axis=1)

    # argmax returns the first True position for rows with any valid point.
    first_idx = np.argmax(finite_point, axis=1)
    rows = np.arange(n_rx)
    first_point = points[rows, first_idx]

    direction = first_point - tx[None, :]
    norm = np.linalg.norm(direction, axis=1)
    use = any_point & np.isfinite(norm) & (norm > 0)

    path_dirs[:, p, :] = direct_unit.astype(np.float32)
    path_dirs[use, p, :] = (
        direction[use] / norm[use, None]
    ).astype(np.float32)

    has_direction[:, p] = (
        np.isfinite(direct_dist) & (direct_dist > 0)
    ) | use
    used_interaction[:, p] = use

# ---------------- PATH VALIDITY ----------------
valid_path = (
    np.isfinite(power_db)
    & np.isfinite(phase_deg)
    & np.isfinite(delay)
    & (power_db > -300)
    & (delay > 0)
    & has_direction
)

usable_rx = (
    np.all(np.isfinite(rx), axis=1)
    & (direct_dist > 0)
    & np.any(valid_path, axis=1)
)

receiver_ids = np.flatnonzero(usable_rx)
print("Total receivers:", n_rx)
print("Usable receivers:", len(receiver_ids))
print("Antenna elements:", NX * NY)
print("Candidate beams:", N_BEAMS)
print("Subcarriers:", N_SC)

# Save the beam power for each receiver/subcarrier/beam.
# Shape: usable receivers x 64 subcarriers x 64 beams.
response_path = OUT / "beam_power_response.npy"
response = np.lib.format.open_memmap(
    response_path, mode="w+", dtype=np.float32,
    shape=(len(receiver_ids), N_SC, N_BEAMS)
)

labels = np.empty((len(receiver_ids), N_SC), dtype=np.int16)
frequency_channel = np.empty((len(receiver_ids), N_SC), dtype=np.complex64)

print("\nCalculating path-based array responses...")

for start in range(0, len(receiver_ids), CHUNK):
    end = min(start + CHUNK, len(receiver_ids))
    ids = receiver_ids[start:end]
    batch = len(ids)

    # Shape: batch x paths x beams
    u_path = path_dirs[ids, :, 0].astype(np.float64)
    v_path = path_dirs[ids, :, 1].astype(np.float64)

    du = u_path[:, :, None] - beam_u[None, None, :]
    dv = v_path[:, :, None] - beam_v[None, None, :]

    # Separable normalized UPA steering response.
    ax = np.exp(1j * k * du[..., None] * x[None, None, None, :]).sum(axis=3)
    ay = np.exp(1j * k * dv[..., None] * y[None, None, None, :]).sum(axis=3)
    array_factor = (ax * ay) / (NX * NY)

    p_db = power_db[ids]
    ph = np.deg2rad(phase_deg[ids])
    tau = delay[ids]
    mask = valid_path[ids]

    amplitude = np.zeros_like(p_db, dtype=np.float64)
    amplitude[mask] = 10 ** (p_db[mask] / 20.0)

    coeff = amplitude * np.exp(1j * ph)
    coeff[~mask] = 0.0

    # Frequency-dependent path coefficient.
    delay_phase = np.exp(
        -1j * 2 * np.pi * tau[:, :, None]
        * freq_offsets[None, None, :]
    )
    path_coeff = (
        coeff[:, :, None] * delay_phase
    )  # batch x paths x subcarriers

    # Coherent path sum: batch x subcarriers x beams.
    h = np.einsum(
        "bpf,bpq->bfq",
        path_coeff,
        array_factor,
        optimize=True
    )

    # The channel's total complex response at each frequency.
    # Sum over path coefficients without beam steering.
    h_total = np.sum(path_coeff, axis=1)
    frequency_channel[start:end] = h_total.astype(np.complex64)

    beam_power = np.abs(h) ** 2
    response[start:end] = beam_power.astype(np.float32)
    labels[start:end] = np.argmax(beam_power, axis=2).astype(np.int16)

    if end % 5000 == 0 or end == len(receiver_ids):
        print(f"Processed {end}/{len(receiver_ids)} receivers")

response.flush()

np.save(OUT / "receiver_indices.npy", receiver_ids)
np.save(OUT / "beam_labels.npy", labels)
np.save(OUT / "beam_directions_xyz.npy", beam_dirs.astype(np.float32))
np.save(OUT / "frequency_offsets_hz.npy", freq_offsets.astype(np.float32))
np.save(OUT / "valid_path_mask.npy", valid_path[receiver_ids])
np.save(OUT / "used_interaction_direction_mask.npy", used_interaction[receiver_ids])

metadata = {
    "array": "32x32 UPA",
    "antenna_elements": NX * NY,
    "carrier_frequency_hz": FC,
    "bandwidth_hz": BANDWIDTH,
    "subcarriers": N_SC,
    "element_spacing_m": float(spacing),
    "candidate_beams": N_BEAMS,
    "total_receivers": int(n_rx),
    "usable_receivers": int(len(receiver_ids)),
    "response_shape": [int(len(receiver_ids)), N_SC, N_BEAMS],
    "label_shape": list(labels.shape),
    "path_model": "Coherent sum of path coefficients and UPA steering responses",
    "assumptions": [
        "Power values are interpreted as dB and converted to amplitude via 10**(power_db/20).",
        "Phase values are interpreted as degrees.",
        "Path phase is treated as a carrier-referenced coefficient.",
        "Subcarrier phase offsets use exp(-j*2*pi*frequency_offset*delay).",
        "Array is assumed to lie in the local XY plane with broadside +Z.",
        "The first finite interaction coordinate is assumed to be the first interaction from the transmitter.",
        "Paths without stored interactions use the direct TX-to-RX direction."
    ],
    "limitations": [
        "These assumptions must be validated against the scenario documentation.",
        "This is a reconstructed simulation, not a measured physical 32x32 array response.",
        "No RF hardware impairments or interference are modeled."
    ]
}

(OUT / "channel_metadata.json").write_text(
    json.dumps(metadata, indent=2), encoding="utf-8"
)

print("\nCHANNEL RESPONSE GENERATED")
print("Response file:", response_path)
print("Response shape:", response.shape)
print("Labels shape:", labels.shape)
print("Unique labels:", np.unique(labels).size)
print("Output folder:", OUT)
print("Validate the documented assumptions before treating labels as final.")

