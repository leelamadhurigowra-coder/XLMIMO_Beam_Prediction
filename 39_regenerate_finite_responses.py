from pathlib import Path
import numpy as np
from scipy.io import loadmat

ROOT = Path.cwd()
DATA = ROOT / "data"
OUT = ROOT / "results" / "experiment_32x32" / "real_channel"

def read(name, key):
    return loadmat(DATA / name)[key]

power = read("power_t001_tx000_r000.mat", "power").astype(np.float64)
phase = np.deg2rad(read("phase_t001_tx000_r000.mat", "phase").astype(np.float64))
delay = read("delay_t001_tx000_r000.mat", "delay").astype(np.float64)

receiver_ids = np.load(OUT / "receiver_indices.npy")
dirs = np.load(OUT / "beam_directions_xyz.npy").astype(np.float64)
pos = np.load(OUT / "array_positions_m.npy").astype(np.float64)

C = 299792458.0
FC = 3.5e9
BANDWIDTH = 100e6
N_SC = 64
NX = NY = 32
wavelength = C / FC
k = 2 * np.pi / wavelength

# Use the actual saved element positions.
x = np.unique(np.round(pos[:, 0], 7))
y = np.unique(np.round(pos[:, 1], 7))

freq = (np.arange(N_SC) - N_SC / 2) * BANDWIDTH / N_SC
beam_u, beam_v = dirs[:, 0], dirs[:, 1]

out_path = OUT / "beam_power_response_finite.npy"
response = np.lib.format.open_memmap(
    out_path, mode="w+", dtype=np.float32,
    shape=(len(receiver_ids), N_SC, len(dirs))
)

valid = (
    np.isfinite(power) & np.isfinite(phase) & np.isfinite(delay)
    & (power > -300) & (delay > 0)
)

# Use the same provisional path directions as the previous script.
inter_pos = read("inter_pos_t001_tx000_r000.mat", "inter_pos").astype(np.float64)
tx = read("tx_pos_t001_tx000_r000.mat", "tx_pos").reshape(-1, 3)[0]
rx = read("rx_pos_t001_tx000_r000.mat", "rx_pos").astype(np.float64)

print("Regenerating finite response tensor...")

for start in range(0, len(receiver_ids), 32):
    end = min(start + 32, len(receiver_ids))
    ids = receiver_ids[start:end]

    # Path directions: first stored interaction, otherwise direct TX-to-RX.
    vec = rx[ids] - tx[None, :]
    dist = np.linalg.norm(vec, axis=1, keepdims=True)
    direct = vec / np.maximum(dist, 1e-12)

    path_dirs = np.repeat(direct[:, None, :], power.shape[1], axis=1)

    for p in range(power.shape[1]):
        points = inter_pos[ids, p]
        point_mask = np.all(np.isfinite(points), axis=2)
        has = point_mask.any(axis=1)
        first = np.argmax(point_mask, axis=1)
        point = points[np.arange(len(ids)), first]
        dvec = point - tx[None, :]
        norm = np.linalg.norm(dvec, axis=1)
        use = has & np.isfinite(norm) & (norm > 0)
        path_dirs[use, p] = dvec[use] / norm[use, None]

    for local, r in enumerate(ids):
        m = valid[r]
        if not np.any(m):
            response[start + local] = 0
            continue

        u = path_dirs[local, m, 0]
        v = path_dirs[local, m, 1]
        amp = 10 ** (power[r, m] / 20.0)
        ph = phase[r, m]
        tau = delay[r, m]

        # Finite masked path sum for every subcarrier and beam.
        du = u[:, None] - beam_u[None, :]
        dv = v[:, None] - beam_v[None, :]
        ax = np.exp(1j * k * du[:, :, None] * x[None, None, :]).sum(axis=2)
        ay = np.exp(1j * k * dv[:, :, None] * y[None, None, :]).sum(axis=2)
        af = (ax * ay) / (NX * NY)

        coeff = amp[:, None] * np.exp(
            1j * (ph[:, None] - 2 * np.pi * tau[:, None] * freq[None, :])
        )
        h = np.einsum("pf,pb->fb", coeff, af, optimize=True)
        pwr = np.abs(h) ** 2
        pwr[~np.isfinite(pwr)] = 0
        response[start + local] = pwr.astype(np.float32)

    if end % 1000 == 0 or end == len(receiver_ids):
        print(f"Processed {end}/{len(receiver_ids)}")

response.flush()

print("\nSaved:", out_path)
print("Response shape:", response.shape)
print("Finite fraction:", float(np.isfinite(response).mean()))
print("NaN count:", int(np.isnan(response).sum()))
print("Infinite count:", int(np.isinf(response).sum()))
print("All finite:", bool(np.isfinite(response).all()))
