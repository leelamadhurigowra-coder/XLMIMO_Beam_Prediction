from pathlib import Path
import json
import time
import joblib
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ROOT = Path.cwd()
OUT = ROOT / "results" / "experiment_32x32"
MODEL = ROOT / "models" / "experiment_32x32" / "random_forest_los_proxy.pkl"

# Explicit evaluation assumptions (not measured site conditions)
ASSUMED_SNR_DB = 10.0
OUTAGE_THRESHOLD_DB = 0.0
CHUNK = 256

def read_mat(filename, key):
    return loadmat(ROOT / "data" / filename)[key]

print("Loading 32x32 LOS-proxy experiment...")

X = np.load(OUT / "frequency_dataset_los_proxy.npy", mmap_mode="r")
labels = np.load(OUT / "los_proxy_beam_labels.npy")
valid = np.load(OUT / "valid_receiver_mask.npy")
directions = np.load(OUT / "beam_directions_xyz.npy").astype(np.float64)
array_pos = np.load(OUT / "array_positions_m.npy").astype(np.float64)

tx = read_mat("tx_pos_t001_tx000_r000.mat", "tx_pos").reshape(-1, 3)[0]
rx = read_mat("rx_pos_t001_tx000_r000.mat", "rx_pos").astype(np.float64)

if not MODEL.exists():
    raise FileNotFoundError(f"Model not found: {MODEL}")

# Match the features used by the baseline training script.
mask = valid & (labels >= 0)
receiver_ids = np.flatnonzero(mask)
X_mean = np.asarray(X[mask].mean(axis=1), dtype=np.float32)
y = labels[mask].astype(np.int64)

finite = np.isfinite(X_mean).all(axis=1)
receiver_ids = receiver_ids[finite]
X_mean = X_mean[finite]
y = y[finite]

_, test_idx = train_test_split(
    np.arange(len(y)), test_size=0.2, random_state=42
)

X_test = X_mean[test_idx]
y_test = y[test_idx]
test_receiver_ids = receiver_ids[test_idx]

model = joblib.load(MODEL)

start = time.perf_counter()
pred = model.predict(X_test)
inference_seconds = time.perf_counter() - start

proba = model.predict_proba(X_test)
model_classes = model.classes_
top_order = np.argsort(proba, axis=1)[:, ::-1]
top5_class_ids = model_classes[top_order[:, :min(5, len(model_classes))]]

# LOS direction from the transmitter to each receiver.
vec = rx[test_receiver_ids] - tx[None, :]
distance = np.linalg.norm(vec, axis=1)
unit = vec / np.maximum(distance[:, None], 1e-12)

# Assumes the array's local x/y axes align with the dataset x/y axes.
u = unit[:, 0]
v = unit[:, 1]

# Direction cosines of the defined codebook beams.
ub = directions[:, 0]
vb = directions[:, 1]

# Recover regular array-grid coordinates from the saved array geometry.
C = 299792458.0
FC = 3.5e9
wavelength = C / FC
k = 2 * np.pi / wavelength

xcoords = np.unique(np.round(array_pos[:, 0], 7))
ycoords = np.unique(np.round(array_pos[:, 1], 7))

# Relative power response of each beam for each LOS direction.
# Separable UPA array factor avoids allocating a huge 3-D tensor.
n_test = len(test_receiver_ids)
beam_power = np.empty((n_test, len(directions)), dtype=np.float32)

for start_idx in range(0, n_test, CHUNK):
    end_idx = min(start_idx + CHUNK, n_test)
    uc = u[start_idx:end_idx, None] - ub[None, :]
    vc = v[start_idx:end_idx, None] - vb[None, :]

    sx = np.exp(1j * k * uc[:, :, None] * xcoords[None, None, :]).sum(axis=2)
    sy = np.exp(1j * k * vc[:, :, None] * ycoords[None, None, :]).sum(axis=2)

    # Normalized power gain relative to ideal coherent array response.
    beam_power[start_idx:end_idx] = (
        np.abs(sx * sy) ** 2 / (len(xcoords) * len(ycoords)) ** 2
    )

oracle_idx = np.argmax(beam_power, axis=1)
oracle_power = beam_power[np.arange(n_test), oracle_idx]
pred_in_range = (pred >= 0) & (pred < len(directions))

pred_power = np.zeros(n_test, dtype=np.float64)
pred_power[pred_in_range] = beam_power[
    np.arange(n_test)[pred_in_range], pred[pred_in_range]
]

gain_ratio = pred_power / np.maximum(oracle_power, 1e-15)
gain_db = 10 * np.log10(np.maximum(gain_ratio, 1e-15))

# Top-5 selection: best physical beam response among model's five candidates.
top5_power = np.zeros(n_test, dtype=np.float64)
for i in range(n_test):
    candidates = top5_class_ids[i]
    candidates = candidates[(candidates >= 0) & (candidates < len(directions))]
    if len(candidates):
        top5_power[i] = np.max(beam_power[i, candidates])

top5_ratio = top5_power / np.maximum(oracle_power, 1e-15)
top5_gain_db = 10 * np.log10(np.maximum(top5_ratio, 1e-15))

# Idealized link calculation using the stated SNR assumption.
snr_linear = 10 ** (ASSUMED_SNR_DB / 10)
selected_sinr = snr_linear * pred_power
oracle_sinr = snr_linear * oracle_power

selected_rate = np.log2(1 + selected_sinr)
oracle_rate = np.log2(1 + oracle_sinr)

selected_sinr_db = 10 * np.log10(np.maximum(selected_sinr, 1e-15))
oracle_sinr_db = 10 * np.log10(np.maximum(oracle_sinr, 1e-15))

report = {
    "experiment": "32x32 LOS-proxy beamforming evaluation",
    "test_receivers": int(n_test),
    "classification_accuracy_vs_proxy_labels": float(accuracy_score(y_test, pred)),
    "assumptions": {
        "array_axis_alignment": "Array local x/y assumed aligned with dataset x/y",
        "snr_db": ASSUMED_SNR_DB,
        "outage_threshold_db": OUTAGE_THRESHOLD_DB,
        "channel": "LOS geometric response only",
        "rate_formula": "log2(1 + assumed_SNR_linear * normalized_beam_power)",
        "limitations": [
            "Not a validated multipath channel simulation",
            "No interference, blockage, hardware loss, or field measurements",
            "SNR is an assumed evaluation setting, not measured data"
        ]
    },
    "mean_predicted_beam_gain_relative_to_codebook_oracle_db": float(np.mean(gain_db)),
    "median_predicted_beam_gain_relative_to_codebook_oracle_db": float(np.median(gain_db)),
    "mean_top5_best_candidate_gain_relative_to_oracle_db": float(np.mean(top5_gain_db)),
    "mean_predicted_beam_normalized_power": float(np.mean(pred_power)),
    "mean_oracle_normalized_power": float(np.mean(oracle_power)),
    "mean_spectral_efficiency_predicted_beam_bps_hz": float(np.mean(selected_rate)),
    "mean_spectral_efficiency_codebook_oracle_bps_hz": float(np.mean(oracle_rate)),
    "mean_spectral_efficiency_difference_bps_hz": float(np.mean(selected_rate - oracle_rate)),
    "predicted_beam_outage_probability_percent": float(np.mean(selected_sinr_db < OUTAGE_THRESHOLD_DB) * 100),
    "oracle_beam_outage_probability_percent": float(np.mean(oracle_sinr_db < OUTAGE_THRESHOLD_DB) * 100),
    "inference_latency_ms_per_receiver": float(inference_seconds * 1000 / n_test),
    "codebook_oracle_definition": "Best response among the 64 defined codebook directions for the LOS geometry"
}

report_path = OUT / "los_domain_evaluation.json"
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\n=== LOS DOMAIN EVALUATION ===")
print("Test receivers:", n_test)
print(f"Classification accuracy vs proxy labels: {report['classification_accuracy_vs_proxy_labels']:.4f}")
print(f"Mean gain vs codebook oracle (dB): {report['mean_predicted_beam_gain_relative_to_codebook_oracle_db']:.4f}")
print(f"Mean top-5 candidate gain vs oracle (dB): {report['mean_top5_best_candidate_gain_relative_to_oracle_db']:.4f}")
print(f"Predicted-beam spectral efficiency: {report['mean_spectral_efficiency_predicted_beam_bps_hz']:.4f} bits/s/Hz")
print(f"Oracle spectral efficiency: {report['mean_spectral_efficiency_codebook_oracle_bps_hz']:.4f} bits/s/Hz")
print(f"Predicted-beam outage: {report['predicted_beam_outage_probability_percent']:.2f}%")
print(f"Oracle outage: {report['oracle_beam_outage_probability_percent']:.2f}%")
print(f"Inference latency: {report['inference_latency_ms_per_receiver']:.4f} ms/sample")
print("\nSaved:", report_path)
print("These are LOS-only simulation estimates under explicit assumptions.")
