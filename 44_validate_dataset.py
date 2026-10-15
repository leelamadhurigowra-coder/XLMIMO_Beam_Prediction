import os
import json
import numpy as np

base = "results/experiment_32x32/real_channel"
response_file = os.path.join(base, "beam_power_response_finite.npy")
label_file = os.path.join(base, "beam_labels_finite.npy")
report_file = os.path.join(base, "transformer_32x32_report.json")
model_file = "models/experiment_32x32/transformer_32x32.pt"

print("=" * 60)
print("XL-MIMO MODEL VALIDATION")
print("=" * 60)

for path in [response_file, label_file, report_file, model_file]:
    print(f"\n{path}")
    print("EXISTS:", os.path.exists(path))
    if os.path.exists(path):
        print("SIZE:", round(os.path.getsize(path) / (1024 * 1024), 2), "MB")

if not os.path.exists(response_file) or not os.path.exists(label_file):
    print("\nFAILED: Response or label file is missing.")
    raise SystemExit(1)

print("\n--- CHANNEL RESPONSE CHECK ---")
H = np.load(response_file, mmap_mode="r")
print("Shape:", H.shape)
print("Data type:", H.dtype)

finite_count = 0
total_count = 0
minimum = float("inf")
maximum = float("-inf")

for start in range(0, H.shape[0], 256):
    block = np.asarray(H[start:start + 256])
    finite = np.isfinite(block)
    finite_count += int(finite.sum())
    total_count += block.size
    if finite.any():
        minimum = min(minimum, float(block[finite].min()))
        maximum = max(maximum, float(block[finite].max()))

print("Finite values:", finite_count, "/", total_count)
print("Finite percentage:", round(100 * finite_count / total_count, 6), "%")
print("Minimum finite value:", minimum)
print("Maximum finite value:", maximum)

print("\n--- LABEL CHECK ---")
y = np.load(label_file, mmap_mode="r")
print("Shape:", y.shape)
print("Data type:", y.dtype)

if y.ndim != 2:
    print("WARNING: Expected a 2D label array.")
else:
    print("Subcarriers:", y.shape[1])
    labels = np.unique(y)
    print("Unique beam labels:", len(labels))
    print("Smallest label:", labels.min())
    print("Largest label:", labels.max())
    print("All labels in range 0-63:", bool(np.all((labels >= 0) & (labels < 64))))

print("\n--- TRAINING REPORT CHECK ---")
if os.path.exists(report_file):
    with open(report_file, "r", encoding="utf-8") as f:
        report = json.load(f)
    for key in [
        "accuracy", "top5_accuracy", "macro_f1",
        "weighted_f1", "test_receivers", "test_samples",
        "training_time_seconds", "predicted_unique_beams"
    ]:
        print(f"{key}: {report.get(key, 'NOT FOUND')}")
else:
    print("Training report missing.")

print("\n--- INITIAL VALIDATION SUMMARY ---")
print("Response and label receiver counts match:",
      H.shape[0] == y.shape[0])
print("Response subcarrier count matches labels:",
      H.shape[1] == y.shape[1] if H.ndim == 3 and y.ndim == 2 else False)
print("All response values finite:", finite_count == total_count)
print("NOTE: These checks do not yet prove there is no data leakage.")
