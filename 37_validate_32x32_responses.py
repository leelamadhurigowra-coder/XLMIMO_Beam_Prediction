from pathlib import Path
import numpy as np
import json

root = Path.cwd()
out = root / "results" / "experiment_32x32" / "real_channel"

response_path = out / "beam_power_response.npy"
labels_path = out / "beam_labels.npy"
indices_path = out / "receiver_indices.npy"

print("Loading generated 32x32 responses...")
responses = np.load(response_path, mmap_mode="r")
labels = np.load(labels_path)
receiver_ids = np.load(indices_path)

print("\n=== SHAPE CHECK ===")
print("Responses:", responses.shape)
print("Labels:", labels.shape)
print("Receiver IDs:", receiver_ids.shape)

print("\n=== NUMERICAL CHECK ===")
print("Finite response values:", np.isfinite(responses).mean() * 100, "%")
print("NaN count:", int(np.isnan(responses).sum()))
print("Infinite count:", int(np.isinf(responses).sum()))
print("Zero response fraction:", float(np.mean(responses == 0)))
print("Negative response count:", int(np.sum(responses < 0)))

print("\n=== LABEL CHECK ===")
print("Label minimum:", int(labels.min()))
print("Label maximum:", int(labels.max()))
print("Unique labels:", int(np.unique(labels).size))

print("\n=== BEAM RESPONSE CHECK ===")
sample = np.asarray(responses[:100])
print("Sample response minimum:", float(np.min(sample)))
print("Sample response maximum:", float(np.max(sample)))
print("Sample response mean:", float(np.mean(sample)))

best = np.max(sample, axis=2)
second = np.partition(sample, -2, axis=2)[:, :, -2]
gap_db = 10 * np.log10(
    np.maximum(best, 1e-30) / np.maximum(second, 1e-30)
)

print("Mean best-vs-second beam gap (dB):", float(np.mean(gap_db)))

print("\nVALIDATION COMPLETE")
