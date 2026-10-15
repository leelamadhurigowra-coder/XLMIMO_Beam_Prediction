from pathlib import Path
import numpy as np
import json

root = Path.cwd()
out = root / "results" / "experiment_32x32" / "real_channel"

response_path = out / "beam_power_response_finite.npy"
response = np.load(response_path, mmap_mode="r")

if response.ndim != 3:
    raise ValueError(f"Unexpected response shape: {response.shape}")

if not np.isfinite(response).all():
    raise ValueError("Response tensor contains NaN or infinity.")

n_receivers, n_subcarriers, n_beams = response.shape
labels = np.empty((n_receivers, n_subcarriers), dtype=np.int16)

print("Generating labels from finite beam responses...")

for start in range(0, n_receivers, 256):
    end = min(start + 256, n_receivers)
    block = np.asarray(response[start:end])
    labels[start:end] = np.argmax(block, axis=2)

    if end % 10000 == 0 or end == n_receivers:
        print(f"Processed {end}/{n_receivers}")

np.save(out / "beam_labels_finite.npy", labels)

unique, counts = np.unique(labels, return_counts=True)
report = {
    "response_shape": list(response.shape),
    "label_shape": list(labels.shape),
    "number_of_beams": int(n_beams),
    "unique_labels": int(len(unique)),
    "label_distribution": {
        str(int(k)): int(v) for k, v in zip(unique, counts)
    },
    "label_source": "argmax of regenerated finite beam-power response",
    "warning": (
        "Labels depend on provisional channel geometry and phase assumptions; "
        "numerical validity does not establish physical validity."
    )
}

(out / "label_generation_report.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)

print("\nLABEL GENERATION COMPLETE")
print("Label shape:", labels.shape)
print("Unique beams used:", len(unique))
print("Beam IDs:", unique.tolist())
print("Saved:", out / "beam_labels_finite.npy")
