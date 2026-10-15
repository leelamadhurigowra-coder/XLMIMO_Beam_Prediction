from pathlib import Path
import torch
import numpy as np
import json

ROOT = Path.cwd()

model_path = ROOT / "models" / "transformer_beam_predictor.pth"
classes_path = ROOT / "models" / "transformer_label_classes.npy"
report_path = ROOT / "results" / "transformer_report.json"

print("\n=== TRANSFORMER MODEL INSPECTION ===")

if model_path.exists():
    print("\nModel found:", model_path)
    checkpoint = torch.load(
        model_path, map_location="cpu", weights_only=False
    )

    if isinstance(checkpoint, dict):
        print("Checkpoint keys:", list(checkpoint.keys()))
        state = checkpoint.get("model_state_dict", checkpoint)

        if isinstance(state, dict):
            print("\nParameter names and shapes:")
            for name, value in state.items():
                if hasattr(value, "shape"):
                    print(f"{name}: {tuple(value.shape)}")
    else:
        print("Saved object type:", type(checkpoint))
else:
    print("\nMODEL NOT FOUND:", model_path)

if classes_path.exists():
    classes = np.load(classes_path)
    print("\nSaved classes:", classes.shape)
    print(classes)

if report_path.exists():
    print("\nPrevious report:")
    print(report_path.read_text(encoding="utf-8")[:5000])

print("\n=== INSPECTION COMPLETE ===")
