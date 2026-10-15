from pathlib import Path
import json
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

ROOT = Path.cwd()
OUT = ROOT / "results" / "experiment_32x32" / "real_channel"
MODEL_DIR = ROOT / "models" / "experiment_32x32"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

response = np.load(
    OUT / "beam_power_response_finite.npy", mmap_mode="r"
)
labels = np.load(OUT / "beam_labels_finite.npy")

# Select a reproducible subset of receiver locations.
rng = np.random.default_rng(42)
n_receivers = len(labels)
selected = np.sort(
    rng.choice(n_receivers, size=min(12000, n_receivers), replace=False)
)

print("Selected receivers:", len(selected))

# Keep the 64 subcarriers for each selected receiver.
raw = np.asarray(response[selected], dtype=np.float32)
y_raw = labels[selected].astype(np.int64)

# Convert powers to dB and normalize using training data only.
X = 10 * np.log10(np.maximum(raw, 1e-20))
X = X.reshape(-1, X.shape[-1]).astype(np.float32)
y = y_raw.reshape(-1)

del raw

# Split by receiver to avoid leakage between subcarriers from one receiver.
receiver_group = np.repeat(np.arange(len(selected)), 64)
train_receivers, test_receivers = train_test_split(
    np.arange(len(selected)), test_size=0.2, random_state=42
)
train_mask = np.isin(receiver_group, train_receivers)
test_mask = np.isin(receiver_group, test_receivers)

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
std[std < 1e-6] = 1
X_train = np.nan_to_num((X_train - mean) / std).astype(np.float32)
X_test = np.nan_to_num((X_test - mean) / std).astype(np.float32)

classes = np.unique(y)
mapping = {int(v): i for i, v in enumerate(classes)}
y_train = np.array([mapping[int(v)] for v in y_train], dtype=np.int64)
y_test_encoded = np.array([mapping[int(v)] for v in y_test], dtype=np.int64)

train_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
    batch_size=1024, shuffle=True, num_workers=0
)

class FastBeamTransformer(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.projection = nn.Linear(64, 32)
        layer = nn.TransformerEncoderLayer(
            d_model=32, nhead=4, dim_feedforward=64,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(32, n_classes)

    def forward(self, x):
        # A single token per sample: fast preliminary Transformer classifier.
        x = self.projection(x).unsqueeze(1)
        return self.head(self.encoder(x)[:, 0])

model = FastBeamTransformer(len(classes)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

print("Training samples:", len(X_train))
print("Test samples:", len(X_test))
print("Classes:", len(classes))
print("Starting 5-epoch prototype...")

start = time.perf_counter()

for epoch in range(5):
    model.train()
    total = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(xb)

    print(f"Epoch {epoch + 1}/5, loss={total / len(X_train):.4f}")

model.eval()
pred_encoded = []
with torch.no_grad():
    for start_idx in range(0, len(X_test), 2048):
        xb = torch.from_numpy(X_test[start_idx:start_idx + 2048]).to(device)
        pred_encoded.extend(model(xb).argmax(dim=1).cpu().numpy())

pred_encoded = np.asarray(pred_encoded)
pred = classes[pred_encoded]
y_true = classes[y_test_encoded]

acc = accuracy_score(y_true, pred)
pm, rm, fm, _ = precision_recall_fscore_support(
    y_true, pred, average="macro", zero_division=0
)
pw, rw, fw, _ = precision_recall_fscore_support(
    y_true, pred, average="weighted", zero_division=0
)

elapsed = time.perf_counter() - start

report = {
    "experiment": "32x32 Transformer quick prototype",
    "receiver_subset": len(selected),
    "test_receivers": len(test_receivers),
    "test_subcarrier_samples": len(y_test),
    "epochs": 5,
    "accuracy": float(acc),
    "macro_precision": float(pm),
    "macro_recall": float(rm),
    "macro_f1": float(fm),
    "weighted_precision": float(pw),
    "weighted_recall": float(rw),
    "weighted_f1": float(fw),
    "training_and_evaluation_seconds": float(elapsed),
    "predicted_unique_beams": int(np.unique(pred).size),
    "note": "Prototype subset; labels depend on unvalidated reconstructed channel assumptions."
}

torch.save(
    {
        "state_dict": model.state_dict(),
        "classes": classes.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist()
    },
    MODEL_DIR / "transformer_32x32_quick.pt"
)

(OUT / "transformer_32x32_quick_report.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)

print("\n=== QUICK TRANSFORMER RESULTS ===")
print(f"Accuracy: {acc:.4f}")
print(f"Macro precision: {pm:.4f}")
print(f"Macro recall: {rm:.4f}")
print(f"Macro F1: {fm:.4f}")
print(f"Weighted F1: {fw:.4f}")
print("Unique predicted beams:", np.unique(pred).size)
print(f"Elapsed time: {elapsed:.1f} seconds")
print("Saved report:", OUT / "transformer_32x32_quick_report.json")
