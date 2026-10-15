from pathlib import Path
import json
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    top_k_accuracy_score
)

ROOT = Path.cwd()
OUT = ROOT / "results" / "experiment_32x32" / "real_channel"
MODEL_DIR = ROOT / "models" / "experiment_32x32"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RESPONSE_PATH = OUT / "beam_power_response_finite.npy"
LABEL_PATH = OUT / "beam_labels_finite.npy"

if not RESPONSE_PATH.exists() or not LABEL_PATH.exists():
    raise FileNotFoundError(
        "Run scripts 39 and 41 first to generate finite responses and labels."
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Load responses and labels.
# Each receiver has 64 subcarriers and 64 beam-power values per subcarrier.
responses = np.load(RESPONSE_PATH, mmap_mode="r")
labels_all = np.load(LABEL_PATH)

if responses.shape[:2] != labels_all.shape:
    raise ValueError(
        f"Response/label mismatch: {responses.shape} vs {labels_all.shape}"
    )
if not np.isfinite(responses).all():
    raise ValueError("Response tensor contains NaN or infinity.")

# Scale beam powers safely and convert to dB.
# Input sequence: 64 subcarriers x 64 candidate beams.
X = 10 * np.log10(np.maximum(np.asarray(responses), 1e-20))
X = np.asarray(X, dtype=np.float32)

# A label is the strongest-response beam at each subcarrier.
# Train one prediction per receiver/subcarrier pair.
n_receivers, n_subcarriers, n_beams = X.shape
X = X.reshape(-1, n_beams)
y = labels_all.reshape(-1).astype(np.int64)

finite = np.isfinite(X).all(axis=1)
X = X[finite]
y = y[finite]

classes = np.unique(y)
class_to_index = {int(c): i for i, c in enumerate(classes)}
y_encoded = np.array([class_to_index[int(v)] for v in y], dtype=np.int64)

print("Samples:", len(y_encoded))
print("Input features:", X.shape[1])
print("Beam classes:", len(classes))

# Split samples by receiver, preventing subcarriers from one receiver
# appearing in both train and test.
receiver_group = np.repeat(np.arange(n_receivers), n_subcarriers)[finite]
unique_receivers = np.unique(receiver_group)

train_receivers, test_receivers = train_test_split(
    unique_receivers, test_size=0.2, random_state=42
)
train_mask = np.isin(receiver_group, train_receivers)
test_mask = np.isin(receiver_group, test_receivers)

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y_encoded[train_mask], y_encoded[test_mask]

# Normalize using training data only.
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
std[std < 1e-6] = 1.0

X_train = ((X_train - mean) / std).astype(np.float32)
X_test = ((X_test - mean) / std).astype(np.float32)

train_ds = TensorDataset(
    torch.from_numpy(X_train), torch.from_numpy(y_train)
)
train_loader = DataLoader(
    train_ds, batch_size=512, shuffle=True, num_workers=0
)

class BeamTransformer(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, 64)
        self.position = nn.Parameter(torch.zeros(1, 1, 64))
        layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128,
            dropout=0.1, activation="gelu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(64)
        self.head = nn.Linear(64, n_classes)

    def forward(self, x):
        # A single token per sample: this is a Transformer classifier,
        # not yet a cross-subcarrier sequence model.
        x = self.input_projection(x).unsqueeze(1)
        x = x + self.position
        x = self.encoder(x)
        return self.head(self.norm(x[:, 0]))

model = BeamTransformer(X_train.shape[1], len(classes)).to(device)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=5e-4, weight_decay=1e-4
)
criterion = nn.CrossEntropyLoss()

print("\nTraining Transformer...")
start_time = time.perf_counter()
epochs = 15

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * len(xb)

    print(
        f"Epoch {epoch + 1:02d}/{epochs} "
        f"loss={total_loss / len(train_ds):.4f}"
    )

training_seconds = time.perf_counter() - start_time

# Evaluate on held-out receiver locations.
model.eval()
all_probs = []
batch_size = 1024

with torch.no_grad():
    for start in range(0, len(X_test), batch_size):
        xb = torch.from_numpy(
            X_test[start:start + batch_size]
        ).to(device)
        all_probs.append(torch.softmax(model(xb), dim=1).cpu().numpy())

probs = np.concatenate(all_probs, axis=0)
pred_encoded = probs.argmax(axis=1)
pred = classes[pred_encoded]
y_true = classes[y_test]

acc = accuracy_score(y_true, pred)
p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
    y_true, pred, average="macro", zero_division=0
)
p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
    y_true, pred, average="weighted", zero_division=0
)

top5 = float(np.mean([
    y_true[i] in classes[np.argsort(probs[i])[-min(5, len(classes)):]]
    for i in range(len(y_true))
]))

report = {
    "model": "32x32 response-input Transformer classifier",
    "device": str(device),
    "antenna_elements": 1024,
    "subcarriers": 64,
    "candidate_beams": 64,
    "test_samples": int(len(y_true)),
    "test_receivers": int(len(test_receivers)),
    "accuracy": float(acc),
    "macro_precision": float(p_macro),
    "macro_recall": float(r_macro),
    "macro_f1": float(f1_macro),
    "weighted_precision": float(p_weighted),
    "weighted_recall": float(r_weighted),
    "weighted_f1": float(f1_weighted),
    "top5_accuracy": float(top5),
    "training_time_seconds": float(training_seconds),
    "predicted_unique_beams": int(np.unique(pred).size),
    "classification_report": classification_report(
        y_true, pred, zero_division=0, output_dict=True
    ),
    "limitations": [
        "This uses modeled beam responses, not measured hardware data.",
        "Channel geometry and phase assumptions remain to be validated.",
        "One token is used per subcarrier sample; temporal/cross-subcarrier attention is not implemented."
    ]
}

torch.save({
    "model_state_dict": model.state_dict(),
    "input_dim": int(X_train.shape[1]),
    "classes": classes.tolist(),
    "mean": mean.tolist(),
    "std": std.tolist()
}, MODEL_DIR / "transformer_32x32.pt")

(OUT / "transformer_32x32_report.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)

print("\n=== 32x32 TRANSFORMER RESULTS ===")
print(f"Accuracy:           {acc:.4f}")
print(f"Macro precision:    {p_macro:.4f}")
print(f"Macro recall:       {r_macro:.4f}")
print(f"Macro F1:           {f1_macro:.4f}")
print(f"Weighted precision: {p_weighted:.4f}")
print(f"Weighted recall:    {r_weighted:.4f}")
print(f"Weighted F1:        {f1_weighted:.4f}")
print(f"Top-5 accuracy:     {top5:.4f}")
print("Predicted unique beams:", np.unique(pred).size)
print(f"Training time:      {training_seconds:.2f}s")
print("Saved model:", MODEL_DIR / "transformer_32x32.pt")
print("Saved report:", OUT / "transformer_32x32_report.json")
