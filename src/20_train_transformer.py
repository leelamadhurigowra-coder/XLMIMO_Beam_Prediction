
import json
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
    classification_report
)

# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_DIR / "results"
MODELS_DIR = PROJECT_DIR / "models"

MODELS_DIR.mkdir(exist_ok=True)

X_FILE = RESULTS_DIR / "frequency_dataset.npy"
Y_FILE = RESULTS_DIR / "beam_labels.npy"
INDEX_FILE = RESULTS_DIR / "beam_label_receiver_indices.npy"

MODEL_FILE = MODELS_DIR / "transformer_beam_predictor.pth"
REPORT_FILE = RESULTS_DIR / "transformer_report.json"

RANDOM_STATE = 42

NUM_SUBCARRIERS = 64
NUM_FEATURES = 19
NUM_BEAMS = 64

BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.0005

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\n========== TRANSFORMER TRAINING ==========\n")
print("Device:", DEVICE)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading dataset...")

# Memory mapping avoids loading the entire 1.65 GB
# feature dataset into RAM at once.
X = np.load(X_FILE, mmap_mode="r")

Y = np.load(Y_FILE)
receiver_indices = np.load(INDEX_FILE)

print("Input shape:", X.shape)
print("Label shape:", Y.shape)
print("Receiver indices:", receiver_indices.shape)

if X.shape != (
    len(Y),
    NUM_SUBCARRIERS,
    NUM_FEATURES
):
    raise ValueError("Input dimensions do not match expected shape.")

if Y.shape[1] != NUM_SUBCARRIERS:
    raise ValueError("Label subcarrier count is incorrect.")

if len(receiver_indices) != len(Y):
    raise ValueError("Receiver indices are not aligned with labels.")


# ============================================================
# 3. ENCODE BEAM LABELS
# ============================================================

label_encoder = LabelEncoder()

Y_encoded = label_encoder.fit_transform(
    Y.reshape(-1)
).reshape(Y.shape)

num_classes = len(label_encoder.classes_)

print("Number of beam classes:", num_classes)

if num_classes > NUM_BEAMS:
    raise ValueError("More beam classes than expected.")

np.save(
    MODELS_DIR / "transformer_label_classes.npy",
    label_encoder.classes_
)


# ============================================================
# 4. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

all_indices = np.arange(len(Y_encoded))

train_indices, test_indices = train_test_split(
    all_indices,
    test_size=0.20,
    random_state=RANDOM_STATE
)

train_indices, val_indices = train_test_split(
    train_indices,
    test_size=0.125,
    random_state=RANDOM_STATE
)

print("\nData split:")
print("Training receivers:", len(train_indices))
print("Validation receivers:", len(val_indices))
print("Testing receivers:", len(test_indices))


# ============================================================
# 5. FEATURE NORMALIZATION
# ============================================================

print("\nCalculating feature normalization...")

# Calculate mean and variance from training data only.
# Use batches to limit RAM consumption.

feature_sum = np.zeros(NUM_FEATURES, dtype=np.float64)
feature_sq_sum = np.zeros(NUM_FEATURES, dtype=np.float64)

total_values = 0

NORM_BATCH_SIZE = 256

for start in range(0, len(train_indices), NORM_BATCH_SIZE):

    batch_indices = train_indices[
        start:start + NORM_BATCH_SIZE
    ]

    batch = np.asarray(
        X[batch_indices],
        dtype=np.float64
    )

    feature_sum += batch.sum(axis=(0, 1))
    feature_sq_sum += (batch ** 2).sum(axis=(0, 1))

    total_values += batch.shape[0] * batch.shape[1]

    print(
        f"Normalization: "
        f"{min(start + NORM_BATCH_SIZE, len(train_indices))}"
        f"/{len(train_indices)}",
        end="\r"
    )

feature_mean = feature_sum / total_values

feature_variance = (
    feature_sq_sum / total_values
    - feature_mean ** 2
)

feature_std = np.sqrt(
    np.maximum(feature_variance, 1e-12)
)

feature_std[feature_std < 1e-8] = 1.0

print("\nNormalization complete.")


# ============================================================
# 6. PYTORCH DATASET
# ============================================================

class BeamDataset(Dataset):

    def __init__(self, X, Y, indices, mean, std):

        self.X = X
        self.Y = Y
        self.indices = indices

        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):

        receiver_index = self.indices[index]

        x = np.array(
            self.X[receiver_index],
            dtype=np.float32
        )

        y = np.array(
            self.Y[receiver_index],
            dtype=np.int64
        )

        x = (x - self.mean) / self.std

        return (
            torch.from_numpy(x),
            torch.from_numpy(y)
        )


train_dataset = BeamDataset(
    X, Y_encoded, train_indices,
    feature_mean, feature_std
)

val_dataset = BeamDataset(
    X, Y_encoded, val_indices,
    feature_mean, feature_std
)

test_dataset = BeamDataset(
    X, Y_encoded, test_indices,
    feature_mean, feature_std
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 7. TRANSFORMER MODEL
# ============================================================

class BeamTransformer(nn.Module):

    def __init__(
        self,
        input_features=19,
        d_model=64,
        num_heads=4,
        num_layers=2,
        num_classes=64,
        dropout=0.1
    ):

        super().__init__()

        # Convert 19 input features into 64-dimensional embeddings
        self.input_projection = nn.Linear(
            input_features,
            d_model
        )

        # Learnable positional embeddings
        self.position_embedding = nn.Parameter(
            torch.zeros(1, NUM_SUBCARRIERS, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):

        # x shape:
        # (batch, 64, 19)

        x = self.input_projection(x)

        x = x + self.position_embedding

        x = self.transformer(x)

        # Output:
        # (batch, 64, num_classes)

        return self.classifier(x)


model = BeamTransformer(
    input_features=NUM_FEATURES,
    d_model=64,
    num_heads=4,
    num_layers=2,
    num_classes=num_classes
).to(DEVICE)

print("\nModel architecture:")
print(model)


# ============================================================
# 8. LOSS FUNCTION AND OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0001
)


# ============================================================
# 9. TRAINING LOOP
# ============================================================

best_val_loss = float("inf")
patience = 4
epochs_without_improvement = 0

history = []

print("\n========== STARTING TRAINING ==========\n")

for epoch in range(EPOCHS):

    model.train()

    train_loss_total = 0.0
    train_batches = 0

    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(batch_x)

        loss = criterion(
            outputs.reshape(-1, num_classes),
            batch_y.reshape(-1)
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        train_loss_total += loss.item()
        train_batches += 1

    avg_train_loss = (
        train_loss_total / train_batches
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss_total = 0.0
    val_batches = 0

    correct = 0
    total = 0

    with torch.no_grad():

        for batch_x, batch_y in val_loader:

            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            outputs = model(batch_x)

            loss = criterion(
                outputs.reshape(-1, num_classes),
                batch_y.reshape(-1)
            )

            val_loss_total += loss.item()
            val_batches += 1

            predictions = outputs.argmax(dim=-1)

            correct += (
                predictions == batch_y
            ).sum().item()

            total += batch_y.numel()

    avg_val_loss = (
        val_loss_total / val_batches
    )

    val_accuracy = correct / total

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.4f}"
    )

    history.append({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "validation_loss": avg_val_loss,
        "validation_accuracy": val_accuracy
    })

    # Save best model
    if avg_val_loss < best_val_loss:

        best_val_loss = avg_val_loss

        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "num_classes": num_classes,
                "input_features": NUM_FEATURES,
                "num_subcarriers": NUM_SUBCARRIERS
            },
            MODEL_FILE
        )

        print("Best model saved!")

    else:

        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:

        print("\nEarly stopping triggered.")

        break


# ============================================================
# 10. LOAD BEST MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# 11. TEST EVALUATION
# ============================================================

print("\n========== TEST EVALUATION ==========\n")

all_predictions = []
all_targets = []
all_probabilities = []

with torch.no_grad():

    for batch_x, batch_y in test_loader:

        batch_x = batch_x.to(DEVICE)

        outputs = model(batch_x)

        probabilities = torch.softmax(
            outputs,
            dim=-1
        )

        predictions = outputs.argmax(dim=-1)

        all_predictions.append(
            predictions.cpu().numpy()
        )

        all_targets.append(
            batch_y.numpy()
        )

        all_probabilities.append(
            probabilities.cpu().numpy()
        )

y_pred = np.concatenate(
    all_predictions,
    axis=0
).reshape(-1)

y_true = np.concatenate(
    all_targets,
    axis=0
).reshape(-1)

y_prob = np.concatenate(
    all_probabilities,
    axis=0
).reshape(-1, num_classes)


# ============================================================
# 12. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision_weighted = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

recall_weighted = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

f1_weighted = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

f1_macro = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

top5_accuracy = top_k_accuracy_score(
    y_true,
    y_prob,
    k=min(5, num_classes),
    labels=np.arange(num_classes)
)

print("Test Accuracy:", accuracy)
print("Top-5 Accuracy:", top5_accuracy)
print("Weighted Precision:", precision_weighted)
print("Weighted Recall:", recall_weighted)
print("Weighted F1:", f1_weighted)
print("Macro F1:", f1_macro)

print("\nClassification report:")

print(
    classification_report(
        y_true,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 13. SAVE REPORT
# ============================================================

report = {
    "model": "Transformer Encoder",
    "test_accuracy": float(accuracy),
    "top5_accuracy": float(top5_accuracy),
    "weighted_precision": float(precision_weighted),
    "weighted_recall": float(recall_weighted),
    "weighted_f1": float(f1_weighted),
    "macro_f1": float(f1_macro),
    "num_test_receivers": int(len(test_indices)),
    "num_test_predictions": int(len(y_true)),
    "num_classes": int(num_classes),
    "num_subcarriers": int(NUM_SUBCARRIERS),
    "training_history": history
}

with open(REPORT_FILE, "w") as f:

    json.dump(
        report,
        f,
        indent=4
    )


# ============================================================
# 14. COMPLETION
# ============================================================

print("\n========== TRAINING COMPLETE ==========\n")

print("Model saved:", MODEL_FILE)
print("Report saved:", REPORT_FILE)

print("\nFinal test accuracy:", accuracy)
print("Final top-5 accuracy:", top5_accuracy)

print("\n========== SUCCESS ==========")