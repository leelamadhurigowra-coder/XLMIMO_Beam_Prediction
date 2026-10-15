
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_DIR / "results"
MODELS_DIR = PROJECT_DIR / "models"

X_FILE = RESULTS_DIR / "frequency_dataset.npy"
Y_FILE = RESULTS_DIR / "beam_labels.npy"

MODEL_FILE = MODELS_DIR / "transformer_beam_predictor.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\n========== TRANSFORMER DIAGNOSTICS ==========\n")
print("Device:", DEVICE)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading dataset...")

X = np.load(
    X_FILE,
    mmap_mode="r"
)

Y = np.load(Y_FILE)

label_encoder = LabelEncoder()

Y_encoded = label_encoder.fit_transform(
    Y.reshape(-1)
).reshape(Y.shape)

num_classes = len(label_encoder.classes_)

print("Input shape:", X.shape)
print("Label shape:", Y.shape)
print("Number of classes:", num_classes)


# ============================================================
# 3. RECREATE THE SAME TEST SPLIT
# ============================================================

all_indices = np.arange(len(Y_encoded))

train_indices, test_indices = train_test_split(
    all_indices,
    test_size=0.20,
    random_state=42
)

train_indices, val_indices = train_test_split(
    train_indices,
    test_size=0.125,
    random_state=42
)

print("\n========== DATA SPLIT ==========\n")

print("Training receivers:", len(train_indices))
print("Validation receivers:", len(val_indices))
print("Testing receivers:", len(test_indices))


# ============================================================
# 4. LOAD SAVED MODEL AND NORMALIZATION VALUES
# ============================================================

print("\nLoading trained Transformer...")

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE,
    weights_only=False
)

feature_mean = checkpoint["feature_mean"]
feature_std = checkpoint["feature_std"]

print("Saved model loaded successfully.")


# ============================================================
# 5. RECREATE TRANSFORMER ARCHITECTURE
# ============================================================

class BeamTransformer(nn.Module):

    def __init__(
        self,
        input_features=19,
        d_model=64,
        num_heads=4,
        num_layers=2,
        num_classes=54,
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
            torch.zeros(1, 64, d_model)
        )

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        # Two Transformer encoder layers
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Classification layer
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):

        # Input shape: (batch, 64, 19)

        x = self.input_projection(x)

        x = x + self.position_embedding

        x = self.transformer(x)

        # Output shape: (batch, 64, 54)
        return self.classifier(x)


model = BeamTransformer(
    input_features=19,
    d_model=64,
    num_heads=4,
    num_layers=2,
    num_classes=num_classes
).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Transformer architecture restored successfully.")


# ============================================================
# 6. CHECK ACTUAL BEAM DISTRIBUTION
# ============================================================

print("\n========== ACTUAL BEAM DISTRIBUTION ==========\n")

actual_counts = np.bincount(
    Y_encoded[test_indices].reshape(-1),
    minlength=num_classes
)

actual_total = actual_counts.sum()

for class_id in np.argsort(actual_counts)[::-1]:

    count = actual_counts[class_id]

    percentage = (
        100 * count / actual_total
    )

    original_beam = label_encoder.classes_[class_id]

    print(
        f"Beam {original_beam}: "
        f"{count} labels ({percentage:.2f}%)"
    )


# ============================================================
# 7. GENERATE TEST PREDICTIONS
# ============================================================

print("\n========== GENERATING PREDICTIONS ==========\n")

all_predictions = []
all_targets = []

BATCH_SIZE = 64

with torch.no_grad():

    for start in range(
        0,
        len(test_indices),
        BATCH_SIZE
    ):

        batch_indices = test_indices[
            start:start + BATCH_SIZE
        ]

        # Load test inputs
        batch_x = np.asarray(
            X[batch_indices],
            dtype=np.float32
        )

        # Load actual labels
        batch_y = Y_encoded[batch_indices]

        # Normalize using training-set statistics
        batch_x = (
            batch_x - feature_mean
        ) / feature_std

        # IMPORTANT:
        # Convert normalized values to float32.
        # This prevents the Double/Float mismatch.
        batch_x = batch_x.astype(
            np.float32
        )

        # Convert NumPy array to PyTorch tensor
        batch_x = torch.from_numpy(
            batch_x
        ).to(DEVICE)

        # Predict beam classes
        outputs = model(batch_x)

        predictions = outputs.argmax(
            dim=-1
        ).cpu().numpy()

        all_predictions.append(
            predictions.reshape(-1)
        )

        all_targets.append(
            batch_y.reshape(-1)
        )

        processed = min(
            start + BATCH_SIZE,
            len(test_indices)
        )

        print(
            f"Processed: {processed}/{len(test_indices)}",
            end="\r"
        )


# Combine predictions from all batches
y_pred = np.concatenate(
    all_predictions
)

y_true = np.concatenate(
    all_targets
)

print("\n\nPrediction generation complete.")


# ============================================================
# 8. CHECK PREDICTED BEAM DISTRIBUTION
# ============================================================

print("\n========== PREDICTED BEAM DISTRIBUTION ==========\n")

predicted_counts = np.bincount(
    y_pred,
    minlength=num_classes
)

for class_id in np.argsort(predicted_counts)[::-1]:

    count = predicted_counts[class_id]

    percentage = (
        100 * count / len(y_pred)
    )

    original_beam = label_encoder.classes_[class_id]

    print(
        f"Beam {original_beam}: "
        f"{count} predictions ({percentage:.2f}%)"
    )


# ============================================================
# 9. CHECK NUMBER OF DISTINCT PREDICTED CLASSES
# ============================================================

actual_classes = np.unique(y_true)

predicted_classes = np.unique(y_pred)

print("\n========== DIAGNOSTIC SUMMARY ==========\n")

print(
    "Distinct classes in actual test labels:",
    len(actual_classes)
)

print(
    "Distinct classes predicted by model:",
    len(predicted_classes)
)

most_predicted_class = np.argmax(
    predicted_counts
)

print(
    "Most frequently predicted beam:",
    label_encoder.classes_[most_predicted_class]
)

print(
    "Most frequently predicted beam percentage:",
    round(
        100 * predicted_counts.max() / len(y_pred),
        2
    ),
    "%"
)

accuracy = np.mean(
    y_true == y_pred
)

print(
    "Recalculated test accuracy:",
    round(float(accuracy), 6)
)


# ============================================================
# 10. SAVE DIAGNOSTIC REPORT
# ============================================================

diagnostic_file = (
    RESULTS_DIR / "transformer_diagnostics.npz"
)

np.savez(
    diagnostic_file,
    actual_counts=actual_counts,
    predicted_counts=predicted_counts,
    actual_classes=actual_classes,
    predicted_classes=predicted_classes,
    accuracy=accuracy
)

print("\nDiagnostic report saved at:")
print(diagnostic_file)

print("\n========== DIAGNOSTICS COMPLETE ==========")