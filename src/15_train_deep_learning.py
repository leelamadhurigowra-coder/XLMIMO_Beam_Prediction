
import os
import json
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report
)
from sklearn.pipeline import Pipeline

# ============================================================
# 1. CREATE OUTPUT FOLDERS
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("=" * 60)
print("DEEP LEARNING - ARTIFICIAL NEURAL NETWORK")
print("=" * 60)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading dataset...")

X_all = np.load(
    "results/random_forest_input_features.npy"
)

Y_all = np.load(
    "results/beam_labels.npy"
)

receiver_indices = np.load(
    "results/beam_label_receiver_indices.npy"
)

print("Feature shape:", X_all.shape)
print("Label shape:", Y_all.shape)


# ============================================================
# 3. ALIGN FEATURES AND LABELS
# ============================================================

X = X_all[receiver_indices]

# Center subcarrier
y = Y_all[:, 32]

print("Aligned feature shape:", X.shape)
print("Target shape:", y.shape)


# ============================================================
# 4. REMOVE INVALID VALUES
# ============================================================

valid_mask = np.isfinite(X).all(axis=1)

X = X[valid_mask]
y = y[valid_mask]

print("Valid samples:", len(X))


# ============================================================
# 5. ENCODE BEAM LABELS
# ============================================================

# Convert beam IDs into consecutive class indices.
# This is useful when some beam IDs are absent.

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

print("Number of beam classes:", len(encoder.classes_))


# ============================================================
# 6. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 7. BUILD NEURAL NETWORK
# ============================================================

print("\nBuilding neural network...")

model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "neural_network",
        MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=0.001,
            batch_size=256,
            learning_rate_init=0.001,
            max_iter=100,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42,
            verbose=True
        )
    )
])


# ============================================================
# 8. TRAIN MODEL
# ============================================================

print("\nTraining neural network...")

model.fit(X_train, y_train)

print("\nTraining completed!")


# ============================================================
# 9. PREDICT
# ============================================================

print("\nGenerating predictions...")

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)


# ============================================================
# 10. EVALUATE
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

# Calculate top-5 accuracy
top5_correct = 0

for i in range(len(y_test)):

    top_indices = np.argsort(y_prob[i])[-5:]

    top_classes = top_indices

    if y_test[i] in top_classes:
        top5_correct += 1

top5_accuracy = top5_correct / len(y_test)


# ============================================================
# 11. DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("DEEP LEARNING RESULTS")
print("=" * 60)

print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy percentage: {accuracy * 100:.2f}%")

print(f"Top-5 Accuracy: {top5_accuracy:.4f}")
print(
    f"Top-5 Accuracy percentage: "
    f"{top5_accuracy * 100:.2f}%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(encoder.classes_)),
        target_names=[
            str(c) for c in encoder.classes_
        ],
        zero_division=0
    )
)


# ============================================================
# 12. SAVE MODEL AND LABEL ENCODER
# ============================================================

joblib.dump(
    model,
    "models/deep_learning_ann.pkl"
)

joblib.dump(
    encoder,
    "models/beam_label_encoder.pkl"
)

print("\nModel saved:")
print("models/deep_learning_ann.pkl")

print("Label encoder saved:")
print("models/beam_label_encoder.pkl")


# ============================================================
# 13. SAVE REPORT
# ============================================================

report = {
    "model": "Artificial Neural Network",
    "task": "Center-subcarrier beam prediction",
    "subcarrier_index": 32,
    "architecture": [128, 64],
    "activation": "ReLU",
    "optimizer": "Adam",
    "num_features": int(X.shape[1]),
    "num_samples": int(len(X)),
    "training_samples": int(len(X_train)),
    "testing_samples": int(len(X_test)),
    "num_beam_classes": int(len(encoder.classes_)),
    "accuracy": float(accuracy),
    "accuracy_percent": float(accuracy * 100),
    "top5_accuracy": float(top5_accuracy),
    "top5_accuracy_percent": float(top5_accuracy * 100)
}

with open(
    "results/deep_learning_report.json",
    "w"
) as f:
    json.dump(report, f, indent=4)

print("\nReport saved:")
print("results/deep_learning_report.json")

print("\nDEEP LEARNING TRAINING FINISHED!")