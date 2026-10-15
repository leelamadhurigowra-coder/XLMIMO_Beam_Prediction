
import os
import json
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report
)

# ============================================================
# 1. CREATE OUTPUT FOLDERS
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("=" * 60)
print("RANDOM FOREST BEAM PREDICTION")
print("=" * 60)


# ============================================================
# 2. LOAD DATASET
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

print("Original feature shape:", X_all.shape)
print("Beam label shape:", Y_all.shape)
print("Receiver index shape:", receiver_indices.shape)


# ============================================================
# 3. ALIGN FEATURES AND LABELS
# ============================================================

print("\nAligning features with usable receivers...")

X = X_all[receiver_indices]

# Select center subcarrier (index 32)
y = Y_all[:, 32]

print("Aligned feature shape:", X.shape)
print("Target shape:", y.shape)


# ============================================================
# 4. REMOVE INVALID FEATURE ROWS
# ============================================================

valid_mask = np.isfinite(X).all(axis=1)

X = X[valid_mask]
y = y[valid_mask]

print("\nFinal dataset information:")
print("Number of samples:", len(X))
print("Number of features:", X.shape[1])
print("Number of unique beam classes:", len(np.unique(y)))


# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 6. CREATE RANDOM FOREST CLASSIFIER
# ============================================================

print("\nCreating Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 7. TRAIN MODEL
# ============================================================

print("\nTraining started...")

model.fit(X_train, y_train)

print("Training completed successfully!")


# ============================================================
# 8. MAKE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)


# ============================================================
# 9. CALCULATE ACCURACY
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

# Calculate top-5 accuracy manually.
# This also handles test samples whose class was not
# represented in the training set.

top5_correct = 0

for i in range(len(y_test)):

    top_indices = np.argsort(y_prob[i])[-5:]

    top_classes = model.classes_[top_indices]

    if y_test[i] in top_classes:
        top5_correct += 1

top5_accuracy = top5_correct / len(y_test)


# ============================================================
# 10. DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("RANDOM FOREST EVALUATION RESULTS")
print("=" * 60)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy percentage: {accuracy * 100:.2f}%")

print(f"\nTop-5 Accuracy: {top5_accuracy:.4f}")
print(
    f"Top-5 Accuracy percentage: "
    f"{top5_accuracy * 100:.2f}%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 11. SAVE TRAINED MODEL
# ============================================================

model_path = "models/random_forest.pkl"

joblib.dump(model, model_path)

print("\nTrained model saved at:", model_path)


# ============================================================
# 12. SAVE EVALUATION REPORT
# ============================================================

report = {
    "model": "Random Forest",
    "task": "Center-subcarrier beam prediction",
    "subcarrier_index": 32,
    "num_features": int(X.shape[1]),
    "num_samples": int(len(X)),
    "training_samples": int(len(X_train)),
    "testing_samples": int(len(X_test)),
    "num_beam_classes": int(len(np.unique(y))),
    "accuracy": float(accuracy),
    "accuracy_percent": float(accuracy * 100),
    "top5_accuracy": float(top5_accuracy),
    "top5_accuracy_percent": float(top5_accuracy * 100)
}

report_path = "results/random_forest_report.json"

with open(report_path, "w") as f:
    json.dump(report, f, indent=4)

print("Evaluation report saved at:", report_path)


# ============================================================
# 13. FINAL STATUS
# ============================================================

print("\n")
print("=" * 60)
print("RANDOM FOREST TRAINING FINISHED")
print("=" * 60)

print("Model: models/random_forest.pkl")
print("Report: results/random_forest_report.json")

print("\nNext step: Deep Learning model training.")