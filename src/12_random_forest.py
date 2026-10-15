
import numpy as np
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ============================================================
# RANDOM FOREST BASELINE MODEL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
# STEP 1: LOAD ENGINEERED FEATURES
# ------------------------------------------------------------

data = np.load(
    RESULTS_DIR / "engineered_features.npz",
    allow_pickle=True
)

print("Available features:", data.files)

# Select numerical features
feature_names = [
    "path_count",
    "mean_power",
    "max_power",
    "mean_delay"
]

X = np.column_stack([
    data[name] for name in feature_names
])

# ------------------------------------------------------------
# STEP 2: LOAD BEAM LABELS
# ------------------------------------------------------------

label_file = RESULTS_DIR / "beam_labels.npy"

if not label_file.exists():

    print("\nBeam labels are not available yet.")
    print("Feature preparation completed.")
    print("Generate optimal beam labels before training.")

    np.save(
        RESULTS_DIR / "random_forest_input_features.npy",
        X
    )

    raise SystemExit

y = np.load(label_file)

# ------------------------------------------------------------
# STEP 3: LOAD DATA SPLITS
# ------------------------------------------------------------

splits = np.load(
    RESULTS_DIR / "data_split_indices.npz"
)

train_idx = splits["train_indices"]
val_idx = splits["val_indices"]
test_idx = splits["test_indices"]

# ------------------------------------------------------------
# STEP 4: PREPARE TRAINING DATA
# ------------------------------------------------------------

X_train = X[train_idx]
X_val = X[val_idx]
X_test = X[test_idx]

y_train = y[train_idx]
y_val = y[val_idx]
y_test = y[test_idx]

# ------------------------------------------------------------
# STEP 5: TRAIN RANDOM FOREST
# ------------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

# ------------------------------------------------------------
# STEP 6: EVALUATE MODEL
# ------------------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nRandom Forest Test Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ------------------------------------------------------------
# STEP 7: SAVE MODEL
# ------------------------------------------------------------

model_path = MODELS_DIR / "random_forest.pkl"

joblib.dump(model, model_path)

# Save evaluation results
report = {
    "model": "Random Forest",
    "test_accuracy": float(accuracy),
    "number_of_trees": 100,
    "maximum_depth": 20,
    "features": feature_names
}

with open(
    RESULTS_DIR / "random_forest_report.json",
    "w"
) as f:
    json.dump(report, f, indent=4)

print("\nModel saved successfully!")
print("Model location:", model_path)
print("Report saved in results folder.")