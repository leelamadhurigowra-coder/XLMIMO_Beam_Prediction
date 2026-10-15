from pathlib import Path
import json
import time
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score
)

ROOT = Path.cwd()
OUT = ROOT / "results" / "experiment_32x32"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ROOT / "models" / "experiment_32x32"
MODELS.mkdir(parents=True, exist_ok=True)

X_path = OUT / "frequency_dataset_los_proxy.npy"
y_path = OUT / "los_proxy_beam_labels.npy"
mask_path = OUT / "valid_receiver_mask.npy"

X_all = np.load(X_path, mmap_mode="r")
y_all = np.load(y_path)
mask = np.load(mask_path)

# Mean-pool the 64 subcarrier features for a simple baseline.
valid = mask & (y_all >= 0)
X = np.asarray(X_all[valid].mean(axis=1), dtype=np.float32)
y = y_all[valid].astype(np.int64)

finite = np.isfinite(X).all(axis=1)
X = X[finite]
y = y[finite]

print("Feature matrix:", X.shape)
print("Labels:", y.shape)
print("Classes:", np.unique(y).size)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest...")
start = time.perf_counter()
model.fit(X_train, y_train)
train_seconds = time.perf_counter() - start

start = time.perf_counter()
pred = model.predict(X_test)
inference_seconds = time.perf_counter() - start

proba = model.predict_proba(X_test)
classes = model.classes_

acc = accuracy_score(y_test, pred)
precision_macro, recall_macro, f1_macro, _ = (
    precision_recall_fscore_support(
        y_test, pred, average="macro", zero_division=0
    )
)
precision_weighted, recall_weighted, f1_weighted, _ = (
    precision_recall_fscore_support(
        y_test, pred, average="weighted", zero_division=0
    )
)

top5 = float(np.mean([yt in model.classes_[np.argsort(row)[-min(5, len(model.classes_)):]] for yt, row in zip(y_test, proba)]))

report = {
    "experiment": "32x32 LOS-proxy baseline",
    "target_warning": "Labels are geometric LOS proxies, not validated multipath-optimal beams.",
    "test_receivers": int(len(y_test)),
    "num_classes_in_model": int(len(classes)),
    "accuracy": float(acc),
    "macro_precision": float(precision_macro),
    "macro_recall": float(recall_macro),
    "macro_f1": float(f1_macro),
    "weighted_precision": float(precision_weighted),
    "weighted_recall": float(recall_weighted),
    "weighted_f1": float(f1_weighted),
    "top5_accuracy": float(top5),
    "training_time_seconds": float(train_seconds),
    "inference_time_seconds_total": float(inference_seconds),
    "inference_time_ms_per_receiver": float(
        inference_seconds * 1000 / len(X_test)
    ),
    "classification_report": classification_report(
        y_test, pred, zero_division=0, output_dict=True
    ),
    "confusion_matrix": confusion_matrix(y_test, pred).tolist()
}

import joblib
joblib.dump(model, MODELS / "random_forest_los_proxy.pkl")
(OUT / "random_forest_los_proxy_report.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)

print("\n=== RANDOM FOREST EVALUATION ===")
print(f"Accuracy:          {acc:.4f}")
print(f"Macro precision:   {precision_macro:.4f}")
print(f"Macro recall:      {recall_macro:.4f}")
print(f"Macro F1:          {f1_macro:.4f}")
print(f"Weighted precision:{precision_weighted:.4f}")
print(f"Weighted recall:   {recall_weighted:.4f}")
print(f"Weighted F1:       {f1_weighted:.4f}")
print(f"Top-5 accuracy:    {top5:.4f}")
print(f"Training time:     {train_seconds:.2f} seconds")
print(f"Inference ms/sample: {inference_seconds*1000/len(X_test):.4f}")
print("\nSaved model to:", MODELS)
print("Saved report to:", OUT / "random_forest_los_proxy_report.json")



