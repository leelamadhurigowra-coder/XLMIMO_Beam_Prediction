from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

root = Path.cwd()
out = root / "results" / "experiment_32x32"

labels = np.load(out / "los_proxy_beam_labels.npy")
mask = np.load(out / "valid_receiver_mask.npy")

y = labels[mask & (labels >= 0)]

# Reproduce the same 80/20 random split used in training.
from sklearn.model_selection import train_test_split

indices = np.arange(len(y))
train_idx, test_idx = train_test_split(
    indices, test_size=0.2, random_state=42
)

y_train = y[train_idx]
y_test = y[test_idx]

values, counts = np.unique(y_train, return_counts=True)
majority_class = values[np.argmax(counts)]
pred = np.full_like(y_test, majority_class)

print("=== MAJORITY-CLASS BASELINE ===")
print("Majority beam:", int(majority_class))
print("Test receivers:", len(y_test))
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("Macro F1:", round(
    f1_score(y_test, pred, average="macro", zero_division=0), 4
))
print("\nCompare these results with the Random Forest.")
