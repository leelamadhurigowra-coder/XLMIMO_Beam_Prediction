from pathlib import Path
from scipy.io import loadmat
import numpy as np

data = Path.cwd() / "data"

inter = loadmat(data / "inter_t001_tx000_r000.mat")["inter"]
pos = loadmat(data / "inter_pos_t001_tx000_r000.mat")["inter_pos"]

print("inter shape:", inter.shape)
print("inter_pos shape:", pos.shape)

print("\nFirst 30 interaction values:")
print(inter[10, :])

print("\nCoordinates for receiver 10, path 0:")
print(pos[10, 0])

print("\nCoordinates for receiver 10, path 1:")
print(pos[10, 1])

print("\nCoordinates for receiver 10, path 2:")
print(pos[10, 2])

finite = inter[np.isfinite(inter)]
print("\nUnique finite interaction values (first 40):")
print(np.unique(finite)[:40])

print("\nInteraction diagnostic complete.")
