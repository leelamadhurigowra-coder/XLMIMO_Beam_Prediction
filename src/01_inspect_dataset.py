
import os
import json
import scipy.io as sio

# ==========================================
# STEP 1: Define dataset path
# ==========================================

DATA_PATH = os.path.join("data")

print("=" * 60)
print("DEEPMIMO DATASET INSPECTION")
print("=" * 60)

# ==========================================
# STEP 2: Check whether dataset exists
# ==========================================

if not os.path.exists(DATA_PATH):
    print("Dataset folder not found!")
    print("Please check the data folder location.")

else:
    print("\nDataset folder found successfully!")

# ==========================================
# STEP 3: List all files
# ==========================================

files = os.listdir(DATA_PATH)

print("\nTotal number of files:", len(files))

print("\nFiles available:")

for file in files:
    print(file)

# ==========================================
# STEP 4: Read simulation parameters
# ==========================================

params_file = os.path.join(DATA_PATH, "params.json")

if os.path.exists(params_file):

    with open(params_file, "r") as f:
        params = json.load(f)

    print("\nSimulation Parameters:")
    print(json.dumps(params, indent=4))

else:
    print("\nparams.json not found!")

# ==========================================
# STEP 5: Inspect MATLAB files
# ==========================================

print("\nInspecting MATLAB files...")

mat_files = [f for f in files if f.endswith(".mat")]

for file in mat_files[:5]:

    file_path = os.path.join(DATA_PATH, file)

    print("\nFile:", file)

    try:
        data = sio.loadmat(file_path)

        print("Variables:", list(data.keys()))

        for key in data:

            if not key.startswith("__"):

                print("Variable:", key)
                print("Shape:", data[key].shape)
                print("Data type:", data[key].dtype)

    except Exception as e:
        print("Could not read file:", e)

print("\nDataset inspection completed!")