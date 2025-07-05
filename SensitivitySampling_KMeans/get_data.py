from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
import os

# ----------------------------------
# Load Covertype Dataset (ID: 31)
# ----------------------------------
print("📥 Fetching Covertype dataset (ID 31)...")
covertype = fetch_ucirepo(id=31)
X_cover = covertype.data.features.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
# y_cover = covertype.data.targets.to_numpy()
print(f"✔ Covertype dataset loaded: {X_cover.shape}")

# ----------------------------------
# Load US Census Dataset (ID: 116)
# ----------------------------------
print("\n📥 Fetching US Census 1990 dataset (ID 116)...")
census = fetch_ucirepo(id=116)
X_census = census.data.features.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
# y_census = census.data.targets.to_numpy()
print(f"✔ US Census dataset loaded: {X_census.shape}")

# ----------------------------------
# Load Tower Dataset from Local file
# ----------------------------------
tower_path = os.path.join("data", "tower.txt")
print("\n📁 Loading Tower dataset from local file:", tower_path)

try:
    X_tower = np.loadtxt(tower_path, delimiter=",", dtype=np.float64)
    print(f"✔ Tower dataset loaded: {X_tower.shape}")
except FileNotFoundError:
    print(f"❌ File not found: {tower_path}. Please make sure 'tower.txt' is inside the 'data/' folder.")
    X_tower = None
