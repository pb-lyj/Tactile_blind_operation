from glob import glob
import numpy as np
import os

NPY_ROOT_DIR = "./organized_data/env_0"
file_paths = sorted(glob(os.path.join(NPY_ROOT_DIR, "episode_*", "tactile.npy")))
print(f"Found {len(file_paths)} tactile.npy files")

for path in file_paths[:5]:
    arr = np.load(path)
    print(f"{path} shape: {arr.shape}")
