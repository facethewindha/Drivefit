import os
import numpy as np

folder = "datasets/box_info_coords/cloud"

files = sorted(os.listdir(folder))[:10]

for f in files:
    x = np.load(os.path.join(folder, f), allow_pickle=True)
    print(f, x.shape, x)
