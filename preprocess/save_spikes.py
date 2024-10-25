import os, sys
sys.path.insert(0, os.getcwd())
import pandas as pd
import h5py
from utils.read_datapaths import *
from utils.myutil import get_exp_id
from tqdm import tqdm

data_root = os.path.join(os.path.dirname(os.getcwd()), 'ALL_DATA')
mouse_names = os.listdir(data_root)

df = pd.DataFrame(read_datapaths(mouse_names))
base = r"C:\Users\suyas\ALL_DATA"

for idx, row in tqdm(df.iterrows(), total=len(df)):
    paths = row["recordings"]
    for path in paths:
        spktimes = np.load(os.path.join(path, r"spike_times.npy"))
        spkclus = np.load(os.path.join(path, r"spike_clusters.npy"))
        data = {
            "spkclus": spkclus,
            "spktimes": spktimes
        }
        exp_id = get_exp_id(path, row["mouse"])
        dest_path = os.path.join(base,row["mouse"],row["probe"],row["loc"],exp_id,"spikes.npy")
        with h5py.File(dest_path, 'w') as f:
            for key, value in data.items():
                f.create_dataset(key, data=value)
