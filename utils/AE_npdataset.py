# Wentao Qiu, 2023-10-07
# qiuwentao1212@gmail.com


import os, sys
import random
import re
import copy as cp
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
import json


if __name__ == '__main__':
    from myutil import read_good_id_from_mat, select_good_units_files,is_date_filename, read_good_ids
    from read_datapaths import *
else:
    from utils.myutil import read_good_id_from_mat,select_good_units_files,is_date_filename, read_good_ids
    from utils.read_datapaths import *

    
class AE_NeuropixelsDataset(Dataset):
    def __init__(self, root: str, batch_size = 32):
        self.root = root
        self.mouse_names = os.listdir(self.root)
        self.batch_size = batch_size

        self.np_file_names = read_good_ids(self.root, self.batch_size, finetune=False)
        self.n_neurons = len(self.np_file_names)
        if self.n_neurons < 1:
            print("No data! Try reducing batch size?")

    def __len__(self):
        return self.n_neurons

    def __getitem__(self, i):
        # file_name, half = self.np_file_names[i]
        # data = np.load(file_name)
        file_name = self.np_file_names[i]
        # Randomly pick 0 or 1 to choose the first half or second half of the data
        half = random.randint(0, 1)
        with h5py.File(file_name, 'r') as f:
            waveform = f['waveform'][()] 
            MaxSitepos = f['MaxSitepos'][()]
        if half == 0:
            data = waveform[..., 0]  # First half
        else:
            try:
                data = waveform[..., 1]  # Second half
            except:
                data = waveform[..., 0]  # First half
        # Handle data being the wrong shape
        if data.shape != (60,30):
            data = np.zeros((60,30))
        return data


# np_root = os.path.join(os.path.dirname(os.getcwd()), 'R_DATA_UnitMatch')
# np_dataset = AE_NeuropixelsDataset(root=np_root,batch_size=32)