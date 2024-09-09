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


if __name__ == '__main__':
    from myutil import read_good_id_from_mat, select_good_units_files,is_date_filename
else:
    from utils.myutil import read_good_id_from_mat,select_good_units_files,is_date_filename

    
class AE_NeuropixelsDataset(Dataset):
    def __init__(self, root: str, batch_size = 32):
        self.root = root
        self.mouse_names = os.listdir(self.root)
        self.batch_size = batch_size
        self.np_file_names = []

        for name in self.mouse_names:
            name_path = os.path.join(self.root, name)
            experiments = os.listdir(name_path)
            for experiment in experiments:
                experiment_path = os.path.join(name_path, experiment)
                good_units_index = read_good_id_from_mat(os.path.join(experiment_path, 'PreparedData.mat'))
                len_good_units = len(good_units_index[good_units_index == 1])
                if len_good_units <= self.batch_size:
                    continue
                good_units_files = select_good_units_files(os.path.join(experiment_path, 'RawWaveforms'), good_units_index)
                # Store each file name twice to represent two data points
                for file in good_units_files:
                    # self.np_file_names.append((file, 0))  # First half of data
                    # self.np_file_names.append((file, 1))  # Second half of data
                    self.np_file_names.append(file)

        self.n_neurons = len(self.np_file_names)

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
            data = waveform[..., 1]  # Second half

        return data