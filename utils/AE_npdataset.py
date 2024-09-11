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
    from myutil import read_good_id_from_mat, select_good_units_files,is_date_filename
    from read_datapaths import *
else:
    from utils.myutil import read_good_id_from_mat,select_good_units_files,is_date_filename
    from utils.read_datapaths import *

    
class AE_NeuropixelsDataset(Dataset):
    def __init__(self, root: str, batch_size = 32):
        self.root = root
        self.mouse_names = os.listdir(self.root)
        self.batch_size = batch_size
        self.np_file_names = []

        for name in self.mouse_names:
            if name=="AV008":
                # skipping AV008 for now due to spike sorting issue
                continue
            name_path = os.path.join(self.root, name)
            paths_from_UM = read_datapaths(name)                  # paths on server read from UnitMatch.mat, for this mouse only
            probes = os.listdir(name_path)                        
            for probe in probes:
                probe_path = os.path.join(name_path, probe)
                locs = os.listdir(probe_path)
                for loc in locs:
                    loc_path = os.path.join(probe_path, loc)
                    experiments = os.listdir(loc_path)
                    for experiment in experiments:
                        experiment_path = os.path.join(loc_path, experiment)
                        try:
                            metadata_file = os.path.join(experiment_path, "metadata.json")
                        except:
                            print(experiment_path)
                            raise ValueError("Did not find metadata.json file for this experiment")
                        metadata = json.load(open(metadata_file))
                        good_units_index = metadata["good_ids"]
                        len_good_units = sum(good_units_index)
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


np_root = os.path.join(os.path.dirname(os.getcwd()), 'R_DATA_UnitMatch')
np_dataset = AE_NeuropixelsDataset(root=np_root,batch_size=32)