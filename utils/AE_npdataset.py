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
            name_path = os.path.join(self.root, name)
            paths_from_UM = read_datapaths(name)                  # paths on server read from UnitMatch.mat, for this mouse only
            probes = os.listdir(name_path)                        
            for probe in probes:
                probe_path = os.path.join(name_path, probe)
                locs = os.listdir(probe_path)
                if os.path.isfile(locs[0]):
                    continue
                for i, p in enumerate(paths_from_UM["probe"]):
                    if p!=probe:
                        del paths_from_UM["mouse"][i]
                        del paths_from_UM["probe"][i]
                        del paths_from_UM["loc"][i]
                        del paths_from_UM["recordings"][i]
                for loc in locs:
                    loc_path = os.path.join(probe_path, loc)
                    experiments = os.listdir(loc_path)
                    for i, l in enumerate(paths_from_UM["loc"]):
                        if l!=loc:
                            del paths_from_UM["mouse"][i]
                            del paths_from_UM["probe"][i]
                            del paths_from_UM["loc"][i]
                            del paths_from_UM["recordings"][i]
                    if len(paths_from_UM["recordings"]) != 1:
                        print("No. of possible experiments: ", len(paths_from_UM["recordings"]))
                        raise ValueError(f"Unable to uniquely determine the experiment from (mouse, probe, location):{name, probe, loc}")
                    
                    server_paths = paths_from_UM["recordings"][0]           # this is now just a list of filepaths

                    for i, experiment in enumerate(experiments):
                        experiment_path = os.path.join(loc_path, experiment)     

                        if len(experiments) != len(server_paths):
                            # check the local data structure matches the one on the server
                            print("No. of experiments on the local machine: ", len(experiments))
                            print("No. of filepaths from UnitMatch.mat: ", len(server_paths))
                            raise ValueError("CAUTION: there is a mismatch between data on local machine and server - please check")
                        
                        good_units_index = read_good_id_from_mat(os.path.join(server_paths[i], 'PreparedData.mat'))
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


np_root = os.path.join(os.path.dirname(os.getcwd()), 'R_DATA_UnitMatch')
np_dataset = AE_NeuropixelsDataset(root=np_root,batch_size=32)