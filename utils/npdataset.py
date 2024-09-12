# Wentao Qiu, 2023-10-07
# qiuwentao1212@gmail.com


import os, sys
import random
import copy as cp
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader,Sampler
import json


if __name__ == '__main__':
    from myutil import read_good_id_from_mat, select_good_units_files,is_date_filename, read_good_ids
else:
    from utils.myutil import read_good_id_from_mat,select_good_units_files,is_date_filename, read_good_ids

    
class NeuropixelsDataset(Dataset):
    def __init__(self, root: str, batch_size = 30, mode = 'train'):
        self.root = root
        self.mouse_names = os.listdir(self.root)
        self.experiment_unit_map = {}  # Maps experiment to its units' file paths
        self.batch_size = batch_size
        self.mode = mode

        self.all_files = read_good_ids(self.root, self.batch_size, finetune=True)

    def __len__(self):
        return len(self.all_files)

    def _normalize_waveform(self, waveform):
        # max-min normalization
        max_val = np.max(waveform)
        min_val = np.min(waveform)
        return (waveform - min_val) / (max_val - min_val)
    
    def _augment(self, data):
        # Apply random augmentations to data, shape [T,C]
        choice = random.choice(["roll_up", "roll_down", "none"])
        if choice == "roll_up":
            # implement roll_up augmentation
            C = data.shape[1]  # Number of channels
            # Indices for odd channels, excluding the last one if C is odd
            odd_indices = np.arange(0, C - 1, 2)
            # Indices for even channels, excluding the last one
            even_indices = np.arange(1, C - 1, 2)
            # Shift odd channels up, excluding the last odd channel
            if len(odd_indices) > 1:  # Check if there are at least 2 odd channels to roll
                data[:, odd_indices[:-1]] = data[:, odd_indices[1:]]
            # Shift even channels up, excluding the last even channel
            if len(even_indices) > 1:  # Check if there are at least 2 even channels to roll
                data[:, even_indices[:-1]] = data[:, even_indices[1:]]
        elif choice == "roll_down":
            # implement roll_down augmentation
            C = data.shape[1]  # Number of channels
            odd_indices = np.arange(2, C, 2)
            even_indices = np.arange(3, C, 2)
            if len(odd_indices) > 0:  # Check if there are odd channels to roll
                data[:, odd_indices] = data[:, odd_indices - 2]
            if len(even_indices) > 0:  # Check if there are even channels to roll
                data[:, even_indices] = data[:, even_indices - 2]
        elif choice == "none":
            # No augmentation
            pass
        return data
        
    def __getitem__(self, i):
        experiment_path, neuron_file = self.all_files[i]
        with h5py.File(neuron_file, 'r') as f:
            waveform = f['waveform'][()] 
            MaxSitepos = f['MaxSitepos'][()]
        # waveform [T,C,2]
        # waveform_fh = self._normalize_waveform(waveform[..., 0])
        # waveform_sh = self._normalize_waveform(waveform[..., 1])
        ## data augmentation version
        if self.mode == 'train':
            waveform_fh = self._augment(waveform[..., 0])
            waveform_sh = self._augment(waveform[..., 1])
        else:
            waveform_fh = waveform[..., 0]
            waveform_sh = waveform[..., 1]
        # ## standard version
        # waveform_fh = waveform[..., 0]
        # waveform_sh = waveform[..., 1]
        return waveform_fh, waveform_sh, MaxSitepos, experiment_path, neuron_file

# class TrainExperimentBatchSampler(Sampler):
#     def __init__(self, data_source, batch_size, shuffle=False):
#         self.data_source = data_source
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.experiment_batches = self._create_batches()

#     def _create_batches(self):
#         batches = []
#         for experiment, unit_paths in self.data_source.experiment_unit_map.items():
#             # Create a mapping from file paths to indices
#             file_to_idx = {file: idx for idx, (exp, file) in enumerate(self.data_source.all_files) if exp == experiment}
#             experiment_indices = [file_to_idx[file] for file in unit_paths]
#             # Store experiment_indices for each experiment
#             batches.append(experiment_indices)
#         return batches

#     def __iter__(self):
#         iter_batches = []
#         for experiment_indices in self.experiment_batches:
#             # Shuffle the indices within each experiment if required
#             if self.shuffle:
#                 random.shuffle(experiment_indices)
#             # Generate random batches for the current experiment
#             iter_batches.extend([
#                 experiment_indices[i:i + self.batch_size]
#                 for i in range(0, len(experiment_indices), self.batch_size)
#             ])
#         # Shuffle the order of batches (experiment order) if required
#         if self.shuffle:
#             random.shuffle(iter_batches)
#         return iter(iter_batches)

#     def __len__(self):
#         # Calculate the total number of batches by summing up the number of batches per experiment
#         total_batches = sum((len(exp_indices) + self.batch_size - 1) // self.batch_size
#                             for exp_indices in self.experiment_batches)
#         return total_batches

class TrainExperimentBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, shuffle=False):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.experiment_batches = self._create_batches()

    def _create_batches(self):
        batches = []
        for experiment, unit_paths in self.data_source.experiment_unit_map.items():
            file_to_idx = {file: idx for idx, (exp, file) in enumerate(self.data_source.all_files) if exp == experiment}
            experiment_indices = [file_to_idx[file] for file in unit_paths]
            batches.append(experiment_indices)
        return batches

    def __iter__(self):
        iter_batches = []
        for experiment_indices in self.experiment_batches:
            if self.shuffle:
                random.shuffle(experiment_indices)
            for i in range(0, len(experiment_indices), self.batch_size):
                batch = experiment_indices[i:i + self.batch_size]
                # Check if the last batch is smaller than batch_size
                if len(batch) < self.batch_size:
                    # Resample additional items from the experiment_indices to fill the batch
                    shortfall = self.batch_size - len(batch)
                    additional_samples = random.choices(experiment_indices, k=shortfall)
                    batch.extend(additional_samples)
                iter_batches.append(batch)
        if self.shuffle:
            random.shuffle(iter_batches)
        return iter(iter_batches)

    def __len__(self):
        total_batches = sum((len(exp_indices) + self.batch_size - 1) // self.batch_size for exp_indices in self.experiment_batches)
        return total_batches

class ValidationExperimentBatchSampler(Sampler):
    """
    Creates one batch per experiment with all data points for validation.
    Optionally shuffles data within each experiment batch in each iteration.
    """
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.experiment_batches = self._create_batches()

    def _create_batches(self):
        batches = []
        for experiment, unit_paths in self.data_source.experiment_unit_map.items():
            # Create a mapping from file paths to indices
            file_to_idx = {file: idx for idx, (exp, file) in enumerate(self.data_source.all_files) if exp == experiment}
            experiment_indices = [file_to_idx[file] for file in unit_paths]
            # Each experiment is a single batch with all its units
            batches.append(experiment_indices)
        return batches

    def __iter__(self):
        iter_batches = []
        for experiment_indices in self.experiment_batches:
            # Shuffle the indices within each experiment if required
            if self.shuffle:
                random.shuffle(experiment_indices)
            iter_batches.append(experiment_indices)
        return iter(iter_batches)

    def __len__(self):
        return len(self.experiment_batches)
    
if __name__ == "__main__":
    # train dataset
    batch_size = 10
    train_data_root = os.path.join(os.path.dirname(os.getcwd()), 'R_DATA_UnitMatch')
    train_dataset = NeuropixelsDataset(root=train_data_root,batch_size=batch_size, mode='train')
    train_sampler = TrainExperimentBatchSampler(train_dataset,batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)

    # Initialize a set to keep track of the experiments we have printed
    printed_experiments = set()
    # Iterate over the dataloader and print batch information
    batch_counter = 0
    for batch in train_loader:
        data1, data2, MaxSitepos, experiment_paths,neuron_files = batch

        # # Check if we have printed this experiment already
        # if experiment_path not in printed_experiments:
        #     print(f"Experiment: {experiment_path}")
        #     print(f"Batch shapes - data1: {data1.shape}, data2: {data2.shape}")
        #     printed_experiments.add(experiment_path)

        # # Stop after printing 3 different experiments
        # if len(printed_experiments) >= 1:
        #     break

    # # test dataset
    # test_data_root = os.path.join(os.getcwd(), os.pardir,os.pardir, 'test_DATA_UnitMatch')
    # test_dataset = NeuropixelsDataset(root=test_data_root,batch_size=0, mode='test')
    # test_sampler = TestExperimentBatchSampler(test_dataset)
    # test_dataloader = DataLoader(test_dataset, batch_sampler=test_sampler)

    # for batch in test_dataloader:
    #     data1, data2, shank_idx, experiment_path = batch
    #     print(f"Experiment: {experiment_path}")
    #     print(f"Batch shapes - data1: {data1.shape}, data2: {data2.shape}, shank_idx: {shank_idx.shape}")

