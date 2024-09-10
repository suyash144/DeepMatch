import numpy as np
import os, sys, time
import re
import copy as cp
import numpy as np
import h5py
import shutil


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.pardir, os.pardir))
    sys.path.insert(0, os.path.join(os.pardir))
    sys.path.insert(0, os.path.join(os.getcwd()))

from utils.myutil import *
from utils.param_fun import *
from utils.read_datapaths import *


if __name__ == "__main__":
    # reshape for all data
    mode = 'train' # 'train' or 'test'
    old_data_root = os.path.join(os.getcwd(), os.pardir, 'data_unitmatch')
    new_data_root = os.path.join(os.getcwd(), os.pardir, 'R_DATA_UnitMatch')           # where data is saved after preprocessing
    mouse_names = os.listdir(old_data_root)

    recordings_dict = read_datapaths(mouse_names)

    for i, mouse in enumerate(recordings_dict["mouse"]):
        experiments = recordings_dict["recordings"][i]
        for experiment in experiments:
            # copy PreparedData.mat to new location
            # PreparedData_path = os.path.join(name_probe_location_date_path, experiment, 'PreparedData.mat')
            # dest_PreparedData_path = PreparedData_path.replace(old_data_root, new_data_root)
            # dest_PreparedData_directory = os.path.dirname(dest_PreparedData_path)
            # os.makedirs(dest_PreparedData_directory, exist_ok=True)
            # shutil.copyfile(PreparedData_path, dest_PreparedData_path)
            # load channel map and channel positions for reshaping
            ChannelPos = np.load(os.path.join(experiment,"channel_positions.npy"))
            ChannelMap = np.load(os.path.join(experiment,"channel_map.npy"))
            params = get_default_param()
            # prepare waveform data for reshaping
            np_path = os.path.join(experiment,"qMetrics" ,'RawWaveforms')
            np_file_names = os.listdir(np_path)

            for np_file_name in np_file_names:
                np_waveform_file = os.path.join(np_path, np_file_name)
                try:
                    data = np.load(np_waveform_file, allow_pickle=True) # (82,384,2)
                except:
                    # Handles the error where data is saved as a HDF5 file, implying it has already been preprocessed.
                    f = h5py.File(np_waveform_file, 'r')
                    Rwaveform = f['waveform']
                    MaxSitepos = f['MaxSitepos']
                    save_waveforms_hdf5(experiment, new_data_root, mouse, np_file_name, Rwaveform, MaxSitepos)
                    print("Saved as HDF5 -> implies this has already been preprocessed")
                    continue
                if data.shape != (82,384,2):
                    print(f"Expected waveform shape to be (82,384,2) - instead got {data.shape}")
                if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                    # print("Data contains NaNs or infs, cleaning required.")
                    # data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                    print(f"Bad data for mouse {mouse}, just fill with zeros and -1")
                    print(np_waveform_file)
                    Rwaveform = np.zeros((1,1,1))
                    MaxSitepos = np.array([-1,-1])
                else:
                    # print('np_file_name', np_file_name)
                    MaxSiteMean, MaxSitepos, sorted_goodChannelMap, sorted_goodpos, Rwaveform = extract_Rwaveforms(data, ChannelPos, ChannelMap, params)
                # print('Rwaveform shape:', Rwaveform.shape, 'MaxSitepos shape:', MaxSitepos.shape)
                # sys.exit()
                save_waveforms_hdf5(experiment, new_data_root, mouse, np_file_name, Rwaveform, MaxSitepos)