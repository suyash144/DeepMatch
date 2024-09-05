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


if __name__ == "__main__":
    # reshape for all data
    mode = 'train' # 'train' or 'test'
    old_data_root = os.path.join(os.getcwd(), os.pardir, 'data_unitmatch')
    new_data_root = os.path.join(os.getcwd(), os.pardir, 'test_R_DATA_UnitMatch')
    mouse_names = os.listdir(old_data_root)
    mouse_names = ['AL031',]

    for name in mouse_names:
        name_path = os.path.join(old_data_root, name)
        probes = os.listdir(name_path)
        # probes = ['19011119461',]
        print(probes)
        for probe in probes:
            name_probe_path = os.path.join(name_path,probe)
            locations = os.listdir(name_probe_path)
            for location in locations:
                name_probe_location_path = os.path.join(name_probe_path, location)
                dates = os.listdir(name_probe_location_path)
                dates = [d for d in dates if is_date_filename(d)] # filter out non-date filenames
                for date in dates:
                    name_probe_location_date_path = os.path.join(name_probe_location_path, date)
                    experiments = os.listdir(name_probe_location_date_path)
                    for experiment in experiments:
                        # copy PreparedData.mat to new location
                        PreparedData_path = os.path.join(name_probe_location_date_path, experiment, 'PreparedData.mat')
                        dest_PreparedData_path = PreparedData_path.replace(old_data_root, new_data_root)
                        dest_PreparedData_directory = os.path.dirname(dest_PreparedData_path)
                        os.makedirs(dest_PreparedData_directory, exist_ok=True)
                        shutil.copyfile(PreparedData_path, dest_PreparedData_path)
                        # load channel map and channel positions for reshaping
                        ChannelPos = load_channel_positions(name,probe,location,date,experiment,mode)
                        ChannelMap = load_channel_map(name,probe,location,date,experiment,mode)
                        params = get_default_param()
                        # prepare waveform data for reshaping
                        np_path = os.path.join(name_probe_location_date_path, experiment, 'RawWaveforms')
                        np_file_names = os.listdir(np_path)
                        print('start processing: ', 'name', name, 'probe', probe, 'location', location, 'date', date, 'experiment', experiment)
                        for np_file_name in np_file_names:
                            np_waveform_file = os.path.join(np_path, np_file_name)
                            data = np.load(np_waveform_file) # (82,384,2)
                            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                                # print("Data contains NaNs or infs, cleaning required.")
                                # data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                                print("Bad data, just fill with zeros and -1")
                                Rwaveform = np.zeros((1,1,1))
                                MaxSitepos = np.array([-1,-1])
                            else:
                                # print('np_file_name', np_file_name)
                                MaxSiteMean, MaxSitepos, sorted_goodChannelMap, sorted_goodpos, Rwaveform = extract_Rwaveforms(data, ChannelPos, ChannelMap, params)
                            # print('Rwaveform shape:', Rwaveform.shape, 'MaxSitepos shape:', MaxSitepos.shape)
                            # sys.exit()
                            dest_path = np_waveform_file.replace(old_data_root, new_data_root)  # dest means destination 
                            dest_directory = os.path.dirname(dest_path)
                            os.makedirs(dest_directory, exist_ok=True)

                            new_data = {
                                "waveform": Rwaveform,
                                "MaxSitepos": MaxSitepos
                            }
                            with h5py.File(dest_path, 'w') as f:
                                for key, value in new_data.items():
                                    f.create_dataset(key, data=value)
                        
                        print('finish processing: ', 'name', name, 'probe', probe, 'location', location, 'date', date, 'experiment', experiment)
                        