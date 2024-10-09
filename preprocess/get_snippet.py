import numpy as np
import os, sys, time
import re
import copy as cp
import numpy as np
import h5py
import shutil
import json


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
    old_data_root = os.path.join(os.path.dirname(os.getcwd()), 'data_unitmatch')
    new_data_root = os.path.join(os.path.dirname(os.getcwd()), 'R_DATA_UnitMatch')           # where data is saved after preprocessing
    server_root = r"\\znas\Lab\Share\UNITMATCHTABLES_ENNY_CELIAN_JULIE\FullAnimal_KSChanMap"
    mouse_names = os.listdir(old_data_root)

    recordings_dict = read_datapaths(mouse_names)

    for i, mouse in enumerate(recordings_dict["mouse"]):
        experiments = recordings_dict["recordings"][i]
        for j, experiment in enumerate(experiments):
            
            # load channel map and channel positions for reshaping
            ChannelPos = np.load(os.path.join(experiment,"channel_positions.npy"))
            ChannelMap = np.load(os.path.join(experiment,"channel_map.npy"))
            params = get_default_param()
            # prepare waveform data for reshaping
            np_path = os.path.join(experiment,"qMetrics" ,'RawWaveforms')
            np_file_names = os.listdir(np_path)
    
            # Construct the path where we want to save processed data locally
            experiment_id = experiment[experiment.find(mouse):]
            experiment_id = experiment_id.replace(mouse, '')
            experiment_id = experiment_id.replace("\\", "_")
            experiment_id = get_exp_id(experiment, mouse)
            probe = recordings_dict["probe"][i]
            loc = recordings_dict["loc"][i]
            dest_path = os.path.join(new_data_root, mouse, probe, loc, experiment_id)
            os.makedirs(dest_path, exist_ok=True)

            um_path = os.path.join(server_root, mouse, probe, loc, "UnitMatch", "Unitmatch.mat")
            good_id = read_good_id_from_mat(um_path, j+1).tolist()

            metadata_file = open(os.path.join(dest_path, "metadata.json"), "w")

            # Store the metadata for the experiment in a json file
            metadata = {
                "mouse": mouse,
                "probe": probe,
                "loc": loc,
                "experiment_id": experiment_id,
                "good_ids": good_id
            }
            json.dump(metadata, metadata_file, indent=6)

            for np_file_name in np_file_names:
                np_waveform_file = os.path.join(np_path, np_file_name)
                try:
                    data = np.load(np_waveform_file, allow_pickle=True) # (82,384,2)
                except:
                    # Handles the error where data is saved as a HDF5 file, implying it has already been preprocessed.
                    f = h5py.File(np_waveform_file, 'r')
                    Rwaveform = f['waveform']
                    MaxSitepos = f['MaxSitepos']
                    save_waveforms_hdf5(dest_path, np_file_name, Rwaveform, MaxSitepos)
                    print("Saved as HDF5 -> implies this has already been preprocessed")
                    continue
                if data.shape != (82,384,2):
                    print(f"Expected waveform shape to be (82,384,2) - instead got {data.shape}")
                if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                    # print("Data contains NaNs or infs, cleaning required.")
                    # data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                    # print(f"Bad data for mouse {mouse}, just fill with zeros and -1")
                    # print(np_waveform_file)
                    print(f"Corrupted data in {np_waveform_file}")
                    continue
                    Rwaveform = np.zeros((1,1,1))
                    MaxSitepos = np.array([-1,-1])
                else:
                    # print('np_file_name', np_file_name)
                    try:
                        MaxSiteMean, MaxSitepos, sorted_goodChannelMap, sorted_goodpos, Rwaveform = extract_Rwaveforms(data, ChannelPos, ChannelMap, params)
                    except:
                        print(f"Failed to extract waveforms for {np_waveform_file}")
                        continue

                save_waveforms_hdf5(dest_path, np_file_name, Rwaveform, MaxSitepos)
        print(f"Finished processing data for experiment {experiment_id}")
    