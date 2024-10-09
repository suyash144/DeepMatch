import scipy.io as spio
import os, sys
import numpy as np  
import h5py
import math,random
import re
import matplotlib.pyplot as plt
import json
import datetime
import pandas as pd
import mat73

def is_date_filename(filename):
    # Define a regular expression pattern for the date format
    pattern = r'^\d{4}-\d{2}-\d{2}$'
    return re.match(pattern, filename) is not None

def read_good_id_from_mat(filepath, recses):
    """
    Reads the GoodID field from the UniqueIDConversion struct in a MATLAB .mat file.
    Args:
    - filepath (str): Path to the .mat file.
    """
    with h5py.File(filepath, 'r') as file:
        # Navigate through the file structure to get to the Good_ID field
        # This might vary depending on the exact structure of your .mat file
        clusinfo = file['UniqueIDConversion']
        good_id = np.array(clusinfo['GoodID']).astype(int)
        recsesall = np.array(clusinfo['recsesAll']).astype(int)
        recsesall = np.squeeze(recsesall, axis=0)
        idx = np.where(recsesall==int(recses))
        good_id = good_id[idx]
        good_id = good_id.reshape(-1)
    return good_id

def select_good_units_files(directory, good_units_value):
    """
    Selects the filenames of the good units based on the good_units_value array.
    Args:
    - directory (str): The directory containing the unit files.
    - good_units_value (list or numpy.ndarray): An array where a value of 1 indicates a good unit.
    Returns:
    - list: A list of filenames corresponding to the good units.
    """
    good_units_files = []
    for index, is_good in enumerate(good_units_value):
        if is_good:
            filename = f"Unit{index}_RawSpikes.npy"
            filepath = os.path.join(directory, filename)
            if os.path.exists(filepath):  # Check if file exists before adding
                good_units_files.append(filepath)
            else:
                print(f"Warning: Expected file {filename} does not exist.")
    return good_units_files

def select_good_units_files_indices(directory, good_units_value):
    """
    Selects the filenames of the good units based on the good_units_value array.
    Args:
    - directory (str): The directory containing the unit files.
    - good_units_value (list or numpy.ndarray): An array where a value of 1 indicates a good unit.
    Returns:
    - list: A list of filenames corresponding to the good units.
    """
    good_units_files = []
    good_units_indices = []
    for index, is_good in enumerate(good_units_value):
        if is_good:
            filename = f"Unit{index}_RawSpikes.npy"
            filepath = os.path.join(directory, filename)
            if os.path.exists(filepath):  # Check if file exists before adding
                good_units_files.append(filepath)
                good_units_indices.append(index)
            else:
                print(f"Warning: Expected file {filename} does not exist.")
                # sys.exit()
    return good_units_files, good_units_indices

def load_channel_positions(mouse,probe,location,date,experiment,mode='train'):
    base = os.path.join(os.getcwd(),os.pardir, os.pardir)
    if mode == 'train':
        path_data = os.path.join(base, 'DATA_UnitMatch', mouse, probe, location,date,experiment)
    elif mode == 'test':
        path_data = os.path.join(base, 'test_DATA_UnitMatch', mouse, probe, location,date,experiment)
    filename = 'channel_positions.npy'
    filepath = os.path.join(path_data, filename)
    channel_positions = np.load(filepath)
    return channel_positions

def load_channel_map(mouse,probe,location,date,experiment,mode='train'):
    base = os.path.join(os.getcwd(),os.pardir, os.pardir)
    if mode == 'train':
        path_data = os.path.join(base, 'DATA_UnitMatch', mouse, probe, location,date,experiment)
    elif mode == 'test':
        path_data = os.path.join(base, 'test_DATA_UnitMatch', mouse, probe, location,date,experiment)
    filename = 'channel_map.npy'
    filepath = os.path.join(path_data, filename)
    channel_map = np.load(filepath)
    return channel_map

def get_default_param(param = None):
    """
    Create param, a dictionary with the default parameters.
    If a dictionary is given, it will add values to it without overwriting existing values.
    Do not need to give a dictionary.
    """
    tmp = {'nTime' : 82, 'nChannels' : 384, 'ChannelRadius' : 110,
           'RnChannels' : 30, 'RnTime' : 60, 
        }
    # if no dictionary is given just returns the default parameters
    if param == None:
        out = tmp
    else:    
        # Add default parameters to param dictionary, does not overwrite pre existing param values
        out = tmp | param
    if out['RnChannels'] %2 !=0:
        print('RnChannels is not even, please check')
    return out

def normalize_waveform(waveform):
    max_val = np.max(waveform)
    min_val = np.min(waveform)
    waveform = (waveform - min_val) / (max_val - min_val)
    return waveform

def load_mouse_data(mouse,probe,location,exps,mode='test'):
    base_path = os.path.dirname(os.getcwd())
    if mode == 'test':
        base_path = os.path.join(base_path,'test_R_DATA_UnitMatch')
    else:
        base_path = os.path.join(base_path,'R_DATA_UnitMatch')
    # first date
    base_path_1 = os.path.join(base_path,mouse,probe,location,exps[0])
    waveform_path_1 = os.path.join(base_path_1,'processed_waveforms')
    with open(os.path.join(base_path_1, "metadata.json")) as f:
        metadata1 = json.load(f)
    good_units_index_1 = metadata1["good_ids"]
    good_units_files_1,good_units_indices_1 = select_good_units_files_indices(waveform_path_1, good_units_index_1)
    # second date
    base_path_2 = os.path.join(base_path,mouse,probe,location,exps[1])
    waveform_path_2 = os.path.join(base_path_2,'processed_waveforms')
    with open(os.path.join(base_path_1, "metadata.json")) as f:
        metadata2 = json.load(f)
    good_units_index_2 = metadata2["good_ids"]
    good_units_files_2,good_units_indices_2 = select_good_units_files_indices(waveform_path_2, good_units_index_2)
    print('day 1 len', len(good_units_files_1))
    print('day 2 len', len(good_units_files_2))
    return good_units_files_1,good_units_indices_1,good_units_files_2,good_units_indices_2

def get_day_MaxSitepos(good_units_files_1,good_units_files_2):
    day1_MaxSitepos = []
    for index_file, filename in enumerate(good_units_files_1):
        with h5py.File(filename, 'r') as f:
            MaxSitepos = f['MaxSitepos'][()]
        day1_MaxSitepos.append(MaxSitepos)
    day1_MaxSitepos = np.array(day1_MaxSitepos)
    day2_MaxSitepos = []
    for index_file, filename in enumerate(good_units_files_2):
        # data = np.load(filename)
        with h5py.File(filename, 'r') as f:
            MaxSitepos = f['MaxSitepos'][()]
        day2_MaxSitepos.append(MaxSitepos)
    day2_MaxSitepos = np.array(day2_MaxSitepos)
    return day1_MaxSitepos,day2_MaxSitepos

# def sort_neurons_by_position(positions):
#     # Sort first by the y-axis values, then by the z-axis values within each y-group
#     sorted_indices = np.lexsort((positions[:, 1], positions[:, 0]))
#     return sorted_indices

def sort_neurons_by_position(positions):
    # Define a custom sort key
    def sort_key(index):
        pos = positions[index]
        # Shank is determined by dividing the y-position by 200
        shank = pos[0] // 200
        # Z depth is the second element
        z_depth = pos[1]
        # Y position within shank, considering the modulo 200 for y
        y_within_shank = pos[0] % 200
        return shank, z_depth, y_within_shank
    indices = np.arange(len(positions))
    sorted_indices = sorted(indices, key=sort_key)
    return np.array(sorted_indices)

# def rearrange_matrix(matrix, day1_MaxSitepos, day2_MaxSitepos):
#     # Sort neurons for day 1 based on their positions
#     day1_sorted_indices = sort_neurons_by_position(day1_MaxSitepos)
#     # Sort neurons for day 2 based on their positions
#     day2_sorted_indices = sort_neurons_by_position(day2_MaxSitepos)
#     # Rearrange rows and columns of the similarity matrix based on sorted indices
#     matrix_rearranged = matrix[day1_sorted_indices, :][:, day2_sorted_indices]
#     return matrix_rearranged,day1_sorted_indices,day2_sorted_indices

def rearrange_matrix(matrix, row_sorted_indices, col_sorted_indices):
    matrix_rearranged = matrix[row_sorted_indices, :][:, col_sorted_indices]
    return matrix_rearranged

def rearrange_four_matrix(matrix_11,matrix_12,matrix_21,matrix_22, day1_sorted_indices, day2_sorted_indices):
    # Rearrange rows and columns of the similarity matrix based on sorted indices
    matrix_11_rearranged = matrix_11[day1_sorted_indices, :][:, day1_sorted_indices]
    matrix_12_rearranged = matrix_12[day1_sorted_indices, :][:, day2_sorted_indices]
    matrix_21_rearranged = matrix_21[day1_sorted_indices, :][:, day2_sorted_indices]
    matrix_22_rearranged = matrix_22[day2_sorted_indices, :][:, day2_sorted_indices]
    return matrix_11_rearranged,matrix_12_rearranged,matrix_21_rearranged,matrix_22_rearranged

def get_sitepos_shank(MaxSitepos):
    shank = MaxSitepos[:,0] // 200
    return shank

def get_Sitepos_dist(day1_MaxSitepos,day2_MaxSitepos):
    # day1 doens't necessarily mean day1, treat 1 and 2 as row and col
    day1_len = len(day1_MaxSitepos)
    day2_len = len(day2_MaxSitepos)
    day1_shank = get_sitepos_shank(day1_MaxSitepos)
    day2_shank = get_sitepos_shank(day2_MaxSitepos)
    dist_matrix_actoss_days = np.zeros((day1_len,day2_len))
    for i in range(day1_len):
        for j in range(day2_len):
            if day1_shank[i] == day2_shank[j]:
                dist_matrix_actoss_days[i,j] = np.linalg.norm(day1_MaxSitepos[i] - day2_MaxSitepos[j])
            else:
                dist_matrix_actoss_days[i,j] = -1 # different shank,-1 means infinite distance
    return dist_matrix_actoss_days

def simple_filter_matrix(sim_matrix,row_MaxSitepos,col_MaxSitepos):
    dist_matrix = get_Sitepos_dist(row_MaxSitepos,col_MaxSitepos)
    # sparsification
    # cosine similarity is between -1 and 1, so we set them to -1
    sim_matrix[dist_matrix==-1] = -1
    return sim_matrix
    
def has_conflict_match(arr):
        unique_arr = np.unique(arr, axis=0)
        print('has conflict match',unique_arr.shape[0] != arr.shape[0])

def find_index_list(arr, x):
    try:
        return arr.index(x)
    except ValueError:
        return -1  #

def find_index_np(arr, x):
    # Find the indices where the condition is True
    indices = np.where(arr == x)
    # np.where() returns a tuple with arrays. For a 1D array, we just need the first element
    indices = indices[0]
    # Check if the indices array is not empty, which means the element was found
    if indices.size > 0:
        return indices[0]  # Return the first index where the value is x
    else:
        return -1 
    
def read_good_ids(root, batch_size, finetune:bool):
    """
    Output depends on whether you want to pre-train the encoder (finetune=False)
    or finetune it via contrastive learning (finetune=True). This must be specified.
    """
    mouse_names = os.listdir(root)
    np_file_names = []
    experiment_unit_map = {}

    for name in mouse_names:
        # if name=="AV008":
        #     # skipping AV008 for now due to spike sorting issue
        #     continue
        name_path = os.path.join(root, name)
        probes = os.listdir(name_path)                        
        for probe in probes:
            probe_path = os.path.join(name_path, probe)
            locs = os.listdir(probe_path)
            for loc in locs:
                loc_path = os.path.join(probe_path, loc)
                experiments = os.listdir(loc_path)
                for experiment in experiments:
                    experiment_path = os.path.join(loc_path, experiment)
                    good_units_files = read_good_files(experiment_path, batch_size)
                    if good_units_files is None:
                        continue
                    if finetune:
                        experiment_unit_map[experiment_path] = good_units_files
                    else:
                        for file in good_units_files:
                            np_file_names.append(file)
    
    if finetune:
        return experiment_unit_map
    else:
        return np_file_names

def read_good_files(experiment_path, batch_size):
    if not os.path.isdir(experiment_path):
        return None
    try:
        metadata_file = os.path.join(experiment_path, "metadata.json")
        metadata = json.load(open(metadata_file))
    except:
        print(experiment_path)
        raise ValueError("Did not find metadata.json file for this experiment")
    good_units_index = metadata["good_ids"]
    len_good_units = sum(good_units_index)
    if len_good_units <= batch_size:
        return None
    good_units_files = select_good_units_files(os.path.join(experiment_path, 'processed_waveforms'), good_units_index)
    return good_units_files

def get_exp_id(experiment_path:str, mouse:str):
    """
    experiment_path should be a full absolute path to the desired experiment folder on the server
    (NOT LOCAL PATH)
    """
    experiment_id = experiment_path[experiment_path.find(mouse):]
    experiment_id = experiment_id.replace(mouse, '')
    experiment_id = experiment_id.replace("\\", "_")
    return experiment_id

def get_unit_id(filepath:str):
    fp = os.path.basename(filepath)
    if fp[:4] == "Unit" and fp[-14:] == "_RawSpikes.npy":
        fp = fp.replace("Unit", "")
        id = fp.replace("_RawSpikes.npy", "")
        try:
            return int(id)
        except:
            raise ValueError(f"Invalid filepath format for this waveform: {filepath}", 
                         "Filename for waveform XX should be UnitXX_RawSpikes.npy")
    else:
        raise ValueError(f"Invalid filepath format for this waveform: {filepath}", 
                         "Filename for waveform XX should be UnitXX_RawSpikes.npy")
    
def exp_id_to_date(exp_id:str):
    date = exp_id[1:]
    date = date[:date.find("_")]
    l = date.split("-")
    date = datetime.date(int(l[0]), int(l[1]), int(l[2]))
    return date

def mtpath_to_expids(mt_path:str, matches:pd.DataFrame):

    sessions = set(matches["RecSes1"].unique())
    sessions = sessions.union(set(matches["RecSes2"].unique()))
    
    exp_folder = os.path.dirname(mt_path)
    for file in os.listdir(exp_folder):
        if not os.path.isdir(os.path.join(exp_folder, file)):
            continue
        else:
            some_exp = os.path.join(exp_folder, file)
            break
    with open(os.path.join(some_exp, "metadata.json")) as f:
        metadata = json.load(f)
    
    um_path = os.path.join(r"\\znas\Lab\Share\UNITMATCHTABLES_ENNY_CELIAN_JULIE\FullAnimal_KSChanMap", 
                    metadata["mouse"], metadata["probe"], metadata["loc"], "UnitMatch", "UnitMatch.mat")
    um = mat73.loadmat(um_path)
    paths = um["UMparam"]["KSDir"]
    exp_ids = {}
    for recses in sessions:
        exp_ids[recses] = get_exp_id(paths[recses-1], metadata["mouse"])
    return exp_ids, metadata