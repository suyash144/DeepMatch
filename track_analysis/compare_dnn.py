# compare different dnn models
import os, sys
import numpy as np
import pandas as pd
import scipy.io
import scipy.stats
import matplotlib.pyplot as plt
import h5py

if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))
    sys.path.insert(0, os.path.join(os.pardir, os.pardir))
    sys.path.insert(0, os.path.join(os.pardir))

def read_good_indices_from_mat(filepath):
    """
    Reads the Good_ID field from the clusinfo struct in a MATLAB .mat file.
    Args:
    - filepath (str): Path to the .mat file.
    """
    with h5py.File(filepath, 'r') as file:
        clusinfo = file['clusinfo']
        good_id = clusinfo['Good_ID']
        good_id = np.array(good_id)
        good_id = good_id.reshape(-1).astype(int)
        # return indices whose value is 1
        good_id = np.where(good_id == 1)[0]
    return good_id, len(good_id)

def find_new_index_from_ori_index(good_units_2D, ori_index):
    index = np.where(good_units_2D[:, 0] == ori_index)[0]
    if len(index) > 0:
        return good_units_2D[index, 1][0]
    else:
        return None

def transform_ori_array_to_new_array(ori_array, good_units_1, good_units_2):
    new_first_column = np.vectorize(lambda x: find_new_index_from_ori_index(good_units_1, x))(ori_array[:, 0])
    new_second_column = np.vectorize(lambda x: find_new_index_from_ori_index(good_units_2, x))(ori_array[:, 1])
    new_array = np.column_stack((new_first_column, new_second_column))
    return new_array

def load_match_pair_dnn_subfolder(mouse,model_name):
    results_save_folder = os.path.join(os.getcwd(),os.pardir,'results',model_name,'dnn')
    filename = 'match_pair_'+mouse+'.npy'
    match_pair = np.load(os.path.join(results_save_folder,filename))
    return match_pair

def find_unique_pairs(pred_1,pred_2):
    set_1 = set(map(tuple, pred_1))
    set_2 = set(map(tuple, pred_2))
    overlap = np.array(list(set_1 & set_2))
    unique_1 = np.array(list(set_1 - set_2))
    unique_2 = np.array(list(set_2 - set_1))
    return overlap,unique_1,unique_2

if __name__ == '__main__':
    base = os.path.join(os.getcwd(),os.pardir)

    # load data
    # mouse = 'AV008'
    # probe = 'Probe0'
    # location = '1'
    # dates = ['2022-03-12','2022-03-13']
    # experiment = 'exp1'

    mouse = 'AL032'
    probe = 'Probe0'
    location = '1'
    dates = ['2019-11-21','2019-11-22']
    experiment = 'exp1'
    date1 = dates[0]
    date2 = dates[1]

    model_name_1 = 'model_12_09_clip'
    model_name_2 = 'model_12_19_AE_clip'

    match_pair_dnn_1 = load_match_pair_dnn_subfolder(mouse,model_name_1)
    match_pair_dnn_2 = load_match_pair_dnn_subfolder(mouse,model_name_2)

    ori_overlap, ori_unique_1, ori_unique_2 = find_unique_pairs(match_pair_dnn_1,match_pair_dnn_2)
    print('ori_unique_1', ori_unique_1)
    print('ori_unique_2', ori_unique_2)
    print('ori_overlap', ori_overlap)
    print('len ori_unique_1', len(ori_unique_1))
    print('len ori_unique_2', len(ori_unique_2))
    print('len ori_overlap', len(ori_overlap))

    # # read good id
    # base_load = os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir)
    # base_load_mouse = os.path.join(base_load, 'original_folder', 'Data_UnitMatch', mouse)
    # path_date1 = os.path.join(base_load_mouse, date1, 'Probe0', '1', 'PreparedData.mat')
    # good_units_1, len_good_units_1 = read_good_indices_from_mat(path_date1)
    # path_date2 = os.path.join(base_load_mouse, date2, 'Probe0', '1', 'PreparedData.mat')
    # good_units_2, len_good_units_2 = read_good_indices_from_mat(path_date2)
    # neuron_num = len_good_units_1 + len_good_units_2
    # good_units = np.concatenate((good_units_1, good_units_2))
    # good_units_indices = np.arange(neuron_num) + 1
    # good_units_2D = np.zeros((neuron_num, 2))
    # good_units_2D[:, 0] = good_units
    # good_units_2D[:, 1] = good_units_indices
    # good_units_1 = good_units_2D[:len_good_units_1, :]
    # good_units_2 = good_units_2D[len_good_units_1:, :]

    # # transform match_pair_dnn to new index
    # match_pair_dnn_new_1 = transform_ori_array_to_new_array(match_pair_dnn_1, good_units_1, good_units_2)
    # match_pair_dnn_new_2 = transform_ori_array_to_new_array(match_pair_dnn_2, good_units_1, good_units_2)
