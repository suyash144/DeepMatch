import os, sys
import numpy as np
import pandas as pd
import scipy.io
import scipy.stats
import matplotlib.pyplot as plt
import h5py


if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.pardir, os.pardir))
    sys.path.insert(0, os.path.join(os.pardir))

from compare_3m import *
from unitmatch import *
from utils.myutil import *

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

def save_as_matlab_cell(array, save_path, cell_name):
    """
    Saves a numpy array as a MATLAB cell array in a .mat file.
    Parameters:
    array (numpy.ndarray): The numpy array to be saved.
    filename (str): The name of the .mat file to save.
    """
    # Preparing the cell array format for MATLAB
    matlab_cell_array = np.empty((1, array.shape[0]), dtype=object)
    for i, row in enumerate(array):
        matlab_cell_array[0, i] = np.array(row)
    scipy.io.savemat(save_path, {cell_name: matlab_cell_array})

def save_transfer(ori_match_pair,mouse,session_pair,model_name,VennPart,good_units_1,good_units_2):
    '''
    VennPart: 'overlap_all','unique_unitmatch','unique_dnn','unique_func','overlap_unitmatch_dnn','overlap_unitmatch_func','overlap_dnn_func'
    '''
    transfered_match_pair = transform_ori_array_to_new_array(ori_match_pair, good_units_1, good_units_2)
    print('len transfered_match_pair',len(transfered_match_pair))
    # save
    results_save_folder = os.path.join(os.getcwd(),os.pardir,'results','transfer_index',model_name)
    if not os.path.exists(results_save_folder):
        os.makedirs(results_save_folder, exist_ok=True)
    filename = VennPart + '_'+mouse + '_'+session_pair+'.mat'
    save_path = os.path.join(results_save_folder,filename)
    cell_name = VennPart
    save_as_matlab_cell(transfered_match_pair, save_path, cell_name)
    return transfered_match_pair

def UM_based_transfer_index():
    pass

if __name__ == '__main__':
    mode = 'test' # 'train' or 'test'
    mouse = 'AV007'
    probe = '19011119461'
    location = '11'
    dates = ['2022-04-06', '2022-04-07']
    exps = ['AV007_2022-04-06_ActivePassive_g0_t0-imec1-ap', 
            'AV007_2022-04-07_ActivePassive_g0_t0-imec1-ap']
    session_pair = '2'
    print('mouse', mouse, 'session_pair', session_pair)

    # load match pair from unitmatch
    match_pair_unitmatch = load_match_pair_unitmatch(mouse,session_pair)
    # load match pair from functional measures
    # match_pair_func = load_match_pair_func(mouse,session_pair)
    match_pair_func = load_match_pair_func_Diff(mouse,session_pair)
    # load match pair from dnn
    model_name = '2024_2_13_ag_ft_SpatioTemporalCNN_V2'
    # model_name = '2024_2_13_ft_SpatioTemporalCNN_V2'
    match_pair_dnn = load_match_pair_dnn(mouse,model_name,session_pair)

    UniqueID, OriID, recses = read_UnitMatch_matfile_v73(mouse, session_pair)
    # the length of the UniqueID, OriID, recses is the same

    '''
    Now the elemment in match_pair are values in OriID.
    However, the element in Pairs in matlab actually are the index
    And previously we constrain that the first element in the bracket always from the first session
    So we can use OriID and recses to find the index, save that for later use in matlab
    When do this operation, pay attention to index difference between python and matlab
    '''
