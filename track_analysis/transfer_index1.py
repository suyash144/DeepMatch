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
    new_array = new_array.astype(int)
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
    # print('len transfered_match_pair',len(transfered_match_pair))
    # save
    results_save_folder = os.path.join(os.getcwd(),os.pardir,'results','transfer_index',model_name)
    if not os.path.exists(results_save_folder):
        os.makedirs(results_save_folder, exist_ok=True)
    filename = VennPart + '_'+mouse + '_'+session_pair+'.mat'
    save_path = os.path.join(results_save_folder,filename)
    cell_name = VennPart
    save_as_matlab_cell(transfered_match_pair, save_path, cell_name)
    return transfered_match_pair

if __name__ == '__main__':
    mode = 'test' # 'train' or 'test'
    # mouse = 'AV007'
    # probe = '19011119461'
    # location = '11'
    # dates = ['2022-04-06', '2022-04-07']
    # exps = ['AV007_2022-04-06_ActivePassive_g0_t0-imec1-ap', 
    #         'AV007_2022-04-07_ActivePassive_g0_t0-imec1-ap']
    # session_pair = '2'

    mouse = 'AL036'
    probe = '19011116882'
    location = '3'
    dates = ['2020-08-04', '2020-08-05']
    exps = ['AL036_2020-08-04_stripe240r1_natIm_g0_t0-imec0-ap', 
            'AL036_2020-08-05_stripe240_natIm_g0_t0-imec0-ap']
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

    good_units_files_1,good_units_indices_1,good_units_files_2,good_units_indices_2 = load_mouse_data(mouse,probe,location,dates,exps,mode)
    len_good_units_1 = len(good_units_files_1)
    len_good_units_2 = len(good_units_files_2)
    overlap_all, unique_unitmatch, unique_dnn, unique_func, overlap_unitmatch_dnn, overlap_unitmatch_func, overlap_dnn_func = plot_comparison_pairs_venn3(match_pair_unitmatch, match_pair_dnn, match_pair_func, mouse, model_name, session_pair,len_good_units_1,len_good_units_2, plot=False, save=False)

    neuron_num = len_good_units_1 + len_good_units_2
    good_units = np.concatenate((good_units_indices_1, good_units_indices_2))
    good_units_indices = np.arange(neuron_num) + 1
    good_units_2D = np.zeros((neuron_num, 2))
    good_units_2D[:, 0] = good_units
    good_units_2D[:, 1] = good_units_indices
    good_units_1 = good_units_2D[:len_good_units_1, :]
    good_units_2 = good_units_2D[len_good_units_1:, :]
    # np.set_printoptions(threshold=np.inf)
    # print('good_units_2D \n',good_units_2D)

    # sanity check with unitmatch index
    # trans_match_pair_unitmatch = save_transfer(match_pair_unitmatch,mouse,session_pair,model_name,'unitmatch',good_units_1,good_units_2)
    # put trans_match_pair_unitmatch and match_pair_unitmatch together, two [N,2] into [N,4]
    # combined_match_pair_unitmatch = np.concatenate((match_pair_unitmatch, trans_match_pair_unitmatch), axis=1)
    # print('combined_match_pair_unitmatch \n',combined_match_pair_unitmatch)
    # print('len combined_match_pair_unitmatch \n',len(combined_match_pair_unitmatch))
    # trans_match_pair_unitmatch = trans_match_pair_unitmatch.tolist()
    # print('trans_match_pair_UM',trans_match_pair_unitmatch)

    # ## save transfered index
    # print('save transfered index')
    # # overlap_all
    # trans_overlap_all = save_transfer(overlap_all,mouse,session_pair,model_name,'overlap_all',good_units_1,good_units_2)
    # # unique_unitmatch
    # trans_unique_unitmatch = save_transfer(unique_unitmatch,mouse,session_pair,model_name,'unique_unitmatch',good_units_1,good_units_2)
    # # print('len unique_unitmatch',len(unique_unitmatch))
    # # print('unique_unitmatch',unique_unitmatch)
    # # print('trans unique unitmatch',trans_unique_unitmatch)
    # # unique_dnn
    # trans_unique_dnn = save_transfer(unique_dnn,mouse,session_pair,model_name,'unique_dnn',good_units_1,good_units_2)
    # # print('len unique_dnn',len(unique_dnn))
    # # print('unique_dnn',unique_dnn)
    # # unique_func
    # trans_unique_func = save_transfer(unique_func,mouse,session_pair,model_name,'unique_func',good_units_1,good_units_2)
    # print('len unique_func',len(unique_func))
    # # overlap_unitmatch_dnn
    # trans_overlap_unitmatch_dnn = save_transfer(overlap_unitmatch_dnn,mouse,session_pair,model_name,'overlap_unitmatch_dnn',good_units_1,good_units_2)
    # # overlap_unitmatch_func
    # trans_overlap_unitmatch_func = save_transfer(overlap_unitmatch_func,mouse,session_pair,model_name,'overlap_unitmatch_func',good_units_1,good_units_2)
    # # overlap_dnn_func
    # trans_overlap_dnn_func = save_transfer(overlap_dnn_func,mouse,session_pair,model_name,'overlap_dnn_func',good_units_1,good_units_2)
    # print('save transfered index done')

    print('match_pair_dnn', match_pair_dnn)