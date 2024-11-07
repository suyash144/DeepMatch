import os, sys
import numpy as np
import pandas as pd
import scipy.io
import scipy.stats
import matplotlib.pyplot as plt
import h5py


if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.pardir, os.pardir))
    sys.path.insert(0, os.getcwd())

from utils.myutil import *

def read_MatchTable_from_csv(mouse, session_pair):
    base_load = os.path.dirname(os.getcwd())
    base_load_mouse = os.path.join(base_load, 'Save_UnitMatch', mouse,'session_pair_'+str(session_pair))
    base_load_data = os.path.join(base_load_mouse, 'MatchTable.csv')
    MatchTable = pd.read_csv(base_load_data)
    return MatchTable

def read_UnitMatch_matfile_v73(mouse, session_pair):
    # Open the HDF5 MATLAB file
    base_load = os.path.dirname(os.getcwd())
    base_load_mouse = os.path.join(base_load, 'Save_UnitMatch', mouse,'session_pair_'+str(session_pair))
    base_load_data = os.path.join(base_load_mouse, 'UnitMatch.mat')
    with h5py.File(base_load_data, 'r') as f:
        # Access the 'UniqueIDConversion' struct
        unique_id_conversion = f['UniqueIDConversion']
        # Extract the fields from the struct
        unique_id = unique_id_conversion['UniqueID'][()].flatten()  # Assuming UniqueID is stored as 1xN
        original_clus_id = unique_id_conversion['OriginalClusID'][()].flatten()  # Assuming OriginalClusID is stored as Nx1
        recses_all = unique_id_conversion['recsesAll'][()].flatten()  # Assuming recsesAll is stored as Nx1
        good_id = unique_id_conversion['GoodID'][()].flatten()  # Assuming GoodID is stored as 1xN
        unique_id = np.round(unique_id).astype(int) # Convert to int
        original_clus_id = original_clus_id.astype(int)
        recses_all = recses_all.astype(int)
        good_id = good_id.astype(bool)
        UniqueID = unique_id[good_id] # the unique ID (across sessions), begins from 1
        OriID = original_clus_id[good_id] # the original ID (for each session), begins from 0
        recses = recses_all[good_id]
        # return unique_id, original_clus_id, recses_all, good_id
        return UniqueID, OriID, recses
    

def visualize_prob_matrix_unitmatch(probs_matrix_1, probs_matrix_2, mouse, session_pair):
    fontsize = 24
    fontsize_title = 28 
    tick_params_dict = {'axis': 'both', 'which': 'both', 'labelsize': 20, 'direction': 'out'}
    fig, ax = plt.subplots(1,2, figsize = (14,10))
    img0 = ax[0].imshow(probs_matrix_1, cmap='viridis')
    ax[0].set_title('Probability Matrix 11-22', fontsize = fontsize)
    ax[0].set_xlabel('Day 2', fontsize = fontsize)
    ax[0].set_ylabel('Day 1', fontsize = fontsize)
    ax[0].tick_params(**tick_params_dict)
    # fig.colorbar(img0, ax=ax[0])
    img1 = ax[1].imshow(probs_matrix_2, cmap='viridis')
    ax[1].set_title('Probability Matrix 12-21', fontsize = fontsize)
    ax[1].set_xlabel('Day 2', fontsize = fontsize)
    ax[1].set_ylabel('Day 1', fontsize = fontsize)
    ax[1].tick_params(**tick_params_dict)
    # fig.colorbar(img1, ax=ax[1])
    fig.suptitle('Probability Matrix across days', fontsize = fontsize_title)
    plt.colorbar(img0, ax=ax, orientation='horizontal')
    fig_save_folder = os.path.join(os.getcwd(),os.pardir,'figures','unitmatch',mouse)
    if not os.path.exists(fig_save_folder):
        os.makedirs(fig_save_folder)
    filename = 'prob_matrix_across_days_'+'session_pair'+ str(session_pair) +'.png'
    plt.savefig(os.path.join(fig_save_folder,filename))
    plt.close()


def visualize_prob_matrix_all_unitmatch(probs_matrix_11, probs_matrix_12, probs_matrix_21, probs_matrix_22, mouse, session_pair):
    '''
    Inputs:
        probs_matrix_11: shape (num_units_day1, num_units_day1)
        probs_matrix_12: shape (num_units_day1, num_units_day2)
        probs_matrix_21: shape (num_units_day1, num_units_day2)
        probs_matrix_22: shape (num_units_day2, num_units_day2)
    '''
    # combine into a big matrix, shape (num_units_day1+num_units_day2, num_units_day1+num_units_day2)
    sim_matrix_1 = np.concatenate((probs_matrix_11, probs_matrix_12), axis = 1)
    probs_matrix_21_T = probs_matrix_21.T # shape (num_units_day2, num_units_day1)
    sim_matrix_2 = np.concatenate((probs_matrix_21_T, probs_matrix_22), axis = 1)
    sim_matrix = np.concatenate((sim_matrix_1, sim_matrix_2), axis = 0)
    fontsize = 24
    fontsize_title = 28 
    tick_params_dict = {'axis': 'both', 'which': 'both', 'labelsize': 20, 'direction': 'out'}
    fig, ax = plt.subplots(1,1, figsize = (14,10))
    img0 = ax.imshow(sim_matrix,cmap = 'viridis')
    # draw a horizontal and vertical line to separate day 1 and day 2
    ax.axvline(x=probs_matrix_11.shape[1], color = 'black', linewidth = 3)
    ax.axhline(y=probs_matrix_11.shape[0], color = 'black', linewidth = 3)
    ax.set_title('Probability Matrix across days', fontsize = fontsize_title)
    ax.set_xlabel('Day 1 and 2', fontsize = fontsize)
    ax.set_ylabel('Day 2 and 1', fontsize = fontsize)
    ax.tick_params(**tick_params_dict)
    # fig.colorbar(img0, ax=ax[0])
    plt.colorbar(img0, ax=ax, orientation='horizontal')
    fig_save_folder = os.path.join(os.getcwd(),os.pardir,'figures','unitmatch',mouse)
    if not os.path.exists(fig_save_folder):
        os.makedirs(fig_save_folder)
    filename = 'prob_matrix_across_days_all_'+ mouse+'_' + 'session_pair_' + session_pair +'.png'
    plt.savefig(os.path.join(fig_save_folder,filename))
    plt.close()

def matlab_get_pairs(UniqueID):
    '''
    Here, each element in the Pairs list represents indices 
    in the UniqueID array where the value of UniqueID matches 
    a particular unique value found in the entire UniqueID array. 
    But, the indices are 1-based, 
    so we mius 1 to each index later when we use it for indexing in python.
    '''    
    uId = np.unique(UniqueID)  # Equivalent to MATLAB's `unique`
    # Step 2: For each unique element in uId, find the indices in UniqueID where UniqueID equals the element
    # This step uses a list comprehension as an equivalent to MATLAB's `arrayfun`
    Pairs = [np.where(UniqueID == x)[0]+1 for x in uId] # +1 because matlab index starts from 1
    # Step 3: Remove elements from Pairs that contain only one index
    # This uses another list comprehension to filter out any array in Pairs with length 1
    Pairs = [pair for pair in Pairs if len(pair) > 1]
    return Pairs

def organize_pairs_from_matlab(Pairs, UniqueID, OriID, recses, mouse, session_pair):
    '''
    Organize the pairs from matlab into a list of lists
    '''
    match_pairs = []
    for pair in Pairs:
        for i in range(len(pair)-1): # len(pair) can be 2 or more, more is because a neuron can be matched to multiple neurons
            py_index_i = pair[i]-1
            recses_i = recses[py_index_i]
            if recses_i == 2:
                continue
            oriid_i = OriID[py_index_i]
            for j in range(i+1, len(pair)):
                py_index_j = pair[j]-1
                recses_j = recses[py_index_j]
                if recses_j == 1:
                    continue
                oriid_j = OriID[py_index_j]
                match_pairs.append([oriid_i, oriid_j])
    match_pairs = np.array(match_pairs)
    results_save_folder = os.path.join(os.getcwd(),os.pardir,'results','unitmatch',mouse)
    if not os.path.exists(results_save_folder):
        os.makedirs(results_save_folder)
    filename = 'match_pair_'+ 'session_pair'+ str(session_pair) +'.npy'
    np.save(os.path.join(results_save_folder,filename),match_pairs)
    return match_pairs

# get pair of units with probability matrix
def get_match_pair_unitmatch(probs_matrix_1, probs_matrix_2,good_units_1,good_units_2,mouse,session_pair,thr = 0.5):
    '''
    thr: threshold, mat_1(i,j) > thr and mat_2(i,j) > thr
    good_units: real indices of indices in probs_matrix
    '''
    match_pairs = []
    for i in range(probs_matrix_1.shape[0]):
        for j in range(probs_matrix_1.shape[1]):
            average_prob = (probs_matrix_1[i,j] + probs_matrix_2[i,j])/2
            # if average_prob > thr:
                # match_pairs.append([good_units_1[i],good_units_2[j]])
            if probs_matrix_1[i,j] >= thr and probs_matrix_2[i,j] >= thr:
                match_pairs.append([good_units_1[i],good_units_2[j]])
    match_pairs = np.array(match_pairs)
    results_save_folder = os.path.join(os.getcwd(),os.pardir,'results','unitmatch',mouse)
    if not os.path.exists(results_save_folder):
        os.makedirs(results_save_folder)
    filename = 'match_pair_'+ 'session_pair'+ str(session_pair) +'.npy'
    np.save(os.path.join(results_save_folder,filename),match_pairs)
    return match_pairs

# # Revised function to ensure UID uniqueness across all match pairs
def get_unique_match_pair_unitmatch(MatchTable,mouse,session_pair,thr = 0.0001):
    # filter MatchTable
    MatchTable = MatchTable[['ID1', 'ID2', 'RecSes1', 'RecSes2', 'UID1', 'UID2','MatchProb']]
    # MatchTable = MatchTable[MatchTable['MatchProb'] >= thr]
    MatchTable = MatchTable[MatchTable['RecSes1'] != MatchTable['RecSes2']]
    # get correct match pairs
    match_pairs = []  # List to store correct match pairs
    used_uids = []
    for index, row in MatchTable.iterrows():
        id1 = row['ID1']
        id2 = row['ID2']
        recses1 = row['RecSes1']
        recses2 = row['RecSes2']
        UID1 = row['UID1']
        UID2 = row['UID2']
        if UID1 != UID2:
            continue
        # Check if the pair has been used
        if [id1, id2] in used_uids or [id2, id1] in used_uids:
            continue
        if recses1 == 1 and recses2 == 2:
            # Find the reverse pair
            reverse_pair = MatchTable[(MatchTable['ID1'] == id2) & (MatchTable['ID2'] == id1) & 
                                      (MatchTable['RecSes1'] == recses2) & (MatchTable['RecSes2'] == recses1) &
                          (MatchTable['UID1'] == UID2) & (MatchTable['UID2'] == UID1)]
        # If reverse pair exists, add to match pairs and mark UIDs as used
        if not reverse_pair.empty:
            match_pairs.append([id1, id2])
            used_uids.append([id1, id2])
            used_uids.append([id2, id1])
    match_pairs = np.array(match_pairs)
    results_save_folder = os.path.join(os.getcwd(),os.pardir,'results','unitmatch',mouse)
    if not os.path.exists(results_save_folder):
        os.makedirs(results_save_folder)
    filename = 'match_pair_'+ 'session_pair'+ str(session_pair) +'.npy'
    np.save(os.path.join(results_save_folder,filename),match_pairs)
    return match_pairs


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
    dates = ['2020-02-24', '2020-02-25']
    exps = ['AL036_2020-02-24_stripe240_NatIm_g0_t0-imec0-ap', 
            'AL036_2020-02-25_stripe240_NatIm_g0_t0-imec0-ap']
    session_pair = '1'
    print('mouse', mouse, 'session_pair', session_pair)

    # MatchTable = read_MatchTable_from_csv(mouse, session_pair)
    # # read good id
    # good_units_files_1,good_units_indices_1,good_units_files_2,good_units_indices_2 = load_mouse_data(mouse,probe,location,dates,exps,mode)
    # len_good_units_1 = len(good_units_files_1)
    # len_good_units_2 = len(good_units_files_2)
    # neuron_num = len_good_units_1 + len_good_units_2 # day 1 neuron num + day 2 neuron num

    # # first method
    # MatchProb = MatchTable['MatchProb'].values    
    # MatchProb = MatchProb.reshape(neuron_num, neuron_num)
    # MatchProb_11 = MatchProb[:len_good_units_1, :len_good_units_1]
    # MatchProb_12 = MatchProb[:len_good_units_1, len_good_units_1:]
    # MatchProb_21 = MatchProb[len_good_units_1:, :len_good_units_1].T
    # MatchProb_22 = MatchProb[len_good_units_1:, len_good_units_1:]

    # # original plot
    # # visualize_prob_matrix_unitmatch(MatchProb_12, MatchProb_21, mouse, session_pair)
    # # rearranged plot by position
    # day1_MaxSitepos, day2_MaxSitepos = get_day_MaxSitepos(good_units_files_1,good_units_files_2)
    # day1_sorted_indices = sort_neurons_by_position(day1_MaxSitepos)
    # day2_sorted_indices = sort_neurons_by_position(day2_MaxSitepos)
    # reProbs_matrix_11, reProbs_matrix_12, reProbs_matrix_21, reProbs_matrix_22 = rearrange_four_matrix(MatchProb_11, MatchProb_12, MatchProb_21, MatchProb_22, day1_sorted_indices, day2_sorted_indices)
    # visualize_prob_matrix_all_unitmatch(reProbs_matrix_11, reProbs_matrix_12, reProbs_matrix_21, reProbs_matrix_22, mouse, session_pair)
    # del reProbs_matrix_11, reProbs_matrix_12, reProbs_matrix_21, reProbs_matrix_22

    # match_pairs = get_match_pair_unitmatch(MatchProb_12, MatchProb_21, good_units_indices_1, good_units_indices_2, mouse, session_pair,thr=0.5)
    
    # second method
    # match_pairs = get_unique_match_pair_unitmatch(MatchTable, mouse, session_pair, thr=0.0001)

    # third method
    # read UniqueID, OriID, recses
    UniqueID, OriID, recses = read_UnitMatch_matfile_v73(mouse, session_pair)
    Pairs = matlab_get_pairs(UniqueID)
    # Pairs_list = [arr.tolist() for arr in Pairs]
    # print('Pairs_list', Pairs_list)
    match_pairs = organize_pairs_from_matlab(Pairs, UniqueID, OriID, recses,mouse, session_pair)
    match_pairs = match_pairs.tolist()
    print('match_pairs', match_pairs)
    print('len', len(match_pairs))

