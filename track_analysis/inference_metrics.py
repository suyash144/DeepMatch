# Wentao Qiu, 2024-2-12
# qiuwentao1212@gmail.com

'''
Calculate the inference metrics for the tracking results
e.g., False Positive Rate for within-day data
'''

import matplotlib.pyplot as plt
import os, sys
import h5py

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment

if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.pardir, os.pardir))
    sys.path.insert(0, os.path.join(os.pardir))
    from dnn import *
else:
    from track_analysis.dnn import *

from models.mymodel import *
from utils.myutil import *
from utils.losses import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 

def visualize_re_ft_sim_matrix_all(sim_matrix_11, sim_matrix_12, sim_matrix_21, sim_matrix_22, mouse, model_name,session_pair):
    # combine into a big matrix, shape (num_units_day1+num_units_day2, num_units_day1+num_units_day2)
    sim_matrix_1 = np.concatenate((sim_matrix_11, sim_matrix_12), axis = 1)
    sim_matrix_21_T = sim_matrix_21.T # shape (num_units_day2, num_units_day1)
    sim_matrix_2 = np.concatenate((sim_matrix_21_T, sim_matrix_22), axis = 1)
    sim_matrix = np.concatenate((sim_matrix_1, sim_matrix_2), axis = 0)
    fontsize = 24
    fontsize_title = 28 
    tick_params_dict = {'axis': 'both', 'which': 'both', 'labelsize': 20, 'direction': 'out'}
    fig, ax = plt.subplots(1,1, figsize = (14,10))
    img0 = ax.imshow(sim_matrix,cmap = 'viridis')
    # draw a horizontal and vertical line to separate day 1 and day 2
    ax.axvline(x=sim_matrix_11.shape[1], color = 'black', linewidth = 3)
    ax.axhline(y=sim_matrix_11.shape[0], color = 'black', linewidth = 3)
    ax.set_title('Similarity Matrix across days', fontsize = fontsize_title)
    ax.set_xlabel('Day 1 and 2', fontsize = fontsize)
    ax.set_ylabel('Day 2 and 1', fontsize = fontsize)
    ax.tick_params(**tick_params_dict)
    # fig.colorbar(img0, ax=ax[0])
    plt.colorbar(img0, ax=ax, orientation='horizontal')
    fig_save_folder = os.path.join(os.getcwd(),os.pardir,'figures','dnn',model_name,mouse,'session_pair'+session_pair)
    if not os.path.exists(fig_save_folder):
        os.makedirs(fig_save_folder, exist_ok=True)
    filename = 're_ft_similarity_matrix_across_days_'+ mouse +'.png'
    plt.savefig(os.path.join(fig_save_folder,filename))
    plt.close()


def tracking_method_for_inference_within_day(sim_matrix_within_day,MaxSitepos_within_day,good_units_within_day,mouse,model_name,session_pair,NoMatchThr=0.5):
    pred_pairs = []
    '''
    tracking algorithm with 'sparsification'
    If the channel position is far away, 
    that element of the graph describing the similarity 
    between a pair of neurons is set to zero and sparsened out. 
    '''
    dist_matrix_within_days = get_Sitepos_dist(MaxSitepos_within_day,MaxSitepos_within_day)
    # sparsification
    # cosine similarity is between -1 and 1, so we set them to -1
    sim_matrix_within_day[dist_matrix_within_days == -1] = -1 
    sim_matrix_within_day[dist_matrix_within_days > 40] = -1
    # Hungarian matching
    matches = hungarian_matching_with_threshold(sim_matrix_within_day, similarity_threshold=NoMatchThr)
    # transfer to neuron index
    for pair in matches:
        pred_pairs.append([good_units_within_day[pair[0]],good_units_within_day[pair[1]]])
    pred_pairs = np.array(pred_pairs)
    # supposed to be correct match pairs
    correct_match_pairs = np.array([[good_units_within_day[i],good_units_within_day[i]] for i in range(len(good_units_within_day))])
    return pred_pairs, correct_match_pairs

def cal_detection_metrics(sim_matrix):
    n = sim_matrix.shape[0]  # Assuming sim_matrix is a square matrix (N x N)
    fp = 0  # False Positives
    fn = 0  # False Negatives
    tp = 0  # True Positives
    # Iterate over each row to find the max similarity score and its index
    for i in range(n):
        row = sim_matrix[i, :]
        max_sim_index = np.argmax(row)  # Index of the maximum similarity score in the row
        if max_sim_index == i:
            tp += 1  # Diagonal element was chosen, so it's a true positive
        else:
            # Non-diagonal element was chosen, so it's both a false positive and a false negative in this context
            fp += 1
            fn += 1
    fp_rate = fp / (fp + tp) if (fp + tp) > 0 else 0
    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    return fp_rate, fn_rate

def cal_detection_metrics_2(match_pair, correct_match_pair):
    # Convert arrays to sets of tuples for easier comparison
    match_pair_set = set(map(tuple, match_pair))
    correct_match_pair_set = set(map(tuple, correct_match_pair))
    # True positives are pairs that are in both match_pair and correct_match_pair
    true_positives = len(match_pair_set.intersection(correct_match_pair_set))
    # False negatives are pairs that are in correct_match_pair but not in match_pair
    false_negatives = len(correct_match_pair_set.difference(match_pair_set))
    # False negative rate is the ratio of false negatives to the total in correct_match_pair
    false_negative_rate = false_negatives / len(correct_match_pair_set)
    return false_negative_rate

if __name__ == '__main__':
    base = os.path.join(os.getcwd(),os.pardir,os.pardir)
    # load model
    # model_name = '2024_2_7_ft_SpatioTemporalCNN_V2_2'
    model_name = '2024_2_13_ag_ft_SpatioTemporalCNN_V2'
    # model_name = '2024_2_13_ft_SpatioTemporalCNN_V2'
    ckpt_path = os.path.join('ModelExp','experiments', model_name, 'ckpt', 'ckpt_epoch_49')
    ckpt_path = os.path.join(base, ckpt_path)
    model = load_model(ckpt_path, device)

    # load data
    mode = 'test' # 'train' or 'test'
    mouse = 'EB037'
    probe = 'Probe0'
    location = '1'
    dates = ['2024-02-06', '2024-02-07']
    exps = ['2024-02-06_exp1', 
            '2024-02-07_exp1']
    session_pair = '1'

    good_units_files_1,good_units_indices_1,good_units_files_2,good_units_indices_2 = load_mouse_data(mouse,probe,location,dates,exps,mode)
    rep_day1_first_half,rep_day1_second_half,rep_day2_first_half,rep_day2_second_half = get_representation_dnn(model,good_units_files_1,good_units_files_2)
    sim_matrix_11, sim_matrix_12, sim_matrix_21, sim_matrix_22 = get_sim_matrix_all(rep_day1_first_half,rep_day1_second_half,rep_day2_first_half,rep_day2_second_half)
    del rep_day1_first_half,rep_day1_second_half,rep_day2_first_half,rep_day2_second_half

    day1_MaxSitepos, day2_MaxSitepos = get_day_MaxSitepos(good_units_files_1,good_units_files_2)
    
    # method 1
    # # filter 
    # ft_sim_matrix_11 = simple_filter_matrix(sim_matrix_11,day1_MaxSitepos,day1_MaxSitepos)
    # ft_sim_matrix_22 = simple_filter_matrix(sim_matrix_22,day2_MaxSitepos,day2_MaxSitepos)
    # ft_sim_matrix_12 = simple_filter_matrix(sim_matrix_12,day1_MaxSitepos,day2_MaxSitepos)
    # ft_sim_matrix_21 = simple_filter_matrix(sim_matrix_21,day1_MaxSitepos,day2_MaxSitepos)
    # day1_sorted_indices = sort_neurons_by_position(day1_MaxSitepos)
    # day2_sorted_indices = sort_neurons_by_position(day2_MaxSitepos)
    # re_ft_sim_matrix_11 = rearrange_matrix(ft_sim_matrix_11,day1_sorted_indices,day1_sorted_indices)
    # re_ft_sim_matrix_22 = rearrange_matrix(ft_sim_matrix_22,day2_sorted_indices,day2_sorted_indices)
    # re_ft_sim_matrix_12 = rearrange_matrix(ft_sim_matrix_12,day1_sorted_indices,day2_sorted_indices)
    # re_ft_sim_matrix_21 = rearrange_matrix(ft_sim_matrix_21,day1_sorted_indices,day2_sorted_indices)

    # # visualize_re_ft_sim_matrix_all(re_ft_sim_matrix_11, re_ft_sim_matrix_12, re_ft_sim_matrix_21, re_ft_sim_matrix_22, mouse, model_name,session_pair)

    # # calculate the detection metrics
    # fp_rate_11, fn_rate_11 = cal_detection_metrics(re_ft_sim_matrix_11)
    # fp_rate_22, fn_rate_22 = cal_detection_metrics(re_ft_sim_matrix_22)

    # method 2
    match_pair_day1, correct_match_pairs_day1 = tracking_method_for_inference_within_day(sim_matrix_11,day1_MaxSitepos,good_units_indices_1,mouse,model_name,session_pair)
    match_pair_day2,correct_match_pairs_day2 = tracking_method_for_inference_within_day(sim_matrix_22,day2_MaxSitepos,good_units_indices_2,mouse,model_name,session_pair)

    fn_rate_11 = cal_detection_metrics_2(match_pair_day1, correct_match_pairs_day1)
    fn_rate_22 = cal_detection_metrics_2(match_pair_day2, correct_match_pairs_day2)

    # print control, Print to the second decimal place
    print('mouse:',mouse,'session_pair:',session_pair)
    print('fn_rate_11:',round(fn_rate_11,2))
    print('fn_rate_22:',round(fn_rate_22,2))

