
### Compare the waveforms of predictions from unitmatch and dnn

import matplotlib.pyplot as plt
from matplotlib_venn import venn2,venn3
import os, sys
import numpy as np

if __name__ == '__main__':
    sys.path.insert(0, os.getcwd())
    print(sys.path)

from utils.myutil import *

# load match pair from unitmatch
def load_match_pair_unitmatch(mouse,session_pair):
    results_save_folder = os.path.join(os.getcwd(),os.pardir,'results','unitmatch',mouse)
    filename = 'match_pair_'+ 'session_pair' + session_pair + '.npy'
    match_pair = np.load(os.path.join(results_save_folder,filename))
    return match_pair

# load match pair from functional measures
def load_match_pair_func(mouse,session_pair):
    results_save_folder = os.path.join(os.getcwd(),os.pardir,'results','func',mouse)
    filename = 'match_pair_'+ 'session_pair' + session_pair + '.npy'
    match_pair = np.load(os.path.join(results_save_folder,filename))
    return match_pair

def load_match_pair_func_Sig(mouse,session_pair):
    results_save_folder = os.path.join(os.getcwd(),os.pardir,'results','func_Sig',mouse)
    filename = 'match_pair_'+ 'session_pair' + session_pair + '.npy'
    match_pair = np.load(os.path.join(results_save_folder,filename))
    return match_pair

def load_match_pair_func_Diff(mouse,session_pair):
    results_save_folder = os.path.join(os.getcwd(),os.pardir,'results','func_Diff',mouse)
    filename = 'match_pair_'+ 'session_pair' + session_pair + '.npy'
    match_pair = np.load(os.path.join(results_save_folder,filename))
    return match_pair

# load match pair from dnn
def load_match_pair_dnn(mouse,model_name,session_pair):
    results_save_folder = os.path.join(os.getcwd(),os.pardir,'results','dnn',model_name,mouse)
    filename = 'match_pair_'+ 'session_pair' + session_pair + '.npy'
    match_pair = np.load(os.path.join(results_save_folder,filename))
    return match_pair

def load_matches(mouse,session_pair,method, model_name=None):
    if method=="dnn":
        results_save_folder = os.path.join(os.path.dirname(os.getcwd()),'results', method, model_name, mouse)
    else:
        results_save_folder = os.path.join(os.getcwd(),os.pardir,'results',method,mouse)
    filename = 'match_pair_'+ 'session_pair' + session_pair + '.npy'
    match_pair = np.load(os.path.join(results_save_folder,filename))
    return match_pair

def plot_comparison_pairs_venn3(pred_unitmatch, pred_dnn, pred_func, mouse, model_name, session_pair, len_good_units_1, len_good_units_2, plot=True, save=False):
    """
    Plots three arrays for comparison.
    Args:
    pred_unitmatch (np.array): First array of predictions.
    pred_dnn (np.array): Second array of predictions.
    pred_func (np.array): Third array of predictions.
    """
    fontsize = 18
    fontsize_num = 16
    fontsize_title = 20
    # Convert arrays to sets of tuples for easier comparison
    set_unitmatch = set(map(tuple, pred_unitmatch))
    set_dnn = set(map(tuple, pred_dnn))
    set_func = set(map(tuple, pred_func))

    # Find overlapping and unique points
    overlap_all = set_unitmatch & set_dnn & set_func
    unique_unitmatch = set_unitmatch - set_dnn - set_func
    unique_dnn = set_dnn - set_unitmatch - set_func
    unique_func = set_func - set_unitmatch - set_dnn

    overlap_unitmatch_dnn = set_unitmatch & set_dnn - set_func
    overlap_unitmatch_func = set_unitmatch & set_func - set_dnn
    overlap_dnn_func = set_dnn & set_func - set_unitmatch

    # Counting the overlaps and unique
    overlap_count_all = len(overlap_all)
    unique_unitmatch_count = len(unique_unitmatch)
    unique_dnn_count = len(unique_dnn)
    unique_func_count = len(unique_func)
    overlap_unitmatch_dnn_count = len(overlap_unitmatch_dnn)
    overlap_unitmatch_func_count = len(overlap_unitmatch_func)
    overlap_dnn_func_count = len(overlap_dnn_func)
    
    if plot:
        plt.figure(figsize=(8, 8))
        venn = venn3(subsets=(unique_unitmatch_count, unique_dnn_count, overlap_unitmatch_dnn_count, 
                              unique_func_count, overlap_unitmatch_func_count, overlap_dnn_func_count, 
                              overlap_count_all),
                     set_labels=('UnitMatch', 'DNN', 'Func'))
        for text in venn.set_labels:
            text.set_fontsize(fontsize)
        for text in venn.subset_labels:
            if text:  # Check if the subset label is not None
                text.set_fontsize(fontsize_num)
        plt.title(f'Comparison of Predictions for {mouse} session pair {session_pair}', fontsize=fontsize_title)
        plt.text(0.95, 0.05, f'nDay1 = {len_good_units_1}\nnDay2 = {len_good_units_2}', fontsize=fontsize, 
                 horizontalalignment='right', verticalalignment='bottom', transform=plt.gcf().transFigure)
        plt.legend(frameon=False)
        fig_save_folder = os.path.join(os.getcwd(), os.pardir, 'figures', 'comparison', model_name)
        if not os.path.exists(fig_save_folder):
            os.makedirs(fig_save_folder)
        filename = f'venn3_match_pair_{mouse}_{session_pair}.png'
        plt.savefig(os.path.join(fig_save_folder, filename))
        plt.close()

    if save:
        results_save_folder = os.path.join(os.getcwd(), os.pardir, 'results', 'comparison', model_name)
        if not os.path.exists(results_save_folder):
            os.makedirs(results_save_folder)
        # Save the overlaps and uniques
        np.save(os.path.join(results_save_folder, f'overlap3_all_{mouse}_{session_pair}.npy'), np.array(list(overlap_all)))
        np.save(os.path.join(results_save_folder, f'unique3_unitmatch_{mouse}_{session_pair}.npy'), np.array(list(unique_unitmatch)))
        np.save(os.path.join(results_save_folder, f'unique3_dnn_{mouse}_{session_pair}.npy'), np.array(list(unique_dnn)))
        np.save(os.path.join(results_save_folder, f'unique3_func_{mouse}_{session_pair}.npy'), np.array(list(unique_func)))
        np.save(os.path.join(results_save_folder, f'overlap3_unitmatch_dnn_{mouse}_{session_pair}.npy'), np.array(list(overlap_unitmatch_dnn)))
        np.save(os.path.join(results_save_folder, f'overlap3_unitmatch_func_{mouse}_{session_pair}.npy'), np.array(list(overlap_unitmatch_func)))
        np.save(os.path.join(results_save_folder, f'overlap3_dnn_func_{mouse}_{session_pair}.npy'), np.array(list(overlap_dnn_func)))
    return np.array(list(overlap_all)), np.array(list(unique_unitmatch)), np.array(list(unique_dnn)), np.array(list(unique_func)), np.array(list(overlap_unitmatch_dnn)), np.array(list(overlap_unitmatch_func)), np.array(list(overlap_dnn_func))

if __name__ == '__main__':
    # train
    mode = 'test' # 'train' or 'test'
    # mouse = 'AL031'
    # probe = '19011116684'
    # location = '1'
    # dates = ['2019-10-01', '2019-10-02']
    # exp_ids = [r"_2019-10-01_1__2019-10-01_bank0_g0__2019-10-01_bank0_g0_imec0_PyKS_output", 
    #         r"_2019-10-02_1__2019-10-02_bank0_g0__2019-10-02_bank0_g0_imec0_PyKS_output"]
    # session_pair = '2'
    mouse = 'AL036'
    probe = '19011116882'
    location = '3'
    dates = ['2020-02-24', '2020-02-25']
    exps = ['_2020-08-04_ephys__2020-08-04_stripe240r1_natIm_g0_imec0_PyKS_output', 
            '_2020-08-05_ephys__2020-08-05_stripe240_natIm_g0__2020-08-05_stripe240_natIm_g0_imec0_PyKS_output']
    session_pair = '2'
    print('mouse', mouse, 'session_pair', session_pair)

    # read good id
    good_units_files_1,good_units_indices_1,good_units_files_2,good_units_indices_2 = load_mouse_data(mouse,probe,location,exps,mode)
    len_good_units_1 = len(good_units_files_1)
    len_good_units_2 = len(good_units_files_2)

    # load match pair from unitmatch
    match_pair_unitmatch = load_matches(mouse,session_pair, "unitmatch")
    # load match pair from functional measures
    # match_pair_func = load_match_pair_func(mouse,session_pair)
    # match_pair_func = load_match_pair_func_Sig(mouse,session_pair)
    match_pair_func = load_matches(mouse,session_pair, "func_Diff")
    # load match pair from dnn
    model_name = 'wentao'
    # model_name = '2024_2_13_ft_SpatioTemporalCNN_V2'
    match_pair_dnn = load_matches(mouse,session_pair, "dnn", model_name)

    # plot comparison (venn 3)
    overlap_all, unique_unitmatch, unique_dnn, unique_func, overlap_unitmatch_dnn, overlap_unitmatch_func, overlap_dnn_func = plot_comparison_pairs_venn3(match_pair_unitmatch, match_pair_dnn, match_pair_func, mouse, model_name, session_pair, len_good_units_1,len_good_units_2,plot=True, save=True)