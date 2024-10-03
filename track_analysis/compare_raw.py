
### Compare the waveforms of predictions from unitmatch and raw

import matplotlib.pyplot as plt
from matplotlib_venn import venn2,venn3
import os, sys

import numpy as np
import torch

if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.pardir, os.pardir))
    sys.path.insert(0, os.getcwd())

from utils.myutil import *

# load match pair from raw prediction (null model)
def load_match_pair_raw(mouse,session_pair):
    results_save_folder = os.path.join(os.getcwd(),os.pardir,'results','raw',mouse)
    filename = 'match_pair_'+ 'session_pair' + str(session_pair) + '.npy'
    match_pair = np.load(os.path.join(results_save_folder,filename))
    return match_pair

# load match pair from unitmatch
def load_match_pair_unitmatch(mouse,session_pair):
    results_save_folder = os.path.join(os.getcwd(),os.pardir,'results','unitmatch',mouse)
    # filename = 'match_pair_'+mouse+'.npy'
    filename = 'match_pair_'+ 'session_pair' + str(session_pair) + '.npy'
    match_pair = np.load(os.path.join(results_save_folder,filename))
    return match_pair

# load match pair from functional measures
def load_match_pair_func(mouse,session_pair):
    results_save_folder = os.path.join(os.getcwd(),os.pardir,'results','func',mouse)
    # filename = 'match_pair_'+mouse+'.npy'
    filename = 'match_pair_'+ 'session_pair' + str(session_pair) + '.npy'
    match_pair = np.load(os.path.join(results_save_folder,filename))
    return match_pair

def plot_comparison_pairs_venn3_raw(pred_unitmatch, pred_raw, pred_func, mouse, plot=True, save=False):
    """
    Plots three arrays for comparison.
    Args:
    pred_unitmatch (np.array): First array of predictions.
    pred_raw (np.array): Second array of predictions.
    pred_func (np.array): Third array of predictions.
    """
    # Convert arrays to sets of tuples for easier comparison
    set_unitmatch = set(map(tuple, pred_unitmatch))
    set_raw = set(map(tuple, pred_raw))
    set_func = set(map(tuple, pred_func))

    # Find overlapping and unique points
    overlap_all = set_unitmatch & set_raw & set_func
    unique_unitmatch = set_unitmatch - set_raw - set_func
    unique_raw = set_raw - set_unitmatch - set_func
    unique_func = set_func - set_unitmatch - set_raw

    overlap_unitmatch_raw = set_unitmatch & set_raw - set_func
    overlap_unitmatch_func = set_unitmatch & set_func - set_raw
    overlap_raw_func = set_raw & set_func - set_unitmatch

    # Counting the overlaps and unique
    overlap_count_all = len(overlap_all)
    unique_unitmatch_count = len(unique_unitmatch)
    unique_raw_count = len(unique_raw)
    unique_func_count = len(unique_func)
    overlap_unitmatch_raw_count = len(overlap_unitmatch_raw)
    overlap_unitmatch_func_count = len(overlap_unitmatch_func)
    overlap_raw_func_count = len(overlap_raw_func)
    
    if plot:
        plt.figure(figsize=(8, 8))
        venn = venn3(subsets=(unique_unitmatch_count, unique_raw_count, overlap_unitmatch_raw_count, 
                              unique_func_count, overlap_unitmatch_func_count, overlap_raw_func_count, 
                              overlap_count_all),
                     set_labels=('UnitMatch', 'Raw', 'Func'))
        plt.title(f'Comparison of Predictions for {mouse}')
        plt.legend()
        fig_save_folder = os.path.join(os.getcwd(), os.pardir, 'figures', 'comparison','raw')
        if not os.path.exists(fig_save_folder):
            os.makedirs(fig_save_folder)
        filename = f'venn3_match_pair_{mouse}.png'
        plt.savefig(os.path.join(fig_save_folder, filename))
        plt.close()

    if save:
        results_save_folder = os.path.join(os.getcwd(), os.pardir, 'results', 'comparison','raw')
        if not os.path.exists(results_save_folder):
            os.makedirs(results_save_folder)
        # Save the overlaps and uniques
        np.save(os.path.join(results_save_folder, f'overlap3_all_{mouse}.npy'), np.array(list(overlap_all)))
        np.save(os.path.join(results_save_folder, f'unique3_unitmatch_{mouse}.npy'), np.array(list(unique_unitmatch)))
        np.save(os.path.join(results_save_folder, f'unique3_raw_{mouse}.npy'), np.array(list(unique_raw)))
        np.save(os.path.join(results_save_folder, f'unique3_func_{mouse}.npy'), np.array(list(unique_func)))
        np.save(os.path.join(results_save_folder, f'overlap3_unitmatch_raw_{mouse}.npy'), np.array(list(overlap_unitmatch_raw)))
        np.save(os.path.join(results_save_folder, f'overlap3_unitmatch_func_{mouse}.npy'), np.array(list(overlap_unitmatch_func)))
        np.save(os.path.join(results_save_folder, f'overlap3_raw_func_{mouse}.npy'), np.array(list(overlap_raw_func)))

    return np.array(list(overlap_all)), np.array(list(unique_unitmatch)), np.array(list(unique_raw)), np.array(list(unique_func)), np.array(list(overlap_unitmatch_raw)), np.array(list(overlap_unitmatch_func)), np.array(list(overlap_raw_func))


'''
### 384 channel version
def plot_paired_channel_activity_shank(pair_indices,mouse,probe,location,dates,experiment,plot=True):
    base_path = os.path.join(os.getcwd(),os.pardir, os.pardir)
    date_1 = dates[0]
    date_2 = dates[1]
    arranged_channel_index,shank_index_array = arranged_channel_index()
    path_waveform_1 = os.path.join(base_path, 'DATA_UnitMatch', mouse, probe,location, date_1, experiment,'RawWaveforms')
    path_waveform_2 = os.path.join(base_path, 'DATA_UnitMatch', mouse, probe,location, date_2, experiment,'RawWaveforms')

    unit_1_name = f"Unit{pair_indices[0]}_RawSpikes.npy"
    unit_1_data = np.load(os.path.join(path_waveform_1, unit_1_name)) # (82,384,2)
    unit_1_shank = find_belonged_channel(unit_1_data,shank_index_array)
    unit_2_name = f"Unit{pair_indices[1]}_RawSpikes.npy"
    unit_2_data = np.load(os.path.join(path_waveform_2, unit_2_name)) # (82,384,2)
    unit_2_shank = find_belonged_channel(unit_2_data,shank_index_array)

    if unit_1_shank != unit_2_shank:
        print('Warning: two units are not in the same shank')
        return 0
    if plot:
        unit_shank = unit_1_shank
        max_channel_1 = det_max_channel(unit_1_data)
        max_channel_2 = det_max_channel(unit_2_data)
        max_channel = (max_channel_1 + max_channel_2) // 2

        fig, axs = plt.subplots(48, 2, figsize=(8, 16), sharex=True, sharey=True)
        channel_start_idx = unit_shank * 96
        channel_end_idx = (unit_shank + 1) * 96
        for channel_index in range(channel_start_idx, channel_end_idx):
            row, col = get_subplot_position(channel_index)
            col = col - 2 * unit_shank
            axs[row, col].plot(unit_1_data[:,channel_index,0],color=(220/255, 64/255, 26/255))
            axs[row, col].plot(unit_2_data[:,channel_index,1],color=(78/255, 183/255, 233/255))
        axs[row, col].axis('off')  # Optional: Hide axes
        fig_save_folder = os.path.join(os.getcwd(),os.pardir,'figures','comparison')
        if not os.path.exists(fig_save_folder):
            os.makedirs(fig_save_folder)
        filename = 'comparison_unit_activity_'+mouse+'_unit'+str(pair_indices[0])+'_unit'+str(pair_indices[1])+'.png'
        plt.savefig(os.path.join(fig_save_folder,filename))
        plt.close()
        print('Unit Activity Image Saved')
        return 1
'''

### 96 channel version
def plot_paired_channel_activity_shank(pair_indices,mouse,probe,location,dates,experiment,method,plot=True):
    base_path = os.path.join(os.getcwd(),os.pardir, os.pardir)
    date_1 = dates[0]
    date_2 = dates[1]
    channel_index_array,shank_index_array = arranged_channel_index()
    path_waveform_1 = os.path.join(base_path, 'test_DATA_UnitMatch', mouse, probe,location, date_1, experiment,'RawWaveforms')
    path_waveform_2 = os.path.join(base_path, 'test_DATA_UnitMatch', mouse, probe,location, date_2, experiment,'RawWaveforms')

    unit_1_name = f"Unit{pair_indices[0]}_RawSpikes.npy"
    unit_1_data = np.load(os.path.join(path_waveform_1, unit_1_name)) # (82,384,2)
    unit_1_shank = find_belonged_shank(unit_1_data,shank_index_array)
    unit_2_name = f"Unit{pair_indices[1]}_RawSpikes.npy"
    unit_2_data = np.load(os.path.join(path_waveform_2, unit_2_name)) # (82,384,2)
    unit_2_shank = find_belonged_shank(unit_2_data,shank_index_array)

    if unit_1_shank != unit_2_shank:
        print('Warning: two units are not in the same shank')
        return 0
    if plot:
        unit_shank = unit_1_shank
        max_channel_1 = det_max_channel(unit_1_data)
        max_channel_2 = det_max_channel(unit_2_data)
        max_channel = (max_channel_1 + max_channel_2) // 2
        shank_start = unit_shank * 96
        shank_end = (unit_shank + 1) * 96 - 1
        start_channel = max(max_channel - 10, shank_start)
        if start_channel % 2 != 0:
            start_channel -= 1
        end_channel = min(max_channel + 10, shank_end)
        if end_channel % 2 != 1:
            end_channel += 1
        num_channels_to_plot = end_channel - start_channel + 1
        if num_channels_to_plot % 2 != 0:
            num_channels_to_plot += 1
        channel_num = num_channels_to_plot // 2

        fig, axs = plt.subplots(channel_num, 2, figsize=(8, 16), sharex=True, sharey=True)
        for channel_index in range(start_channel, end_channel + 1):
            row = (channel_index - start_channel) // 2
            col = (channel_index - start_channel) % 2
            # axs[row, col].plot(unit_1_data[:, channel_index, 0], color=(220/255, 64/255, 26/255))
            axs[row, col].plot(unit_1_data[:, channel_index, 1], color=(78/255, 183/255, 233/255))
            # axs[row, col].plot(unit_2_data[:, channel_index, 1], color=(78/255, 183/255, 233/255))
            axs[row, col].axis('off')
        fig_save_folder = os.path.join(os.getcwd(),os.pardir,'figures','comparison', method)
        if not os.path.exists(fig_save_folder):
            os.makedirs(fig_save_folder)
        # filename = 'comparison_unit_activity_'+mouse+'_unit'+str(pair_indices[0])+'_unit'+str(pair_indices[1])+'.png'
        # filename = 'comparison_unit_activity_'+mouse+'_unit'+str(pair_indices[1])+'.png'
        filename = 'half2_same_unit_activity_'+mouse+'_unit'+str(pair_indices[0])+'.png'
        plt.savefig(os.path.join(fig_save_folder,filename))
        plt.close()
        print('Unit Activity Image Saved')
        return 1
    

if __name__ == '__main__':
    base = os.path.join(os.getcwd(),os.pardir)

    # load data
    # mouse = 'AV008'

    # mouse = 'AL032'
    session_pair = 1

    # mouse = 'CB016'
    # mouse = 'JF067'
    mouse = 'CB015'

    # load match pair from unitmatch
    match_pair_unitmatch = load_match_pair_unitmatch(mouse,session_pair)
    # load match pair from functional measures
    match_pair_func = load_match_pair_func(mouse,session_pair)
    # load match pair from raw
    match_pair_raw = load_match_pair_raw(mouse,session_pair)

    # plot comparison (venn 3)
    overlap_all, unique_unitmatch, unique_raw, unique_func, overlap_unitmatch_raw, overlap_unitmatch_func, overlap_raw_func = plot_comparison_pairs_venn3_raw(match_pair_unitmatch, match_pair_raw, match_pair_func, mouse, plot=True, save=True)