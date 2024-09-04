### see predicted pairs directly by feeding the raw data into clip probability model



import matplotlib.pyplot as plt
import os, sys
import h5py

import numpy as np
import torch


if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.pardir, os.pardir))
    sys.path.insert(0, os.path.join(os.pardir))
    from dnn import load_mouse_data, get_sim_matrix, get_prob_matrix, get_match_pair_prob,get_match_pair_sim
else:
    from track_analysis.dnn import load_mouse_data, get_sim_matrix, get_prob_matrix, get_match_pair_prob,get_match_pair_sim

from utils.myutil import *
from utils.losses import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_loss = ClipLoss2D().to(device)
clip_loss.eval()

def get_representation_raw(good_units_files_1,good_units_files_2):
    # first, get the probability distribution within day for day 1 and day 2
    waveform_day1_first_half = []
    waveform_day1_second_half = []
    print('day 1 len', len(good_units_files_1))
    for index_file, filename in enumerate(good_units_files_1):
        # data = np.load(filename)
        with h5py.File(filename, 'r') as f:
            data = f['waveform'][()]
        waveform_day1_first_half.append(data[..., 0])
        waveform_day1_second_half.append(data[..., 1])
    waveform_day1_first_half = np.array(waveform_day1_first_half)
    waveform_day1_second_half = np.array(waveform_day1_second_half)
    waveform_day2_first_half = []
    waveform_day2_second_half = []
    print('day 2 len', len(good_units_files_2))
    for index_file, filename in enumerate(good_units_files_2):
        # data = np.load(filename)
        with h5py.File(filename, 'r') as f:
            data = f['waveform'][()]
        waveform_day2_first_half.append(data[..., 0])
        waveform_day2_second_half.append(data[..., 1])

    waveform_day2_first_half = np.array(waveform_day2_first_half)
    waveform_day2_second_half = np.array(waveform_day2_second_half)
    
    waveform_day1_first_half = torch.from_numpy(waveform_day1_first_half).to(device)
    waveform_day1_second_half = torch.from_numpy(waveform_day1_second_half).to(device)
    waveform_day2_first_half = torch.from_numpy(waveform_day2_first_half).to(device)
    waveform_day2_second_half = torch.from_numpy(waveform_day2_second_half).to(device)

    return waveform_day1_first_half, waveform_day1_second_half, waveform_day2_first_half, waveform_day2_second_half


def visualize_sim_matrix_raw(sim_matrix_1, sim_matrix_2, mouse, probe, location, dates, exps):
    fontsize = 24
    fontsize_title = 28 
    tick_params_dict = {'axis': 'both', 'which': 'both', 'labelsize': 20, 'direction': 'out'}
    fig, ax = plt.subplots(1,2, figsize = (14,10))
    img0 = ax[0].imshow(sim_matrix_1,cmap = 'coolwarm')
    ax[0].set_title('Similarity Matrix 11-22', fontsize = fontsize)
    ax[0].set_xlabel('Day 2', fontsize = fontsize)
    ax[0].set_ylabel('Day 1', fontsize = fontsize)
    ax[0].tick_params(**tick_params_dict)
    # fig.colorbar(img0, ax=ax[0])
    img1 = ax[1].imshow(sim_matrix_2,cmap = 'coolwarm')
    ax[1].set_title('Similarity Matrix 12-21', fontsize = fontsize)
    ax[1].set_xlabel('Day 2', fontsize = fontsize)
    ax[1].set_ylabel('Day 1', fontsize = fontsize)
    ax[1].tick_params(**tick_params_dict)
    # fig.colorbar(img1, ax=ax[1])
    fig.suptitle('Similarity Matrix across days', fontsize = fontsize_title)
    plt.colorbar(img0, ax=ax, orientation='horizontal')
    fig_save_folder = os.path.join(os.getcwd(),os.pardir,'figures','raw')
    if not os.path.exists(fig_save_folder):
        os.makedirs(fig_save_folder)
    filename = 'sim_matrix_across_days_'+mouse+'_'+probe+'_'+location+'_'+ dates[0]+'_'+dates[1]+'_'+ exps[0]+'_'+exps[1]+'.png'
    plt.savefig(os.path.join(fig_save_folder,filename))
    plt.close()


def visualize_prob_matrix_raw(probs_matrix_1, probs_matrix_2, mouse, probe, location, dates, exps):
    fontsize = 24
    fontsize_title = 28 
    tick_params_dict = {'axis': 'both', 'which': 'both', 'labelsize': 20, 'direction': 'out'}
    fig, ax = plt.subplots(1,2, figsize = (14,10))
    img0 = ax[0].imshow(probs_matrix_1,cmap = 'coolwarm')
    ax[0].set_title('Probability Matrix 11-22', fontsize = fontsize)
    ax[0].set_xlabel('Day 2', fontsize = fontsize)
    ax[0].set_ylabel('Day 1', fontsize = fontsize)
    ax[0].tick_params(**tick_params_dict)
    # fig.colorbar(img0, ax=ax[0])
    img1 = ax[1].imshow(probs_matrix_2,cmap = 'coolwarm')
    ax[1].set_title('Probability Matrix 12-21', fontsize = fontsize)
    ax[1].set_xlabel('Day 2', fontsize = fontsize)
    ax[1].set_ylabel('Day 1', fontsize = fontsize)
    ax[1].tick_params(**tick_params_dict)
    # fig.colorbar(img1, ax=ax[1])
    fig.suptitle('Probability Matrix across days', fontsize = fontsize_title)
    plt.colorbar(img0, ax=ax, orientation='horizontal')
    fig_save_folder = os.path.join(os.getcwd(),os.pardir,'figures','raw')
    if not os.path.exists(fig_save_folder):
        os.makedirs(fig_save_folder)
    filename = 'prob_matrix_across_days_'+mouse+'_'+probe+'_'+location+'_'+ dates[0]+'_'+dates[1]+'_'+ exps[0]+'_'+exps[1]+'.png'
    plt.savefig(os.path.join(fig_save_folder,filename))
    plt.close()


def save_filter_prediction_raw(pred_pairs,mouse,probe,location,dates,exps,session_pair):
    base_path = os.path.join(os.getcwd(),os.pardir, os.pardir)
    path_waveform_1 = os.path.join(base_path, 'test_ONE_DATA_UnitMatch', mouse, probe,location, dates[0], exps[0],'RawWaveforms')
    path_waveform_2 = os.path.join(base_path, 'test_ONE_DATA_UnitMatch', mouse, probe,location, dates[1], exps[1],'RawWaveforms')

    # only keep pairs with max channel distance < 10
    pred_pairs_filtered = []
    for pair_indices in pred_pairs:
        unit_1_name = f"Unit{pair_indices[0]}_RawSpikes.npy"
        neuron_file = os.path.join(path_waveform_1, unit_1_name)
        # unit_1_data = np.load(os.path.join(path_waveform_1, unit_1_name)) # (82,384,2)
        with h5py.File(neuron_file, 'r') as f:
            unit_1_data = f['waveform'][()]
            unit_1_shank = f['shank'][()]
        
        unit_2_name = f"Unit{pair_indices[1]}_RawSpikes.npy"
        neuron_file = os.path.join(path_waveform_2, unit_2_name)
        # unit_2_data = np.load(os.path.join(path_waveform_2, unit_2_name)) # (82,384,2)
        with h5py.File(neuron_file, 'r') as f:
            unit_2_data = f['waveform'][()]
            unit_2_shank = f['shank'][()]
            
        if unit_1_shank == unit_2_shank:
            pred_pairs_filtered.append(pair_indices)
    
    pred_pairs_filtered = np.array(pred_pairs_filtered)
    results_save_folder = os.path.join(os.getcwd(),os.pardir,'results','raw',mouse)
    if not os.path.exists(results_save_folder):
        os.makedirs(results_save_folder)
    filename = 'match_pair_'+ 'session_pair' + str(session_pair) + '.npy'
    np.save(os.path.join(results_save_folder,filename),pred_pairs_filtered)
    return pred_pairs_filtered

if __name__ == '__main__':
    # load data

    # mouse = 'AV008'
    # session_pair = 1
    # probe = 'Probe0'
    # location = '1'
    # dates = ['2022-03-12','2022-03-13']
    # exps = ['exp1','exp1']

    # mouse = 'AL032'
    # session_pair = 1
    # probe = 'Probe0'
    # location = '1'
    # dates = ['2019-11-21','2019-11-22']
    # exps = ['exp1','exp1']

    # mouse = 'CB016'
    # session_pair = 1
    # probe = 'Probe0'
    # location = '1'
    # dates = ['2021-09-28','2021-09-29']
    # exps = ['exp1','exp1']

    # mouse = 'JF067'
    # session_pair = 1
    # probe = 'Probe0'
    # location = '1'
    # dates = ['2022-02-14','2022-02-15']
    # exps = ['exp1','exp1']

    mouse = 'CB015'
    session_pair = 1
    probe = '19011110242'
    location = '1'
    dates = ['2021-09-10', '2021-09-11']
    exps = ['CB015_2021-09-10_NatImages_g0_t0-imec0-ap', 'CB015_2021-09-11_NatImages_g0_t0-imec0-ap']

    good_units_files_1,good_units_indices_1,good_units_files_2,good_units_indices_2 = load_mouse_data(mouse,probe,location,dates,exps)
    waveform_day1_first_half, waveform_day1_second_half, waveform_day2_first_half, waveform_day2_second_half = get_representation_raw(good_units_files_1,good_units_files_2)
    sim_matrix_12, sim_matrix_21 = get_sim_matrix(clip_loss,waveform_day1_first_half, waveform_day1_second_half, waveform_day2_first_half, waveform_day2_second_half)
    probs_matrix_12, probs_matrix_21 = get_prob_matrix(clip_loss,waveform_day1_first_half, waveform_day1_second_half, waveform_day2_first_half, waveform_day2_second_half)

    visualize_sim_matrix_raw(sim_matrix_12, sim_matrix_21, mouse, probe, location, dates, exps)
    visualize_prob_matrix_raw(probs_matrix_12, probs_matrix_21, mouse, probe, location, dates, exps)

    # match_pair = get_match_pair_prob(probs_matrix_12, probs_matrix_21, good_units_indices_1, good_units_indices_2, mouse, thr = 0.9)
    match_pair = get_match_pair_sim(sim_matrix_12, sim_matrix_21, good_units_indices_1, good_units_indices_2, mouse, thr = 0.8)
    match_pair = save_filter_prediction_raw(match_pair,mouse,probe,location,dates,exps,session_pair)
    
    # has_duplicate(match_pair[:,0])
    # has_duplicate(match_pair[:,1])

    match_pair = match_pair.tolist()
    print('match_pair', match_pair)
    print('len match_pair', len(match_pair))
