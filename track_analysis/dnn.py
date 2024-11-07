# Wentao Qiu, 2023-12-06
# qiuwentao1212@gmail.com

import matplotlib.pyplot as plt
import os, sys
import h5py

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment

if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.pardir, os.pardir))
    sys.path.insert(0, os.getcwd())

from models.mymodel import *
from utils.myutil import *
from utils.losses import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(ckpt_path, device):
    model = SpatioTemporalCNN_V2(n_channel=30,n_time=60,n_output=256).to(device)
    model = model.double()
    # Load the model state from the checkpoint file
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


def get_representation_dnn(model,good_units_files_1,good_units_files_2):
    waveform_day1_first_half = []
    waveform_day1_second_half = []
    for index_file, filename in enumerate(good_units_files_1):
        with h5py.File(filename, 'r') as f:
            data = f['waveform'][()]
        waveform_day1_first_half.append(data[..., 0])
        waveform_day1_second_half.append(data[..., 1])
    waveform_day1_first_half = np.array(waveform_day1_first_half)
    waveform_day1_second_half = np.array(waveform_day1_second_half)
    waveform_day2_first_half = []
    waveform_day2_second_half = []
    for index_file, filename in enumerate(good_units_files_2):
        with h5py.File(filename, 'r') as f:
            data = f['waveform'][()]
        waveform_day2_first_half.append(data[..., 0])
        waveform_day2_second_half.append(data[..., 1])

    waveform_day2_first_half = np.array(waveform_day2_first_half)
    waveform_day2_second_half = np.array(waveform_day2_second_half)
    # change to tensor for model
    waveform_day1_first_half = torch.from_numpy(waveform_day1_first_half).to(device)
    waveform_day1_second_half = torch.from_numpy(waveform_day1_second_half).to(device)
    waveform_day2_first_half = torch.from_numpy(waveform_day2_first_half).to(device)
    waveform_day2_second_half = torch.from_numpy(waveform_day2_second_half).to(device)
    # get representation
    model.eval()
    with torch.no_grad():
        rep_day1_first_half = model(waveform_day1_first_half) # representation
        rep_day1_second_half = model(waveform_day1_second_half)
        rep_day2_first_half = model(waveform_day2_first_half)
        rep_day2_second_half = model(waveform_day2_second_half)
    return rep_day1_first_half,rep_day1_second_half,rep_day2_first_half,rep_day2_second_half
    
def get_sim_matrix_all(rep_day1_first_half,rep_day1_second_half,rep_day2_first_half,rep_day2_second_half):
    sim_matrix_11 = clip_sim(rep_day1_first_half, rep_day1_second_half).detach().cpu().numpy()
    sim_matrix_12 = clip_sim(rep_day1_first_half, rep_day2_second_half).detach().cpu().numpy()
    sim_matrix_21 = clip_sim(rep_day1_second_half, rep_day2_first_half).detach().cpu().numpy()
    sim_matrix_22 = clip_sim(rep_day2_first_half, rep_day2_second_half).detach().cpu().numpy()
    return sim_matrix_11, sim_matrix_12, sim_matrix_21, sim_matrix_22

def visualize_sim_matrix_all(sim_matrix_11, sim_matrix_12, sim_matrix_21, sim_matrix_22, mouse, model_name,session_pair):
    '''
    Inputs:
        sim_matrix_11: shape (num_units_day1, num_units_day1)
        sim_matrix_12: shape (num_units_day1, num_units_day2)
        sim_matrix_21: shape (num_units_day1, num_units_day2)
        sim_matrix_22: shape (num_units_day2, num_units_day2)
    '''
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
    filename = 'similarity_matrix_across_days_'+ mouse +'.png'
    plt.savefig(os.path.join(fig_save_folder,filename))
    plt.close()

### get the distribution of the similarity/probability for same neurons and neighboring neurons (respectively)
def get_neighboring_neurons(good_units_files_row,good_units_files_col,day1_MaxSitepos,day2_MaxSitepos,days_type='within_day'):
    dist_matrix_across_days = get_Sitepos_dist(day1_MaxSitepos,day2_MaxSitepos)
    N,M = len(good_units_files_row), len(good_units_files_col)
    # for within day
    neighboring_neurons = []
    if days_type == 'within_day':
        assert N == M
        for i in range(N):
            # only upper triangle
            for j in range(i+1, N):
                if good_units_files_row[i] != good_units_files_col[j]:
                    dist = dist_matrix_across_days[i,j]
                    if dist >= 0 and dist < 50:
                        neighboring_neurons.append([i,j])
    neighboring_neurons = np.array(neighboring_neurons)
    return neighboring_neurons

def get_combined_neurons(good_units_files_row,good_units_files_col,day1_MaxSitepos,day2_MaxSitepos):
    dist_matrix_across_days = get_Sitepos_dist(day1_MaxSitepos,day2_MaxSitepos)
    N,M = len(good_units_files_row), len(good_units_files_col)
    combined_neurons = []
    for i in range(N):
        for j in range(M):
            dist = dist_matrix_across_days[i,j]
            if dist >= 0 and dist < 50:
                combined_neurons.append([i,j])
    combined_neurons = np.array(combined_neurons)
    return combined_neurons

# plot neighboring neurons and same neurons together
def get_sim_distribution_within_day_tg(sim_matrix, combined_neurons, mouse,probe,location,exps,model_name,session_pair,nbins,days_index):
    '''
    Inputs: 
    sim_matrix: shape (num_units_row, num_units_col)
    neighboring_neurons: shape (num_pairs, 2)
    days_index: 0 for day 1, 1 for day 2
    '''
    tg_dist_values = []
    for pair in combined_neurons:
        i,j = pair
        tg_dist_values.append(sim_matrix[i,j])
    tg_dist_values = np.array(tg_dist_values)
    median_value = np.median(tg_dist_values)
    fontsize = 24
    fontsize_title = 28
    fig, ax = plt.subplots(1,1, figsize = (14,10))
    # x axis, values of similarity value, y axis, number of pairs
    min_val = np.min(tg_dist_values)
    max_val = np.max(tg_dist_values)
    bins = np.linspace(min_val, max_val, nbins)
    ax.hist(tg_dist_values, bins = bins, alpha = 0.5, label = 'combined neurons',density=True)
    ax.set_title('Similarity Distribution (combined)', fontsize = fontsize_title)
    ax.axvline(x=median_value, color='r', linestyle='--', label='Median Sim')
    ax.set_xlabel('Similarity', fontsize = fontsize)
    ax.set_ylabel('Probability Density', fontsize = fontsize)
    ax.legend(fontsize = fontsize)
    fig_save_folder = os.path.join(os.getcwd(),os.pardir,'figures','dnn',model_name,mouse,'session_pair'+session_pair)
    if not os.path.exists(fig_save_folder):
        os.makedirs(fig_save_folder, exist_ok=True)
    # exact_day = dates[days_index]
    exp = exps[days_index]
    filename = 'similarity_distribution_combined_'+ mouse +'_'+ exp +'.png'
    plt.savefig(os.path.join(fig_save_folder,filename))
    plt.close()
    return median_value

# plot neighboring neurons and same neurons separately
def get_sim_distribution_within_day_sp(sim_matrix, neighboring_neurons, mouse,probe,location,dates,exps,model_name,session_pair,nbins,days_index):
    '''
    Inputs: 
    sim_matrix: shape (num_units_row, num_units_col)
    neighboring_neurons: shape (num_pairs, 2)
    days_index: 0 for day 1, 1 for day 2
    '''
    neighboring_dist_values = []
    same_dist_values = []
    N,M = sim_matrix.shape
    # same neurons similarity
    for i in range(N):
        same_dist_values.append(sim_matrix[i,i])
    # neighboring neurons similarity
    for pair in neighboring_neurons:
        i,j = pair
        neighboring_dist_values.append(sim_matrix[i,j])
    same_dist_values = np.array(same_dist_values)
    neighboring_dist_values = np.array(neighboring_dist_values)
    min_val = min(np.min(same_dist_values), np.min(neighboring_dist_values))
    max_val = max(np.max(same_dist_values), np.max(neighboring_dist_values))
    bins = np.linspace(min_val, max_val, nbins)
    fontsize = 24
    fontsize_title = 28
    fig, ax = plt.subplots(1,1, figsize = (14,10))
    # x axis, values of similarity value, y axis, number of pairs
    ax.hist(same_dist_values, bins = bins, alpha = 0.5, label = 'same neurons',density=True)
    ax.hist(neighboring_dist_values, bins = bins, alpha = 0.5, label = 'neighboring neurons',density=True)
    # weights_same = np.ones_like(same_dist_values) / len(same_dist_values)
    # weights_neighbors = np.ones_like(neighboring_dist_values) / len(neighboring_dist_values)
    # ax.hist(same_dist_values, bins=nbins, alpha=0.5, weights=weights_same, label='same neurons.', density=True)
    # ax.hist(neighboring_dist_values, bins=nbins, alpha=0.5, weights=weights_neighbors, label='neighboring neurons', density=True)
    
    ax.set_title('Similarity Distribution', fontsize = fontsize_title)
    ax.set_xlabel('Similarity', fontsize = fontsize)
    ax.set_ylabel('Probability Density', fontsize = fontsize)
    ax.legend(fontsize = fontsize)
    fig_save_folder = os.path.join(os.getcwd(),os.pardir,'figures','dnn',model_name,mouse,'session_pair'+session_pair)
    if not os.path.exists(fig_save_folder):
        os.makedirs(fig_save_folder, exist_ok=True)
    exact_day = dates[days_index]
    exp = exps[days_index]
    filename = 'similarity_distribution_'+mouse+'_'+exp+'.png'
    plt.savefig(os.path.join(fig_save_folder,filename))
    plt.close()

def get_sim_distribution_across_day_tg(sim_matrix_12,sim_matrix_21, combined_neurons,mouse,probe,location,exps,model_name,session_pair,boundary_similarity_value,nbins):
    N,M = sim_matrix_12.shape
    dist_values_12 = []
    dist_values_21 = []
    for pair in combined_neurons:
        i,j = pair
        dist_values_12.append(sim_matrix_12[i,j])
        dist_values_21.append(sim_matrix_21[i,j])
    dist_values_12 = np.array(dist_values_12)
    dist_values_21 = np.array(dist_values_21)
    median_value_12 = np.median(dist_values_12)
    median_value_21 = np.median(dist_values_21)
    return median_value_12, median_value_21

    # dist_values = []
    # for pair in combined_neurons:
    #     i,j = pair
    #     average_value = (sim_matrix_12[i,j] + sim_matrix_21[i,j]) / 2
    #     dist_values.append(average_value)
    # dist_values = np.array(dist_values)
    # median_value = np.median(dist_values)
    # fontsize = 24
    # fontsize_title = 28
    # fig, ax = plt.subplots(1,1, figsize = (14,10))
    # # x axis, values of similarity value, y axis, number of pairs
    # min_val = np.min(dist_values)
    # max_val = np.max(dist_values)
    # bins = np.linspace(min_val, max_val, nbins)
    # ax.hist(dist_values, bins = bins, alpha = 0.5, label = "'neighboring' neurons", density=True)
    # ax.axvline(x=boundary_similarity_value, color='k', linestyle='--', label='WD-Thr Sim')
    # ax.axvline(x=median_value, color='r', linestyle='--', label='Median Sim')
    # ax.set_title('Similarity Distribution', fontsize = fontsize_title)
    # ax.set_xlabel('Similarity', fontsize = fontsize)
    # ax.set_ylabel('Probability Density', fontsize = fontsize)
    # ax.legend(fontsize = fontsize)
    # fig_save_folder = os.path.join(os.getcwd(),os.pardir,'figures','dnn',model_name,mouse,'session_pair'+session_pair)
    # if not os.path.exists(fig_save_folder):
    #     os.makedirs(fig_save_folder, exist_ok=True)
    # filename = 'similarity_distribution_'+mouse+'_'+ 'across_days' +'.png'
    # plt.savefig(os.path.join(fig_save_folder,filename))
    # plt.close()
    # return median_value

def get_sim_distribution_intersect(sim_matrix,neighboring_neurons,mouse,probe,location,exps,model_name,session_pair,nbins,days_index):  
    '''
    Inputs: 
    sim_matrix: shape (num_units_row, num_units_col)
    neighboring_neurons: shape (num_pairs, 2)
    '''
    neighboring_dist_values = []
    same_dist_values = []
    N, M = sim_matrix.shape
    assert N == M
    for i in range(N):
        same_dist_values.append(sim_matrix[i, i])
    for pair in neighboring_neurons:
        i, j = pair
        neighboring_dist_values.append(sim_matrix[i, j])

    # Convert lists to numpy arrays
    same_dist_values = np.array(same_dist_values)
    neighboring_dist_values = np.array(neighboring_dist_values)
    min_val = min(np.min(same_dist_values), np.min(neighboring_dist_values))
    max_val = max(np.max(same_dist_values), np.max(neighboring_dist_values))
    bins = np.linspace(min_val, max_val, nbins)
    same_counts, _ = np.histogram(same_dist_values, bins=bins, density=True)
    neighboring_counts, _ = np.histogram(neighboring_dist_values, bins=bins, density=True)

    # Smooth histograms
    sigma = 0.6  # Adjust this value as needed
    same_smoothed = gaussian_filter1d(same_counts, sigma)
    # print('same_smoothed', same_smoothed)
    neighboring_smoothed = gaussian_filter1d(neighboring_counts, sigma)
    # print('neighboring_smoothed', neighboring_smoothed)

    # Find the boundary similarity value
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_width = bin_centers[1] - bin_centers[0]
    boundary_similarity_value = None
    for i in range(len(bin_centers)):
        threshold =  neighboring_smoothed[i] 
        if threshold == 0:
            if same_smoothed[i] > threshold:
                # boundary_similarity_value = bin_centers[i] - bin_width/2
                boundary_similarity_value = bin_centers[i]
                break
        else:
            if same_smoothed[i] > threshold:
                boundary_similarity_value = bin_centers[i] - bin_width/2
                # boundary_similarity_value = bin_centers[i]
                break

    # Plot the distribution over the entire range
    fontsize = 24
    fontsize_title = 28
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    # Plot the smoothed histograms
    ax.plot(bin_centers, same_smoothed, label='same neurons')
    ax.plot(bin_centers, neighboring_smoothed, label='neighboring neurons')
    if boundary_similarity_value is not None:
        # Plot the strict boundary similarity value
        ax.axvline(x=boundary_similarity_value, color='k', linestyle='--', label='Boundary similarity')
    ax.set_title('Similarity Distribution (Sm)', fontsize=fontsize_title)
    ax.set_xlabel('Similarity', fontsize=fontsize)
    ax.set_ylabel('Number of Pairs', fontsize=fontsize)
    ax.legend(fontsize=fontsize)
    fig_save_folder = os.path.join(os.getcwd(), os.pardir, 'figures', 'dnn', model_name, mouse, 'session_pair' + session_pair)
    if not os.path.exists(fig_save_folder):
        os.makedirs(fig_save_folder, exist_ok=True)
    # exact_day = dates[days_index]
    exp = exps[days_index]
    filename = 'similarity_distribution_intersect_' + mouse + '_' + exp+'.png'
    plt.savefig(os.path.join(fig_save_folder, filename))
    plt.close()
    return boundary_similarity_value

# get pair of units with highest similarity
def get_match_pair_above_SimThr(sim_matrix_1,sim_matrix_2,day1_MaxSitepos,day2_MaxSitepos,good_units_1,good_units_2,mouse,model_name,session_pair,thr = 0.8):
    '''
    thr: threshold, mat_1(i,j) > thr and mat_2(i,j) > thr
    good_units: real indices of indices in probs_matrix
    '''
    dist_matrix_across_days = get_Sitepos_dist(day1_MaxSitepos,day2_MaxSitepos)
    # filtering
    sim_matrix_1[dist_matrix_across_days == -1] = -1 
    sim_matrix_1[dist_matrix_across_days > 40] = -1
    sim_matrix_2[dist_matrix_across_days == -1] = -1
    sim_matrix_2[dist_matrix_across_days > 40] = -1
    match_pair = []
    for i in range(sim_matrix_1.shape[0]):
        for j in range(sim_matrix_1.shape[1]):
            if sim_matrix_1[i,j] > thr and sim_matrix_2[i,j] > thr:
                match_pair.append([good_units_1[i],good_units_2[j]])
    match_pair = np.array(match_pair)
    # save results
    results_save_folder = os.path.join(os.getcwd(),os.pardir,'results','dnn',model_name,mouse)
    if not os.path.exists(results_save_folder):
        os.makedirs(results_save_folder)
    filename = 'match_pair_'+ 'session_pair' + str(session_pair) + '.npy'
    np.save(os.path.join(results_save_folder,filename),match_pair)
    return match_pair

def hungarian_matching_with_threshold(similarity_matrix, similarity_threshold, separator=None):
    N, M = similarity_matrix.shape
    max_dim = max(N, M)
    # Convert similarity to cost and apply threshold
    high_cost = 1e9  # A very large cost for unacceptable pairs
    cost_matrix = np.where(similarity_matrix >= similarity_threshold, 1 - similarity_matrix, high_cost)
    if separator:
        # Assign high cost to all the within-day pairs so these aren't selected
        cost_matrix[:separator, :separator] = high_cost
        cost_matrix[separator:, separator:] = high_cost
    # Pad the matrix to make it square if necessary
    if N != M:
        padded_cost_matrix = np.pad(cost_matrix, ((0, max_dim - N), (0, max_dim - M)), mode='constant', constant_values=high_cost)
    else:
        padded_cost_matrix = cost_matrix
    # Apply the Hungarian Algorithm
    row_ind, col_ind = linear_sum_assignment(padded_cost_matrix)
    # Filter out the dummy and high-cost assignments
    matches = [[i, j] for i, j in zip(row_ind, col_ind) if i < N and j < M and similarity_matrix[i, j] >= similarity_threshold]
    matches = np.array(matches)
    return matches


def tracking_method_for_inference(sim_matrix_12,sim_matrix_21,day1_MaxSitepos,day2_MaxSitepos,good_units_1,good_units_2,mouse,model_name,session_pair,NoMatchThr=0.8):
    pred_pairs = []
    '''
    tracking algorithm with 'sparsification'
    If the channel position is far away, 
    that element of the graph describing the similarity 
    between a pair of neurons is set to zero and sparsened out. 
    '''
    sim_matrix_across_days = (sim_matrix_12 + sim_matrix_21) / 2
    dist_matrix_across_days = get_Sitepos_dist(day1_MaxSitepos,day2_MaxSitepos)
    # sparsification
    # cosine similarity is between -1 and 1, so we set them to -1
    sim_matrix_across_days[dist_matrix_across_days == -1] = -1 
    sim_matrix_across_days[dist_matrix_across_days > 40] = -1
    # Hungarian matching
    matches = hungarian_matching_with_threshold(sim_matrix_across_days, similarity_threshold=NoMatchThr)
    # transfer to neuron index
    for pair in matches:
        pred_pairs.append([good_units_1[pair[0]],good_units_2[pair[1]]])
    pred_pairs = np.array(pred_pairs)
    # save results
    results_save_folder = os.path.join(os.getcwd(),os.pardir,'results','dnn',model_name,mouse)
    if not os.path.exists(results_save_folder):
        os.makedirs(results_save_folder)
    filename = 'match_pair_'+ 'session_pair' + str(session_pair) + '.npy'
    np.save(os.path.join(results_save_folder,filename),pred_pairs)
    return pred_pairs

if __name__ == '__main__':
    base = os.getcwd()
    # load model
    model_name = 'wentao'
    # model_name = '2024_2_13_ft_SpatioTemporalCNN_V2'
    ckpt_path = os.path.join('ModelExp','experiments', model_name, 'ckpt', 'ckpt_epoch_49')
    ckpt_path = os.path.join(base, ckpt_path)
    model = load_model(ckpt_path, device)

    # load data
    mode = 'test' # 'train' or 'test'
    mouse = 'AL036'
    probe = '19011116882'
    location = '3'
    dates = ['2020-08-04', '2020-08-05']
    exps = ['_2020-08-04_ephys__2020-08-04_stripe240r1_natIm_g0_imec0_PyKS_output', 
            '_2020-08-05_ephys__2020-08-05_stripe240_natIm_g0__2020-08-05_stripe240_natIm_g0_imec0_PyKS_output']
    session_pair = '2'
    print('mouse',mouse,'session_pair',session_pair)
    good_units_files_1,good_units_indices_1,good_units_files_2,good_units_indices_2 = load_mouse_data(mouse,probe,location,exps,mode)
    rep_day1_first_half,rep_day1_second_half,rep_day2_first_half,rep_day2_second_half = get_representation_dnn(model,good_units_files_1,good_units_files_2)
    sim_matrix_11, sim_matrix_12, sim_matrix_21, sim_matrix_22 = get_sim_matrix_all(rep_day1_first_half,rep_day1_second_half,rep_day2_first_half,rep_day2_second_half)
    del rep_day1_first_half,rep_day1_second_half,rep_day2_first_half,rep_day2_second_half

    # original plot
    # visualize_sim_matrix_all(sim_matrix_11, sim_matrix_12, sim_matrix_21, sim_matrix_22, mouse, probe, location, dates, exps, model_name)
    
    # rearranged plot by position
    day1_MaxSitepos, day2_MaxSitepos = get_day_MaxSitepos(good_units_files_1,good_units_files_2)
    day1_sorted_indices = sort_neurons_by_position(day1_MaxSitepos)
    day2_sorted_indices = sort_neurons_by_position(day2_MaxSitepos)

    resim_matrix_11, resim_matrix_12, resim_matrix_21, resim_matrix_22 = rearrange_four_matrix(sim_matrix_11, sim_matrix_12, sim_matrix_21, sim_matrix_22, day1_sorted_indices, day2_sorted_indices)
    visualize_sim_matrix_all(resim_matrix_11, resim_matrix_12, resim_matrix_21, resim_matrix_22, mouse, model_name,session_pair)
    del resim_matrix_11, resim_matrix_12, resim_matrix_21, resim_matrix_22

    # Within-day similarity distribution
    # get neighboring neurons within day 1 and day 2
    neighboring_neurons_within_day1 = get_neighboring_neurons(good_units_files_1,good_units_files_1,day1_MaxSitepos,day1_MaxSitepos)
    neighboring_neurons_within_day2 = get_neighboring_neurons(good_units_files_2,good_units_files_2,day2_MaxSitepos,day2_MaxSitepos)
    # get combined neurons
    combined_neurons_within_day1 = get_combined_neurons(good_units_files_1,good_units_files_1,day1_MaxSitepos,day1_MaxSitepos)
    combined_neurons_within_day2 = get_combined_neurons(good_units_files_2,good_units_files_2,day2_MaxSitepos,day2_MaxSitepos)
    # plot within-day sim distribution respectively for day 1 and day 2
    max_neuron_day = max(sim_matrix_11.shape[0], sim_matrix_22.shape[0])
    nbins = max_neuron_day // 5
    # get_sim_distribution_within_day_sp(sim_matrix_11, neighboring_neurons_within_day1,mouse,probe,location,dates,exps,model_name,nbins,days_index=0)  
    # get_sim_distribution_within_day_sp(sim_matrix_22, neighboring_neurons_within_day2,mouse,probe,location,dates,exps,model_name,nbins,days_index=1)
    boundary_similarity_value_day1 = get_sim_distribution_intersect(sim_matrix_11, neighboring_neurons_within_day1,mouse,probe,location,exps,model_name,session_pair,nbins,days_index=0)  
    boundary_similarity_value_day2 = get_sim_distribution_intersect(sim_matrix_22, neighboring_neurons_within_day2,mouse,probe,location,exps,model_name,session_pair,nbins,days_index=1)
    # if max_neuron_day > 100:
    #     within_day_sim_thr = max(boundary_similarity_value_day1, boundary_similarity_value_day2)
    # else:
    #     within_day_sim_thr =  (boundary_similarity_value_day1 + boundary_similarity_value_day2) / 2
    # within_day_sim_thr = min(boundary_similarity_value_day1, boundary_similarity_value_day2)
    within_day_sim_thr = max(boundary_similarity_value_day1, boundary_similarity_value_day2)
    # within_day_sim_thr =  (boundary_similarity_value_day1 + boundary_similarity_value_day2) / 2
    # print('within_day_sim_thr',within_day_sim_thr)
    # plot within-day sim distribution for combined neurons
    median_value_within_day1 = get_sim_distribution_within_day_tg(sim_matrix_11, combined_neurons_within_day1, mouse,probe,location,exps,model_name,session_pair,nbins,days_index=0)
    median_value_within_day2 = get_sim_distribution_within_day_tg(sim_matrix_22, combined_neurons_within_day2, mouse,probe,location,exps,model_name,session_pair,nbins,days_index=1)
    # if max_neuron_day > 100:
    #     median_value_within_day = min(median_value_within_day1, median_value_within_day2)
    # else:
    #     median_value_within_day = (median_value_within_day1 + median_value_within_day2) / 2
    # median_value_within_day = max(median_value_within_day1, median_value_within_day2)
    median_value_within_day = min(median_value_within_day1, median_value_within_day2)
    # median_value_within_day = (median_value_within_day1 + median_value_within_day2) / 2
    # print('median_value_within_day',median_value_within_day)

    # Across-day similarity distribution
    # get combined neurons
    combined_neurons_across_day = get_combined_neurons(good_units_files_1,good_units_files_2,day1_MaxSitepos,day2_MaxSitepos)
    # plot across-day sim distribution
    median_value_across_day_12,median_value_across_day_21 = get_sim_distribution_across_day_tg(sim_matrix_12,sim_matrix_21, combined_neurons_across_day,mouse,probe,location,exps,model_name,session_pair,within_day_sim_thr,nbins)
    median_value_across_day = max(median_value_across_day_12, median_value_across_day_21)
    # print('median_value_across_day',median_value_across_day)
    # get across-day sim threshold
    across_day_sim_thr = within_day_sim_thr - (median_value_within_day - median_value_across_day)
    # print('across_day_sim_thr',across_day_sim_thr)

    # Hungarian tracking method
    pred_pairs = tracking_method_for_inference(sim_matrix_12,sim_matrix_21,day1_MaxSitepos,day2_MaxSitepos,good_units_indices_1,good_units_indices_2,mouse,model_name,session_pair,NoMatchThr=across_day_sim_thr)
    print('pred_pairs',pred_pairs)
    print('len(pred_pairs)',len(pred_pairs))

    # Similarity threshold tracking method
    # artifical_thr = 0.8
    # match_pair = get_match_pair_above_SimThr(sim_matrix_12, sim_matrix_21, day1_MaxSitepos, day2_MaxSitepos, good_units_indices_1, good_units_indices_2,mouse,model_name,session_pair,thr=artifical_thr)

    # sim_matrix_across_days = (sim_matrix_12 + sim_matrix_21) / 2
    # pair_index_1 = find_index_np(132,good_units_indices_1)
    # pair_index_2 = find_index_np(139,good_units_indices_2)
    # median_value_within_day = np.round(median_value_within_day,4)
    # within_day_sim_thr = np.round(within_day_sim_thr,4)
    # median_value_across_day = np.round(median_value_across_day,4)
    # across_day_sim_thr = np.round(across_day_sim_thr,4)
    # sim_value = np.round(sim_matrix_across_days[pair_index_1,pair_index_2],4) + 0.9
    # print('median_value_within_day',median_value_within_day)
    # print('within_day_sim_thr',within_day_sim_thr)
    # print('median_value_across_day',median_value_across_day)
    # print('across_day_sim_thr',across_day_sim_thr)
    # print('sim value of pair [132,139]',sim_value)
    # print(pred_pairs)
    # print(len(pred_pairs))