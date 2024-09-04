

import matplotlib.pyplot as plt
import os, sys

import numpy as np
import torch


if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.pardir, os.pardir))
    sys.path.insert(0, os.path.join(os.pardir))

from models.mymodel import *
from utils.myutil import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(ckpt_path, device):
    model = SpatioTemporalCNN_V2(n_channel=30,n_time=60,n_output=256).to(device)
    model = model.double()
    # Load the model state from the checkpoint file
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

def load_mouse_data(mouse,probe,location,dates,experiment):
    base_path = os.path.join(os.getcwd(),os.pardir,os.pardir)
    base_path = os.path.join(base_path,'test_DATA_UnitMatch')
    # first date
    date_1 = dates[0]
    base_path_1 = os.path.join(base_path,mouse,probe,location,date_1,experiment)
    waveform_path_1 = os.path.join(base_path_1,'RawWaveforms')
    good_units_index_1 = read_good_id_from_mat(os.path.join(base_path_1, 'PreparedData.mat'))
    good_units_files_1,good_units_indices_1 = select_good_units_files_indices(waveform_path_1, good_units_index_1)
    # second date
    date_2 = dates[1]
    base_path_2 = os.path.join(base_path,mouse,probe,location,date_2,experiment)
    waveform_path_2 = os.path.join(base_path_2,'RawWaveforms')
    good_units_index_2 = read_good_id_from_mat(os.path.join(base_path_2, 'PreparedData.mat'))
    good_units_files_2,good_units_indices_2 = select_good_units_files_indices(waveform_path_2, good_units_index_2)
    return good_units_files_1,good_units_indices_1,good_units_files_2,good_units_indices_2

def get_neuron_waveform(ori_idx,mouse,probe,location,date,experiment):
    base_path = os.path.join(os.getcwd(),os.pardir, os.pardir)
    path_waveform = os.path.join(base_path, 'test_DATA_UnitMatch', mouse, probe,location, date, experiment,'RawWaveforms')
    unit_name = f"Unit{ori_idx}_RawSpikes.npy"
    unit_waveform = np.load(os.path.join(path_waveform, unit_name)) # (82,384,2)
    return unit_waveform

def get_subplot_indices(channel_indices,row_number):
    group_indices = channel_indices // 2  # Indices of the groups (pairs of channels)
    rows = group_indices % row_number
    cols = (group_indices // row_number) * 2 + channel_indices % 2
    return rows, cols

def visualize_paired_layer_outputs(layer_outputs_1,layer_outputs_2, mouse, idx_1, idx_2,order):
    # get keys of layer_outputs_1
    keys = layer_outputs_1.keys()
    for key in keys:
        layer_output_1 = layer_outputs_1[key]
        layer_output_2 = layer_outputs_2[key]
        mean_output_1 = layer_output_1.mean(dim=0) # (T,C)
        mean_output_2 = layer_output_2.mean(dim=0) # (T,C)
        channel_num = mean_output_1.shape[1]
        col_number = 8
        row_number = channel_num // col_number
        fig, axs = plt.subplots(row_number, col_number, figsize=(12, 24), sharex=True, sharey=True)
        # fig, axs = plt.subplots(row_number, col_number, figsize=(12, 24))
        for channel_idx in range(channel_num):
            row, col = get_subplot_indices(channel_idx,row_number)
            axs[row, col].plot(mean_output_1[:,channel_idx].cpu().detach().numpy(),color=(220/255, 64/255, 26/255))
            axs[row, col].plot(mean_output_2[:,channel_idx].cpu().detach().numpy(),color=(78/255, 183/255, 233/255))
            # axs[row, col].set_title(f"Channel {channel_idx}")
            axs[row, col].axis('off')  # Optional: Hide axes

        base = os.path.join(os.getcwd(),os.pardir)
        path_figures = os.path.join(base, 'figures',mouse)
        if not os.path.exists(path_figures):
            os.makedirs(path_figures)
        plt.savefig(os.path.join(path_figures,'units' + str(idx_1) + '_' + str(idx_2) +'Layer'+str(key)+'_Output' + '_' + str(order) +'.png'))
        print('Layer Output Image Saved')
        # plt.show()
        plt.close()
        

if __name__ == '__main__':
    base = os.path.join(os.getcwd(),os.pardir)
    # load model
    # ckpt_path = "./experiments/2023_12_13_clip/ckpt/ckpt_epoch_14" 
    ckpt_path = "./experiments/2023_12_19_AE_clip/ckpt/ckpt_epoch_9" 
    ckpt_path = os.path.join(base, ckpt_path)
    model = load_model(ckpt_path, device)

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


    idx_1 = 7
    idx_2 = 5
    unit_waveform_1 = get_neuron_waveform(idx_1,mouse,probe,location,dates[0],experiment)
    unit_waveform_2 = get_neuron_waveform(idx_2,mouse,probe,location,dates[1],experiment)

    unit_waveform_1 = torch.from_numpy(unit_waveform_1).to(device)
    unit_waveform_2 = torch.from_numpy(unit_waveform_2).to(device)

    unit_waveform_1 = unit_waveform_1.unsqueeze(0)
    unit_waveform_2 = unit_waveform_2.unsqueeze(0)

    unit_waveform_11 = unit_waveform_1[...,0]
    unit_waveform_12 = unit_waveform_1[...,1]
    unit_waveform_21 = unit_waveform_2[...,0]
    unit_waveform_22 = unit_waveform_2[...,1]

    # get layer outputs
    x, layer_outputs_11 = model(unit_waveform_11)
    x, layer_outputs_12 = model(unit_waveform_12)
    x, layer_outputs_21 = model(unit_waveform_21)
    x, layer_outputs_22 = model(unit_waveform_22)

    # visualize paired layer outputs
    order = 1122
    visualize_paired_layer_outputs(layer_outputs_11,layer_outputs_22, mouse, idx_1, idx_2,order)
    # order = 1221
    # visualize_paired_layer_outputs(layer_outputs_12,layer_outputs_21, mouse, idx_1, idx_2,order)




