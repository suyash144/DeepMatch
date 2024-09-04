### see reconstructed waveforms from the trained autoencoder


import os,sys
import h5py
import matplotlib.pyplot as plt

import numpy as np
import torch

if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.pardir, os.pardir))
    sys.path.insert(0, os.path.join(os.pardir))

from utils.myutil import *
from utils.losses import *
from utils.visualize import *
from models.mymodel import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_AE(ckpt_path, device):
    model = SpatioTemporalAutoEncoder_V2(n_channel=30,n_time=60,n_output=256).to(device)
    model = model.double()
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()          
    return model

def load_mouse_data(mouse,probe,location,dates,exps,mode='test'):
    base_path = os.path.join(os.getcwd(),os.pardir,os.pardir)
    if mode == 'test':
        base_path = os.path.join(base_path,'test_R_DATA_UnitMatch')
    # first date
    base_path_1 = os.path.join(base_path,mouse,probe,location,dates[0],exps[0])
    waveform_path_1 = os.path.join(base_path_1,'RawWaveforms')
    good_units_index_1 = read_good_id_from_mat(os.path.join(base_path_1, 'PreparedData.mat'))
    good_units_files_1,good_units_indices_1 = select_good_units_files_indices(waveform_path_1, good_units_index_1)
    # second date
    base_path_2 = os.path.join(base_path,mouse,probe,location,dates[1],exps[1])
    waveform_path_2 = os.path.join(base_path_2,'RawWaveforms')
    good_units_index_2 = read_good_id_from_mat(os.path.join(base_path_2, 'PreparedData.mat'))
    good_units_files_2,good_units_indices_2 = select_good_units_files_indices(waveform_path_2, good_units_index_2)
    print('day 1 len', len(good_units_files_1))
    print('day 2 len', len(good_units_files_2))
    return good_units_files_1,good_units_indices_1,good_units_files_2,good_units_indices_2


def reconstruct_wavefroms(model,pseudo_nidx,good_units_files,good_units_indices,half=0):
    neuron_file = good_units_files[pseudo_nidx]
    neuron_idx = good_units_indices[pseudo_nidx]
    with h5py.File(neuron_file, 'r') as f:
        waveform = f['waveform'][()]
    # we have first half and second half average waveform, choose one
    waveform = waveform[...,half]
    # waveform = normalize_waveform(waveform)
    waveform = torch.from_numpy(waveform).to(device)
    with torch.no_grad():
        recon_waveform = model(waveform.unsqueeze(0).double())
    recon_waveform = recon_waveform.squeeze(0).detach().cpu().numpy()
    waveform = waveform.detach().cpu().numpy()
    return waveform, recon_waveform, neuron_idx

if __name__ == '__main__':
    base = os.path.join(os.getcwd(),os.pardir)
    # load model
    model_name = '2024_2_6_AE_SpatioTemporalAutoEncoder_V2'
    ckpt_path = os.path.join('AE_experiments', model_name, 'ckpt', 'ckpt_epoch_299')
    ckpt_path = os.path.join(base, ckpt_path)
    model = load_model_AE(ckpt_path, device)

    # load data
    ### Seen
    mouse = 'AL032'
    probe = 'Probe0'
    location = '1'
    dates = ['2019-11-21','2019-11-22']
    exps = ['exp1','exp1']
    session_pair = 1

    # ### Unseen
    # mouse = 'AL032'
    # probe = '19011111882'
    # location = '2'
    # dates = ['2019-12-03', '2019-12-04']
    # exps = ['AL032_2019-12-03_stripe192_natIm_g0_t0-imec0-ap', 'AL032_2019-12-04_stripe192_gratings_g0_t0-imec0-ap']
    # session_pair = 2

    good_units_files_1,good_units_indices_1,good_units_files_2,good_units_indices_2 = load_mouse_data(mouse,probe,location,dates,exps)
    pseudo_nidx = 50
  
    # day 1
    ## first half
    waveform_1, recon_waveform_1, neuron_idx = reconstruct_wavefroms(model,pseudo_nidx,good_units_files_1,good_units_indices_1,half=0)  
    date = dates[0]
    experiment = exps[0]
    foldername = 'AE'
    waveform_1 = waveform_1[...,np.newaxis]
    recon_waveform_1 = recon_waveform_1[...,np.newaxis]
    AE_waveform = np.concatenate((waveform_1, recon_waveform_1), axis=2)
    plot_channel_activity_single(AE_waveform,mouse,probe,location,date,experiment,neuron_idx,foldername)
    plot_channel_activity_snippet(AE_waveform,mouse,probe,location,date,experiment,neuron_idx,foldername)
    
    ## second half
    waveform_2, recon_waveform_2, neuron_idx = reconstruct_wavefroms(model,pseudo_nidx,good_units_files_1,good_units_indices_1,half=1)
    date = dates[0]
    experiment = exps[0]
    foldername = 'AE'
    waveform_2 = waveform_2[...,np.newaxis]
    recon_waveform_2 = recon_waveform_2[...,np.newaxis]
    
    original_waveform = np.concatenate((waveform_1, waveform_2), axis=2)
    foldername = 'Original'
    plot_channel_activity_single(original_waveform,mouse,probe,location,date,experiment,neuron_idx,foldername)
    plot_channel_activity_snippet(original_waveform,mouse,probe,location,date,experiment,neuron_idx,foldername)