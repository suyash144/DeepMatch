
import numpy as np
import os, sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.pardir))
    sys.path.insert(0, os.path.join(os.pardir, os.pardir))
    from myutil import *
    from param_fun import *
    from visualize import *

def channel_shift_up(data):
    C = data.shape[1]  # Number of channels
    # Indices for odd channels, excluding the last one if C is odd
    odd_indices = np.arange(0, C - 1, 2)
    # Indices for even channels, excluding the last one
    even_indices = np.arange(1, C - 1, 2)
    # Shift odd channels up, excluding the last odd channel
    if len(odd_indices) > 1:  # Check if there are at least 2 odd channels to roll
        data[:, odd_indices[:-1]] = data[:, odd_indices[1:]]
    # Shift even channels up, excluding the last even channel
    if len(even_indices) > 1:  # Check if there are at least 2 even channels to roll
        data[:, even_indices[:-1]] = data[:, even_indices[1:]]
    return data

def channel_shift_down(data):
    # implement roll_down augmentation
    C = data.shape[1]  # Number of channels
    # Indices for odd channels, starting from the second one
    odd_indices = np.arange(2, C, 2)
    # Indices for even channels, starting from the second one if C > 1
    even_indices = np.arange(3, C, 2)
    # Shift odd channels down, keeping the top odd channel unchanged
    if len(odd_indices) > 0:  # Check if there are odd channels to roll
        data[:, odd_indices] = data[:, odd_indices - 2]
    # Shift even channels down, keeping the top even channel unchanged
    if len(even_indices) > 0:  # Check if there are even channels to roll
        data[:, even_indices] = data[:, even_indices - 2]
    return data


if __name__ == "__main__":
    base = os.path.join(os.getcwd(),os.pardir, os.pardir)
    # mouse = 'CB015'
    # probe = '19011110242'
    # location = '1'
    # date = '2021-09-10'
    # experiment = 'CB015_2021-09-10_NatImages_g0_t0-imec0-ap'

    mouse = 'AL032'
    probe = '19011111882'
    location = '1'
    date = '2019-11-08'
    experiment = 'AL032_2019-11-08_4shank_g0_t0-imec0-ap'

    experiment_path = os.path.join(base, 'DATA_UnitMatch', mouse, probe,location, date, experiment)
    good_units_value = read_good_id_from_mat(os.path.join(experiment_path, 'PreparedData.mat'))
    path_waveform = os.path.join(experiment_path,'RawWaveforms')
    good_units_files,good_units_indices = select_good_units_files_indices(path_waveform, good_units_value)
    # print('good_units_indices', good_units_indices)
    
    good_unit_idx = good_units_indices[16]
    filename = f"Unit{good_unit_idx}_RawSpikes.npy"
    path_neuron = os.path.join(path_waveform, filename)
    neuron_waveform = np.load(path_neuron)
    print(neuron_waveform.shape)

    ChannelPos = load_channel_positions(mouse,probe,location,date,experiment)
    ChannelMap = load_channel_map(mouse,probe,location,date,experiment)
    max_zPos = np.max(ChannelPos[:,1])
    min_zPos = np.min(ChannelPos[:,1])
    # print('max_zPos', max_zPos, 'min_zPos', min_zPos)

    params = get_default_param()
    MaxSiteMean, MaxSitepos,sorted_goodChannelMap,sorted_goodpos, Rwaveform = extract_Rwaveforms(neuron_waveform, ChannelPos, ChannelMap, params)
    # print('MaxSiteMean', MaxSiteMean)
    # print('MaxSitepos', MaxSitepos)
    # print('sorted_goodChannelMap', sorted_goodChannelMap)
    # print('sorted_goodpos', sorted_goodpos)
    print('Rwaveform', Rwaveform.shape)

    # visualize
    # plot_channel_activity_single(Rwaveform, mouse, probe, location, date, experiment, good_unit_idx, foldername = 'single',plot = False, save = True)
    plot_channel_activity_snippet(Rwaveform, mouse, probe, location, date, experiment, good_unit_idx, foldername = 'snippet', plot = False, save = True)

    Rwaveform_fh = Rwaveform[:,:,0]
    Rwaveform_sh = Rwaveform[:,:,1]

    # implement roll_up augmentation
    # AgRwaveform_fh = channel_shift_up(Rwaveform_fh)   
    # AgRwaveform_sh = channel_shift_up(Rwaveform_sh)
    # implement roll_down augmentation
    AgRwaveform_fh = channel_shift_down(Rwaveform_fh)
    AgRwaveform_sh = channel_shift_down(Rwaveform_sh)
    # concatenate to [T, C, 2]
    AgRwaveform = np.stack([AgRwaveform_fh, AgRwaveform_sh], axis = 2)

    # visualize
    # plot_channel_activity_single(AgRwaveform, mouse, probe, location, date, experiment, good_unit_idx, foldername = 'Agsingle',plot = False, save = True)
    plot_channel_activity_snippet(AgRwaveform, mouse, probe, location, date, experiment, good_unit_idx, foldername = 'Agsnippet', plot = False, save = True)