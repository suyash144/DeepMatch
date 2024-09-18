# This file will contain all the necessary functions for processing waveforms

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend
import scipy as sp
import h5py, os

import pandas as pd

def detrend_waveform(waveform):
    """
    This function accepts the raw waveforms (nTime, nChannels, CV)
    The output is the same shape, and linearly detrended across time.
    """ 
    return detrend(waveform, axis = 1)

def get_spatialfp(waveform):
    """
    Input: waveform np array (nTime, nChannels, first/second half)
    By taking the maximum value along the time axis of the absolute values
    Output: np array (nChannels, first/second half)
    spatialfp = spatial footprint
    """
    SpatialFP = np.max(np.abs(waveform), axis = 0)
    return SpatialFP

def get_max_site(SpatialFP):
    """
    Input: SpatialFP (nChannels, first/second half)
    By taking the index of maximum argument along the channel axis,
    Output: (first/second half), which gives the maximum site for each unit in first/second half
    """
    MaxSite = np.argmax(SpatialFP, axis = 0)
    return MaxSite


def sort_good_channels(goodChannelMap, goodpos):
    '''
    Sorts the good channels by their y-axis value and then by their z-axis value.
    '''
    # Step 1: Identify the unique y-axis values and sort them
    unique_y_values = np.unique(goodpos[:, 0])
    unique_y_values.sort()

    # Safety check: ensure there are exactly two unique y-axis values
    if len(unique_y_values) != 2:
        # TODO: adapt this code to be robust to Neuropixels 1.0 recordings as well as 2.0.
        # For now we are only using Neuropixels 2.0 data -> if we enter this block it means there was a mistake in spike sorting
        # Therefore the fix for now is to use return 0s so that we can handle this in extract_Rwaveforms

        # print(f"Channel Map: {goodChannelMap}")
        # print(f"Pos: {goodpos}")
        # raise ValueError(f"There should be exactly two unique y-axis values for Neuropixels 2.0 shank - instead got {len(unique_y_values)}: [{unique_y_values}]")
        return [-1],[-1]
    # Step 2: Split channels based on the y-axis value
    # channels_y_min = goodChannelMap[goodpos[:, 0] == unique_y_values[0]]
    # channels_y_max = goodChannelMap[goodpos[:, 0] == unique_y_values[1]]

    channels_y_min_indices = np.where(goodpos[:, 0] == unique_y_values[0])[0]
    channels_y_max_indices = np.where(goodpos[:, 0] == unique_y_values[1])[0]

    channels_y_min = goodChannelMap[channels_y_min_indices]
    channels_y_max = goodChannelMap[channels_y_max_indices]

    pos_y_min = goodpos[channels_y_min_indices]
    pos_y_max = goodpos[channels_y_max_indices]
    
    # Step 3: Sort each group by the z-axis value
    z_min_sorted_indices = np.argsort(goodpos[goodpos[:, 0] == unique_y_values[0], 1])
    z_max_sorted_indices = np.argsort(goodpos[goodpos[:, 0] == unique_y_values[1], 1])

    channels_y_min_sorted = channels_y_min[z_min_sorted_indices]
    channels_y_max_sorted = channels_y_max[z_max_sorted_indices]

    pos_y_min_sorted = pos_y_min[z_min_sorted_indices]
    pos_y_max_sorted = pos_y_max[z_max_sorted_indices]

    # Step 4: Interleave the channels from the two groups
    sorted_goodChannelMap = np.empty_like(goodChannelMap)
    sorted_goodChannelMap[::2] = channels_y_min_sorted[:len(sorted_goodChannelMap)//2]  # Even indices
    sorted_goodChannelMap[1::2] = channels_y_max_sorted[:len(sorted_goodChannelMap)//2]  # Odd indices
    sorted_goodpos = np.empty_like(goodpos)
    sorted_goodpos[::2, :] = pos_y_min_sorted
    sorted_goodpos[1::2, :] = pos_y_max_sorted
    return sorted_goodChannelMap, sorted_goodpos


def extract_Rwaveforms(waveform, ChannelPos,ChannelMap, param):
    """
    Using waveforms, ChannelPos and param, to find the max channel for each unit and cv, this function also
    returns good idx's / positions, by selecting channels within ChannelRadius (default 150 um) 
    Input: waveform (nTime, nChannels, CV), ChannelPos (nChannels, 2), param (dictionary)
    """

    nChannels = param['nChannels']
    nTime = param['nTime']
    RnChannels = param['RnChannels']
    RnTime = param['RnTime']
    ChannelRadius = param['ChannelRadius']
    # original time 0-82, new time 11-71
    start_time,end_time = (nTime - RnTime) // 2, (nTime + RnTime) // 2
    waveform = waveform[start_time:end_time,:,:] # selecting the middle 60 time points (11-71)
    waveform = detrend_waveform(waveform) # detrend the waveform
    MeanCV = np.mean(waveform, axis = 2) # average of each cv
    SpatialFootprint = get_spatialfp(MeanCV) # choose max time 
    MaxSiteMean = get_max_site(SpatialFootprint) # argument of MaxSite
    MaxSitepos = ChannelPos[MaxSiteMean,:] #gives the 2-d positions of the max sites

    # Finds the indices where the distance from the max site mean is small
    goodidx = np.empty(nChannels, dtype=bool)
    for i in range(ChannelPos.shape[0]): #looping over each site
        dist = np.linalg.norm(ChannelPos[MaxSiteMean,:] - ChannelPos[i,:])
        good = dist < ChannelRadius
        goodidx[i] = good

    goodChannelMap = ChannelMap[goodidx] #selecting the good channels
    goodpos = ChannelPos * np.tile(goodidx, (2,1)).T
    goodpos = goodpos[goodidx,:]
    sorted_goodChannelMap,sorted_goodpos = sort_good_channels(goodChannelMap, goodpos)
    # if sorted_goodChannelMap[0]==-1 and sorted_goodpos[0]==-1:
    #     # we have a spike sorting error so need to 0 this recording
    #     return np.array([-1,-1]), np.array([-1,-1]), [0], [0], np.zeros((1,1,1))
    Rwaveform = waveform[:, sorted_goodChannelMap, :] #selecting the good channels
    
    ## this part is tricks to make the data proper for DNN training
    GlobalMean = np.mean(Rwaveform) # mean of all channels and time points
    Rwaveform = Rwaveform - GlobalMean # subtracting the global mean, zero mean is good for DNN
    # padding the data to make it proper for DNN
    NewGlobalMean = np.mean(Rwaveform)
    z_sorted_goodpos = np.unique(sorted_goodpos[:,1])
    mean_z_sorted_goodpos = np.mean(z_sorted_goodpos)
    z_MaxSitepos = MaxSitepos[1]
    num_good_channels = np.sum(goodidx)
    # print('num_good_channels',num_good_channels)
    padding_needed = RnChannels - num_good_channels
    pad_before = 0
    pad_after = 0
    if z_MaxSitepos < mean_z_sorted_goodpos:
        pad_before = padding_needed  # Pad at the beginning if MaxSitepos is below the mean z position
    else:
        pad_after = padding_needed  # Pad at the end if MaxSitepos is above the mean z position
    Rwaveform = np.pad(Rwaveform, ((0, 0), (pad_before, pad_after), (0, 0)), 'constant', constant_values=(NewGlobalMean, NewGlobalMean))
    
    return MaxSiteMean, MaxSitepos, sorted_goodChannelMap, sorted_goodpos, Rwaveform


def save_waveforms_hdf5(dest_path, np_file_name, Rwaveform, MaxSitepos):
    """
    Saves the preprocessed, reduced waveform and the max site position as a HDF5 file.
    Saves in new_data_root/mouse/probe/loc/experiment_id/np_file_name
    """

    dest_path = os.path.join(dest_path, "processed_waveforms", np_file_name)
    dest_directory = os.path.dirname(dest_path)
    os.makedirs(dest_directory, exist_ok=True)

    new_data = {
        "waveform": Rwaveform,          # (60,30,2) 
        "MaxSitepos": MaxSitepos
    }
    with h5py.File(dest_path, 'w') as f:
        for key, value in new_data.items():
            f.create_dataset(key, data=value)
