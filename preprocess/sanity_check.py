

import os, sys
import numpy as np  
import h5py
import matplotlib.pyplot as plt

import torch


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.pardir, os.pardir))
    sys.path.insert(0, os.path.join(os.pardir))

from utils.myutil import *
from utils.param_fun import *

def vis_channel_positions(channel_positions,experiment):
    '''
    channel_positions: (n_channels,2), 2 for x and y
    '''
    plt.scatter(channel_positions[:,0],channel_positions[:,1],s=2)
    plt.title(f'Channel positions of {experiment}')
    plt.show()

# find a pair value in a 2D array
def find_pair_value(array, value):
    '''
    array: 2D array
    value: a pair value
    '''
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i,j] == value:
                return i,j
    return -1,-1

if __name__ == "__main__":

    mouse = 'EB037'
    probe = 'Probe1'
    location = '1'
    date = '2024-02-08'
    exp = '2024-02-08_exp1'
    mode = 'test'

    # Load channel positions and map
    ChannelPos = load_channel_positions(mouse,probe,location,date,exp,mode)
    # print('channelPos',ChannelPos)
    # vis_channel_positions(ChannelPos,exp)

    # # check the number that ChannelPos[:,i] ==2865
    print(np.sum(ChannelPos[:,1] == 2165.))
    # # find the pair that ChannelPos[i,j] == [0,2865]
    # target_pair = [0,2865]
    # count = np.sum(np.all(ChannelPos == target_pair, axis=1))
    # print('count',count)
    print('channelPos',ChannelPos)
    
    # ChannelMap = load_channel_map(mouse,probe,location,date,exp,mode)
    # print('channelMap',ChannelMap)
