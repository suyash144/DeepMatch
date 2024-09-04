# Wentao Qiu, 2023-10-06
# qiuwentao1212@gmail.com
# recommand visualization color=(220/255, 64/255, 26/255) and color=(78/255, 183/255, 233/255)

import os, sys
import numpy as np  
import matplotlib.pyplot as plt


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.pardir))
    sys.path.insert(0, os.path.join(os.pardir, os.pardir))
    from param_fun import *
else:
    from utils.param_fun import *

def plot_channel_position(position_data):
    plt.figure(figsize=(12,24))
    plt.scatter(position_data[:,0],position_data[:,1])

def plot_channel_activity_single(data,mouse,probe,location,date,experiment,unit_index,foldername,plot =False,save = True):
    '''plot a single channel activity, data is (nTime, nChannels, first/second half)'''
    MeanCV = np.mean(data, axis = 2) # average of each cv
    SpatialFootprint = get_spatialfp(MeanCV) # choose max time 
    MaxSiteMean = get_max_site(SpatialFootprint) # argument of MaxSite
    if foldername == 'AE':
        color1 = (51/255, 81/255, 227/255)
        color2 = (255/255, 165/255, 0/255)
        label1 = 'ori'
        label2 = 'rec'
    else:
        color1 = (220/255, 64/255, 26/255)
        color2 = (78/255, 183/255, 233/255)
        label1 = 'fh'
        label2 = 'sh'
    plt.figure()
    plt.plot(data[:,MaxSiteMean,0],color=color1,label = label1)
    plt.plot(data[:,MaxSiteMean,1],color=color2,label = label2)
    plt.legend()
    plt.xlabel('Time(ms)')
    plt.ylabel('Voltage(uV)')
    plt.tight_layout()
    # save to folder figures/mouse/date/probe/experiment
    if save:
        base = os.path.join(os.getcwd(),os.pardir)
        path_figures = os.path.join(base, 'figures',mouse,probe,location,date,experiment,foldername)
        if not os.path.exists(path_figures):
            os.makedirs(path_figures)   
        plt.savefig(os.path.join(path_figures,'Unit'+str(unit_index)+'_RawSpikes_sg.png'))
        print('Unit Activity Image Saved')
    if plot:
        plt.show()

def plot_channel_activity_snippet(data,mouse,probe,location,date,experiment,unit_index,foldername,plot =False,save = True):
    '''plot all-channel activity, data is (nTime, RnChannels, first/second half)'''
    RnChannels = data.shape[1]
    nRow = RnChannels//2
    nCol = 2
    fig, axs = plt.subplots(nRow, nCol, figsize=(12, 24), sharex=True, sharey=True)
    if foldername == 'AE':
        color1 = (51/255, 81/255, 227/255)
        color2 = (255/255, 165/255, 0/255)
    else:
        color1 = (220/255, 64/255, 26/255)
        color2 = (78/255, 183/255, 233/255)
    for channel_index in range(RnChannels):
        row = channel_index // nCol
        col = channel_index % nCol
        axs[row, col].plot(data[:,channel_index,0],color=color1)
        axs[row, col].plot(data[:,channel_index,1],color=color2)
        axs[row, col].axis('off')  # Optional: Hide axes
    plt.tight_layout()
    # save to folder figures/mouse/date/probe/experiment
    if save:
        base = os.path.join(os.getcwd(),os.pardir)
        path_figures = os.path.join(base, 'figures',mouse,probe,location,date,experiment,foldername)
        if not os.path.exists(path_figures):
            os.makedirs(path_figures)   
        plt.savefig(os.path.join(path_figures,'Unit'+str(unit_index)+'_RawSpikes.png'))
        print('Unit Activity Image Saved')
    if plot:
        plt.show()
