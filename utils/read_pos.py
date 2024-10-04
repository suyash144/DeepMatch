import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
if __name__=="__main__":
    from myutil import get_unit_id
else:
    from utils.myutil import get_unit_id

def read_pos(path):
    files = os.listdir(path)
    x = []
    y = []
    filenames = []

    for file in files:
        fp = os.path.join(path, file)
        with h5py.File(fp, 'r') as f:
            # waveform = f['waveform'][()] 
            MaxSitepos = f['MaxSitepos'][()]
        x.append(MaxSitepos[0])
        y.append(MaxSitepos[1])
        filenames.append(get_unit_id(file))
    output = {"file":filenames, "x":x, "y":y}

    return output


if __name__=="__main__":
    path = r"C:\Users\suyas\R_DATA_UnitMatch\AL036\19011116882\3\_2020-02-14_ephys__2020-02-14_stripe240_audioVis_g0_PyKS_output\processed_waveforms"
    dic = read_pos(path)
    plt.hist(dic["x"])
    plt.hist(dic["y"])
    plt.show()
    plt.scatter(dic["x"],dic["y"])
    plt.show()