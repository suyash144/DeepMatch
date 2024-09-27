import numpy as np
import h5py
import os
import matplotlib.pyplot as plt

path = r"C:\Users\suyas\R_DATA_UnitMatch\AL036\19011116882\3\_2020-02-14_ephys__2020-02-14_stripe240_audioVis_g0_PyKS_output\processed_waveforms"

files = os.listdir(path)

x = []
y = []

for file in files:
    fp = os.path.join(path, file)
    with h5py.File(fp, 'r') as f:
        waveform = f['waveform'][()] 
        MaxSitepos = f['MaxSitepos'][()]

    x.append(MaxSitepos[0])
    y.append(MaxSitepos[1])
print(x)
print(y)

plt.hist(x)
plt.hist(y)
plt.show()