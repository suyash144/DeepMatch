import numpy as np
import h5py
import matplotlib.pyplot as plt

waveform = r"\\zinu\Subjects\CB015\2021-09-13\ephys\CB015_2021-09-13_NatImages_g0\pyKS\output\qMetrics\RawWaveforms"

# np.load(r"C:\Users\suyas\test_R_DATA_UnitMatch\CB015\CB015_2021-09-11_breathingHPC_g0\Unit0_RawSpikes.npy")

# we can read these as hdf5 files. problem was loading them as npy arrays. 
f = h5py.File(r"C:\Users\suyas\test_R_DATA_UnitMatch\CB015\CB015_2021-09-11_breathingHPC_g0\Unit0_RawSpikes.npy", 'r')

print(list(f.keys()))