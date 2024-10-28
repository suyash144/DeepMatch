import numpy as np
import os, sys
sys.path.insert(0, os.getcwd())
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.stats import fisher_exact
import mat73
import h5py
from utils.myutil import mtpath_to_expids


def pairwise_histogram_correlation(A, B):
    # Initialize the output correlation matrix
    num_histograms = A.shape[0]
    correlation_matrix = np.zeros((num_histograms, num_histograms))

    # Compute pairwise correlations
    for i in range(num_histograms):
        for j in range(num_histograms):
            # Correlate the i-th histogram in A with the j-th histogram in B
            correlation_matrix[i, j] = np.corrcoef(A[i], B[j])[0, 1]
    
    return correlation_matrix


mouse = "AL031"
probe = "19011116684"
loc = "1"

test_data_root = os.path.join(os.path.dirname(os.getcwd()), "ALL_DATA")
server_root = r"\\znas\Lab\Share\UNITMATCHTABLES_ENNY_CELIAN_JULIE\FullAnimal_KSChanMap"
mt_path = os.path.join(test_data_root, mouse, probe, loc, "new_matchtable.csv")
mt = pd.read_csv(mt_path)
um_path = os.path.join(server_root, mouse, probe, loc, "UnitMatch", "UnitMatch.mat")
um = mat73.loadmat(um_path, verbose=False)
recsesAll = um["UniqueIDConversion"]["recsesAll"].astype(int)
goodid = um["UniqueIDConversion"]['GoodID'].astype(bool)
recses = recsesAll[goodid]
expids, metadata = mtpath_to_expids(mt_path, mt)
OriID = um["UniqueIDConversion"]["OriginalClusID"].astype(int)


plt.hist(mt['ISICorr'], bins=500)
plt.show()

recompute = True
nclus = np.sqrt(len(mt))
if not nclus.is_integer():
    print("Warning: Match table does not have the expected length. Likely to cause downstream errors.")
nclus = int(nclus)

if 'ISICorr' not in mt.columns or recompute:
    print("Computing ISI, this will take some time...")

    # Define ISI bins
    ISIbins = np.concatenate(([0], 5 * 10 ** np.arange(-4, 0.1, 0.1), [np.inf]))
    ISIMat = np.zeros((len(ISIbins) - 1, 2, nclus))
    insuff_spks = []
    insuff_fr = []

    for clusid in range(nclus):
        session = recses[clusid]
        exp = expids[session]
        spikes_path = os.path.join(os.path.dirname(mt_path), exp, "spikes.npy")
        with h5py.File(spikes_path, 'r') as f:
            clusters = f['spkclus'][()] 
            times = f['spktimes'][()]
            times = times/30000                     # divide by 30kHz sampling rate to get sample times in seconds
        for cv in range(2):
            idx1 = np.where(clusters == OriID[clusid])[0]
            if idx1.size > 0:
                if idx1.size < 50 and cv == 0:
                    print(f"Warning: Fewer than 50 spikes for neuron {clusid}, please check your inclusion criteria")
                # Split idx1 into two halves
                if cv == 0:
                    idx1 = idx1[:len(idx1) // 2]
                else:
                    idx1 = idx1[len(idx1) // 2:]
                nspkspersec, _ = np.histogram(times[idx1], bins=np.arange(min(times[idx1]), max(times[idx1]) + 1))
                ISIMat[:, cv, clusid], _ = np.histogram(np.diff(times[idx1].astype(float)), bins=ISIbins)
                if sum(ISIMat[:, cv, clusid]) == 0:
                    if len(idx1) < 2:
                        insuff_spks.append(clusid)
                    elif min(np.diff(times[idx1])) > 5:
                        insuff_fr.append(clusid)
    correlation_matrix = pairwise_histogram_correlation(ISIMat[:, 0, :].T, ISIMat[:, 1, :].T)
    # Correlation between ISIs
    # ISICorr = np.corrcoef(ISIMat[:, 0, :], ISIMat[:, 1, :], rowvar=False)

    # Saving results in the DataFrame
    mt.insert(len(mt.columns), 'newISI', correlation_matrix.ravel())

    print("DONE")