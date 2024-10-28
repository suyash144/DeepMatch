import numpy as np
import os, sys
sys.path.insert(0, os.getcwd())
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.stats import fisher_exact
import mat73
import h5py
from tqdm import tqdm
from utils.myutil import mtpath_to_expids, get_exp_id
from utils.read_datapaths import read_datapaths


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
df = pd.DataFrame(read_datapaths(['AL031']))
df = df.loc[df["loc"]=='1']
exp_dict = {get_exp_id(df["recordings"][0][i], "AL031"):df["recordings"][0][i] for i in range(len(df["recordings"][0]))}
recompute = True
nclus = len(recses)
if nclus**2 != len(mt):
    print("Warning - number of good units is inconsistent with length of match table.")

if 'ISICorr' not in mt.columns or recompute:

    # Define ISI bins
    ISIbins = np.concatenate(([0], 5 * 10 ** np.arange(-4, 0.1, 0.1)))
    ISIMat = np.zeros((len(ISIbins) - 1, 2, nclus))

    for clusid in tqdm(range(nclus)):
        session = recses[clusid]
        exp = expids[session]
        # spikes_path = os.path.join(os.path.dirname(mt_path), exp, "spikes.npy")
        # with h5py.File(spikes_path, 'r') as f:
        #     clusters = f['spkclus'][()] 
        #     times = f['spktimes'][()]
        # times = times/30000                     # divide by 30kHz sampling rate to get sample times in seconds

        # LOAD PRE-PROCESSED DATA
        path = os.path.join(exp_dict[exp], "PreparedData.mat")
        prepdata = mat73.loadmat(path, verbose=False)
        clusters = prepdata["sp"]["clu"]
        times = prepdata["sp"]["st"]
        for cv in range(2):
            idx1 = np.where(clusters == OriID[goodid][clusid])[0]
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
    correlation_matrix = pairwise_histogram_correlation(ISIMat[:, 0, :].T, ISIMat[:, 1, :].T)
    # Correlation between ISIs
    # ISICorr = np.corrcoef(ISIMat[:, 0, :], ISIMat[:, 1, :], rowvar=False)

    # Saving results in the DataFrame
    mt.insert(len(mt.columns), 'newISI', correlation_matrix.ravel())
    plt.plot(mt.loc[:500,'newISI'])
    plt.plot(mt.loc[:500,'ISICorr'])
    plt.show()