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

test_data_root = os.path.join(os.path.dirname(os.getcwd()), "ALL_DATA")
server_root = r"\\znas\Lab\Share\UNITMATCHTABLES_ENNY_CELIAN_JULIE\FullAnimal_KSChanMap"
mice = os.listdir(test_data_root)
df = pd.DataFrame(read_datapaths(mice))

# Define ISI bins
ISIbins = np.concatenate(([0], 5 * 10 ** np.arange(-4, 0.1, 0.1)))
for mouse in tqdm(mice):
    name_path = os.path.join(test_data_root, mouse)
    probes = os.listdir(name_path)
    for probe in probes:
        name_probe = os.path.join(name_path, probe)
        locations = os.listdir(name_probe)
        for loc in locations:
            name_probe_location = os.path.join(name_probe, loc)
            if not os.path.isdir(name_probe_location):
                continue
            mt_path = os.path.join(name_probe_location, "new_matchtable.csv")
            mt = pd.read_csv(mt_path)
            if 'newISI' in mt.columns:
                continue
            um_path = os.path.join(server_root, mouse, probe, loc, "UnitMatch", "UnitMatch.mat")
            um = mat73.loadmat(um_path, verbose=False)
            recsesAll = um["UniqueIDConversion"]["recsesAll"].astype(int)
            goodid = um["UniqueIDConversion"]['GoodID'].astype(bool)
            recses = recsesAll[goodid]
            expids, metadata = mtpath_to_expids(mt_path, mt)
            OriID = um["UniqueIDConversion"]["OriginalClusID"].astype(int)
            d = df.loc[(df["mouse"]==mouse)&(df["probe"]==probe)&(df["loc"]==loc)]
            recs = d['recordings'].item()
            exp_dict = {get_exp_id(recs[i], mouse):recs[i] for i in range(len(recs))}
            recompute = True
            nclus = len(recses)
            if nclus**2 != len(mt):
                print("Warning - number of good units is inconsistent with length of match table.")
            ISIMat = np.zeros((len(ISIbins) - 1, 2, nclus))
            for clusid in tqdm(range(nclus)):
                session = recses[clusid]
                exp = expids[session]
                spikes_path = os.path.join(os.path.dirname(mt_path), exp, "spikes.npy")
                with h5py.File(spikes_path, 'r') as f:
                    clusters = f['spkclus'][()]
                    times = f['spktimes'][()]
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
            correlation_matrix = np.tanh(0.5*np.arctanh(correlation_matrix) + 0.5*np.arctanh(correlation_matrix.T))

            # Saving results in the DataFrame
            mt.insert(len(mt.columns), 'newISI', correlation_matrix.ravel())
            mt.to_csv(mt_path)

            l = os.listdir(name_probe_location)
            for item in l:
                if item[:14]!="new_matchtable" or item=="new_matchtable.csv":
                    continue
                mt_path = os.path.join(name_probe_location, item)
                mt = pd.read_csv(mt_path)
                mt.insert(len(mt.columns), 'newISI', correlation_matrix.ravel())
                mt.to_csv(mt_path)

                
            