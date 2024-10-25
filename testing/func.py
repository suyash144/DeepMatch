import numpy as np
import os
import pandas as pd
from scipy.stats import zscore
from scipy.stats import fisher_exact

test_data_root = os.path.join(os.path.dirname(os.getcwd()), "ALL_DATA")
mt_path = os.path.join(test_data_root, "AL031", "19011116684", "1", "new_matchtable.csv")
mt = pd.read_csv(mt_path)

recompute = True
nclus = np.sqrt(len(mt))

if 'ISICorr' not in mt.columns or recompute:
    print("Computing ISI, this will take some time...")

    # Define ISI bins
    ISIbins = np.concatenate(([0], 5 * 10 ** np.arange(-4, 0.1, 0.1)))
    ISIMat = np.zeros((len(ISIbins) - 1, 2, nclus))
    FR = np.zeros((2, nclus))

    for clusid in range(1,nclus+1):
        for cv in range(2):
            idx1 = np.where((sp.spikeTemplates == OriID[clusid]) & (sp.RecSes == recses[clusid]))[0]
            if idx1.size > 0:
                if idx1.size < 50 and cv == 0:
                    print(f"Warning: Less than 50 spikes for neuron {clusid}, please check your inclusion criteria")

                # Split idx1 into two halves
                if cv == 0:
                    idx1 = idx1[:len(idx1) // 2]
                else:
                    idx1 = idx1[len(idx1) // 2:]

                # Compute Firing rate
                nspkspersec, _ = np.histogram(sp.st[idx1], bins=np.arange(min(sp.st[idx1]), max(sp.st[idx1]) + 1))
                FR[cv, clusid] = np.nanmean(nspkspersec)

                # Compute ISI histogram
                ISIMat[:, cv, clusid], _ = np.histogram(np.diff(sp.st[idx1].astype(float)), bins=ISIbins)

    # Correlation between ISIs
    ISICorr = np.corrcoef(ISIMat[:, 0, :], ISIMat[:, 1, :])[0, 1]
    ISICorr = np.tanh(0.5 * np.arctanh(ISICorr) + 0.5 * np.arctanh(ISICorr))

    # Rank normalization and significance (assuming getRank is defined)
    ISIRank, ISISig = getRank(np.arctanh(ISICorr), SessionSwitch)

    # Saving results in the DataFrame
    mt['ISICorr'] = ISICorr.ravel()
    mt['ISIRank'] = ISIRank.ravel()
    mt['ISISig'] = ISISig.ravel()
