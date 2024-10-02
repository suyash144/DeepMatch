import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import dnn_dist
from sklearn.neighbors import KernelDensity

test_data_root = r"C:\Users\suyas\R_DATA_UnitMatch"

def compare_isi_with_dnnsim(mt_path:str):
    """
    Pass in path to the match table csv file.
    Shows the ISI correlation histogram for that set of experiments.
    """
    if not os.path.exists(mt_path):
        raise ValueError(f"Matchtable not found at {mt_path}")
    mt = pd.read_csv(mt_path)
    matches = np.sqrt(len(mt))
    if (matches).is_integer():
        matches = int(matches)
    else:
        raise ValueError("Length of matchtable is not (no. of neurons)^2")
    same_neuron = mt.loc[(mt["RecSes1"]==mt["RecSes2"]) & (mt["ID1"]==mt["ID2"]), ["DNNSim", "ISICorr"]]
    sorted = mt.sort_values(by = "DNNSim", ascending=False).head(matches)
    sorted = sorted.loc[:, ["DNNSim", "ISICorr"]]
    unsorted = mt.head(matches)
    assert len(same_neuron) == len(unsorted) == len(sorted)
    plt.hist(sorted["ISICorr"], bins = 500, label="Matches (as per DNNSim)", fc = (0, 0, 1, 0.9))
    plt.hist(same_neuron["ISICorr"], bins = 500, label="Comparing units to themselves", fc = (0, 1, 0, 0.5))
    plt.hist(unsorted["ISICorr"], bins = 500, label="Random selection", fc = (1, 0, 0, 0.5))
    plt.legend()
    plt.xlabel("ISI Correlation")
    plt.show()

def roc_curve_old(mt_path:str):
    """
    Input: Full absolute path to the matchtable with the DNNSim outputs you want to use.
    Match table must be fully filled in with DNNSim (every single row)
    """
    mt = pd.read_csv(mt_path)
    DNN_matches = mt.sort_values(by = "DNNSim", ascending=False)
    actual_matches = mt.loc[(mt["RecSes1"]==mt["RecSes2"]) & (mt["ID1"]==mt["ID2"]), ["RecSes1", "RecSes2", "ID1", "ID2"]]

    y = []
    x = []
    tp, fp, tn, fn = 0,0,0,0
    for m in tqdm(np.array_split(DNN_matches, 500)):
        new_tp = sum(actual_matches.index.isin(m.index))
        tp += new_tp
        fp += len(m) - new_tp
        fn = len(actual_matches) - tp
        tn = len(DNN_matches) - len(actual_matches) - fp
        recall = tp/(tp+fn)
        fpr = fp/(tn+fp)
        y.append(recall)
        x.append(fpr)
    x.append(1)
    y.append(y[-1])
    auc = np.trapz(y, x)
    print(auc)
    plt.plot(x,y)
    plt.ylabel("Recall")
    plt.xlabel("False positive rate")
    plt.grid()
    plt.show()

def threshold_isi(mt_path:str, normalise:bool=True, kde:bool=False):
    thresh = dnn_dist.get_threshold(mt_path, False)
    if not os.path.exists(mt_path):
        raise ValueError(f"Matchtable not found at {mt_path}")
    mt = pd.read_csv(mt_path)
    within = mt.loc[(mt["RecSes1"]==mt["RecSes2"]), ["DNNSim", "ISICorr", "ID1", "ID2"]]          # Only keep within-day bits
    across = mt.loc[(mt["RecSes1"]!=mt["RecSes2"]), ["DNNSim", "ISICorr"]]                        # Only keep across-day bits

    matches_across = across.loc[mt["DNNSim"]>=thresh, ["ISICorr"]]
    non_matches = within.loc[(mt["ID1"]!=mt["ID2"]), ["ISICorr"]]
    same_within = within.loc[(mt["ID1"]==mt["ID2"]), ["ISICorr"]]
    if kde:
        # Do kernel density estimation and plot distributions instead of histograms
        # This looks cleaner but runs slower
        m_a = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(matches_across)
        n_m = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(non_matches)
        s_w = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(same_within)
        x = np.linspace(min(non_matches["ISICorr"]), max(same_within["ISICorr"]), 1000).reshape(-1, 1)
        m_a = np.exp(m_a.score_samples(x))
        n_m = np.exp(n_m.score_samples(x))
        s_w = np.exp(s_w.score_samples(x))
        plt.plot(x, m_a, "r", label="Matches across days (above DNNSim threshold)")
        plt.plot(x, n_m, "b", label="Different units within days")
        plt.plot(x, s_w, "g", label="Same units within days")
    else:
        plt.hist(matches_across["ISICorr"], bins=500, alpha=0.5, fc = "red", density=normalise, label="Matches across days (above DNNSim threshold)")
        plt.hist(non_matches["ISICorr"], bins=500, alpha=0.5, fc = "blue", density=normalise, label="Different units within days")
        plt.hist(same_within["ISICorr"], bins=500, alpha=0.5, fc = "green", density=normalise, label="Same units within days")
    plt.legend()
    plt.xlabel("ISI correlation")
    plt.title("ISI correlation histogram")
    plt.show()

def roc_curve(mt_path:str):
    thresh = dnn_dist.get_threshold(mt_path, False)
    if not os.path.exists(mt_path):
        raise ValueError(f"Matchtable not found at {mt_path}")
    mt = pd.read_csv(mt_path)
    within = mt.loc[(mt["RecSes1"]==mt["RecSes2"]), ["DNNSim", "ISICorr", "ID1", "ID2"]]          # Only keep within-day bits
    across = mt.loc[(mt["RecSes1"]!=mt["RecSes2"]), ["DNNSim", "ISICorr"]]                        # Only keep across-day bits

    matches_across = across.loc[mt["DNNSim"]>=thresh, ["ISICorr"]]
    non_matches = within.loc[(mt["ID1"]!=mt["ID2"]), ["ISICorr"]]
    same_within = within.loc[(mt["ID1"]==mt["ID2"]), ["ISICorr"]]

    

# mt_path = os.path.join(test_data_root, "AL031", "19011116684", "1", "new_matchtable.csv")
# mt_path = os.path.join(test_data_root, "AL032", "19011111882", "2", "new_matchtable.csv")
mt_path = os.path.join(test_data_root, "AL036", "19011116882", "3", "new_matchtable.csv")       # 2497 neurons
# compare_isi_with_dnnsim(mt_path)
# roc_curve(mt_path)
threshold_isi(mt_path, normalise=True, kde=False)