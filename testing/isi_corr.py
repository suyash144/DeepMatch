import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

test_data_root = r"C:\Users\suyas\R_DATA_UnitMatch"

def compare_isi(mt_path:str):
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

def roc_curve(mt_path:str):

    mt = pd.read_csv(mt_path)
    DNN_matches = mt.sort_values(by = "DNNSim", ascending=False)
    actual_matches = mt.loc[(mt["RecSes1"]==mt["RecSes2"]) & (mt["ID1"]==mt["ID2"]), ["RecSes1", "RecSes2", "ID1", "ID2"]]

    y = []
    x = []
    tp, fp, tn, fn = 0,0,0,0
    for m in tqdm(range(1, len(DNN_matches), 100)):
        preds = DNN_matches.head(m)
        tp = sum(actual_matches.index.isin(preds.index))
        fp = m - tp
        fn = sum(actual_matches.index.isin(preds.index)==False)
        tn = len(DNN_matches) - len(actual_matches) - fp
        recall = tp/(tp+fn)
        fpr = fp/(tn+fp)
        y.append(recall)
        x.append(fpr)
        if tp == len(actual_matches):
            break
    plt.plot(x,y)
    plt.show()

# mt_path = os.path.join(test_data_root, "AL031", "19011116684", "1", "new_matchtable.csv")
# mt_path = os.path.join(test_data_root, "AL032", "19011111882", "2", "new_matchtable.csv")
mt_path = os.path.join(test_data_root, "AL036", "19011116882", "3", "new_matchtable.csv")       # 2497 neurons
# compare_isi(mt_path)
roc_curve(mt_path)