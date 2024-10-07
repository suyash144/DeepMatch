import numpy as np
import pandas as pd
import os, sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import dnn_dist
from sklearn.neighbors import KernelDensity
import json
import mat73
if __name__ == '__main__':
    sys.path.insert(0, os.getcwd())

from utils.myutil import get_exp_id, exp_id_to_date
from utils.read_pos import read_pos


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
    if not os.path.exists(mt_path):
        raise ValueError(f"Matchtable not found at {mt_path}")
    mt = pd.read_csv(mt_path)
    thresh = dnn_dist.get_threshold(mt, metric="DNNSim", vis=False)
    within = mt.loc[(mt["RecSes1"]==mt["RecSes2"]), ["DNNSim", "ISICorr", "ID1", "ID2"]]          # Only keep within-day bits
    across = mt.loc[(mt["RecSes1"]!=mt["RecSes2"]), ["DNNSim", "ISICorr"]]                        # Only keep across-day bits

    diff = np.median(within["DNNSim"]) - np.median(across["DNNSim"])
    thresh = thresh - diff

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

def roc_curve(mt_path:str, dnn_metric:str="DNNSim", um_metric:str="TotalScore", 
              filter=True, dist_thresh=None, dc=True, one_pair=False):
    """
    dnn_metric can be "DNNSim" or "DNNProb".
    um_metric can be "TotalScore", "MatchProb" or "ScoreExclCentroid".
    filter is a bool that decides whether or not to do spatial filtering.
    thresh sets the threshold for spatial filtering (or if None then just reject worse half of matches)
    dc sets whether or not to do drift correction.
    If one_pair is True then only uses one pair of consecutive recordings to generate the curve. 
    Otherwise, uses all the pairs of recordings in the matchtable.
    """

    if not os.path.exists(mt_path):
        raise ValueError(f"Matchtable not found at {mt_path}")
    
    mt = pd.read_csv(mt_path)
    if one_pair:
        # To just test one pair of recordings on consecutive days:
        mt = mt.loc[(mt["RecSes1"].isin([4,5])) & (mt["RecSes2"].isin([4,5])),:]
    thresh = dnn_dist.get_threshold(mt, metric=dnn_metric, vis=False)
    if um_metric=="MatchProb":
        thresh_um=0.5
    else:
        if um_metric=="ScoreExclCentroid":
            col = mt.loc[:, "WavformSim":"LocTrajectorySim"]
            mt[um_metric] = col.mean(axis=1)
        thresh_um = dnn_dist.get_threshold(mt, metric=um_metric, vis=False)
    within = mt.loc[(mt["RecSes1"]==mt["RecSes2"]), [dnn_metric, "ISICorr", "ID1", "ID2", um_metric]]                                              # Only keep within-day bits
    across = mt.loc[(mt["RecSes1"]!=mt["RecSes2"]), [dnn_metric, "ISICorr", um_metric, "RecSes1", "RecSes2", "ID1", "ID2"]]                        # Only keep across-day bits

    # Correct for different median similarities between within- and across-day sets.
    diff = np.median(within[dnn_metric]) - np.median(across[dnn_metric])
    thresh = thresh - diff

    diff_um = np.median(within[um_metric])- np.median(across[um_metric])
    thresh_um = thresh_um - diff_um

    matches_across = across.loc[mt[dnn_metric]>=thresh, ["ISICorr", "RecSes1", "RecSes2", "ID1", "ID2"]]
    non_matches = within.loc[(mt["ID1"]!=mt["ID2"]), ["ISICorr"]]
    same_within = within.loc[(mt["ID1"]==mt["ID2"]), ["ISICorr"]]

    um_matches = across.loc[mt[um_metric]>=thresh_um, ["ISICorr"]]

    if filter:
        print("Spatial filtering...")
        print(f"Matches before spatial filtering: {len(matches_across)}")
        matches_across = spatial_filter(mt_path, matches_across, dist_thresh, dc, not one_pair)
        print("Filtered out bad matches (using Euclidean distances)")
        print(f"Matches after spatial filtering: {len(matches_across)}")

    sorted_within = within.sort_values(by = "ISICorr", ascending=False)
    sorted_across = across.sort_values(by = "ISICorr", ascending=False)

    tp_g, fp_g, tp_r, fp_r, tp_um, fp_um = 0,0,0,0,0,0
    N_w = len(non_matches)
    P_w = len(same_within)
    N_a = len(across) - len(matches_across)
    P_a = len(matches_across)
    N_um = len(across) - len(um_matches)
    P_um = len(um_matches)
    recall_g, fpr_g, recall_r, fpr_r, recall_um, fpr_um = [], [], [], [], [], []

    for idx, row in sorted_within.iterrows():
        if row["ID1"]==row["ID2"]:
            tp_g+=1
        else:
            fp_g+=1
        recall_g.append(tp_g/P_w)
        fpr_g.append(fp_g/N_w)
    for idx, row in sorted_across.iterrows():
        if idx in matches_across.index:
            tp_r+=1
        else:
            fp_r+=1
        if idx in um_matches.index:
            tp_um += 1
        else:
            fp_um += 1
        recall_r.append(tp_r/P_a)
        fpr_r.append(fp_r/N_a)
        recall_um.append(tp_um/P_um)
        fpr_um.append(fp_um/N_um)

    plt.plot(fpr_g, recall_g, "g", label="Same units within days")
    print(f"Green AUC = {np.trapz(recall_g, fpr_g)}")
    plt.plot(fpr_r, recall_r, "r", label=f"Matches across days (as per DNN {dnn_metric})")
    print(f"Red AUC = {np.trapz(recall_r, fpr_r)}")
    plt.plot(fpr_um, recall_um, "b", label=f"UnitMatch {um_metric}")
    print(f"Blue AUC = {np.trapz(recall_um, fpr_um)}")
        
    plt.grid()
    plt.legend()
    plt.show()

def auc_one_pair(mt:pd.DataFrame, rec1:int, rec2:int, dnn_metric:str="DNNSim", 
                 um_metric:str="TotalScore", dist_thresh=None):
    """
    Returns the AUC figures for DNN and UnitMatch when comparing across a pair of sessions.
    rec1 and rec2 are the RecSes IDs we want to compare.
    dnn_metric can be "DNNSim" or "DNNProb".
    um_metric can be "TotalScore", "MatchProb" or "ScoreExclCentroid".
    thresh sets the threshold for spatial filtering (or if None then just reject worse half of matches)
    """

    mt = mt.loc[(mt["RecSes1"].isin([rec1,rec2])) & (mt["RecSes2"].isin([rec1,rec2])),:]
    thresh = dnn_dist.get_threshold(mt, metric=dnn_metric, vis=False)
    if um_metric=="MatchProb":
        thresh_um=0.5
    else:
        if um_metric=="ScoreExclCentroid":
            col = mt.loc[:, "WavformSim":"LocTrajectorySim"]
            mt[um_metric] = col.mean(axis=1)
        thresh_um = dnn_dist.get_threshold(mt, metric=um_metric, vis=False)
    within = mt.loc[(mt["RecSes1"]==mt["RecSes2"]), [dnn_metric, "ISICorr", "ID1", "ID2", um_metric]]                                              # Only keep within-day bits
    across = mt.loc[(mt["RecSes1"]!=mt["RecSes2"]), [dnn_metric, "ISICorr", um_metric, "RecSes1", "RecSes2", "ID1", "ID2"]]                        # Only keep across-day bits

    # Correct for different median similarities between within- and across-day sets.
    diff = np.median(within[dnn_metric]) - np.median(across[dnn_metric])
    thresh = thresh - diff

    diff_um = np.median(within[um_metric]) - np.median(across[um_metric])
    thresh_um = thresh_um - diff_um

    matches_across = across.loc[mt[dnn_metric]>=thresh, ["ISICorr", "RecSes1", "RecSes2", "ID1", "ID2"]]
    um_matches = across.loc[mt[um_metric]>=thresh_um, ["ISICorr"]]
    matches_across = spatial_filter(mt_path, matches_across, dist_thresh, plot_drift=False)
    sorted_across = across.sort_values(by = "ISICorr", ascending=False)

    tp_r, fp_r, tp_um, fp_um = 0,0,0,0
    N_a = len(across) - len(matches_across)
    P_a = len(matches_across)
    N_um = len(across) - len(um_matches)
    P_um = len(um_matches)
    recall_r, fpr_r, recall_um, fpr_um = [], [], [], []

    for idx, row in sorted_across.iterrows():
        if idx in matches_across.index:
            tp_r+=1
        else:
            fp_r+=1
        if idx in um_matches.index:
            tp_um += 1
        else:
            fp_um += 1
        recall_r.append(tp_r/P_a)
        fpr_r.append(fp_r/N_a)
        recall_um.append(tp_um/P_um)
        fpr_um.append(fp_um/N_um)
    dnn_auc = np.trapz(recall_r, fpr_r)
    um_auc = np.trapz(recall_um, fpr_um)
    return dnn_auc, um_auc

def spatial_filter(mt_path:str, matches:pd.DataFrame, dist_thresh=None, drift_corr=True, plot_drift=True):
    """
    Input is a dataframe of potential matches (according to some threshold, e.g. DNNSim)
    Output is a reduced dataframe after filtering out matches that are spatially distant.
    dist_thresh can take a numerical value, or set it to None to just reject the 50% of matches 
    with greatest Euclidean distance.
    """

    sessions = set(matches["RecSes1"].unique())
    sessions = sessions.union(set(matches["RecSes2"].unique()))

    exp_folder = os.path.dirname(mt_path)
    for file in os.listdir(exp_folder):
        if not os.path.isdir(os.path.join(exp_folder, file)):
            continue
        else:
            some_exp = os.path.join(exp_folder, file)
            break
    with open(os.path.join(some_exp, "metadata.json")) as f:
        metadata = json.load(f)
    
    um_path = os.path.join(r"\\znas\Lab\Share\UNITMATCHTABLES_ENNY_CELIAN_JULIE\FullAnimal_KSChanMap", 
                    metadata["mouse"], metadata["probe"], metadata["loc"], "UnitMatch", "UnitMatch.mat")
    um = mat73.loadmat(um_path)
    paths = um["UMparam"]["KSDir"]
    exp_ids = {}
    for recses in sessions:
        exp_ids[recses] = get_exp_id(paths[recses-1], metadata["mouse"])
    test_data_root = mt_path[:mt_path.find(metadata["mouse"])]
    positions = {}
    for recses, exp_id in exp_ids.items():
        fp = os.path.join(test_data_root, metadata["mouse"], metadata["probe"], 
                          metadata["loc"], exp_id, "processed_waveforms")
        pos_dict = read_pos(fp)
        positions[recses] = pd.DataFrame(pos_dict)
    filtered_matches = matches.copy()
    if drift_corr:
        corrections = get_corrections(matches, positions)
        if plot_drift:
            days, drift = visualise_drift_correction(corrections, exp_ids, plot_drift)
    for idx, match in matches.iterrows():
        if drift_corr:
            dist = drift_corrected_dist(corrections, positions, match)
        else:
            dist = drift_corrected_dist(None, positions, match, True)
        filtered_matches.loc[idx, "dist"] = dist
        if dist_thresh and dist > dist_thresh:
            filtered_matches.drop(idx, inplace=True)
    if not dist_thresh:
        filtered_matches.sort_values(by = "dist", inplace=True)
        return filtered_matches.head(len(filtered_matches)//2)
    else:
        return filtered_matches

def get_corrections(matches, positions):

    drift_correct_dict = {}
    drift_correct_dict["rec1"] = []
    drift_correct_dict["rec2"] = []
    drift_correct_dict["ydiff"] = []
    for idx, row in tqdm(matches.iterrows(), desc="Building drift correction dataframe", total=len(matches)):
        recses1 = row["RecSes1"]
        recses2 = row["RecSes2"]
        pos1 = positions[recses1]
        pos2 = positions[recses2]
        y1 = pos1.loc[pos1["file"]==row["ID1"],"y"].item()
        y2 = pos2.loc[pos2["file"]==row["ID2"],"y"].item()
        dy = y2 - y1
        df = pd.DataFrame(drift_correct_dict)
        if sum((df["rec1"]==recses1) & (df["rec2"]==recses2)) > 0:
            ydiffs = df.loc[(df["rec1"]==recses1) & (df["rec2"]==recses2), "ydiff"].item()
            ydiffs.append(dy)
            drift_correct_dict = df.to_dict(orient="list")
        else:
            drift_correct_dict["rec1"].append(recses1)
            drift_correct_dict["rec2"].append(recses2)
            drift_correct_dict["ydiff"].append([dy])
    medians = [0] * len(drift_correct_dict["ydiff"])
    for i, l in enumerate(drift_correct_dict["ydiff"]):
        medians[i] = np.median(l)
    drift_correct_dict["ydiff"] = medians
    return pd.DataFrame(drift_correct_dict)

def drift_corrected_dist(corrections, positions, match, nocorr=False):
    """
    Returns Euclidean distance between two neurons in a pair.
    If nocorr is set to True, it does no drift correction.
    By default, nocorr is False so we do drift correction.
    """
    r1 = match["RecSes1"]
    r2 = match["RecSes2"]
    id1 = match["ID1"]
    id2 = match["ID2"]
    pos1_df = positions[r1]
    pos2_df = positions[r2]
    pos1 = pos1_df.loc[pos1_df["file"]==id1, ["x","y"]]
    pos2 = pos2_df.loc[pos2_df["file"]==id2, ["x","y"]]
    if nocorr:
        y2 = pos2["y"].item()
    else:
        correction = corrections.loc[(corrections["rec1"]==r1) & (corrections["rec2"]==r2), "ydiff"]
        y2 = pos2["y"].item() - correction
    dist = np.sqrt((pos1["x"].item() - pos2["x"].item())**2 + (pos1["y"].item() - y2)**2)
    return dist.item()

def visualise_drift_correction(corrections, exp_ids, vis):
    c = corrections[corrections["rec1"]==1]
    c = c.sort_values(by = "rec2")
    date0 = exp_id_to_date(exp_ids[1])
    delta_days = []
    for idx, row in c.iterrows():
        id = exp_ids[row["rec2"]]
        date = exp_id_to_date(id)
        delta = (date - date0).days
        delta_days.append(delta)

    if vis:
        plt.plot(delta_days, c["ydiff"])
        plt.xlabel("Days since first recording at this location")
        plt.ylabel("Drift")
        plt.show()
    return delta_days, c["ydiff"].values


test_data_root = os.path.join(os.path.dirname(os.getcwd()), "R_DATA_UnitMatch")
# mt_path = os.path.join(test_data_root, "AL031", "19011116684", "1", "new_matchtable.csv")
# mt_path = os.path.join(test_data_root, "AL032", "19011111882", "2", "new_matchtable.csv")
mt_path = os.path.join(test_data_root, "AL036", "19011116882", "3", "new_matchtable.csv")       # 2497 neurons
# compare_isi_with_dnnsim(mt_path)
# roc_curve(mt_path, dnn_metric="DNNSim", um_metric="MatchProb", one_pair=True, filter=True, dc=True)
# threshold_isi(mt_path, normalise=True, kde=True)
mt = pd.read_csv(mt_path)

dnn_auc, um_auc = auc_one_pair(mt, 1,2)
print(dnn_auc, um_auc)