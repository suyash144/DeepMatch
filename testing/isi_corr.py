import numpy as np
import pandas as pd
import os, sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import dnn_dist
import seaborn as sns
import time
from scipy.stats import linregress
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LinearRegression
if __name__ == '__main__':
    sys.path.insert(0, os.getcwd())

from utils.myutil import get_exp_id, exp_id_to_date, mtpath_to_expids
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
                 um_metric:str="MatchProb", dist_thresh=None, mt_path=None, within50=True):
    """
    Returns the AUC figures for DNN and UnitMatch when comparing across a pair of sessions.
    rec1 and rec2 are the RecSes IDs we want to compare.
    dnn_metric can be "DNNSim" or "DNNProb".
    um_metric can be "TotalScore", "MatchProb" or "ScoreExclCentroid".
    thresh sets the threshold for spatial filtering (or if None then just reject worse half of matches)
    """

    mt = mt.loc[(mt["RecSes1"].isin([rec1,rec2])) & (mt["RecSes2"].isin([rec1,rec2])),:]
    if len(mt) < 40:
        return None, None, None, None
    try:
        thresh = dnn_dist.get_threshold(mt, metric=dnn_metric, vis=False)
    except:
        return None, None, None, None
    if um_metric=="MatchProb":
        thresh_um=0.5
    else:
        if um_metric=="ScoreExclCentroid":
            col = mt.loc[:, "WavformSim":"LocTrajectorySim"]
            mt[um_metric] = col.mean(axis=1)
        thresh_um = dnn_dist.get_threshold(mt, metric=um_metric, vis=False)
    within = mt.loc[(mt["RecSes1"]==mt["RecSes2"]), [dnn_metric, "ISICorr", "ID1", "ID2", um_metric, "RecSes1", "RecSes2"]]
    across = mt.loc[(mt["RecSes1"]!=mt["RecSes2"]), [dnn_metric, "ISICorr", um_metric, "RecSes1", "RecSes2", "ID1", "ID2"]]
    # Correct for different median similarities between within- and across-day sets.
    diff = np.median(within[dnn_metric]) - np.median(across[dnn_metric])
    thresh = thresh - diff
    diff_um = np.median(within[um_metric]) - np.median(across[um_metric])
    thresh_um = thresh_um - diff_um

    if within50:
        # Only consider pairs that are within 50 microns.
        across = spatial_filter(mt_path, across, 50, True, False)

    # Apply thresholds to generate matches for DNN and UnitMatch respectively
    matches_across = across.loc[mt[dnn_metric]>=thresh, ["ISICorr", "RecSes1", "RecSes2", "ID1", "ID2", dnn_metric]]
    um_matches = across.loc[mt[um_metric]>=thresh_um, ["ISICorr", "RecSes1", "RecSes2", "ID1", "ID2", um_metric]]
    # Only allow a match if it is above threshold when comparing in both directions
    matches_across = directional_filter(matches_across)
    um_matches = directional_filter(um_matches)
    if len(matches_across)==0:
        print("no DNN matches found!")
        return None, None, None, None
    # Remove split units from each set of matches
    matches_across = remove_split_units(mt_path, within, matches_across, thresh, "DNNSim")
    um_matches = remove_split_units(mt_path, within, um_matches, thresh_um, "MatchProb")
    # Do spatial filtering in DNN
    matches_across = spatial_filter(mt_path, matches_across, dist_thresh, plot_drift=False)
    # Resolve conflict matches by only keeping the match with highest DNNSim
    matches_across, dnn_conflicts = remove_conflicts(matches_across, dnn_metric)
    um_matches, um_conflicts = remove_conflicts(um_matches, um_metric)

    # Calculate AUCs using final sets of matches
    sorted_across = across.sort_values(by = "ISICorr", ascending=False)
    discard_DNN, discard_UM = False, False
    tp_r, fp_r, tp_um, fp_um = 0,0,0,0
    N_a = len(across) - len(matches_across)
    P_a = len(matches_across)
    N_um = len(across) - len(um_matches)
    P_um = len(um_matches)
    if P_a < 40:
        # fewer than 40 matches found by DNN - discard the AUC for this session pair
        discard_DNN = True
        P_a = 100
    if P_um < 40:
        # same for UnitMatch. Set P values to 100 to avoid division by 0 errors.
        discard_UM = True
        P_um = 100
    recall_r, fpr_r, recall_um, fpr_um = [], [], [], []
    if not discard_DNN or not discard_UM:
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
    if discard_UM:
        um_auc = None
        P_um = 0
    else:
        um_auc = np.trapz(recall_um, fpr_um)
    if discard_DNN:
        dnn_auc = None
        P_a = 0
    else:
        dnn_auc = np.trapz(recall_r, fpr_r)
    return dnn_auc, um_auc, P_a, P_um

def spatial_filter(mt_path:str, matches:pd.DataFrame, dist_thresh=None, drift_corr=True, plot_drift=False):
    """
    Input is a dataframe of potential matches (according to some threshold, e.g. DNNSim)
    Output is a reduced dataframe after filtering out matches that are spatially distant.
    dist_thresh can take a numerical value, or set it to None to just reject the 50% of matches 
    with greatest Euclidean distance.

    Args:
        mt_path (str): Path to the matchtable.
        matches (pd.DataFrame): DataFrame containing potential matches.
        dist_thresh (Optional[float]): Maximum allowed Euclidean distance. If None, filters out the 50% most distant matches.
        drift_corr (bool): Whether to apply drift correction. Defaults to True.
        plot_drift (bool): Whether to plot drift correction visualization. Defaults to False.
    """
    pos_start = time.time()
    exp_ids, metadata = mtpath_to_expids(mt_path, matches)
    test_data_root = mt_path[:mt_path.find(metadata["mouse"])]
    positions = {}
    for recses, exp_id in exp_ids.items():
        fp = os.path.join(test_data_root, metadata["mouse"], metadata["probe"], 
                          metadata["loc"], exp_id, "processed_waveforms")
        pos_dict = read_pos(fp)
        positions[recses] = pd.DataFrame(pos_dict)
    if drift_corr:
        corrections = get_corrections(matches, positions)
    # plot_distances(matches, positions, corrections=corrections)
    matches_with_dist = vectorized_drift_corrected_dist(corrections, positions, matches)
    if not dist_thresh:
        matches_with_dist.sort_values(by = "dist", inplace=True)
        return matches_with_dist.head(len(matches)//2)
    else:
        return matches_with_dist.loc[matches_with_dist["dist"]<dist_thresh]

def directional_filter(matches: pd.DataFrame):
    filtered_matches = matches.copy()
    for idx, row in matches.iterrows():
        i1 = row["ID1"]
        i2 = row["ID2"]
        r1 = row["RecSes1"]
        r2 = row["RecSes2"]

        # Check for the reverse match
        reverse_match = matches.loc[
            (matches["RecSes1"] == r2) & (matches["ID1"] == i2) & 
            (matches["RecSes2"] == r1) & (matches["ID2"] == i1)
        ]

        if reverse_match.empty:
            filtered_matches = filtered_matches.drop(idx)  # Drop if no reverse match is found
    return filtered_matches

def remove_split_units(mt_path:str, within:pd.DataFrame, matches:pd.DataFrame, threshold, metric):
    """
    Takes a dataframe of matches and returns a smaller dataframe with potential split units removed.
    Slightly drastic approach but necessary for the ISI correlation comparisons to make sense.

    Args:
        mt_path: Path to the matchtable csv file for this location.
        within: Dataframe containing only the pairs within days (necessary to identify split units)
        matches: The dataframe of matches which we want to edit.
        threshold: The threshold used to define what is a match and what isn't.
        metric: The metric used to define the threshold above. Usually DNNSim for the DNN method and MatchProb for UM.
    """

    matches_within = within.loc[within[metric]>=threshold, ["ISICorr", "ID1", "ID2", "RecSes1", "RecSes2"]]
    matches_within = directional_filter(matches_within)
    off_diag = matches_within.loc[matches_within["ID1"]!=matches_within["ID2"]]
    split_units = []
    for idx, row in off_diag.iterrows():
        i1 = row["ID1"]
        i2 = row["ID2"]
        r1 = row["RecSes1"]
        r2 = row["RecSes2"]
        split_units.append((r1,i1))
        split_units.append((r2,i2))
    for idx, row in matches.iterrows():
        i1 = row["ID1"]
        i2 = row["ID2"]
        r1 = row["RecSes1"]
        r2 = row["RecSes2"]
        if (r1,i1) in split_units or (r2,i2) in split_units:
            matches = matches.drop(idx)
    return matches

def remove_conflicts(matches:pd.DataFrame, metric:str):
    indices_to_drop = []
    for idx, match in matches.iterrows():
        id1 = match["ID1"]
        id2 = match["ID2"]
        r1 = match["RecSes1"]
        r2 = match["RecSes2"]

        neuron1=matches.loc[(matches["ID1"]==id1) & (matches["RecSes1"]==r1),:]
        if len(neuron1)>1:
            neuron1 = neuron1.sort_values(by=[metric], ascending=False)
            indices_to_drop += neuron1.tail(len(neuron1)-1).index.to_list()
        
        neuron2=matches.loc[(matches["ID2"]==id2) & (matches["RecSes2"]==r2),:]
        if len(neuron2)>1:
            neuron2 = neuron2.sort_values(by=[metric], ascending=False)
            indices_to_drop += neuron2.tail(len(neuron2)-1).index.to_list()
    indices_to_drop = list(set(indices_to_drop))
    return matches.drop(indices_to_drop), len(indices_to_drop)

def get_corrections(matches, positions):
    drift_correct_dict = {"rec1": [], "rec2": [], "shank": [], "ydiff": []}
    for idx, row in matches.iterrows():
        recses1 = row["RecSes1"]
        recses2 = row["RecSes2"]
        pos1 = positions[recses1]
        pos2 = positions[recses2]
        # Get x, y values
        pos1_vals = pos1.loc[pos1["file"] == row["ID1"], ["x", "y"]]
        pos2_vals = pos2.loc[pos2["file"] == row["ID2"], ["x", "y"]]
        if pos1_vals.empty or pos2_vals.empty:
            # Handle missing data cases
            continue
        y1, x1 = pos1_vals.iloc[0]["y"], pos1_vals.iloc[0]["x"]
        y2, x2 = pos2_vals.iloc[0]["y"], pos2_vals.iloc[0]["x"]
        shank1 = int(round(x1, -2))
        shank2 = int(round(x2, -2))
        if shank1 != shank2:
            continue  # Skip if shank mismatch
        dy = y2 - y1
        # Check if this session pair and shank already exist
        existing_idx = None
        for i, (r1, r2, s) in enumerate(zip(drift_correct_dict["rec1"], drift_correct_dict["rec2"], drift_correct_dict["shank"])):
            if r1 == recses1 and r2 == recses2 and s == shank1:
                existing_idx = i
                break
        if existing_idx is not None:
            drift_correct_dict["ydiff"][existing_idx].append(dy)
        else:
            drift_correct_dict["rec1"].append(recses1)
            drift_correct_dict["rec2"].append(recses2)
            drift_correct_dict["shank"].append(shank1)
            drift_correct_dict["ydiff"].append([dy])
    # Calculate medians for each ydiff list
    drift_correct_dict["ydiff"] = [np.median(l) for l in drift_correct_dict["ydiff"]]
    output = pd.DataFrame(drift_correct_dict)
    return output

def vectorized_drift_corrected_dist(corrections, positions, matches:pd.DataFrame, nocorr=False):
    """
    Vectorized drift-corrected distance computation.
    """

    # Extract the x, y positions
    matches = matches.assign(
        x1=matches.apply(lambda row: positions[row['RecSes1']].loc[positions[row['RecSes1']]['file'] == row['ID1'], 'x'].values[0], axis=1),
        y1=matches.apply(lambda row: positions[row['RecSes1']].loc[positions[row['RecSes1']]['file'] == row['ID1'], 'y'].values[0], axis=1),
        x2=matches.apply(lambda row: positions[row['RecSes2']].loc[positions[row['RecSes2']]['file'] == row['ID2'], 'x'].values[0], axis=1),
        y2=matches.apply(lambda row: positions[row['RecSes2']].loc[positions[row['RecSes2']]['file'] == row['ID2'], 'y'].values[0], axis=1)
    )

    # Calculate shanks (rounded x coordinates)
    matches['shank1'] = matches['x1'].round(-2).astype(int)
    matches['shank2'] = matches['x2'].round(-2).astype(int)

    # Apply corrections where necessary
    if not nocorr:
        matches["idx"] = matches.index
        matches = matches.merge(
            corrections[['rec1', 'rec2', 'shank', 'ydiff']], 
            how='left', 
            left_on=['RecSes1', 'RecSes2', 'shank1'], 
            right_on=['rec1', 'rec2', 'shank']
        )
        matches.index = matches["idx"]
        matches.drop(['rec1', 'rec2', 'shank', 'idx'], axis=1, inplace=True)
        # Apply the drift correction conditionally
        matches['y2_corr'] = np.where(
            matches['shank1'] != matches['shank2'], 
            1000, 
            matches['y2'] - matches['ydiff'].fillna(0)
        )
    else:
        matches['y2_corr'] = matches['y2']

    # Calculate the Euclidean distance
    matches['dist'] = np.sqrt((matches['x1'] - matches['x2'])**2 + (matches['y1'] - matches['y2_corr'])**2)

    return matches

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
    x1 = pos1["x"].item()
    x2 = pos2["x"].item()
    shank1 = int(round(x1, -2))
    shank2 = int(round(x2, -2))
    if shank1 != shank2 and not nocorr:
        return 1000
    if nocorr:
        y2 = pos2["y"].item()
    else:
        correction = corrections.loc[(corrections["rec1"]==r1) & (corrections["rec2"]==r2) 
                                     & (corrections["shank"]==shank1), "ydiff"]
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

def auc_over_days(mt_path:str, vis:bool, within50:bool=True):
    mt = pd.read_csv(mt_path)
    sessions = set(mt["RecSes1"].unique())
    dnn_auc, um_auc, delta_days_d, delta_days_u, numbers_d, numbers_u = [], [], [], [], [], []
    exp_ids,_ = mtpath_to_expids(mt_path, mt)
    for r1 in tqdm(sessions):
        for r2 in tqdm(sessions):
            if r1>=r2:
                continue
            dnn, um, n_dnn, n_um = auc_one_pair(mt, r1, r2, mt_path=mt_path, dist_thresh=20, within50=within50)
            date1 = exp_id_to_date(exp_ids[r1])
            date2 = exp_id_to_date(exp_ids[r2])
            if dnn is not None and um is not None:
                dnn_auc.append(dnn)
                delta_days_d.append((date2-date1).days)
                numbers_d.append(n_dnn)
            # if um is not None:
                um_auc.append(um)
                delta_days_u.append((date2-date1).days)
                numbers_u.append(n_um)
    if vis:
        plt.scatter(delta_days_d, dnn_auc, c="r", label="DNN", s=numbers_d, alpha=0.3)
        plt.scatter(delta_days_u, um_auc, c="b", label="UM", s=numbers_u, alpha=0.3)
        plt.xlabel("Delta days")
        plt.ylabel("AUC")
        plt.ylim((0.36,1.05))
        plt.legend()
        model1 = LinearRegression()
        model1.fit(np.array(delta_days_d).reshape(-1,1), dnn_auc, sample_weight=numbers_d)
        line1 = model1.predict(np.array(delta_days_d).reshape(-1,1))
        model2 = LinearRegression()
        model2.fit(np.array(delta_days_u).reshape(-1,1), um_auc, sample_weight=numbers_u)
        line2 = model2.predict(np.array(delta_days_u).reshape(-1,1))
        plt.plot(delta_days_d, line1, label="DNN", color="r")
        plt.plot(delta_days_u, line2, label="UM", color="b")
        # sns.regplot(x = delta_days_d, y = dnn_auc, label="DNN", color="r")
        # sns.regplot(x = delta_days_u, y = um_auc, label="UM", color="b")
        plt.show()
    if len(set(delta_days_d)) > 1:
        dnn_slope, dnn_intercept, dnn_r, dnn_p, dnn_std = linregress(delta_days_d, dnn_auc)
    else:
        dnn_slope, dnn_intercept = None, None
    if len(set(delta_days_u)) > 1:
        um_slope, um_intercept, um_r, um_p, um_std = linregress(delta_days_u, um_auc)
    else:
        um_slope, um_intercept = None, None
    return dnn_slope, dnn_intercept, um_slope, um_intercept

def plot_distances(matches:pd.DataFrame, positions, rec1=None, rec2=None, corrections=None):
    if corrections is None:
        nocorr= True
    else:
        nocorr = False
    dists = []
    if rec1 is not None:
        matches = matches.loc[matches["RecSes1"].isin([rec1,rec2]) & matches["RecSes2"].isin([rec1,rec2]),:]
    for idx, row in matches.iterrows():
        dist = drift_corrected_dist(corrections, positions, row, nocorr)
        dists.append(dist)
    # plt.hist(dists, bins = 100)
    # plt.show()
    return dists

def plot_dist_thresh(mt_path, matches:pd.DataFrame, positions, corrections):
    match = matches.sample(1)
    mt = pd.read_csv(mt_path)

    # thresholds = [10,50,100,150,200,250]
    thresholds = [100]

    dists = []
    dists_dc = []

    fig, axs = plt.subplots(nrows=1, ncols=3)
    i = 0
    r1 = match["RecSes1"].item()
    r2 = match["RecSes2"].item()
    dists.append(plot_distances(matches, positions, r1, r2, None))
    dists_dc.append(plot_distances(matches, positions, r1, r2, corrections))
    aucs = []
    print("got to here")
    for t in tqdm(thresholds):
        dnn, um = auc_one_pair(mt, r1, r2, dist_thresh=t)
        aucs.append(dnn)
        print("done one threshold")
    axs[i,0].hist(dists)
    axs[i,0].set_title("Distance histogram of matches")
    axs[i,1].hist(dists_dc)
    axs[i,1].set_title("After drift correction")
    axs[i,2].plot(thresholds, aucs, color = "r", label="DNN AUCs")
    axs[i,2].axhline(um, label="UnitMatch AUC", color="b")
    axs[i,2].set_title("AUCs over varying distance thresholds")
    i +=1
    print("finished making plots")
    plt.show()

def all_mice_auc_over_days(test_data_root:str):
    mice = os.listdir(test_data_root)
    dnn_a, dnn_b, um_a, um_b = [], [], [], []
    fails = []
    insufficient_matches = []
    for mouse in tqdm(mice):
        mouse_path = os.path.join(test_data_root, mouse)
        probes = os.listdir(mouse_path)
        for probe in tqdm(probes):
            probe_path = os.path.join(mouse_path, probe)
            locs = os.listdir(probe_path)
            for loc in tqdm(locs):
                mt_path = os.path.join(test_data_root, mouse, probe, loc, "new_matchtable.csv")
                # da, db, ua, ub = auc_over_days(mt_path, vis=False)
                try:
                    da, db, ua, ub = auc_over_days(mt_path, vis=False)
                    if da is not None:
                        dnn_a.append(da)
                        dnn_b.append(db)
                    else:
                        insufficient_matches.append((mouse, probe, loc, "DNN"))
                        print(f"Not enough DNN matches for any session pairs in {mouse, probe, loc}")
                    if ua is not None:
                        um_a.append(ua)
                        um_b.append(ub)
                    else:
                        insufficient_matches.append((mouse, probe, loc, "UM"))
                        print(f"Not enough UM matches for any session pairs in {mouse, probe, loc}")
                    print(f"Done with {mouse, probe, loc}")
                except:
                    print(f"FAIL - An error has occured for {mouse, probe, loc}")
                    fails.append((mouse, probe, loc))
    print("Fails: ", fails)
    print("Insufficient matches: ", insufficient_matches)
    plt.scatter(dnn_a, dnn_b, label="DNN", color="r")
    plt.scatter(um_a, um_b, label="UM", color="b")
    plt.legend()
    plt.grid()
    plt.xlabel("Gradient of AUC over days")
    plt.ylabel("y-intercept")
    plt.show()
    return dnn_a, dnn_b, um_a, um_b

def ext_data_fig5(mt_path:str, name):
    mt = pd.read_csv(mt_path)
    sessions = set(mt["RecSes1"].unique())
    dnn_auc, um_auc, day1_d, day1_u, day2_d, day2_u, numbers_d, numbers_u = [], [], [], [], [], [], [], []
    exp_ids,_ = mtpath_to_expids(mt_path, mt)
    for r1 in tqdm(sessions):
        for r2 in tqdm(sessions):
            if r1>=r2:
                continue
            dnn, um, n_dnn, n_um = auc_one_pair(mt, r1, r2, mt_path=mt_path, dist_thresh=20)
            date1 = exp_id_to_date(exp_ids[r1])
            date2 = exp_id_to_date(exp_ids[r2])
            if dnn is not None:
                dnn_auc.append(dnn)
                day1_d.append(date1)
                day2_d.append(date2)
                numbers_d.append(n_dnn)
            if um is not None:
                um_auc.append(um)
                day1_u.append(date1)
                day2_u.append(date2)
                numbers_u.append(n_um)
    dnn = pd.DataFrame({"DNN_day1": pd.to_datetime(day1_d), "DNN_day2": pd.to_datetime(day2_d), "DNNauc": dnn_auc, "DNN_n":numbers_d})
    um = pd.DataFrame({"UM_day1": pd.to_datetime(day1_u), "UM_day2":pd.to_datetime(day2_u), "UMauc": um_auc, "UM_n": numbers_u})

    day0_dnn = min(min(dnn['DNN_day1']), min(dnn['DNN_day2']))
    day0_um = min(min(um['UM_day1']), min(um['UM_day2']))
    dnn['DNN_day1'] = (dnn['DNN_day1'] - day0_dnn).dt.days
    dnn['DNN_day2'] = (dnn['DNN_day2'] - day0_dnn).dt.days
    um['UM_day1'] = (um['UM_day1'] - day0_um).dt.days
    um['UM_day2'] = (um['UM_day2'] - day0_um).dt.days

    if len(dnn["DNN_day1"].unique()) >= len(dnn["DNN_day2"].unique()):
        days = dnn["DNN_day1"].unique()
    else:
        days = dnn["DNN_day2"].unique()
    auc = np.zeros((len(days), len(days)))
    n_matches = np.zeros((len(days), len(days)))
    days.sort()
    for idx, row in dnn.iterrows():
        i = np.where(days==row["DNN_day1"])
        j = np.where(days==row["DNN_day2"])
        auc[i,j] = row["DNNauc"]
        n_matches[i,j] = row["DNN_n"]

    if len(um["UM_day1"].unique()) >= len(um["UM_day2"].unique()):
        days = um["UM_day1"].unique()
    else:
        days = um["UM_day2"].unique()
    aucum = np.zeros((len(days), len(days)))
    n_matchesum = np.zeros((len(days), len(days)))
    days.sort()
    for idx, row in um.iterrows():
        i = np.where(days==row["UM_day1"])
        j = np.where(days==row["UM_day2"])
        aucum[i,j] = row["UMauc"]
        n_matchesum[i,j] = row["UM_n"]

    min_auc = min(auc.min(), aucum.min())
    max_auc = max(auc.max(), aucum.max())
    min_n = min(n_matches.min(), n_matchesum.min())
    max_n = max(n_matches.max(), n_matchesum.max()) 
    fig = plt.figure()
    ax1 = fig.add_subplot(222)
    cax1 = ax1.matshow(auc, vmin=min_auc, vmax=max_auc)
    ax1.set_title("AUC as per ISI correlation")
    ax2 = fig.add_subplot(221)
    cax2 = ax2.matshow(n_matches, vmin=min_n, vmax=max_n)
    ax2.set_title("Number of matches found")
    fig.colorbar(cax1,fraction=0.046, pad=0.04)
    fig.colorbar(cax2,fraction=0.046, pad=0.04)
    ax3 = fig.add_subplot(224)
    cax3 = ax3.matshow(aucum, vmin=min_auc, vmax=max_auc)
    ax3.set_title("AUC as per ISI correlation")
    ax4 = fig.add_subplot(223)
    cax4 = ax4.matshow(n_matchesum, vmin=min_n, vmax=max_n)
    ax4.set_title("Number of matches found")
    fig.colorbar(cax3,fraction=0.046, pad=0.04)
    fig.colorbar(cax4,fraction=0.046, pad=0.04)
    fig.tight_layout()
    results_fig_folder = r"C:\Users\suyas\results_figs"
    save_path = os.path.join(results_fig_folder, name)
    plt.savefig(save_path+'.png', format='png')


if __name__ == "__main__":
    test_data_root = os.path.join(os.path.dirname(os.getcwd()), "R_DATA_UnitMatch")
    # test_data_root = os.path.join(os.path.dirname(os.getcwd()), "scratch_data")
    # mt_path = os.path.join(test_data_root, "AL031", "19011116684", "1", "new_matchtable.csv")
    # mt_path = os.path.join(test_data_root, "AL032", "19011111882", "2", "new_matchtable.csv")
    mt_path = os.path.join(test_data_root, "AL036", "19011116882", "3", "new_matchtable.csv")       # 2497 neurons
    # mt_path = os.path.join(test_data_root, "AV009", "Probe1", "IMRO_6", "new_matchtable.csv")
    # compare_isi_with_dnnsim(mt_path)
    # roc_curve(mt_path, dnn_metric="DNNSim", um_metric="MatchProb", filter=True, dc=True)
    # threshold_isi(mt_path, normalise=True, kde=True)
    # mt = pd.read_csv(mt_path)
    # mt_paths = []
    # # mt_paths.append(os.path.join(test_data_root, "AL031", "19011116684", "1", "new_matchtable.csv"))
    # mt_paths.append(os.path.join(test_data_root, "AL032", "19011111882", "2", "new_matchtable.csv"))
    # mt_paths.append(os.path.join(test_data_root, "AL036", "19011116882", "3", "new_matchtable.csv"))
    # mt_paths.append(os.path.join(test_data_root, "AV008", "Probe0", "IMRO_9", "new_matchtable.csv"))
    # mt_paths.append(os.path.join(test_data_root, "CB017", "19011110803", "2", "new_matchtable.csv"))
    # for i, mt_path in enumerate(mt_paths):
    #     ext_data_fig5(mt_path, str(i))
    # ext_data_fig5(mt_path, "AV009")

    # dnn_auc, um_auc = auc_one_pair(mt, 1, 2)
    # print(dnn_auc, um_auc)

    dnn_slope, dnn_intercept, um_slope, um_intercept = auc_over_days(mt_path, vis=True, within50=True)

    # Get out the y = ax + b parameters for each (mouse, probe, loc)
    # dnn_a, dnn_b, um_a, um_b = all_mice_auc_over_days(test_data_root)
    # print(dnn_a, dnn_b, um_a, um_b)
