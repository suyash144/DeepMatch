import numpy as np
import pandas as pd
import os, sys
sys.path.insert(0, os.getcwd())
from testing.isi_corr import *
from testing.dnn_dist import *
from matplotlib_venn import venn3
from tqdm import tqdm
from scipy.stats import sem


def func_matches(mt:pd.DataFrame, rec1:int, rec2:int, metric:str):

    mt = mt.loc[(mt["RecSes1"].isin([rec1,rec2])) & (mt["RecSes2"].isin([rec1,rec2])),:]
    mt = mt.loc[mt["RecSes1"]!=mt["RecSes2"]]
    mt["uid"] = mt["RecSes1"]*1e6 + mt["ID1"]
    unique_ids = mt["uid"].unique()
    func_matches_indices = []
    if metric=="ISICorr" or metric=="refPopCorr":
        # apply Fisher z-transformation to correlation values
        mt[metric] = np.arctanh(mt[metric])
    for id in unique_ids:
        df = mt.loc[mt["uid"]==id,[metric]].nlargest(2, metric)
        if df.iloc[0].item() - df.iloc[1].item() > 0.2:
            # only accept a functional match if there is some distance (0.1)
            # between it and the next best correlation value in z-space.
            func_matches_indices.append(df.index[0])
    return func_matches_indices

def get_matches(mt:pd.DataFrame, rec1:int, rec2:int, dnn_metric:str="DNNSim", 
                um_metric:str="MatchProb", dist_thresh=None, mt_path=None, within50=True):
    
    mt = mt.loc[(mt["RecSes1"].isin([rec1,rec2])) & (mt["RecSes2"].isin([rec1,rec2])),:]
    thresh = dnn_dist.get_threshold(mt, metric=dnn_metric, vis=False)
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
    dnn_matches = across.loc[mt[dnn_metric]>=thresh, ["ISICorr", "RecSes1", "RecSes2", "ID1", "ID2", dnn_metric]]
    um_matches = across.loc[mt[um_metric]>=thresh_um, ["ISICorr", "RecSes1", "RecSes2", "ID1", "ID2", um_metric]]
    # Only allow a match if it is above threshold when comparing in both directions
    dnn_matches = directional_filter(dnn_matches)
    um_matches = directional_filter(um_matches)
    # Remove split units from each set of matches
    dnn_matches = remove_split_units(within, dnn_matches, thresh, "DNNSim")
    um_matches = remove_split_units(within, um_matches, thresh_um, "MatchProb")
    if len(dnn_matches)!=0:
        # Do spatial filtering in DNN
        dnn_matches = spatial_filter(mt_path, dnn_matches, dist_thresh, plot_drift=False)
    # Resolve conflict matches by only keeping the match with highest DNNSim
    dnn_matches, dnn_conflicts = remove_conflicts(dnn_matches, dnn_metric)
    um_matches, um_conflicts = remove_conflicts(um_matches, um_metric)

    return dnn_matches.index.to_list(), um_matches.index.to_list()

def save_diagrams(mouse:str, probe:str, loc:str, venn:bool, bar:bool):

    test_data_root = os.path.join(os.path.dirname(os.getcwd()), "ALL_DATA")
    mt_path = os.path.join(test_data_root, mouse, probe, loc, "new_matchtable.csv")
    mt = pd.read_csv(mt_path)
    sessions = mt["RecSes1"].unique()
    venn_dir = r"C:\Users\suyas\results_figs\venn_diagrams"
    dnn_perc, um_perc, dnn_n, um_n = [], [], [], []    
    for r1 in tqdm(sessions):
        for r2 in tqdm(sessions):
            if r1 >= r2 or abs(r2-r1)>1:
                continue
            func = func_matches(mt, r1, r2, "refPopCorr")
            dnn, um = get_matches(mt, r1, r2, mt_path=mt_path, dist_thresh=20)
            func, dnn, um = set(func), set(dnn), set(um)
            if venn:
                venn3([func, dnn, um], ('Functional', 'DNN', 'UnitMatch'))
                filename = "_".join((mouse, loc, str(r1), str(r2))) + '.png'
                savepath = os.path.join(venn_dir, mouse+"_rpc", filename)
                plt.savefig(savepath, bbox_inches='tight')
                plt.clf()
            if bar:
                if len(func)==0:
                    continue
                fd = len(dnn.intersection(func))
                fu = len(um.intersection(func))
                dnn_perc.append(fd / len(func))
                um_perc.append(fu / len(func))
                dnn_n.append(len(dnn))
                um_n.append(len(um))
    if bar:
        labels = ["DNN", "UnitMatch"]
        percs = [np.mean(dnn_perc), np.mean(um_perc), sem(dnn_perc), sem(um_perc)]
        numbers = [np.mean(dnn_n), np.mean(um_n), sem(dnn_n), sem(um_n)]
        plt.subplot(1, 2, 1)
        plt.bar(labels, percs[:2], yerr=percs[2:], capsize=10)
        plt.ylabel("Percentage of functional matches found")
        plt.subplot(1, 2, 2)
        plt.bar(labels, numbers[:2], yerr=numbers[2:], capsize=10)
        plt.ylabel("Number of matches found")
        plt.tight_layout()
        savepath_bar = os.path.join(venn_dir, mouse+"_rpc", "barcharts.png")
        plt.savefig(savepath_bar, bbox_inches='tight')
        plt.clf()


if __name__=="__main__":
    save_diagrams("AL031", "19011116684", "1", venn=True, bar=True)