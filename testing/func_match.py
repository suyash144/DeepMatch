import numpy as np
import pandas as pd
import os, sys
sys.path.insert(0, os.getcwd())
from testing.isi_corr import *
from testing.dnn_dist import *
from testing.hungarian import hungarian_matches
from testing.similarity_matrices import read_depths
from matplotlib_venn import venn3
from tqdm import tqdm
from scipy.stats import sem


def test_metric(mt:pd.DataFrame, metric, vis:bool=False, rank:bool=False):
    mt = mt[mt[metric].notnull()]
    within = mt.loc[mt["RecSes1"]==mt["RecSes2"]]

    same_neuron = within.loc[within["ID1"]==within["ID2"]]
    diff_neuron = within.loc[within["ID1"]!=within["ID2"]]

    same = np.histogram(same_neuron[metric], bins=50, density=True)[0]
    diff = np.histogram(diff_neuron[metric], bins=50, density=True)[0]
    chi_squared_dist = np.sum((same - diff)**2) / np.sum(same + diff)
    if vis:
        plt.hist(same_neuron[metric], bins=50, alpha=0.4, density=True, label="Same neuron")
        plt.hist(diff_neuron[metric], bins=50, alpha=0.4, density=True, label="Diff. neurons")
        plt.xlabel(metric)
        plt.legend()
        plt.show()
    mt["uid1"] = mt["RecSes1"]*1e6 + mt["ID1"]
    mt["uid2"] = mt["RecSes2"]*1e6 + mt["ID2"]
    uids = mt["uid1"].unique().astype(int)
    ranks = []
    for id in uids:
        try:
            diag_idx = mt.loc[(mt["uid1"]==id) & (mt["uid2"]==id)].index.item()
        except:
            continue
        ranked = mt.loc[mt["uid1"]==id, metric].rank(ascending=False).astype(int)
        ranks.append(ranked[diag_idx])
    plt.hist(ranks, bins = np.arange(1, 20, 0.5))
    plt.xlabel(f"Rank of diagonal value according to {metric}")
    plt.show()
    return chi_squared_dist, np.mean(ranks)

def func_matches(matchtable:pd.DataFrame, metric:str, thresh=None):

    mt = matchtable.loc[matchtable["RecSes1"]!=matchtable["RecSes2"]]
    if thresh:
        within = mt.loc[mt["RecSes1"]==mt["RecSes2"]]
        mt = remove_split_units(within, mt, thresh, "DNNSim")
    mt["uid"] = mt["RecSes1"]*1e6 + mt["ID1"]
    unique_ids = mt["uid"].unique()
    func_matches_indices = []
    if metric=="ISICorr" or metric=="refPopCorr" or metric=="newISI":
        # apply Fisher z-transformation to correlation values
        mt[metric] = np.arctanh(mt[metric])
    for id in unique_ids:
        df = mt.loc[mt["uid"]==id,[metric]].nlargest(2, metric)
        if df.iloc[0].item() - df.iloc[1].item() > 0.2:
            # only accept a functional match if there is some distance (0.2)
            # between it and the next best correlation value in z-space.
            func_matches_indices.append(df.index[0])
    func_matches = directional_filter(mt.loc[func_matches_indices])
    func_matches = func_matches.loc[func_matches["RecSes1"]<func_matches["RecSes2"]]
    return func_matches.index

def get_matches(mt:pd.DataFrame, dnn_metric:str="DNNSim", um_metric:str="MatchProb", dist_thresh=None, mt_path=None):
    
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

    # Apply thresholds to generate matches for DNN and UnitMatch respectively
    dnn_matches = across.loc[mt[dnn_metric]>=thresh, ["ISICorr", "RecSes1", "RecSes2", "ID1", "ID2", dnn_metric]]
    um_matches = across.loc[mt[um_metric]>=thresh_um, ["ISICorr", "RecSes1", "RecSes2", "ID1", "ID2", um_metric]]
    # Only allow a match if it is above threshold when comparing in both directions
    dnn_matches = directional_filter(dnn_matches)
    um_matches = directional_filter(um_matches)
    # Remove split units from each set of matches
    # dnn_matches = remove_split_units(within, dnn_matches, thresh, "DNNSim")
    # um_matches = remove_split_units(within, um_matches, thresh, "DNNSim")
    if len(dnn_matches)!=0:
        # Do spatial filtering in DNN
        dnn_matches = spatial_filter(mt_path, dnn_matches, dist_thresh, plot_drift=False)
    # Resolve conflict matches by only keeping the match with highest DNNSim
    dnn_matches, dnn_conflicts = remove_conflicts(dnn_matches, dnn_metric)
    um_matches, um_conflicts = remove_conflicts(um_matches, um_metric)

    dnn_matches = dnn_matches.loc[dnn_matches["RecSes1"]<dnn_matches["RecSes2"]]
    um_matches = um_matches.loc[um_matches["RecSes1"]<um_matches["RecSes2"]]

    return dnn_matches.index.to_list(), um_matches.index.to_list(), thresh

def save_diagrams(mouse:str, probe:str, loc:str, venn:bool, bar:bool, save:bool):

    test_data_root = os.path.join(os.path.dirname(os.getcwd()), "ALL_DATA")
    mt_path = os.path.join(test_data_root, mouse, probe, loc, "wentao_model.csv")
    mt = pd.read_csv(mt_path)
    sessions = mt["RecSes1"].unique()
    depths = read_depths("AL036", "19011116882", "3")
    venn_dir = r"C:\Users\suyas\results_figs\venn_diagrams"
    dnn_rec, um_rec, dnn_n, um_n, dnn_prec, um_prec = [], [], [], [], [], []
    for r1 in tqdm(sessions):
        for r2 in tqdm(sessions):
            # if r1!=20 or r2 !=21:            # to compare same sessions as Wentao
            #     continue
            if r1 >= r2 or abs(r2-r1)>1:
                continue
            df = mt.loc[(mt["RecSes1"].isin([r1,r2])) & (mt["RecSes2"].isin([r1,r2])),:]
            dnn, um, thresh = get_matches(df, mt_path=mt_path, dist_thresh=20)
            func = func_matches(df, "refPopCorr")
            hung = hungarian_matches(df, r1, r2, depths, mt_path, thresh)
            func, dnn, um = set(func), set(hung), set(um)
            if venn:
                venn3([func, dnn, um], ('Functional', 'DNN', 'UnitMatch'))
                if save:
                    filename = "_".join((mouse, loc, str(r1), str(r2))) + '.png'
                    savepath = os.path.join(venn_dir, mouse+"unidirectional", filename)
                    plt.savefig(savepath, bbox_inches='tight')
                    plt.clf()
                else:
                    plt.show()
            if bar:
                if len(func)==0:
                    continue
                fd = len(dnn.intersection(func))
                fu = len(um.intersection(func))
                dnn_rec.append(fd / len(func))
                um_rec.append(fu / len(func))
                dnn_n.append(len(dnn))
                um_n.append(len(um))
                dnn_prec.append(fd / len(dnn))
                um_prec.append(fu / len(um))
    if bar:
        labels = ["DNN", "UnitMatch"]
        recs = [np.mean(dnn_rec), np.mean(um_rec), sem(dnn_rec), sem(um_rec)]
        numbers = [np.mean(dnn_n), np.mean(um_n), sem(dnn_n), sem(um_n)]
        precs = [np.mean(dnn_prec), np.mean(um_prec), sem(dnn_prec), sem(um_prec)]
        plt.subplot(1, 3, 1)
        plt.bar(labels, recs[:2], yerr=recs[2:], capsize=10)
        for i in range(len(dnn_rec)):
            plt.scatter([0, 1], [dnn_rec[i], um_rec[i]], alpha=0.7, c="r")
            plt.plot([0, 1], [dnn_rec[i], um_rec[i]], "r", alpha=0.7)
        plt.ylabel("Recall (percentage of functional matches found)")
        plt.subplot(1, 3, 2)
        plt.bar(labels, numbers[:2], yerr=numbers[2:], capsize=10)
        for i in range(len(dnn_n)):
            plt.scatter([0, 1], [dnn_n[i], um_n[i]], alpha=0.7, c="r")
            plt.plot([0, 1], [dnn_n[i], um_n[i]], "r", alpha=0.7)
        plt.ylabel("Number of matches found")
        plt.subplot(1, 3, 3)
        plt.bar(labels, precs[:2], yerr=precs[2:], capsize=10)
        for i in range(len(dnn_prec)):
            plt.scatter([0, 1], [dnn_prec[i], um_prec[i]], alpha=0.7, c="r")
            plt.plot([0, 1], [dnn_prec[i], um_prec[i]], "r", alpha=0.7)
        plt.ylabel("Precision (percentage of matches found that are functional)")
        plt.tight_layout()
        if save:
            savepath_bar = os.path.join(venn_dir, mouse+"unidirectional", "barcharts.png")
            plt.savefig(savepath_bar, bbox_inches='tight')
            plt.clf()
        else:
            plt.show()


if __name__=="__main__":
    save_diagrams("AL036", "19011116882", "3", venn=True, bar=True, save=True)
    # test_data_root = os.path.join(os.path.dirname(os.getcwd()), "ALL_DATA")
    # mt_path = os.path.join(test_data_root, "AL036", "19011116882", "3", "new_matchtable.csv")
    # mt = pd.read_csv(mt_path)
    # print(test_metric(mt, "DNNSim", rank=True))