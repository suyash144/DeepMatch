import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
sys.path.insert(0, os.getcwd())
from track_analysis.dnn import hungarian_matching_with_threshold
from testing.similarity_matrices import compare_two_recordings, read_depths
from testing.isi_corr import spatial_filter, remove_split_units, directional_filter


def hungarian_matches(df, r1, r2, depths, mt_path, thresh):
    sim_matrix, indices = compare_two_recordings(df, r1, r2, "depth", depths)
    separator = int(np.sqrt(len(df.loc[(df["RecSes1"]==r1) & (df["RecSes2"]==r1)])))
    matches = hungarian_matching_with_threshold(sim_matrix, thresh, separator)
    hung_idx = [indices[i,j] for i, j in zip(matches[:,0], matches[:,1])]
    hung_matches = df.loc[hung_idx]
    # within = df.loc[df["RecSes1"]==df["RecSes2"]]
    # hung_matches = remove_split_units(within, hung_matches, thresh, "DNNSim")
    hung_matches = directional_filter(hung_matches)
    hung_matches = hung_matches.loc[hung_matches["RecSes1"]<hung_matches["RecSes2"]]
    if len(hung_matches) > 0:
        hung = spatial_filter(mt_path, hung_matches, 20, True, False).index.to_list()
    else:
        return []
    return hung

if __name__=="__main__":
    test_data_root = os.path.join(os.path.dirname(os.getcwd()), "ALL_DATA")
    mt_path = os.path.join(test_data_root, "AL036", "19011116882", "3", "wentao_model.csv")
    depths = read_depths("AL036", "19011116882", "3")
    mt = pd.read_csv(mt_path)
    sim_matrix, idx = compare_two_recordings(mt, 1, 2, "depth", depths)
    matches = hungarian_matching_with_threshold(sim_matrix)         # try manually setting sim_thresh
    res = np.zeros(sim_matrix.shape)
    fig, (ax1, ax2) = plt.subplots(ncols = 2)
    for m in matches:
        res[m[0], m[1]] = 1
    ax1.matshow(res)
    ax1.set_title("Assignments made by Hungarian algorithm")
    ax2.matshow(sim_matrix)
    ax2.set_title("Similarity matrix")
    plt.show()

