import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
sys.path.insert(0, os.getcwd())
from track_analysis.dnn import hungarian_matching_with_threshold
from testing.func_match import get_matches, func_matches
from testing.similarity_matrices import compare_two_recordings, read_depths

def hungarian_matches(mouse, probe, loc):
    test_data_root = os.path.join(os.path.dirname(os.getcwd()), "ALL_DATA")
    mt_path = os.path.join(test_data_root, mouse, probe, loc, "wentao_model.csv")
    depths = read_depths(mouse, probe, loc)
    mt = pd.read_csv(mt_path)
    sessions = mt["RecSes1"].unique()
    for r1 in tqdm(sessions):
        for r2 in tqdm(sessions):
            if r1 >= r2 or abs(r2-r1)>1:
                continue
            df = mt.loc[(mt["RecSes1"].isin([r1, r2])) & (mt["RecSes2"].isin([r1,r2]))]
            sim_matrix, indices = compare_two_recordings(df, r1, r2, "depth", depths)
            # Do spatial steps here
            matches = hungarian_matching_with_threshold(sim_matrix)
            hung = [indices[i,j] for i, j in zip(matches[:,0], matches[:,1])]
            func = func_matches(df, r1, r2, "refPopCorr")
            dnn, um = get_matches(df, r1, r2, mt_path=mt_path, dist_thresh=20)
            raise

if __name__=="__main__":
    # test_data_root = os.path.join(os.path.dirname(os.getcwd()), "ALL_DATA")
    # mt_path = os.path.join(test_data_root, "AL036", "19011116882", "3", "wentao_model.csv")
    # depths = read_depths("AL036", "19011116882", "3")
    # mt = pd.read_csv(mt_path)
    # sim_matrix = compare_two_recordings(mt, 1, 2, "depth", depths)
    # matches = hungarian_matching_with_threshold(sim_matrix)         # try manually setting sim_thresh
    # res = np.zeros(sim_matrix.shape)
    # fig, (ax1, ax2) = plt.subplots(ncols = 2)
    # for m in matches:
    #     res[m[0], m[1]] = 1
    # ax1.matshow(res)
    # ax1.set_title("Assignments made by Hungarian algorithm")
    # ax2.matshow(sim_matrix)
    # ax2.set_title("Similarity matrix")
    # plt.show()


    hungarian_matches("AL036", "19011116882", "3")
