import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.getcwd())
from track_analysis.dnn import hungarian_matching_with_threshold
from testing.func_match import get_matches
from testing.similarity_matrices import compare_two_recordings, read_depths



if __name__=="__main__":
    test_data_root = os.path.join(os.path.dirname(os.getcwd()), "ALL_DATA")
    mt_path = os.path.join(test_data_root, "AL036", "19011116882", "3", "wentao_model.csv")
    depths = read_depths("AL036", "19011116882", "3")
    sim_matrix = compare_two_recordings(mt_path, 1, 2, "depth", depths)
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
