import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import mat73
from scipy.stats import rankdata

def create_sim_mat(df, col):
    sessions = df["RecSes1"].iloc[1] == df["RecSes2"].iloc[1]
    if sessions:
        l1 = l2 = int(np.sqrt(len(df)))
    else:
        l1 = len(df["ID1"].unique())
        l2 = len(df["ID2"].unique())
        assert l1 * l2 == len(df)
    if col=="index":
        vals = np.array(df.index)
    else:
        vals = np.array(df[col])
    vals = vals.reshape((l1, l2))
    return vals

def create_concat_mat(df11, df12, df21, df22, col, sort_method="id", rec1=None, rec2=None, depths=None):
    s11 = create_sim_mat(df11, col)
    s12 = create_sim_mat(df12, col)
    s21 = create_sim_mat(df21, col)
    s22 = create_sim_mat(df22, col)

    if sort_method=="depth":
        s11 = reorder_by_depth(s11, depths, rec1, rec1)
        s12 = reorder_by_depth(s12, depths, rec1, rec2)
        s21 = reorder_by_depth(s21, depths, rec2, rec1)
        s22 = reorder_by_depth(s22, depths, rec2, rec2)
        
    top_row = np.concatenate((s11, s12), axis = 1)
    bottom_row =  np.concatenate((s21, s22), axis = 1)

    sim_matrix = np.concatenate((top_row, bottom_row))
    return sim_matrix

def read_depths(mouse, probe, loc):
    base = r"\\znas\Lab\Share\UNITMATCHTABLES_ENNY_CELIAN_JULIE\FullAnimal_KSChanMap"
    # Find Unitmatch.mat for each recording
    um_path = os.path.join(base, mouse, probe, loc, "UnitMatch", "UnitMatch.mat")
    um = mat73.loadmat(um_path, verbose=False)
    pl = um["WaveformInfo"]["ProjectedLocation"]        # shape [3 x Nclus x 2]
    x = pl[1,:,:]                                      # shape [Nclus x 2]
    y = pl[2,:,:]
    x = np.array(np.round(np.mean(x, axis=1), decimals=-2))
    y = np.array(np.mean(y, axis=1))
    if len(np.unique(x))!=4:
        print(f"CAUTION: Found {len(np.unique(x))} clusters of x values rather than 4.")
    uid = um["UniqueIDConversion"]["OriginalClusID"]
    urs = um["UniqueIDConversion"]["recsesAll"]
    goodids = um["UniqueIDConversion"]["GoodID"]
    uid, urs = uid[goodids==1], urs[goodids==1]
    depth = x * 1e6 + y
    depth_dict = {"RecSes": urs, "ID": uid, "IDrank": '', "depth": depth}
    depth_df = pd.DataFrame(depth_dict)
    return depth_df

def compare_two_recordings(df:pd.DataFrame, rec1:int, rec2:int, sort_method = "id", depths = None, vis=False):
    """
    df: matchtable as a pandas DataFrame
    rec1: integer corresponding to the RecSes1 that you want to select
    rec2: integer corresponding to the RecSes2 that you want to select
    sort_method: how you want the results to be sorted (depth or id)
    depths: only required if sort_method="depth". can read depths directly from matlab using the
    read_depths function.
    """
    # Pick out the relevant columns and ensure they are sorted
    col = df.loc[:, "WavformSim":"LocTrajectorySim"]
    df["NoLocScore"] = col.mean(axis=1)
    df = df.loc[:, ["RecSes1", "RecSes2", "ID1", "ID2", "DNNSim", "MatchProb", "NoLocScore"]]
    df11 = df.loc[(df["RecSes1"] == rec1) & (df["RecSes2"] == rec1), :]
    df12 = df.loc[(df["RecSes1"] == rec1) & (df["RecSes2"] == rec2), :]
    df21 = df.loc[(df["RecSes1"] == rec2) & (df["RecSes2"] == rec1), :]
    df22 = df.loc[(df["RecSes1"] == rec2) & (df["RecSes2"] == rec2), :]
    
    if sort_method == "depth":
        sim_matrix = create_concat_mat(df11, df12, df21, df22, "DNNSim", "depth", rec1, rec2, depths)
        indices = create_concat_mat(df11, df12, df21, df22, "index", "depth", rec1, rec2, depths)
        um_output = create_concat_mat(df11, df12, df21, df22, "MatchProb", "depth", rec1, rec2, depths)
        um_score = create_concat_mat(df11, df12, df21, df22, "NoLocScore", "depth", rec1, rec2, depths)
    elif sort_method == "id":
        sim_matrix = create_concat_mat(df11, df12, df21, df22, "DNNSim", "id")
        um_output = create_concat_mat(df11, df12, df21, df22, "MatchProb")
        um_score = create_concat_mat(df11, df12, df21, df22, "NoLocScore")
    else:
        raise ValueError("""Please pick a sorting method from 'depth' or 'id'
                         Default is id as this requires no info about spatial positions of neurons.
                         Depth gives better results though.""")
    if vis:
        fig, (ax1, ax2, ax3) = plt.subplots(ncols = 3)
        ax1.matshow(sim_matrix)
        ax1.set_title("DNN Similarity matrix")

        ax2.matshow(um_output)
        ax2.set_title("UnitMatch match probabilities")

        ax3.matshow(um_score)
        ax3.set_title("UnitMatch score (no centroid)")

        plt.show()
    return sim_matrix, indices

def reorder_by_depth(matrix:np.ndarray, depths, recses1:int, recses2:int):
    """
    Matrix should compare just one recording session against another.
    """
    depths1 = depths.loc[depths["RecSes"]==recses1, :]
    depths2 = depths.loc[depths["RecSes"]==recses2, :]
    depths1.loc[:, "IDrank"] = depths1["ID"].rank()-1
    depths2.loc[:, "IDrank"] = depths2["ID"].rank()-1
    res = np.empty(matrix.shape)
    n1_depths = np.empty((matrix.shape[0],))
    n2_depths = np.empty((matrix.shape[1],))
    for i, j in np.ndindex(matrix.shape):
        depth_i = depths1.loc[depths1["IDrank"]==i, "depth"]
        depth_i = depth_i.tolist()
        if len(depth_i) != 1:
            print(len(depth_i))
            raise ValueError("Unable to uniquely identify the neuron to find its depth")
        n1_depths[i] = depth_i[0]

        depth_j = depths2.loc[depths2["IDrank"]==j, "depth"]
        depth_j = depth_j.tolist()
        if len(depth_j) != 1:
            print(len(depth_j))
            raise ValueError("Unable to uniquely identify the neuron to find its depth")
        n2_depths[j] = depth_j[0]
    n1_depths = (rankdata(n1_depths) - 1).astype(int)
    n2_depths = (rankdata(n2_depths) - 1).astype(int)
    for idxi, idxj in np.ndindex(matrix.shape):
        new_i = n1_depths[idxi]
        new_j = n2_depths[idxj]
        res[new_i, new_j] = matrix[idxi, idxj]
    
    return res

        
if __name__=="__main__":
    # sims = df["DNNSim"]
    # probs = df["DNNProb"]

    # # match_fraction = 

    # print(sims.quantile(0.99))

    # plt.hist(sims, bins = 500)
    # # plt.hist(probs, bins = 500)
    # # plt.show()
    # print(max(probs))


    # df = pd.read_csv(r"C:\Users\suyas\R_DATA_UnitMatch\AL032\19011111882\2\new_matchtable.csv")
    path_to_csv = r"C:\Users\suyas\R_DATA_UnitMatch\AL036\19011116882\3\new_matchtable.csv"
    # df = pd.read_csv(r"C:\Users\suyas\R_DATA_UnitMatch\AV008\Probe0\IMRO_7\new_matchtable.csv")

    depths = read_depths("AL036", "19011116882", "3")

    sim_matrix,idx = compare_two_recordings(path_to_csv, 19, 20, "depth", depths)

    # df = pd.read_csv(path_to_csv)
    # df = df.loc[:, ["RecSes1", "RecSes2", "ID1", "ID2", "DNNSim", "MatchProb"]]
    # df11 = df.loc[(df["RecSes1"] == 19) & (df["RecSes2"] == 20), :]
    # mat = create_sim_mat(df11, "DNNSim")
    # t = reorder_by_depth(mat, proj_loc, 19, 20)
    # plt.matshow(t)
    # plt.show()