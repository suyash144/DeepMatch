import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import mat73

def create_sim_mat(df):
    sessions = df["RecSes1"].iloc[1] == df["RecSes2"].iloc[1]
    if sessions:
        l1 = l2 = int(np.sqrt(len(df)))
    else:
        l1 = len(df["ID1"].unique())
        l2 = len(df["ID2"].unique())
        assert l1 * l2 == len(df)
    mat = np.empty((l1, l2))
    for n in range(len(df)):
        neuron = df.iloc[n, :]
        s = neuron["DNNSim"]
        mat[n//l2, n % l2] = s
    return mat

def create_concat_mat(df11, df12, df21, df22):
    s11 = create_sim_mat(df11)
    s12 = create_sim_mat(df12)
    s21 = create_sim_mat(df21)
    s22 = create_sim_mat(df22)
        
    top_row = np.concatenate((s11, s12), axis = 1)
    bottom_row =  np.concatenate((s21, s22), axis = 1)

    sim_matrix = np.concatenate((top_row, bottom_row))
    return sim_matrix

def read_depths(mouse, probe, loc):
    base = r"\\znas\Lab\Share\UNITMATCHTABLES_ENNY_CELIAN_JULIE\FullAnimal_KSChanMap"
    # Find Unitmatch.mat for each recording
    um_path = os.path.join(base, mouse, probe, loc, "UnitMatch", "UnitMatch.mat")
    um = mat73.loadmat(um_path)
    pl = um["WaveformInfo"]["ProjectedLocation"]        # shape [3 x Nclus x 2]
    pl = pl[1,:,:]                                      # shape [Nclus x 2]
    pl = np.mean(pl, axis=1)
    # pl has length N_clus and is ordered by clusID
    # Equivalently, ordered by RecSes1 -> ID1 -> RecSes2 -> ID2
    pl = np.array(pl)
    pl = np.repeat(pl, len(pl))
    return pl

def compare_two_recordings(df:pd.DataFrame, rec1:int, rec2:int, sort_method = "depth", depths = None):
    """
    df: dataframe object containing whole matchtable csv
    rec1: integer corresponding to the RecSes1 that you want to select
    rec2: integer corresponding to the RecSes2 that you want to select
    sort_method: how you want the results to be sorted (depth or id)
    depths: only required if sort_method="depth". can read depths directly from matlab using the
    read_depths function.
    """
    # Pick out the relevant columns and ensure they are sorted
    df = df.loc[:, ["RecSes1", "RecSes2", "ID1", "ID2", "DNNSim"]].sort_values(by=["RecSes1", 'RecSes2', 'ID1', 'ID2'])
    if sort_method == "depth":
        df.insert(len(df.columns), "depth", depths)
        df11 = df.loc[(df["RecSes1"] == rec1) & (df["RecSes2"] == rec1), :].sort_values(by=["depth", "RecSes1", 'RecSes2', 'ID1', 'ID2'])
        df12 = df.loc[(df["RecSes1"] == rec1) & (df["RecSes2"] == rec2), :].sort_values(by=["depth", "RecSes1", 'RecSes2', 'ID1', 'ID2'])
        df21 = df.loc[(df["RecSes1"] == rec2) & (df["RecSes2"] == rec1), :].sort_values(by=["depth", "RecSes1", 'RecSes2', 'ID1', 'ID2'])
        df22 = df.loc[(df["RecSes1"] == rec2) & (df["RecSes2"] == rec2), :].sort_values(by=["depth", "RecSes1", 'RecSes2', 'ID1', 'ID2'])
    elif sort_method == "id":
        df11 = df.loc[(df["RecSes1"] == rec1) & (df["RecSes2"] == rec1), :]
        df12 = df.loc[(df["RecSes1"] == rec1) & (df["RecSes2"] == rec2), :]
        df21 = df.loc[(df["RecSes1"] == rec2) & (df["RecSes2"] == rec1), :]
        df22 = df.loc[(df["RecSes1"] == rec2) & (df["RecSes2"] == rec2), :]
    else:
        raise ValueError("Please pick a sorting method from 'depth' or 'id'. Default is depth.")
    sim_matrix = create_concat_mat(df11, df12, df21, df22)
    plt.matshow(sim_matrix)
    plt.show()


# sims = df["DNNSim"]
# probs = df["DNNProb"]

# # match_fraction = 

# print(sims.quantile(0.99))

# plt.hist(sims, bins = 500)
# # plt.hist(probs, bins = 500)
# # plt.show()
# print(max(probs))


# df = pd.read_csv(r"C:\Users\suyas\R_DATA_UnitMatch\AL032\19011111882\2\new_matchtable.csv")
df = pd.read_csv(r"C:\Users\suyas\R_DATA_UnitMatch\AL036\19011116882\3\new_matchtable.csv")
# df = pd.read_csv(r"C:\Users\suyas\R_DATA_UnitMatch\AV008\Probe0\IMRO_7\new_matchtable.csv")

# proj_loc = read_depths("AL036", "19011116882", "3")

compare_two_recordings(df, 19, 20, "id")