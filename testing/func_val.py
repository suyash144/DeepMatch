import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import mat73
from tqdm import tqdm

def rank_list(numbers):
    # Create a sorted version of the list
    sorted_numbers = sorted(set(numbers))
    
    # Create a dictionary to map each number to its sorted index (rank)
    ranks = {number: index for index, number in enumerate(sorted_numbers)}
    
    # Replace each number in the original list with its rank
    return [ranks[number] for number in numbers]

def create_sim_mat(df, col):
    sessions = df["RecSes1"].iloc[1] == df["RecSes2"].iloc[1]
    if sessions:
        l1 = l2 = int(np.sqrt(len(df)))
    else:
        l1 = len(df["ID1"].unique())
        l2 = len(df["ID2"].unique())
        assert l1 * l2 == len(df)
    mat = np.empty((l1, l2))
    df["ID1"] = rank_list(df["ID1"])
    df["ID2"] = rank_list(df["ID2"])
    for n in range(len(df)):
        neuron = df.iloc[n, :]
        s = neuron[col]
        mat[int(neuron["ID1"]), int(neuron["ID2"])] = s
    return mat

def create_concat_mat(df11, df12, df21, df22, col):
    s11 = create_sim_mat(df11, col)
    s12 = create_sim_mat(df12, col)
    s21 = create_sim_mat(df21, col)
    s22 = create_sim_mat(df22, col)
        
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
    y = pl[2,:,:]                                      # shape [Nclus x 2]
    y = np.array(np.mean(y, axis=1))
    x = pl[1,:,:]                                      # shape [Nclus x 2]
    x = np.array(np.round(np.mean(x, axis=1), decimals=-2))

    if len(np.unique(x))!=4:
        print(f"CAUTION: Found {len(np.unique(x))} clusters of x values rather than 4.")

    uid = um["UniqueIDConversion"]["OriginalClusID"]
    urs = um["UniqueIDConversion"]["recsesAll"]
    goodids = um["UniqueIDConversion"]["GoodID"]
    uid, urs = uid[goodids==1], urs[goodids==1]
    # pl = np.repeat(pl, len(pl))
    depth_dict = {"RecSes": urs, "ID": uid, "x": x, "depth": y}
    depth_df = pd.DataFrame(depth_dict).sort_values(by = ["RecSes", "ID"])
    return depth_df

def compare_two_recordings(path_to_csv:str, rec1:int, rec2:int, sort_method = "id", depth_df = None):
    """
    path_to_csv: path to matchtable csv
    rec1: integer corresponding to the RecSes1 that you want to select
    rec2: integer corresponding to the RecSes2 that you want to select
    sort_method: how you want the results to be sorted (depth or id)
    depths: only required if sort_method="depth". can read depths directly from matlab using the
    read_depths function.
    """
    # Pick out the relevant columns and ensure they are sorted
    df = pd.read_csv(path_to_csv)
    df = df.loc[:, ["RecSes1", "RecSes2", "ID1", "ID2", "DNNSim", "MatchProb"]]
    if sort_method == "depth":
        # Add depths to the dataframe.
        # for idx, row in tqdm(df.iterrows()):
            # i = depths_dict["ID"]==row["ID1"]
            # j = depths_dict["RecSes"]==row["RecSes1"]
            # depth = depths_dict["depth"][i&j]
            # if len(depth) != 1:
            #     print("Unable to order by depth as depth_dict does not uniquely identify neurons")
            #     print("Switching to sorting by ID")
            #     sort_method = "id"
            #     break
            # row["depth"] = depth[0]
        df11 = df.loc[(df["RecSes1"] == rec1) & (df["RecSes2"] == rec1), :].sort_values(by=["ID2", "ID1"])
        df12 = df.loc[(df["RecSes1"] == rec1) & (df["RecSes2"] == rec2), :].sort_values(by=["ID2", "ID1"])
        df21 = df.loc[(df["RecSes1"] == rec2) & (df["RecSes2"] == rec1), :].sort_values(by=["ID2", "ID1"])
        df22 = df.loc[(df["RecSes1"] == rec2) & (df["RecSes2"] == rec2), :].sort_values(by=["ID2", "ID1"])
        
        for d in [df11, df12, df21, df22]:
            depth_d = depth_df[depth_df["RecSes"]==d["RecSes1"].iloc[1]]          # subset of depth_df corresponding to d
            d.insert(len(d.columns), "x", np.resize(depth_d["x"], len(d)))
            d.insert(len(d.columns), "y", np.resize(depth_d["depth"], len(d)))
            d.sort_values(by = ["x", "y"], inplace=True)
    
    
    else:
        df11 = df.loc[(df["RecSes1"] == rec1) & (df["RecSes2"] == rec1), :]
        df12 = df.loc[(df["RecSes1"] == rec1) & (df["RecSes2"] == rec2), :]
        df21 = df.loc[(df["RecSes1"] == rec2) & (df["RecSes2"] == rec1), :]
        df22 = df.loc[(df["RecSes1"] == rec2) & (df["RecSes2"] == rec2), :]
    
    sim_matrix = create_concat_mat(df11, df12, df21, df22, "DNNSim")
    fig, (ax1, ax2) = plt.subplots(ncols = 2)
    ax1.matshow(sim_matrix)
    ax1.set_title("DNN Similarity matrix")

    um_output = create_concat_mat(df11, df12, df21, df22, "MatchProb")
    ax2.matshow(um_output)
    ax2.set_title("UnitMatch match probabilities")

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
path_to_csv = r"C:\Users\suyas\R_DATA_UnitMatch\AL036\19011116882\3\new_matchtable.csv"
# df = pd.read_csv(r"C:\Users\suyas\R_DATA_UnitMatch\AV008\Probe0\IMRO_7\new_matchtable.csv")

proj_loc = read_depths("AL036", "19011116882", "3")

compare_two_recordings(path_to_csv, 19, 20, "depth", proj_loc)