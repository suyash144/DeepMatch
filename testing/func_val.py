import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

df = pd.read_csv(r"C:\Users\suyas\R_DATA_UnitMatch\AL031\19011116684\1\new_matchtable.csv")

# To compare two recordings on adjacent days
rows = ((df["RecSes1"] == 3) | (df["RecSes1"] == 4)) & ((df["RecSes2"] == 3) | (df["RecSes2"] == 4))

df = df.loc[rows, :]

df11 = df.loc[(df["RecSes1"] == 3) & (df["RecSes2"] == 3), ["RecSes1", "RecSes2", "ID1", "ID2", "DNNSim"]].sort_values(by=['ID1', 'ID2'])
df12 = df.loc[(df["RecSes1"] == 3) & (df["RecSes2"] == 4), ["RecSes1", "RecSes2", "ID1", "ID2", "DNNSim"]].sort_values(by=['ID1', 'ID2'])
df21 = df.loc[(df["RecSes1"] == 4) & (df["RecSes2"] == 3), ["RecSes1", "RecSes2", "ID1", "ID2", "DNNSim"]].sort_values(by=['ID1', 'ID2'])
df22 = df.loc[(df["RecSes1"] == 4) & (df["RecSes2"] == 4), ["RecSes1", "RecSes2", "ID1", "ID2", "DNNSim"]].sort_values(by=['ID1', 'ID2'])

# sims = df["DNNSim"]
# probs = df["DNNProb"]

# # match_fraction = 

# print(sims.quantile(0.99))

# plt.hist(sims, bins = 500)
# # plt.hist(probs, bins = 500)
# # plt.show()
# print(max(probs))

sim_matrix = create_concat_mat(df11, df12, df21, df22)


plt.matshow(sim_matrix)
plt.show()