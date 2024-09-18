import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_sim_mat(df):
    l = int(np.sqrt(len(df)))
    mat = np.empty((l, l))
    for n in range(len(df)):
        neuron = df.iloc[n, :]
        s = neuron["DNNSim"]
        mat[n//l, n % l] = s
    return mat

df = pd.read_csv(r"C:\Users\suyas\R_DATA_UnitMatch\AL031\19011116684\1\new_matchtable.csv")

# To compare two recordings on adjacent days
rows = ((df["RecSes1"] == 3) | (df["RecSes1"] == 4)) & ((df["RecSes2"] == 3) | (df["RecSes2"] == 4))

tby3 = df.loc[(df["RecSes1"] == 3) & (df["RecSes2"] == 3), ["ID1", "ID2", "DNNSim"]].sort_values(by=['ID1', 'ID2'])
tby4 = df.loc[(df["RecSes1"] == 3) & (df["RecSes2"] == 4), ["ID1", "ID2", "DNNSim"]].sort_values(by=['ID1', 'ID2'])
fby3 = df.loc[(df["RecSes1"] == 4) & (df["RecSes2"] == 3), ["ID1", "ID2", "DNNSim"]].sort_values(by=['ID1', 'ID2'])
fby4 = df.loc[(df["RecSes1"] == 4) & (df["RecSes2"] == 4), ["ID1", "ID2", "DNNSim"]].sort_values(by=['ID1', 'ID2'])

df = df.loc[rows, :]

# sims = df["DNNSim"]
# probs = df["DNNProb"]

# # match_fraction = 

# print(sims.quantile(0.99))

# plt.hist(sims, bins = 500)
# # plt.hist(probs, bins = 500)
# # plt.show()
# print(max(probs))

sim_matrix = create_sim_mat(tby3)


plt.matshow(sim_matrix)
plt.show()