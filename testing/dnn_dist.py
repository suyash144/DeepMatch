import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


def dnn_sim_dist(mt_path):
    mt = pd.read_csv(mt_path)
    on_diag = mt.loc[(mt["RecSes1"]==mt["RecSes2"]) & (mt["ID1"]==mt["ID2"]), ["DNNSim"]]
    off_diag = mt.loc[((mt["RecSes1"]==mt["RecSes2"]) & (mt["ID1"]==mt["ID2"]))==False, ["DNNSim"]]
    assert sum(on_diag.index.isin(off_diag.index)) == 0
    assert len(off_diag)+ len(on_diag) == len(mt)
    plt.hist(on_diag["DNNSim"], bins = 500, alpha = 0.5)
    plt.hist(off_diag["DNNSim"], bins = 500, alpha = 0.5)
    plt.show()


test_data_root = r"C:\Users\suyas\R_DATA_UnitMatch"


mt_path = os.path.join(test_data_root, "AL036", "19011116882", "3", "new_matchtable.csv")       # 2497 neurons

dnn_sim_dist(mt_path)