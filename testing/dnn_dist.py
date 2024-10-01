import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.neighbors import KernelDensity


def dnn_sim_dist(mt_path):
    mt = pd.read_csv(mt_path)
    on_diag = mt.loc[(mt["RecSes1"]==mt["RecSes2"]) & (mt["ID1"]==mt["ID2"]), ["DNNSim"]]
    off_diag = mt.loc[((mt["RecSes1"]==mt["RecSes2"]) & (mt["ID1"]==mt["ID2"]))==False, ["DNNSim"]]
    off_diag = off_diag.sample(len(on_diag))

    # sanity check that the categories are being loaded correctly
    assert sum(on_diag.index.isin(off_diag.index)) == 0

    weights_on = np.ones_like(on_diag) / len(on_diag)
    weights_off = np.ones_like(off_diag) / len(off_diag)
    plt.hist(on_diag["DNNSim"], bins = 500, alpha = 0.5, density=True, label="On diagonal")
    plt.hist(off_diag["DNNSim"], bins = 500, alpha = 0.5, density=True, label="Off diagonal")
    kde_on = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(on_diag)
    kde_off = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(off_diag)
    x = np.linspace(min(off_diag["DNNSim"]), max(on_diag["DNNSim"]), 1000).reshape(-1, 1)
    plt.plot(x, np.exp(kde_on.score_samples(x)))
    plt.plot(x, np.exp(kde_off.score_samples(x)))

    plt.legend()
    plt.xlabel("DNNSim")
    plt.title("Normalised histograms for on- and off-diagonal DNNSim values")
    plt.show()


test_data_root = r"C:\Users\suyas\R_DATA_UnitMatch"

# mt_path = os.path.join(test_data_root, "AL031", "19011116684", "1", "new_matchtable.csv")
mt_path = os.path.join(test_data_root, "AL036", "19011116882", "3", "new_matchtable.csv")       # 2497 neurons

dnn_sim_dist(mt_path)