import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.neighbors import KernelDensity


def get_threshold(mt:pd.DataFrame, metric:str="DNNSim", vis:bool=True):
    mt = mt.loc[(mt["RecSes1"]==mt["RecSes2"]), :]              # Only use within day rows to compute threshold

    # On-diagonal means same neuron. Off-diagonal means different neurons.
    on_diag = mt.loc[(mt["ID1"]==mt["ID2"]), [metric]]
    off_diag = mt.loc[(mt["ID1"]!=mt["ID2"]), [metric]]

    # sanity check that the categories are being loaded correctly
    assert sum(on_diag.index.isin(off_diag.index)) == 0

    # Kernel density estimation (distributions are more useful than histograms)
    kde_on = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(on_diag)
    kde_off = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(off_diag)
    x = np.linspace(min(off_diag[metric]), max(on_diag[metric]), 1000).reshape(-1, 1)
    y_on = np.exp(kde_on.score_samples(x))
    y_off = np.exp(kde_off.score_samples(x))

    # Find the threshold where the distributions intersect
    thresh=np.argwhere(np.diff(np.sign(y_off - y_on)))
    if len(thresh) > 1:
        vis = True
        print(thresh)
        thresh = thresh[-1]
    print(f"Threshold: {metric} = {x[thresh].item()}")

    if vis:
        # visualise the results
        plt.hist(on_diag[metric], bins = 500, alpha = 0.5, density=True, label="On diagonal")
        plt.hist(off_diag[metric], bins = 500, alpha = 0.5, density=True, label="Off diagonal")
        plt.plot(x, y_on, label="On diagonal")
        plt.plot(x, y_off, label="Off diagonal")
        plt.axvline(x = x[thresh].item())
        
        plt.grid()
        plt.legend()
        plt.xlabel(metric)
        plt.title(f"Normalised histograms for on- and off-diagonal {metric} values in the same recording")
        plt.show()
    return x[thresh].item()

def plot_hist(mt_path:str, thresh=None):
    mt = pd.read_csv(mt_path)
    mt = mt.loc[(mt["RecSes1"]!=mt["RecSes2"]), :]              # Select the rows we want to use

    # On-diagonal means same neuron. Off-diagonal means different neurons.
    # on_diag = mt.loc[(mt["ID1"]==mt["ID2"]), ["DNNSim"]]
    # off_diag = mt.loc[(mt["ID1"]!=mt["ID2"]), ["DNNSim"]]

    # sanity check that the categories are being loaded correctly
    # assert sum(on_diag.index.isin(off_diag.index)) == 0

    # visualise the results
    # plt.hist(on_diag["DNNSim"], bins = 500, alpha = 0.5, density=True, label="On diagonal")
    # plt.hist(off_diag["DNNSim"], bins = 500, alpha = 0.5, density=True, label="Off diagonal")
    plt.hist(mt["DNNSim"], bins = 500, alpha = 0.5, density=True)
    if thresh:
        # if a threshold is provided then plot it as a vertical line.
        plt.axvline(x=thresh)

    plt.grid()
    # plt.legend()
    plt.xlabel("DNNSim")
    # plt.title("Normalised histograms for on- and off-diagonal DNNSim values")
    plt.show()

if __name__ == "__main__":
    test_data_root = r"C:\Users\suyas\R_DATA_UnitMatch"

    # mt_path = os.path.join(test_data_root, "AL031", "19011116684", "1", "new_matchtable.csv")
    mt_path = os.path.join(test_data_root, "AL036", "19011116882", "3", "new_matchtable.csv")       # 2497 neurons
    mt = pd.read_csv(mt_path)
    thresh = get_threshold(mt, vis=False)
    plot_hist(mt_path, thresh)