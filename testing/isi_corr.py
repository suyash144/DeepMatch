import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

test_data_root = r"C:\Users\suyas\R_DATA_UnitMatch"

def compare_isi(test_data_root, mouse, probe, loc):
    mt_path = os.path.join(test_data_root, mouse, probe, loc, "new_matchtable.csv")
    if not os.path.exists(mt_path):
        raise ValueError(f"Matchtable not found for (mouse, probe, loc: {mouse}, {probe}, {loc})")
    mt = pd.read_csv(mt_path)
    matches = np.sqrt(len(mt))
    if (matches).is_integer():
        matches = int(matches)
    else:
        raise ValueError("Length of matchtable is not (no. of neurons)^2")
    sorted = mt.sort_values(by = "DNNSim", ascending=False).head(matches)
    sorted = sorted.loc[:, ["DNNSim", "ISICorr"]]
    unsorted = mt.head(matches)
    plt.hist(sorted["ISICorr"], bins = 500, label="Matches (as per DNNSim)")
    plt.hist(unsorted["ISICorr"], bins = 500, label="Random selection")
    plt.legend()
    plt.show()


# compare_isi(test_data_root, "AL031", "19011116684", "1")
# compare_isi(test_data_root, "AL032", "19011111882", "2")
compare_isi(test_data_root, "AL036", "19011116882", "3")