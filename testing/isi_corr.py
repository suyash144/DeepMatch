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
    plt.hist(sorted["ISICorr"], bins = 100)
    plt.show()


compare_isi(test_data_root, "AL031", "19011116684", "1")