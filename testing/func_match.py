import numpy as np
import pandas as pd
import os


def func_matches(mt:pd.DataFrame, metric:str):
    mt["uid"] = mt["RecSes1"]*1e6 + mt["ID1"]
    unique_ids = mt["uid"].unique()
    func_matches_indices = []
    for id in unique_ids:
        df = mt.loc[mt["uid"]==id,[metric]].nlargest(2, metric)
        if df.iloc[0].item() - df.iloc[1].item() > 0.05:
            func_matches_indices.append(df.index[0])
    return func_matches_indices


mouse = "AL036"
probe = "19011116882"
loc = "3"

test_data_root = os.path.join(os.path.dirname(os.getcwd()), "ALL_DATA")
server_root = r"\\znas\Lab\Share\UNITMATCHTABLES_ENNY_CELIAN_JULIE\FullAnimal_KSChanMap"
mt_path = os.path.join(test_data_root, mouse, probe, loc, "new_matchtable.csv")
mt = pd.read_csv(mt_path)
idx = func_matches(mt, "ISICorr")

print(len(idx))
print(len(mt))
print(len(idx) / len(mt))