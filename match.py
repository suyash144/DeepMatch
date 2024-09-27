import os
import numpy as np
import pandas as pd
import mat73
from utils.myutil import get_exp_id
from utils.read_pos import read_pos

def generate_matches(mt_path, sample_size, rec1, rec2):

    server_root = r"\\znas\Lab\Share\UNITMATCHTABLES_ENNY_CELIAN_JULIE\FullAnimal_KSChanMap"

    loc = os.path.basename(os.path.dirname(mt_path))
    probe = os.path.basename(os.path.dirname(os.path.dirname(mt_path)))
    mouse = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(mt_path))))
    um_path = os.path.join(server_root, mouse, probe, loc, "UnitMatch", "UnitMatch.mat")
    um = mat73.loadmat(um_path)

    path_list = um["UMparam"]["KSDir"]
    path_dict = {}
    for idx, path in enumerate(path_list):
        p = get_exp_id(path, mouse)
        path_dict[int(idx+1)] = p
    expid1 = path_dict[rec1]
    expid2 = path_dict[rec2]
    exp1 = os.path.join(os.path.dirname(mt_path), expid1, "processed_waveforms")
    exp2 = os.path.join(os.path.dirname(mt_path), expid2, "processed_waveforms")

    x1, y1 = read_pos(exp1)
    x2, y2 = read_pos(exp2)

    df = pd.read_csv(mt_path)
    df = df.loc[(df["RecSes1"]==rec1) & (df["RecSes2"]==rec2),["DNNProb", "DNNSim", "ID1", "ID2"]]
    putative_matches = df.sort_values(by = "DNNSim", ascending=False).head(sample_size)

    for row in putative_matches.iterrows():
        pass


mt_path = r"C:\Users\suyas\R_DATA_UnitMatch\AL036\19011116882\3\new_matchtable.csv"

generate_matches(mt_path, 10, 19, 20)