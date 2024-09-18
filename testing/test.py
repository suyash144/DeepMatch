import os, sys

if __name__ == '__main__':
    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, os.path.join(os.getcwd(), os.pardir))

from utils.losses import *
from utils.npdataset import NeuropixelsDataset, ValidationExperimentBatchSampler
import numpy as np
from models.mymodel import *
from torch.utils.data import DataLoader
from torch import unsqueeze
import tqdm
import pandas as pd
import mat73
from utils.myutil import get_exp_id, get_unit_id


def inference(test_data_root:str, mouse:str, probe:str, loc:str, model_name:str, device = "cpu"):
    """
    Arguments:
    mouse: mouse name
    probe: probe
    loc: location on probe
    These 3 arguments together should also specify the filepath to the folder of experiments
    for this specific combination of (mouse, probe, loc)
    model_name: name of saved (trained) model you wish to load for testing (within ModelExp/experiments)
    device: optional, specify if e.g. using cuda
    """

    # Load the trained model
    model = SpatioTemporalCNN_V2(n_channel=30,n_time=60,n_output=256).to(device)
    model = model.double()
    ckpt_folder = os.path.join(os.getcwd(), "ModelExp", "experiments", model_name, "ckpt")
    ckpt_lst = os.listdir(ckpt_folder)
    ckpt_lst.sort(key=lambda x: int(x.split('_')[-1]))
    read_path = os.path.join(ckpt_folder, ckpt_lst[-1])
    print('Loading checkpoint from %s'%(read_path))
    checkpoint = torch.load(read_path)
    model.load_state_dict(checkpoint['model'])
    clip_loss = CustomClipLoss().to(device)
    clip_loss.load_state_dict(checkpoint['clip_loss'])
    model.eval()
    clip_loss.eval()

    # Load projector
    projector = Projector(input_dim=256, output_dim=128, hidden_dim=128, n_hidden_layers=1, dropout=0.1).to(device)
    projector = projector.double()

    # test_data_root = os.path.join(os.path.dirname(os.getcwd()), 'R_DATA_UNITMATCH')
    test_dataset = NeuropixelsDataset(root=test_data_root, batch_size=32, mode='val', m=mouse, p=probe, l=loc)
    test_sampler = ValidationExperimentBatchSampler(test_dataset, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler)

    print(f"Length of test dataset: {len(test_dataset)}")

    server_root = r"\\znas\Lab\Share\UNITMATCHTABLES_ENNY_CELIAN_JULIE\FullAnimal_KSChanMap"

    with torch.no_grad():
        progress_bar = tqdm.tqdm(total=len(test_loader))
        
        mt_path = os.path.join(test_data_root, mouse, probe, loc, "matchtable.csv")
        try:
            mt = pd.read_csv(mt_path)
            mt.insert(len(mt.columns), "DNNProb", '', allow_duplicates=False)
            mt.insert(len(mt.columns), "DNNSim", '', allow_duplicates=False)
        except:
            print("No matchtable found for this combination of (mouse, probe, location): ")
            print(f"Mouse: {mouse}, Probe: {probe}, Location: {loc}")
            raise ValueError()
        um_path = os.path.join(server_root, mouse, probe, loc, "UnitMatch", "UnitMatch.mat")
        um = mat73.loadmat(um_path)
        path_list = um["UMparam"]["KSDir"]
        path_dict = {}
        for idx, path in enumerate(path_list):
            p = get_exp_id(path, mouse)
            path_dict[p] = int(idx+1)

        for estimates_i, _,_, exp_ids_i, filepaths_i in test_loader:
            if torch.cuda.is_available():
                estimates_i = estimates_i.cuda()
            bsz_i = estimates_i.shape[0]
            exp_id_i = exp_ids_i[0]                     # this should be the same for all files in the batch
            rec_ses1 = path_dict[exp_id_i]

            # Forward pass
            enc_estimates_i = model(estimates_i)        # shape [bsz, 256]

            for _, candidates_j,_,exp_ids_j,filepaths_j in test_loader:
                if torch.cuda.is_available():
                    candidates_j = candidates_j.cuda()
                bsz_j = candidates_j.shape[0]
                enc_candidates_j = model(candidates_j)
                exp_id_j = exp_ids_j[0]
                recses2 = path_dict[exp_id_j]

                for unit_idx_i, est in enumerate(enc_estimates_i):
                    id1 = get_unit_id(filepaths_i[unit_idx_i])
                    est.unsqueeze_(0)
                    for unit_idx_j, cand in enumerate(enc_candidates_j):
                        id2 = get_unit_id(filepaths_j[unit_idx_j])
                        cand.unsqueeze_(0)
                        prob = clip_prob(est, cand).item()
                        sim = clip_sim(est, cand).item()
                        # Write to the matchtable now that we have ID1, ID2, RecSes1 and RecSes2
                        row = ((mt["ID1"]==id1) & (mt["ID2"]==id2) 
                                     & (mt["RecSes1"]==rec_ses1) & (mt["RecSes2"]==recses2))
                        mt.loc[row, "DNNProb"] = prob
                        mt.loc[row, "DNNSim"] = sim

            progress_bar.update(1)
        
        mt.to_csv(os.path.join(test_data_root, mouse, probe, loc, "new_matchtable.csv"))
        progress_bar.close()



if __name__ == '__main__':
    # example args to check inference function works
    base = r"C:\Users\suyas\R_DATA_UnitMatch"

    # to test for one specific set of recordings
    # inference(base, "AL031", "19011116684", "1", "test")

    # to test for all sets of recordings
    mice = os.listdir(base)
    for mouse in mice:
        if mouse=="AV008":
            continue
        name_path = os.path.join(base, mouse)
        probes = os.listdir(name_path)
        for probe in probes:
            name_probe = os.path.join(name_path, probe)
            locations = os.listdir(name_probe)
            for location in locations:
                name_probe_location = os.path.join(name_probe, location)
                inference(base, mouse, probe, location, "test")