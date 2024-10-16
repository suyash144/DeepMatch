import os, sys

if __name__ == '__main__':
    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, os.path.join(os.getcwd(), os.pardir))

from utils.losses import *
from utils.npdataset import NeuropixelsDataset, ValidationExperimentBatchSampler
import numpy as np
from models.mymodel import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import mat73
from utils.myutil import get_exp_id, get_unit_id
from scipy.special import softmax

def load_trained_model(model_name:str, device="cpu"):
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

    # Can also return projector if needed
    return model

def write_to_matchtable(model, test_data_root, test_loader, mouse, probe, loc, fast=True):
    """
    fast mode assumes that the matchtable and data are sorted consistently so that elements in similarity
    and probability matrices map to rows in match table trivially.
    slow mode does not but takes much longer.
    """

    server_root = r"\\znas\Lab\Share\UNITMATCHTABLES_ENNY_CELIAN_JULIE\FullAnimal_KSChanMap"

    with torch.no_grad():
        progress_bar = tqdm(total=len(test_loader))
        
        mt_path = os.path.join(test_data_root, mouse, probe, loc, "matchtable.csv")

        if os.path.exists(os.path.join(test_data_root, mouse, probe, loc, "new_matchtable.csv")):
            print("New matchtable already exists - continuing would overwrite.")
            return
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

        print("Loaded um and mt. path_dict created")
        for estimates_i, _,_, exp_ids_i, filepaths_i in tqdm(test_loader):
            if torch.cuda.is_available():
                estimates_i = estimates_i.cuda()
            bsz_i = estimates_i.shape[0]
            exp_id_i = exp_ids_i[0]                     # this should be the same for all files in the batch
            recses1 = path_dict[exp_id_i]

            # Forward pass
            enc_estimates_i = model(estimates_i)        # shape [bsz, 256]

            for _, candidates_j,_,exp_ids_j,filepaths_j in tqdm(test_loader):
                if torch.cuda.is_available():
                    candidates_j = candidates_j.cuda()
                bsz_j = candidates_j.shape[0]
                enc_candidates_j = model(candidates_j)
                exp_id_j = exp_ids_j[0]
                recses2 = path_dict[exp_id_j]

                if fast:
                    s = clip_sim(enc_estimates_i, enc_candidates_j)

                    rows = ((mt["RecSes1"]==recses1) & (mt["RecSes2"]==recses2))
                    mt.loc[rows, "DNNSim"] = s.flatten()
                    mt.loc[rows, "DNNProb"] = F.softmax(s, dim=1).flatten()
                    
                else:
                    # do things the slow way 
                    for unit_idx_i, est in tqdm(enumerate(enc_estimates_i)):
                        id1 = get_unit_id(filepaths_i[unit_idx_i])
                        est.unsqueeze_(0)
                        sims = {}
                        for unit_idx_j, cand in tqdm(enumerate(enc_candidates_j)):
                            id2 = get_unit_id(filepaths_j[unit_idx_j])
                            cand.unsqueeze_(0)
                            # prob = clip_prob(est, cand).item()
                            sim = clip_sim(est, cand).item()
                            # Write to the matchtable now that we have ID1, ID2, RecSes1 and RecSes2
                            row = ((mt["ID1"]==id1) & (mt["ID2"]==id2) 
                                    & (mt["RecSes1"]==recses1) & (mt["RecSes2"]==recses2))
                            row_idx = np.argwhere(row).item()
                            sims[row_idx] = float(sim)
                            mt.loc[row_idx, "DNNSim"] = sim
                        probs = softmax(np.array(list(sims.values())))
                        for i ,row in enumerate(sims):
                            mt.loc[row, "DNNProb"] = probs[i]
            progress_bar.update(1)
        
        mt.to_csv(os.path.join(test_data_root, mouse, probe, loc, "new_matchtable.csv"))
        progress_bar.close()

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

    model = load_trained_model(model_name, device)

    test_dataset = NeuropixelsDataset(root=test_data_root, batch_size=1, mode='val', m=mouse, p=probe, l=loc)
    test_sampler = ValidationExperimentBatchSampler(test_dataset, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler)

    print(f"Length of test dataset: {len(test_dataset)}")

    write_to_matchtable(model, test_data_root, test_loader, mouse, probe, loc)

def inference_one_pair(rec1:str, rec2:str, model_name:str, device="cpu"):
    """
    rec1 and rec2 should be full absolute paths to the recordings we want to compare
    """

    loc = os.path.basename(os.path.dirname(rec1))
    probe = os.path.basename(os.path.dirname(os.path.dirname(rec1)))
    mouse = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(rec1))))
    test_data_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(rec1))))

    model = load_trained_model(model_name=model_name, device=device)
    
    # test dataset contains all recordings under this 
    test_dataset = NeuropixelsDataset(root=test_data_root, batch_size=1, mode='val', m=mouse, p=probe, l=loc)
    # so now drop all the ones we don't want to compare
    new_dict = {}
    for key in test_dataset.experiment_unit_map:
        if key == os.path.basename(rec1) or key == os.path.basename(rec2):
            new_dict[key] = test_dataset.experiment_unit_map[key]
    test_dataset.experiment_unit_map = new_dict
    test_dataset.all_files = [(exp, file) for exp, files in test_dataset.experiment_unit_map.items() for file in files]

    test_sampler = ValidationExperimentBatchSampler(test_dataset, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler)
    print(f"Length of test dataset: {len(test_dataset)}")

    write_to_matchtable(model, test_data_root, test_loader, mouse, probe, loc)
    

if __name__ == '__main__':
    # example args to check inference function works
    base = r"C:\Users\suyas\R_DATA_UnitMatch"

    # AL032 recordings
    # rec1=r"C:\Users\suyas\R_DATA_UnitMatch\AL032\19011111882\2\_2019-11-21_ephys_K1_PyKS_output"
    # rec2=r"C:\Users\suyas\R_DATA_UnitMatch\AL032\19011111882\2\_2019-11-22_ephys_K1_PyKS_output"

    # AL036 recordings
    rec1 = r"C:\Users\suyas\R_DATA_UnitMatch\AL036\19011116882\3\_2020-07-01_ephys__2020-07-01_stripe240_natIm_g0__2020-07-01_stripe240_natIm_g0_imec0_PyKS_output"
    rec2 = r"C:\Users\suyas\R_DATA_UnitMatch\AL036\19011116882\3\_2020-08-04_ephys__2020-08-04_stripe240r1_natIm_g0_imec0_PyKS_output"
    
    # AV008 recordings
    # rec1 = r"C:\Users\suyas\R_DATA_UnitMatch\AV008\Probe0\IMRO_7\_2022-03-21_ephys__2022-03-21_SparseNoiseNaturalImages_g0__2022-03-21_SparseNoiseNaturalImages_g0_imec0_pyKS_output"
    # rec2 = r"C:\Users\suyas\R_DATA_UnitMatch\AV008\Probe0\IMRO_7\_2022-03-22_ephys__2022-03-22_SparseNoiseNaturalImages_g0__2022-03-22_SparseNoiseNaturalImages_g0_imec0_pyKS_output"
    
    # to test on one specific PAIR of recordings
    # inference_one_pair(rec1, rec2, model_name = "incl_AV008")

    # to test on one specific SET of recordings (ie one group with same (mouse, probe, loc))
    # inference(base, "AL031", "19011116684", "1", "incl_AV008")

    # to test on ALL sets of recordings
    mice = os.listdir(base)
    fails = []
    for mouse in mice:
        name_path = os.path.join(base, mouse)
        probes = os.listdir(name_path)
        for probe in probes:
            name_probe = os.path.join(name_path, probe)
            locations = os.listdir(name_probe)
            for location in locations:
                name_probe_location = os.path.join(name_probe, location)
                try:
                    inference(base, mouse, probe, location, "incl_AV008")
                except:
                    fails.append((mouse, probe, location))
                    print(f"Error for {mouse, probe, location}")
    print(fails)