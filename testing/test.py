import os, sys

if __name__ == '__main__':
    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, os.path.join(os.getcwd(), os.pardir))

from utils.losses import *
from utils.npdataset import NeuropixelsDataset, ValidationExperimentBatchSampler
import numpy as np
from models.mymodel import *
from torch.utils.data import DataLoader
import tqdm
from utils import metric
import pandas as pd
import mat73
from utils.myutil import get_exp_id


def test(mouse:str, probe:str, loc:str, model_name:str, device = "cpu"):
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
    # model_name = "test"
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

    test_data_root = os.path.join(os.path.dirname(os.getcwd()), 'R_DATA_UNITMATCH')
    test_dataset = NeuropixelsDataset(root=test_data_root, batch_size=32, mode='val', m=mouse, p=probe, l=loc)
    test_sampler = ValidationExperimentBatchSampler(test_dataset, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler)

    print(f"Length of test dataset: {len(test_dataset)}")

    server_root = r"\\znas\Lab\Share\UNITMATCHTABLES_ENNY_CELIAN_JULIE\FullAnimal_KSChanMap"

    with torch.no_grad():
        progress_bar = tqdm.tqdm(total=len(test_loader))

        # Initialise the matrix that will store the pairwise probabilities for every combination of neurons
        prob_matrix = np.empty((len(test_dataset), len(test_dataset)))
        # And the similarity matrix
        sim_matrix = np.empty((len(test_dataset), len(test_dataset)))
        # This matrix will store the experiment id for each entry, corresponding to the 2 above.
        which_exp = np.empty((len(test_dataset), len(test_dataset)))
        # And this will store the unit id (within the experiment).
        unit_id = np.empty((len(test_dataset), len(test_dataset)))
        
        mt_path = os.path.join(test_data_root, mouse, probe, loc, "matchtable.csv")
        try:
            matchtable = pd.read_csv(mt_path)
        except:
            print("No matchtable found for this combination of (mouse, probe, location): ")
            print(f"Mouse: {mouse}, Probe: {probe}, Location: {loc}")
            raise ValueError()
        um_path = os.path.join(server_root, mouse, probe, loc, "UnitMatch", "UnitMatch.mat")
        um = f = mat73.loadmat(um_path)
        path_list = um["UMparam"]["KSDir"]
        path_dict = {}
        for i, path in enumerate(path_list):
            p = get_exp_id(path, mouse)
            path_dict[p] = i+1

        for i, (estimates_i, candidates_i,_, exp_ids_i, filepaths_i) in enumerate(test_loader):
            if torch.cuda.is_available():
                estimates_i = estimates_i.cuda()
            bsz_i = estimates_i.shape[0]
            exp_id_i = exp_ids_i[0]                     # this should be the same for all files in the batch
            rec_ses1 = path_dict[exp_id_i]

            # Forward pass
            enc_estimates_i = model(estimates_i)        # shape [bsz, 256]

            for j, (estimates_j, candidates_j,_,exp_ids_j,filepaths_j) in enumerate(test_loader):
                if torch.cuda.is_available():
                    candidates_j = candidates_j.cuda()
                bsz_j = candidates_j.shape[0]
                enc_candidates_j = model(candidates_j)
                exp_id_j = exp_ids_j[0]
                recses2 = path_dict[exp_id_j]

                prob_matrix[i:i+bsz_i, j:j+bsz_j] = clip_prob(enc_estimates_i, enc_candidates_j)
                sim_matrix[i:i+bsz_i, j:j+bsz_j] = clip_sim(enc_estimates_i, enc_candidates_j)
            progress_bar.update(1)
        progress_bar.close()
    return prob_matrix, sim_matrix

if __name__ == '__main__':
    # example args to check test function works
    # probs, sims = test("AL032", "19011111882", "2", "test")
    probs, sims = test("AL031", "19011116684", "1", "test")