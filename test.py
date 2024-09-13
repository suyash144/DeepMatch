from utils.losses import *
from utils.npdataset import NeuropixelsDataset, ValidationExperimentBatchSampler
import numpy as np
from models.mymodel import *
import os
from torch.utils.data import DataLoader
import tqdm
from utils import metric

device = "cpu"


# Load the trained model
model = SpatioTemporalCNN_V2(n_channel=30,n_time=60,n_output=256).to(device)
model = model.double()
model_name = "test"
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

# Load the data to use for testing
# Want same (mouse, probe, location) but 2 different days

# Processed data:
day_1 = r"C:\Users\suyas\R_DATA_UnitMatch\AL032\19011111882\2\_2019-11-21_ephys_K1_PyKS_output"
day_2 = r"C:\Users\suyas\R_DATA_UnitMatch\AL032\19011111882\2\_2019-11-22_ephys_K1_PyKS_output"

train_data_root = os.path.join(os.path.dirname(os.getcwd()), 'test_data')
test_dataset = NeuropixelsDataset(root=train_data_root, batch_size=32, mode='val')
test_sampler = ValidationExperimentBatchSampler(test_dataset, shuffle = True)
test_loader = DataLoader(test_dataset, batch_sampler=test_sampler)

print(f"Length of test dataset: {len(test_dataset)}")

losses = metric.AverageMeter()
experiment_accuracies = []

with torch.no_grad():
    progress_bar = tqdm.tqdm(total=len(test_loader))
    for estimates, candidates,_,_,_ in test_loader:
        if torch.cuda.is_available():
            estimates = estimates.cuda()
            candidates = candidates.cuda()

        bsz = estimates.shape[0]
        # Forward pass
        enc_estimates = model(estimates) # shape [bsz, channel*time]
        enc_candidates = model(candidates) # shape [bsz, channel*time]
        proj_estimates = projector(enc_estimates)
        proj_candidates = projector(enc_candidates)
        loss_clip = clip_loss(proj_estimates, proj_candidates)
        loss = loss_clip
        losses.update(loss.item(), bsz)

        probs = clip_prob(enc_estimates, enc_candidates)
        predicted_indices = torch.argmax(probs, dim=1)  # Get the index of the max probability for each batch element
        ground_truth_indices = torch.arange(bsz, device=device)  # Diagonal indices as ground truth
        correct_predictions = (predicted_indices == ground_truth_indices).sum().item()  # Count correct predictions
        accuracy = correct_predictions / bsz
        experiment_accuracies.append(accuracy)
        progress_bar.update(1)
    progress_bar.close()

    print(f"Average loss: {losses.avg}")
    print(f"Experiment accuracies: {np.mean(experiment_accuracies)}")

# Construct similarity matrix