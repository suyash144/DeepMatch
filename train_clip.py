# Wentao Qiu, 2023-10-07
# qiuwentao1212@gmail.com


from hashlib import sha1
import logging
import os
from pathlib import Path
import typing as tp
import sys
import argparse
import h5py

import numpy as np
import tqdm 
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard  import SummaryWriter

from utils import metric
from utils.losses import *
from utils.npdataset import NeuropixelsDataset, TrainExperimentBatchSampler, ValidationExperimentBatchSampler
from models.mymodel import *


logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def validation(epoch, model, val_loader, clip_loss, writer):
    model.eval()  # set the model to evaluation mode
    clip_loss.eval()
    losses = metric.AverageMeter()
    experiment_accuracies = []
    if torch.cuda.is_available():
        model = model.cuda()

    with torch.no_grad():
        progress_bar = tqdm.tqdm(total=len(val_loader), desc='Epoch {:3d}'.format(epoch))
        for estimates, candidates,_,_,_ in val_loader:
            if torch.cuda.is_available():
                estimates = estimates.cuda()
                candidates = candidates.cuda()
            bsz = estimates.shape[0]
            # Forward pass
            estimates_output = model(estimates) # shape [bsz, channel*time]
            candidates_output = model(candidates) # shape [bsz, channel*time]
            loss_clip = clip_loss(estimates_output, candidates_output)
            loss = loss_clip
            losses.update(loss.item(), bsz)
            probs = clip_loss.get_probabilities(estimates_output, candidates_output)
            predicted_indices = torch.argmax(probs, dim=1)  # Get the index of the max probability for each batch element
            ground_truth_indices = torch.arange(bsz, device=estimates_output.device)  # Diagonal indices as ground truth
            correct_predictions = (predicted_indices == ground_truth_indices).sum().item()  # Count correct predictions
            accuracy = correct_predictions / bsz
            experiment_accuracies.append(accuracy)
            progress_bar.update(1)

        progress_bar.close()
    
    print('Epoch: %d'%(epoch), 'Validation Loss: %.9f'%(losses.avg), 'Validation Accuracy: %.9f'%(np.mean(experiment_accuracies)))
    writer.add_scalar('Validation/Loss', losses.avg, epoch)
    writer.add_scalar('Validation/Accuracy', np.mean(experiment_accuracies), epoch)
    return

def train(epoch, model, optimizer, train_loader, clip_loss, VICReg_loss, writer):
    model.train()
    clip_loss.train()
    losses = metric.AverageMeter()
    iteration = len(train_loader) * epoch
    if torch.cuda.is_available():
        model = model.cuda()

    progress_bar = tqdm.tqdm(total=len(train_loader), desc='Epoch {:3d}'.format(epoch))
    for estimates, candidates,_,_,_ in train_loader:
        bsz = estimates.shape[0]
        if torch.cuda.is_available():
            estimates = estimates.cuda()
            candidates = candidates.cuda()
        optimizer.zero_grad()
        estimates_output = model(estimates)
        candidates_output = model(candidates)
        loss_clip = clip_loss(estimates_output, candidates_output)
        # loss_reg = VICReg_loss(estimates_output, candidates_output)
        # loss = loss_clip + 0.2* loss_reg
        loss = loss_clip
        # update metric
        losses.update(loss.item(), bsz)
        loss.backward()
        optimizer.step()
        progress_bar.update(1)
        iteration += 1
        if iteration % 50 == 0:
            writer.add_scalar('Train/Loss', losses.avg, iteration)
            writer.add_scalar('Train/ClipLoss', loss_clip, iteration)
    
    progress_bar.close()
    print(' Epoch: %d'%(epoch), 'Loss: %.9f'%(losses.avg))
    return 

def run(args):
    save_folder = os.path.join('./ModelExp/experiments',  args.exp_name)
    ckpt_folder = os.path.join(save_folder,  'ckpt')
    log_folder = os.path.join(save_folder,  'log')
    os.makedirs(ckpt_folder, exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)
    writer = SummaryWriter(log_dir=log_folder)

    train_data_root = os.path.join(os.getcwd(), os.pardir, 'test_DATA_UnitMatch')
    np_dataset = NeuropixelsDataset(root=train_data_root,batch_size=args.batchsize, mode='train')
    train_sampler = TrainExperimentBatchSampler(np_dataset, args.batchsize, shuffle=True)
    train_loader = DataLoader(np_dataset, batch_sampler=train_sampler)
    val_sampler = ValidationExperimentBatchSampler(np_dataset, shuffle = True)
    val_loader = DataLoader(np_dataset, batch_sampler=val_sampler)

    # # print len of train and test
    print('train dataset length: %d'%(len(np_dataset)))

    # model = SpatioTemporalCNN(T = 82, C = 96).to(device)
    model = SpatioTemporalCNN_V2(n_channel=46,n_time=82,n_output=256).to(device)
    model = model.double()
    clip_loss = ClipLoss1D().to(device)
    # clip_loss = ClipLoss2D().to(device)
    VICReg_loss = VICReg(sim_coeff=1.0,std_coeff=1.0, cov_coeff=0.05).to(device)
    
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model_parameters = list(model.parameters()) + list(clip_loss.parameters())
    optimizer = optim.Adam(model_parameters, lr=args.lr)

    if args.cont:
        # load latest checkpoint
        ckpt_lst = os.listdir(ckpt_folder)
        ckpt_lst.sort(key=lambda x: int(x.split('_')[-1]))
        read_path = os.path.join(ckpt_folder, ckpt_lst[-1])
        print('load checkpoint from %s'%(read_path))
        checkpoint = torch.load(read_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        clip_loss.load_state_dict(checkpoint['clip_loss'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0

    for epoch in range(start_epoch, args.total_epoch):
        train(epoch, model, optimizer, train_loader, clip_loss, VICReg_loss, writer)
        if epoch % args.save_freq == 0:
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'clip_loss': clip_loss.state_dict(),
                'epoch': epoch,
            }
            save_file = os.path.join(ckpt_folder, 'ckpt_epoch_%s'%(str(epoch)))
            torch.save(state, save_file)

        # with torch.no_grad():
        # validate and test
        validation(epoch, model, val_loader, clip_loss, writer)
        # test(epoch, model, test_loader, writer)
    return 


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--exp_name', '-e', type=str, required=True, help="The checkpoints and logs will be save in ./checkpoint/$EXP_NAME")
    arg_parser.add_argument('--lr', '-l', type=float, default=1e-5, help="Learning rate")
    arg_parser.add_argument('--save_freq', '-s', type=int, default=1, help="frequency of saving model")
    arg_parser.add_argument('--total_epoch', '-t', type=int, default=20, help="total epoch number for training")
    arg_parser.add_argument('--cont', '-c', action='store_true', help="whether to load saved checkpoints from $EXP_NAME and continue training")
    arg_parser.add_argument('--batchsize', '-b', type=int, default=30, help="batch size")
    args = arg_parser.parse_args()

    run(args)

    
