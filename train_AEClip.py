# Wentao Qiu, 2023-10-07
# qiuwentao1212@gmail.com


import logging
import os, sys
import argparse

import numpy as np
import tqdm 
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard  import SummaryWriter

from utils import metric
from utils.losses import *
from utils.npdataset import NeuropixelsDataset, TrainExperimentBatchSampler, ValidationExperimentBatchSampler
from models.mymodel import *

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def validation(epoch,encoder,decoder,projector,val_loader,clip_loss,AE_loss,writer):
    encoder.eval()  
    decoder.eval()
    projector.eval()
    clip_loss.eval()
    losses = metric.AverageMeter()
    experiment_accuracies = []

    if torch.cuda.is_available():
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        projector = projector.cuda()
        clip_loss = clip_loss.cuda()
        AE_loss = AE_loss.cuda()

    with torch.no_grad():
        progress_bar = tqdm.tqdm(total=len(val_loader), desc='Epoch {:3d}'.format(epoch))
        for estimates, candidates,_,_,_ in val_loader:
            if torch.cuda.is_available():
                estimates = estimates.cuda()
                candidates = candidates.cuda()
                
            bsz = estimates.shape[0]
            # Forward pass
            enc_estimates = encoder(estimates)
            enc_candidates = encoder(candidates)
            proj_estimates = projector(enc_estimates)
            proj_candidates = projector(enc_candidates)
            dec_estimates = decoder(enc_estimates)
            dec_candidates = decoder(enc_candidates)

            loss_clip = clip_loss(proj_estimates, proj_candidates)
            loss_AE = AE_loss(dec_estimates, estimates) + AE_loss(dec_candidates, candidates)
            # loss_reg = VICReg_loss(estimates_output, candidates_output)
            # loss = loss_clip + 0.2* loss_reg
            loss = loss_clip + loss_AE
            losses.update(loss.item(), bsz)

            probs = clip_prob(enc_estimates, enc_candidates)
            predicted_indices = torch.argmax(probs, dim=1)  # Get the index of the max probability for each batch element
            ground_truth_indices = torch.arange(bsz, device=device)  # Diagonal indices as ground truth
            correct_predictions = (predicted_indices == ground_truth_indices).sum().item()  # Count correct predictions
            accuracy = correct_predictions / bsz
            experiment_accuracies.append(accuracy)
            progress_bar.update(1)

        progress_bar.close()
    
    print('Epoch: %d'%(epoch), 'Validation Loss: %.9f'%(losses.avg), 'Validation Accuracy: %.9f'%(np.mean(experiment_accuracies)))
    writer.add_scalar('Validation/Loss', losses.avg, epoch)
    writer.add_scalar('Validation/ClipLoss', loss_clip, epoch)
    writer.add_scalar('Validation/AELoss', loss_AE, epoch)
    writer.add_scalar('Validation/Accuracy', np.mean(experiment_accuracies), epoch)
    return

def train(epoch,encoder,decoder,projector,optimizer,train_loader,clip_loss,AE_loss,VICReg_loss,writer):
    encoder.train()
    decoder.train()
    projector.train()
    clip_loss.train()
    losses = metric.AverageMeter()
    iteration = len(train_loader) * epoch
    
    progress_bar = tqdm.tqdm(total=len(train_loader), desc='Epoch {:3d}'.format(epoch))
    
    if torch.cuda.is_available():
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        projector = projector.cuda()
        clip_loss = clip_loss.cuda()
        AE_loss = AE_loss.cuda()
        VICReg_loss = VICReg_loss.cuda()

    for estimates, candidates,_,_,_ in train_loader:
        bsz = estimates.shape[0]
        if torch.cuda.is_available():
            estimates = estimates.cuda()
            candidates = candidates.cuda()

        optimizer.zero_grad()
        enc_estimates = encoder(estimates)
        enc_candidates = encoder(candidates)
        proj_estimates = projector(enc_estimates)
        proj_candidates = projector(enc_candidates)
        dec_estimates = decoder(enc_estimates)
        dec_candidates = decoder(enc_candidates)
        loss_clip = clip_loss(proj_estimates, proj_candidates)
        loss_AE = AE_loss(dec_estimates, estimates) + AE_loss(dec_candidates, candidates)
        # loss_reg = VICReg_loss(estimates_output, candidates_output)
        # loss = loss_clip + 0.2* loss_reg
        loss = loss_clip + loss_AE

        # update metric
        losses.update(loss.item(), bsz)
        loss.backward()
        if metric.check_gradients(encoder):
            print("Stopping training due to gradient issues.")
            return  # Exit the training function
        
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        progress_bar.update(1)
        iteration += 1

        if iteration % 50 == 0:
            writer.add_scalar('Train/Loss', losses.avg, iteration)
            writer.add_scalar('Train/ClipLoss', loss_clip, iteration)
            writer.add_scalar('Train/AELoss', loss_AE, iteration)
    
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

    train_data_root = os.path.join(os.getcwd(), os.pardir, 'R_DATA_UnitMatch')
    np_dataset = NeuropixelsDataset(root=train_data_root,batch_size=args.batchsize, mode='train')
    train_sampler = TrainExperimentBatchSampler(np_dataset, args.batchsize, shuffle=True)
    train_loader = DataLoader(np_dataset, batch_sampler=train_sampler)
    val_sampler = ValidationExperimentBatchSampler(np_dataset, shuffle = True)
    val_loader = DataLoader(np_dataset, batch_sampler=val_sampler)

    # print len of train and test
    print('train dataset length: %d'%(len(np_dataset)))

    # model = SpatioTemporalCNN(T = 82, C = 96).to(device)
    encoder = SpatioTemporalCNN_V3(n_channel=46,n_time=82,n_output=256).to(device)
    decoder = Decoder_SpatioTemporalCNN_V3(n_channel=46,n_time=82,n_input=256).to(device)
    projector = Projector(input_dim=256, output_dim=128, hidden_dim=256, n_hidden_layers=1, dropout=0.1).to(device)
    encoder = encoder.double()
    decoder = decoder.double()
    projector = projector.double()
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    projector = projector.to(device)
    clip_loss = ClipLoss1D().to(device)
    # clip_loss = ClipLoss2D().to(device)
    VICReg_loss = VICReg(sim_coeff=1.0,std_coeff=1.0, cov_coeff=0.05).to(device)
    AE_loss = AELoss(lambda1 = 0.05,lambda2 = 0.95).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model_parameters = list(encoder.parameters()) + list(decoder.parameters()) + list(projector.parameters()) + list(clip_loss.parameters())
    optimizer = optim.Adam(model_parameters, lr=args.lr)

    if args.cont:
        # load latest checkpoint
        ckpt_lst = os.listdir(ckpt_folder)
        ckpt_lst.sort(key=lambda x: int(x.split('_')[-1]))
        read_path = os.path.join(ckpt_folder, ckpt_lst[-1])
        print('load checkpoint from %s'%(read_path))
        checkpoint = torch.load(read_path)
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        projector.load_state_dict(checkpoint['projector'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        clip_loss.load_state_dict(checkpoint['clip_loss'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0

    for epoch in range(start_epoch, args.total_epoch):
        train(epoch, encoder, decoder, projector, optimizer, train_loader, clip_loss, AE_loss, VICReg_loss, writer)
        if epoch % args.save_freq == 0:
            state = {
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'projector': projector.state_dict(),
                'optimizer': optimizer.state_dict(),
                'clip_loss': clip_loss.state_dict(),
                'epoch': epoch,
            }
            save_file = os.path.join(ckpt_folder, 'ckpt_epoch_%s'%(str(epoch)))
            torch.save(state, save_file)

        # with torch.no_grad():
        # validate and test
        validation(epoch, encoder, decoder, projector, val_loader, clip_loss, AE_loss, writer)
    # test(epoch, model, test_loader, writer)
    return

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--exp_name', '-e', type=str, required=True, help="The checkpoints and logs will be save in ./checkpoint/$EXP_NAME")
    # arg_parser.add_argument('--finetune', '-f', type=str, required=True, help="Load the AE encoder from the path ./checkpoint/$finetune")
    arg_parser.add_argument('--lr', '-l', type=float, default=1e-5, help="Learning rate")
    arg_parser.add_argument('--save_freq', '-s', type=int, default=1, help="frequency of saving model")
    arg_parser.add_argument('--total_epoch', '-t', type=int, default=32, help="total epoch number for training")
    arg_parser.add_argument('--cont', '-c', action='store_true', help="whether to load saved checkpoints from $EXP_NAME and continue training")
    arg_parser.add_argument('--batchsize', '-b', type=int, default=32, help="batch size")
    args = arg_parser.parse_args()

    run(args)

    
