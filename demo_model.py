# Wentao Qiu, 2023-10-07
# qiuwentao1212@gmail.com

import random
import typing as tp
import matplotlib.pyplot as plt
import os, sys
from sklearn.manifold import TSNE

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from models.mymodel import SpatioTemporalCNN, SpatioTemporalAutoEncoder
from utils.visualize import *

def load_model(ckpt_path, device):
    # Define the model structure (this must match the structure of the saved model)
    model = SpatioTemporalCNN().to(device)
    model = model.double()
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model'])
    return model


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(1, 82, 96).to(device)   # shape (batch_size, channels, length)
    model = SpatioTemporalAutoEncoder(T=82, C=96).to(device)
    y = model(x)
    print(y.shape)

    
