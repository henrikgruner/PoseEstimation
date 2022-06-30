
import sys
sys.path.append('../models')
sys.path.append('../data')
sys.path.append('..')
sys.path.append('logs')
sys.path.append('../6D')
sys.path.append('../Fresh')
from resnet import resnet50, resnet101, ResnetHead
import matplotlib.pyplot as plt
from loss import loss_frobenius
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import ModelNetSO3
import os
from helper import save_network
from model import ResnetRS
import numpy as np
import glob
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch import Tensor
from dataloader import get_upna_loaders

dl_train, dl_eval = get_upna_loaders(1, True, '')

data = next(iter(dl_train))
plt.imshow(data[0].cpu().numpy()[0].transpose(1,2,0))
plt.axis('off')
plt.savefig('person')
print(data[1])
print(data[0].shape)

