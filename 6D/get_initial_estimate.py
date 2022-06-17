import sys
sys.path.append('../models')
sys.path.append('../data')
sys.path.append('../Fresh/')
sys.path.append('logs')
from model import ResnetRS
from loss import loss_frobenius
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from utility import *
import os
import spatialmath as sm

from model import ResnetRS
import numpy as np
import glob
from dataset import get_modelnet_loaders, get_eval_loader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch import Tensor
import matplotlib.pyplot as plt


def test_angle_SO3(model, opt, dl_eval, device):
    '''
    Either for validation or for testing

    '''

    model.eval()
    with torch.no_grad():

        angle_errors = []
        for img, ex, _, _, _, _ in dl_eval:

            img = img.to(device)
            # rotation matrix
            R = ex[:, :3, :3].to(device)

            # preditc rot matrix

            out = model(img)
            # Does not need to reshape
            out = symmetric_orthogonalization((out.view(-1, 3, 3)))

            angle_errors.append(angle_error(out, R).mean().item())

    return angle_errors


def load_network(path, model, opt, model_name):
    modelcheckpoint = torch.load(path)
    model.load_state_dict(modelcheckpoint['model_state_dict'])
    opt.load_state_dict(modelcheckpoint['optimizer_state_dict'])
    epoch = modelcheckpoint['epoch']

    return model, opt, epoch

# TODO
# Test resnetrs
# Spit out R from image


def get_Rinit_from_net(model, image, device):
    model.eval()
    with torch.no_grad():
        img = image.to(device, dtype=torch.float)
        # predict rot matrix
        out = model(img)
        # Does not need to reshape
        out = symmetric_orthogonalization((out.view(-1, 3, 3)))
    return out


def get_random_T_init(T):
    R = ex[:, :3, :3]
    new_R = get_Rinit_random(R, 0.1)
    # Fix translation here
    #
    raise NotImplementedError


def get_Rinit_random(R, max_change):
    return R * (1 + np.randon.uniform(max_change))


def get_small_change_T(T, mac, mtc):
    '''
    Function takes a transformation matrix
    and alters it randomly in the ranges given
    Input:
    - T - transform matrix
    - mac - Maximum angle change (degrees)
    - mtc - Maximum translation change (meters)
    Output:
    - T_random = slightly altered T
    '''
    rand = np.random.uniform
    change = rand(-mac, mac)
    x_change, y_change, z_change = rand(-mtc, mtc, size=3)

    R = sm.SO3.AngleAxis(change, get_random_unit_axis(), unit='deg')
    transl = np.array([x_change, y_change, z_change])
    T_new = sm.SE3.Rt(R, transl)

    return T_new * T


if __name__ == '__main__':
    dataset = 'SO3'
    dataset_dir = '../data/datasets/'
    batch_size = 1
    dl_eval = get_eval_loader(dataset, batch_size, dataset_dir, ['toilet'])
    img, ex, _, _, _, _ = next(iter(dl_eval))

    model_name = 'resnetrs101'

    device = torch.device("cuda" if(
        torch.cuda.is_available()) else "cpu")
    devices = [d for d in range(torch.cuda.device_count())]
    device_names = [torch.cuda.get_device_name(d) for d in devices]

    print("cuda: ", torch.cuda.is_available())
    print("count: ", torch.cuda.device_count())
    print("names: ", device_names)
    print("device ids:", devices)

    model = ResnetRS.create_pretrained(
        model_name, in_ch=3, num_classes=1)

    model = nn.DataParallel(model, device_ids=devices)
    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0)
    path = '../Fresh/logs/run022/saved_models/resnetrs101_state_dict_28.pkl'
    model, opt, epoch = load_network(
        path, model, opt, model_name)

    R = get_Rinit_from_net(model, img, device).float().to('cpu')
    R_real = ex[:, :3, :3].float().to('cpu')

    print(R.to('cpu'))
    print(R_real.to('cpu'))
    print(angle_error(R, R_real))
    print(angle_error(R_real, R))
