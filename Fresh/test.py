import torch
from model import ResnetRS
import numpy as np
import sys
sys.path.append('../models')
sys.path.append('../data')
sys.path.append('..')
sys.path.append('../CosyPose')
sys.path.append('logs')
from model import ResnetRS
from loss import loss_frobenius
import torch
from viz import get_img, rotate_by_180
import time
import torch.nn as nn
import torch.nn.functional as F
import ModelNetSO3
import os
from helper import save_network
from model import ResnetRS
import numpy as np
import glob
from dataset import get_modelnet_loaders, get_eval_loader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch import Tensor
import matplotlib.pyplot as plt


def loss_frobenius(R_pred, R_true):
    difference = R_true - R_pred
    frob_norm = torch.linalg.matrix_norm(difference, ord='fro')
    return frob_norm.mean()


def angle_error_np(R_1, R_2):
    tr = np.trace(np.matmul(R_1.transpose(), R_2))
    angle = np.arccos((tr - 1) / 2) * (180.0 / np.pi)
    return angle


def angle_error(t_R1, t_R2):
    ret = torch.empty((t_R1.shape[0]), dtype=t_R1.dtype, device=t_R1.device)
    rotation_offset = torch.matmul(t_R1.transpose(1, 2), t_R2)
    tr_R = torch.sum(rotation_offset.view(-1, 9)
                     [:, ::4], axis=1)  # batch trace
    cos_angle = (tr_R - 1) / 2
    if torch.any(cos_angle < -1.1) or torch.any(cos_angle > 1.1):
        raise ValueError(
            "angle out of range, input probably not proper rotation matrices")
    cos_angle = torch.clamp(cos_angle, -1, 1)
    angle = torch.acos(cos_angle)
    return angle * (180 / np.pi)


def symmetric_orthogonalization(x):
    """
    Code from https://github.com/amakadia/svd_for_pose
    Maps 9D input vectors onto SO(3) via symmetric orthogonalization.
    x: should have size [batch_size, 9]
    Output has size [batch_size, 3, 3], where each inner 3x3 matrix is in SO(3).
    """
    m = x.view(-1, 3, 3)
    u, s, v = torch.svd(m)
    vt = torch.transpose(v, 1, 2)
    det = torch.det(torch.matmul(u, vt))
    det = det.view(-1, 1, 1)
    vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
    r = torch.matmul(u, vt)
    return r


'''
gg = np.array([0.1, 1, 2, 3])
all_angles = [1, 2]
print('acc pi/24', (gg < np.pi / 6).sum() / len(all_angles))
'''
batch_size = 1
dataset = 'SO3'
dataset_dir = '../data/datasets/'


classes = ['sofa']


dl_train, dl_eval = get_modelnet_loaders(
    dataset, batch_size, dataset_dir, classes)

print(len(dl_train))
print(len(dl_eval))
