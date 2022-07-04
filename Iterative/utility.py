import sys
sys.path.append('../data')
sys.path.append('..')
from rotation_representation import symmetric_orthogonalization
sys.path.append('logs')
import matplotlib.pyplot as plt
from torch import Tensor
import torchvision


import glob
import numpy as np

import os
import ModelNetSO3
import torch.nn.functional as F
import torch.nn as nn
import time
import torch
from loss import loss_frobenius





def angle_error(t_R1, t_R2):
    ret = torch.empty((t_R1.shape[0]), dtype=t_R1.dtype, device=t_R1.device)
    rotation_offset = torch.matmul(
        t_R1.transpose(1, 2).double(), t_R2.double())
    tr_R = torch.sum(rotation_offset.view(-1, 9)
                     [:, ::4], axis=1)  # batch trace
    cos_angle = (tr_R - 1) / 2
    if torch.any(cos_angle < -1.1) or torch.any(cos_angle > 1.1):
        raise ValueError(
            "angle out of range, input probably not proper rotation matrices")
    cos_angle = torch.clamp(cos_angle, -1, 1)
    angle = torch.acos(cos_angle)
    return angle * (180 / np.pi)



def combine(R, tx,ty,tz, device='cuda'):
    T_pred = torch.ones((model_output.shape[0], 4, 4)).to(device)
    T_pred[:, :3, :3] = R_new
    T_pred[:, 0, 3] = tx
    T_pred[:, 1, 3] = ty
    T_pred[:, 2, 3] = tz
    T_pred[:, 3, :3] = torch.tensor(0)

    return T_pred


def calculate_T_pred(model_output, T_init, device, rot_repr='SVD'):
    '''
    SE3_parametrization
    R:           output of network
    R^k:         input pose of network, i.e., R in ex_init
    v_x,v_y,z_y: model output translation
    x,y,z:       input translation of network t in ex_init


    Idea from https://github.com/ylabbe/cosypose/
    Implemenation modified from https://github.com/olaals/end-to-end-RGB-pose-estimation-baseline
    R^(k+1) = RR^(k+1)
    T_delta*T_init = new T_CO.
    '''

    dR = symmetric_orthogonalization(model_output[:, :9])
    transl = model_output[:, 9:12]

    vx,vy,vz = transl[:, 0],transl[:, 1],transl[:, 2]

    R_k = T_init[:, :3, :3]

    flen = 50
    sw = 36
    img_res = 320

    ppm = sw / img_res
    fx = fy = flen / ppm

    z_k = T_init[:, 2, 3]

    z_new = vz * z_k

    x_k = T_init[:, 0, 3]
    y_k = T_init[:, 1, 3]
    x_new = (vx / fx + x_k / z_k) * z_new
    y_new = (vy / fy + y_k / z_k) * z_new
    
    R_k = R_k.float()
    R_new = torch.einsum('bij,bjk->bik', dR, R_k)

    T_pred = combine(R_new, x_new, y_new, z_new, device)

    return T_pred





def load_network_2(path, model):
    
    modelcheckpoint = torch.load(path)

    model.load_state_dict(modelcheckpoint['model_state_dict'])
    opt.load_state_dict(modelcheckpoint['optimizer_state_dict'])
    epoch = modelcheckpoint['epoch']

    return model
