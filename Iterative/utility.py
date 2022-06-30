import sys
sys.path.append('../data')
sys.path.append('..')
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



def combine(R, t, device='cpu'):
    ex = torch.zeros((R.shape[0], 4, 4)).to(device)
    ex[:, :3, :3] = R
    ex[:, 0, 3] = t[0].clone().detach().requires_grad_(True)
    ex[:, 1, 3] = t[1].clone().detach().requires_grad_(True)
    ex[:, 2, 3] = t[2].clone().detach().requires_grad_(True)
    ex[:, 3, 3] = torch.tensor(1)
    return ex


def calculate_T_CO_pred(model_output, T_CO_init, rot_repr='SVD'):
    bsz = model_output.shape[0]
    if rot_repr == 'SVD':
        assert model_output.shape == (bsz, 12)
    elif rot_repr == '6D':
        assert model_output.shape == (bsz, 9)
    elif rot_repr == '3D':
        assert model_output.shape == (bsz, 6)

    assert T_CO_init.shape == (bsz, 4, 4)

    device = model_output.device
    bsz = T_CO_init.shape[0]
    T_CO_pred = torch.ones((bsz, 4, 4)).to(device)
    if rot_repr == 'SVD':
        dR = symmetric_orthogonalization(model_output[:, 0:9])
        vxvyvz = model_output[:, 9:12]
    elif rot_repr == '6D':
        dR = compute_rotation_matrix_from_ortho6d(model_output[:, 0:6])
        vxvyvz = model_output[:, 6:9]
    elif rot_repr == '3D':
        dR = vec_3d_to_SO3(model_output[:, 0:3])
        vxvyvz = model_output[:, 3:6]
    else:
        assert False
    vx = vxvyvz[:, 0]
    vy = vxvyvz[:, 1]
    vz = vxvyvz[:, 2]
    #vz = 1+nn.Tanh()(vz)*0.2

    R_k = T_CO_init[:, :3, :3]

    flen = 50
    sw = 36
    img_res = 320

    ppm = sw / img_res
    fx = fy = flen / ppm

    z_k = T_CO_init[:, 2, 3]

    z_kp1 = vz * z_k

    x_k = T_CO_init[:, 0, 3]
    y_k = T_CO_init[:, 1, 3]
    x_kp1 = (vx / fx + x_k / z_k) * z_kp1
    y_kp1 = (vy / fy + y_k / z_k) * z_kp1
    R_k = R_k.float()
    R_kp1 = torch.einsum('bij,bjk->bik', dR, R_k)

    # assemble
    T_CO_pred[:, :3, :3] = R_kp1
    T_CO_pred[:, 0, 3] = x_kp1
    T_CO_pred[:, 1, 3] = y_kp1
    T_CO_pred[:, 2, 3] = z_kp1
    T_CO_pred[:, 3, :3] = torch.tensor(0)

    return T_CO_pred


def SE3_parameterization(R, translation, ex, configs={'fx': 50, 'fy': 50}):
    '''
    SE3_parametrization
    R:           output of network
    R^k:         input pose of network, i.e., R in ex_init
    v_x,v_y,z_y: model output translation
    x,y,z:       input translation of network t in ex_init


    R^(k+1) = RR^(k+1)
    T_delta*T_CO_init = new T_CO.
    '''

    flen = 50
    sw = 36
    img_res = 320

    ppm = sw / img_res
    fx = fy = flen / ppm
    #vx = vy = img_res / 2
    vx, vy, vz = translation[:, 0], translation[:, 1], translation[:, 2]
    x, y, z = ex[:, 0, 3], ex[:, 1, 3], ex[:, 2, 3]

    z_new = vz * z

    x_new = (vx / fx + x / z) * z_new
    y_new = (vy / fy + y / z) * z_new

    R_new = torch.einsum('bij,bjk->bik', R, ex[:, :3, :3])

    ex_curr = combine(R_new, [x_new, y_new, z_new])

    return ex_curr


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


def load_network_2(path, model):
    
    modelcheckpoint = torch.load(path)

    model.load_state_dict(modelcheckpoint['model_state_dict'])
    opt.load_state_dict(modelcheckpoint['optimizer_state_dict'])
    epoch = modelcheckpoint['epoch']

    return model
