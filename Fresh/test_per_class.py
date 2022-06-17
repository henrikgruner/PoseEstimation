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


# TODO - Save net per tenth

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


def test_angle_SO3(model, opt, dl_eval, device):
    '''
    Either for validation or for testing

    '''

    model.eval()
    with torch.no_grad():

        angle_errors = []
        for i, (img, ex, cs, di, gg, dd) in enumerate(dl_eval):

            img = img.to(device)
            # rotation matrix
            R = ex[:, :3, :3].to(device)

            # preditc rot matrix

            out = model(img)
            # Does not need to reshape
            out = symmetric_orthogonalization((out.view(-1, 3, 3)))
            angle = angle_error(out, R).mean().item()
            original_angle = angle
            if(angle > 150):
                outx, outy, outz = rotate_by_180(
                    out)
                anglex = angle_error(outx.unsqueeze(dim=0).to(
                    'cuda').double(), R).mean().item()
                angley = angle_error(outy.unsqueeze(dim=0).to(
                    'cuda').double(), R).mean().item()
                anglez = angle_error(outz.unsqueeze(dim=0).to(
                    'cuda').double(), R).mean().item()
                angle = np.min([anglex, angley, anglez])

                print(round(original_angle), round(angle), round(
                    anglex), round(angley), round(anglez))

            angle_errors.append(angle)

    return angle_errors


def load_network(path, model, opt, model_name):
    modelcheckpoint = torch.load(path)

    model.load_state_dict(modelcheckpoint['model_state_dict'])
    opt.load_state_dict(modelcheckpoint['optimizer_state_dict'])
    epoch = modelcheckpoint['epoch']

    return model, opt, epoch

# Frobenius norm


# Brief setup
batch_size = 1
dataset = 'SO3'
dataset_dir = '../data/datasets/'


model_name = 'resnetrs101'


device = torch.device("cuda" if(
    torch.cuda.is_available()) else "cpu")
devices = [d for d in range(torch.cuda.device_count())]
device_names = [torch.cuda.get_device_name(d) for d in devices]

print("cuda: ", torch.cuda.is_available())
print("count: ", torch.cuda.device_count())
print("names: ", device_names)
print("device ids:", devices)


# SOFA


def test_angle(path, save_path, class_type='chair'):
    dl_eval = get_eval_loader(dataset, batch_size, dataset_dir, [class_type])
    model = ResnetRS.create_pretrained(
        model_name, in_ch=3, num_classes=1)

    model = nn.DataParallel(model, device_ids=devices)
    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0)

    model, opt, epoch = load_network(
        path, model, opt, model_name)

    val_angle_errors = test_angle_SO3(
        model, opt, dl_eval, device)

    print('Class: {} Mean: {} Median: {} Max: {} std: {}'.format(class_type, np.mean(
        val_angle_errors), np.median(val_angle_errors), np.max(val_angle_errors), np.std(val_angle_errors)))

    plt.xlabel('Mean angle error')
    plt.ylabel('Frequency')

    plt.title(class_type)
    plt.hist(val_angle_errors, bins=50)
    plt.savefig(class_type)
    plt.close()
    return val_angle_errors


classes = ['bathtub', 'table']
paths = ['../Fresh/logs/run026/saved_models/resnetrs101_state_dict_47.pkl']
save_paths = ['run-026-epoch47-all']


# automatically find the latest run folder
classes = ['bathtub', 'bed', 'chair', 'desk', 'dresser',
           'monitor', 'night_stand', 'sofa', 'table', 'toilet']


for i, path in enumerate(paths):
    print(path)
    latex = []
    all_angles = []
    for c in classes:
        if not os.path.isdir(save_paths[i]):
            os.makedirs(save_paths[i])

        val_angle_errors = test_angle(path, save_paths[i], c)

        all_angles.extend(val_angle_errors)
        data = dict()
        data["row1"] = [round(np.mean(
            val_angle_errors), 2), round(np.median(val_angle_errors), 2), round(np.std(val_angle_errors), 2)]

        texdata = "\\hline\n"
        for label in sorted(data):
            if label == "z":
                texdata += "\\hline\n"
            texdata += f"{label} & {' & '.join(map(str,data[label]))} \\\\\n"

        print(texdata, end="")
        latex.append(texdata)

    with open(save_paths[i] + '/dump.txt', 'w') as file:
        for g in latex:
            file.write(g)

print('Median:', np.median(all_angles))
print('Mean: ', np.mean(all_angles))
all = np.array(all_angles)


print('acc pi/24', (all < 30).sum() / len(all_angles))
print('acc pi/12', (all < 15).sum() / len(all_angles))
print('acc pi/24', (all < 7.5).sum() / len(all_angles))
