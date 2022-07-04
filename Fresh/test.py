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
from helper import save_network, load_network
from render_utility import render_from_id
from model import ResnetRS
import numpy as np
import glob
from dataset import get_modelnet_loaders, get_eval_loader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch import Tensor
from PIL import Image
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
#batch_size = 1
#dataset = 'SO3'
#dataset_dir = '../data/datasets/'
def combine_imgs(img1, img2):
    gs1 = np.mean(img1, axis=2)
    gs2 = np.mean(img2, axis=2)
    img = np.zeros((gs1.shape[0], gs1.shape[1], 3))
    img[:, :, 0] = gs1
    img[:, :, 1] = gs2
    img[:, :, 2] = 255
    return img

def plot(imgs, titles, save_title, angles):
    fig = plt.figure(figsize=(10, 7))

    # setting values to rows and column variables
    rows = 1
    columns = len(imgs)
    
    for i, img in enumerate(imgs):
        print(img.shape)    
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(titles[i]+str(angles[i]))

    plt.savefig(os.path.join('illustrations',save_title))
    plt.close()


#classes = ['sofa']

def test_plot_SO3(model, opt, dl_eval, device):
    '''
    Either for validation or for testing

    '''

    model.eval()
    with torch.no_grad():

        angle_errors = []

        for i, (img, ex, class_id, di, cad_idx) in enumerate(dl_eval):
            if(i > 100):
                break
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
                angles = [anglex,angley,anglez]

                index_min = np.argmin(angles)

                out_out = [outx,outy,outz]

                angle = angles[index_min]
                out = out_out[index_min]
            ex[:,2,3] = -2.5
            ex_out = ex.detach().clone()
            ex_out[:,:3,:3] = out



               

            gt_img = render_from_id(di[0], ex[0].detach().cpu().numpy())
            ren_img = render_from_id(di[0], ex_out[0].detach().cpu().numpy())
            combined_img = combine_imgs(gt_img, ren_img)

            plt.imshow(gt_img)
            plt.axis('off')
            plt.savefig('ex/gt'+str(round(angle,1))+di[0]+'.png')
            plt.close()
            plt.imshow(ren_img)
            plt.axis('off')
            plt.savefig('ex/ren'+str(round(angle,1))+di[0]+'.png')
            plt.close()
            plt.imshow(combined_img)
            plt.axis('off')
            plt.savefig('ex/comb'+str(round(angle,1))+di[0]+'.png')
            plt.close()

path = 'logs/run26/saved_models/resnetrs101_state_dict_47.pkl'

model_name = 'resnetrs101'


device = torch.device("cuda")
devices = [d for d in range(torch.cuda.device_count())]
device_names = [torch.cuda.get_device_name(d) for d in devices]



model = ResnetRS.create_pretrained(
    model_name, in_ch=3, num_classes=1)

model = nn.DataParallel(model, device_ids=devices)
model = model.to(device)
opt = torch.optim.SGD(model.parameters(), lr=0)

model = load_network(
    path, model)
batch_size = 1
dataset = 'SO3'
dataset_dir = 'datasetSO3/'
classes = ['bathtub', 'bed', 'chair', 'desk', 'dresser',
           'monitor', 'night_stand', 'sofa', 'table', 'toilet']

#img = render_from_id(466, class_id, ex[0])
dl_eval = get_eval_loader(dataset, batch_size, dataset_dir, classes, shuffle = 'true')
test_plot_SO3(model, opt, dl_eval, device)

