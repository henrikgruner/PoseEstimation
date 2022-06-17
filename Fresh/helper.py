from model import ResnetRS
import torch.nn as nn
import torch
import sys
import os
from datetime import datetime
import glob
import pyrender
import trimesh as tm
os.environ['PYOPENGL_PLATFORM'] = 'egl'

def save_network(epoch, model, opt, model_name, path):
    opt_name = opt.__class__.__name__

    NAME = str(model_name) + '_state_dict_{}.pkl'.format(epoch)
    PATH = os.path.join(path, NAME)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
    }, PATH)


def load_network(path, model, opt, model_name, out_dim, numclasses):
    modelcheckpoint = torch.load(path)

    model.load_state_dict(modelcheckpoint['model_state_dict'])
    opt.load_state_dict(modelcheckpoint['optimizer_state_dict'])
    epoch = modelcheckpoint['epoch']

    return model, opt, epoch


import sys
sys.path.append('../models')
sys.path.append('../data')
sys.path.append('../6D')
sys.path.append('..')
sys.path.append('ModelNet40-norm-ply')
sys.path.append('logs')
from model import ResnetRS
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


def load_network(path, model):
    modelcheckpoint = torch.load(path)

    model.load_state_dict(modelcheckpoint['model_state_dict'])

    return model

def get_scene(mesh, ex, flen, img_res, sw):
    ex = np.linalg.inv(ex)
    scene = pyrender.Scene()
    scene.bg_color = (255, 255, 255)
    scene.add(mesh)

    ppm = sw / img_res
    fx = fy = flen / ppm
    vx = vy = img_res / 2

    camera = pyrender.IntrinsicsCamera(fx, fy, vx, vy)
    light = pyrender.SpotLight(color=np.ones(3), intensity=30.0,
                               innerConeAngle=np.pi / 8.0,
                               outerConeAngle=np.pi / 3.0)
    scene.add(light, pose=ex)
    scene.add(camera, pose=ex)

    return scene

# 6D/ModelNet40/NormalizedModelNet40/sofa/train/sofa_0681.ply
def render_from_id(cad_id, class_id, ex_curr,foldergg, dataset='ModelNet40-norm-ply', train=False):

    if(train):
        folder = 'train'
    else:
        folder = 'test'

    class_str = 'toilet'


    filename = class_str + '_'+str(class_id).zfill(4)+'.ply'

    path = os.path.join(dataset, class_str, folder, filename)

    mesh_org = tm.load(path)
    mesh = pyrender.Mesh.from_trimesh(mesh_org)
    img = render_image(mesh, ex_curr)
    plt.imshow(img)
    plt.title(str(i))
    plt.savefig(foldergg+'/'+str(i))
    plt.close()

      
    return img


def render_image(mesh, ex, flen=112, sw=112, img_res=224):

    scene = get_scene(mesh, ex, flen, img_res, sw)
    r = pyrender.OffscreenRenderer(
        viewport_width=img_res, viewport_height=img_res)
    color, depth = r.render(scene)
    color = color / 255
    r.delete()

    return color



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

    model_svd = load_network(
        path_svd, model_svd)
    model_gs = load_network(path_gs, model_gs)

    val_angle_errors = test_angle_SO3(
        model, opt, dl_eval, device)

    print('Class: {} Mean: {} Median: {} Max: {} std: {}'.format(class_type, np.mean(
        val_angle_errors), np.median(val_angle_errors), np.max(val_angle_errors), np.std(val_angle_errors)))

    plt.xlabel('Mean angle error')
    plt.ylabel('Frequency')

    plt.title(class_type)
    plt.hist(val_angle_errors, bins=50)
    plt.savefig(save_path + '/' + class_type + ".png")
    plt.close()
    return val_angle_errors


def viz(path_svd, path_gs, classes):

    svd_dim = 9
    ortho_dim = 6
    curr_class = 100
    model_svd = ResnetRS.create_pretrained(
        model_name, in_ch=3, num_classes=9)

    #model_gs = ResnetRS.create_pretrained(
    #    model_name, in_ch=3, num_classes=6)

    model_svd = nn.DataParallel(model_svd, device_ids=devices)
    #model_gs = nn.DataParallel(model_gs, device_ids=devices)

    model_svd = model_svd.to(device)
    #model_gs = model_gs.to(device)

    model_svd = load_network(
        path_svd, model_svd)

    # model_gs = load_network(
    #    path_gs, model_gs)
    index =345
    dl_eval = get_eval_loader(dataset, batch_size, dataset_dir, classes)

    for img, ex_in, c_id, _, cam, cad_id in dl_eval:
        model_svd.eval()
        with torch.no_grad():
            out = model_svd(img.to(device))
            R = symmetric_orthogonalization((out.view(-1, 3, 3)))

        angle = angle_error(R.to('cpu'), ex_in[:,:3,:3].to('cpu'))
        print(angle)

        img = img.to('cpu').detach().numpy()
        img = img[0].transpose(1, 2, 0)
        
        #ex = torch.zeros((R.shape[0], 4, 4)).to(device)
        #ex[:, :3, :3] = R
        #ex[:, 0, 3] = 0
        #ex[:, 1, 3] = 0
        #ex[:, 2, 3] = -2.5
        ex_in[:, 2, 3] = -5
        #ex[:, 3, 3] = torch.tensor(1)
        #render_from_id(cad_id, 0, ex.to('cpu').numpy()[0], foldergg = 'model')
        render_from_id(cad_id, 345, ex_in.to('cpu').numpy()[0], foldergg = 'true')
        plt.imshow(img)
        plt.savefig("org"+str(index))
        plt.close()
        index +=1



        #print(ex)
        #print(c_id)
        #print(cam)
        #print(cad_id)
        #exit()


classes = ['bathtub', 'bed', 'desk', 'dresser',
           'monitor', 'night_stand', 'chair', 'sofa', 'table', 'toilet']
classes = ['toilet']
path_svd = '../Fresh/logs/run026/saved_models/resnetrs101_state_dict_47.pkl'
path_gs = '../Fresh/logs/run027/saved_models/resnetrs101_state_dict_32.pkl'
if __name__ == '__main__':
    viz(path_svd, path_gs, classes) 
# automatically find the latest run folder
