import matplotlib.pyplot as plt
import torch
import os
import torch
import os
import numpy as np
from prepare import get_sampled_rotation_matrices_by_axisAngle
import glob
import torch
import sys
sys.path.append('../6D')
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch import Tensor
import time

from model_fetch import Model

class ModelNetDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, sample_num=1024):
        super(ModelNetDataset, self).__init__()
        self.paths = [os.path.join(data_folder, i)
                      for i in os.listdir(data_folder)]
        self.sample_num = sample_num
        self.size = len(self.paths)
        print(f"dataset size: {self.size}")

    def __getitem__(self, index):
        fpath = self.paths[index % self.size]
        pc = np.loadtxt(fpath)
        pc = np.random.permutation(pc)
        return pc[:self.sample_num, :].astype(float)

    def __len__(self):
        return self.size

def load_network(path, model, opt, model_name, out_dim, numclasses):
    modelcheckpoint = torch.load(path)

    model.load_state_dict(modelcheckpoint['model_state_dict'])
    opt.load_state_dict(modelcheckpoint['optimizer_state_dict'])
    epoch = modelcheckpoint['epoch']

    return model, opt, epoch


def compute_geodesic_distance_from_two_matrices(m1, m2):
    batch=m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3
    
    cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()) )
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda())*-1 )
    
    
    theta = torch.acos(cos)
    
    #theta = torch.min(theta, 2*np.pi - theta)
    
    
    return theta
def visualize(pc, pred_r, gt_r, title = 'none'):
    pc = torch.autograd.Variable(pc.float().to('cpu'))  # num*3

    pc_pred = torch.bmm(pc, pred_r.float())
    pc_gt = torch.bmm(pc, gt_r)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc[...,0], pc[...,1],pc[...,2])
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc_pred[..., 0], pc_pred[..., 1], pc_pred[..., 2])
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc_gt[..., 0], pc_gt[..., 1], pc_gt[..., 2])
    ax.axis('off')
    plt.savefig(title)

def test(model, test_folder, device, batch):
    angle_errors = []
    val_loss = []
    geos = []
    model.eval()
    test_path_list = [os.path.join(test_folder, i)
                      for i in os.listdir(test_folder)]

    with torch.no_grad():
        for i, path in enumerate(test_path_list):
            tmp = torch.load(path)
            pc2 = tmp['pc'].cpu().cuda()
            gt_rmat = tmp['rgt'].cpu().cuda()

            out = model(pc2.transpose(1, 2))
            if(i % 10 == 0):
                visualize(pc2.detach().cpu(), out.detach().cpu(), gt_rmat.detach().cpu(), str(i))



            geo = compute_geodesic_distance_from_two_matrices(out, gt_rmat).detach().to('cpu').numpy()
            geos.append(geo)

    return angle_errors, geo

data_folder = 'modelnet40_manually_aligned/chair'
train_folder = os.path.join(data_folder, 'train_pc')
val_folder = os.path.join(data_folder, 'test_fix')
train_dataset = ModelNetDataset(train_folder, sample_num=1024)

dl_train = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)




batch = 1
pc = next(iter(dl_train))
point_num = 1024
device = 'cuda'
pc = torch.autograd.Variable(pc.float().to('cpu'))  # num*3
model = Model("svd").to(device)
path = 'logs/run007/saved_models/resnetrs101_state_dict_560.pkl'
opt = torch.optim.SGD(model.parameters(), lr = 0)
load_network(path, model, opt, 'gg','gg', 'gg')
angle_errors, geo = test(model, val_folder, 'cuda', 1)

print(np.mean(geo*180/np.pi))