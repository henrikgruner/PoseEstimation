import torch
from dataset import MyDataset
from models import Model
import numpy as np
from torch.utils.data import DataLoader
import torch
from scipy.ndimage import gaussian_filter1d
from pyquaternion import Quaternion

import matplotlib.pyplot as plt


def loss_frobenius(R_pred, R_true):
    difference = R_true - R_pred
    frob_norm = torch.linalg.matrix_norm(difference, ord='fro')

    return frob_norm.mean()


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

def getAngle(R1,R2):
    '''without the clamp -< might induce erros'''
    R1 = R1.detach().cpu().numpy()
    R2 = R2.detach().cpu().numpy()
    R = np.dot(R1,R2.T)
    theta = (np.trace(R) -1)/2
    theta = np.clip(theta, -1, 1)
    return np.arccos(theta) * (180/np.pi)


def train_model(model, dl_train,dl_eval, device = 'cuda' , epochs = 50, direct=False):
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    angle_errors = []
    out_errors = []
    for e in range(epochs):
        if e in [20,30,40,50,60,70,80,90]:
            for g in opt.param_groups:
                g['lr'] *= 0.5

        for i, (R, axes) in enumerate(train_loader):

            opt.zero_grad()
            R = R.to(device)
            axes = axes.to(device)
            axes = axes.view(-1,9)
            outR = model(axes)

            loss = loss_frobenius(outR, R)
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), clip = 1)
            opt.step()
            if(direct):
                angle =getAngle(outR, R)
                angle_errors.append(angle)
            else:
                angle = angle_error(outR, R)
                angle_errors.append(angle.detach().cpu().numpy())

        out_errors.append(np.mean(angle_errors))

    with torch.no_grad():
        test_angle_errors = []
        for i, (R, axes) in enumerate(dl_eval):
            opt.zero_grad()
            R = torch.tensor(R).to(device)
            outR = model(axes.to(device).view(-1,9))

            angle = angle_error(outR, R)

            test_angle_errors.append(angle.detach().cpu().numpy())
        test_angle = np.mean(test_angle_errors)

    return out_errors,test_angle


def plot(results, title, restricted = False):
    if(restricted):
        title
    for key, value in results.items():
        
        plt.plot(value, label=key)
        if(restricted):
            plt.ylim([0,5])
            plt.xlim([25,50])

    plt.xlabel('Epoch')
    plt.ylabel('Mean error angle (degrees)')
    plt.title(title)
    plt.legend(loc="upper right")
    if(restricted):
        title +='restricted'
    plt.savefig(title)
    plt.close()


max_angles = [np.pi/4, np.pi/2, np.pi]
pltangles = ["45", "90", "180"]


results ={'Euler':[], 'Quat': [], 'SVD': [], '5D':[], '6D': [], 'Direct': []}
for i,max_angle in enumerate(max_angles):

    dataset = MyDataset(max_angle, 30000)      
    testdata = MyDataset(max_angle, 6000)

    train_loader = DataLoader(
    dataset,
    batch_size=128,
    num_workers=1,
    drop_last = True)   

    test_loader = DataLoader(
    testdata,
    batch_size=128,
    num_workers=1,
    drop_last = True)

    for key in results.keys():

        if key == 'Direct':
            direct = True
        else:
            direct = False
        results[key], test_angle = train_model(Model(representation = key), train_loader, test_loader, direct = direct)
        print(key,': ',test_angle)
        print(key,': ', results[key])
   

    plot(results, pltangles[i])
    plot(results, pltangles[i], restricted= True)



