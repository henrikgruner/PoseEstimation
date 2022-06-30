import sys
sys.path.append('../6D/data')
sys.path.append('../models')
sys.path.append('../data')
sys.path.append('..')
sys.path.append('logs')
sys.path.append('../CosyPose')
from dataset import get_6D_stripped_loader
import os

import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from model import ResnetRS
from helper import save_network
from helper import save_network
import numpy as np
import glob
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch import Tensor

def load_network(path, model, opt):
    modelcheckpoint = torch.load(path)

    model.load_state_dict(modelcheckpoint['model_state_dict'])
    opt.load_state_dict(modelcheckpoint['optimizer_state_dict'])
    epoch = modelcheckpoint['epoch']

    return model, opt, epoch


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



def loss_frobenius(R_pred, R_true):
    difference = R_true - R_pred
    frob_norm = torch.linalg.matrix_norm(difference, ord='fro')

    return frob_norm.mean()


def train_SO3(model, opt, dl_train, device, lossfunc = None, translation = True):
    '''
        input:
        model : network
        opt : optimizer
        criterion : loss function
        dl_train : dataloader with training data
    '''
    model.train()
    epoch_loss = []
    angle_errors = []
    for index, (img, ex, _, _) in enumerate(dl_train):
        opt.zero_grad()
        ex = ex.to(device)
        img = img.to(device, dtype = torch.float)
        # rotation matrix
        R = ex[:, :3, :3].to(device)

        # preditc rot matrix

        out = model(img)
        out_R = out[:,:9]


        #out = compute_rotation_matrix_from_ortho6d(out)
        out_R = symmetric_orthogonalization(out_R)
        loss_R = loss_frobenius(R, out_R)
        
        if(translation):
            out_t = out[:,9:12]
            out_x = out_t[:,0]
            out_y = out_t[:,1]
            out_z = out_t[:,2]


            flen = 50
            sw = 36
            img_res = 320

            ppm = sw / img_res
            fx = fy = flen / ppm
            x_gt = ex[:, 0, 3]
            y_gt = ex[:, 1, 3]
            z_gt = ex[:, 2, 3]

            z = out_z * -2.5
            x = (out_x/fx)
            y = (out_y/fy)

            transl =torch.stack((x,y,z),1)

            transl_gt = torch.stack((x_gt, y_gt, z_gt), 1)
            loss_tr = F.mse_loss(transl, transl_gt, 'mean')
            loss = loss_R +loss_tr
        else:
            loss = loss_R


        epoch_loss.append(loss.item())

        angle = angle_error(out_R, R).mean().item()
        angle_errors.append(angle)

        loss.backward()
        opt.step()

    return epoch_loss, angle_errors


def test_SO3(model, opt, dl_eval, device, lossfunc = None,translation=False):
    '''
    Either for validation or for testing

    '''

    model.eval()
    with torch.no_grad():
        val_loss = []
        angle_errors = []
        for index, (img, ex, _, _) in enumerate(dl_eval):
            opt.zero_grad()
            ex = ex.to(device)
            img = img.to(device, dtype = torch.float)
            # rotation matrix
            R = ex[:, :3, :3].to(device)

            # preditc rot matrix

            out = model(img)
            out_R = out[:,0:9]
            out_R = symmetric_orthogonalization(out_R)

            loss_R = loss_frobenius(R, out_R)
            if(translation):

                out_t = out[:,9:12]
                out_x = out_t[:,0]
                out_y = out_t[:,1]
                out_z = out_t[:,2]


                flen = 50
                sw = 36
                img_res = 320

                ppm = sw / img_res
                fx = fy = flen / ppm
                x_gt = ex[:, 0, 3]
                y_gt = ex[:, 1, 3]
                z_gt = ex[:, 2, 3]

                z = out_z * -2.5
                x = (out_x/fx)
                y = (out_y/fy)

                transl =torch.stack((x,y,z),1)

                transl_gt = torch.stack((x_gt, y_gt, z_gt), 1)
                loss_tr = torch.norm(transl_gt-transl)
                loss = loss_R +loss_tr
            else:
                loss = loss_R


            #out = compute_rotation_matrix_from_ortho6d(out)




            val_loss.append(loss.item())

            angle = angle_error(out_R, R).mean().item()
            angle_errors.append(angle)



    return val_loss, angle_errors



batch_size = 256

#datapath = '../6D/data/'
epochs = 1000
drop_epochs = []
save_interval = 10
model_name = 'resnetrs101'
curr_epoch = 0
num_classes = 1

classes = ['toilet1','toilet2','toilet3']
datapath = '../data/dataset_2/'
DIR = 'logs2'
# automatically find the latest run folder
n = len(glob.glob(DIR + '/run*'))
NEW_DIR = 'run' + str(n + 1).zfill(3)

SAVE_PATH = os.path.join(DIR, NEW_DIR)
# create new directory

PATH = 'saved_models'
MODEL_SAVE_PATH = os.path.join(SAVE_PATH, PATH)
if not os.path.isdir(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

device = torch.device("cuda")
devices = [d for d in range(torch.cuda.device_count())]
device_names = [torch.cuda.get_device_name(d) for d in devices]

print("cuda: ", torch.cuda.is_available())
print("count: ", torch.cuda.device_count())
print("names: ", device_names)
print("device ids:", devices)


lr = 0.05

dl_train, dl_eval = get_6D_stripped_loader(batch_size, datapath, classes)

translation = True

model = ResnetRS.create_pretrained(
    model_name, in_ch=3,  num_classes=num_classes, out_features = 12)




model = nn.DataParallel(model, device_ids=devices)
model = model.to(device)
opt = torch.optim.SGD(model.parameters(), lr=lr)
curr_epoch = 0
resume = False

LOAD_PATH = 'logs2/run028/saved_models/resnetrs101_state_dict_70.pkl'
if(resume):
    NAME = str(model_name) + '_state_dict_{}.pkl'.format(curr_epoch)
    LOAD_PATH = os.path.join(LOAD_PATH, NAME)
    model, opt, epoch = load_network(
    'logs2/run028/saved_models/resnetrs101_state_dict_70.pkl', model, opt)
    for g in opt.param_groups:
        g['lr'] = lr
    print("Resuming training from epoch: ", epoch)

lossfunc = None

writer_train = SummaryWriter(
    log_dir=os.path.join(SAVE_PATH, 'train'), comment=f"_{model_name}_{opt.__class__.__name__}_{lr}_train")



for e in range(curr_epoch, epochs):
    verbose = e % int(save_interval) == 0 or e == (epochs - 1)

    epoch_time = time.time()

    if e in drop_epochs:
        lr *= 0.1
        for g in opt.param_groups:
            g['lr'] = lr

    train_loss, train_angle_errors = train_SO3(model, opt, dl_train, device, lossfunc, translation)
    val_loss,val_angle_errors = test_SO3(model, opt, dl_eval, device, lossfunc, translation)

    average_train_loss = (sum(train_loss) / len(train_loss))
    average_eval_loss = (sum(val_loss) / len(val_loss))
    average_train_angle_error = (sum(train_angle_errors)/len(train_angle_errors))
    average_eval_angle_error = (sum(val_angle_errors)/len(val_angle_errors))
    median_train_angle_error = np.median(train_angle_errors)
    median_eval_angle_error = np.median(val_angle_errors)
    epoch_time = (time.time() - epoch_time)

    writer_train.add_scalar('Loss/train', average_train_loss,e)
    writer_train.add_scalar('Loss/test', average_eval_loss,e)

    writer_train.add_scalar('MeanAngleError/train', average_train_angle_error, e)
    writer_train.add_scalar('MeanAngleError/test', average_eval_angle_error, e)
    writer_train.add_scalar('MedianAngleError/train', median_train_angle_error, e)
    writer_train.add_scalar('MedianAngleError/test', median_eval_angle_error, e)

    print(e, "/", epochs, ": ", "Training loss: ", average_train_loss,"train-MAE: ",average_train_angle_error,
         "Validation loss: ", average_eval_loss,"Val-MAE: ",average_eval_angle_error, "time: ",epoch_time)

    if(verbose):
        save_network(e, model, opt, model_name, MODEL_SAVE_PATH)

