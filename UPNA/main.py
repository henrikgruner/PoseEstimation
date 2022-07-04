import sys
sys.path.append('../models')
sys.path.append('../data')
sys.path.append('..')
sys.path.append('logs')
sys.path.append('../6D')
sys.path.append('../Fresh')

from resnet import resnet50, resnet101, ResnetHead
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
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch import Tensor
from dataloader import get_upna_loaders
from rotation_representation import *




def train_SO3(model, opt, dl_train, device, lossfunc=None):
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
    for index, (img, ex, _, _, _, _) in enumerate(dl_train):
        opt.zero_grad()
        img = img.to(device)
        # rotation matrix
        R = ex[:, :3, :3].to(device)

        # predict rot matrix
        out = model(img)
        #out = compute_rotation_matrix_from_ortho6d(out)
        out = symmetric_orthogonalization(out)

        if(lossfunc is None):
            loss = loss_frobenius(R, out)
        else:
            loss = lossfunc(R, out)

        epoch_loss.append(loss.item())

        loss.backward()

        opt.step()

        angle = angle_error(out.detach(), R).mean().item()
        angle_errors.append(angle)

    return epoch_loss, angle_errors


def test_SO3(model, opt, dl_eval, device, lossfunc=None):
    '''
    Either for validation or for testing

    '''

    model.eval()
    with torch.no_grad():
        val_loss = []
        angle_errors = []
        for img, ex, _, _, _, _ in dl_eval:

            img = img.to(device)
            # rotation matrix
            R = ex[:, :3, :3].to(device)

            # preditc rot matrix

            out = model(img)
            #out = compute_rotation_matrix_from_ortho6d(out)
            out = symmetric_orthogonalization(out)

            angle = angle_error(out, R).mean().item()

            angle_errors.append(angle)

            if(lossfunc is None):
                loss = loss_frobenius(R, out)
            else:
                loss = lossfunc(R, out)

            val_loss.append(loss.item())
            angle_errors.append(angle_error(out, R).mean().item())

    return val_loss, angle_errors


def load_network(path, model, opt, model_name, out_dim, numclasses):
    modelcheckpoint = torch.load(path)

    model.load_state_dict(modelcheckpoint['model_state_dict'])
    opt.load_state_dict(modelcheckpoint['optimizer_state_dict'])
    epoch = modelcheckpoint['epoch']

    return model, opt, epoch

# Frobenius norm


# Brief setup
rot_rep = 'SVD'
rot_dim = 6
num_classes = 1
batch_size = 128
epochs = 50
drop_epochs = []
save_interval = 1
model_name = 'resnetrs101'
ngpu = 4
lr = 0.01
DIR = 'logs'

lossfunc = None
# automatically find the latest run folder
n = len(glob.glob(DIR + '/run*'))
NEW_DIR = 'run' + str(n + 1).zfill(3)

SAVE_PATH = os.path.join(DIR, NEW_DIR)
# create new directory

PATH = 'saved_models'
MODEL_SAVE_PATH = os.path.join(SAVE_PATH, PATH)
if not os.path.isdir(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

device = torch.device("cuda" if(
    torch.cuda.is_available() and ngpu > 0) else "cpu")
devices = [d for d in range(torch.cuda.device_count())]
device_names = [torch.cuda.get_device_name(d) for d in devices]

print("cuda: ", torch.cuda.is_available())
print("count: ", torch.cuda.device_count())
print("names: ", device_names)
print("device ids:", devices)
classes = ['bathtub', 'table', 'toilet']

dl_train, dl_eval = get_upna_loaders(batch_size, True, '')


model = ResnetRS.create_pretrained(
    model_name, in_ch=3, num_classes=num_classes)

'''
base = resnet101(pretrained=True, progress=True)

num_classes = 1
model = ResnetHead(base, num_classes, 0, 512, 9)
'''

model = nn.DataParallel(model, device_ids=devices)
model = model.to(device)

opt = torch.optim.SGD(model.parameters(), lr=lr)

curr_epoch = 0
resume = False

LOAD_PATH = ''
if(resume):
    NAME = str(model_name) + '_state_dict_{}.pkl'.format(curr_epoch)
    LOAD_PATH = os.path.join(LOAD_PATH, NAME)
    model, opt, epoch = load_network(
        LOAD_PATH, model, opt, model_name, rot_dim, num_classes)
    for g in opt.param_groups:
        g['lr'] = lr
    print("Resuming training from epoch: ", epoch)


writer_train = SummaryWriter(
    log_dir=os.path.join(SAVE_PATH, 'train'), comment=f"_{model_name}_{opt.__class__.__name__}_{lr}_train")


for e in range(curr_epoch, epochs):
    verbose = e % int(save_interval) == 0 or e == (epochs - 1)

    epoch_time = time.time()

    if e in drop_epochs:
        lr *= 0.1
        for g in opt.param_groups:
            g['lr'] = lr

    train_loss, train_angle_errors = train_SO3(
        model, opt, dl_train, device, lossfunc)
    val_loss, val_angle_errors = test_SO3(
        model, opt, dl_eval, device, lossfunc)

    average_train_loss = (sum(train_loss) / len(train_loss))
    average_eval_loss = (sum(val_loss) / len(val_loss))
    average_train_angle_error = (
        sum(train_angle_errors) / len(train_angle_errors))
    average_eval_angle_error = (sum(val_angle_errors) / len(val_angle_errors))
    median_train_angle_error = np.median(train_angle_errors)
    median_eval_angle_error = np.median(val_angle_errors)
    epoch_time = (time.time() - epoch_time)

    writer_train.add_scalar('Loss/train', average_train_loss, e)
    writer_train.add_scalar('Loss/test', average_eval_loss, e)

    writer_train.add_scalar('MeanAngleError/train',
                            average_train_angle_error, e)
    writer_train.add_scalar('MeanAngleError/test', average_eval_angle_error, e)
    writer_train.add_scalar('MedianAngleError/train',
                            median_train_angle_error, e)
    writer_train.add_scalar('MedianAngleError/test',
                            median_eval_angle_error, e)

    print(e, "/", epochs, ": ", "Training loss: ", average_train_loss, "train-MAE: ", average_train_angle_error, "train-MED: ", median_train_angle_error,
          "Validation loss: ", average_eval_loss, "Val-MAE: ", average_eval_angle_error, "train-MED: ", median_eval_angle_error, "time: ", epoch_time)

    if(verbose):
        save_network(e, model, opt, model_name, MODEL_SAVE_PATH)
