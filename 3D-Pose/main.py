import sys
sys.path.append('../models')
sys.path.append('dataset/')
sys.path.append('configs/')
sys.path.append('..')
sys.path.append('logs')
sys.path.append('../CosyPose')
from model import ResnetRS
from loss import loss_frobenius, rotate_by_180
import torch
import time
from config import config_parser, get_new_dir,cuda_confirm
import torch.nn as nn
import torch.nn.functional as F
import ModelNetSO3
import os
from helper import save_network, load_network
from model import ResnetRS
import numpy as np
import glob
from dataset import get_modelnet_loaders
from torch.utils.tensorboard import SummaryWriter
from rotation_representation import *
import torchvision
import argparse
from torch import Tensor 
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-c', type=str, default="example.yaml",
                    help='config, default = resnet')
arg_parser.add_argument('-dir', type=str, default="configs/",
                    help='config, default = resnet')

args = arg_parser.parse_args()




def train_SO3(model, opt, dl_train, device, lossfunc = 'Frobenius', rot_rep = "SVD"):
    '''
        input:
        model : network
        opt : optimizer
        criterion : loss function
        dl_train : dataloader with training data
    '''
    func = {"SVD": symmetric_orthogonalization, "6D":compute_rotation_matrix_from_ortho6d, "5D": compute_rotation_matrix_from_ortho5d, "quat": compute_rotation_matrix_from_quaternion}
    model.train()
    epoch_loss = []
    angle_errors = []
    for index, (img, ex, _, _, _) in enumerate(dl_train):
        opt.zero_grad()
        img = img.to(device)
        # rotation matrix
        R = ex[:, :3, :3].to(device)

        # preditc rot matrix

        out = model(img)

        out = func[rot_rep](out)

        angle = angle_error(out, R).mean().item()
        '''
        if(angle > 140):
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
            out = out_out[index_min]
            angle = angles[index_min]
        '''
        loss = loss_frobenius(R, out)
        angle_errors.append(angle)

        epoch_loss.append(loss.item())

        loss.backward()
        opt.step()

    return epoch_loss, angle_errors


def test_SO3(model, opt, dl_eval, device, lossfunc = 'Frobenius', rot_rep = "SVD"):
    '''
    Either for validation or for testing
    '''

    func = {"SVD": symmetric_orthogonalization, "6D":compute_rotation_matrix_from_ortho6d, "5D": compute_rotation_matrix_from_ortho5d, "quat": compute_rotation_matrix_from_quaternion}
    model.eval()
    with torch.no_grad():
        val_loss = []
        angle_errors = []
        for img, ex, _, _, _ in dl_eval:

            img = img.to(device)
            # rotation matrix
            R = ex[:, :3, :3].to(device)

            # preditc rot matrix

            out = model(img)
            #out = compute_rotation_matrix_from_ortho6d(out)
            out = func[rot_rep](out)


            angle = angle_error(out, R).mean().item()
            angle_errors.append(angle)

            
            if(lossfunc == 'Frobenius'):
                loss = loss_frobenius(R, out)
            elif(lossfunc == 'Geodesic'):
                raise NotImplementedError
            else:
                raise NotImplementedError
            
            val_loss.append(loss.item())


    return val_loss, angle_errors




# Frobenius norm
# Brief setup

lr, batch_size, opt_name, model_name, classes, rot_rep, rot_dim, epochs, drop_epochs, lossfunc, num_classes, dataset_dir, resume, save_interval, schedule = config_parser(args)


SAVE_PATH, MODEL_SAVE_PATH = get_new_dir(rot_rep)

device, devices = cuda_confirm()


dl_train, dl_eval = get_modelnet_loader(batch_size, True , dataset_dir = dataset_dir)

model = ResnetRS.create_pretrained(
    model_name, in_ch=3,out_features = rot_dim, num_classes=rot_dim)

model = nn.DataParallel(model, device_ids=devices)
model = model.to(device)
opt = torch.optim.SGD(model.parameters(), lr=lr)
curr_epoch = 0

if(resume):
    LOAD_PATH, curr_epoch = resume_training(model_name, args)
    model, opt, epoch = load_network(
    LOAD_PATH, model, opt, model_name, rot_dim, num_classes)

    # Change learning rate if applicable
    for g in opt.param_groups:
        g['lr'] = lr

    print("Resuming training from epoch: ", curr_epoch)


writer_train = SummaryWriter(
    log_dir=os.path.join(SAVE_PATH, 'train'), comment=f"_{model_name}_{opt.__class__.__name__}_{lr}_train")



for e in range(curr_epoch, epochs):
    verbose = e % int(save_interval) == 0 or e == (epochs - 1)

    epoch_time = time.time()

    if e in drop_epochs:
        lr *= 0.1
        for g in opt.param_groups:
            g['lr'] = lr

    train_loss, train_angle_errors = train_SO3(model, opt, dl_train, device, lossfunc)
    val_loss,val_angle_errors = test_SO3(model, opt, dl_eval, device, lossfunc)

    average_train_loss = (sum(train_loss) / len(train_loss))
    average_eval_loss = (sum(val_loss) / len(val_loss))
    average_train_angle_error = (sum(train_angle_errors)/len(train_angle_errors))
    average_eval_angle_error = (sum(val_angle_errors)/len(val_angle_errors))
    epoch_time = (time.time() - epoch_time)

    writer_train.add_scalar('Loss/train', average_train_loss,e)
    writer_train.add_scalar('Loss/test', average_eval_loss,e)

    writer_train.add_scalar('MeanAngleError/train', average_train_angle_error, e)
    writer_train.add_scalar('MeanAngleError/test', average_eval_angle_error, e)

    print(e, "/", epochs, ": ", "Training loss: ", average_train_loss,"train-MAE: ",average_train_angle_error,
         "Validation loss: ", average_eval_loss,"Val-MAE: ",average_eval_angle_error, "time: ",epoch_time)

    if(verbose):
        save_network(e, model, opt, model_name, MODEL_SAVE_PATH)

