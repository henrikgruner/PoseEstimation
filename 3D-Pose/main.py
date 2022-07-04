import sys
sys.path.append('../models')
sys.path.append('../data')
sys.path.append('..')
sys.path.append('logs')
sys.path.append('../CosyPose')
from model import ResnetRS
from loss import loss_frobenius, rotate_by_180
import torch
import time
from dataloader import get_modelnet_loader
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
arg_parser.add_argument("-r", "--rot_rep", type=str, default='SVD', help="category")

args = arg_parser.parse_args()

def compute_geodesic_distance_from_two_matrices(m1, m2):
    '''
    Taken from:
    https://github.com/airalcorn2/pytorch-geodesic-loss/blob/master/geodesic_loss.py
    '''
    batch=m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3
    
    cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()) )
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda())*-1 )
    
    
    theta = torch.acos(cos)
    
    #theta = torch.min(theta, 2*np.pi - theta)
    
    
    return theta


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






def train_SO3(model, opt, dl_train, device, lossfunc = None, rot_rep = "SVD"):
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


def test_SO3(model, opt, dl_eval, device, lossfunc = None, rot_rep = "SVD"):
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

            
            if(lossfunc is None):
                loss = loss_frobenius(R, out)
            else:
                loss = lossfunc(R,out)
                
            val_loss.append(loss.item())


    return val_loss, angle_errors




# Frobenius norm
# Brief setup

rot_rep = args.rot_rep
batch_size = 512
dataset = 'SO3'
dataset_dir = 'datasetSO3/'
rot_dim = {"SVD":9, "5D": 5, "6D": 6,"quat": 4}
epochs = 50
drop_epochs = []
save_interval = 1
model_name = 'resnetrs101'
ngpu = 4
lr = 0.05
#lossfunc =GeodesicLoss()
lossfunc = None
DIR = rot_rep + '/logs'
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


dl_train, dl_eval = get_modelnet_loader(batch_size,True, dataset_dir = '../data/')

model = ResnetRS.create_pretrained(
    model_name, in_ch=3,out_features = rot_dim[rot_rep], num_classes=rot_dim[rot_rep],)

model = nn.DataParallel(model, device_ids=devices)
model = model.to(device)
opt = torch.optim.SGD(model.parameters(), lr=lr)

curr_epoch = 47
resume = False
LOAD_PATH = '/logs/run026/saved_models'
if(resume):
    NAME = str(model_name) + '_state_dict_{}.pkl'.format(curr_epoch)
    LOAD_PATH = os.path.join(LOAD_PATH, NAME)
    model, opt, epoch = load_network(
    '../Fresh/logs/run026/saved_models/resnetrs101_state_dict_47.pkl', model, opt, model_name, rot_dim, num_classes)
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

