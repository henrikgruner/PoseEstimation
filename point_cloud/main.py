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
import argparse
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-c", "--category", type=str, default='chair', help="category")
arg_parser.add_argument("-r", "--rot", type=str, default='6D', help="category")
args = arg_parser.parse_args()

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

def compute_geodesic_distance_from_two_matrices(m1, m2):
    batch=m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3
    
    cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()) )
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda())*-1 )
    
    
    theta = torch.acos(cos)
    
    #theta = torch.min(theta, 2*np.pi - theta)
    
    
    return theta



def geodesic(R1, R2, reduction = "mean"):
    eps = 1e-7
    R_diffs = R1 @ R2.permute(0, 2, 1)
     # See: https://github.com/pytorch/pytorch/issues/7500#issuecomment-502122839.
    traces = R_diffs.diagonal(dim1=-2, dim2=-1).sum(-1)
    dists = torch.acos(torch.clamp(
        (traces - 1) / 2, -1 + eps, 1 - eps))
    if reduction == "none":
        return dists
    elif reduction == "mean":
        return dists.mean()
    elif reduction == "sum":
        return dists.sum()


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


def loss_frobenius(R_pred, R_true):
    difference = R_true - R_pred
    frob_norm = torch.linalg.matrix_norm(difference, ord='fro')
    return frob_norm.mean()

category = args.category 
print(category)
data_folder = 'modelnet40_manually_aligned/'+category
train_folder = os.path.join(data_folder, 'train_pc')
val_folder = os.path.join(data_folder, 'test_fix')
train_dataset = ModelNetDataset(train_folder, sample_num=1024)


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


def compute_rotation_matrix_from_ortho6d(poses):
    """
    Code from https://github.com/papagina/RotationContinuity
    On the Continuity of Rotation Representations in Neural Networks
    Zhou et al. CVPR19
    https://zhouyisjtu.github.io/project_rotation/rotation.html
    """
    assert poses.shape[-1] == 6
    x_raw = poses[..., 0:3]
    y_raw = poses[..., 3:6]
    x = x_raw / torch.norm(x_raw, p=2, dim=-1, keepdim=True)
    z = torch.cross(x, y_raw, dim=-1)
    z = z / torch.norm(z, p=2, dim=-1, keepdim=True)
    y = torch.cross(z, x, dim=-1)
    matrix = torch.stack((x, y, z), -1)
    return matrix


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


def train(model, train_loader, device, batch):
    angle_errors = []
    epoch_loss = []
    geos = []
    model.train()

    for pc in train_loader:
        opt.zero_grad()

        batch = pc.shape[0]

        point_num = 1024

        pc1 = torch.autograd.Variable(pc.float().to(device))  # num*3

        gt_rmat = get_sampled_rotation_matrices_by_axisAngle(
            batch)  # batch*3*3

        gt_rmats = gt_rmat.contiguous().view(batch, 1, 3, 3).expand(
            batch, point_num, 3, 3).contiguous().view(-1, 3, 3)

        # (batch*point_num)*3*1
        pc2 = torch.bmm(gt_rmats, pc1.view(-1, 3, 1))
        pc_out = pc2.view(batch, point_num, 3)  # batch,p_num,3

        gg = pc_out.transpose(1, 2)

        model_out = model(gg)

        angle = angle_error(model_out, gt_rmat).mean().item()
        angle_errors.append(angle)
        geo = compute_geodesic_distance_from_two_matrices(model_out, gt_rmat).detach().to('cpu').numpy()
        geos.append(geo)

        # out = compute_rotation_matrix_from_ortho6d(out)

        if(lossfunc is None):
            loss = loss_frobenius(gt_rmat, model_out)
        else:
            loss = lossfunc(gt_rmat, model_out)

        epoch_loss.append(loss.item())
        loss.backward()
        opt.step()

    return epoch_loss, angle_errors, geo


def test(model, test_folder, device, batch):
    angle_errors = []
    val_loss = []
    geos = []
    model.eval()
    test_path_list = [os.path.join(test_folder, i)
                      for i in os.listdir(test_folder)]

    with torch.no_grad():
        for path in test_path_list:
            tmp = torch.load(path)
            pc2 = tmp['pc'].cpu().cuda()
            gt_rmat = tmp['rgt'].cpu().cuda()

            out = model(pc2.transpose(1, 2))

            angle = angle_error(out, gt_rmat).mean().item()
            angle_errors.append(angle)
            geo = compute_geodesic_distance_from_two_matrices(out, gt_rmat).detach().to('cpu').numpy()
            geos.append(geo)

            if(lossfunc is None):
                loss = loss_frobenius(gt_rmat, out)
            else:
                loss = lossfunc(gt_rmat, out)

            val_loss.append(loss.item())

    return val_loss, angle_errors, geo


def cuda_confirm():
    bz = {16: 32, 32: 64, 39: 32, 40: 32}
    print("cuda: ", torch.cuda.is_available())
    print("count: ", torch.cuda.device_count())
    devices = [d for d in range(torch.cuda.device_count())]
    device_names = [torch.cuda.get_device_name(d) for d in devices]
    mem = []
    for d in devices:
        mem.append(round(torch.cuda.get_device_properties(
            d).total_memory / (1024 * 1024 * 1024)))
    print("Maximum batch size in power of two is :", bz[min(mem)])
    print("names: ", device_names)
    print("device ids:", devices)

    return devices, bz[min(mem)]

# Brief setup

rot_dim = 6
num_classes = 1
batch_size = 128
epochs = 250
drop_epochs = []
save_interval = 5
model_name = 'resnetrs101'
ngpu = 4
lr = 0.01
DIR = 'logs'
ngpu = 4
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





model = Model(args.rot)
device = torch.device("cuda" if(
    torch.cuda.is_available() and ngpu > 0) else "cpu")
devices, max_batch_size_per_gpu = cuda_confirm()



model = nn.DataParallel(model, device_ids=devices)
model = model.to(device)
# model = nn.DataParallel(model, device_ids=devices)


dl_train = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)


opt = torch.optim.SGD(model.parameters(), lr=lr)

curr_epoch = 0
resume = False

LOAD_PATH = ''
if(resume):
    NAME = str(model_name) + '_state_dict_{}.pkl'.format(curr_epoch)
    LOAD_PATH = os.path.join(LOAD_PATH, NAME)
    model, opt, epoch = load_network(
        'logs/run002/saved_models/resnetrs101_state_dict_49.pkl', model, opt, model_name, rot_dim, num_classes)
    for g in opt.param_groups:
        g['lr'] = lr
    print("Resuming training from epoch: ", epoch)


writer_train = SummaryWriter(
    log_dir=os.path.join(SAVE_PATH, 'train'), comment=f"_{model_name}_{opt.__class__.__name__}_{lr}_train")

for e in range(epochs):
    verbose = e % int(save_interval) == 0 or e == (epochs - 1)

    epoch_time = time.time()

    if e in [10,50, 80, 120]:
        lr *= 0.1
        for g in opt.param_groups:
            g['lr'] *= 0.7

    train_loss, train_angle_errors, geodesic_train = train(
        model, dl_train, device, batch_size)
    val_loss, val_angle_errors, geodesic_test = test(
        model, val_folder, device, batch_size)
    # test(model, dl_eval, device, batch_size)
    average_train_loss = round(np.mean(train_loss),1)
    average_eval_loss = round(np.mean(val_loss),1)
    average_train_angle_error = round(np.mean(train_angle_errors),1)
    average_eval_angle_error = round(np.mean(val_angle_errors),1)

    median_train_geos = round(np.median(geodesic_train),3)
    median_eval_geos = round(np.median(geodesic_test),3)
    average_train_geos = round(np.mean(geodesic_train),3)
    average_eval_geos = round(np.mean(geodesic_test),3)
    epoch_time = (time.time() - epoch_time)

    writer_train.add_scalar('Loss/train', average_train_loss, e)
    writer_train.add_scalar('Loss/test', average_eval_loss, e)

    writer_train.add_scalar('GeoAngleError/train',
                            average_train_geos, e)
    writer_train.add_scalar('GeoAngleError/test', average_eval_geos, e)
    writer_train.add_scalar('GeoAngleError/train',
                            median_train_geos, e)
    writer_train.add_scalar('GeoAngleError/test',
                            median_eval_geos, e)

    print(e, "/", epochs, ": ", "Training loss: ", average_train_loss, "train-MAE: ", average_train_angle_error, "Median-Geo: ", median_train_geos, "geo-train", average_train_geos,
          "Validation loss: ", average_eval_loss, "Val-MAE: ", average_eval_angle_error, "Median-geo: ", median_eval_geos, "geo",average_eval_geos, "time: ", epoch_time)

    if(verbose and e > 200):
        save_network(e, model, opt, model_name, MODEL_SAVE_PATH)
