import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender
import sys

sys.path.append('../data')
sys.path.append('../Fresh')
import time
import threading
import torchvision
import multichannel_resnet
from multichannel_resnet import get_arch as Resnet
import random
import trimesh as tm
from utility import *
import torch.nn as nn
import pandas as pd
from dataset import get_6D_loader, get_6D_eval_loader
import torch
from render_utility import *
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('-start',type = int, default=0)
parser.add_argument('-end', type = int, default=0)
args = parser.parse_args()
## MÅL
# rendre bildene i omgang -> en klasse.
def load_network(path, model, opt, model_name, out_dim, numclasses):
    modelcheckpoint = torch.load(path)

    model.load_state_dict(modelcheckpoint['model_state_dict'])
    opt.load_state_dict(modelcheckpoint['optimizer_state_dict'])
    epoch = modelcheckpoint['epoch']

    return model, opt, epoch

def generate(model, dl_train, start, end):
    dataset_path = 'dataset_iteration_2'
    model.train()
    epoch_loss = []
    model.eval()
    imgs, depths, TCOs, init_imgs, depths_init, TCO_inits, ids, vertices, TCO_new, rendered_img, cids = [
            ], [], [], [], [], [], [], [], [], [], []

    mapping = {'bathtub': 0, 'chair': 1, 'sofa': 2, 'toilet': 3, 'airplane': 4}

    with torch.no_grad():
        starttime = time.time()
        for index, (images, curr_images, depth, verts, ex, ex_curr, class_id, cad_id) in enumerate(dl_train, start):               

            if(index == 800):
                print('time:', time.time()-starttime)
                df = pd.DataFrame()
                df['Images'] = imgs
                df['Init_Images'] = init_imgs
                df['Init_depth'] = depths_init
                df['Rendered_img'] = rendered_img
                df['Vertices'] = vertices
                df['Extrinsic'] = TCOs
                df['Extrinsic_rendered'] = TCO_new
                df['Class'] = cids
                df['Extrinsic_init'] = TCO_inits
                df['Cad_id'] = ids
                df.to_pickle('dataset_iteration_2/homo.pkl')
                exit()


            ex = ex.to(device)
            verts = verts.to(device)
            gt_img = images.to(device, dtype=torch.float)
            curr_images = curr_images.to(device, dtype=torch.float)
            depth = depth.to(device)

            model_input = torch.cat(
                [depth.unsqueeze(dim=1), curr_images, gt_img], dim=1)

            out = model(model_input)

            ex_curr_new = calculate_T_CO_pred(
                out, ex_curr.to(device), rot_repr='SVD')

            rendered_images = render_to_batch(cad_id.cpu(), class_id, ex_curr.detach().cpu().numpy(), train = False, img_res = 320)

            # Pass the inputs to the model

            imgs.append(images.detach().cpu().numpy()[0])
            init_imgs.append(curr_images.detach().cpu().numpy()[0])
            depths_init.append(depth.detach().cpu().numpy()[0])
            TCOs.append(ex.detach().cpu().numpy()[0])
            vertices.append(verts.detach().cpu().numpy()[0])
            TCO_inits.append(ex_curr.detach().cpu().numpy()[0])
            TCO_new.append(ex_curr_new.detach().cpu().numpy()[0])
            ids.append(cad_id.item())
            cids.append(class_id.item())
            rendered_img.append(rendered_images.detach().cpu().numpy()[0])


       



if __name__ == '__main__':
    rot_rep = 'SVD'
    rot_dim = 9

    num_classes = 1
    batch_size = 1

    dataset_dir = 'data/'
    dataset = ''
    epochs = 300
    drop_epochs = []
    save_interval = 1
    model_name = 'resnetrs101'
    ngpu = 2
    lr = 0.05


    curr_epoch = 199

    # create new directory

    PATH = 'saved_models'

    device = torch.device("cuda" if(
        torch.cuda.is_available() and ngpu > 0) else "cpu")
    devices = [d for d in range(torch.cuda.device_count())]
    device_names = [torch.cuda.get_device_name(d) for d in devices]

    print("cuda: ", torch.cuda.is_available())
    print("count: ", torch.cuda.device_count())
    print("names: ", device_names)
    print("device ids:", devices)

    classes = ['chair']

    dl_train, dl_eval = get_6D_loader(
        dataset, batch_size, dataset_dir, classes, pre=True, RGBD=False, shuffle = False)

    resnet101_7_channel = Resnet(101, 7)


    model = resnet101_7_channel(True)
    model = nn.DataParallel(model, device_ids=devices)
    model = model.to(device)

    opt = torch.optim.SGD(model.parameters(), lr=lr)

    resume = True
    curr_epoch = 199
    LOAD_PATH = 'logs/run014/saved_models/resnetrs101_state_dict_199.pkl'
    if(resume):
        NAME = str(model_name) + '_state_dict_{}.pkl'.format(curr_epoch)
        LOAD_PATH = os.path.join(LOAD_PATH, NAME)
        model, opt, epoch = load_network(
            'logs/run014/saved_models/resnetrs101_state_dict_199.pkl', model, opt, model_name, rot_dim, num_classes)
        for g in opt.param_groups:
            g['lr'] = lr
    print(args.start, args.end)
    generate(model, dl_eval, args.start, args.end)