import sys
sys.path.append('../models')
sys.path.append('../modelnet')
sys.path.append('../data')
sys.path.append('..')
from render_utility import *
import time as time
import random
import numpy as np
import yaml
import pandas as pd
import torch
import torchvision
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import argparse
parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-name', type=int, default=1,
                    help='Number of gpus for CUDA, default = 2')

args = parser.parse_args()

'''
start = time.time()
path = 'ModelNet40-norm-ply/airplane/train/airplane_0001.ply'
config = os.path.join('configs', 'config_1.yaml')
img, depth, TCO,, verts = render_img(path, config)
fuck it - 10 rendering per bilde?
Dataset -> ønsker
img, depth (org?), TCO, TCO_init, meshnavn(test), og verts
'''


def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


if __name__ == '__main__':

    config_path = os.path.join('configs', 'config_1.yaml')
    config = read_yaml(config_path)
    path = ''
    dataset = 'ModelNet40-norm-ply'
    new_dataset = '6D-dataset-2'

    path = os.path.join(path, dataset)


    mapping = {'bathtub': 0, 'chair': 1, 'sofa': 2, 'toilet': 3, 'airplane': 4}
    classes = ['toilet']
    tt = ['train', 'test']

    for t in tt:
        for c in classes:
            imgs, depths, TCOs, init_imgs, depths_init, TCO_inits, ids, vertices = [
            ], [], [], [], [], [], [], []
            if not c.startswith("."):
                read_dir = os.path.join(path, c, t)
                read_files = [os.path.join(read_dir, filename)
                              for filename in os.listdir(read_dir)]
                for r in read_files:
                    for _ in range(10):
                        cad_id = int(
                            r.split('/')[-1].split('.')[0].split('_')[-1])

                        img, depth, TCO, verts, TCO_init, img_init, depth_init = render_img(
                            r, config_path)
                        imgs.append(img)
                        depths.append(depth)
                        TCOs.append(TCO)
                        vertices.append(verts)
                        init_imgs.append(img_init)
                        depths_init.append(depth_init)
                        TCO_inits.append(TCO_init)
                        ids.append(cad_id)

                df = pd.DataFrame()
                df['Images'] = imgs
                df['Init_Images'] = init_imgs
                df['Init_depth'] = depths_init
                df['Vertices'] = vertices
                df['Extrinsic'] = TCOs
                df['Depth'] = depths
                df['Class'] = mapping[c]
                df['Extrinsic_init'] = TCO_inits
                df['Cad_id'] = ids
                df.to_pickle('dataset/' +str(args.name)+ str(c) + '_' + t + '.pkl')
