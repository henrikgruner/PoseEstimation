import numpy as np
import torch
import os
from os.path import join as pjoin
import trimesh
import argparse
import sys
import tqdm 
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0,pjoin(BASEPATH, '../..'))

def normalize_vector( v):
    batch=v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).to(v.device)))
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    return v


def get_sampled_rotation_matrices_by_axisAngle( batch):
    
    theta = torch.autograd.Variable(torch.FloatTensor(np.random.uniform(-1,1, batch)*np.pi).cuda()) #[0, pi] #[-180, 180]
    sin = torch.sin(theta)
    axis = torch.autograd.Variable(torch.randn(batch, 3).cuda())
    axis = normalize_vector(axis) #batch*3
    qw = torch.cos(theta)
    qx = axis[:,0]*sin
    qy = axis[:,1]*sin
    qz = axis[:,2]*sin
    
    # Unit quaternion rotation matrices computatation  
    xx = (qx*qx).view(batch,1)
    yy = (qy*qy).view(batch,1)
    zz = (qz*qz).view(batch,1)
    xy = (qx*qy).view(batch,1)
    xz = (qx*qz).view(batch,1)
    yz = (qy*qz).view(batch,1)
    xw = (qx*qw).view(batch,1)
    yw = (qy*qw).view(batch,1)
    zw = (qz*qw).view(batch,1)
    
    row0 = torch.cat((1-2*yy-2*zz, 2*xy - 2*zw, 2*xz + 2*yw), 1) #batch*3
    row1 = torch.cat((2*xy+ 2*zw,  1-2*xx-2*zz, 2*yz-2*xw  ), 1) #batch*3
    row2 = torch.cat((2*xz-2*yw,   2*yz+2*xw,   1-2*xx-2*yy), 1) #batch*3
    
    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch,1,3), row2.view(batch,1,3)),1) #batch*3*3
    
    return matrix 

def pc_normalize(pc):
    centroid = (np.max(pc, axis=0) + np.min(pc, axis=0)) /2
    pc = pc - centroid
    scale = np.linalg.norm(np.max(pc, axis=0) - np.min(pc, axis=0))
    pc = pc / scale
    return pc, centroid, scale

if __name__ == "__main__": 
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-d", "--data_dir", type=str, default='dataset/modelnet40_manually_aligned', help="Path to modelnet dataset")
    arg_parser.add_argument("-c", "--category", type=str, default='chair', help="category")
    arg_parser.add_argument("-f", "--fix_test", action='store_false', help="for fair comparision")
    args = arg_parser.parse_args()

    sample_num = 4096
    for mode in ['train', 'test']:
        in_folder = pjoin(args.data_dir, args.category, mode)
        out_folder = pjoin(args.data_dir, args.category, mode + '_pc')
        os.makedirs(out_folder, exist_ok=True)


        lst = [i for i in os.listdir(in_folder) if i[-4:] == '.off']
        lst.sort()
        for p in tqdm.tqdm(lst):
            in_path = pjoin(in_folder, p)
            out_path = pjoin(out_folder, p.replace('.off','.pts'))
            if os.path.exists(out_path) and mode == 'train':
                continue
            mesh = trimesh.load(in_path, force='mesh')
            pc, _ = trimesh.sample.sample_surface(mesh, sample_num)
            pc = np.array(pc)
            pc, centroid, scale = pc_normalize(pc) 
            np.savetxt(out_path, pc)
            
            if mode == 'test' and args.fix_test:
                fix_folder = pjoin(args.data_dir, args.category, mode + '_fix')
                os.makedirs(fix_folder, exist_ok=True)
                fix_path = pjoin(fix_folder, p.replace('.off','.pt'))
                pc = np.random.permutation(pc)[:1024,:]
                #each instance sample 10 rotations for test
                rgt = get_sampled_rotation_matrices_by_axisAngle(10).cpu()
                pc = torch.bmm(rgt, torch.Tensor(pc).unsqueeze(0).repeat(10,1,1).transpose(2,1))
                data_dict = {'pc':pc.transpose(1,2), 'rgt':rgt,'centroid':centroid, 'scale':scale}
                torch.save(data_dict, fix_path)