import torch
import torch.nn as nn
import sys
import os
from os.path import join as pjoin
import numpy as np
import torch
import torch.functional as F
import torch.nn

"""
Most of the rotation representation functions are from https://github.com/papagina/RotationContinuity
    On the Continuity of Rotation Representations in Neural Networks
    Zhou et al. CVPR19
    https://zhouyisjtu.github.io/project_rotation/rotation.html
"""


def symmetric_orthogonalization(x):
    """Maps 9D input vectors onto SO(3) via symmetric orthogonalization.
    x: should have size [batch_size, 9]
    Output has size [batch_size, 3, 3], where each inner 3x3 matrix is in SO(3).

    This is from Levinson et Al.
    
    """
    m = x.view(-1, 3, 3)
    d = m.device
    u, s, v = torch.svd(m.cpu())
    u, v = u.to(d), v.to(d)
    vt = torch.transpose(v, 1, 2)
    det = torch.det(torch.bmm(u, vt))
    det = det.view(-1, 1, 1)
    vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
    r = torch.bmm(u, vt)
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

def normalize_vector( v):
    """
    Code from https://github.com/papagina/RotationContinuity
    On the Continuity of Rotation Representations in Neural Networks
    Zhou et al. CVPR19
    https://zhouyisjtu.github.io/project_rotation/rotation.html
    """
    batch=v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).to(v.device)))
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    return v
    
# u, v batch*n
def cross_product( u, v):
    """
    Code from https://github.com/papagina/RotationContinuity
    On the Continuity of Rotation Representations in Neural Networks
    Zhou et al. CVPR19
    https://zhouyisjtu.github.io/project_rotation/rotation.html
    """
    batch = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)#batch*3
        
    return out
        

def stereographic_unproject(a, axis=None):
    """
	Inverse of stereographic projection: increases dimension by one.
    Code from https://github.com/papagina/RotationContinuity
    On the Continuity of Rotation Representations in Neural Networks
    Zhou et al. CVPR19
    https://zhouyisjtu.github.io/project_rotation/rotation.html
    """
    batch=a.shape[0]
    if axis is None:
        axis = a.shape[1]
    s2 = torch.pow(a,2).sum(1) #batch
    ans = torch.autograd.Variable(torch.zeros(batch, a.shape[1]+1).cuda()) #batch*6
    unproj = 2*a/(s2+1).view(batch,1).repeat(1,a.shape[1]) #batch*5
    if(axis>0):
        ans[:,:axis] = unproj[:,:axis] #batch*(axis-0)
    ans[:,axis] = (s2-1)/(s2+1) #batch
    ans[:,axis+1:] = unproj[:,axis:]	 #batch*(5-axis)		# Note that this is a no-op if the default option (last axis) is used
    return ans

def compute_rotation_matrix_from_euler(euler):
    """
    Code from https://github.com/papagina/RotationContinuity
    On the Continuity of Rotation Representations in Neural Networks
    Zhou et al. CVPR19
    https://zhouyisjtu.github.io/project_rotation/rotation.html
    """
    batch=euler.shape[0]
        
    c1=torch.cos(euler[:,0]).view(batch,1)#batch*1 
    s1=torch.sin(euler[:,0]).view(batch,1)#batch*1 
    c2=torch.cos(euler[:,2]).view(batch,1)#batch*1 
    s2=torch.sin(euler[:,2]).view(batch,1)#batch*1 
    c3=torch.cos(euler[:,1]).view(batch,1)#batch*1 
    s3=torch.sin(euler[:,1]).view(batch,1)#batch*1 
        
    row1=torch.cat((c2*c3,          -s2,    c2*s3         ), 1).view(-1,1,3) #batch*1*3
    row2=torch.cat((c1*s2*c3+s1*s3, c1*c2,  c1*s2*s3-s1*c3), 1).view(-1,1,3) #batch*1*3
    row3=torch.cat((s1*s2*c3-c1*s3, s1*c2,  s1*s2*s3+c1*c3), 1).view(-1,1,3) #batch*1*3
        
    matrix = torch.cat((row1, row2, row3), 1) #batch*3*3
     
        
    return matrix

#a batch*5
#out batch*3*3

def compute_rotation_matrix_from_ortho5d(a):
    """
    Code from https://github.com/papagina/RotationContinuity
    On the Continuity of Rotation Representations in Neural Networks
    Zhou et al. CVPR19
    https://zhouyisjtu.github.io/project_rotation/rotation.html
    """
    batch = a.shape[0]
    proj_scale_np = np.array([np.sqrt(2)+1, np.sqrt(2)+1, np.sqrt(2)]) #3
    proj_scale = torch.autograd.Variable(torch.FloatTensor(proj_scale_np).cuda()).view(1,3).repeat(batch,1) #batch,3
    
    u = stereographic_unproject(a[:, 2:5] * proj_scale, axis=0)#batch*4
    norm = torch.sqrt(torch.pow(u[:,1:],2).sum(1)) #batch
    u = u/ norm.view(batch,1).repeat(1,u.shape[1]) #batch*4
    b = torch.cat((a[:,0:2], u),1)#batch*6
    matrix = compute_rotation_matrix_from_ortho6d(b)
    return matrix


def compute_rotation_matrix_from_quaternion(quaternion):
    """
    Code from https://github.com/papagina/RotationContinuity
    On the Continuity of Rotation Representations in Neural Networks
    Zhou et al. CVPR19
    https://zhouyisjtu.github.io/project_rotation/rotation.html
    """
    batch=quaternion.shape[0]
    
    quat = normalize_vector(quaternion)
    
    qw = quat[...,0].view(batch, 1)
    qx = quat[...,1].view(batch, 1)
    qy = quat[...,2].view(batch, 1)
    qz = quat[...,3].view(batch, 1)

    # Unit quaternion rotation matrices computatation  
    xx = qx*qx
    yy = qy*qy
    zz = qz*qz
    xy = qx*qy
    xz = qx*qz
    yz = qy*qz
    xw = qx*qw
    yw = qy*qw
    zw = qz*qw
    
    row0 = torch.cat((1-2*yy-2*zz, 2*xy - 2*zw, 2*xz + 2*yw), 1) #batch*3
    row1 = torch.cat((2*xy+ 2*zw,  1-2*xx-2*zz, 2*yz-2*xw  ), 1) #batch*3
    row2 = torch.cat((2*xz-2*yw,   2*yz+2*xw,   1-2*xx-2*yy), 1) #batch*3
    
    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch,1,3), row2.view(batch,1,3)),1) #batch*3*3
    
    return matrix    

class Model(nn.Module):
    def __init__(self, representation="SVD"):
        super(Model, self).__init__()
        self.representation = representation
        self.dimension = {"SVD":9, "6D":6, "5D": 5, "Quat": 4, "Euler": 3, "Direct": 9}
        self.func = {"SVD":symmetric_orthogonalization, "6D":compute_rotation_matrix_from_ortho6d, "5D": compute_rotation_matrix_from_ortho5d, "Quat": compute_rotation_matrix_from_quaternion, "Euler": compute_rotation_matrix_from_euler}
        self.out_channel = self.dimension[representation]

        self.fc1 = nn.Linear(9, 128)  # 5*5 from image dimension
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, self.out_channel)




    # pt b*point_num*3

    def forward(self, input):
        
        x = torch.nn.functional.relu(self.fc1(input))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)

        if(self.representation == "Direct"):
            out = x.view(-1,3,3)
        else:
            try:
                out = self.func[self.representation](x)
            except Exception as err:
                print("Rotation could not be found")
            
        return out


  