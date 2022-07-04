import torch
import torch.nn as nn
import sys
import os
from os.path import join as pjoin
import numpy as np
import torch
import torch.functional as F
import torch.nn
sys.path.append("../")
from rotation_representation import *


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


  