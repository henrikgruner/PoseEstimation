import numpy as np
from pyquaternion import Quaternion
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader

def getR(x,y,z):

   
    cx = np.cos(x)
    cy = np.cos(y)
    cz = np.cos(z)
    sx = np.sin(x)
    sy = np.sin(y)
    sz = np.sin(z)
    R = np.array([[cy*cz, -sy, cy*sz],
                [ cx*sy*cz+sx*sz, cx*cy,  cx*sy*sz-sx*cz],
                [sx*sy*cz-cx*sz, sx*cy,  sx*sy*sz+cx*cz]])
   
    return R

initial_pose = np.array([[1,0,0], [0,1,0], [0,0,1]]).astype(np.float32)



def sample_points(max_angle, samples = 50000):
    dataset ={'R':[], 'angles': [], 'axes': []}
    for i in range(samples):
        x,y,z = np.random.uniform(-max_angle, max_angle, size = 3)
        R = getR(x,y,z)
      
        dataset['R'].append(R)
        dataset['angles'].append([x,y,z])
        dataset['axes'].append(np.array(R.dot(initial_pose.T).T))

    for key in dataset.keys():
        dataset[key] = np.array(dataset[key])
    return dataset


class MyDataset(Dataset):
    def __init__(self, max_angle, samples = 50):
        self.data = sample_points(max_angle, samples)
        self.R = self.data['R']
        self.axes = self.data['axes']
        

    def __getitem__(self, index):
        R_out = self.R[index]
        axes_out = self.axes[index]

        return torch.tensor(axes_out).float(), torch.tensor(R_out).float()
    
    def __len__(self):
        return len(self.R)


if __name__ == '__main__':
    data = sample_points(45, 1)
    print(np.dot(data['R'][0].T, data['R'][0]))
    print(np.dot(data['axes'],data['R']))