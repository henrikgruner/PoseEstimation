import torch
from scipy.spatial.transform import Rotation as R
import spatialmath as sm
import numpy as np


def loss_frobenius(R_pred, R_true):
    difference = R_true - R_pred
    frob_norm = torch.linalg.matrix_norm(difference, ord='fro')

    return frob_norm.mean()


def rotate_by_180(R_guess):
    Rx = sm.SO3.Rx(np.pi)
    Ry = sm.SO3.Ry(np.pi)
    Rz = sm.SO3.Rz(np.pi)
    Rx = torch.tensor(R_guess.to('cpu') @ Rx.data[0], dtype=torch.float64)
    Ry = torch.tensor(R_guess.to('cpu') @ Ry.data[0], dtype=torch.float64)
    Rz = torch.tensor(R_guess.to('cpu') @ Rz.data[0], dtype=torch.float64)
    return Rx, Ry, Rz


if __name__ == '__main__':
    T = torch.tensor([[[0.3598, 0.8291, -0.428],
                      [0.0176, 0.4526, 0.8915],
                      [0.9329, -0.3283, 0.1483]],

                      [[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]]])
    R = T[:, :3, :3]
    Rx, Ry, Rz = rotate_by_180(R)

    print(Rx[0], '\n')
    print(Rx[1], '\n')
