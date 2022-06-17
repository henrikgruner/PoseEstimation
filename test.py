import numpy as np
import math
T = np.array([[0.3598, 0.8291, -0.428, -0.0698],
              [0.0176, 0.4526, 0.8915, 0.1329],
              [0.9329, - 0.3283, 0.1483, - 2.5796],
              [0., 0., 0., 1.]])


R = T[:3, :3]

axis, theta = [4, 1, 3], 0


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


from numpy import cross, eye, dot
from scipy.linalg import expm, norm


def M(axis, theta):
    return expm(cross(eye(3), axis / norm(axis) * theta))


print(M(axis, 0))
print(M(axis, np.pi))
