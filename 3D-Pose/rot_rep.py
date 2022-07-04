import torch
import math
import roma

"""
Taken from Ola Alstad
Code from: https://github.com/olaals/end-to-end-RGB-pose-estimation-baseline/blob/main/rotation_representation.py

Contains:
- mapping to orthogonal 6d
- mapping to SO(3) via symmetric orthogonalization
- mapping with Lie Algebra
"""


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


def so3_exp_map(
    log_rot: torch.Tensor, eps: float = 0.0001
):
    """
    Copied from Pytorch3D : https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/so3.html#so3_exp_map
    A helper function that computes the so3 exponential map and,
    apart from the rotation matrix, also returns intermediate variables
    that can be re-used in other functions.
    """
    _, dim = log_rot.shape
    if dim != 3:
        raise ValueError("Input tensor shape has to be Nx3.")

    nrms = (log_rot * log_rot).sum(1)
    # phis ... rotation angles
    rot_angles = torch.clamp(nrms, eps).sqrt()
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
    skews = hat(log_rot)
    skews_square = torch.bmm(skews, skews)

    R = (
        # pyre-fixme[16]: `float` has no attribute `__getitem__`.
        fac1[:, None, None] * skews
        + fac2[:, None, None] * skews_square
        + torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]
    )

    return R


def hat(v: torch.Tensor) -> torch.Tensor:
    """
    Copied from Pytorch3D:  https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/so3.html#so3_exp_map
    Compute the Hat operator [1] of a batch of 3D vectors.
    Args:
        v: Batch of vectors of shape `(minibatch , 3)`.
    Returns:
        Batch of skew-symmetric matrices of shape
        `(minibatch, 3 , 3)` where each matrix is of the form:
            `[    0  -v_z   v_y ]
             [  v_z     0  -v_x ]
             [ -v_y   v_x     0 ]`
    Raises:
        ValueError if `v` is of incorrect shape.
    [1] https://en.wikipedia.org/wiki/Hat_operator
    """

    N, dim = v.shape
    if dim != 3:
        raise ValueError("Input vectors have to be 3-dimensional.")

    h = torch.zeros((N, 3, 3), dtype=v.dtype, device=v.device)

    x, y, z = v.unbind(1)

    h[:, 0, 1] = -z
    h[:, 0, 2] = y
    h[:, 1, 0] = z
    h[:, 1, 2] = -x
    h[:, 2, 0] = -y
    h[:, 2, 1] = x
    return h


def vec_3d_to_SO3(x):
    """
    Based on the Lie Group mapping from the minimal vector space to the SO3 manifold (rot mat)
    """
    device = x.device
    bsz = x.shape[0]
    assert x.shape == (bsz, 3)

    rot_mats = so3_exp_map(x)
    return rot_mats


transform_output = {'SVD': (9, symmetric_orthogonalization), '6D': (
    6, compute_rotation_matrix_from_ortho6d), '3D': (3, vec_3d_to_SO3)}


def get_scene_parameters():
    '''
    camera is a dict with the following values:
    - Focal length
    - Sensor width
    - Img res
    Need
    - f_x, f_y, v_x, v_y and v_z
    '''
    sw = 36
    img_res = 320
    flen = 50
    ppm = sw / img_res
    fx = fy = flen / ppm
    vx = vy = img_res / 2

    return fx, fy, vx, vy


def parametrization(output, T_init, fx, fy, vx, vy, rot_rep):
    '''
    The formulas are taken from CosyPose paper 
    https://arxiv.org/pdf/2008.08465.pdf
    '''
    vector_length, representation_func = transform_output[rot_rep]
    # Transform the first part of the output to SO3
    R = representation_func(output[:, 0: vector_length])
    vx, vy, vz = output[:, vector_len:]

    #assert translation.shape[1:] == 3
    xk, yk, zk = T_init[:, 2]

    # xk1 = x_{k+1}
    zk1 = vz * zk
    yk1 = (vy / fy + yk / zk) * zk1
    xk1 = (vx / fx + kk / zk) * zk1

    batch_size = output.size[0]
    T_out = torch.zeros((batch_size, 4, 4)).to(device)

    R_new = R @ T_init[:, :3, :3]

    T_out[:, :3, :3] = R_new
    T_out[:, :3, 3] = torch.tensor([xk1, yk1, zk1])
    T_out[:, 3, 3] = 1

    return T_out
