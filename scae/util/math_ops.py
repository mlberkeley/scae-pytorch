import torch
import torch.nn as nn
import numpy as np


def geometric_transform(pose_tensors, similarity=False, nonlinear=True, as_3x3=False):
    """
    Converts parameter tensor into an affine or similarity transform.
    :param pose_tensor: [..., 6] tensor.
    :param similarity: bool.
    :param nonlinear: bool; applies nonlinearities to pose params if True.
    :param as_matrix: bool; convers the transform to a matrix if True.
    :return: [..., 3, 3] tensor if `as_matrix` else [..., 6] tensor.
    """
    trans_xs, trans_ys, scale_xs, scale_ys, thetas, shears = pose_tensors.split(1, dim=-1)

    if nonlinear:
        trans_xs = nn.functional.tanh(trans_xs * 5.)
        trans_ys = nn.functional.tanh(trans_ys * 5.)
        scale_xs = nn.functional.sigmoid(scale_xs) + 1e-2
        scale_ys = nn.functional.sigmoid(scale_ys) + 1e-2
        shears = nn.functional.tanh(shears * 5.)
        thetas *= 2. * np.pi
    else:
        scale_xs = torch.abs(scale_xs) + 1e-2
        scale_ys = torch.abs(scale_ys) + 1e-2

    cos_thetas, sin_thetas = torch.cos(thetas), torch.sin(thetas)

    if similarity:
        scales = scale_xs
        poses = [scales * cos_thetas, -scales * sin_thetas, trans_xs,
                scales * sin_thetas, scales * cos_thetas, trans_ys]
    else:
        poses = [
            scale_xs * cos_thetas + shears * scale_ys * sin_thetas,
            -scale_xs * sin_thetas + shears * scale_ys * cos_thetas,
            trans_xs,
            scale_ys * sin_thetas,
            scale_ys * cos_thetas,
            trans_ys
        ]
    poses = torch.cat(poses, -1)  # shape (... , 6)

    # Convert poses to 3x3 A matrix so: [y, 1] = A [x, 1]
    if as_3x3:
        poses = poses.reshape(poses.shape[:-1], 2, 3)
        bottom_pad = torch.zeros(poses.shape[:-1], 1, 3)
        bottom_pad[..., 2] = 1
        # shape (... , 2, 3) + shape (... , 1, 3) = shape (... , 3, 3)
        poses = torch.stack([poses, bottom_pad], dim=-2)
    return poses
