# import functools  # TODO: update to python 3.8+ to use functools.cached_property

import numpy as np

import torch
import torch.nn as nn

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
        trans_xs = torch.tanh(trans_xs * 5.)
        trans_ys = torch.tanh(trans_ys * 5.)
        scale_xs = torch.sigmoid(scale_xs) + 1e-2
        scale_ys = torch.sigmoid(scale_ys) + 1e-2
        shears = torch.tanh(shears * 5.)
        thetas = thetas * 2. * np.pi
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


class MixtureDistribution(torch.distributions.Distribution):
    def __init__(self, mixing_logits, means, var=1, distribution=torch.distributions.Normal):
        """

        :param mixing_logits: size (batch_size, num_distributions, C, H, W)
        :param means:         size (batch_size, num_distributions, C, H, W)
        :param var:
        """
        super().__init__()
        self._mixing_logits = mixing_logits
        self._means = means
        self._var = var
        self._distributions = distribution(loc=means, scale=var)

    @property
    def mixing_log_prob(self):
        return self._mixing_logits - self._mixing_logits.exp().sum(dim=1, keepdims=True).log()

    @property
    def mixing_prob(self):
        return torch.softmax(self._mixing_logits, dim=1)

    def log_prob(self, value):
        """

        :param value: shape (batch_size, C, H, W)
        :return:      shape (batch_size)
        """
        batch_size = value.shape[0]
        # y      shape (batch_size, 1,                 C, H, W)
        y = value.unsqueeze(1)
        # logits shape (batch_size, num_distributions, C, H, W)
        logits = self._distributions.log_prob(y)
        # multiply probabilities over channels, multiply by mixing probs
        logits = logits.sum(dim=2) + self.mixing_log_prob.sum(dim=2)
        # sum probs over distributions
        logits = logits.exp().sum(dim=1).log()
        return logits

    def mean(self, idx=None):
        # returns weighted average over all distributions
        if idx is not None:
            return (self._means[idx] * self.mixing_prob[idx]).sum(dim=0).detach()
        else:
            return (self._means * self.mixing_prob).sum(dim=1).detach()

