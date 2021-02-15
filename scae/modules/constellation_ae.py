import numpy as np
import torch.nn as nn
import torch
from scae.modules import attention as _attention
from scae.modules import capsule as _capsule
from attrdict import AttrDict
import scae.util.math as math
import torch.nn.functional as F
import torch.distributions as D
import collections


class ConstellationCapsule(nn.Module):

    """Decoder."""

    def __init__(self, input_dims, n_caps, n_caps_dims, n_votes, **capsule_kwargs):
        super(ConstellationCapsule, self).__init__()
        self._input_dims = input_dims
        self._n_caps = n_caps
        self._n_caps_dims = n_caps_dims
        self._n_votes = n_votes
        if capsule_kwargs:
            self._capsule_kwargs = capsule_kwargs
        else:
            self._capsule_kwargs = {"n_caps_params": 32, "n_hiddens": 128,
                                    "learn_vote_scale": True, "deformations": True,
                                    "noise_type": 'uniform', "noise_scale": 4.,
                                    "similarity_transform": False}
        self.build()

    def build(self):
        self.capsule = _capsule.CapsuleLayer(self._input_dims,
                                             self._n_caps, self._n_caps_dims, self._n_votes, **self._capsule_kwargs)

    def forward(self, h, x, presence=None):

        batch_size = int(x.shape[0])
        self.vote_shape = [batch_size, self._n_caps, self._n_votes, 6]

        res = self.capsule(h)

        # The paper literally just grabs these coordinates from our OV * OP
        # composition. Constructing a two-dimensional vote.
        res.vote = res.vote[..., :-1, -1]

        # Reshape dimensions for constellation specific.
        def pool_dim(x, dim_begin, dim_end):
            combined_shape = list(
                x.shape[:dim_begin]) + [-1] + list(x.shape[dim_end:])
            return x.view(combined_shape)

        for k, v in res.items():
            print(k, v.shape)
            if len(v.shape) == 4:
                print(v.shape)
                res[k] = pool_dim(v, 1, 2)
                print(res[k].shape)

        likelihood = _capsule.OrderInvariantCapsuleLikelihood(self._n_votes,
                                                              res.vote, res.scale,
                                                              res.vote_presence)
        ll_res = likelihood(x, presence)
        # print(ll_res)
