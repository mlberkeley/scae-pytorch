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
from easydict import EasyDict


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

        batch_size, n_input_points = int(x.shape[0]), int(x.shape[0])
        self.vote_shape = [batch_size, self._n_caps, self._n_votes, 6]

        res = self.capsule(h)

        # The paper literally just grabs these coordinates from our OV * OP
        # composition. Constructing a two-dimensional vote.
        res.vote = res.vote[..., :-1, -1]

        # Reshape dimensions for constellation specific use case.
        def pool_dim(x, dim_begin, dim_end):
            combined_shape = list(
                x.shape[:dim_begin]) + [-1] + list(x.shape[dim_end:])
            return x.view(combined_shape)

        for k, v in res.items():
            print("raw out : ", v.shape)
            if k == "vote" or k == "scale":
                res[k] = pool_dim(v, 1, 3)
            if k == "vote_presence":
                print("ROSA PARKS: ", v.shape)
                res[k] = pool_dim(v, 1, 3)
                print("ROSA PARKS: ", res[k].shape)

        likelihood = _capsule.OrderInvariantCapsuleLikelihood(self._n_votes,
                                                              res.vote, res.scale,
                                                              res.vote_presence)
        ll_res = likelihood(x, presence)

        #
        # Mixing KL div.
        #

        soft_layer = torch.nn.Softmax(dim=1)
        mixing_probs = soft_layer(ll_res.mixing_logits)
        prior_mixing_log_prob = math.scalar_log(1. / n_input_points)
        mixing_kl = mixing_probs * \
            (ll_res.mixing_log_prob - prior_mixing_log_prob)
        mixing_kl = torch.mean(torch.sum(mixing_kl, -1))

        #
        # Sparsity loss.
        #

        from_capsule = ll_res.is_from_capsule

        # torch implementation of tf.one_hot
        idx = torch.eye(self._n_caps)
        wins_per_caps = torch.stack([idx[from_capsule[b].type(
            torch.LongTensor)] for b in range(from_capsule.shape[0])])

        if presence is not None:
            wins_per_caps *= torch.expand_dims(presence, -1)

        wins_per_caps = torch.sum(wins_per_caps, 1)

        has_any_wins = torch.gt(wins_per_caps, 0).float()
        should_be_active = torch.gt(wins_per_caps, 1).float()

        # https://pytorch.org/docs/stable/generated/torch.nn.MultiLabelSoftMarginLoss.html
        # From math, looks to be same as `tf.nn.sigmoid_cross_entropy_with_logits`.

        # TODO: not rigorous cross-implementation
        softmargin_loss = torch.nn.MultiLabelSoftMarginLoss()
        sparsity_loss = softmargin_loss(should_be_active,
                                        res.pres_logit_per_caps)

        # sparsity_loss = tf.reduce_sum(sparsity_loss * has_any_wins, -1)
        # sparsity_loss = tf.reduce_mean(sparsity_loss)

        caps_presence_prob = torch.max(torch.reshape(
            res.vote_presence, [batch_size, self._n_caps, self._n_votes]),
            2)[0]

        #
        # Constructing loss ensemble.
        #
        print(torch.mean(res.scale))

        return EasyDict(
            mixing_kl=mixing_kl,
            sparsity_loss=sparsity_loss,
            caps_presence_prob=caps_presence_prob,
            mean_scale=torch.mean(res.scale)
        )
