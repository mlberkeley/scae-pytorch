import numpy as np
import torch.nn as nn
import torch
from attrdict import AttrDict
import scae.util.math_utils as math_utils
from torch.distributions.bernoulli import Bernoulli
from util import math as math
import torch.nn.functional as F

class CapsuleLayer(nn.Module):
    _n_transform_params = 6
    def __init__(self, n_caps, n_caps_dims, n_votes, n_caps_params=None,
               n_hiddens=128, learn_vote_scale=False, deformations=True,
               noise_type=None, noise_scale=0., similarity_transform=True,
               caps_dropout_rate=0.0):
      super(CapsuleLayer, self).__init__()
      self._n_caps = n_caps
      self._n_caps_dims = n_caps_dims
      self._n_caps_params = n_caps_params

      self._n_votes = n_votes
      self._n_hiddens = n_hiddens
      self._learn_vote_scale = learn_vote_scale
      self._deformations = deformations
      self._noise_type = noise_type
      self._noise_scale = noise_scale

      self._similarity_transform = similarity_transform
      self._caps_dropout_rate = caps_dropout_rate

      assert n_caps_dims == 2, ('This is the only value implemented now due to '
                              'the restriction of similarity transform.')
      
      self.output_shapes = (
            [self._n_votes, self._n_transform_params],  # CPR_dynamic
            [1, self._n_transform_params],  # CCR
            [1],  # per-capsule presence
            [self._n_votes],  # per-vote-presence
            [self._n_votes],  # per-vote scale
            )
      self.splits = [np.prod(i).astype(np.int32) for i in self.output_shapes]
      self.n_outputs = sum(self.splits)
      
      self.build(self._n_hiddens, self._n_caps_params, self.n_outputs)
      
    def build(self, n_hiddens, n_params, n_out):
        self.mlp = nn.ModuleList()
        for n_hidden in n_hiddens + [n_params]:
            self.mlp.append(nn.Linear(n_hidden))
            self.mlp.append(nn.ReLU())
        
        self.caps_mlp = nn.ModuleList()
        for n_hidden in [n_hiddens, n_out]:
            self.caps_mlp.append(nn.Linear(n_hidden, bias = False))
            self.caps_mlp.append(nn.ReLU())
        
        
    def forward(self, x, parent_transform=None, parent_presence=None):
        batch_size = x.shape[0]
        batch_shape = [batch_size, self._n_caps]
        if self._n_caps_params is not None:
            raw_caps_params = self.mlp(x)
            caps_params = torch.reshape(raw_caps_params,
                                   batch_shape + [self._n_caps_params])
        else:
            assert x.shape[:2].as_list() == batch_shape
            caps_params = x
        
        if self._caps_dropout_rate == 0.0:
            caps_exist = torch.ones(batch_shape + [1], dtype=torch.float32)
        else:
            pmf = Bernoulli(1. - self._caps_dropout_rate, dtype=torch.float32)
            caps_exist = pmf.sample(batch_shape + [1])
        
        caps_params = torch.cat([caps_params, caps_exist], -1)
        all_params = self.caps_mlp(caps_params)
        all_params = torch.split(all_params, self.splits, -1)
        res = [torch.reshape(i, batch_shape + s)
           for (i, s) in zip(all_params, self.output_shapes)]
        cpr_dynamic = res[0]
        
        # res = [snt.AddBias()(i) for i in res[1:]]
        ccr, pres_logit_per_caps, pres_logit_per_vote, scale_per_vote = res
        
        if self._caps_dropout_rate != 0.0:
            pres_logit_per_caps += math.safe_log(caps_exist)
        
        cpr_static = tf.get_variable('cpr_static', shape=[1, self._n_caps,
                                                      self._n_votes,
                                                      self._n_transform_params])
        
        pres_logit_per_caps = math.add_noise(pres_logit_per_caps)
        pres_logit_per_vote = math.add_noise(pres_logit_per_vote)       
        
        if parent_transform is None:
            ccr = math.geometric_transform(ccr)
        else:
            ccr = parent_transform
        if not self._deformations:
            cpr_dynamic = torch.zeros_like(cpr_dynamic)
        
        cpr = math.geometric_transform(cpr_dynamic + cpr_static)
        ccr_per_vote = [2].tile([self._n_votes])(ccr)
        votes = torch.matmul(ccr_per_vote, cpr)
        
        if parent_presence is not None:
            pres_per_caps = parent_presence
        else:
            pres_per_caps = torch.sigmoid(pres_logit_per_caps)
        
        pres_per_vote = pres_per_caps * torch.sigmoid(pres_logit_per_vote)

        if self._learn_vote_scale:
            scale_per_vote = F.softplus(scale_per_vote + .5) + 1e-2
        else:
            scale_per_vote = torch.zeros_like(scale_per_vote) + 1.

        return AttrDict(
            vote=votes,
            scale=scale_per_vote,
            vote_presence=pres_per_vote,
            pres_logit_per_caps=pres_logit_per_caps,
            pres_logit_per_vote=pres_logit_per_vote,
            dynamic_weights_l2=tf.nn.l2_loss(cpr_dynamic) / batch_size,
            raw_caps_params=raw_caps_params,
            raw_caps_features=features,
        )
        
        
        
        
        
        
        
        
            
