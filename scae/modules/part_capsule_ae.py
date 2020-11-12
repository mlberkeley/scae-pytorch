import numpy as np
import torch.nn as nn
import torch
from attrdict import AttrDict

import scae.util.math_utils as math_utils


class ConvAttn(nn.Module):
    def __init__(self, n_caps, n_dims, in_channels=128):
        super(ConvAttn, self).__init__()
        self._n_caps = n_caps
        self._n_dims = n_dims
        self._in_channels = in_channels

        self._feat = nn.Conv2d(in_channels, n_dims * n_caps, kernel_size=1, stride=1)
        self._attn = nn.Conv2d(in_channels, n_caps, kernel_size=1, stride=1)

    def forward(self, x):
        """
        Compute and apply per-capsule attention masks
        :param x: Tensor of shape (batch_size, self._in_channels, H, W)
        :return: Tensor of shape (batch_size, self._n_caps, self._n_dims)
        """
        # TODO: add visualization of attn maps for debugging
        batch_size = x.shape[0]

        a = self._attn(x)
        a = a.reshape(batch_size, self._n_caps, 1, -1)  # flatten image dims (HxW)
        a_mask = nn.functional.softmax(a, dim=-1)

        f = self._feat(x)
        f = f.reshape(batch_size, self._n_caps, self._n_dims, -1)  # flatten image dims (HxW)

        # weighted sum over all image pixels, different weighting for each capsule
        return (f * a_mask).sum(-1, keepdim=False)


class CapsuleImageEncoder(nn.Module):
    def __init__(self,
               n_caps=16,
               n_caps_dims=6,
               n_features=16,
               noise_scale=4.,
               similarity_transform=False):
        super(CapsuleImageEncoder, self).__init__()
        self._n_caps = n_caps
        self._n_caps_dims = n_caps_dims
        self._n_features = n_features
        self._noise_scale = noise_scale
        self._similarity_transform = similarity_transform

        # Image embedding encoder
        channels = [1, 128, 128, 128, 128]
        strides = [2, 2, 1, 1]
        layers = []
        for i in range(4):
            layers.append(nn.Conv2d(channels[i], channels[i+1], kernel_size=3, stride=strides[i]))
            layers.append(nn.ReLU())
        self._encoder = nn.Sequential(layers)

        # Conv attention
        self._splits = [self._n_caps_dims, self._n_features, 1]  # 1 for presence
        self._n_dims = sum(self._splits)
        self._attn = ConvAttn(n_caps, self._n_dims)

    def forward(self, x):
        """

        :param x: Image tensor of shape (batch_size, 1, H, W)
        :return:
        """
        batch_size = x.shape[0]
        img_embedding = self._encoder(x)  # img_embedding shape (batch_size, C, H, W)
        preds = self._attn(img_embedding)  # preds shape (batch_size, self._n_caps, self._n_dims)

        poses, features, presence_logits = preds.split(self._splits, dim=-1)

        # Tensor of shape (batch_size, self._n_caps, 6)
        poses = math_utils.geometric_transform(poses, self._similarity_transform)

        if self._n_features == 0:
            features = None

        presence_logits = presence_logits.squeeze(-1)
        if self._noise_scale > 0.:  # TODO: why do this???
            # Add uniform [-self._noise_scale/2, self._noise_scale/2] noise to logits
            presence_logits += ((torch.rand(presence_logits.shape) - .5) * self._noise_scale)
        presences = nn.functional.sigmoid(presence_logits)

        return AttrDict(
            poses=poses,
            features=features,
            presences=presences,
            presence_logits=presence_logits,
            img_embedding=img_embedding
        )


def get_nonlin(name):
    nonlin = getattr(torch.functional, name, None)
    if nonlin:
        return nonlin
    else:
        raise ValueError('Invalid nonlinearity: "{}".'.format(name))


class TemplateImageDecoder(nn.Module):
    def __init__(self,
               output_size=(40, 40),
               template_size=(11, 11),
               n_channels=1,
               colorize_templates=True,
               template_nonlin='sigmoid',
               color_nonlin='sigmoid',
               use_alpha_channel=True):
        super(TemplateImageDecoder, self).__init__()
        self._output_size = output_size
        self._template_size = template_size
        self._n_channels = n_channels
        self._colorize_templates = colorize_templates

        self._template_nonlin = get_nonlin(template_nonlin)
        self._color_nonlin = get_nonlin(color_nonlin)
        self._use_alpha_channel = use_alpha_channel

        assert len(self._template_size) == 2, 'Template size must be of dim 2'
        self.init_templates()
        self.bg_value = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
        self.bg_logit = torch.nn.Parameter(torch.zeros(1), requires_grad=True)

    def init_templates(self):
        template_shape = [self._n_caps, self._n_channels] + list(self._template_size)  # torch generally uses (N, C, H, W)
        n_elems = np.prod(template_shape[1:])

        # make each templates orthogonal to each other at init
        n = max(self._n_caps, n_elems)
        q, _ = torch.qr(torch.rand(n, n))
        ts = q[:self._n_caps, :n_elems].reshape(template_shape)

        t_min = ts.min()
        t_max = ts.max()
        ts = (ts - t_min) / (t_max - t_min)  # normalize values to [0,1]

        if self._use_alpha_channel:
            alphas = torch.zeros(self._n_caps, 1, *self._template_size)
            ts = torch.stack([ts, alphas], dim=1)

        self.templates = torch.nn.Parameter(self._template_nonlin(ts), requires_grad=True)

    def forward(self, capsules, bg_image=None):
        """

        :param capsules:
        :param bg_image: size (N, C, H, W)
        :return:
        """
        batch_size = capsules.poses.size[0]
        n_dims = self._n_channels + 1 if self._use_alpha_channel else self._n_channels

        template_batch_shape = [batch_size, self._n_caps, n_dims] + list(self._output_size)

        # poses shape (batch_size * self._n_caps, 2, 3)
        poses = capsules.poses.view(-1, 2, 3)
        grid_coords = nn.functional.affine_grid(theta=poses, size=self._output_size)

        # templates             shape             (self._n_caps, n_dims, self._template_size)
        # template_stack        shape (batch_size* self._n_caps, n_dims, self._template_size)
        # transformed_templates shape (batch_size, self._n_caps, n_dims, self._output_size)
        template_stack = self.templates.expand(batch_size, 1, 1, 1)  # TODO: see if auto broadcasting over batch dim works
        transformed_templates = nn.functional.grid_sample(template_stack, grid_coords).view(template_batch_shape)

        if bg_image is None:
            bg_value = torch.nn.functional.sigmoid(self.bg_value)
            bg_image = torch.zeros(batch_size, self._n_channels, *self._output_size) + bg_value

        if self._use_alpha_channel:
            # template_logits    shape (batch_size, self._n_caps, self._output_size)
            # capsules.presences shape (batch_size, self._n_caps)
            presence_probs = capsules.presences.view(batch_size, self._n_caps, 1, 1)
            template_logits = transformed_templates[:, :, -1, ...] + torch.log(presence_probs)  # TODO: make log safe
        else:
            raise NotImplementedError()
            # temperature_logit = tf.get_variable('temperature_logit', shape=[1])
            # temperature = tf.nn.softplus(temperature_logit + .5) + 1e-4
            # template_mixing_logits = transformed_templates / temperature

        # TODO: add template colorization from features

        # mixture_logits shape (batch_size, self._n_caps + 1, self._output_size)
        # mixture_means  shape (batch_size, self._n_caps + 1, self._n_channels, self._output_size)
        mixture_logits = torch.stack([template_logits, self.bg_logit.view(1, 1, 1, 1)], dim=1)
        mixture_means = torch.stack([transformed_templates[:, :, :-1, ...], bg_image], dim=1)
        mixture_pdf = math_utils.MixtureDistribution(mixture_logits, mixture_means)

        return AttrDict(
            raw_templates=self.templates,
            mixture_means=mixture_means,
            mixture_logits=mixture_logits,
            pdf=mixture_pdf
        )


class PartCapsuleAE(nn.Module):
    def __init__(
            self,
            encoder,
            decoder,
            input_key,
            label_key=None,
            n_classes=None,
            dynamic_l2_weight=0.,
            caps_ll_weight=0.,
            vote_type='soft',
            pres_type='enc',
            img_summaries=False,
            stop_grad_caps_input=True,
            stop_grad_caps_target=True,
            prior_sparsity_loss_type='kl',
            prior_within_example_sparsity_weight=0.,
            prior_between_example_sparsity_weight=0.,
            prior_within_example_constant=0.,
            posterior_sparsity_loss_type='kl',
            posterior_within_example_sparsity_weight=0.,
            posterior_between_example_sparsity_weight=0.,
            primary_caps_sparsity_weight=0.,
            weight_decay=0.,
            feed_templates=True,
            prep='none'):
        super(PartCapsuleAE, self).__init__()
        self._encoder = encoder
        self._decoder = decoder
        self._input_key = input_key
        self._label_key = label_key
        self._n_classes = n_classes

        self._dynamic_l2_weight = dynamic_l2_weight
        self._caps_ll_weight = caps_ll_weight
        self._vote_type = vote_type
        self._pres_type = pres_type
        self._img_summaries = img_summaries

        self._stop_grad_caps_input = stop_grad_caps_input
        self._stop_grad_caps_target = stop_grad_caps_target
        self._prior_sparsity_loss_type = prior_sparsity_loss_type
        self._prior_within_example_sparsity_weight = prior_within_example_sparsity_weight
        self._prior_between_example_sparsity_weight = prior_between_example_sparsity_weight
        self._prior_within_example_constant = prior_within_example_constant
        self._posterior_sparsity_loss_type = posterior_sparsity_loss_type
        self._posterior_within_example_sparsity_weight = posterior_within_example_sparsity_weight
        self._posterior_between_example_sparsity_weight = posterior_between_example_sparsity_weight
        self._primary_caps_sparsity_weight = primary_caps_sparsity_weight
        self._weight_decay = weight_decay
        self._feed_templates = feed_templates

