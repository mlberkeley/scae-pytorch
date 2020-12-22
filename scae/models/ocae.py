import wandb
from easydict import EasyDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl


class OCAE(pl.LightningModule):

    """Reconstruct parts from object predictions.

    TODO: More information from paper here...
    """

    def __init__(self, encoder, decoder, args):
        super(OCAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.lr = args.ocae_lr
        self.lr_decay = args.ocae_lr_decay
        self.weight_decay = args.ocae_weight_decay

        self.n_classes = args.num_classes
        self.mse = nn.MSELoss()

    def forward(self, points):
        """Forward pass.

        Additional details here..."""

        capsules = self.encoder(points)
        reconstruction = self.decoder(capsules.poses, capsules.presences)
        return capsules, reconstruction

    def training_step(self, batch, batch_idx):
        # TODO
        # Pulled from domas' code
        pass

    def training_epoch_end(self, outputs):
        # TODO
        # Pulled from domas' code
        pass

    def configure_optimizers(self):
        # TODO
        # Pulled from domas' code
        pass
