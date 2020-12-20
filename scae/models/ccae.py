import wandb
from easydict import EasyDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl


def rec_to_wandb_im(x, **kwargs):  # TODO: move to utils
    # TODO: unpack reconstruction template components
    return to_wandb_im(x, **kwargs)


class CCAE(pl.LightningModule):

    """Reconstruct points from constellations (groupings of points).

    Two-dimensional points as "parts" in the SCAE paradigm. Object capsule
    behavior and reconstruction logic is nearly identical to OCAE."""

    def __init__(self, encoder, decoder, args):
        super(CCAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.lr = args.ccae_lr
        self.lr_decay = args.ccae_lr_decay
        self.weight_decay = args.ccae_weight_decay

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
