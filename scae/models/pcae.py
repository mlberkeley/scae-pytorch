import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
from easydict import EasyDict


def to_wandb_im(x):  # TODO: move to utils
    if len(x.shape) == 3:
        x = x.permute(1, 2, 0)
    return x.cpu().numpy()


class PCAE(pl.LightningModule):
    def __init__(self, encoder, decoder, args):
        super(PCAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.n_classes = args.num_classes
        self.lr = args.pcae_lr
        self.dynamic_l2_weight = 0.
        self.caps_ll_weight = 0.
        self.primary_caps_sparsity_weight = 0.
        self.weight_decay = 0.

    def forward(self, image):
        capsules = self.encoder(image)
        reconstruction = self.decoder(capsules.poses, capsules.presences)
        return capsules, reconstruction

    def agg_losses(self, losses):
        loss = -losses.rec_ll
        return loss

    def training_step(self, batch, batch_idx):
        # img    shape (batch_size, C,     H, W)
        # labels shape (batch_size)
        img, labels = batch

        capsules, reconstruction = self(img)
        rec_ll = reconstruction.pdf.log_prob(img).mean(dim=0)
        self.log('rec_log_likelihood', rec_ll, prog_bar=True)

        if batch_idx % 10 == 0:
            gt_img = wandb.Image(to_wandb_im(img[0]), caption='gt_image')
            rec_img = wandb.Image(to_wandb_im(img[0]), caption='rec_image')
            self.logger.experiment.log({'train_img': [gt_img, rec_img]}, commit=False)
        return rec_ll

    def training_epoch_end(self, outputs):
        ...
        # self.log('train_acc_epoch', self.accuracy.compute())

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)
