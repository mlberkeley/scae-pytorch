import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
from easydict import EasyDict


def to_wandb_im(x, **kwargs):  # TODO: move to utils
    x = x.detach()
    if len(x.shape) == 3:
        # Torch uses C, H, W
        x = x.permute(1, 2, 0)
    if x.shape[-1] == 2:
        # channels = val, alpha
        val = x[..., 0]
        alpha = x[..., 1]
        # convert to RGBA
        x = torch.stack([val]*3 + [alpha], dim=-1)
    return wandb.Image(x.cpu().numpy(), **kwargs)


class PCAE(pl.LightningModule):
    def __init__(self, encoder, decoder, args):
        super(PCAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.n_classes = args.num_classes
        self.lr = args.pcae_lr
        self.weight_decay = .001

    def forward(self, image):
        capsules = self.encoder(image)
        reconstruction = self.decoder(capsules.poses, capsules.presences)
        return capsules, reconstruction

    def agg_losses(self, losses):
        loss = -losses.rec_ll
        return loss

    def training_step(self, batch, batch_idx):
        # img    shape (batch_size, C, H, W)
        # labels shape (batch_size)
        img, labels = batch

        capsules, rec = self(img)
        rec_ll = rec.pdf.log_prob(img).mean()
        self.log('rec_log_likelihood', rec_ll, prog_bar=True, on_step=True)

        if batch_idx % 10 == 0:
            n = 5
            gt_imgs = [to_wandb_im(img[i], caption='gt_image') for i in range(n)]
            rec_imgs = [to_wandb_im(rec.pdf.mean(idx=i), caption='rec_image') for i in range(n)]
            gt_rec_imgs = [None]*(2*n)
            gt_rec_imgs[::2], gt_rec_imgs[1::2] = gt_imgs, rec_imgs  # interweave

            template_imgs = [to_wandb_im(t, caption=f'tmp_{i}') for i, t in enumerate(rec.raw_templates)]
            mixture_mean_imgs = [to_wandb_im(t, caption=f'tmp_{i}') for i, t in enumerate(rec.mixture_means[0])]
            mixture_logit_imgs = [to_wandb_im(t, caption=f'tmp_{i}') for i, t in enumerate(rec.mixture_logits[0])]

            self.logger.experiment.log({
                'train_imgs': gt_rec_imgs,
                'templates': template_imgs,
                'mixture_means': mixture_mean_imgs,
                'mixture_logits': mixture_logit_imgs},
                commit=True)
        return -rec_ll + 100*torch.norm(self.decoder.templates)

    def training_epoch_end(self, outputs):
        ...
        # self.log('train_acc_epoch', self.accuracy.compute())

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
