import wandb
from easydict import EasyDict
import torch_optimizer as optim

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl


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

def rec_to_wandb_im(x, **kwargs):  # TODO: move to utils
    # TODO: unpack reconstruction template components
    return to_wandb_im(x, **kwargs)


class PCAE(pl.LightningModule):
    def __init__(self, encoder, decoder, args):
        super(PCAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.lr = args.pcae_lr
        self.lr_decay = args.pcae_lr_decay
        self.weight_decay = args.pcae_weight_decay

        self.n_classes = args.num_classes
        self.mse = nn.MSELoss()

    def forward(self, image):
        capsules = self.encoder(image)
        reconstruction = self.decoder(capsules.poses, capsules.presences)
        return capsules, reconstruction

    def training_step(self, batch, batch_idx):
        # img    shape (batch_size, C, H, W)
        # labels shape (batch_size)
        img, labels = batch
        batch_size = img.shape[0]

        capsules, rec = self(img)
        rec_ll = rec.pdf.log_prob(img).view(batch_size, -1).sum(dim=-1).mean()
        self.log('rec_log_likelihood', rec_ll, prog_bar=True)

        temp_l1 = F.relu(self.decoder.templates).sum()
        self.log('temp_l1', temp_l1, prog_bar=True)

        rec_mse = self.mse(rec.pdf.mean(), img)
        self.log('rec_mse', rec_mse, prog_bar=True)

        losses = EasyDict(
            rec_log_likelihood=rec_ll,
            temp_l1=temp_l1,
            rec_mse=rec_mse
        )

        if batch_idx == 100:  # % 10 == 0:
            n = 8
            gt_imgs = [to_wandb_im(img[i], caption='gt_image') for i in range(n)]
            rec_imgs = [rec_to_wandb_im(rec.pdf.mean(idx=i), caption='rec_image') for i in range(n)]
            gt_rec_imgs = [None]*(2*n)
            gt_rec_imgs[::2], gt_rec_imgs[1::2] = gt_imgs, rec_imgs  # interweave

            template_imgs = [to_wandb_im(t, caption=f'tmp_{i}') for i, t in enumerate(rec.raw_templates)]
            mixture_mean_imgs = [to_wandb_im(t, caption=f'tmp_{i}') for i, t in enumerate(rec.mixture_means[0])]
            mixture_logit_imgs = [to_wandb_im(t, caption=f'tmp_{i}') for i, t in enumerate(rec.mixture_logits[0])]

            self.logger.experiment.log({
                'train_imgs': gt_rec_imgs,
                'templates': template_imgs,
                'mixture_means': mixture_mean_imgs,
                'mixture_logits': mixture_logit_imgs,
                'epoch': self.current_epoch},
                commit=False)

        loss = - rec_ll \
               + temp_l1 * self.weight_decay \
               - (capsules.presences).sum()

        if torch.isnan(loss).any():  # TODO: try grad clipping?
            raise ValueError('loss is nan')

        return loss

    def configure_optimizers(self):
        opt = optim.RAdam([
            {'params': self.encoder.parameters(), 'weight_decay': 0},
            {'params': self.decoder.parameters(), 'lr': self.lr * 50, 'weight_decay': 0}
        ], lr=self.lr, weight_decay=self.weight_decay)

        lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=self.lr_decay)

        return [opt], [lr_sched]
