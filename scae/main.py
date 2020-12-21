import argparse
import os

import wandb
from easydict import EasyDict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from scae.args import parse_args

def main():
    args = parse_args()

    if args.debug or not args.non_deterministic:
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # torch.set_deterministic(True) # grid_sampler_2d_backward_cuda does not have a deterministic implementation

    if args.debug:
        torch.autograd.set_detect_anomaly(True)


    if args.dataset == 'mnist':
        args.num_classes = 10
        args.im_channels = 1

        from torchvision.datasets import MNIST

        t = transforms.Compose([
            transforms.RandomCrop(size=(40, 40), pad_if_needed=True),
            transforms.ToTensor()
        ])

        train_dataset = MNIST('data', train=True, transform=t, download=True)
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.data_workers)

        val_dataset = MNIST('data', train=False, transform=t, download=True)
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.data_workers)
    else:
        raise NotImplementedError()


    logger = WandbLogger(
        project=args.log_project,
        name=args.log_run_name,
        config=args, offline=not args.log_upload)


    if args.model == 'ccae':
        from scae.modules.constellation_ae import SetTransformer, ConstellationCapsule
        from scae.models.ccae import CCAE

        encoder = SetTransformer()
        decoder = ConstellationCapsule()
        model = CCAE(encoder, decoder, args)

        # logger.watch(encoder._encoder, log='all', log_freq=args.log_frequency)
        # logger.watch(decoder, log='all', log_freq=args.log_frequency)
    elif args.model == 'pcae':
        from scae.modules.part_capsule_ae import CapsuleImageEncoder, TemplateImageDecoder
        from scae.models.pcae import PCAE

        encoder = CapsuleImageEncoder(
            args.pcae_num_caps, args.pcae_caps_dim, args.pcae_feat_dim)
        decoder = TemplateImageDecoder(
            args.pcae_num_caps, use_alpha_channel=args.alpha_channel, output_size=(40, 40))
        model = PCAE(encoder, decoder, args)

        logger.watch(encoder._encoder, log='all', log_freq=args.log_frequency)
        logger.watch(decoder, log='all', log_freq=args.log_frequency)
    elif args.model == 'ocae':
        from scae.modules.object_capsule_ae import SetTransformer, ImageCapsule
        from scae.models.ocae import OCAE

        encoder = SetTransformer()
        decoder = ImageCapsule()
        model = OCAE(encoder, decoder, args)

        #  TODO: after ccae #
    else:
        raise NotImplementedError()

    # Execute Experiment
    trainer = pl.Trainer(gpus=1, max_epochs=args.num_epochs, logger=logger)
    trainer.fit(model, train_dataloader)

if __name__ == "__main__":
    main()
