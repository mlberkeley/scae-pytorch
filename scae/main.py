import argparse
import os
import wandb
import torch
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from easydict import EasyDict


def parse_args():
    parser = argparse.ArgumentParser()
    # Trainer params
    parser.add_argument('-bs', '--batch_size', type=int, default=2048)
    parser.add_argument('-es', '--num_epochs', type=int, default=150)
    parser.add_argument('--model', type=str, default='PCAE', help='PCAE')

    # Dataset Params
    parser.add_argument('--data', type=str, default="MNIST")
    # parser.add_argument('--use_aug', action="store_true", default=False, help="If this is true, using data augmentation")

    # Sub AutoEncoder Params
    pcae_args = parser.add_argument_group('PCAE Params')
    pcae_args.add_argument('--pcae_n_caps', type=int, default=16)
    pcae_args.add_argument('--pcae_caps_dim', type=int, default=6)
    pcae_args.add_argument('--pcae_feat_dim', type=int, default=16)
    pcae_args.add_argument('--pcae_lr', type=int, default=5e-2)
    pcae_args.add_argument('--alpha_channel', action="store_true", default=False)

    ocae_args = parser.add_argument_group('OCAE Params')
    ocae_args.add_argument('--ocae_lr', type=float, default=1e-1)

    # Logging Params
    logger_args = parser.add_argument_group('Logger Params')
    logger_args.add_argument('--name', type=str, default=None)
    logger_args.add_argument('--project', type=str, default='StackedCapsuleAutoEncoders')

    return EasyDict(vars(parser.parse_args()))


def main():
    args = parse_args()

    # Init Dataset
    if args.data == 'MNIST':
        args.num_classes = 10
        args.im_channels = 1

        from torchvision.datasets import MNIST
        t = transforms.Compose([
            transforms.RandomCrop(size=(40, 40), pad_if_needed=True),
            transforms.ToTensor()
        ])
        train_dataset = MNIST('data', train=True, transform=t, download=True)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
        val_dataset = MNIST('data', train=False, transform=t, download=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    else:
        raise NotImplementedError(args.data)

    # Init Logger
    logger = WandbLogger(name=args.name, project=args.project, config=args)

    # Init Model
    if args.model == 'PCAE':
        from scae.modules.part_capsule_ae import CapsuleImageEncoder, TemplateImageDecoder
        from scae.models.pcae import PCAE
        encoder = CapsuleImageEncoder(args.pcae_n_caps, args.pcae_caps_dim, args.pcae_feat_dim)
        decoder = TemplateImageDecoder(args.pcae_n_caps, use_alpha_channel=args.alpha_channel, output_size=(40, 40))
        model = PCAE(encoder, decoder, args)

        logger.watch(encoder._encoder, log='all', log_freq=10)
        logger.watch(decoder, log='all', log_freq=10)
    else:
        raise NotImplementedError()

    # Execute Experiment
    trainer = pl.Trainer(gpus=1, max_epochs=args.num_epochs, logger=logger)
    trainer.fit(model, train_dataloader)


if __name__ == "__main__":
    main()

