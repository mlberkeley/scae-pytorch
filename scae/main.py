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


def parse_args():
    parser = argparse.ArgumentParser(
        prog='scae',
        description='Training/evaluation/inference script for Stacked Capsule Autoencoders',
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # todo(maximsmol): add inference and evaluation
    parser.add_argument(
        '-n', '--batch-size',
        type=int, default=128,
        help='number of samples to per mini-batch')
    parser.add_argument(
        '-N', '--num-epochs',
        type=int, default=3000,
        metavar='NEPOCHS',
        help='number of epochs to train for')

    parser.add_argument(
        '--model',
        type=str.lower, default='pcae',
        choices=['ccae', 'pcae', 'ocae', 'scae'],
        # todo(maximsmol): change default
        help='which part of the model to run')


    parser.add_argument(
        '--dataset',
        type=str.lower, default='mnist',
        choices=['mnist'],
        help='dataset to use [default = MNIST]')
    parser.add_argument(
        '--data-workers',
        type=int, default=len(os.sched_getaffinity(0)),
        metavar='NWORKERS',
        help='number of data loader workers to spawn')


    pcae_args = parser.add_argument_group('PCAE Parameters')
    pcae_args.add_argument(
        '--pcae-num-caps',
        type=int, default=16,
        metavar='PCAE_NCAPS',
        help='number of capsules')
    pcae_args.add_argument(
        '--pcae-caps-dim',
        type=int, default=6,
        help='number of dimensions per capsule')
    pcae_args.add_argument(
        '--pcae-feat-dim',
        type=int, default=16,
        help='number of feature dimensions per capsule')
    pcae_args.add_argument(
        '--pcae-lr',
        type=float, default=1e-4,
        help='learning rate')
    # .998 = 1-(1-.96)**1/20, equiv to .96 every 20 epochs
    pcae_args.add_argument(
        '--pcae-lr-decay',
        type=float, default=.998,
        help='learning rate decay')
    pcae_args.add_argument(
        '--pcae-weight-decay',
        type=float, default=.01,
        help='weight decay')
    pcae_args.add_argument(
        '--alpha-channel',
        action='store_true', default=False,
        help='expect input images to have an alpha channel')


    ocae_args = parser.add_argument_group('OCAE Parameters')
    ocae_args.add_argument(
        '--ocae-lr',
        type=float, default=1e-1,
        help='learning rate')


    logger_args = parser.add_argument_group('Logger Parameters')
    logger_args.add_argument(
        '--logger-run-name',
        type=str, default=None,
        help='run name to use')
    logger_args.add_argument(
        '--logger-project',
        type=str, default='SCAE',
        help='project to log to')

    return EasyDict(vars(parser.parse_args()))


def main():
    args = parse_args()

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # torch.set_deterministic(True) # grid_sampler_2d_backward_cuda does not have a deterministic implementation
    torch.autograd.set_detect_anomaly(True)

    # Init Dataset
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

    # Init Logger
    logger = WandbLogger(name=args.logger_run_name, project=args.logger_project, config=args, offline=True)

    # Init Model
    if args.model == 'ccae':
        from scae.modules.constellation_ae import (SetTransformer,
                                                   ConstellationCapsule)
        from scae.models.ccae import CCAE

        encoder = SetTransformer()
        decoder = ConstellationCapsule()
        model = CCAE(encoder, decoder, args)

        # logger.watch(encoder._encoder, log='all', log_freq=10)
        # logger.watch(decoder, log='all', log_freq=10)

    if args.model == 'pcae':
        from scae.modules.part_capsule_ae import (CapsuleImageEncoder,
                                                  TemplateImageDecoder)
        from scae.models.pcae import PCAE

        encoder = CapsuleImageEncoder(
            args.pcae_num_caps, args.pcae_caps_dim, args.pcae_feat_dim)
        decoder = TemplateImageDecoder(
            args.pcae_num_caps, use_alpha_channel=args.alpha_channel, output_size=(40, 40))
        model = PCAE(encoder, decoder, args)

        logger.watch(encoder._encoder, log='all', log_freq=10)
        logger.watch(decoder, log='all', log_freq=10)
    elif args.model == 'ocae':
        from scae.modules.object_capsule_ae import (SetTransformer,
                                                    ImageCapsule)
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
