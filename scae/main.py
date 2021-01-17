import argparse
import os
from pathlib import Path

import wandb
from easydict import EasyDict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning.callbacks as cb

from scae.args import parse_args

data_path = Path('data')

norm_3c = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
norm_1c = transforms.Normalize([0.449], [0.226])

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

    dataloader_args = EasyDict(batch_size=args.batch_size, shuffle=False,
                               num_workers=0 if args.debug else args.data_workers)
    if args.dataset == 'mnist':
        args.num_classes = 10
        args.im_channels = 1
        args.image_size = (40, 40)

        from torchvision.datasets import MNIST

        t = transforms.Compose([
            transforms.RandomCrop(size=(40, 40), pad_if_needed=True),
            transforms.ToTensor(),
            # norm_1c
        ])
        train_dataloader = DataLoader(MNIST(data_path/'mnist', train=True, transform=t, download=True),
                                      **dataloader_args)
        val_dataloader = DataLoader(MNIST(data_path/'mnist', train=False, transform=t, download=True),
                                    **dataloader_args)
    elif args.dataset == 'usps':
        args.num_classes = 10
        args.im_channels = 1
        args.image_size = (40, 40)

        from torchvision.datasets import USPS

        t = transforms.Compose([
            transforms.RandomCrop(size=(40, 40), pad_if_needed=True),
            transforms.ToTensor(),
            # norm_1c
        ])
        train_dataloader = DataLoader(USPS(data_path/'usps', train=True, transform=t, download=True),
                                      **dataloader_args)
        val_dataloader = DataLoader(USPS(data_path/'usps', train=False, transform=t, download=True),
                                    **dataloader_args)
    elif args.dataset == 'cifar10':
        args.num_classes = 10
        args.im_channels = 3
        args.image_size = (32, 32)

        from torchvision.datasets import CIFAR10

        t = transforms.Compose([
            transforms.ToTensor()
        ])
        train_dataloader = DataLoader(CIFAR10(data_path/'cifar10', train=True, transform=t, download=True),
                                      **dataloader_args)
        val_dataloader = DataLoader(CIFAR10(data_path/'cifar10', train=False, transform=t, download=True),
                                    **dataloader_args)
    elif args.dataset == 'svhn':
        args.num_classes = 10
        args.im_channels = 3
        args.image_size = (32, 32)

        from torchvision.datasets import SVHN

        t = transforms.Compose([
            transforms.ToTensor()
        ])
        train_dataloader = DataLoader(SVHN(data_path/'svhn', split='train', transform=t, download=True),
                                      **dataloader_args)
        val_dataloader = DataLoader(SVHN(data_path/'svhn', split='test', transform=t, download=True),
                                    **dataloader_args)
    else:
        raise NotImplementedError()


    logger = WandbLogger(
        project=args.log.project,
        name=args.log.run_name,
        entity=args.log.team,
        config=args, offline=not args.log.upload)

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

        encoder = CapsuleImageEncoder(args)
        decoder = TemplateImageDecoder(args)
        model = PCAE(encoder, decoder, args)

        logger.watch(encoder._encoder, log='all', log_freq=args.log.frequency)
        logger.watch(decoder, log='all', log_freq=args.log.frequency)
    elif args.model == 'ocae':
        from scae.modules.object_capsule_ae import SetTransformer, ImageCapsule
        from scae.models.ocae import OCAE

        encoder = SetTransformer()
        decoder = ImageCapsule()
        model = OCAE(encoder, decoder, args)

        #  TODO: after ccae
    else:
        raise NotImplementedError()

    # Execute Experiment
    lr_logger = cb.LearningRateMonitor(logging_interval='step')
    best_checkpointer = cb.ModelCheckpoint(save_top_k=1, monitor='val_rec_ll', filepath=logger.experiment.dir)
    last_checkpointer = cb.ModelCheckpoint(save_last=True, filepath=logger.experiment.dir)
    trainer = pl.Trainer(gpus=1, max_epochs=args.num_epochs, logger=logger,
                         callbacks=[lr_logger, best_checkpointer, last_checkpointer])
    trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == "__main__":
    main()
