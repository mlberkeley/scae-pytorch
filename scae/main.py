import argparse
import os
import wandb
import torch

from scae.modules.stacked_capsule_ae import CapsNet

def main():
    parser = argparse.ArgumentParser()
    # Trainer params
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-e', '--num_epochs', type=int, default=150)
    # Dataset
    parser.add_argument('--train_data', type=str, default="MNIST")
    parser.add_argument('--use_augmentation', action="store_true", default=False,
                        help="If this is true, using data augmentation (CIFAR10)")
    # Estimator hyperparams
    est_args = parser.add_argument_group('Estimator hyperparams')
    est_args.add_argument('--miest_width', type=int, default=1024)
    est_args.add_argument('--miest_lr', type=float, default=1e-4)

    enc_args = parser.add_argument_group('Encoder hyperparams')
    enc_args.add_argument('--encoder_lr', type=float, default=1e-1)
    enc_args.add_argument('--encoder_weight_decay', type=float, default=1e-4)

    # Logging hyperparameters
    logger_args = parser.add_argument_group('Logger Config')
    logger_args.add_argument('--name', type=str, default="")
    logger_args.add_argument('--project', type=str, default='thresholding')
    logger_args.add_argument('--log_interval', type=int, default=-1, help="Steps per logging")
    logger_args.add_argument('--save_interval', type=int, default=1, help="Epochs per saving")

    hparams = parser.parse_args()

    if hparams.train_data == "Tiny-ImageNet-C":
        hparams.num_classes = 200
    elif hparams.train_data == "cifar10":
        hparams.num_classes = 10
    elif hparams.train_data == 'cifar100':
        hparams.num_classes = 100
    else:
        raise NotImplementedError

    if hparams.pretrain_encoder < 0:
        hparams.pretrain_encoder = hparams.num_epochs

    hparams.host_name = os.uname()[1]

    wandb_kwargs = {}
    peel_args = ['name', 'project']

    for arg in peel_args:
        val = getattr(hparams, arg, None)
        wandb_kwargs[arg] = val

    wandb_kwargs['name'] = wandb_kwargs['name'].format(hparams)
    wandb.init(config=hparams, **wandb_kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(hparams)
    print(device)

    dataset =

    encoder = model.SingleBottleneck(hparams)
    discriminator = D.CMIEstimator(hparams)

    model = CapsNet(hparams, device)

    model.train(dataset)


if __name__ == "__main__":
    main()

