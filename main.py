import argparse
import wandb

from common.modules import *
from IB.solver import Solver_MNIST as Solver

data_size = 784
latent_size = 256
output_size = 10

def build_IB():
    hidden_size = 1024
    dims = [data_size, hidden_size, hidden_size, latent_size]
    encoder = []
    for i, o in zip(dims[:-2], dims[1:-1]):
        encoder.append(nn.Linear(i, o))
        encoder.append(nn.BatchNorm1d(o)) if wandb.config.bneck_use_batchnorm else None
        encoder.append(nn.ReLU(o))
        encoder.append(nn.Dropout(p=wandb.config.bneck_dropout_p)) if wandb.config.bneck_dropout_p else None
    i, o = dims[-2:]
    encoder.append(nn.Linear(i, o))
    encoder.append(nn.ReLU())
    encoder = nn.Sequential(*encoder)

    decoder = nn.Sequential(
        nn.Linear(latent_size, output_size)
    )
    return IB(encoder, decoder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Train Amounts
    parser.add_argument('-b', '--batch_size', type=int, default=100)
    parser.add_argument('-e', '--num_epochs', type=int, default=200)
    # Learning Rates
    parser.add_argument('--bneck_lr', type=float, default=1e-4)
    parser.add_argument('--bneck_lr_decay', type=float, default=.97)
    # Regularization Params
    parser.add_argument('--bneck_dropout_p', type=float, default=0)
    parser.add_argument('--bneck_use_batchnorm', type=int, default=1)
    wandb.init(project="domain-invariance", tags=['IB'], config=parser.parse_args())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bottleneck = build_IB()
    print(bottleneck)

    mine_solver = Solver(bottleneck, device)
    mine_solver.train()