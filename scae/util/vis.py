import torch
import matplotlib.pyplot as plt
from functools import partial
import numpy as np
from easydict import EasyDict


def show_img(tensor, min=0, max=1):
    img = tensor.cpu().numpy()

    fig, ax = plt.subplots()
    im = ax.imshow(img, vmin=min, vmax=max, norm=False)
    fig.colorbar(im, ax=ax)

    plt.show()


def plot_image_tensor(X):
    plt.figure()
    num_imgs = X.shape[0]
    len_y = int(np.floor(np.sqrt(num_imgs)))
    len_x = int(np.ceil(num_imgs / len_y))

    fig, ax = plt.subplots(len_y, len_x)
    for i in range(len_y):
        for j in range(len_x):
            idx = i * len_x + j
            if idx < num_imgs:
                ax[i, j].imshow(X[idx][0].detach().cpu().numpy())
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)
    fig.subplots_adjust(hspace=0.1)  # No horizontal space between subplots
    fig.subplots_adjust(wspace=0)
    fig.show()

def plot_image_tensor_2D(X):
    plt.figure()
    len_y, len_x = X.shape[:2]
    fig, ax = plt.subplots(len_y, len_x)
    for i in range(len_y):
        for j in range(len_x):
            ax[i, j].imshow(X[i, j][0].detach().cpu().numpy())
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)
    fig.subplots_adjust(hspace=0.1)  # No horizontal space between subplots
    fig.subplots_adjust(wspace=0)
    fig.show()

# with open('scae/save.pkl', 'rb') as input:
#     capsules_l = pickle.load(input)
#     rec_l = pickle.load(input)
