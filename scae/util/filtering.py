from scipy import ndimage
import numpy as np
import torch
import torch.nn.functional as F

def torch_sobel_filter(tensor):
    filter = torch.tensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]])
    f = filter.expand(1, 3, 3, 3)
    sobel = F.conv2d(tensor.unsqueeze(0), f, stride=1, padding=1)
    return sobel.squeeze(dim=1).repeat(3, 1, 1)

def sobel_filter(img):
    return ndimage.sobel(img)

def normalize(tensor):
    med = torch.median(tensor)
    tensor = torch.abs(tensor - med)
    sobelmax = torch.max(tensor)
    sobelmin = torch.min(tensor)
    return torch.div(tensor - sobelmin, sobelmax - sobelmin + 1e-8)