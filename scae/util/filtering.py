from scipy import ndimage
import numpy as np
import torch
import kornia

def torch_sobel_filter(tensor):
    out = kornia.filters.spatial_gradient(tensor.unsqueeze(0))
    return torch.sum(out, dim=2).squeeze(0)

def sobel_filter(img):
    return ndimage.sobel(img)

def normalize(tensor):
    med = torch.median(tensor)
    tensor = torch.abs(tensor - med)
    sobelmax = torch.max(tensor)
    sobelmin = torch.min(tensor)
    return torch.div(tensor - sobelmin, sobelmax - sobelmin + 1e-8)