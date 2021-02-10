from scipy import ndimage
import numpy as np
import torch
import kornia
from PIL import Image

def torch_sobel_filter(tensor):
    out = torch.abs(kornia.filters.spatial_gradient(tensor.unsqueeze(0)))
    out = torch.sum(out, dim=2)
    m = torch.max(out)
    out = torch.div(out, m + 1e-8)
    #im = Image.fromarray(torch.sum(out, dim=2).detach().cpu().numpy()[0].transpose(2,1,0), mode="RGB")
    #im.save(r"C:\Users\nrdas\Downloads\MLAB\temp.png")
    #print("Min, Max:", torch.min(out), torch.max(out))

    #raise ZeroDivisionError

    # print(torch.sum(out, dim=2).detach().cpu().numpy()[0].transpose(1,2,0).shape)
    # im = Image.fromarray(torch.sum(out, dim=2).detach().cpu().numpy()[0].transpose(2,1,0), mode="RGB")
    # im.save(r"C:\Users\nrdas\Downloads\MLAB\temp.png")
    # raise ZeroDivisionError
    return out.squeeze(0)

def sobel_filter(img):
    return ndimage.sobel(img)

def normalize(tensor):
    med = torch.median(tensor)
    tensor = torch.abs(tensor - med)
    sobelmax = torch.max(tensor)
    sobelmin = torch.min(tensor)
    return torch.div(tensor - sobelmin, sobelmax - sobelmin + 1e-8)