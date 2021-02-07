import torch
import kornia


def sobel_filter(input: torch.Tensor):
    output = kornia.filters.spatial_gradient(input.unsqueeze(0)).sum(dim=2).squeeze(0)
    return output


def normalize(tensor):
    med = torch.median(tensor)
    tensor = torch.abs(tensor - med)
    sobelmax = torch.max(tensor)
    sobelmin = torch.min(tensor)
    return torch.div(tensor - sobelmin, sobelmax - sobelmin + 1e-8)