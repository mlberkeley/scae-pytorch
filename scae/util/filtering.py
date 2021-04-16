import torch
import kornia


def sobel_filter(input: torch.Tensor):
    output = kornia.filters.spatial_gradient(input.unsqueeze(0)).sum(dim=2).squeeze(0)

    output = torch.abs(output - output.view(3, -1).median(-1)[0].unsqueeze(-1).unsqueeze(-1))

    output = output.sum(0, keepdim=True)
    sobelmax = torch.max(output)
    sobelmin = torch.min(output)
    return torch.div(output - sobelmin, sobelmax - sobelmin + 1e-8)
