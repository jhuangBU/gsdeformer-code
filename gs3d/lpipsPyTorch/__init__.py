import torch

from .modules.lpips import LPIPS


def lpips(x: torch.Tensor,
          y: torch.Tensor,
          net_type: str = 'alex',
          version: str = '0.1',
          mask: torch.Tensor = None,
          background: float = 1.0):
    r"""Function that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        x, y (torch.Tensor): the input tensors to compare.
        net_type (str): the network type to compare the features: 
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
        mask (torch.Tensor): optional mask where 1=keep, 0=ignore.
        background (float): value to fill masked regions. Default: 1.0.
    """
    if mask is not None:
        assert mask.shape == x.shape, \
            f"Mask shape {mask.shape} must match image shape {x.shape}"
        x = x * mask + background * (1 - mask)
        y = y * mask + background * (1 - mask)

    device = x.device
    criterion = LPIPS(net_type, version).to(device)
    return criterion(x, y)
