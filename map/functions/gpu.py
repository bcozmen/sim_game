from functools import wraps

import numpy as np
import torch
import torch.nn.functional as F
from numba import njit




def compute_on_gpu(func):
    """
    Decorator: moves a numpy array argument to the best available device,
    runs *func*, then returns the result as numpy.
    Supports (H, W) and (C, H, W) inputs.
    """
    @wraps(func)
    def wrapper(x, *args, **kwargs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #if not torch tensor, convert to tensor and move to device
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)
        x = x.to(device)
        result = func(x, *args, **kwargs)
        if isinstance(result, torch.Tensor):
            return result.cpu().numpy()
        tensor, *rest = result
        return (tensor.cpu().numpy(), *rest)
    return wrapper

@compute_on_gpu
def masked_avg(array, mask, radius):
    if not isinstance(mask, torch.Tensor):
        mask = torch.from_numpy(mask)
    mask = mask.to(array.device, dtype=array.dtype)

    # Ensure 4D: (N, C, H, W)
    if array.dim() == 2:
        array = array.unsqueeze(0).unsqueeze(0)
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif array.dim() == 3:
        array = array.unsqueeze(1)
        mask = mask.unsqueeze(1)

    k = 2 * radius + 1

    kernel = torch.ones((1, 1, k, k), device=array.device, dtype=array.dtype)

    masked_array = array * mask

    numerator = F.conv2d(masked_array, kernel, padding=radius)
    denominator = F.conv2d(mask, kernel, padding=radius)

    result = numerator / denominator.clamp(min=1e-8)
    result[denominator == 0] = 0

    return result.squeeze()




@compute_on_gpu
def binary_dilation(x, k=3, iterations=1):   
    # ensure shape (1,1,H,W)
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        x = x.unsqueeze(1)

    x = x.float()

    for _ in range(iterations):
        x = F.max_pool2d(x, kernel_size=k, stride=1, padding=k // 2)

    x = (x > 0)

    return x.squeeze()

@compute_on_gpu
def binary_erosion(x, k=3, iterations=1):
    # ensure shape (1,1,H,W)
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        x = x.unsqueeze(1)

    x = x.float()

    for _ in range(iterations):
        x = -F.max_pool2d(-x, kernel_size=k, stride=1, padding=k // 2)

    x = (x > 0)

    return x.squeeze()



