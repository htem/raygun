import torch

def get_base_optimizer(optim):
    if isinstance(optim, str):
        base_optim = getattr(torch.optim, optim)
    else:
        base_optim = optim
    return base_optim

        