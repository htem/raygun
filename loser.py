import numpy as np
import warnings

import torch
from torch.nn.modules.distance import PairwiseDistance
from torch.nn.modules.module import Module
from torch.nn import functional as F
from torch.nn import _reduction as _Reduction
from torch.nn.modules.loss import _Loss

from torch import Tensor, BoolTensor
from typing import Callable, Optional
import random

def lossFunctionN2V(samples, labels, masks):
    '''
    The loss function as described in Eq. 7 of the original Noise2Void paper.
    '''
        
    errors=(labels-torch.mean(samples,dim=0))**2

    # Average over pixels and batch
    loss= torch.sum( errors *masks  ) /torch.sum(masks)
    return loss

def maskedMSE(src: Tensor, mask: Tensor or BoolTensor, target: Tensor) -> Tensor:
        if not isinstance(mask, BoolTensor):
            mask = torch.gt(mask, 0)
        # pad = tuple()
        # for i, j in zip(torch.tensor(mask.size()), torch.tensor(src.size())):
        #     p = int(torch.div(torch.sub(i, j), 2))
        #     pad += p, p
        # src = F.pad(src, pad[::-1])
        
        # Average over pixels and batch
        loss = torch.mean(torch.sub(src[mask], target[mask])**2 /torch.sum(mask))
        return loss.requires_grad_()