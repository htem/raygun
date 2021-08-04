import numpy as np
import warnings

import torch
from torch.nn.modules.distance import PairwiseDistance
from torch.nn.modules.module import Module
from torch.nn import functional as F
from torch.nn import _reduction as _Reduction
from torch.nn.modules.loss import _Loss

from torch import Tensor
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

class MaskedMSELoss(_Loss):
    r"""Creates a criterion that measures the mean squared error (squared L2 norm) between
    each non-masked element in the input :math:`x` and target :math:`y`.

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left( x_n - y_n \right)^2,

    where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), &  \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  &  \text{if reduction} = \text{`sum'.}
        \end{cases}

    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.

    The mean operation still operates over all the elements, and divides by :math:`n`.

    The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(N, *)` where :math:`*` means, any number of additional
          dimensions
        - Mask: :math:`(N, *)`, same shape as the input
        - Target: :math:`(N, *)`, same shape as the input

    Examples::

        ...
    """
    __constants__ = ['reduction']

    def __init__(self, reduction: str = 'mean') -> None:
        super(MaskedMSELoss, self).__init__(reduction)

    def forward(self, input: Tensor, mask: Tensor, target: Tensor) -> Tensor:
        masked_input = input[mask > 0]
        masked_target = target[mask > 0]
        return F.mse_loss(masked_input, masked_target, reduction=self.reduction)