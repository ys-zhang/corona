from typing import Optional
import torch
from torch import nn

from corona.core.utils import make_parameter


class LinearSensitivity(nn.Module):

    weight: Optional[torch.Tensor]
    bias: Optional[torch.Tensor]
    mm: bool  # if weight act to base using matrix multiply, default False

    def __init__(self, weight=None, bias=None, mm: bool=False, *, name: Optional[str]=None):
        super().__init__()
        if weight is None:
            self.register_parameter('weight', None)
        else:
            self.weight = make_parameter(weight)
        if bias is None:
            self.register_parameter('bias', None)
        else:
            self.bias = make_parameter(bias)
        self.mm = mm
        self.name = name

    def forward(self, base, **kwargs):
        try:
            if not self.mm:
                wb = self.weight * base
            else:
                wb = self.weight @ base
        except TypeError:
            wb = base
        try:
            result = wb + self.bias
        except TypeError:
            result = wb
        return result
