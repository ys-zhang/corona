import numpy as np
import torch
from torch import Tensor
from torch.nn import Module, Parameter
from corona.utils import sens
from corona.const import MAX_YR_LEN


class DiscountLayer(Module):

    _RATE_INIT_UPPER_LIMIT = .05
    _RATE_INIT_LOWER_LIMIT = .01

    def __init__(self, forward_rate=None, time_len=MAX_YR_LEN, *, context):
        super().__init__()
        if forward_rate is None:
            self.time_len = time_len
            self.forward_rate = Parameter(Tensor(self.time_len))
            self.reset_parameters()
        else:
            assert forward_rate.dim() == 1
            self.time_len = forward_rate.nelement()
            self.forward_rate = Parameter(forward_rate)
        self.context = context

    def reset_parameters(self):
        self.forward_rate.data.uniform_(self._RATE_INIT_UPPER_LIMIT,
                                        self._RATE_INIT_UPPER_LIMIT)

    def set_parameter(self, forward_rate):
        if isinstance(forward_rate, Tensor):
            if forward_rate.dim() == 1:
                self.forward_rate.data.fill_(forward_rate)
            else:
                self.forward_rate.data.set_(forward_rate)
        elif isinstance(forward_rate, np.ndarray):
            self.forward_rate.data.set_(torch.from_numpy(
                forward_rate * np.ones(self.time_len)).double())
        else:
            self.forward_rate.data.fill_(forward_rate)

    def forward(self, cf, pos=1., weight=None, bias=None):
        forward_rate = 1. + sens(self.forward_rate, weight, bias)
        discount_rate_divider = torch.cumprod(forward_rate, 0)
        return cf / discount_rate_divider / torch.pow(forward_rate, pos - 1.)


class ConstantDiscountRateLayer(DiscountLayer):

    def __init__(self,  forward_rate, time_len=MAX_YR_LEN, *, context):
        Module.__init__(self)
        self.time_len = time_len
        if isinstance(forward_rate, float):
            forward_rate = torch.tensor([forward_rate])
        self._forward_rate = Parameter(forward_rate)
        self.forward_rate = self._forward_rate.expand(time_len)
        self.context = context
