"""
Layer of Probability

`ProbabilityLayer`, `SelectionLayer`

"""
import torch
import numpy as np
from torch import Tensor
from torch.nn import Module, Parameter
from corona.const import MAX_YR_LEN
from corona.utils import repeat, time_slice


class ProbabilityLayer(Module):

    SEX_IDX, AGE_IDX = 0, 1

    def __init__(self, qx=None, kx=None):
        super().__init__()
        if qx is None:
            self.qx = Parameter(torch.zeros(2, MAX_YR_LEN))
        elif isinstance(qx, Parameter):
            self.qx = qx
        elif isinstance(qx, Tensor):
            self.qx = Parameter(qx)
        elif isinstance(qx, np.ndarray):
            self.qx = Parameter(torch.from_numpy(qx))
        else:
            self.qx = Parameter(torch.tensor(qx))

        if kx is None:
            self.register_parameter('kx', None)
        elif isinstance(kx, Parameter):
            self.kx = kx
        elif isinstance(kx, Tensor):
            self.kx = Parameter(kx)
        elif isinstance(kx, np.ndarray):
            self.kx = Parameter(torch.from_numpy(kx))
        else:
            self.kx = Parameter(torch.tensor(kx))

        self.qx_mth, self.kx_mth = None, None
        self.update_monthly()

    def update_monthly(self):
        self.qx_mth = repeat(torch.pow(self.qx + 1., 1. / 12) - 1., 12)
        if self.kx is not None:
            self.kx_mth = repeat(self.kx, 12)

    def set_parameter(self, px, kx=None):
        self.px.data.set_(px)
        if kx and self.kx is not None:
            self.kx.data.set_(kx)
        self.update_monthly()
        return self

    def forward(self, mp_idx, annual=False)->Tensor:
        px = self.qx if annual else self.qx_mth
        kx = self.kx if annual else self.kx_mth
        try:
            px = px * (1. - kx)
        except TypeError:
            pass
        sex = mp_idx[:, self.SEX_IDX].long()
        age = mp_idx[:, self.AGE_IDX].long()
        index = age if annual else age * 12
        px = px.index_select(sex)
        return time_slice(px, index, 0.)

    def combine_selection_factor(self, sele_layer):
        return SelectedProbabilityLayer(self, sele_layer)


class Inevitable(Module):

    def __init__(self):
        super().__init__()

    def forward(self, mp_idx, annual=False):
        return mp_idx.new_full((1,), 1).double()


class SelectionLayer(Module):

    def __init__(self, fac=None, *, context):
        super().__init__()
        self.context = context
        if fac is None:
            self.fac = Parameter(torch.zeros(2, MAX_YR_LEN))
        elif isinstance(fac, Parameter):
            self.fac = fac
        elif isinstance(fac, Tensor):
            self.fac = Parameter(fac)
        elif isinstance(fac, np.ndarray):
            self.fac = Parameter(torch.from_numpy(fac))
        else:
            self.fac = Parameter(torch.tensor(fac))

        self.fac_mth = None
        self.update_monthly()

    def update_monthly(self):
        self.fac_mth = repeat(self.fac, 12)

    def set_parameter(self, selection_factor):
        self.fac.data.set_(selection_factor)
        self.update_monthly()
        return self

    def forward(self, mp_idx, annual)->Tensor:
        fac = self.fac if annual else self.fac_mth
        sex = mp_idx[:, ProbabilityLayer.SEX_IDX].long()
        return fac[sex, :]

    def combine_prob(self, prob_layer: Module):
        return SelectedProbabilityLayer(prob_layer, self)


class SelectedProbabilityLayer(Module):

    def __init__(self, prob_layer: Module, sele_layer: Module=None):
        super().__init__()
        self.context = sele_layer.context
        self.prob_layer = prob_layer
        self.sele_layer = sele_layer

    def forward(self, mp_idx, annual):
        p = self.prob_layer(mp_idx, annual)
        try:
            s = self.sele_layer(mp_idx, annual)
            return s * p
        except TypeError:
            return p


class AbstractInForceLayer(Module):

    def forward(self, qx_lst, context)->list:
        raise NotImplementedError()