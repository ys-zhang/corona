from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from ...conf import MAX_YR_LEN
from ...utils import repeat, time_slice, make_parameter

__all__ = ['Probability', 'Inevitable']


class Probability(Module):
    """ Class represents a Probability Table

    Attributes:
        - :attr:`name` (str)
           the name of Table.
        - :attr:`qx` (:class:`~torch.nn.Parameter`)
           annual probability with rows as sex, columns as age
        - :attr:`kx` (:class:`~torch.nn.Parameter`)

    """
    SEX_IDX, AGE_IDX, DUR_IDX = 0, 1, 4
    SUPPORTED_SENS_TYPES = frozenset({'table', 'aftMonthly', 'aftDur'})

    name: Optional[str]
    qx: torch.Tensor
    kx: Optional[torch.Tensor]
    sens_model: Optional[torch.nn.Module]

    def __init__(self, qx=None, kx=None, sens_model=None, sens_type=None, *, name=None):
        """

        :param qx: tensor represents the probability
        :param kx: ratio of
        :param sens_model:
        :param sens_type:
        :param name: name of the Probability
        """
        super().__init__()
        assert sens_type is None or sens_type in self.SUPPORTED_SENS_TYPES
        self.name = name
        self.sens_model = sens_model
        self.sens_type = sens_type
        if qx is None:
            self.qx = Parameter(torch.zeros(2, MAX_YR_LEN, dtype=torch.double))
        else:
            self.qx = make_parameter(qx, pad_n_col=MAX_YR_LEN, pad_mode=1)
        if kx is None:
            self.register_parameter('kx', None)
        else:
            self.kx = make_parameter(kx, pad_n_col=MAX_YR_LEN, pad_mode=1)

    def sens_qx(self):
        return self.sens_model(self.qx) if self.sens_type == 'table' else self.qx

    def monthly_probs(self):
        """ Monthly version of the probability tables

        :returns: qx_mth, kx_mth
        """
        qx_mth = repeat(torch.pow(self.sens_qx() + 1., 1. / 12) - 1., 12)
        if self.kx is not None:
            kx_mth = repeat(self.kx, 12)
        else:
            kx_mth = None
        return self.sens_model(qx_mth) if self.sens_type == 'aftMonthly' else qx_mth, kx_mth

    def set_parameter(self, qx, kx=None):
        """Set Parameter with new tensor value

        :param Tensor qx:
        :param Tensor kx:
        """
        self.qx.data.set_(qx)
        if kx and self.kx is not None:
            self.kx.data.set_(kx)
        return self

    def new_with_kx(self, kx, name=None):
        """ Create a new instance of Probability with same qx but new kx

        :param Union[Tensor, Parameter, ndarray, list] kx: new kx
        :param str name: name of new Probability
        :rtype: Probability
        """
        if name is None:
            name = self.name
        return Probability(self.qx, kx, self.sens_model, self.sens_type, name=name)

    def new_with_sens_model(self, sens_model, sens_type, name=None):
        if name is None:
            name = self.name
        return Probability(self.qx, self.kx, sens_model, sens_type, name=name)

    def forward(self, mp_idx, mp_val, annual=False)->Tensor:
        if annual:
            qx = self.sens_qx()
            kx = self.kx
        else:
            qx, kx = self.monthly_probs()
        try:
            qx = qx * (1. - kx)
        except TypeError:
            pass
        sex = mp_idx[:, self.SEX_IDX].long()
        age = mp_idx[:, self.AGE_IDX].long()
        if annual:
            index = age
        else:
            dur = mp_idx[:, self.DUR_IDX]
            index = age * 12 + dur
        qx = qx.index_select(0, sex)
        if self.sens_type == 'aftDur':
            return self.sens_model(time_slice(qx, index, 0.))
        else:
            return time_slice(qx, index, 0.)

    def extra_repr(self):
        return self.name


class Inevitable(Module):

    def __init__(self):
        super().__init__()

    def forward(self, mp_idx, mp_val, annual=False):
        return mp_val.new_full((1,), 1)
