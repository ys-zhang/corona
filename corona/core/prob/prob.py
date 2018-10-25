from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from ...conf import MAX_YR_LEN
from ...utils import repeat, time_slice, make_parameter
from ..mm import LinearSensitivity

__all__ = ['Probability', 'Inevitable', 'SelectionFactor']


class Probability(Module):
    """ Class represents a Probability Table

    Attributes:

    - :attr:`name` (str)
       the name of the Module.
    - :attr:`qx` (:class:`Parameter`)
       annual probability with rows as sex, columns as age
    - :attr:`kx` (:class:`Parameter`)
       proportion of death caused by critical illness in mortality
    - :attr:`sens_model`
       model act as a sensitive layer to probabilities
    - :attr:`sens_type`
       how :attr:`sens_model` act to probs, can be chosen from *'table', 'aftMonthly' and 'aftDur'*:
          - table: the sens_model act directly on the Parameter qx
          - aftMonthly: the sens_model act on the monthly probability
          - aftDur: the sens_model act on the monthly probability which is sliced after duration of policy
          - sele: sens_type for :class:`SelectionFactor` @prlife

    """
    SEX_IDX, AGE_IDX, DUR_IDX = 0, 1, 4
    SUPPORTED_SENS_TYPES = frozenset({'table', 'aftMonthly', 'aftDur', 'sele'})

    name: Optional[str]
    qx: torch.Tensor
    kx: Optional[torch.Tensor]
    sens_model: Optional[LinearSensitivity]
    sens_type: Optional[str]

    def __init__(self, qx=None, kx=None, sens_model: Optional[Module]=None,
                 sens_type: Optional[str]=None, *, name: Optional[str]=None):
        """

        :param qx: tensor represents the probability, annual probability with rows as sex, columns as age
        :param kx: proportion of death caused by critical illness in mortality
        :param sens_model: model act as a sensitive layer to probabilities
        :param sens_type: how `sens_model` act to probs, can be chosen from *'table', 'aftMonthly' , 'aftDur', 'sele'*:
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
        if self.sens_type == 'table':
            return self.sens_model(self.qx)
        else:
            return self.qx

    def monthly_probs(self):
        """ Monthly version of the probability tables

        :returns: qx_mth, kx_mth
        """
        qx_mth = repeat(torch.pow(self.sens_qx() + 1., 1. / 12) - 1., 12)
        if self.kx is not None:
            kx_mth = repeat(self.kx, 12)
        else:
            kx_mth = None
        if self.sens_type == 'aftMonthly':
            qx_mth = self.sens_model(qx_mth)
        return qx_mth, kx_mth

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
            return self.annual(mp_idx)
        sex = mp_idx[:, self.SEX_IDX]
        age = mp_idx[:, self.AGE_IDX]
        dur = mp_idx[:, self.DUR_IDX]
        mth = age * 12 + dur

        if self.sens_type == 'sele':
            qx_yr = self.sens_model(time_slice(self.sens_qx()[sex, :], age, 0.),  sex)
            qx = repeat(torch.pow(qx_yr + 1., 1. / 12) - 1., 12)
            if self.kx is not None:
                qx = qx * (1. - repeat(time_slice(self.kx, age, 0.), 12))
            return time_slice(qx, dur, 0.)
        elif self.sens_type == 'aftDur':
            qx, kx = self.monthly_probs()
            if kx is not None:
                qx = qx * (1. - kx)
            return self.sens_model(time_slice(qx[sex, :], mth, 0.))
        else:
            qx, kx = self.monthly_probs()
            if kx is not None:
                qx = qx * (1. - kx)
            return time_slice(qx[sex, :], mth, 0.)

    def annual(self, mp_idx):
        qx = self.sens_qx()
        kx = self.kx
        if kx is not None:
            qx = qx * (1. - kx)
        sex = mp_idx[:, self.SEX_IDX]
        age = mp_idx[:, self.AGE_IDX]
        qx = qx[sex, :]
        if self.sens_type == 'aftDur':
            return self.sens_model(time_slice(qx, age, 0.))
        elif self.sens_type == 'sele':
            return self.sens_model(time_slice(qx, age, 0.), sex=sex)
        else:
            return time_slice(qx, age, 0.)

    def extra_repr(self):
        return self.name


class Inevitable(Module):

    def __init__(self):
        super().__init__()

    def forward(self, mp_idx, mp_val, annual=False):
        return mp_val.new_full((1,), 1)


class SelectionFactor(LinearSensitivity):
    def __init__(self, weight, *, name=None):
        super().__init__(weight=weight, mm=False, name=name)

    def forward(self, base, **kwargs):
        sex = kwargs['sex']  # type: Tensor
        sele_fac = self.weight[sex, :]
        return sele_fac * base
