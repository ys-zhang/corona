import torch
from torch import Tensor
from torch.nn import Module, Parameter
from corona.conf import MAX_YR_LEN
from corona.utils import repeat, time_slice, make_parameter


__all__ = ['Probability', 'SelectionFactor', 'SelectedProbability',
           'Inevitable']


class Probability(Module):
    """ Class represents a Proabability Table

    Attributes:
        - :attr:`name` (str)
           the name of Table.
        - :attr:`qx` (:class:`~torch.nn.Parameter`)
           annual probability with rows as sex, columns as age
        - :attr:`kx` (:class:`~torch.nn.Parameter`)

    """
    SEX_IDX, AGE_IDX = 0, 1

    # noinspection PyArgumentList
    def __init__(self, qx=None, kx=None, *, name=''):
        super().__init__()
        self.name = name
        if qx is None:
            self.qx = Parameter(torch.zeros(2, MAX_YR_LEN, dtype=torch.double))
        else:
            self.qx = make_parameter(qx, pad_n_col=MAX_YR_LEN, pad_mode=1)

        if kx is None:
            self.register_parameter('kx', None)
        else:
            self.kx = make_parameter(kx, pad_n_col=MAX_YR_LEN, pad_mode=1)

    def monthly_probs(self):
        """ Monthly version of the probability tables

        :returns: qx_mth, kx_mth
        """
        qx_mth = repeat(torch.pow(self.qx + 1., 1. / 12) - 1., 12)
        if self.kx is not None:
            kx_mth = repeat(self.kx, 12)
        else:
            kx_mth = None
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

    def new_with_kx(self, kx, name):
        """ Create a new instance of Probability with same qx but new kx

        :param Union[Tensor, Parameter, ndarray, list] kx: new kx
        :param str name: name of new Probability
        :rtype: Probability
        """
        return Probability(self.qx, kx, name=name)

    def forward(self, mp_idx, mp_val, annual=False)->Tensor:
        if annual:
            px = self.qx
            kx = self.kx
        else:
            px, kx = self.monthly_probs()
        try:
            px = px * (1. - kx)
        except TypeError:
            pass
        sex = mp_idx[:, self.SEX_IDX].long()
        age = mp_idx[:, self.AGE_IDX].long()
        index = age if annual else age * 12
        px = px.index_select(0, sex)
        return time_slice(px, index, 0.)

    def combine_selection_factor(self, sele_layer):
        """Create a

        :param SelectionFactor sele_layer:
        :rtype: SelectedProbability
        """
        return SelectedProbability(self, sele_layer)

    def extra_repr(self):
        return self.name


class Inevitable(Module):

    def __init__(self):
        super().__init__()

    def forward(self, mp_idx, mp_val, annual=False):
        return mp_val.new_full((1,), 1)


class SelectionFactor(Module):

    def __init__(self, fac=None, *, name=''):
        super().__init__()
        self.name = name
        if fac is None:
            self.fac = Parameter(torch.zeros(2, MAX_YR_LEN))
        else:
            self.fac = make_parameter(fac)

    def monthly_factor(self):
        return repeat(self.fac, 12)

    def set_parameter(self, selection_factor):
        self.fac.data.set_(selection_factor)
        return self

    def forward(self, mp_idx, mp_val, annual)->Tensor:
        fac = self.fac if annual else self.monthly_factor()
        sex = mp_idx[:, Probability.SEX_IDX].long()
        return fac[sex, :]

    def combine_prob(self, prob_layer: Module):
        return SelectedProbability(prob_layer, self)

    def extra_repr(self):
        return self.name


class SelectedProbability(Module):

    def __init__(self, prob_layer: Module, sele_layer: Module=None):
        super().__init__()
        self.prob_layer = prob_layer
        self.sele_layer = sele_layer
        if sele_layer:
            self.name = f'{prob_layer.name}|{sele_layer.name}'
        else:
            self.name = prob_layer.name

    def extra_repr(self):
        return self.name

    def forward(self, mp_idx, mp_val, annual):
        p = self.prob_layer(mp_idx, mp_val, annual)
        try:
            s = self.sele_layer(mp_idx, mp_val, annual)
            return s * p
        except TypeError:
            return p



