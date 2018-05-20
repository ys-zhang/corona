import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np


class Repeat(torch.autograd.Function):
    """Repeat a tensor

    * :attr:`ts_input`: (Tensor) the input tensor
    * :attr:`times`: (int) repeat how many times default 12
    * :attr: `axis`: (int) axis along which to repeat values default -1

    >>> Repeat.apply(torch.range(0, 1), 2)
    """
    @staticmethod
    def forward(ctx, ts_input: Tensor, times: int=12, axis=-1):
        ctx.split_times = ts_input.shape[axis]
        ctx.axis = axis
        np_input = ts_input.detach().numpy()
        np_output = np_input.repeat(times, axis)
        return torch.from_numpy(np_output)

    @staticmethod
    def backward(ctx, grad_output):
        return torch.stack(grad_output.split(ctx.split_times, ctx.axis))\
            .sum(0), None, None


repeat = Repeat.apply
"""repeat tensor for `times` along `axis`

Args:
    - ts_input: Tensor
    - times: default 12
    - axis: default -1
"""

def repeat_array(array: Tensor, repeats: int=12)->Tensor:
    """Repeat a 1-D array for several times

    :param array: tensor to be repeated
    :param repeats: time to be repeated
    :return: repeated array

    >>> repeat_array(torch.range(1, 2), 3)
    tensor([ 1,  1,  1,  2,  2,  2])
    """
    n = array.nelement()
    return array.unsqueeze(1).expand(n, repeats)\
        .contiguous().view(n * repeats)


def repeat_table(table: Tensor, repeats: int=12)->Tensor:
    """Repeat a 2-D tensor in column

    >>> import numpy as np
    >>> repeat_table(torch.from_numpy(np.array([[2, 3], [1, 5]])), 2)
    tensor([[ 2,  2,  3,  3].
            [ 1,  1,  5,  5]])
    """
    shape = table.shape
    return table.unsqueeze(2).expand(*shape, repeats).contiguous()\
        .view(shape[0], shape[1] * repeats)


def sens(ts: Tensor, weight=None, bias=None)->Tensor:
    r""" Affine function for sensitive test.
    if dimension of weight is 1 then `weight` is point wise multiplied to `ts`
    else `torch.nn.functional.linear` is used.

    .. math::

        \text{out}_{ij} = \text{ts}_{ij} * \text{weight}_i + \text{bias}_i

        \text{out} = \text{weight}\text{ts} + \text{bias}

    :param ts: original tensor
    :param weight: multiplier
    :param bias: bias part
    """

    rst = ts
    if weight:
        if weight.dim() == 1:
            rst = rst * weight + bias if bias else weight * rst
        else:
            rst = F.linear(ts, weight, bias)
    elif bias:
        rst += bias
    return rst


def time_slice1d(ts: Tensor, aft, pad_value):
    r"""
    .. math::

        $$ out_i = ts_{i + aft} * \chi_{i + aft < n} +
               pad_value * \chi_{i + aft \ge n} $$

    """
    if aft > 0:
        return torch.cat((ts[aft:], torch.full((aft,), pad_value)))
    else:
        return ts


def time_slice(tbl: Tensor, aft: Tensor, pad_value=0)->Tensor:
    r"""
    .. math::

        $$ out_{i, j} = ts_{i + aft, j} * \chi_{j + aft < n} +
               pad_value * \chi_{j + aft \ge n} $$

    """
    aft = aft.long()
    ts_lst = torch.unbind(tbl, 0)
    aft_lst = aft.tolist()
    return torch.stack([time_slice1d(ts, i, pad_value)
                        for ts, i in zip(ts_lst, aft_lst)], 0)


def time_trunc1d(ts: Tensor, aft, fill)->Tensor:
    r"""
    .. math::

        $$ out_i = ts_i * \chi_{i < aft} +
               fill * \chi_{i \ge aft } $$

    if fill is None then :math:`fill = out_{aft-1}`
    """
    rst = ts.clone()
    rst[aft:] = fill if fill is not None else rst[aft-1]
    return rst


def time_trunc(tbl: Tensor, aft: Tensor, fill=0)->Tensor:
    r"""
    .. math::

        $$ out_{i, j} = ts_{i, j} * \chi_{j < aft} +
               fill * \chi_{j \ge aft } $$

    if fill is None then :math:`fill_i = out_{i, aft-1}`
    """
    aft = aft.long()
    aft_lst = aft.tolist()
    ts_lst = torch.unbind(tbl, 0)
    return torch.stack([time_trunc1d(ts, i, fill)
                        for ts, i in zip(ts_lst, aft_lst)], 0)


class WaitingPeriod(nn.Module):
    """ Represents Waiting Period for liabilities.
    first `period_in_mth` columns of each input in `inputs` is annihilated.

    Construction:
        :parameter int period_in_mth: length of the waiting period
            in number of months

    Inputs:
        :parameter Tensor *inputs: 2-D Tensors
    """
    def __init__(self, period_in_mth: int):
        super().__init__()
        assert isinstance(period_in_mth, int)
        self.period_in_mth = period_in_mth

    def forward(self, *seq):
        if len(seq) > 1:
            seq = [x.copy() for x in seq]
            for x in seq:
                x[:, :self.period_in_mth] = 0
            return seq
        else:
            rst = seq[0].copy()
            rst[:, :self.period_in_mth] = 0
            return rst


class Flip(torch.autograd.Function):
    """ flip a tensor
    The dim of the tensor can only be 1 or 2.
    if dim > 1 then flip the last axis
    """
    @staticmethod
    def _flip(array):
        return np.flip(array, -1).copy()

    @staticmethod
    def forward(ctx, ts_input: Tensor):
        np_input = ts_input.detach().numpy()
        return torch.from_numpy(Flip._flip(np_input))

    @staticmethod
    def backward(ctx, grad_output):
        return torch.from_numpy(Flip._flip(grad_output.detach().numpy()))


flip = Flip.apply
"""Flip a tensor
The dim of the tensor can only be 1 or 2.
if dim == 2 then flip each row
"""


class CF2M(nn.Module):
    r"""Convert an annual cash flow to monthly, by repeating or insert into
    a matrix of zeros

    if `only_at_month` is `None` (default), the cash flow is repeated 12 times
    along column, else only the column mode 12 equals `only_at_month` is not
    zero, which means `only_at_month` can only be from 0 to 11.

    .. math::

        $$ output_{i, j} = input_{i, j // 12} $$
        $$ output_{i, j} = \big{\chi}_{j \equiv \text{onlyAtMonth}} *
            input_{i, j // 12} $$


    .. :attr::`only_at_month`

        if not None, then result will be zero at all months except the value of
        this attribute.

    """
    def __init__(self, only_at_month=None):
        super().__init__()
        self.only_at_month = only_at_month

    def forward(self, table):
        if self.only_at_month is None:
            return repeat(table)
        else:
            raise NotImplementedError(self.only_at_month)
