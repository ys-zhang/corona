import weakref
import re
import enum
import warnings

import torch
import collections
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np
from cytoolz import isiterable, keyfilter, valmap


class CashFlow:
    """Cash Flow Data, return type of :class:`~core.contract.Clause`

    Attributes:

        :attr:`cf` Blank Cash Flow
        :attr:`p` probability of the cash flow
        :attr:`qx` probability of kick out of the contract
        :attr:`lx` in force number before the cash flow happens
        :attr:`t_offset` time offset to begining of each time interval

    """

    __slots__ = ("cf", "p", "qx", 'lx', 't_offset')

    def __init__(self, cf, p, qx, lx, t_offset):
        self.cf = cf
        self.p = p
        self.qx = qx
        self.lx = lx
        self.t_offset = t_offset

    def __getitem__(self, item):
        return CashFlow(self.cf[item], self.p[item],
                        self.qx[item], self.lx[item], self.t_offset)

    @property
    def pcf(self):
        """ cf * p

        :rtype: Tensor
        """
        return self.cf * self.p

    @property
    def icf(self):
        """ cf * p * lx

        :rtype: Tensor
        """
        return self.pcf * self.lx

    @property
    def lx2(self):
        """ (1 - qx) * lx

        :rtype: Tensor
        """
        return self.lx * self.px

    @property
    def px(self):
        """ 1 - qx

        :rtype: Tensor
        """
        return 1 - self.qx

    @px.setter
    def px(self, px):
        self.qx = 1 - px

    def lx_mul_(self, ts):
        """ Make CashFlow has a interface compatible to :class:`GResult`.
        see :meth:`GResult.lx_mul_`
        """
        self.lx = self.lx * ts
        return self

    def remove_offset(self, forward_rate):
        """ Equivalent CashFlow with t_offset=0

        :param Tensor forward_rate: discount rate
        :rtype: CashFlow
        """
        if self.t_offset != 0:
            return CashFlow(self.cf / forward_rate.add(1).pow(self.t_offset),
                            self.p, self.qx, self.lx, 0)
        else:
            return self

    def full_offset(self, forward_rate):
        """ Equivalent CashFlow with t_offset=1

        :param Tensor forward_rate: discount rate
        :rtype: CashFlow
        """
        if self.t_offset != 1:
            return CashFlow(self.cf * forward_rate.add(1).pow(1 - self.t_offset),
                            self.p, self.qx, self.lx, 1)
        else:
            return self

    def remove_offset_(self, forward_rate):
        if self.t_offset != 0:
            self.cf.div_(forward_rate.add(1).pow(self.t_offset))
            self.t_offset = 0.
        return self

    def full_offset_(self, forward_rate):
        if self.t_offset != 1:
            self.cf.mul_(forward_rate.add(1).pow(1 - self.t_offset))
            self.t_offset = 1.
        return self


class GResult:
    r""" Result Type of :class:`~core.contract.ClauseGroup`.

    Attributes:
        :attr:`results`:
            An `OrderedDict` holding results of sub clause-like modules of the
            :class:`~core.contract.ClauseGroup` generated the `GResult` object
        :attr:`qx`:
            A `Tensor` represents the equivalent `qx` of the whole GResult
            treated as a single entity.
        :attr:`lx`:
            A `Tensor` represents the number of in force before any of sub results
            happens.

            .. note:

                The value of this :attr:`lx` is linked to `lx`s of its :attr:`results`
                *just **after** initialization*. Once :attr:`lx` is reset by `=`,
                `lx`s of `results` is updated by

                .. math:
                    \text{results.lx}_{\text{new}} = \text{results.lx}_{\text{old}}
                        * \text{lx}_{\text{new}} / \text{lx}_{\text{old}}

    """
    __slots__ = ('results', '_lx', 'qx')

    def __init__(self, results, qx=None, lx=1):
        """

        :param results: OrderedDict of :class:`CashFlow`s or :class:`GResults`.
        :type results: OrderedDict
        :param Tensor qx: equivalent qx if the GResult object is treated as a single entity
        :param lx: number of in force before any of the cash flows of this GResult happen
        :type lx: float or Tensor
        """
        self.results = results  # type: collections.OrderedDict
        self._lx = lx
        self.qx = qx

    @property
    def lx(self):
        return self._lx

    @lx.setter
    def lx(self, lx):
        _lx = self._lx
        self._lx = lx
        for v in self.results.values():  # type: GResult or CashFlow
            v.lx = v.lx * lx / _lx

    def lx_mul_(self, ts):
        """Multiply `lx` by `ts` equivalent to::
            g_result.lx = lx * ts

        this method is faster and returns `self`.

        :param Tensor ts: the multiplier
        :rtype: GResult
        """
        self._lx = self._lx * ts
        for v in self.results.values():  # type: GResult or CashFlow
            v.lx = v.lx * ts
        return self

    def __getitem__(self, item):
        return self.results[item]

    def __setitem__(self, key, value):
        self.results[key] = value

    def keys(self):
        return self.results.keys()

    def values(self):
        return self.results.values()

    def items(self):
        return self.results.items()

    def __iter__(self):
        return self.items()

    def flat(self):
        """ Flat result of the collection
        :rtype: FResult
        """
        return FResult(self)


class FResult(collections.OrderedDict):
    """ Container to store a **flat** collection of :class:`CashFlow`s with the name of
    the clause generates the cash flow as the key.
    """
    __slots__ = ()

    def __init__(self, raw_result):
        """

        :param OrderedDict raw_result: raw result
        :type raw_result: GResult
        """
        super().__init__(self._make_iter(raw_result))

    @staticmethod
    def _make_iter(raw_result):
        for i, v in raw_result.items():
            if isinstance(v, GResult):
                yield from FResult._make_iter(v)
            elif isinstance(v, CashFlow):
                yield i, v
            else:
                raise TypeError(v)

    def cf(self, value_only=False):
        if value_only:
            return [v.cf for v in self.values()]
        else:
            return collections.OrderedDict(((k, v.cf) for k, v in self.items()))

    def p(self, value_only=False):
        if value_only:
            return [v.p for v in self.values()]
        else:
            return collections.OrderedDict(((k, v.p) for k, v in self.items()))

    def qx(self, value_only=False):
        if value_only:
            return [v.qx for v in self.values()]
        else:
            return collections.OrderedDict(((k, v.qx) for k, v in self.items()))

    def px(self, value_only=False):
        if value_only:
            return [v.px for v in self.values()]
        else:
            return collections.OrderedDict(((k, v.px) for k, v in self.items()))

    def t_offset(self, value_only=False):
        if value_only:
            return [v.t_offset for v in self.values()]
        else:
            return collections.OrderedDict(((k, v.t_offset) for k, v in self.items()))

    def apply_(self, func):
        for k, cf in self.values():
            self[k] = func(cf)
        return self


class Repeat(torch.autograd.Function):
    """Repeat a tensor

    Inputs:

        * `ts_input`: (Tensor) the input tensor
        * `times`: (int) repeat how many times default 12
        * `axis`: (int) axis along which to repeat values default -1

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
        return torch.cat((ts[aft:], torch.full((aft,), pad_value, dtype=torch.double)))
    else:
        return ts


def time_slice(tbl: Tensor, aft: Tensor, pad_value=0.)->Tensor:
    r"""
    .. math::

        out_{i, j} = ts_{i, j + aft} * \chi_{j + aft < n} +
            pad_value * \chi_{j + aft \ge n}

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


def time_push1d(ts: Tensor, to)->Tensor:
    r"""
    .. math::

        $$ out_i = ts_{i - \text{to}} * \chi_{i \ge \text{to}}
    """
    rst = ts.new_zeros(ts.shape)
    rst[to:] = ts[:(ts.nelement()-to)]
    return rst


def time_push(tbl: Tensor, to: Tensor)->Tensor:
    to = to.long()
    to_lst = to.tolist()
    ts_lst = torch.unbind(tbl, 0)
    return torch.stack([time_push1d(ts, i)
                        for ts, i in zip(ts_lst, to_lst)], 0)


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


def account_value(a0: Tensor, rate: Tensor, after_credit: Tensor=None, before_credit: Tensor=None):
    r""" calculate account value

    .. math::

        a_{i, t+1} = (a_{i, t} + \text{before_credit}_{i, t}) * rate_{i, t}
         + \text{after_credit}_{i, t}

    :param a0: 1-D tensor represents initial av of each model point
    :param rate: 1-D tensor, credit interest of each model point at each time
    :param after_credit: increment  or reduction of account value after credit
    :param before_credit: increment  or reduction of account value
        before credit
    :return: the account value of each model point at each time,
        the first column is a0

    .. note::
        rate is directly multiplied to the account value of last month
    """
    rate = 1 + rate  # type: Tensor
    if before_credit is None:
        b = after_credit
    elif after_credit is None:
        b = before_credit * rate
    else:
        b = before_credit * rate + after_credit

    rate = torch.cat((rate.new_ones((rate.shape[0], 1)), rate), 1)[:, :-1]
    if b is not None:
        a0b = torch.cat((a0.reshape(-1, 1), b), 1)[:, :-1].unsqueeze(1)
        mask = rate.new_ones((rate.shape[1], rate.shape[1]),
                             requires_grad=False).triu().unsqueeze(0)
        cum_rate = rate.cumprod(1).unsqueeze(1)
        expanded_cum_rate = cum_rate.expand(*rate.shape, rate.shape[1])
        factor = expanded_cum_rate / cum_rate.transpose(1, 2) * mask
        return (a0b @ factor).squeeze()
    else:
        return a0.reshape(-1, 1) * rate.cumprod(1)


def make_parameter(param, *, pad_n_col=None, pad_value=None, pad_mode=None) -> nn.Parameter:
    if isinstance(param, nn.Parameter):
        return param
    elif isinstance(param, Tensor):
        pass
    elif isinstance(param, np.ndarray):
        param = torch.from_numpy(param)
    elif isinstance(param, int) or isinstance(param, float):
        param = torch.tensor([param], dtype=torch.double)
    else:
        param = torch.tensor(list(param), dtype=torch.double)
    if pad_mode is not None:
        if param.shape[1] < pad_n_col:
            param = pad(param, pad_n_col, pad_value, pad_mode)
        elif param.shape[1] > pad_n_col:
            warnings.warn("param len longer than pad_n_col", RuntimeWarning)
            param = param[:, :pad_n_col]
    return nn.Parameter(param)


def make_model_dict(modules, default_key=None):
    assert modules is None or isiterable(modules)
    if isinstance(modules, dict):
        return ModuleDict(modules, default_key=default_key)
    elif modules is not None:
        modules = dict([(p.context, p) for p in modules])
        return ModuleDict(modules, default_key)
    else:
        return ModuleDict(default_key=default_key)


class PadMode(enum.IntEnum):
    """ Different modes of padding mechanism.

    - :attr:`Constant`: 0, Pad left with constant value.
    - :attr:`LastValue`, :attr:`LastColumn`: 1, Pad left with the last Column
    - :attr:`MaxLastValueAnd`: 2 Pad left with max(lastColumn, value)
    - :attr:`MinLastValueAnd`: 3 Pad left with min(lastColumn, value)
    - :attr:`Max`: 4 Pad left with the maximum of the row
    - :attr:`Min`: 5 Pad left with the minimum of the row
    - :attr:`Average`: 6 Pad left with the average of the row
    - :attr:`Mode`: 6 Pad left with the mode of the row
    """
    Constant = 0
    """Pad left with constant value"""
    LastValue = 1
    """Pad left with the last Column"""
    LastColumn = 1
    """Same as `LastValue`"""
    MaxLastValueAnd = 2
    """Pad left with max(lastColumn, value)"""
    MinLastValueAnd = 3
    """Pad left with min(lastColumn, value)"""
    Max = 4
    """Pad left with the maximum of the row"""
    Min = 5
    """Pad left with the minimum of the row"""
    Avg = 6
    """Pad left with the average of the row"""
    Mode = 7
    """Pad left with the mode of the row"""


def pad(raw_table, n_col, pad_value, pad_mode):
    if raw_table.nelement() == 1:
        table = torch.full((1, n_col), raw_table)
    elif raw_table.dim() == 1:
        table = torch.unsqueeze(raw_table, 0)
    elif n_col and n_col > raw_table.shape[1]:
        pad = (0, n_col - raw_table.shape[1])
        if pad_mode == PadMode.Constant:
            table = F.pad(raw_table, pad, 'constant', pad_value)
        else:
            if pad_mode == PadMode.LastValue:
                pad_value = raw_table[:, -1]
            elif pad_mode == PadMode.MinLastValueAnd:
                pad_value = torch.nn.functional.threshold(raw_table[:, -1], pad_value, pad_value)
            elif pad_mode == PadMode.MaxLastValueAnd:
                pad_value = -torch.nn.functional.threshold(-raw_table[:, -1], -pad_value, -pad_value)
            elif pad_mode == PadMode.Max:
                pad_value = raw_table.max(1)[0]
            elif pad_mode == PadMode.Min:
                pad_value = raw_table.min(1)[0]
            elif pad_mode == PadMode.Avg:
                pad_value = raw_table.meam(1)
            elif pad_mode == PadMode.Mode:
                pad_value = raw_table.mode(1)[0]
            else:
                raise NotImplementedError(f'pad_mode: {pad_mode}')
            table = torch.cat((raw_table, torch.unsqueeze(pad_value, 1).expand((-1, pad[1]))), 1)
    elif n_col:
        table = raw_table[:, :n_col]
    else:
        table = raw_table
    return table


class Lambda(nn.Module):

    def __init__(self, lambd, repre=None):
        super().__init__()
        self.lambd = lambd
        self.repr = repre

    def forward(self, *inputs):
        return self.lambd(*inputs)

    def __repr__(self):
        if self.repr:
            return str(self.repr)
        else:
            return super().__repr__()


class ClauseReferable:

    def __init__(self):
        self._clause = None
        """ weak proxy of clause the ratio table belongs to  """

    @property
    def clause(self):
        return self._clause()

    # don't use property.setter
    # to avoid circular reference when nn.Model.modules() is called
    def set_clause_ref(self, clause):
        self._clause = weakref.ref(clause)


class ContractReferable:

    def __init__(self):
        self._contract = None
        """ weak ref of contract the ratio table belongs to  """

    @property
    def contract(self):
        return self._contract()

    # don't use property.setter
    # to avoid circular reference when nn.Model.modules() is called
    def set_contract_ref(self, contract):
        self._contract = weakref.ref(contract)


class ModuleDict(nn.Module):

    def __init__(self, module_dict: dict=None, default_key=None):
        super().__init__()
        self.default_key = default_key
        if module_dict is not None:
            for k, m in module_dict.items():
                self.add_module(str(k), m)

    def __getitem__(self, key):
        try:
            return self._modules[str(key)]
        except KeyError:
            return self._modules[str(self.default_key)]

    def __setitem__(self, key, module):
        assert isinstance(module, nn.Module)
        return setattr(self, str(key), module)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        return list(self._modules.keys())

    def re_select(self, patten):
        patten = re.compile(patten)
        return list(keyfilter(patten.fullmatch, self._modules).values())

    def forward(self, *args, **kwargs):
        return valmap(lambda md: md(*args, **kwargs), self._modules)
