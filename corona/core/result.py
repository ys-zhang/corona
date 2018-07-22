"""
Classes of results returned by contract related Modules
"""
import collections
import sqlite3
from typing import Optional, Union, Iterable

import numpy as np
import pandas as pd
import cytoolz
from torch import Tensor

from corona.mp import ModelPointSet
from corona.conf import MAX_MTH_LEN, MAX_YR_LEN
from corona.utils import time_slice, numpy_row_slice


class CashFlow:
    """Cash Flow Data, return type of :class:`~core.contract.Clause`

    Attributes:

        :attr:`cf` Blank Cash Flow
        :attr:`p` probability of the cash flow
        :attr:`qx` probability of kick out of the contract
        :attr:`lx` in force number before the cash flow happens
        :attr:`base_val`: value of the base of the cashflow
        :attr:`ratio_val`: value of the ratio of the cashflow
        :attr:`meta_data` meta_data of the Clause instance generated the cash flow
        :attr:`shape` shape of the cash flow
    """

    cf: Union[Tensor, np.ndarray]
    p: Union[Tensor, np.ndarray]
    lx: Union[Tensor, np.ndarray]
    base_val: Union[Tensor, np.ndarray]
    ratio_val: Union[Tensor, np.ndarray]
    meta_data: dict
    mp_idx: Optional[Tensor]
    mp_val: Optional[Tensor]
    _future_sliced: bool

    __slots__ = ("cf", "p", "qx", 'lx', 'base_val',
                 'ratio_val', 'annual',
                 'meta_data', 'mp_idx', 'mp_val', '_future_sliced')

    def __init__(self, cf, p, qx, lx, base, ratio, annual, meta_data,
                 mp_idx=None, mp_val=None, future_sliced=False):
        self.cf = cf
        self.p = p
        self.qx = qx
        self.lx = lx
        self.base_val = base
        self.ratio_val = ratio
        self.annual = annual
        self.meta_data = meta_data
        self.mp_idx = mp_idx
        self.mp_val = mp_val
        self._future_sliced = future_sliced

    @property
    def future_sliced(self):
        return self._future_sliced

    @future_sliced.setter
    def future_sliced(self, val):
        assert val or (not self._future_sliced)
        self._future_sliced = val

    def slice_future_(self):
        if self.future_sliced:
            return self
        dur = self.mp_idx[:, 4]
        def trunc(var):
            if isinstance(var, Tensor) and var.dim() == 2:
                return time_slice(var, dur)
            elif isinstance(var, np.ndarray):
                try:
                    return numpy_row_slice(var, dur.detach().numpy(), self.annual)
                except AttributeError:
                    return numpy_row_slice(var, dur, self.annual)
            else:
                return var
        self.cf = trunc(self.cf)
        self.p = trunc(self.p)
        self.qx = trunc(self.qx)
        self.lx = trunc(self.lx)
        self.base_val = trunc(self.base_val)
        self.ratio_val = trunc(self.ratio_val)
        self.future_sliced = True
        return self

    def __getitem__(self, item):
        return CashFlow(self.cf[item], self.p[item],
                        self.qx[item], self.lx[item], 
                        self.base_val[item], self.ratio_val[item],
                        self.annual, self.meta_data.copy(),
                        self.mp_idx[item[0], :], self.mp_val[item[0], :],
                        self.truncated)

    def __repr__(self):
        return self.dataframe().__repr__()

    @property
    def shape(self):
        mp, _ = self.mp_idx.shape
        t = MAX_YR_LEN if self.annual else MAX_MTH_LEN
        return mp, t

    @property
    def t_offset(self):
        return self.meta_data['t_offset']
    
    @t_offset.setter
    def t_offset(self, offset):
        self.meta_data['t_offset'] = offset

    def numpy(self, convert_mp=True):
        broadcaster = np.ones(self.shape)
        cf = (self.cf.detach().numpy() if isinstance(self.cf, Tensor) else self.cf) * broadcaster
        p = (self.p.detach().numpy() if isinstance(self.p, Tensor) else self.p) * broadcaster
        qx = (self.qx.detach().numpy() if isinstance(self.qx, Tensor) else self.qx) * broadcaster
        lx = (self.lx.detach().numpy() if isinstance(self.lx, Tensor) else self.lx) * broadcaster
        base_val = (self.base_val.detach().numpy() if isinstance(self.base_val, Tensor) else self.base_val) \
            * broadcaster
        ratio_val = (self.ratio_val.detach().numpy() if isinstance(self.ratio_val, Tensor) else self.ratio_val) \
            * broadcaster
        if convert_mp:
            mp_idx = self.mp_idx.detach().numpy()
            mp_val = self.mp_val.detach().numpy()
        else:
            mp_idx = self.mp_idx
            mp_val = self.mp_val
        return CashFlow(cf, p, qx, lx, base_val, ratio_val, self.annual,
                        self.meta_data.copy(), mp_idx, mp_val, self._future_sliced)

    def dataframe(self, val_date=None, date_type='date')->pd.DataFrame:
        cash_flow = self.numpy(convert_mp=False).slice_future_()
        mp_num, n_col = self.shape
        freq = 'Y' if self.annual else 'M'
        if val_date is None:
            row_idx = range(n_col)
        else:
            if date_type.lower() == 'period':
                row_idx = pd.period_range(val_date, periods=n_col, freq=freq, name='t')
            elif date_type.lower() == 'date':
                row_idx = pd.date_range(val_date, periods=n_col, freq=freq, name='t')
            else:
                raise ValueError(date_type)
        cf = cash_flow.cf.T
        p = cash_flow.p.T
        qx = cash_flow.qx.T
        lx = cash_flow.lx.T
        icf = cash_flow.icf.T
        base_val = cash_flow.base_val.T
        ratio_val = cash_flow.ratio_val.T
        col_idx = pd.MultiIndex.from_product(
            [['cf', 'p', 'qx', 'lx', 'icf', 'base_val', 'ratio_val'], range(mp_num)],
            names=('val', 'mp')
        )
        return pd.DataFrame(
            np.hstack([cf, p, qx, lx, icf, base_val, ratio_val]),
            index=row_idx, columns=col_idx
        ).swaplevel(axis=1).sort_index(axis=1).stack(0).sort_index(level='mp').swaplevel()

    def __getattr__(self, item):
        try:
            return self.meta_data[item]
        except KeyError:
            raise AttributeError(item)

    def cashflows(self):
        """ for compatibility of :attr:`GResult.cashflows` """
        yield self

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
            meta_data = self.meta_data.copy()
            meta_data['t_offset'] = 0
            return CashFlow(self.cf / forward_rate.add(1).pow(self.t_offset),
                            self.p, self.qx, self.lx, self.base_val,
                            self.ratio_val, self.annual, meta_data,
                            self.mp_idx, self.mp_val, self._future_sliced)
        else:
            return self

    def full_offset(self, forward_rate):
        """ Equivalent CashFlow with t_offset=1

        :param Tensor forward_rate: discount rate
        :rtype: CashFlow
        """
        if self.t_offset != 1:
            meta_data = self.meta_data.copy()
            meta_data['t_offset'] = 1
            return CashFlow(self.cf * forward_rate.add(1).pow(1 - self.t_offset),
                            self.p, self.qx, self.lx, self.base_val,
                            self.ratio_val, self.annual, meta_data,
                            self.mp_idx, self.mp_val, self._future_sliced)
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
    _mp_val: Tensor
    _mp_idx: Tensor
    __slots__ = ('results', '_lx', 'qx', '_mp_idx', '_mp_val', '_future_sliced',
                 'msc_results')

    def __init__(self, results, qx=None, lx=1, mp_idx=None, mp_val=None,
                 future_sliced=False, msc_results=None):
        """

        :param results: OrderedDict of :class:`CashFlow`s or :class:`GResults`.
        :type results: OrderedDict
        :param Tensor qx: equivalent qx if the GResult object is treated as a single entity
        :param float or Tensor lx: number of in force before any of the cash flows of this GResult happens.
        :param dict msc_results: dict of miscellaneous results, for example account value.
        """
        self.results = results  # type: collections.OrderedDict
        self._lx = lx
        self.qx = qx
        self.msc_results = {} if msc_results is None else msc_results
        self.mp_idx = mp_idx
        self.mp_val = mp_val
        self._future_sliced = future_sliced

    @property
    def mp_idx(self):
        """ index part of the input model points

        :rtype: Tensor
        """
        return self._mp_idx

    @mp_idx.setter
    def mp_idx(self, idx):
        if idx is not None:
            self._mp_idx = idx
            for v in self.values():
                v.mp_idx = idx

    @property
    def mp_val(self):
        """ value part of the input model points

        :rtype: Tensor
        """
        return self._mp_val

    @mp_val.setter
    def mp_val(self, val):
        if val is not None:
            self._mp_val = val
            for v in self.values():
                v.mp_val = val

    @property
    def lx(self):
        return self._lx

    @lx.setter
    def lx(self, lx):
        _lx = self._lx
        self._lx = lx
        for v in self.results.values():  # type: GResult or CashFlow
            v.lx = v.lx * lx / _lx

    @property
    def future_sliced(self):
        return self._future_sliced

    @future_sliced.setter
    def future_sliced(self, val: bool):
        assert val or (not self._future_sliced)
        for v in self.values():
            v.future_sliced = val

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

    def slice_future_(self):
        if self.future_sliced:
            return self
        for cf in self.cashflows():
            cf.slice_future_()
        self.future_sliced = True
        return self

    def __getitem__(self, item):
        return self.results[item]

    def __setitem__(self, key, value):
        self.results[key] = value

    def __getattr__(self, item):
        try:
            return self.results[item]
        except KeyError:
            try:
                return self.msc_results[item]
            except KeyError:
                raise AttributeError(item)

    def keys(self):
        return self.results.keys()

    def values(self):
        return self.results.values()

    def items(self):
        return self.results.items()

    def cashflows(self)->Iterable[CashFlow]:
        """ Generator of cashflows in the GResult
        """
        for v in self.values():
            yield from v.cashflows()

    def __iter__(self):
        raise NotImplemented

    def flat(self, val_date=None):
        """ Flat result of the collection
        
        :rtype: FResult
        """
        if self.mp_idx is not None and self.mp_val is not None:
            return FResult(self, val_date=val_date)
        else:
            raise RuntimeError('mp info missing')


class FResult(collections.OrderedDict):
    """ Container to store a **flat** collection of :class:`CashFlow`s with the name of
    the clause generates the cash flow as the key. The internal data are numpy ndarrays
    """

    MODEL_POINT_DATA_SET_CLASS = ModelPointSet

    def __init__(self, g_result, val_date=None):
        """
        :param GResult g_result: g_result
        """
        super().__init__(self._make_iter(g_result))
        self._mp_idx = g_result.mp_idx.detach().numpy()
        self._mp_val = g_result.mp_val.detach().numpy()
        self.val_date = val_date
        self.msc_results = dict((k, v.detach().numpy()) for k, v in g_result.msc_results.items())

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            try:
                return self.msc_results[item]
            except KeyError:
                raise AttributeError(item)

    @staticmethod
    def _make_iter(raw_result):
        for i, v in raw_result.items():
            if isinstance(v, GResult):
                yield from FResult._make_iter(v)
            elif isinstance(v, CashFlow):
                yield i, v.numpy()
            else:
                raise TypeError(v)

    @property
    def cf(self):
        return collections.OrderedDict(((k, v.cf) for k, v in self.items()))

    def pcf(self, value_only=False):
        if value_only:
            return [v.pcf for v in self.values()]
        else:
            return collections.OrderedDict(((k, v.pcf) for k, v in self.items()))

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

    def icf(self, value_only=False):
        if value_only:
            return [v.icf for v in self.values()]
        else:
            return collections.OrderedDict(((k, v.icf) for k, v in self.items()))

    def lx(self, value_only=False):
        if value_only:
            return [v.lx for v in self.values()]
        else:
            return collections.OrderedDict(((k, v.lx) for k, v in self.items()))

    @property
    def meta_data(self)->dict:
        """ meta_data of contained cashflows in the FResult
        """
        return cytoolz.merge_with(np.array, *[v.meta_data for v in self.values()])
    
    def to_sql(self, conn, val_date):
        """ save the FResult to a sql database

        :param conn: connection of a database, see pandas.to_sql
        :param val_date: valuation date
        """
        for var_name, cf in self.items():
            df = cf.dataframe(val_date, 'date')
            df.to_sql(var_name, conn)
        pd.DataFrame.from_dict(self.meta_data).to_sql('meta_data', conn)

    def to_sqlite(self, path, val_date):
        with sqlite3.connect(path) as conn:
            self.to_sql(conn, val_date)
