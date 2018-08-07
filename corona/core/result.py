"""
Classes of results returned by contract related Modules
"""
import collections
import sqlite3
from typing import Iterable

import numpy as np
import pandas as pd
import cytoolz
from torch import Tensor
from corona.conf import MAX_MTH_LEN, MAX_YR_LEN


class CashFlow:
    """Cash Flow Data, return type of :class:`~core.contract.Clause`

    Attributes:

        :attr:`cf` Blank Cash Flow
        :attr:`p` probability of the cash flow
        :attr:`qx` probability of kick out of the contract
        :attr:`lx` in force number before the cash flow happens
        :attr:`base`: value of the base of the cashflow
        :attr:`ratio`: value of the ratio of the cashflow
        :attr:`meta_data` meta_data of the Clause instance generated the cash flow
        :attr:`shape` shape of the cash flow
    """

    __slots__ = ("cf", "p", "qx", 'lx', 'base', 'ratio', 'meta_data', 'mp_idx', 'mp_val')

    def __init__(self, cf, p, qx, lx, base, ratio, meta_data, mp_idx=None, mp_val=None):
        self.cf = cf
        self.p = p
        self.qx = qx
        self.lx = lx
        self.base = base
        self.ratio = ratio
        self.meta_data: dict = meta_data
        self.mp_idx = mp_idx
        self.mp_val = mp_val

    def __getitem__(self, item):
        return CashFlow(self.cf[item], self.p[item], self.qx[item], self.lx[item],
                        self.base[item], self.ratio[item], self.meta_data.copy(),
                        self.mp_idx[item[0], :], self.mp_val[item[0], :])

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
        base_val = (self.base.detach().numpy() if isinstance(self.base, Tensor) else self.base) * broadcaster
        ratio_val = (self.ratio.detach().numpy() if isinstance(self.ratio, Tensor) else self.ratio) * broadcaster
        if convert_mp:
            mp_idx = self.mp_idx.detach().numpy()
            mp_val = self.mp_val.detach().numpy()
        else:
            mp_idx = self.mp_idx
            mp_val = self.mp_val
        return CashFlow(cf, p, qx, lx, base_val, ratio_val,
                        self.meta_data.copy(), mp_idx, mp_val)

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
            [['cf', 'p', 'qx', 'lx', 'icf', 'base_val', 'ratio_val'], range(mp_num), [self.context]],
            names=('val', 'mp', 'context')
        )
        return pd.DataFrame(
            np.hstack([cf, p, qx, lx, icf, base_val, ratio_val]),
            index=row_idx, columns=col_idx
        ).stack([2, 1]).sort_index(level=[-1, 0])

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
                            self.p, self.qx, self.lx, self.base, self.ratio,
                            meta_data, self.mp_idx, self.mp_val)
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
                            self.p, self.qx, self.lx, self.base, self.ratio,
                            meta_data, self.mp_idx, self.mp_val)
        else:
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

    __slots__ = ('results', '_lx', 'qx', '_mp_idx', '_mp_val', 'msc_results')

    def __init__(self, results, qx=None, lx=1, mp_idx=None, mp_val=None, msc_results=None):
        """

        :param results: OrderedDict of :class:`CashFlow`s or :class:`GResults`.
        :type results: OrderedDict
        :param Tensor qx: equivalent qx if the GResult object is treated as a single entity
        :param float or Tensor lx: number of in force before any of the cash flows of this GResult happens.
        :param Tensor mp_idx: index part of model point
        :param Tensor mp_val: value part of model point
        :param dict msc_results: dict of miscellaneous results, for example account value.
        """
        self.results = results  # type: collections.OrderedDict
        self._lx = lx
        self.qx = qx
        self.msc_results = {} if msc_results is None else msc_results
        self.mp_idx = mp_idx
        self.mp_val = mp_val

    @property
    def mp_idx(self)->Tensor:
        """ index part of the input model points """
        return self._mp_idx

    @mp_idx.setter
    def mp_idx(self, idx):
        if idx is not None:
            self._mp_idx = idx
            for v in self.values():
                v.mp_idx = idx

    @property
    def mp_val(self)->Tensor:
        """ value part of the input model points """
        return self._mp_val

    @mp_val.setter
    def mp_val(self, val):
        if val is not None:
            self._mp_val = val
            for v in self.values():
                v.mp_val = val

    @property
    def lx(self):
        """ number of in force just before the Contract or ClauseGroup which generates the Group Result """
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

    def to_sql(self, conn, val_date, batch_id=0):
        """ save the FResult to a sql database

        :param conn: connection of a database, see pandas.to_sql
        :param val_date: valuation date
        :param batch_id: the batch identifier used to identify data in the database
        """
        def add_batch_id(x):
            x['batch_id'] = batch_id
            return x.set_index('batch_id', append=True)
        for cf in self.cashflows():
            df = add_batch_id(cf.dataframe(val_date, 'date'))
            df.to_sql(cf.name, conn, if_exists='append')
        cf_meta_date = cytoolz.merge_with(np.array, *[v.meta_data for v in self.cashflows()])
        meta_frame = add_batch_id(pd.DataFrame.from_dict(cf_meta_date)).set_index('name', append=True)
        meta_frame.to_sql('cashflow_meta_data', conn, if_exists='append')

    def to_sqlite(self, path, val_date, batch_id=0):
        with sqlite3.connect(path) as conn:
            self.to_sql(conn, val_date, batch_id)

    def to_msgpack(self, val_date, batch_id=0, mode='flat', *, path=None):
        """Save the result to a message pack file

        :param Union[date, str] val_date: valuation date of the
        :param int batch_id: which batch the input model point is.
        :param str mode: mode the result will be saved
        :param str path: the path of the msg pack will be saved to
        :return:
        """
        assert mode == 'flat'
        import msgpack
        d = {
            'meta' : {
                'val_date': val_date,
                'batch_id': batch_id,
                'mode'    : mode, },
            'value': [self]
        }
        if path:
            with open(path, 'w+b') as file:
                msgpack.dump(d, file, default=encoder)
        else:
            return msgpack.packb(d, default=encoder)

    def to_hdf(self, val_date, batch_id=0, mode='flat', *, path=None):
        assert mode == 'flat'
        import h5py

        if path is None:
            path = f"{val_date}_batch{batch_id}.h5"

        def add_attr(file, d: dict):
            for k, v in d.items():
                file.attrs[k] = v

        def save_mp(file: h5py.File):
            gp = file.create_group('mp')
            mp_idx = self.mp_idx.detach().numpy()
            mp_val = self.mp_val.detach().numpy()
            gp.create_dataset('mp_idx', data=mp_idx, compression="gzip", compression_opts=9)
            gp.create_dataset('mp_val', data=mp_val, compression="gzip", compression_opts=9)

        def save_cashflow(gp: h5py.Group, cf: CashFlow):
            name = cf.name
            cf_gp = gp.create_group(name)
            add_attr(cf_gp, cf.meta_data)
            cf_gp.create_dataset('cf', data=np.array(encoder(cf.cf)),
                                 compression="gzip", compression_opts=9)
            cf_gp.create_dataset('p', data=np.array(encoder(cf.p)),
                                 compression="gzip", compression_opts=9)
            cf_gp.create_dataset('base', data=np.array(encoder(cf.base)),
                                 compression="gzip", compression_opts=9)
            cf_gp.create_dataset('ratio', data=np.array(encoder(cf.ratio)),
                                 compression="gzip", compression_opts=9)

        with h5py.File(path, 'w') as f:
            # add meta information
            meta = {
                'val_date': val_date,
                'batch_id': batch_id,
                'mode'    : mode,
            }
            add_attr(f, meta)
            save_mp(f)
            group = f.create_group('cashflows')
            for cashflow in self.cashflows():
                save_cashflow(group, cashflow)

    def to_excel(self, mp, val_date=None, path=None):
        assert mp is not None
        rst = {}
        def get_mp(val_, mp_):
            try:
                return encoder(val_[mp_, :])
            except TypeError:
                return encoder(val_)
            except IndexError:
                return encoder(val_[mp_])
        for cf in self.cashflows():
            rst[cf.name] = {
                'base' : get_mp(cf.base, mp),
                'ratio': get_mp(cf.ratio, mp),
                'cf'   : get_mp(cf.cf, mp),
                'lx'   : get_mp(cf.lx, mp),
                'qx'   : get_mp(cf.qx, mp),
                'p'    : get_mp(cf.p, mp),
                'icf'  : get_mp(cf.icf, mp),
            }

        base = pd.DataFrame.from_dict({k: v['base'] for k, v in rst.items()})
        ratio = pd.DataFrame.from_dict({k: v['ratio'] for k, v in rst.items()})
        cf = pd.DataFrame.from_dict({k: v['cf'] for k, v in rst.items()})
        qx = pd.DataFrame.from_dict({k: v['qx'] for k, v in rst.items()})
        lx = pd.DataFrame.from_dict({k: v['lx'] for k, v in rst.items()})
        p = pd.DataFrame.from_dict({k: v['p'] for k, v in rst.items()})
        icf = pd.DataFrame.from_dict({k: v['icf'] for k, v in rst.items()})

        pd.concat()


def encoder(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, Tensor):
        return obj.tolist()
    elif isinstance(obj, CashFlow):
        return {
            '__type__': 'CashFlow',
            'meta'    : {k: encoder(v) for k, v in obj.meta_data.items()},
            'cf'      : encoder(obj.cf),
            'p'       : encoder(obj.p),
            'qx'      : encoder(obj.qx),
            'base'    : encoder(obj.base),
            'ratio'   : encoder(obj.ratio),
        }
    elif isinstance(obj, GResult):
        return {
            '__type__'   : 'GResult',
            'msc_results': {k: encoder(v) for k, v in obj.msc_results.items()},
            'lx'         : encoder(obj.lx),
            'qx'         : encoder(obj.qx),
            'mp_idx'     : encoder(obj.mp_idx),
            'mp_val'     : encoder(obj.mp_val),
            'cashflows'  : [encoder(cf) for cf in obj.cashflows()],
        }
    else:
        return obj


def msgpack_decoder(obj):
    if b'__type__' in obj:
        return msgpack_decoder.decoder_dict[obj['__type__']](obj)
    elif isinstance(obj, list):
        return np.array(obj)
    else:
        return obj


msgpack_decoder.decoder_dict = {

}
