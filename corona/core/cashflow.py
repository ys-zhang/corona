"""
Classes of results returned by contract related Modules
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Any
import dataclasses as dc

import torch
from torch import Tensor, tensor


__all__ = ['CashFlow']


@dc.dataclass(eq=False)
class CashFlow:
    """Cash Flow Data, return type of :class:`~core.contract.Clause`

    Attributes:

        :attr:`cf` Blank Cash Flow
        :attr:`p` probability of the cash flow
        :attr:`qx` probability of kick out of the contract
        :attr:`lx` in force number **before** the cash flow happens
        :attr:`base`: value of the base of the cashflow
        :attr:`ratio`: value of the ratio of the cashflow
        :attr:`mp`: model point information
        :attr:`meta_data` meta_data of the Clause instance generated the cash flow
    """

    cf: Tensor                = dc.field(default_factory=lambda: tensor(0., dtype=torch.double), repr=False)
    p: Tensor                 = dc.field(default_factory=lambda: tensor(0., dtype=torch.double), repr=False)
    qx: Tensor                = dc.field(default_factory=lambda: tensor(0., dtype=torch.double), repr=False)
    lx: Tensor                = dc.field(default_factory=lambda: tensor(1., dtype=torch.double), repr=False)
    base: Tensor              = dc.field(default_factory=lambda: tensor(1., dtype=torch.double), repr=False)
    ratio: Tensor             = dc.field(default_factory=lambda: tensor(1., dtype=torch.double), repr=False)
    mp: Tuple[Tensor, Tensor] = dc.field(default=(), repr=False)
    meta_data: Dict[str, Any] = dc.field(default_factory=dict)
    children: List[CashFlow]  = dc.field(default_factory=list)
    calculator_results: dict  = dc.field(default_factory=dict, repr=False)

    def copy(self)->CashFlow:
        return CashFlow(self.cf, self.p, self.qx, self.lx, self.base, self.ratio,
                        self.mp, self.meta_data.copy(), self.children.copy())

    # -------------------------------  probability related properties -------------------------------------
    @property
    def px(self)->Tensor:
        """ 1 - qx """
        return 1 - self.qx
    
    @property
    def pcf(self)->Tensor:
        """ cf * p """
        return self.cf * self.p

    @property
    def icf(self)->Tensor:
        """ pcf * lx = cf * p * lx """
        return self.pcf * self.lx

    @property
    def lx_aft(self)->Tensor:
        """ (1 - qx) * lx """
        return self.lx * self.px
    
    def lx_mul_(self, factor):
        self.lx = self.lx * factor
        for cf in self.children:
            cf.lx_mul_(factor)
        return self
    
    def update_lx(self, lx):
        _lx = self.lx
        self.lx = lx
        for cf in self.children:
            cf.update_lx(cf.lx * lx / _lx)
        return self
    
    # -------------------------------  discount related properties -------------------------------------
    @property
    def t_offset(self)->float:
        return self.meta_data['t_offset']

    def remove_offset(self, forward_rate)->CashFlow:
        """ Equivalent CashFlow with t_offset=0, the result is a shallow copy.

        :param Tensor forward_rate: discount rate
        :rtype: CashFlow
        """
        rst = self.copy()
        if self.children:
            rst.children = [cf.remove_offset(forward_rate) for cf in rst.children]
        elif self.t_offset > 0:
            rst.cf = self.cf / forward_rate.add(1).pow(self.t_offset)
            rst.meta_data.update(t_offset=0.)
        return rst

    def full_offset(self, forward_rate)->CashFlow:
        """ Equivalent CashFlow with t_offset=1, the result is a shallow copy.

        :param Tensor forward_rate: discount rate
        :rtype: CashFlow
        """
        rst = self.copy()
        if self.children:
            rst.children = [cf.full_offset(forward_rate) for cf in rst.children]
        elif self.t_offset < 1:
            rst.cf = self.cf * forward_rate.add(1).pow(1 - self.t_offset)
            rst.meta_data.update(t_offset=1.)
        return rst
    
    # -------------------------------  serializing methods --------------------------------------
    def to_sql(self, conn, batch_id):
        pass
