""" Modules and other utilities help to define a Product or Contract

Clause
^^^^^^

A :class:`clause` in `corona` is a concept that mimics a single clause
in insurance product defining obligations of policy holders and insurer
involving cash flow (e.g. benefits, fees etc.)

A typical example of a clause or benefit can be::

    2 times of premium payed will be payed in case the insurant dead within the policy term

usually 3 kind of critical information is required by common actuarial calculation:
    #. the claim of this benefit is proportional to `SA`, which is contained in the model point,
    #. the ratio equals to 2 in all cases,
    #. the probability attached to this clause can only be death probabilities.

We bundle all these information defines a clause in a contract into a single instance of class `Clause`:
    #. base
        the value property of modelpoint (may transformed) like *sum assured*,
        *account value* or *gross premium*.
    #. ratio_table
        the ratio_table is a instance of
        :class:`~corona.table.Table` in case the ratio varies w.r.t.
        payment term or issue age.s
        Three classes :class:`~corona.table.PmtLookupTable`,
        :class:`~corona.table.AgeIndexedTable`
        and :class:`~corona.table.PmtAgeLookupTable` can be used in different cases.
        When the clause represent the credit cashflow of the linked Account of a Universal
        Link product. A instance of Credit Strategy can supplied as a ratio table.
        See module :mod:`~corona.core.creditstrat` for detail.

    #. probs
        probability bundle.

but only these are not enough for calculation, For Example we have such cases:
    #. The contract may remains effective after the clause is triggered.
    #. Different probabilities are used in valuations for different purposes.


additional control parameters and some middle layers should be supplied to
support these cases.
    #. virtual
        this parameter is introduced for the first case, when virtual is True
        then the probabilities will not be considered
        in *in force* calculation.
    #. excluded_in_contexts, default_context
        A iterable of contexts can be supplied to a Clause, the cash flow produced
        by this clause will be 0 under these contexts.

        the probability of default_context is used when the real context's
        probability not included in the param `probs`.

    .. note::
        **What is a context?**

        *Context* is a central concept in `corona`. It is widely used when your model has to base on different kinds of
        package of assumptions. 
        
        Typical contexts are *PRICING*, *GAAP*, *CROSS*, etc.

        When we refer to *discount rate* or *probability* in the *PRICING* context, we mean
        the discount rate and probabilities used in calculating the *gross premium* and *cash value*.

        Whats more, in different contexts, a cash flow may also have different meanings. For example,
        within the *GAAP* context, cashflows come from the linked account of a product
        whose design type is Universal is 0. Another case is the credit rate.
        When we test how our company will be like under different

#. mth_converter


Clause Group
^^^^^^^^^^^^

Usually a Contract in real world contains many liabilities and clause.
The clauses may have dependencies and may be only triggered in certain
circumstances. Thus we need more structures to  define a real Contract.
We provide 2 classes to help describe the relationship between clauses"

#. :class:`ParallelGroup`
    this class defines a relation like `or`, the order of clauses within this
    kind of group makes no difference to the calculation of probability and
    cash flow.

    In the view of an actuary, these clauses are independent
    and may be triggered simultaneously. The probability of this kind of
    group is the sum of all the clauses included.

#. :class:`SequentialGroup`
    this class defines a relation like `and`, the order of clauses within this
    kind of group is critical to the calculation of probability and
    cash flow.

    the probability of each clause included are linked by a copula function.

All types of groups support nesting.
"""
import re
import random
import weakref
import abc
from collections import OrderedDict
from functools import reduce, lru_cache
from itertools import accumulate
from operator import mul
from typing import Optional, Iterable, Dict, Callable, List, Union
from contextlib import ExitStack

import numpy as np
import torch
from torch import Tensor
from cytoolz import isiterable
from torch.nn import Module

from ..conf import MAX_YR_LEN
from ..utils import CF2M, repeat, account_value, make_model_dict, Lambda, \
    make_parameter, ClauseReferable, ContractReferable, ModuleDict, time_push, time_slice
from .cashflow import CashFlow


@lru_cache(128)
def parse_context(context: str):
    try:
        if '@' not in context:
            return None, context
        else:
            n, c = str.rsplit(context, '@', 1)
            return re.compile(n), c
    except TypeError:
        return None, None


# def in_force(g_result):
#     cfs = [cf for cf in g_result.values()]
#     pxs = [1 - cf.qx for cf in cfs]
#     px = reduce(lambda x, y: x * y, pxs)  # type: torch.Tensor
#     if_end = px.cumprod(1)
#     if_begin = if_end / px[:, :1]
#     g_result.lx = if_begin
#     return g_result

class CalculationContext:

    def __init__(self, *args):
        if len(args) == 1:
            self.repr_str = args[0]
    
    def __repr__(self):
        return self.repr_str


# ================================= Converters ================================


class BaseConverter(Module, ClauseReferable, ContractReferable):
    """Base of Converter Class, convert model point to base of benefits
    """
    BFT_INDICATOR = torch.tril(torch.ones(MAX_YR_LEN, MAX_YR_LEN, dtype=torch.double))
    BFT_IDX = 3  # policy term index

    def __init__(self, *, mth_converter: Optional[Union[int, Module]]=None):
        """
        :param mth_converter: Module used to convert an annual result to a monthly result, default None
           if mth_converter is None then it will be replace with the mth_converter of the Clause
              the BaseConverter lives in.
           if mth_converter is Integer then :class:`~util.CF2M` is used
           You can use :class:`~util.Lambda` to wrap a callable
        """
        super().__init__()
        if isinstance(mth_converter, int) or mth_converter is None:
            self.mth_converter = CF2M(only_at_month=mth_converter)
        else:
            self.mth_converter = mth_converter

    @classmethod
    def bft_indicator(cls, mp_idx: Tensor):
        return cls.BFT_INDICATOR[mp_idx[:, cls.BFT_IDX].long() - 1, :]

    def forward(self, mp_idx: Tensor, mp_val: Tensor, context: str, annual: bool=False):
        if annual:
            return self.annual_result(mp_idx, mp_val, context)
        else:
            return self.mth_converter(self.annual_result(mp_idx, mp_val, context))

    def annual_result(self, mp_idx: Tensor, mp_val: Tensor, context: str):
        raise NotImplementedError
    

class BaseSelector(BaseConverter):

    def __init__(self, idx, *, mth_converter: Optional[Union[int, Module]]=None):
        super().__init__(mth_converter=mth_converter)
        self.idx = idx

    def annual_result(self, mp_idx: Tensor, mp_val: Tensor, context: str):
        return mp_val[:, self.idx].unsqueeze(1).expand(mp_val.shape[0], MAX_YR_LEN)

    def __repr__(self):
        return str(self.idx)


class PremPayed(BaseConverter):
    FAC = torch.tril(torch.ones(MAX_YR_LEN, MAX_YR_LEN, dtype=torch.double)).cumsum(1)

    def __init__(self, pmt_idx=2, prem_idx=0, *, mth_converter: Optional[Union[int, Module]]=None):
        super().__init__(mth_converter=mth_converter)
        self.pmt_idx = pmt_idx
        self.prem_idx = prem_idx

    def annual_result(self, mp_idx: Tensor, mp_val: Tensor, context: str):
        prem = mp_val[:, self.prem_idx]
        pmt = mp_idx[:, self.pmt_idx].long()
        fac = self.FAC[pmt - 1, :]
        idc = self.bft_indicator(mp_idx)
        return fac * prem.reshape(-1, 1) * idc


class WaitingPeriod(BaseConverter):
    """ Represents Waiting Period for liabilities.

    Construction:
        :parameter int mth: length of the waiting period in number of months.
        :parameter BaseConverter after: base that used after the waiting period.
        :parameter BaseConverter during: base that used during the waiting period.
    """

    def __init__(self, mth: int, after: BaseConverter, during: BaseConverter=None,
                 *, mth_converter: Optional[Union[int, Module]] = None):
        super().__init__(mth_converter=mth_converter)
        assert isinstance(mth, int) and 1 <= mth <= 12
        self.mth = mth
        self.after = after
        if during is None:
            self.during = PremPayed()  # Premium Payed is the default during BaseConverter
        else:
            self.during = during

    def forward(self, mp_idx: Tensor, mp_val: Tensor, context: str, annual: bool=False):
        if annual:
            return self.annual_result(mp_idx, mp_val, context=context)
        else:
            aft = self.after(mp_idx, mp_val, annual=False)[:, self.mth:]
            dur = self.during(mp_idx, mp_val, annual=False)[:, :self.mth]
            return torch.cat((dur, aft), dim=1)
    
    def annual_result(self, mp_idx: Tensor, mp_val: Tensor, context: str):
        dur_ratio, aft_ratio = self.mth / 12, 1 - self.mth / 12
        aft = self.aft(mp_idx, mp_val, annual=True)
        dur = self.during(mp_idx, mp_val, annual=True)
        return torch.cat((dur[:, :1] * dur_ratio + aft[:, :1] * aft_ratio, aft[:, 1:]), dim=1)


class WaiveSelf(BaseConverter):
    FAC = torch.from_numpy(np.tril(np.ones((MAX_YR_LEN, MAX_YR_LEN), dtype=np.double), -1)
                           .T[::-1, :].cumsum(1)[:, ::-1].copy())

    def __init__(self, pmt_idx=2, prem_idx=0, *, rate, mth_converter: Optional[Union[int, Module]]=None):
        super().__init__(mth_converter=mth_converter)
        self.pmt_idx = pmt_idx
        self.prem_idx = prem_idx
        self.rate = make_parameter(rate)

    def annual_result(self, mp_idx: Tensor, mp_val: Tensor, context: str):
        prem = mp_val[:, self.prem_idx]
        pmt = mp_idx[:, self.pmt_idx].long()
        p_fac = self.FAC[pmt - 1, :]
        fac = ((1 + self.rate).pow(p_fac) - 1) / self.rate
        return fac * prem.reshape(-1, 1)


class Waive(BaseConverter):

    def __init__(self, sa_idx=1, *, rate, mth_converter: Optional[Union[int, Module]]=None):
        super().__init__(mth_converter=mth_converter)
        self.sa_idx = sa_idx
        self.rate = make_parameter(rate)

    def annual_result(self, mp_idx: Tensor, mp_val: Tensor, context: str):
        sa = mp_val[:, self.sa_idx].reshape(-1, 1)
        bft = mp_idx[:, self.BFT_IDX].long()
        p_fac = WaiveSelf.FAC[bft, :]
        fac = ((1 + self.rate).pow(p_fac) - 1) / self.rate
        return fac * sa


class DescSA(BaseConverter):
    
    def __init__(self, sa_idx=1, *, mth_converter: Optional[Union[int, Module]]=None):
        super().__init__(mth_converter=mth_converter)
        self.sa_idx = sa_idx

    def annual_result(self, mp_idx: Tensor, mp_val: Tensor, context: str):
        sa = mp_val[:, self.sa_idx].reshape(-1, 1)
        fac = WaiveSelf.FAC[mp_idx[:, self.BFT_IDX].long(), :]
        return fac * sa


class OnesBase(BaseConverter):

    def annual_result(self, mp_idx: Tensor, mp_val: Tensor, context: str):
        return mp_val.new_ones((mp_val.shape[0], MAX_YR_LEN))


class AccountValue(BaseConverter):

    def __init__(self, *, mth_converter: Optional[Union[int, Module]]=None):
        super().__init__(mth_converter=mth_converter)
        # this value is set as the name of the previous AClause by side effect: ChangeAccountValue
        # see AccountValueCalculator.account_value for the use of key
        self.key = AccountValueCalculator.INITIAL_AV_KEY

    def forward(self, mp_idx: Tensor, mp_val: Tensor, context: str, annual: bool=False):
        assert not annual, "Account value does not support annual"
        try:
            return self.contract.AccountValueCalculator[self.key]
        except KeyError:
            return self.contract.AccountValueCalculator(mp_idx, mp_val, context=context, key=self.key)

    def annual_result(self, mp_idx: Tensor, mp_val: Tensor, context: str):
        raise RuntimeError("Account value does not support annual")


# ==============================  Clauses =====================================
class ClauseLike:
    """ The base class of Clause and ClauseGroup.
    """
    pass


class Clause(Module, ContractReferable, ClauseLike):
    r""" A typical Clause of an insurance contract.

    Attributes:
        - :attr:`name` (str)
           name of the clause, should be unique in the contract it belongs to
        - :attr:`base` (:class:`BaseConverter`)
           base of the clause, calculate the value with the ratio will multiply to.
        - :attr:`t_offset` (float) time offset.
           For example, if t_offset=0.5, the clause is assumed to be triggered
           at the middle of a month in a monthly module
           or at the middle of a year in an annual module.
           **This value is used in discounting cash flow in other modules**
        - :attr:`ratio_tables` (:class:`~utils.ModuleDict`)
           A Module holds modules to calculation the ratio of `base` under some context
        - :attr:`prob_tables` (:class:`~utils.ModuleDict`)
           A Module holds modules to lookup probability under some context
        - :attr:`default_context` (str)
           The fallback context when the assumption of current contest is not found.
        - :attr:`mth_converter` (:class:`~torch.nn.Module`)
           module convert annual cash flow to monthly cash flow,
        - :attr:`virtual` (bool)
           whether if the prob of this clause will not involved in *in force* or *lx* calculation.
           for example a clause represents survival benefit should be virtual.
        - :attr:`contexts_exclude` (Callable)
           judging whether this clause's cash flow is excluded or treat as zero under the input context,
           *but the probability is still in consideration.*
    """

    name: str
    default_context: str
    t_offset: float
    virtual: bool
    _context_exclude_repr: str
    contexts_exclude: Callable[[str], bool]
    base: BaseConverter
    ratio_table: ModuleDict
    prob_tables: ModuleDict
    mth_converter: Callable
    meta_data: Dict[str, object]

    def __init__(self, name, ratio_tables, base, t_offset, *, mth_converter=None,
                 prob_tables=None, default_context='DEFAULT', virtual=False,
                 contexts_exclude=()):
        """

        :param str name: name of the clause
        :param ratio_tables: input can be both A dict and a single module.
           if the input is a dict, the keys should be contexts.
           if the input is a single module, the `default_context` is treated as the key
           if the input is a float, then it will convert to a module using :class:`~util.Lambda`, acting like a
           constant multiplier.
        :type ratio_tables: Module or dict[str, Module] or float
        :param base: base of the clause, calculate the value with the ratio will multiply to.
           input can be a  :class:`BaseConverter` or integer or `dict` of these.
           if the input is a integer, the base of the clause will be a :class:`BaseSelector`.
           if the input is a dict, the keys should be contexts.
        :type base: int or BaseConverter
        :param float t_offset: time offset
        :param mth_converter: input can be both A module and a integer.
           if `monthly_converter` is integer or None then :class:`~util.CF2M` is used.
           you can use :class:`~util.Lambda` to wrap a callable
        :type mth_converter: Module or int
        :param prob_tables: input can be both A dict and a single module.
           if the input is a dict, the keys should be contexts.
           if the input is a single module, the `default_context` is treated as the key
        :type prob_tables: Module or dict[str, Module]
        :param str default_context: default context, Default 'DEFAULT'
        :param bool virtual: default `False`
        :param contexts_exclude: contexts under which the cash flow of this clause should be excluded.
           Input can be a string, an iterable or a callable.

               - str: can be a single context or contexts separated by comma. if the string is started with `~`,
                  the semantics is *'exclude except'*.
               - iterable: contexts of the iterable is excluded
               - callable: returns `True` if the cash flow should be excluded under the input context.
        :type contexts_exclude: str or Iterable[str] or Callable

        """
        super().__init__()
        self.name = name
        self.t_offset = t_offset
        self.virtual = virtual
        
        self._setup_default_context(default_context, contexts_exclude)
        self._setup_base(base, mth_converter)
        # set up ratio tables
        self._setup_ratio_tables(ratio_tables)
        # set up probabilities
        self._setup_probabilities(prob_tables)
        self._setup_clause_ref()

        # dictionary meta information of the clause
        self.meta_data = {
            'class': self.__class__.__name__,
            'name': name,
            'ratio_tables': str(ratio_tables),
            'base': str(base),
            't_offset': t_offset,
            'mth_converter': str(mth_converter),
            'prob_tables': str(prob_tables),
            'default_context': default_context,
            'virtual': virtual,
            'context_exclude': str(contexts_exclude),
        }

    def _setup_clause_ref(self):
        for md in self.modules():
            if isinstance(md, ClauseReferable):
                md.set_clause_ref(self)
        return self

    def _setup_base(self, base: Union[int, BaseConverter], mth_converter):
        # set up mth converter
        if mth_converter is None or isinstance(mth_converter, int):
            self.mth_converter = CF2M(only_at_month=mth_converter)
        else:
            self.mth_converter = mth_converter

        # set up base modules & params
        def new_base(b: Union[int, BaseConverter]):
            if isinstance(b, int):
                b = BaseSelector(b)
            elif not isinstance(b, BaseConverter):
                raise TypeError(b)
            if not b.mth_converter:  # if mth converter is left blank then mth converter of the clause is used
                b.mth_converter = self.mth_converter
            return b
        self.base = new_base(base)
        return self

    def _setup_ratio_tables(self, ratio_tables):
        if not isiterable(ratio_tables):
            if isinstance(ratio_tables, float) or isinstance(ratio_tables, int):
                _r = float(ratio_tables)
                ratio_tables = Lambda(lambda mp_idx, _: _.new([_r]).expand(_.shape[0], MAX_YR_LEN),
                                      repre=ratio_tables)
            ratio_tables = {self.default_context: ratio_tables}
        self.ratio_tables = make_model_dict(ratio_tables, self.default_context)
        return self

    def _setup_probabilities(self, prob_tables):
        if not isiterable(prob_tables):
            prob_tables = {self.default_context: prob_tables}
        self.prob_tables = make_model_dict(prob_tables, self.default_context)
        return self
    
    def _setup_default_context(self, default_context, contexts_exclude):
        self.default_context = default_context
        if isinstance(contexts_exclude, str):
            self._context_exclude_repr = contexts_exclude
            if contexts_exclude.startswith('~'):
                _context_exclude = set((x.strip()
                                        for x in contexts_exclude[1:].split(',')))
                self.contexts_exclude = lambda x: x not in _context_exclude
            else:
                _context_exclude = set((x.strip()
                                        for x in contexts_exclude.split(',')))
                self.contexts_exclude = lambda x: x in _context_exclude
        elif isiterable(contexts_exclude):
            _context_exclude = set(contexts_exclude)
            self._context_exclude_repr = ','.join(
                (str(x) for x in contexts_exclude))
            self.contexts_exclude = lambda x: x in _context_exclude
        else:
            assert callable(contexts_exclude)
            self._context_exclude_repr = repr(contexts_exclude)
            self.contexts_exclude = contexts_exclude
        return self
    
    def calc_ratio(self, mp_idx, mp_val, context, annual)->Tensor:
        """Calculate ratio of modelpoint under some context. If context can't be find in the 
        relative assumptions, then `default_context` of the Clause is used

        :param Tensor mp_idx: index part of model point
        :param Tensor mp_val: value part of model point
        :param str context: calculation context
        :param bool annual: annual or monthly result
        :return: ratio
        """
        r = self.ratio_tables[context](mp_idx, mp_val)
        return r if annual else repeat(r, 12)

    def calc_prob(self, mp_idx, mp_val, context, annual)->Tensor:
        """Calculate probability of modelpoint under some context. If context can't be find in the 
        relative assumptions, then `default_context` of the Clause is used

        :param Tensor mp_idx: index part of model point
        :param Tensor mp_val: value part of model point
        :param str context: calculation context
        :param bool annual: annual or monthly result
        :return: probability
        """
        return self.prob_tables[context](mp_idx, mp_val, annual=annual)
    
    def calc_base(self, mp_idx, mp_val, context, annual)->Tensor:
        """Calculate base of modelpoint under some context. If context can't be find in the 
        relative assumptions, then `default_context` of the Clause is used

        :param Tensor mp_idx: index part of model point
        :param Tensor mp_val: value part of model point
        :param str context: calculation context
        :param bool annual: annual or monthly result
        :return: base
        """
        return self.base(mp_idx, mp_val, context=context, annual=annual)

    def forward(self, mp_idx: Tensor, mp_val: Tensor, context=None, annual: bool=False, *, calculator_results=None):
        """Cash flow calculation.

        :param Tensor mp_idx: index part of model point
        :param Tensor mp_val: value part of model point
        :param str context: calculation context
        :param bool annual: annual or monthly result
        :param calculator_results:
        :type calculator_results: dict[str, Tensor]
        :return: cash flow detail
        :rtype: CashFlow
        """
        pattern, context = parse_context(context)
        p = self.calc_prob(mp_idx, mp_val, context, annual)
        qx = mp_val.new_zeros(()) if self.virtual else p
        if self.contexts_exclude(context):
            cf = mp_val.new_zeros(())
            r = mp_val.new_zeros(())
            b = mp_val.new_zeros(())
        elif pattern is None or pattern.fullmatch(self.name):
            r = self.calc_ratio(mp_idx, mp_val, context, annual)
            b = self.calc_base(mp_idx, mp_val, context, annual)
            if annual:
                cf = r * b
            else:
                cf = time_slice(r * b, mp_idx[:, 4])
        else:
            cf = mp_val.new_zeros(())
            r = mp_val.new_zeros(())
            b = mp_val.new_zeros(())

        meta_data = self.meta_data.copy()
        meta_data.update(context=context, annual=annual)
        return CashFlow(cf, p, qx, mp_val.new_ones(()), b, r, (mp_idx, mp_val), meta_data)

    def extra_repr(self):
        return "$NAME$: {}\n".format(self.name) +\
               "$VIRTUAL$: {}\n".format(self.virtual) + \
               "$DEFAULT_CONTEXT$: {}\n".format(self.default_context) + \
               "$CONTEXTS_EXCLUDE$: {}".format(self._context_exclude_repr)


class SideEffect(abc.ABC):
    """Side effect of a impure :class:`DirtyClause`
    """
    def __init__(self):
        self.clause = None
        """weak proxy of the dirty clause"""

    @abc.abstractmethod
    def __call__(self, contract):
        raise NotImplementedError

    @property
    def name(self)->str:
        return self.__class__.__name__

    @classmethod
    @abc.abstractmethod
    def CalculatorClass(cls)->type:
        raise NotImplementedError


class ChangeAccountValue(SideEffect):
    """SideEffect that will change the account value.
    Used in account value calculation.
    """

    @classmethod
    def CalculatorClass(cls):
        return AccountValueCalculator

    def _handle_waiting_priod(self, base: WaitingPeriod):
        if isinstance(base.during, AccountValue):
            base.during.key = self.clause.name
        if isinstance(base.after, AccountValue):
            base.after.key = self.clause.name
    
    def _handle_account_value(self, base: AccountValue):
        base.key = self.clause.name

    def _handle_base(self, base: BaseConverter):
        if isinstance(base, AccountValue):
            self._handle_account_value(base)
        elif isinstance(base, WaitingPeriod):
            self._handle_waiting_priod(base)

    def __call__(self, contract):
        """
        :param Contract contract:
        """
        hit_the_clause = False
        for cl in contract.all_clauses():
            if cl == self.clause:
                hit_the_clause = True
            elif hit_the_clause:
                self._handle_base(cl.base)
                        

class DirtyClause(Clause):
    """A Clause that is impure and has a :class:`SideEffect`
    """
    side_effect: SideEffect

    def __init__(self, *args, side_effect: SideEffect, **kwargs):
        super().__init__(*args, **kwargs)
        # assert isinstance(side_effect, SideEffect)
        self.side_effect = side_effect
        self.side_effect.clause = weakref.proxy(self)

    def extra_repr(self):
        rst = super().extra_repr()
        return rst + f'\n$SIDE_EFFECT$: {self.side_effect.name}'

    @property
    def CalculatorClass(self)->type:
        return self.side_effect.CalculatorClass()


class AClause(DirtyClause):
    """:class:`DirtyClause` that will change the Account Value of the contract.
    with :class:`ChangeAccountValue` as SideEffect
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, side_effect=ChangeAccountValue(), **kwargs)


class ClauseGroup(ModuleDict, ClauseLike):
    """ Base of all Clause containers
    :class:`ParallelGroup` and :class:`SequentialGroup` inherit directly from this class.

    """

    def __init__(self, *clause, name=None):
        """
        :param clause: clause like objects including :class:`Clause`,
            :class:`ParallelGroup`, :class:`SequentialGroup` etc
        :param name: a optional name of this group
        """
        if len(clause) == 1 and isiterable(clause[0]):
            clause = clause[0]
        d = OrderedDict((n, cl) for cl, n in zip(clause, self._get_dict_name(clause)))
        super().__init__(d)
        self.name: Optional[str] = name

    def clauses(self):
        """Clauses contained directly in this group.

        :rtype: Iterable[Clause]
        """
        return filter(lambda md: isinstance(md, Clause), self.children())

    def named_clauses(self)->dict:
        """Dict of clauses contained directly in this group with name as key.

        :rtype: Dict[str, Clause]
        """
        return {name: md for name, md in self.named_children() if isinstance(md, Clause)}
    
    def clause_groups(self):
        """Subgroup of clauses contained directly in this group.

        :rtype: Iterable[ClauseGroup]
        """
        return filter(lambda md: isinstance(md, ClauseGroup), self.children())

    def named_clause_groups(self)->dict:
        """Dict of subgroup of clauses contained directly in this group with name as key.

        :rtype: Dict[str, ClauseGroup]
        """
        return {name: md for name, md in self.named_children() if isinstance(md, ClauseGroup)}

    def search_clause(self, name):
        """Get clause by name. It will search among all clauses contained in this group,
        including those contained by sub groups

        :rtype: Clause
        """
        named_clauses = self.named_clauses()
        if name in named_clauses:
            return named_clauses[name]
        for grp in self.clause_groups():
            try:
                return grp.search_clause(name)
            except KeyError:
                pass
        else:
            raise KeyError("can't find clause with name: {}".format(name))

    def all_clauses(self):
        """iterator of all clauses contained in this group.

        :rtype: Iterable[Clause]
        """
        for cl in self.values():
            if isinstance(cl, Clause):
                yield cl
            elif isinstance(cl, ClauseGroup):
                yield from cl.all_clauses()

    @staticmethod
    def _get_dict_name(modules):
        d = {}
        for module in modules:
            if hasattr(module, 'name') and module.name is not None:
                yield module.name
            else:
                name = module.__class__.__name__
                i = d.get(name, 0) + 1
                d[name] = i
                yield f'{name}#{i}'

    def raw_result(self, *args, **kwargs)->List[CashFlow]:
        """Calc and return all results of children model

        :return: unmodified GResult
        :rtype: GResult
        """
        return [v(*args, **kwargs) for v in self.values() if isinstance(v, ClauseLike)]

    def forward(self, *args, **kwargs)->CashFlow:
        raise NotImplementedError


class ParallelGroup(ClauseGroup):
    """ A container define a list of "Clause" like objects with the order plays
    make no difference to calculation. The clauses within are independent.
    """
    def __init__(self, *clause, name=None, t_offset=None):
        super().__init__(*clause, name=name)
        self.t_offset = t_offset
        if self.t_offset is not None:
            for cl in self.clauses():
                cl.t_offset = self.t_offset

    def forward(self, mp_idx, mp_val, context=None, annual=False,
                *, calculator_results=None):
        """

        :param Tensor mp_idx: index part of model point
        :param Tensor mp_val: value part of model point
        :param str context: calculation context
        :param bool annual: annual or monthly result
        :param calculator_results:
        :type calculator_results: dict[str, Tensor]
        :return: group result of cash flow detail
        :rtype: CashFlow
        """
        raw_results = self.raw_result(mp_idx, mp_val, context, annual,
                                      calculator_results=calculator_results)
        qx = sum(r.qx for r in raw_results)
        return CashFlow(qx=qx, children=raw_results)


class SequentialGroup(ClauseGroup):
    r""" A container define a list of "Clause" like objects with the order plays
    a critical role in calculation. 
    
    The probability of the joint distribution at each time is calculated using a copula function.

    the default copula:

    .. math::
         \text{px}_i = \Pi_{k=0}^{i-1}(1 - \text{prob}_k)

    .. note::
       The probabilities of each clause
       is linked by a **copula** function which takes 2 arguments:

       #. list of probability (of kicked off from in force) tensors
       #. the context name

       And returns a list of tensor represents the number of in force just
       before the clause can be triggered  i.e. *px*.

    """
    # noinspection PyArgumentList
    copula: Callable

    def __init__(self, *clause, copula=None, name=None):
        """
        :param clause: clause like objects including :class:`Clause`,
            :class:`ParallelGroup`, :class:`SequentialGroup` etc
        :param copula: a custom copula function
        :param name: a optional name of this group
        """
        super().__init__(*clause, name=name)
        if copula is None:
            self.copula = Lambda(lambda lst, context: list(accumulate((1 - x for x in lst), mul)),
                                 repre='DefaultCopula')
        else:
            self.copula = copula

    def forward(self, mp_idx, mp_val, context=None, annual=False, *, calculator_results=None):
        """

        :param Tensor mp_idx: index part of model point
        :param Tensor mp_val: value part of model point
        :param str context: calculation context
        :param bool annual: annual or monthly result
        :param calculator_results:
        :type calculator_results: dict[str, Tensor]
        :return: group result of cash flow detail
        :rtype: CashFlow
        """
        sub_results = self.raw_result(mp_idx, mp_val, context, annual, calculator_results=calculator_results)
        qx_lst = [r.qx for r in sub_results]
        pp_lst = self.copula(qx_lst, context)
        raw_result = CashFlow(qx=1 - pp_lst[-1], children=sub_results)
        for r, px in zip(sub_results, pp_lst[:-1]):
            r.lx_mul_(px)  # update lx
        return raw_result


class Contract(Module):

    name: str
    clauses: ClauseGroup

    def __init__(self, name: str, clauses: ClauseGroup):
        """

        :param str name: name of the contract
        :param ClauseGroup clauses: clauses of the contract
        """
        super().__init__()
        self.name = name
        self.clauses = clauses
        related_calculator_class = set(cl.CalculatorClass for cl in self.dirty_clauses())
        self._calculators = tuple(cc(self) for cc in related_calculator_class)
        for c in self._calculators:
            setattr(self, c.contract_attr_name, c)
        # self.av_calculator = AccountValueCalculator(self)
        self._setup_contract_ref()
        if self.clauses.name is None:
            self.clauses.name = self.name

    def _setup_contract_ref(self):
        for md in self.modules():
            if isinstance(md, ContractReferable):
                md.set_contract_ref(self)

    def search_clause(self, name)->Optional[Clause]:
        return self.clauses.search_clause(name)

    def all_clauses(self):
        """ Iterator of all clauses in the contract """
        yield from self.clauses.all_clauses()

    def dirty_clauses(self) -> Iterable[DirtyClause]:
        """ Iterator of dirty clauses in the contract """
        for cl in self.all_clauses():
            if isinstance(cl, DirtyClause):
                yield cl

    def calculators(self):
        """

        :return: calculators in the contract
        :rtype: Iterable[Calculator]
        """
        # yield self.av_calculator
        for c in self._calculators:
            yield c

    @staticmethod
    def in_force(cashflow: CashFlow)->CashFlow:
        if not cashflow.children:
            px = 1 - cashflow.qx
            cashflow.lx = px.cumprod(1) / px[:, :1]
            return cashflow
        else:
            px = reduce(lambda x, y: x * y, (1 - cf.qx for cf in cashflow.children))
            in_force_before = px.cumprod(1) / px[:, :1]
            cashflow.update_lx(in_force_before)
            return cashflow

    def forward(self, mp_idx: Tensor, mp_val: Tensor, context: Optional[str]=None, annual: bool=False):
        """

        :param Tensor mp_idx: index part of model point
        :param Tensor mp_val: value part of model point
        :param str context: calculation context
        :param bool annual: annual or monthly result
        :rtype: CashFlow
        """
        with ExitStack() as stack:
            calculator_results = dict(((calc.type(), stack.enter_context(calc(mp_idx, mp_val, context, annual)))
                                       for calc in self.calculators()))
            rst = self.clauses(mp_idx, mp_val, context, annual, calculator_results=calculator_results)
            rst.calculator_results.update(calculator_results)
        return self.in_force(rst)

    def extra_repr(self):
        return "$NAME$: {}".format(self.name)


# ============================== Calculators ==================================

class Calculator(abc.ABC):
    """ Abstract Calculator """

    contract: Contract

    def __init__(self, contract):
        self.contract = weakref.proxy(contract)  # type: Contract
        self.set_up_calculator()
        # trigger side effects
        for cl in self.related_side_effect_clauses():
            cl.side_effect(self.contract)

    @property
    def contract_attr_name(self)->str:
        """ attribute name of the calculator in a contract """
        return self.__class__.__name__

    @classmethod
    def type(cls)->str:
        """ used as key word in `calculator_results` see forward method of `Clause` """
        tp = cls.__name__.replace('Calculator', '')
        return tp if tp else cls.__name__

    @abc.abstractmethod
    def set_up_calculator(self):
        """ call in the __init__ method before every setups except the
            setup of :attr:`contract`
        """
        pass

    @abc.abstractmethod
    def related_side_effect_clauses(self)->Iterable[DirtyClause]:
        raise NotImplementedError

    def __call__(self, mp_idx, mp_val, context, annual, *args):
        raise NotImplementedError

    @abc.abstractmethod
    def __enter__(self):
        """ the result of this function is stored in `calculator_results`, see forward method of :class:`Clause`
        """
        raise NotImplementedError


class AccountValueCalculator(Calculator):

    INITIAL_AV_KEY = "___HiMyFriend___" + str(random.random())

    bf_crd: Optional[Iterable[Clause]]
    aft_crd: Optional[Iterable[Clause]]
    crd: Optional[Iterable[AClause]]

    def __init__(self, contract: Contract):
        super().__init__(contract)
        self._av_store = {}

    def related_side_effect_clauses(self):
        return [x for x in self.contract.dirty_clauses() if isinstance(x, AClause)]

    @staticmethod
    def is_av_based(clause: Clause):
        base = clause.base
        if isinstance(base, AccountValue):
            return True
        elif isinstance(base, WaitingPeriod) and (isinstance(base.during, AccountValue)
                                                  or isinstance(base.after, AccountValue)):
            return True
        else:
            return False

    def set_up_calculator(self):
        clauses = self.related_side_effect_clauses()
        cls_bf_crd = []  # clauses before credit
        cls_crd = []  # clauses act like credit
        cls_aft_crd = []  # clauses after credit
        _meet_av_cl = False
        for cl in clauses:
            if self.is_av_based(cl):
                cls_crd.append(cl)
                _meet_av_cl = True
                continue
            if _meet_av_cl:
                cls_aft_crd.append(cl)
            else:
                cls_bf_crd.append(cl)

        self.bf_crd = cls_bf_crd
        """ clauses affect account value before credit """
        self.aft_crd = cls_aft_crd
        """ clauses affect account value after credit """
        self.crd = cls_crd
        """ clauses act like credit """

    def before_credit(self, mp_idx: Tensor, mp_val: Tensor, context: str, annual: bool=False)->List[Tensor]:
        return [cl(mp_idx, mp_val, context, annual=annual).cf for cl in self.bf_crd]

    def after_credit(self, mp_idx: Tensor, mp_val: Tensor, context: str, annual: bool=False)->List[Tensor]:
        return [cl(mp_idx, mp_val, context, annual=annual).cf for cl in self.aft_crd]

    def rates(self, mp_idx: Tensor, mp_val: Tensor, context: str, annual: bool=False)->List[Tensor]:
        return [cl.calc_ratio(mp_idx, mp_val, context, annual).add(1.) for cl in self.crd]

    @staticmethod
    def sum(tensor_lst):
        try:
            return sum(tensor_lst)
        except TypeError:
            return None

    def account_value(self, mp_idx, mp_val, context, annual=False, *, memorize=False):
        a0 = mp_val[:, 2]  # the initial account value
        rates_lst = self.rates(mp_idx, mp_val, context, annual)  # type: torch.Tensor
        after_credit_lst = self.after_credit(mp_idx, mp_val, context, annual)
        before_credit_lst = self.before_credit(mp_idx, mp_val, context, annual)
        dur_mth = mp_idx[:, 4]
        if not after_credit_lst:
            after_credit = None
        else:
            after_credit = time_slice(self.sum(after_credit_lst), dur_mth)
        if not before_credit_lst:
            before_credit = None
        else:
            before_credit = time_slice(self.sum(before_credit_lst),  dur_mth)
        rates = time_slice(reduce(mul, rates_lst) - 1, dur_mth)
        av = account_value(a0, rates, after_credit, before_credit)
        av = time_push(av, dur_mth)
        if memorize:
            self._av_store[self.INITIAL_AV_KEY] = av
            curr = av
            # todo optimize annihilation of account value before dur_mth
            mask = (av > 0.)  # type: torch.Tensor
            mask = mask.double()

            if before_credit_lst is not None:
                for cf, cl in zip(before_credit_lst, self.bf_crd):
                    curr = curr + cf * mask
                    self._av_store[cl.name] = curr
            for r, cl in zip(rates_lst, self.crd):
                curr = curr * r
                self._av_store[cl.name] = curr
            if after_credit_lst is not None:
                for cf, cl in zip(after_credit_lst, self.aft_crd):
                    curr = curr + cf * mask
                    self._av_store[cl.name] = curr
        else:
            return av

    def __getitem__(self, item):
        return self._av_store[item]

    def __call__(self, mp_idx, mp_val, context, annual=False, *, key=None):
        # _, context = parse_context(context)
        # self.account_value(mp_idx, mp_val, context, annual, memorize=True)
        if key is not None:
            rst = self._av_store[key]
            self._av_store.clear()
            return rst
        else:
            _, context = parse_context(context)
            self.account_value(mp_idx, mp_val, context, annual, memorize=True)
            return self

    def __enter__(self):
        return self._av_store[self.INITIAL_AV_KEY]

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._av_store.clear()
        return False
