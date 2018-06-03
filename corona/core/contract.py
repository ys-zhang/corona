""" Modules and other utilities help to define a Product or Contract

Clause
^^^^^^

A `clause` in the context of `corona` is a concept that mimics a single clause
in insurance product defining obligations of policy holders and insurer
involving cash flow (e.g. benefits, fees etc.)

.. note::

    A typical definition of a clause or benefit is like::

        2 times of premium payed will be payed
        in case the insurant dead within the policy term

usually 3 kind of critical information related to calculation is provided:

#. the claim of this benefit is proportional to `SA`,
   which is contained in the model point,
#. the ratio equals to 2 in all cases,
#. the probability bundled to this clause can only be death probabilities.

We bundle all these information defines a clause in a contract
into a single instance of class `Clause`:

#. Base selection and transformation
    the value property of modelpoint (may transformed) like *sum assured*,
    *account value* or *gross premium*, two kind of parameters can be supplied:

    * base_control_param
    * base_converter
#. ratio_table
    the ratio_table is a instance of
    :class:`corona.table.Table` in case the ratio varies w.r.t.
    payment term or issue age.
    Three classes :class:`corona.table.PmtLookupTable`,
    :class:`corona.table.AgeIndexedTable`
    and :class:`corona.table.PmtAgeLookupTable` can be used in different cases.
#. probs
    probability bundle.

but only these are not enough for calculation, For Example we have such cases:

    #. The contract may remains effective after the clause is triggered
    #. Different probabilities are used in valuations with different purposes.


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
        *Contexts* are used when the calculation related to different kind of
        settings of assumptions. You should provide a context
        for most assumption related classes,
        for example :class:`corona.core.discount.DiscountLayer` and
        :class:`corona.core.prob.SelectionFactor`

        Examples of Contests: PRICING, GAAP, CROSS, etc.

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
from collections import OrderedDict
from functools import reduce
from operator import mul
from typing import Callable, Optional
from contextlib import ExitStack

import numpy as np
import torch
from cytoolz import isiterable, pluck, accumulate, valfilter, valmap
from torch.nn import Module

from corona.conf import MAX_YR_LEN
from corona.utils import CF2M, repeat, account_value, make_model_dict, Lambda, \
    make_parameter, ClauseReferable, ContractReferable, ModuleDict, CResult


def parse_context(context: str):
    try:
        if '@' not in context:
            return None, context
        else:
            n, c = str.rsplit(context, '@', 1)
            return re.compile(n), c
    except TypeError:
        return None, None


class Clause(Module, ContractReferable):
    r""" A typical Clause of an insurance contract.

    Attributes:
        - :attr:`name` (str)
           name of the clause, should be unique in the contract it belongs to
        - :attr:`prob_assumptions` (ModuleList)
           probability modules attached
        - :attr:`default_context` (str)
        - :attr:`mth_converter` (Union[Module, int])
           module convert annual cash flow to monthly whether cash flow, if
           `monthly_converter` is integer or None then
           :class:`corona.util.CF2M` is used
        - :attr:`virtual` (bool) whether if the prob of this clause will not involved in
           *in force* or *lx* calculation.
        - :attr:`contexts_exclude` (set) set of context names in which
           this clause's cash flow is excluded or treat as zero,
           but the probability is still in consideration.
    """

    def __init__(self, name, ratio_tables, base: Callable=None,
                 *, mth_converter=None,
                 prob_tables=None, default_context='DEFAULT', virtual=False,
                 contexts_exclude=()):
        super().__init__()
        self.name = name
        self.default_context = default_context
        self.virtual = virtual
        if isinstance(contexts_exclude, str):
            self._context_exclude_repr = contexts_exclude
            if contexts_exclude[0] == '~':
                _context_exclude = set((x.strip() for x in contexts_exclude[1:].split(',')))
                self.contexts_exclude = lambda x: x not in _context_exclude
            else:
                _context_exclude = set((x.strip() for x in contexts_exclude.split(',')))
                self.contexts_exclude = lambda x: x in _context_exclude
        elif isiterable(contexts_exclude):
            _context_exclude = set(contexts_exclude)
            self._context_exclude_repr = ','.join((str(x) for x in contexts_exclude))
            self.contexts_exclude = lambda x: x in _context_exclude
        else:
            assert callable(contexts_exclude)
            self._context_exclude_repr = repr(contexts_exclude)
            self.contexts_exclude = contexts_exclude

        # set up base modules & params
        if isinstance(base, int):
            self.base = BaseSelector(base)
        elif isinstance(base, BaseConverter):
            self.base = base
        else:
            raise TypeError(base)

        # set up ratio tables
        if not isiterable(ratio_tables):
            if isinstance(ratio_tables, float) or isinstance(ratio_tables, int):
                _r = float(ratio_tables)
                ratio_tables = Lambda(lambda mp_idx, _: _.new([_r]).expand(_.shape[0], MAX_YR_LEN),
                                      repre=ratio_tables)
            ratio_tables = {self.default_context: ratio_tables}
        self.ratio_tables = make_model_dict(ratio_tables, self.default_context)

        # set up probabilities
        if not isiterable(prob_tables):
            prob_tables = {self.default_context: prob_tables}
        self.prob_tables = make_model_dict(prob_tables, self.default_context)

        # set up mth converter
        if mth_converter is None or isinstance(mth_converter, int):
            self.mth_converter = CF2M(mth_converter)
        else:
            self.mth_converter = mth_converter

        self._setup_clause_ref()

    def _setup_clause_ref(self):
        for md in self.modules():
            if isinstance(md, ClauseReferable):
                md.set_clause_ref(self)

    def calc_ratio(self, mp_idx, mp_val, context, annual):
        r = self.ratio_tables[context](mp_idx, mp_val)
        return r if annual else repeat(r, 12)

    def calc_prob(self, mp_idx, mp_val, context, annual):
        return self.prob_tables[context](mp_idx, mp_val, annual=annual)

    def forward(self, mp_idx, mp_val, context=None, annual=False, *, calculator_results=None):
        pattern, context = parse_context(context)
        p = self.calc_prob(mp_idx, mp_val, context, annual)
        r = self.calc_ratio(mp_idx, mp_val, context, annual)
        val = self.base(mp_idx, mp_val, context)
        if not annual and not isinstance(self.base, AccountValue):
            val = self.mth_converter(val)
        cf = r * val
        if self.contexts_exclude(context):
            pcf = 0
        elif pattern is None or pattern.fullmatch(self.name):
            pcf = cf * p
        else:
            pcf = 0
        return CResult(pcf, cf, (mp_val.new_zeros((1,)) if self.virtual else p))

    def extra_repr(self):
        return f"$NAME$: {self.name}\n" \
               f"$VIRTUAL$: {self.virtual}\n" \
               f"$DEFAULT_CONTEXT$: {self.default_context}\n" \
               f"$CONTEXTS_EXCLUDE$: {self._context_exclude_repr}"


class SideEffect:

    def __init__(self):
        self.clause = None
        """weak proxy of the dirty clause"""

    def __call__(self, contract):
        raise NotImplementedError

    @property
    def name(self):
        return self.__class__.__name__


class ChangeAccountValue(SideEffect):

    def __call__(self, contract):
        """
        :param Contract contract:
        """
        hit_the_clause = False
        for cl in contract.all_clauses():
            base = cl.base
            if cl == self.clause:
                hit_the_clause = True
                if isinstance(base, AccountValue) and base.key is None:
                    base.key = AccountValueCalculator.INITIAL_AV_KEY
            elif hit_the_clause and isinstance(base, AccountValue):
                base.key = self.clause.name


class DirtyClause(Clause):
    """A Clause that is impure or have side effects
    """
    def __init__(self, *args, side_effect, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(side_effect, SideEffect)
        self.side_effect = side_effect
        self.side_effect.clause = weakref.proxy(self)

    def extra_repr(self):
        rst = super().extra_repr()
        return rst + f'\n$SIDE_EFFECT$: {self.side_effect.name}'


class AClause(DirtyClause):
    """A Clause that will change the Account Value of the contract
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, side_effect=ChangeAccountValue(), **kwargs)


class ClauseGroup(ModuleDict):

    def __init__(self, *clause, name=None, **kwargs):
        """
        :param clause: list of clause like objects including Clause,
            ParallelGroup etc
        :param name: a optional name of this group
        """
        if len(clause) == 1 and isiterable(clause[0]):
            clause = clause[0]
        d = OrderedDict(((n, cl) for cl, n in zip(clause, self._get_dict_name(clause))))
        super().__init__(d)
        if name is not None:
            self.name = name

    def clauses(self):
        return filter(lambda md: isinstance(md, Clause), self.children())

    def named_clauses(self)->dict:
        return valfilter(lambda md: isinstance(md, Clause), dict(self.named_children()))

    def clause_groups(self):
        return filter(lambda md: isinstance(md, ClauseGroup), self.children())

    def named_clause_groups(self)->dict:
        return valfilter(lambda md: isinstance(md, ClauseGroup), self.named_children())

    def search_clause(self, name):
        named_clauses = self.named_clauses()
        if name in named_clauses:
            return named_clauses[name]
        for grp in self.clause_groups():
            try:
                return grp.search_clause(name)
            except KeyError:
                pass
        else:
            raise KeyError(f"can't find clause with name: {name}")

    def all_clauses(self):
        for cl in self.children():
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
                name = module._get_name()
                i = d.get(name, 0) + 1
                d[name] = i
                yield f'{name}#{i}'

    def forward(self, *args, **kwargs):
        return valmap(lambda md: md(*args, **kwargs),
                      valfilter(lambda md: isinstance(md, Clause) or isinstance(md, ClauseGroup),
                                self._modules))


class ParallelGroup(ClauseGroup):
    """ A container define a list of "Clause" like objects with the order plays
    make no difference to calculation. The clauses within are independent.
    """
    def __init__(self, *clause, name=None):
        super().__init__(*clause, name=name)

    def forward(self, mp_idx, mp_val, context=None, annual=False, *, calculator_results=None):
        simple_results = super().forward(mp_idx, mp_val, context, annual, calculator_results=calculator_results)
        sub_results = list(simple_results.values())
        icf = sum(pluck(0, sub_results))
        cf = sum(pluck(1, sub_results))
        p = sum(pluck(2, sub_results))
        return CResult(icf, cf, p)


class SequentialGroup(ClauseGroup):
    r""" A container define a list of "Clause" like objects with the order plays
    a critical role in calculation. The probability of the whole
    group is calculated using a copula function.

    the default copula:

    .. math::
         \text{output}_i = \Pi_{k=0}^{i-1}(1 - \text{prob}_k)

    .. note::
       The probabilities of each clause
       is linked by a **copula** function which takes 2 arguments:

       #. list of probability (of kicked off from in force) tensors
       #. the context name

       And returns a list of tensor represents the inforce number just
       before the clause can be triggered  i.e. *px*.

    """
    # noinspection PyArgumentList
    def __init__(self, *clause, copula=None, name=None):
        """
        :param clause: list of clause like objects including Clause,
            ParallelGroup etc
        :param copula: a custom copula function
        :param name: a optional name of this group
        """
        super().__init__(*clause, name=name)
        if copula is None:
            self.copula = Lambda(lambda lst, context:
                                 list(accumulate(mul, (1 - x for x in lst),
                                                 initial=1))[:-1],
                                 repre='DefaultCopula')
        else:
            self.copula = copula

    def forward(self, mp_idx, mp_val, context=None, annual=False, *, calculator_results=None):
        simple_results = super().forward(mp_idx, mp_val, context, annual, calculator_results=calculator_results)
        sub_results = list(simple_results.values())
        icf_lst = list(pluck(0, sub_results))
        p_lst = list(pluck(2, sub_results))
        pp_lst = self.copula(p_lst, context)
        cf = sum(pluck(1, sub_results))
        p = 1 - pp_lst[-1] * (1 - p_lst[-1])
        return CResult(sum((icf * pp for icf, pp in zip(icf_lst, pp_lst))), cf, p)


# ================= Contract Definition ======================


class Contract(Module):

    def __init__(self, name, clauses: ClauseGroup):
        super().__init__()
        self.name = name
        self.clauses = clauses
        self.av_calculator = AccountValueCalculator(self)
        self._setup_contract_ref()

    def _setup_contract_ref(self):
        for md in self.modules():
            if isinstance(md, ContractReferable):
                md.set_contract_ref(self)

    def search_clause(self, name)->Optional[Clause]:
        return self.clauses.search_clause(name)

    def all_clauses(self):
        """ Iterator of all clauses in the contract """
        yield from self.clauses.all_clauses()

    def dirty_clauses(self):
        """ Iterator of dirty clauses in the contract """
        for cl in self.all_clauses():
            if isinstance(cl, DirtyClause):
                yield cl

    def calculators(self):
        yield self.av_calculator

    def forward(self, mp_idx, mp_val, context=None, annual=False):
        with ExitStack() as stack:
            calculator_results = dict(((calc.type(), stack.enter_context(calc(mp_idx, mp_val, context, annual)))
                                       for calc in self.calculators()))
            return self.clauses(mp_idx, mp_val, context, annual, calculator_results=calculator_results)

    def extra_repr(self):
        return f"$NAME$: {self.name}"


# ============================= Converters ==============================


class BaseConverter(Module, ClauseReferable, ContractReferable):
    # noinspection PyArgumentList
    BFT_INDICATOR = torch.tril(torch.ones(MAX_YR_LEN, MAX_YR_LEN, dtype=torch.double))
    BFT_IDX = 3

    def __init__(self):
        super().__init__()

    @classmethod
    def bft_indicator(cls, mp_idx):
        return cls.BFT_INDICATOR[mp_idx[:, cls.BFT_IDX].long() - 1, :]

    def forward(self, mp_idx, mp_val, context=None):
        raise NotImplementedError


class BaseSelector(BaseConverter):

    def __init__(self, idx):
        super().__init__()
        self.idx = idx

    def forward(self, mp_idx, mp_val, context=None):
        return mp_val[:, self.idx].unsqueeze(1).expand(mp_val.shape[0], MAX_YR_LEN)

    def __repr__(self):
        return str(self.idx)


class PremPayed(BaseConverter):
    # noinspection PyArgumentList
    FAC = torch.tril(torch.ones(MAX_YR_LEN, MAX_YR_LEN, dtype=torch.double)).cumsum(1)

    def __init__(self, pmt_idx=2, prem_idx=0):
        super().__init__()
        self.pmt_idx = pmt_idx
        self.prem_idx = prem_idx

    def forward(self, mp_idx, mp_val, context=None):
        prem = mp_val[:, self.prem_idx]
        pmt = mp_idx[:, self.pmt_idx].long()
        fac = self.FAC[pmt - 1, :]
        idc = self.bft_indicator(mp_idx)
        return fac * prem.reshape(-1, 1) * idc


class WaiveSelf(BaseConverter):
    FAC = torch.from_numpy(np.tril(np.ones((MAX_YR_LEN, MAX_YR_LEN),
                                           dtype=np.double), -1)
                           .T[::-1, :].cumsum(1)[:, ::-1].copy())

    def __init__(self, pmt_idx=2, prem_idx=0, *, rate):
        super().__init__()
        self.pmt_idx = pmt_idx
        self.prem_idx = prem_idx
        self.rate = make_parameter(rate)

    def forward(self, mp_idx, mp_val, context=None):
        prem = mp_val[:, self.prem_idx]
        pmt = mp_idx[:, self.pmt_idx].long()
        p_fac = self.FAC[pmt - 1, :]
        fac = ((1 + self.rate).pow(p_fac) - 1) / self.rate
        return fac * prem.reshape(-1, 1)


class Waive(BaseConverter):

    def __init__(self, sa_idx=1, *, rate):
        super().__init__()
        self.sa_idx = sa_idx
        self.rate = make_parameter(rate)

    def forward(self, mp_idx, mp_val, context=None):
        sa = mp_val[:, self.sa_idx].reshape(-1, 1)
        bft = mp_idx[:, self.BFT_IDX].long()
        p_fac = WaiveSelf.FAC[bft, :]
        fac = ((1 + self.rate).pow(p_fac) - 1) / self.rate
        return fac * sa


class DescSA(BaseConverter):

    def __init__(self, sa_idx=1):
        super().__init__()
        self.sa_idx = sa_idx

    def forward(self, mp_idx, mp_val, context=None):
        sa = mp_val[:, self.sa_idx].reshape(-1, 1)
        fac = WaiveSelf.FAC[mp_idx[:, self.BFT_IDX].long(), :]
        return fac * sa


class OnesBase(BaseConverter):

    def forward(self, mp_idx, mp_val, context=None):
        return mp_val.new_ones((mp_val.shape[0], MAX_YR_LEN))


class AccountValue(BaseConverter):

    def __init__(self):
        super().__init__()
        self.key = None

    def forward(self, mp_idx, mp_val, context=None):
        try:
            return self.contract.av_calculator[self.key]
        except KeyError:
            return self.contract.av_calculator(mp_idx, mp_val, context=context,
                                               key=self.key)


class Calculator:
    """ Abstract Calculator """

    def __init__(self, contract, *args, **kwargs):
        self.contract = weakref.proxy(contract)  # type: Contract
        self.set_up_calculator()
        # trigger side effects
        for cl in self.related_side_effect_clauses():
            cl.side_effect(self.contract)

    @classmethod
    def type(cls):
        return cls.__name__

    def set_up_calculator(self):
        raise NotImplementedError

    def related_side_effect_clauses(self):
        raise NotImplementedError

    def __call__(self, mp_idx, mp_val, context, annual, *args):
        raise NotImplementedError


class AccountValueCalculator(Calculator):

    INITIAL_AV_KEY = "___INITIAL_AV____" + str(random.random())

    def __init__(self, contract):
        super().__init__(contract)
        self._av_store = {}

    def related_side_effect_clauses(self):
        return (x for x in self.contract.dirty_clauses() if isinstance(x, AClause))

    def set_up_calculator(self):
        clauses = list(self.related_side_effect_clauses())
        cls_bf_crd = []
        cls_crd = []
        cls_aft_crd = []
        _meet_av_cl = False
        for cl in clauses:
            if isinstance(cl.base, AccountValue):
                cls_crd.append(cl)
                _meet_av_cl = True
                continue
            if _meet_av_cl:
                cls_aft_crd.append(cl)
            else:
                cls_bf_crd.append(cl)

        self.bf_crd = cls_bf_crd if cls_bf_crd else None
        """ clauses affect account value before credit """
        self.aft_crd = cls_aft_crd if cls_aft_crd else None
        """ clauses affect account value after credit """
        self.crd = cls_crd if cls_crd else None
        """ clauses act like credit """

    def before_credit(self, mp_idx, mp_val, context, annual=False):
        try:
            return [cl(mp_idx, mp_val, context, annual=annual).cf
                    for cl in self.bf_crd]
        except TypeError:
            return None

    def after_credit(self, mp_idx, mp_val, context, annual=False):
        try:
            return [cl(mp_idx, mp_val, context, annual=annual).cf
                    for cl in self.aft_crd]
        except TypeError:
            return None

    def rates(self, mp_idx, mp_val, context, annual=False):
        return [1 + cl.calc_ratio(mp_idx, mp_val, context, annual) for cl in self.crd]

    @staticmethod
    def sum(tensor_lst):
        try:
            return sum(tensor_lst)
        except TypeError:
            return None

    def account_value(self, mp_idx, mp_val, context, annual=False, *, memorize=False):
        a0 = mp_val[:, 2]  # the initiate account value
        rates_lst = self.rates(mp_idx, mp_val, context, annual)
        after_credit_lst = self.after_credit(mp_idx, mp_val, context, annual)
        before_credit_lst = self.before_credit(mp_idx, mp_val, context, annual)
        after_credit = self.sum(after_credit_lst)
        before_credit = self.sum(before_credit_lst)
        rates = reduce(mul, rates_lst) - 1
        av = account_value(a0, rates, after_credit, before_credit)
        if memorize:
            self._av_store[self.INITIAL_AV_KEY] = av
            curr = av
            if before_credit_lst is not None:
                for cf, cl in zip(before_credit_lst, self.bf_crd):
                    curr += cf
                    self._av_store[cl.name] = curr
            for r, cl in zip(rates_lst, self.crd):
                curr *= r
                self._av_store[cl.name] = curr
            if after_credit_lst is not None:
                for cf, cl in zip(after_credit_lst, self.aft_crd):
                    curr += cf
                    self._av_store[cl.name] = curr
        else:
            return av

    def __getitem__(self, item):
        return self._av_store[item]

    def __call__(self, mp_idx, mp_val, context, annual=False, *, key=None):
        _, context = parse_context(context)
        self.account_value(mp_idx, mp_val, context, annual, memorize=True)
        if key is not None:
            rst = self._av_store[key]
            self._av_store = {}
            return rst
        else:
            return self

    def __enter__(self):
        return self._av_store

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._av_store = {}
        return False
