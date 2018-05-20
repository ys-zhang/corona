""" Modules and other utilities help to define a Product or Contract

Clause and Clause Groups
^^^^^^^^^^^^^^^^^^^^^^^^

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

"""
import re
from typing import Callable, List
from operator import mul
import numpy as np
import torch
from cytoolz import get, isiterable, pluck, keyfilter, accumulate
from torch.nn import Module, ModuleList, Parameter
from corona.const import MAX_YR_LEN
from corona.utils import CF2M, repeat


class AbstractClause(Module):
    """Abstract Class for Clause Definition.

    Attributes:
        - :attr:`name` (str)
           name of the clause
        - :attr:`probs` (ModuleList)
           probability modules attached
        - :attr:`default_context` (str)
           default context to lookup probability

    """
    def __init__(self, name, *, probs=None, default_context=None):
        super().__init__()
        self.name = name
        assert probs is None or isiterable(probs)
        if isinstance(probs, dict):
            probs = list(probs.values())
            self.probs = ModuleList(pluck(1, probs))
            self._asmp_index = dict((k, i) for i, k in
                                    enumerate(pluck(0, probs)))
        elif probs is not None:
            probs = [(p.context, p) for p in probs]
            self.probs = ModuleList(probs)
            self._asmp_index = dict((k, i) for i, k in
                                    enumerate(pluck(0, probs)))
        else:
            self.probs = ModuleList()
            self._asmp_index = {}
        self.default_context = default_context

    def set_prob(self, p: Module, context=None):
        """Set or add a probability module without duplication checking.

        :param str context: name of the prob module
        :param Module p: probability module
        """
        if context is None:
            context = p.context
        try:
            idx = self._asmp_index[context]
            self.probs[idx] = p
        except KeyError:
            self.probs.append(p)
            self._asmp_index[context] = len(self._asmp_index)
        return self

    def set_probs(self, p_lst, context_lst=None):
        """Set or add a list probability module without duplication checking.

        :param str context_lst: context list of the prob modules
        :param List[Module] p_lst: probability module
        """
        if context_lst is None:
            context_lst = [p.context for p in p_lst]
        for n, p in zip(context_lst, p_lst):
            self.set_prob(p, n)
        return self

    def add_prob(self, p: Module, context=None):
        """Add a probability module with duplication checking.

        :param str context: context of the prob module
        :param Module p: probability module
        """
        if context is None:
            context = p.context
        assert context not in self._asmp_index
        return self.set_prob(p, context)

    def add_probs(self, p_lst, context_lst=None):
        """Set or add a list probability module with duplication checking.

        :param str context_lst: context list of the prob modules
        :param List[Module] p_lst: probability module
        """
        if context_lst is None:
            context_lst = [p.context for p in p_lst]
        for n, p in zip(context_lst, p_lst):
            self.add_prob(p, n)
        return self

    def get_prob(self, context=None)->Module:
        """Get associated prob module by context,
        if `context` can not be found then
        the probability with context as `default_context` attribute of
        the instance is returned.

        :param str context: context of prob module to lookup.
        """
        default_context = self.default_context
        return self.probs[self._asmp_index.get(context, default_context)]

    def select_prob(self, patten: str):
        """Select probability modules by regular expression with `fullmatch`.

        :param str patten: regular expression.
        :return: list of probability modules with name can be
            full matched by pattern.
        """
        patten = re.compile(patten)
        idx = list(keyfilter(patten.fullmatch, self._asmp_index).values())
        return get(idx, self.probs)

    def forward(self, mp_idx, mp_val, *, context=None):
        raise NotImplementedError('Abstract Class')


class Clause(AbstractClause):
    r""" A typical Clause of an insurance contract.

    Attributes:
        - :attr:`name` (str)
           name of the clause
        - :attr:`prob_assumptions` (ModuleList)
           probability modules attached
        - :attr:`default_prob_name` (str)
           name of default probability module
        - :attr:`monthly_converter` (Union[Module, int])
           module convert annual cash flow to monthly cash flow, if
           `monthly_converter` is integer or None then
           :class:`corona.util.CF2M` is used
        - :attr:`virtual` (bool) if the prob of this clause not in part of
           *in force* or *lx* calculation.
        - :attr:`excluded_in_contexts` (set) set of context names in which
           this clause's cash flow is excluded or treat as zero,
           but the probability is still in consideration.
    """

    def __init__(self, name, ratio_table, base_control_param=None,
                 *, mth_converter=None, base_converter: Callable=None,
                 probs=None, default_context=None, virtual=False,
                 excluded_in_contexts=()):
        super().__init__(name, probs=probs, default_context=default_context)
        self.ratio_table = ratio_table
        if mth_converter is None or isinstance(mth_converter, int):
            self.mth_converter = CF2M(mth_converter)
        else:
            self.mth_converter = mth_converter
        self.base_control_param = base_control_param
        self.base_converter = base_converter
        self.virtual = virtual
        self.excluded_in_contexts = set(excluded_in_contexts)

    def forward(self, mp_idx, mp_val, *, annual=False, context=None):
        p = self.get_prob(context)(mp_idx, annual=annual)
        if context not in self.excluded_in_contexts:
            r = self.ratio_table(mp_idx)
        else:
            r = 0
        if self.base_converter is not None:
            val = self.base_converter(mp_idx, mp_val,
                                      control_param=self.base_control_param)
        else:
            val = mp_val[:, self.base_control_param]
        if not annual:
            r = repeat(r, 12)
            val = self.mth_converter(val)
        cf = r * val
        return cf * p, cf, (mp_val.new_zeros((1,)) if self.virtual else p)


class SynchronousClauseGroup(Module):

    def __init__(self, *clause_list):
        super().__init__()
        if not clause_list:
            clause_list = None
        elif len(clause_list) == 1 and isiterable(clause_list[0]):
            clause_list = clause_list[0]
        else:
            raise ValueError(clause_list)
        self.clauses = ModuleList(clause_list)

    def forward(self, mp_idx, mp_val, *, annual=False, context=None):
        sub_results = [s(mp_idx, mp_val, annual=annual, context=context)
                       for s in self.clauses]
        icf = sum(pluck(0, sub_results))
        cf = sum(pluck(1, sub_results))
        p = sum(pluck(2, sub_results))
        return icf, cf, p


class SequentialClauseGroup(Module):
    """ A container define a list of "Clause" like objects with order plays
    a critical role in inforce calculation. The probability of the whole
    group is calculated using a copula function.

    the default copula:

    .. math::
         output_i = \Pi_{k=0}^{i-1}(1 - prob_k)

    .. note::
       The probabilities of each clause
       is linked by a **copula** function which takes 2 arguments:

       #. list of probability (of kicked off from in force) tensors
       #. the context name

       And returns a list of tensor represents the inforce number just
       before the clause can be triggered  i.e. *px*.

    """
    # noinspection PyArgumentList
    def __init__(self, *clause_list, copula=None):
        """

        :param clause_list: list of clause like objects including Clause,
            SynchronousClauseGroup etc
        :param copula: a custom copula function
        """
        super().__init__()
        if not clause_list:
            clause_list = None
        elif len(clause_list) == 1 and isiterable(clause_list[0]):
            clause_list = clause_list[0]
        else:
            raise ValueError(clause_list)
        self.clauses = ModuleList(clause_list)
        if copula is None:
            self.copula = \
                lambda lst, context: list(accumulate(mul, (1 - x for x in lst),
                                                     initial=1))[:-1]
        else:
            self.copula = copula

    def forward(self, mp_idx, mp_val, *, annual=False, context=None):
        sub_results = [s(mp_idx, mp_val, annual=annual, context=context)
                       for s in self.clauses]
        icf_lst = list(pluck(0, sub_results))
        p_lst = list(pluck(2, sub_results))
        pp_lst = self.copula(p_lst, context)
        cf = sum(pluck(1, sub_results))
        p = 1 - pp_lst[-1] * (1 - p_lst[-1])
        return sum((icf * pp for icf, pp in zip(icf_lst, pp_lst))), cf, p


class BaseConverter(Module):
    BFT_INDICATOR = torch.tril(torch.ones(MAX_YR_LEN, MAX_YR_LEN))
    BFT_IDX = 3

    @classmethod
    def bft_indicator(cls, mp_idx):
        return cls.BFT_INDICATOR[mp_idx[:, cls.BFT_IDX].long() - 1, :]

    def forward(self, mp_idx, mp_val, control_param=None):
        raise NotImplementedError('BaseConverter')


class PremPayed(BaseConverter):
    FAC = torch.tril(torch.ones(MAX_YR_LEN, MAX_YR_LEN)).cumsum(1)

    def __init__(self, pmt_idx=2, prem_idx=0):
        super().__init__()
        self.pmt_idx = pmt_idx
        self.prem_idx = prem_idx

    def forward(self, mp_idx, mp_val, control_param=None):
        prem = mp_val[:, self.prem_idx]
        pmt = mp_idx[:, self.pmt_idx].long()
        fac = self.FAC[pmt - 1, :]
        idc = self.bft_indicator(mp_idx)
        return fac * prem.reshape(-1, 1) * idc


class WaiveSelf(BaseConverter):
    FAC = torch.from_numpy(np.tril(np.ones((MAX_YR_LEN, MAX_YR_LEN)), -1)
                           .T[::-1, :].cumsum(1)[:, ::-1].copy())

    def __init__(self, pmt_idx=2, prem_idx=0, *, rate):
        super().__init__()
        self.pmt_idx = pmt_idx
        self.prem_idx = prem_idx
        if not isinstance(rate, Parameter):
            self.rate = Parameter(torch.tensor(rate))
        else:
            self.rate = rate

    def forward(self, mp_idx, mp_val, control_param=None):
        prem = mp_val[:, self.prem_idx]
        pmt = mp_idx[:, self.pmt_idx].long()
        p_fac = self.FAC[pmt - 1, :]
        fac = ((1 + self.rate).pow(p_fac) - 1) / self.rate
        return fac * prem.reshape(-1, 1)


class Waive(BaseConverter):

    def __init__(self, sa_idx=1, *, rate):
        super().__init__()
        self.sa_idx = sa_idx
        if not isinstance(rate, Parameter):
            self.rate = Parameter(torch.tensor(rate))
        else:
            self.rate = rate

    def forward(self, mp_idx, mp_val, control_param=None):
        sa = mp_val[:, self.sa_idx].reshape(-1, 1)
        bft = mp_idx[:, self.BFT_IDX].long()
        p_fac = WaiveSelf.FAC[bft, :]
        fac = ((1 + self.rate).pow(p_fac) - 1) / self.rate
        return fac * sa


class DescSA(BaseConverter):

    def __init__(self, sa_idx=1):
        super().__init__()
        self.sa_idx = sa_idx

    def forward(self, mp_idx, mp_val, control_param=None):
        sa = mp_val[:, self.sa_idx].reshape(-1, 1)
        fac = WaiveSelf[mp_idx[:, self.BFT_IDX].long(), :]
        return fac * sa
