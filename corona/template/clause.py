from enum import IntEnum
from typing import Dict, List, Optional

from corona import PremPayed, SumAssured, Waive
from corona.core import Clause, Inevitable
from corona.core import PremIncome, BaseConverter
from corona.core.lookup import TableType, TableIndex
from corona.utils import *
from corona.prophet import ProphetTable

__all__ = ['TableType',
           'Premium', 'Loading']


@template
class Premium(ClauseTemplate):
    name: str = 'Premium'

    def __call__(self)->Clause:
        return Clause(self.name, 1., PremIncome(),
                      t_offset=0., prob_tables=Inevitable(),
                      virtual=True, contexts_exclude=())


@template
class Loading(ClauseTemplate):
    name: str = 'Loading'
    table_type: TableType = None
    loading: Dict[TableIndex, List[float]] = field(
        default=None,
        metadata={
            'Prophet': {
                'type': 'GenericTable',
                'formula': lambda self: "PREM_LOADING_{}_{}".format(self.table_type.value, self.prod_name),
            }}
    )
    cv_loading: Optional[Dict[TableIndex, List[float]]] = field(
        default=None,
        metadata={
            'Prophet': {
                'type': 'GenericTable',
                'formula': lambda self: "CV_LOADING_{}_{}".format(self.table_type.value, self.prod_name),
            }}
    )
    prod_name: str = field(
        default=None,
        metadata={
            'Prophet': {
                'type': 'ProductIdentifier',
                'formula': None
            }}
    )

    def __post_init__(self):
        if self.prod_name is None:
            assert self.loading is not None
            if isinstance(self.table_type, str):
                self.table_type = TableType(self.table_type)
            return
        if self.loading is None:
            self.loading = ProphetTable.get_table(f"PREM_LOADING_{self.table_type.value}_{self.prod_name}")\
                .dataframe.T.to_dict('list')
        if self.loading is not None:
            try:
                self.cv_loading = self.loading = ProphetTable\
                    .get_table(f"CV_LOADING_{self.table_type.value}_{self.prod_name}")\
                    .dataframe.T.to_dict('list')
            except KeyError:
                pass

    @property
    def ratio_table(self):
        klass = self.table_type.TableClass
        kwargs = {'GP': self.loading}
        if self.cv_loading:
            kwargs.update(CV=self.cv_loading)
        return {k: klass({pmt: [x / 100. for x in val] for pmt, val in v.items()})
                for k, v in kwargs.items()}

    def __call__(self, **kwargs):
        name = kwargs.get('name', self.name)
        ratio_tables = kwargs.get('ratio_tables', self.ratio_table)
        base = kwargs.get('base', PremIncome())
        t_offset = kwargs.get('t_offset', 0.)
        default_context = kwargs.get('default_context', 'GP')
        prob_tables = kwargs.get('prob_tables', Inevitable())
        contexts_exclude = kwargs.get('contexts_exclude', r"!GP|CV")
        virtual = kwargs.get('virtual', True)
        return Clause(name, ratio_tables, base, t_offset=t_offset,
                      default_context=default_context, prob_tables=prob_tables,
                      virtual=virtual, contexts_exclude=contexts_exclude)


class DeathBenType(IntEnum):
    RoP      = 1  # 1: %ROP
    SA       = 3  # 3: %SA
    MaxRoPCV = 4  # 4: max(%ROP, CV)
    WoP      = 5  # 5: %SA
    minRoP   = 7  # 7: %minROP
    RoP18    = 8  # 8: %ROP < 18 and SA >= 18
    MinSARoP = 9  # 9: %MinSAPremPayed

    def base(self, waiting_period: Optional[int]=None, within_waiting_period=False, **kwargs)->BaseConverter:
        mth_converter = kwargs.get('mth_converter', None)
        if self == self.RoP:
            pmt_idx = kwargs.get('prm_idx', 2)
            prem_idx = kwargs.get('prm_idx', 0)
            rst = PremPayed(pmt_idx, prem_idx, mth_converter=mth_converter,
                            waiting_period=waiting_period,
                            within_waiting_period=within_waiting_period)
        elif self == self.SA:
            sa_idx = kwargs.get('sa_idx', 1)
            rst = SumAssured(sa_idx, mth_converter=mth_converter,
                             waiting_period=waiting_period,
                             within_waiting_period=within_waiting_period)
        elif self == self.WoP:
            sa_idx = kwargs.get('sa_idx', 1)
            rate = kwargs['rate']
            rst = Waive(sa_idx, rate=rate, mth_converter=mth_converter,
                        waiting_period=waiting_period,
                        within_waiting_period=within_waiting_period)
        else:
            raise NotImplementedError(self)

        return rst


@template
class DeathBenefit(ClauseTemplate):
    name: str = "DeathBenefit"
    baseType: int = field(default=None, metadata={
        'Prophet': {
            'type': 'Parameter',
            'variable': 'DEATH_BEN_TYPE',
            'formula': lambda self, table: DeathBenType(table[self.prod_name].DEATH_BEN_TYPE),
        }
    })
    ratio: float = field(default=1., metadata={
        'Prophet': {
            'type': 'Parameter',
            'variable': ('DTH_BEN_ROP_PC', 'DTH_BEN_SA_PC'),
            'formula': lambda self, table: max(table[self.prod_name].DTH_BEN_ROP_PC,
                                               table[self.prod_name].DTH_BEN_SA_PC),
        }
    })
    waiting_period: Optional[int] = field(default=None, metadata={
        'Prophet': {
            'type': 'Parameter',
            'variable': 'WAITING_PERIOD_M',
        }
    })
    prod_name: str = field(default=None, metadata={
        'Prophet': {
            'type': 'ProductIdentifier',
        }
    })

    def __post_init__(self):
        if self.prod_name is None:
            return
        elif self.prod_name[:3] == 'IUL':
            parameter_table = ProphetTable.get_table('PARAMET_U')
        elif self.prod_name[:3] == 'ITL' or self.prod_name[:3] == 'IPL':
            parameter_table = ProphetTable.get_table('PARAMET_C')
        else:
            raise ValueError(self.prod_name)
        if self.baseType is None:
            self.baseType = DeathBenType(parameter_table[self.prod_name].DEATH_BEN_TYPE),

    def __call__(self, *args, **kwargs):
        pass
