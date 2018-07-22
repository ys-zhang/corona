# corona
An actuarial framework based on Pytorch (in development)

- 产品定义：core.contract
- 假设表：table
- prophet 兼容相关： prophet
- 模型点输入：mp

### TODO

 - Simply the result system.

### Create a product or contract

```{python3}
from corona.core.contract import *
from corana.core.probabilty import *
from corona.core.probability.cn import *

contract = Contract("UL_EXAMPLE",
                     SequentialGroup(
                         AClause('FEE', ratio_tables=-10., base=OnesBase(), t_offset=0,
                                 prob_tables=Inevitable(),
                                 virtual=True),
                         AClause('CREDIT-H1', ratio_tables=KeepCurrentCredit(.5),
                                 base=AccountValue(), t_offset=0.5,
                                 prob_tables=Inevitable(),
                                 contexts_exclude='~PROFIT',
                                 virtual=True),
                         ParallelGroup(
                             Clause('DB-INSIDE', ratio_tables=1.,
                                    base=AccountValue(), t_offset=0.5,
                                    prob_tables=CL13_I, contexts_exclude='GAAP, SOME_OTHER', virtual=True),
                             Clause('DB-OUTSIDE', ratio_tables=.6,
                                    base=AccountValue(), t_offset=0.5,
                                    prob_tables=CL13_I),
                             name='DB'),
                         AClause('CREDIT-H2', ratio_tables=KeepCurrentCredit(.5),
                                 base=AccountValue(), t_offset=0.5,
                                 prob_tables=Inevitable(),
                                 contexts_exclude='~PROFIT',
                                 virtual=True),
                        )
                     )
```