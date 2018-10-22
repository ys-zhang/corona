from .contract import Contract
from .contract import Clause,  AClause, SequentialGroup, ParallelGroup
from .contract import BaseSelector, OnesBase, WaitingPeriod, PremPayed, DescSA,\
    Waive, WaiveSelf, AccountValue
from .creditstrat import ConstantCredit, KeepCurrentCredit
from .prob.prob import Probability, Inevitable
from .prob import cn

# alias for ClauseGroup
SeqGrp = SequentialGroup
PrlGrp = ParallelGroup

