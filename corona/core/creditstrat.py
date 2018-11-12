"""Strategy of credit interest, used as ratio table of a clause related to
credit of account value

"""
from corona.core.utils import make_parameter
from ..conf import MAX_YR_LEN
from corona.core.table import RatioTableBase


class CreditRateBase(RatioTableBase):

    def __init__(self, *args):
        super().__init__()

    def forward(self, mp_idx, mp_val=None):
        raise NotImplementedError


class ConstantCredit(CreditRateBase):
    """ Constant credit interest for all model points and all future months
    """
    def __init__(self, ratio_of_mth, rate):
        super().__init__()
        self.rate = make_parameter(rate)
        self.ratio_of_mth = ratio_of_mth

    def forward(self, mp_idx, mp_val=None):
        return mp_val.new_full((mp_val.shape[0], MAX_YR_LEN),
                               self.rate.add(1.).pow(self.ratio_of_mth / 12.).sub(1.))


class KeepCurrentCredit(CreditRateBase):
    """ Maintain current credit interest for all model points and all future months
    """
    def __init__(self, ratio_of_mth, crd_idx=3):
        super().__init__()
        self.crd_idx = crd_idx
        self.ratio_of_mth = ratio_of_mth

    def forward(self, mp_idx, mp_val=None):
        return mp_val[:, self.crd_idx].add(1.).pow(self.ratio_of_mth / 12.)\
            .sub(1.).unsqueeze(1).expand(mp_val.shape[0], MAX_YR_LEN)
