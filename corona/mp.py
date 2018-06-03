"""ModelPoint

"""
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy
import types
from cytoolz import get, valmap
import numpy as np
import pandas as pd


__all__ = ['ModelPointSet', 'ToNewBusiness', 'Scale', 'to_tensor',
           'ConstantCreditInterest', 'Compose', 'Lambda',
           'BatchSamplerFromIndex']


class ModelPointSet(Dataset):

    _INDEX_ORDER = ['sex', 'age', 'pmt', 'bft', 'mth']

    _INDEX_MAP = {
        'sex': 'SEX',
        'age': 'AGE_AT_ENTRY',
        'pmt': 'PREM_PAYBL_Y',
        'bft': 'POL_TERM_Y',
        'mth': 'DURATIONIF_M',
    }

    _VALUE_ORDER = ['prem', 'sa', 'av', 'crd']

    _N_POLS_IF = 'INIT_POLS_IF'

    _VALUE_MAP = {
        'prem': 'ANNUAL_PREM',
        'sa': 'SUM_ASSURED',
        'av': 'INIT_UNFDU',  # account value
        'crd': 'CURR_CRD_INT',  # credit interest
    }

    @classmethod
    def idx_map(cls):
        """override this function to define the map
         between dataframe column names to index names"""
        return cls._INDEX_MAP

    @classmethod
    def val_map(cls):
        """override this function to define the map
        between dataframe column names to value names"""
        return cls._VALUE_MAP

    @classmethod
    def idx_order(cls):
        return cls._INDEX_ORDER

    @classmethod
    def val_order(cls):
        return cls._VALUE_ORDER

    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.transform = transform
        self.dataframe = dataframe.sort_index()
        d = valmap(lambda s: s.values, dict(dataframe.items()))
        size = len(self.dataframe)
        idx_order, val_order = self.idx_order(), self.val_order()
        idx_map, val_map = self.idx_map(), self.val_map()

        mp_idx = np.array([d.get(x, np.zeros(size)) for x in get(idx_order, idx_map)]).T
        mp_val = np.array([d.get(x, np.full(size, np.nan)) for x in get(val_order, val_map)]).T

        self.n_pols_if = dataframe[self._N_POLS_IF].values

        self.mp_idx = mp_idx
        self.mp_val = mp_val * self.n_pols_if.reshape(-1, 1)
        self.batch_indicator = dataframe.index

    def __getattr__(self, item):
        if item in self.idx_map():
            rst = self.mp_idx[:, self.idx_order().index(item)]
        elif item in self.val_map():
            rst = self.mp_val[:, self.val_order().index(item)]
        else:
            raise AttributeError(f"{item}")
        return rst

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        mp_idx = self.mp_idx[idx, :]
        mp_val = self.mp_val[idx, :]
        if self.transform is None:
            return mp_idx, mp_val
        else:
            return self.transform(mp_idx, mp_val)

    @property
    def sp_code_batch_sampler(self):
        return BatchSamplerFromIndex(self)

    def data_loader(self, **kwargs):
        return DataLoader(self, batch_sampler=self.sp_code_batch_sampler, **kwargs)


class ToNewBusiness:
    """ Transform to mp represent new business """
    def __init__(self, mth_idx=4, prem_idx=0, av_idx=2):
        self.mth_idx = mth_idx
        self.prem_idx = prem_idx
        self.av_idx = av_idx

    def __call__(self, mp_index, mp_val):
        idx = mp_index.copy()
        idx[self.mth_idx] = 0
        val = mp_val.copy()
        val[self.av_idx] *= val[self.prem_idx] / val[self.av_idx]
        return idx, val


def to_tensor(mp_index, mp_val):
    return from_numpy(mp_index).long(), from_numpy(mp_val)


class Scale:

    def __init__(self, ratio, exclude_idx_lst=(3,)):
        self.ratio = ratio
        self.exclude_idx_lst = exclude_idx_lst

    def __call__(self, mp_index, mp_val):
        mp_val2 = mp_val.copy()
        mp_val2 *= self.ratio
        if self.exclude_idx_lst is not None:
            mp_val2[self.exclude_idx_lst] = mp_val[self.exclude_idx_lst]
        return mp_index, mp_val2


class ConstantCreditInterest:

    def __init__(self, val, crd_idx=3):
        self.val = val
        self.crd_idx = crd_idx

    def __call__(self, mp_index, mp_val):
        mp_val2 = mp_val.copy()
        mp_val2[self.crd_idx] = self.val
        return mp_index, mp_val


# ------------------- Transform from torchvision ----------------
class Compose:
    """Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, mp_idx, mp_val):
        for t in self.transforms:
            mp_idx, mp_val = t(mp_idx, mp_val)
        return mp_idx, mp_val

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, mp_idx, mp_val):
        return self.lambd(mp_idx, mp_val)

    def __repr__(self):
        return self.__class__.__name__ + '()'


# ------------------ Sampler ------------------------------

class BatchSamplerFromIndex:

    def __init__(self, mp_or_batch_index):
        if isinstance(mp_or_batch_index, pd.Index):
            self.batch_index = mp_or_batch_index
        elif isinstance(mp_or_batch_index, ModelPointSet):
            self.batch_index = mp_or_batch_index.batch_indicator
        else:
            raise ValueError('mp_or_batch_index should be a pandas index or ModelPointSet')
        self.labels = sorted(set(self.batch_index))
        self.locations = dict(((label, self._get_loc(label))
                               for label in self.labels))

    def _get_loc(self, key):
        rst = self.batch_index.get_loc(key)
        if isinstance(rst, slice):
            return list(range(rst.start, rst.stop))
        else:
            return np.flatnonzero(rst)

    def __len__(self):
        return len(self.labels)

    def __iter__(self):
        for label in self.labels:
            yield self.locations[label]
