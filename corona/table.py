""" Modules used for defining Lookup Tables and Config Enums used for
Config Padding mechanism are defined in this Module.

Use `Table` for lookup input are exactly the row number of the table, else use
`LookupTable`.
"""
import torch
from torch.nn import Module, Parameter
from torch import Tensor
import numpy as np
from .utils import ClauseReferable, ContractReferable, pad


class Table(Module):
    r""" Defines a Table of parameters of an insurance contract or an assumption
    with padding mechanism supported throw `pad_mode` and `pad_value`.

    .. math::

        \text{out}_{i, j} = \text{table}_{\text{index}_i, j}

    All Table are inherited from this class.

    Attributes:
        - :attr:`name` (str)
           the name of Table.
        - :attr:`table` (Tensor)
           the table will be indexed, the dim of the table should be not more
           than 2. Padding can be used for long tedious table.
        - :attr:`n_col` (int)
           the total column number of the table after padding.

    """

    def __init__(self, name: str, table: Tensor, n_col: int=None,
                 *, pad_value: float=0, pad_mode=0):
        """
        :param str name: the name of Table
        :param Tensor table: raw table
        :param int n_col: if None(default), then no padding
            will be act on the input `table`, if provided the n_col is the
            total column number of the table after padding.
        :param float pad_value: value needed for the `pad_mode` to work,
            for example pad_value is the valued filled
            if `pad_mode=PadMode.Constant`.
        :param Union[int, PadMode] pad_mode: how to perform the padding,
            Constant padding by default.
        """
        super().__init__()
        self.name = name.strip()
        self.table = Parameter(table)
        self.n_col = n_col
        if self.table.nelement() == 1:
            self._need_lookup = False
        elif self.table.dim() == 1:
            self.n_col = self.table.nelement()
            self._need_lookup = False
        elif n_col and n_col > self.table.shape[1]:
            self._need_lookup = True
        else:
            self.n_col = self.table.shape[1]

        self.pad_value = pad_value
        self.pad_mode = pad_mode

    def forward(self, index: Tensor):
        """
        :param index: 1-D Tensor, index for index select
        :return: rows at `index`
        """
        table = pad(self.table, self.n_col, self.pad_value, self.pad_mode)
        if self._need_lookup:
            return torch.index_select(table, 0, index.long())
        else:
            return table.expand(index.nelement(), self.n_col)


class LookupTable(Table):
    r""" Defines a LookupTable of parameters of an insurance contract
    or an assumption with padding mechanism.

    First the input is converted into row number by ‘looking up'
    in `index_table` then the rows are selected in `table`.

    .. math::

         \text{out}_{i, j} =
            \text{table}_{\text{index_table}[\text{lookup}], \;  j} 

    All LookupTable Tables are inherited from this class.

    Attributes:
        - :attr:`name` (str)
           the name of Table.
        - :attr:`table` (Tensor)
           the table will be indexed, the dim of the table should be no more
           than 2. Padding can be used for long tedious table.
        - :attr:`index_table`
           the row index of lookup value
        - :attr:`n_col` (int)
           the total column number of the table after padding.

    Inputs:
         - :attr:`lookup` (Tensor):

    Shape:
        - index: dim >= 1
    """
    def __init__(self, name: str, table: Tensor, n_col: int=None,
                 *, index_col_num: int=1, index_table=None,
                 pad_value: float=0, pad_mode=0):
        """

        :param str name: the name of Table, used as the key in the storage
            WeakValueDict.
        :param Tensor table: the table will be indexed,
            the dim of the table should be not more than 2.
            Padding can be used for long tedious table.
        :param int n_col: if None(default), then no padding will
            be act on the input `table`, if provided the n_col is the total
            column number of the table after padding.
        :param int index_col_num: if :attr:`index_table` is omitted, the first
            :attr:`index_col_num` columns of :attr:`table` are selected
            as :attr:`index_table`
        :param Tensor index_table: table used to be convert input
            to row numbers. if omitted then :attr:`index_col_num` is used
        :param float pad_value: value needed for the `pad_mode` to work,
            for example pad_value is the valued filled
            if `pad_mode=PadMode.Constant`.
        :param Union[int, PadMode] pad_mode: how to perform the padding,
            Constant padding by default.
        """

        if n_col is None:
            n_col = table.shape[1] - index_col_num

        if index_table is None:
            self.raw_index_table = table[:, :index_col_num].long()
            super().__init__(name, table[:, :index_col_num], n_col,
                             pad_mode=pad_mode, pad_value=pad_value)
        else:
            self.raw_index_table: Tensor = index_table.long()
            super().__init__(name, table, n_col,
                             pad_mode=pad_mode, pad_value=pad_value)

        if self.raw_index_table.dim() == 2:
            self.sparse_index_table = \
                torch.sparse.LongTensor(self.raw_index_table.t(),
                                        torch.arange(0, self.raw_index_table.shape[0]).long(),
                                        torch.Size((self.raw_index_table.max(0)[0] + 1).tolist()))
            self.index_table = self.sparse_index_table.to_dense()
            self._is_one_dim_index = False
        elif self.raw_index_table.dim() == 1:
            self.sparse_index_table = None
            i = self.raw_index_table.max().item()
            _index_table = np.full(i + 1, self.table.shape[0])
            _index_table[self.raw_index_table.numpy()] = np.arange(self.raw_index_table.nelement())
            self.index_table = torch.from_numpy(_index_table).long()
            self._is_one_dim_index = True
        else:
            raise RuntimeError("dim of index_table bigger than 2")

    def forward(self, lookup: Tensor):
        """

        :param Tensor lookup: lookup value for index select
        """
        if self._is_one_dim_index:
            index = self.index_table[lookup]
        else:
            index = self.index_table.index(torch.unbind(lookup, 1))
        return super().forward(index)


class RatioTableBase(Module, ClauseReferable, ContractReferable):
    """ Base Clase of RatioTable
    """
    def forward(self, *inputs):
        raise NotImplementedError


class PmtLookupTable(LookupTable, RatioTableBase):
    """LookupTable with `payment term` as the lookup key
    the result after `benefit term` is set to 0.

    Inputs:
        - :attr:`mp_idx` (`Tensor`)

    """
    PMT_IDX = 2

    def forward(self, mp_idx, mp_val=None):
        """

        :param Tensor mp_idx:
        :param Tensor mp_val:
        :return:
        """
        pmt = mp_idx[:, self.PMT_IDX]
        return super().forward(pmt)


class AgeIndexedTable(Table, RatioTableBase):
    """Table with `issue age` as index
    the result after `benefit term` is set to 0.

    Inputs:
        - :attr:`mp_idx` (`Tensor`)

    """
    AGE_IDX = 1

    def forward(self, mp_idx, mp_val=None):
        """

        :param Tensor mp_idx:
        :param Tensor mp_val:
        :return:
        """
        age = mp_idx[:, self.AGE_IDX]
        return super().forward(age)


class PmtAgeLookupTable(LookupTable, RatioTableBase):
    """LookupTable with `payment term` and `issue age` as the lookup key
    the result after `benefit term` is set to 0.

    Inputs:
        - :attr:`mp_idx` (`Tensor`)

    """
    PMT_IDX = PmtLookupTable.PMT_IDX
    AGE_IDX = AgeIndexedTable.AGE_IDX

    def forward(self, mp_idx, mp_val=None):
        """

        :param Tensor mp_idx:
        :param Tensor mp_val:
        :return:
        """
        pmt = mp_idx[:, self.PMT_IDX]
        age = mp_idx[:, self.AGE_IDX]
        return super().forward(torch.stack([pmt, age], 0).t())
