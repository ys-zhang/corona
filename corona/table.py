""" Modules used for defining Lookup Tables and Config Enums used for
Config Padding mechanism are defined in this Module.

Use `Table` for lookup input are exactly the row number of the table, else use
`LookupTable`.
"""
import enum
import weakref
import torch
from torch.nn import Module, Parameter, functional
from torch import Tensor
import numpy as np


class PadMode(enum.IntEnum):
    """ Different modes of padding mechanism.

    - :attr:`Constant`: 0, Pad left with constant value.
    - :attr:`LastValue`, :attr:`LastColumn`: 1, Pad left with the last Column
    - :attr:`MaxLastValueAnd`: 2 Pad left with max(lastColumn, value)
    - :attr:`MinLastValueAnd`: 3 Pad left with min(lastColumn, value)
    - :attr:`Max`: 4 Pad left with the maximum of the row
    - :attr:`Min`: 5 Pad left with the minimum of the row
    - :attr:`Average`: 6 Pad left with the average of the row
    - :attr:`Mode`: 6 Pad left with the mode of the row
    """
    Constant = 0
    """Pad left with constant value"""
    LastValue = 1
    """Pad left with the last Column"""
    LastColumn = 1
    """Same as `LastValue`"""
    MaxLastValueAnd = 2
    """Pad left with max(lastColumn, value)"""
    MinLastValueAnd = 3
    """Pad left with min(lastColumn, value)"""
    Max = 4
    """Pad left with the maximum of the row"""
    Min = 5
    """Pad left with the minimum of the row"""
    Avg = 6
    """Pad left with the average of the row"""
    Mode = 7
    """Pad left with the mode of the row"""


class Table(Module):
    r""" Defines a Table of parameters of an insurance contract or an assumption
    with padding mechanism supported throw `pad_mode` and `pad_value`.

    .. math::

        \text{out}_{i, j} = \text{table}_{\text{index}_i, j}

    instances will be kept in a WeakValueDict, with their names as keys.

    All Table are inherited from this class.

    Attributes:
        - :attr:`name` (str)
           the name of Table, used as the key in the storage WeakValueDict.
        - :attr:`table` (Tensor)
           the table will be indexed, the dim of the table should be not more
           than 2. Padding can be used for long tedious table.
        - :attr:`n_col` (Optional[int])
           if None(default), then no padding will be act on the input `table`,
           if provided the n_col is the total column number of the table
           after padding.
        - :attr:`raw_table` (Parameter)
           the raw table before padding

    Inputs:
         - :attr:`index` (Tensor)
            index for index select

    Shape:
        - index: 1-D

    """

    _TABLES = weakref.WeakValueDictionary()

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
        assert name.strip() not in self._TABLES
        self._TABLES[name] = self
        self.name = name.strip()
        self.raw_table = Parameter(table)
        self.n_col = n_col
        if self.raw_table.nelement() == 1:
            self.table = torch.full((1, n_col), self.raw_table)
            self._need_lookup = False
        elif self.raw_table.dim() == 1:
            self.table = torch.unsqueeze(self.raw_table, 0)
            self.n_col = self.raw_table.nelement()
            self._need_lookup = False
        elif n_col and n_col > self.raw_table.shape[1]:
            self._need_lookup = True
            pad = (0, n_col - self.raw_table.shape[1])
            if pad_mode == PadMode.Constant:
                self.table = functional.pad(self.raw_table, pad,
                                            'constant', pad_value)
            else:
                if pad_mode == PadMode.LastValue:
                    pad_value = self.raw_table[:, -1]
                elif pad_mode == PadMode.MinLastValueAnd:
                    pad_value = torch.nn.functional\
                        .threshold(self.raw_table[:, -1], pad_value, pad_value)
                elif pad_mode == PadMode.MaxLastValueAnd:
                    pad_value = -torch.nn.functional\
                        .threshold(-self.raw_table[:, -1], -pad_value,
                                   -pad_value)
                elif pad_mode == PadMode.Max:
                    pad_value = self.raw_table.max(1)[0]
                elif pad_mode == PadMode.Min:
                    pad_value = self.raw_table.min(1)[0]
                elif pad_mode == PadMode.Avg:
                    pad_value = self.raw_table.meam(1)
                elif pad_mode == PadMode.Mode:
                    pad_value = self.raw_table.mode(1)[0]
                else:
                    raise NotImplementedError(f'pad_mode: {pad_mode}')
                self.table = torch.cat((self.raw_table,
                                        torch.unsqueeze(pad_value, 1)
                                        .expand((-1, pad[1]))), 1)
        elif n_col:
            self.table = self.raw_table[:, :n_col]
        else:
            self.table = self.raw_table
            self.n_col = self.raw_table.shape[1]

    def forward(self, index: Tensor):
        if self._need_lookup:
            return torch.index_select(self.table, 0, index.long())
        else:
            return self.table.expand(index.nelement(), self.n_col)


class LookupTable(Table):
    r""" Defines a LookupTable of parameters of an insurance contract
    or an assumption with padding mechanism
    supported throw `pad_mode` and `pad_value`,

    First the input is converted into row number bu ‘looking up'
    in`index_table` then the rows are selected in `table`
    is not the row number of table.

    .. math::

        \text{out}_{i, j} =
            \text{table}_{\text{index_table}[\text{lookup}], j}

    instances will be kept in a WeakValueDict, with their names as keys.

    All Table are inherited from this class.

    Attributes:
        - :attr:`name` (str)
           the name of Table, used as the key in the storage WeakValueDict.
        - :attr:`table` (Tensor)
           the table will be indexed, the dim of the table should be not more
           than 2. Padding can be used for long tedious table.
        - :attr:`index_table`
           table storage the row index of lookup value
        - :attr:`n_col` (Optional[int])
           if None(default), then no padding will be act on the input `table`,
           if provided the n_col is the total column number of the table
           after padding.
        - :attr:`raw_table` (Parameter)
           the raw table before padding
        - :attr:`raw_index_table` (SparseTensor)
           sparse version of :attr:`index_table`

    Inputs:
         - :attr:`lookup` (Tensor): lookup value for index select

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
                                        torch.arange(0, self.raw_index_table
                                                     .shape[0]).long(),
                                        torch.Size((self.raw_index_table
                                                    .max(0)[0] + 1).tolist()))
            self.index_table = self.sparse_index_table.to_dense()
            self._is_one_dim_index = False
        elif self.raw_index_table.dim() == 1:
            self.sparse_index_table = None
            i = self.raw_index_table.max().item()
            _index_table = np.full(i+1, self.raw_table.shape[0])
            _index_table[self.raw_index_table.numpy()] =\
                np.arange(self.raw_index_table.nelement())
            self.index_table = torch.from_numpy(_index_table).long()
            self._is_one_dim_index = True
        else:
            raise RuntimeError(f"dim of index_table bigger than 2")

    def forward(self, lookup: Tensor):
        if self._is_one_dim_index:
            index = self.index_table[lookup]
        else:
            index = self.index_table.index(torch.unbind(lookup, 1))
        return torch.index_select(self.table, 0, index)


class PmtLookupTable(LookupTable):
    """LookupTable with `payment term` as the lookup key
    the result after `benefit term` is set to 0.

    Inputs:
        - :attr:`mp_idx` (`Tensor`)

    """
    PMT_IDX = 2

    def forward(self, mp_idx):
        pmt = mp_idx[:, self.PMT_IDX]
        return super().forward(pmt)


class AgeIndexedTable(Table):
    """Table with `issue age` as index
    the result after `benefit term` is set to 0.

    Inputs:
        - :attr:`mp_idx` (`Tensor`)

    """
    AGE_IDX = 1

    def forward(self, mp_idx):
        age = mp_idx[:, self.AGE_IDX]
        return super().forward(age)


class PmtAgeLookupTable(LookupTable):
    """LookupTable with `payment term` and `issue age` as the lookup key
    the result after `benefit term` is set to 0.

    Inputs:
        - :attr:`mp_idx` (`Tensor`)

    """
    PMT_IDX = PmtLookupTable.PMT_IDX
    AGE_IDX = AgeIndexedTable.AGE_IDX

    def forward(self, mp_idx):
        pmt = mp_idx[:, self.PMT_IDX]
        age = mp_idx[:, self.AGE_IDX]
        return super().forward(torch.stack([pmt, age], 0).t())
