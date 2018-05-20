import unittest
import torch
from corona import table


class TestTable(unittest.TestCase):

    def setUp(self):
        self.table_tensor = torch.rand(2, 8)
        self.input = torch.tensor([1, 0, 1])
        self.n_col = 10
        self.name = 'test'

    def test_constant_padding(self):
        t = table.Table(self.name, self.table_tensor,
                        n_col=self.n_col, pad_value=10)
        print(t.table)
        rst = t(self.input)
        rst.sum().backward()
        print(t.raw_table.grad)
        t2 = table.Table(self.name+'_', torch.tensor([2, 3, 3, 23, 3]))
        print(t2(self.input))

    def test_last_value_padding(self):
        t = table.Table(self.name, self.table_tensor,
                        n_col=self.n_col, pad_mode=1)
        print(t.table[:, -4:])
        rst = t(self.input)
        rst.sum().backward()
        print(t.raw_table.grad)

    def test_max_padding(self):
        t = table.Table(self.name, self.table_tensor,
                        n_col=self.n_col, pad_mode=4)
        print(t.raw_table.max(1)[0])
        print(t.table[:, -4:])
        rst = t(self.input)
        rst.sum().backward()
        print(t.raw_table.grad)

    def test_min_padding(self):
        t = table.Table(self.name, self.table_tensor,
                        n_col=self.n_col,
                        pad_mode=table.PadMode.Min)
        print(t.raw_table.min(1)[0])
        print(t.table[:, -4:])
        rst = t(self.input)
        rst.sum().backward()
        print(t.raw_table.grad)

    def test_lookup_table(self):
        t1 = table.LookupTable(
            self.name,
            self.table_tensor,
            self.n_col,
            index_table=torch.tensor([[4, 5], [3, 3]])
        )
        print(t1.table)
        print(t1(torch.tensor([[3, 3], [4, 5]])))

        t2 = table.LookupTable(
            self.name + '_',
            self.table_tensor,
            self.n_col,
            index_table=torch.tensor([4, 5])
        )
        print(t2(torch.tensor([5, 4])))
