from corona.utils import *
import unittest


class TestUtils(unittest.TestCase):

    def test_time_slice1d(self):
        ts = torch.rand(10)
        print(ts)
        print(time_slice1d(ts, 4, 0))

    def test_time_slice(self):
        ts = torch.rand(3, 4)
        aft = torch.tensor([2, 3, 0])
        print(ts)
        print(time_slice(ts, aft))

    def test_time_trunc(self):
        ts = torch.rand(3, 4)
        aft = torch.tensor([2, 3, 0])
        print(ts)
        print(time_trunc(ts, aft))

    def test_couple(self):
        ts = torch.tensor([[1, 2],  [3, 4], [5, 6]])
        cpl = Couple(2)
        print(torch.unbind(ts, 0))
        print(cpl(*torch.unbind(ts, 0)))

    def test_flip(self):
        ts1 = torch.arange(0, 10, requires_grad=True)
        flip(ts1)[0].backward()
        print(ts1.grad)
        ts2 = torch.rand(2, 3, requires_grad=True)
        flip(ts2)[:, 1:].sum().backward()
        print(ts2.grad)

    def test_repeat(self):
        ts = torch.tensor([[1, 2]], requires_grad=True)
        a = repeat(ts, 2, 0)
        print(a)
        a[:, 0].sum().backward()
        print(ts.grad)
