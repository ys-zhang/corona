from corona.utils import *
import unittest
import numpy as np


class TestUtils(unittest.TestCase):

    def test_time_slice1d(self):
        ts = torch.rand(10)
        ats = ts.numpy().copy()
        ats = np.roll(ats, -4)
        ats[-4:] = 0
        self.assertSequenceEqual(time_slice1d(ts, 4, 0).tolist(), list(ats))

    def test_time_slice(self):
        ts = torch.rand(3, 4)
        aft = torch.tensor([2, 3, 0])
        target = np.zeros((3, 4))
        target[0, :2] = ts.numpy()[0, 2:]
        target[1, :1] = ts.numpy()[1, 3:]
        target[2, :4] = ts.numpy()[2, 0:]
        self.assertEqual(np.sum(np.abs(time_slice(ts, aft).numpy()-target)), 0)

    def test_time_trunc(self):
        ts = torch.rand(3, 4)
        aft = torch.tensor([2, 3, 0])
        target = np.zeros((3, 4))
        target[0, :2] = ts.numpy()[0, :2]
        target[1, :3] = ts.numpy()[1, :3]
        self.assertEqual(np.sum(np.abs(time_trunc(ts, aft).numpy()-target)), 0)

    def test_flip(self):
        ts1 = torch.arange(0, 10, requires_grad=True)
        self.assertEqual(flip(ts1).tolist(), list(range(9, -1, -1)))
        flip(ts1)[0].backward()
        self.assertEqual(ts1.grad.tolist(), [0] * 9 +[1])
        ts2 = torch.rand(2, 3, requires_grad=True)
        flip(ts2)[:, 1:].sum().backward()
        self.assertEqual(ts2.grad.tolist(), [[1, 1, 0]] * 2)

    def test_repeat(self):
        ts = torch.tensor([[1, 2]], requires_grad=True)
        a = repeat(ts, 2, 0)
        self.assertEqual(a.tolist(), [[1,2]] * 2, 'repeat failed')
        a[:, 0].sum().backward()
        self.assertEqual(ts.grad.tolist(), [[2, 0]])

    def test_push(self):
        ts = torch.tensor([1, 2, 3])
        print(time_push1d(ts, 1))
