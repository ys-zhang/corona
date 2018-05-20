import unittest
from corona.predict.nb import *
from torch.nn.functional import *
import numpy as np


class TestNB(unittest.TestCase):

    def setUp(self):
        self.unit = torch.from_numpy(np.array(
            [
                [2053, 1934, 1818, 1704, 1592, 1483],  # a1
                [2053, 1934, 1818, 1704, 1592, 1483],  # a2
                # [1, 1, 1, 1, 1, 1],  # b1
                # [1, 1, 1, 1, 1, 1],  # b2
            ], dtype=float
        ))

        self.nb_plan = torch.from_numpy(np.array(
            [
                [1, 1, 2, 2],  # a1
                [2, 2, 1, 1],  # a2
                # [1, 2, 3, 4],  # b1
                # [1, 2, 3, 4],  # b2
            ], dtype=float
        ))

    def testConvolve(self):
        print('\n')
        print(convolve(self.unit, self.nb_plan) )

    def testConv(self):
        a = torch.tensor([[[1,2]],
                           [[2,4]]])
        b = torch.tensor([[[1, 1, 0, 2, 1]],
                          [[1, 1, 0, 2, 1]]])
        print(np.convolve([1, 1, 0, 2, 1], [2, 1]))
        print(conv1d(b.double(), a.double(), padding=1))

    def testConv2(self):
        a = torch.from_numpy(np.array(
            [[[1, 2],
             [2, 4]]]))
        b = torch.from_numpy(np.array(
            [[[1, 1, 0, 2, 1],
             [1, 1, 0, 2, 1]]]))
        print(np.convolve([1, 1, 0, 2, 1], [2, 1]))
        print(conv1d(b.double(), a.double(), padding=1))
