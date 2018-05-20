from corona.core.discount import *
import unittest
import torch


class TestDiscountLayer(unittest.TestCase):

    def setUp(self):

        self.layer1 = \
            DiscountLayer(forward_rate=torch.tensor([1., 1.]))

        self.layer2 = \
            ConstantDiscountRateLayer(2, 1.)

        self.input = torch.ones(2)

    def test1(self):
        print(self.layer1(self.input))
        print(self.layer2(self.input, 0))
