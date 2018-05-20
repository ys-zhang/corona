import unittest
from corona.prophet import *
import numpy as np
from corona.mp import ModelPointSet

class TestFacTable(unittest.TestCase):

    def setUp(self):

        self.ppp_file = open('./data/SB_PPP_IPL001.fac', errors='ignore')
        self.ppp_table = self.ppp_file.readlines()
        self.ppp_age_file = open('./data/PREM_LOADING_PPPAGE_IPL002.fac',
                                 errors='ignore')
        self.ppp_age_table = self.ppp_age_file.readlines()
        self.mp_file = open('./data/IPL004.RPT', errors='ignore')

    def testGenericReader(self):
        print('\n')
        # print(ProphetTable.read_generic_table('./data/SB_PPP_IPL001.fac'))
        # print(ProphetTable.read_generic_table('./data/PREM_LOADING_PPPAGE_IPL002.fac'))
        # print(ProphetTable.read_generic_table('./data/FACTOR_ITL007.fac'))
        # print(ProphetTable.read_generic_table('./data/CROSS_Discount_Rate_StressUp.fac'))
        print(ProphetTable.read_generic_table('./data/PARAMET_ASSUMPTION.fac'))

    def testMPReader(self):
        print('\n')
        df = ProphetTable.read_modelpoint_table(self.mp_file)
        s = ModelPointSet(df)
        print(s.mp_idx)
        print(s.mp_val)
