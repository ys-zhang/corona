import unittest
from corona.core import *
from corona.core.creditstrat import *
from corona.core import cn
from corona.core.contract import AccountValueCalculator, ClauseReferable
import pandas as pd
import torch
import numpy as np

ul_con = Contract("UL_EXAMPLE",
                  SequentialGroup(
                      AClause('FEE', ratio_tables=-10., t_offset=0,
                              base=OnesBase(), prob_tables=Inevitable(),
                              virtual=True),
                      AClause('CREDIT_H1', ratio_tables=KeepCurrentCredit(.5),
                              base=AccountValue(), prob_tables=Inevitable(),
                              contexts_exclude='~PROFIT', t_offset=.5,
                              virtual=True),
                      ParallelGroup(
                          Clause('DB_INSIDE', ratio_tables=1., t_offset=.5,
                                 base=AccountValue(),
                                 prob_tables=cn.CL13_I, contexts_exclude='GAAP, SOME_OTHER', virtual=True),
                          Clause('DB_OUTSIDE', ratio_tables=.6, t_offset=.5,
                                 base=AccountValue(),
                                 prob_tables=cn.CL13_I),
                          name='DB'),
                      AClause('CREDIT_H2', ratio_tables=KeepCurrentCredit(.5),
                              base=AccountValue(), prob_tables=Inevitable(),
                              contexts_exclude='~PROFIT', t_offset=.5,
                              virtual=True),))


class TestContract(unittest.TestCase):

    def setUp(self):
        # df = ProphetTable.read_modelpoint_table('./data/IUL042.RPT')
        # mp = ModelPointSet(df.dataframe, transform=to_tensor)
        # self.dl = mp.data_loader()
        self.mp_idx = torch.tensor([[0, 12, 1, 88, 23],
                                    [1, 42, 1, 105, 1]]).long()
        self.mp_value = torch.tensor([[10000, 10000, 10000, 0.04],
                                      [10000, 10000, 10000, 0.05]], dtype=torch.double)
        self.contract = ul_con

    @unittest.skip('跳过contract.__repr__')
    def testRepr(self):
        print(self.contract)

    def testSearchClause(self):
        cl = self.contract.search_clause('FEE')
        self.assertEqual(cl.name, 'FEE')
        self.assertRaises(KeyError, self.contract.search_clause, "FOO")

    def testAllClauses(self):
        lst = list(self.contract.all_clauses())
        self.assertSequenceEqual([x.name for x in lst],
                                 ['FEE', 'CREDIT_H1', 'DB_INSIDE', 'DB_OUTSIDE',
                                  'CREDIT_H2'])

    def testSideEffect(self):
        lst = [c for c in self.contract.all_clauses() if isinstance(c.base, AccountValue)]
        self.assertSequenceEqual([c.base.key for c in lst],
                                 ['FEE', 'CREDIT_H1', 'CREDIT_H1', 'CREDIT_H1'])
        contract = \
            Contract("UL_EXAMPLE2",
                     SequentialGroup(
                         AClause('CREDIT-H1', ratio_tables=KeepCurrentCredit(.5), t_offset=0.,
                                 base=AccountValue(), prob_tables=Inevitable(),
                                 virtual=True),
                         ParallelGroup(
                             Clause('DB-INSIDE', ratio_tables=1., t_offset=.5,
                                    base=AccountValue(),
                                    prob_tables=cn.CL13_I, contexts_exclude='GAAP, SOME_OTHER', virtual=True),
                             Clause('DB-OUTSIDE', ratio_tables=.6, t_offset=.5,
                                    base=AccountValue(), prob_tables=cn.CL13_I),
                             name='DB'
                         ),
                         AClause('CREDIT-H2', ratio_tables=KeepCurrentCredit(.5), t_offset=.5,
                                 base=AccountValue(), prob_tables=Inevitable(),
                                 virtual=True),
                        )
                     )
        self.assertSequenceEqual([c.base.key for c in contract.all_clauses() if isinstance(c.base, AccountValue)],
                                 [AccountValueCalculator.INITIAL_AV_KEY, 'CREDIT-H1', 'CREDIT-H1', 'CREDIT-H1'])

    def testAVCalculator(self):
        mp_idx = torch.tensor([[0, 12, 1, 88, 0],
                               [1, 42, 1, 105, 0]]).long()
        mp_value = torch.tensor([[10000, 10000, 10000, 0.04],
                                 [10000, 10000, 10000, 0.05]], dtype=torch.double)

        av_gaap = self.contract.AccountValueCalculator.account_value(mp_idx, mp_value, 'GAAP')
        av_default = self.contract.AccountValueCalculator.account_value(mp_idx, mp_value, None)
        self.assertAlmostEqual(av_gaap.numpy()[:, -1].sum(), 1853632.77909629)
        self.assertAlmostEqual(av_default.numpy()[:, -1].sum(), 1853632.77909629)
        df = pd.DataFrame(av_gaap.numpy()).T
        df.to_clipboard()

    def testRef(self):
        for c in self.contract.all_clauses():
            self.assertEqual(c.contract, self.contract)
            for rtb in c.modules():
                if isinstance(rtb, ClauseReferable):
                    self.assertEqual(rtb.clause, c)

    # @unittest.skip('跳过contract.__repr__')
    def testForward(self):
        crst = self.contract(self.mp_idx, self.mp_value, 'GAAP')
        crst2 = self.contract(self.mp_idx, self.mp_value, 'PROFIT')
        print(crst)


class TestBaseConverter(unittest.TestCase):

    def setUp(self):
        self.mp_idx = (torch.rand(3, 5).abs() * 10).long().clamp(min=1)
        self.mp_val = torch.rand(3, 4).abs().double() * 1000
        self.np = self.mp_idx[:, 2]
        self.nb = self.mp_idx[:, 3]
        self.prem = self.mp_val[:, 0]
        print(self.np)
        print(self.nb)
        print(self.prem)

    def testPremPayed(self):
        converter = PremPayed()
        rst = converter(self.mp_idx, self.mp_val, annual=True)

        def f(p, b, prm):
            r = np.arange(MAX_YR_LEN) + 1.
            r[p-1:] = r[p-1]
            r *= prm
            r[b:] = 0
            return r

        arr_rst = np.stack((f(p, b, prm) for p, b, prm in zip(self.np, self.nb, self.prem) ))
        self.assertEqual(np.sum(np.abs(arr_rst - rst.numpy())), 0)
