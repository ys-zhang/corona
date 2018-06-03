import unittest
from corona.core.contract import *
from corona.core.creditstrat import *
from corona.core.prob import *
from corona.core.prob.cn import *
from corona.prophet import *
from corona.mp import *
import pandas as pd


class TestContract(unittest.TestCase):

    def setUp(self):
        df = ProphetTable.read_modelpoint_table('./data/IUL042.RPT')
        mp = ModelPointSet(df, transform=to_tensor)
        self.dl = mp.data_loader()
        self.mp_idx = torch.tensor([[0, 12, 1, 88, 23],
                                    [1, 42, 1, 105, 1]]).long()
        self.mp_value = torch.tensor([[10000, 10000, 10000, 0.04],
                                      [10000, 10000, 10000, 0.05]], dtype=torch.double)
        self.contract = \
            Contract("UL_EXAMPLE",
                     SequentialGroup(
                         AClause('FEE', ratio_tables=-10.,
                                 base=OnesBase(), prob_tables=Inevitable(),
                                 virtual=True),
                         AClause('CREDIT-H1', ratio_tables=KeepCurrentCredit(.5),
                                 base=AccountValue(), prob_tables=Inevitable(),
                                 contexts_exclude='~PROFIT',
                                 virtual=True),
                         ParallelGroup(
                             Clause('DB-INSIDE', ratio_tables=1.,
                                    base=AccountValue(),
                                    prob_tables=CL13_I, contexts_exclude='GAAP, SOME_OTHER', virtual=True),
                             Clause('DB-OUTSIDE', ratio_tables=.6,
                                    base=AccountValue(),
                                    prob_tables=CL13_I),
                             name='DB'),
                         AClause('CREDIT-H2', ratio_tables=KeepCurrentCredit(.5),
                                 base=AccountValue(), prob_tables=Inevitable(),
                                 contexts_exclude='~PROFIT',
                                 virtual=True),
                        )
                     )

    # @unittest.skip('跳过contract.__repr__')
    def testRepr(self):
        print(self.contract)

    def testSearchClause(self):
        cl = self.contract.search_clause('FEE')
        self.assertEqual(cl.name, 'FEE')
        self.assertRaises(KeyError, self.contract.search_clause, "FOO")

    def testAllClauses(self):
        lst = list(self.contract.all_clauses())
        self.assertSequenceEqual([x.name for x in lst],
                                 ['FEE', 'CREDIT-H1', 'DB-INSIDE', 'DB-OUTSIDE',
                                  'CREDIT-H2'])

    def testSideEffect(self):
        lst = [c for c in self.contract.all_clauses() if isinstance(c.base, AccountValue)]
        self.assertSequenceEqual([c.base.key for c in lst],
                                 ['FEE', 'CREDIT-H1', 'CREDIT-H1', 'CREDIT-H1'])
        contract = \
            Contract("UL_EXAMPLE2",
                     SequentialGroup(
                         AClause('CREDIT-H1', ratio_tables=KeepCurrentCredit(.5),
                                 base=AccountValue(), prob_tables=Inevitable(),
                                 virtual=True),
                         ParallelGroup(
                             Clause('DB-INSIDE', ratio_tables=1.,
                                    base=AccountValue(),
                                    prob_tables=CL13_I, contexts_exclude='GAAP, SOME_OTHER', virtual=True),
                             Clause('DB-OUTSIDE', ratio_tables=.6,
                                    base=AccountValue(),
                                    prob_tables=CL13_I),
                             name='DB'
                         ),
                         AClause('CREDIT-H2', ratio_tables=KeepCurrentCredit(.5),
                                 base=AccountValue(), prob_tables=Inevitable(),
                                 virtual=True),
                     )
                     )
        self.assertSequenceEqual([c.base.key for c in contract.all_clauses() if isinstance(c.base, AccountValue)],
                                 [AccountValueCalculator.INITIAL_AV_KEY, 'CREDIT-H1', 'CREDIT-H1', 'CREDIT-H1'])

    def testAVCalculator(self):
        mp_idx = torch.tensor([[0, 12, 1, 88, 23],
                               [1, 42, 1, 105, 1]]).long()
        mp_value = torch.tensor([[10000, 10000, 10000, 0.04],
                                 [10000, 10000, 10000, 0.05]], dtype=torch.double)
        av_gaap = self.contract.av_calculator.account_value(mp_idx, mp_value, 'GAAP')
        av_default = self.contract.av_calculator.account_value(mp_idx, mp_value, None)
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
        df = pd.DataFrame(crst2.pcf.detach().numpy()).T
        df.to_clipboard()
        print(crst)
        # pd.DataFrame(CL13_I.qx.data.detach().numpy()).T.to_clipboard()


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
        rst = converter(self.mp_idx, self.mp_val)

        def f(p, b, prm):
            r = np.arange(MAX_YR_LEN) + 1.
            r[p-1:] = r[p-1]
            r *= prm
            r[b:] = 0
            return r

        arr_rst = np.stack((f(p, b, prm) for p, b, prm in zip(self.np, self.nb, self.prem) ))
        self.assertEqual(np.sum(np.abs(arr_rst - rst.numpy())), 0)
