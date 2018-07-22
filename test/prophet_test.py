import unittest
from corona.prophet import *
from corona.core.prob import *


class TestFacTable(unittest.TestCase):

    def setUp(self):
        pass
        # self.ppp_file = open('./data/SB_PPP_IPL001.fac', errors='ignore')
        # self.ppp_table = self.ppp_file.readlines()
        # self.ppp_age_file = open('./data/PREM_LOADING_PPPAGE_IPL002.fac',
        #                          errors='ignore')
        # self.ppp_age_table = self.ppp_age_file.readlines()
        # self.mp_file = './data/ITL006.RPT'

    def tearDown(self):
        ProphetTable.clear_cache()

    def testGenericReader(self):
        print('\n')
        # print(ProphetTable.read_generic_table('./data/SB_PPP_IPL001.fac'))
        # print(ProphetTable.read_generic_table('./data/PREM_LOADING_PPPAGE_IPL002.fac'))
        # print(ProphetTable.read_generic_table('./data/FACTOR_ITL007.fac'))
        print(ProphetTable.read_generic_table('./data/PREM_LOADING_PPPAGE_IPL002.fac'))
        # print(ProphetTable.read_generic_table('./data/PARAMET_ASSUMPTION.fac'))
        df = ProphetTable.read_generic_table('./data/CROSS_Lapse.fac')
        # print(df)
        with self.assertWarns(RuntimeWarning):
            print(type(df), type(df.T), type(df['1']))
        print(df[1, 1, 'PAR', 2])
        print(list(df._ALL_TABLES_.keys()))

    def testProbabilityTable(self):
        df1 = ProphetTable.read_probability_table('./data/CL03_M.fac', './data/CL03_F.fac')
        df = ProphetTable.read_probability_table('./data/CL13_1_U.fac')
        self.assertSequenceEqual(list(df.m.values), list(df.f.values))
        self.assertEqual(set(df._ALL_TABLES_.keys()), {'CL03', 'CL13_1_U'})
        with self.assertWarns(RuntimeWarning):
            self.assertIsInstance(df.as_probability(), Probability)
            self.assertIsInstance(df.as_selection_factor(), SelectionFactor)

    def testMPReader(self):
        print('\n')
        df = ProphetTable.read_modelpoint_table(self.mp_file)
        print(list(df._ALL_TABLES_.keys()))
        # print(df)
        s = df.as_modelpoint()
        # print(s.mp_idx)
        print(s.prem.sum())

    def testTableOfTable(self):
        cng_gaap_config = read_table_of_table('./data/TABLES/TABLE_CONFIG/CNGAAP_TABLE_CONFIG.fac')
        global_table = read_table_of_table('./data/TABLES/GLOBAL.fac')
        with self.assertWarns(RuntimeWarning):
            self.assertEqual(cng_gaap_config, global_table['RUN_12']['CNG_TABLE_CONFIG_TBL'])
        self.assertEqual(cng_gaap_config, global_table['CNG_TABLE_CONFIG_TBL', 'RUN_12'])
        self.assertEqual(cng_gaap_config, global_table['CNG_TABLE_CONFIG_TBL']['RUN_12'])
        self.assertEqual(cng_gaap_config, global_table.RUN_13['CNG_TABLE_CONFIG_TBL'])
        print(global_table.RUN_13['CNG_TABLE_CONFIG_TBL'].TABLE_NAME)

    def testReadFolder(self):
        read_assumption_tables('./data/TABLES', prob_folder='MORT',
                               param_pattern=r'PARAMET_.+', tot_pattern='GLOBAL|.*_TABLE_CONFIG',
                               exclude_folder='CROSS_LASTVAL',
                               exclude_pattern='PRICING_AGE_TBL')

    def testPrlifeReader(self):
        rst = prlife_read('./data/TABLES')
        print(dict([
            [key, [v.tablename for v in val]] for key, val in rst.items()
        ]))

    def testHDFSaver(self):
        prlife_read('./data/TABLES')
        ProphetTable.cache_to_hdf('./tables.h5')

    def testSqliteSaver(self):
        prlife_read('./test/data/TABLES')
        ProphetTable.cache_to_sqlite('./tables.db')
