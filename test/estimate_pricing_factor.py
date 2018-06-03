from corona import mp
from corona.core import contract as ct
from corona.core import prob as p
from corona.prophet import ProphetTable
from corona.table import PmtLookupTable
import pandas as pd
import torch

mp_df1 = ProphetTable.read_modelpoint_table('./data/IPL004.RPT')

mp1 = mp.ModelPointSet(mp_df1, transform=mp.Compose([mp.Scale(2),
                                                 mp.ToNewBusiness(),
                                                 mp.to_tensor]))

print('mp1')
for i in range(10):
    print(mp1[i])

mp_df2 = pd.read_csv('./data/10123002.csv')


class OnlineDataMPSet(mp.ModelPointSet):

    @classmethod
    def idx_map(cls):
        return {
            'sex': 'SEX',
            'age': 'ISSAGE',
            'pmt': 'PRMTERM',
            'bft': 'INSTERM',
            'mth': None,
        }

    @classmethod
    def val_map(cls):
        return {
            'prem': 'PREM',
            'sa': 'SA',
            'av': None,
            'crd': None,
        }


mp2 = OnlineDataMPSet(mp_df2, transform=mp.to_tensor)
print('mp2')
for i in range(10):
    print(mp2[i])

for d in mp1.data_loader():
    print(d)

model = ct.Contract(
    ct.SequentialGroup([
        ct.Clause('prem_income', ratio_tables=-1., base_control_param=0,
                  prob_tables={'PRICING': p.Inevitable()},
                  default_context='PRICING', virtual=True),
        ct.Clause('loading',
                  ratio_tables=PmtLookupTable('loading_table',
                                              table=torch.zeros(80, 40),  # loading 不超过 40 年
                                              n_col=105,
                                              index_table=torch.range(1, 80)  # 支持 1 - 80 年交
                                              ),
                  base_control_param=0,
                  prob_tables={'PRICING': p.Inevitable()},
                  default_context='PRICING', virtual=True),
        ct.Clause('db', ratio_tables=1., base_control_param=1, prob_tables={
            'PRICING': p.Probability()
        })
    ]))
