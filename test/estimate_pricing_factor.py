from corona import mp
from corona.core import contract as ct
from corona.prophet import ProphetTable
import pandas as pd

mp_df1 = ProphetTable.read_modelpoint_table('./data/IPL004.RPT')

mp1 = mp.ModelPointSet(mp_df1,
                           transform=mp.Compose([mp.Scale(2),
                                                 mp.ToNewBusiness(),
                                                 mp.to_tensor]))

print('mp1')
for i in range(10):
    print(mp1[i])

mp_df2 = pd.read_csv('./data/10123002.csv')


class OnlineDataMPset(mp.ModelPointSet):

    def idx_map(cls):
        return {
            'sex': 'SEX',
            'age': 'ISSAGE',
            'pmt': 'PRMTERM',
            'bft': 'INSTERM',
            'mth': None,
        }

    def val_map(cls):
        return {
            'prem': 'PREM',
            'sa': 'SA',
            'av': None
        }


mp2 = OnlineDataMPset(mp_df2, transform=mp.to_tensor)
print('mp2')
for i in range(10):
    print(mp2[i])

for d in mp1.data_loader():
    print(d)