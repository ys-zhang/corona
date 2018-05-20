import collections

MAX_YR_LEN = 107
MAX_MTH_LEN = MAX_YR_LEN * 12
MAX_ISSUE_AGE = 105

AccountValues = collections.namedtuple(
    'AccountValues',
    ['beforeDeduction', 'beforeCredit', 'halfCredit',
     'beforeBonus', 'afterBonus'])
