"""
utils for Sungard Prophet Format file support for example .fac .RPT file

tables are read as instances of :class:`ProphetTable`. Once a table is read, it
is cached by `ProphetTable` and indexed by `tablename` except model point tables.
With the help of this mechanism we implement `Table of Table`.

For now 5 kinds of Prophet Tables are supported:
    #.  `GenericTable`
    #.  `ModelPoint`
    #.  `Parameter`
    #.  `Probability`
    #.  `TableOfTable`

and 5 functions are provided to help user read tables by path:
    #.  `read_generic_table`
    #.  `read_modelpoint_table`
    #.  `read_parameter_table`
    #.  `read_probability_table`
    #.  `read_table_of_table`

A `ProphetTable` is just like a pandas DataFrame, except that:
    #. `[]` can select both row and column, but **we strongly recommend only use it when selecting rows**.
       At this version, a warn will be thrown out if a column is selected and returned.
    #. Dot expression can be used to select column just like a DataFrame, for example `GLOBAL.RUN_99`
       is column "RUN_99" of table "GLOBAL". **We strongly recommend you to use the dot expression only
       to select columns**.
    #. Unlike `DataFrame`, there is no `loc` or `iloc` attribute in `ProphetTable`
    #. When selecting cells with string value from a TableOfTable, the result can be different from other types
       of ProphetTables. First the result is looked up in the cache, if there is a table cached with
       the result string as its `tablename`, the cached table is returned in place of the result string.

    .. note::

            When `[]` is triggered,  first a function like :attr:`DataFrame.loc` is tried,
            then a function like :attr:`DataFrame.iloc` is tried and at last the semantics
            of `[]` in :class:`pandas.DataFrame` is tried. If all these failed a KeyError is
            raised.


Example
-------

.. code::

    prlife_read("./Tables")
    GLOBAL = ProphetTable.get_table('GLOBAL')  # global table is a Table of Table
    GLOBAL.T # transpose the table
    RUN13 = GLOBAL.RUN_13 # run 13 configuration, good style
    RUN13 == GLOBAL['RUN_13'] # True, but bad style

    # returns CNG_TABLE_CONFIG_TBL itself of run 13 not the table name
    CNG_TABLE_CONFIG_TBL = GLOBAL.RUN_13['CNG_TABLE_CONFIG_TBL']
    CNG_TABLE_CONFIG_TBL == GLOBAL['CNG_TABLE_CONFIG_TBL', 'RUN_13'] # True, good style

    # CNG_TABLE_CONFIG_TBL itself is a TableOfTable thus you can keep selecting like a chain
    lapse_table = GLOBAL.RUN_13['CNG_TABLE_CONFIG_TBL'].TABLE_NAME['LAPSE']
    # some times you may want `tablename` not table it self. You can use the
    # `dataframe` attribute of a TableOfTable
    lapse_table_name = GLOBAL.RUN_13['CNG_TABLE_CONFIG_TBL'].TABLE_NAME.dataframe['LAPSE']  # type: str
    lapse_table2017 = ProphetTable.get_table(lapse_table_name + '2017')

"""
import enum
import os
import warnings
import sqlite3
from typing import Dict

import pyparsing as pp
import pandas as pd
import numpy as np
from cytoolz import get

__all__ = ['ProphetTable', 'read_generic_table', 'read_parameter_table',
           'read_probability_table', 'read_modelpoint_table',
           'read_table_of_table', 'read_assumption_tables', 'prlife_read']


# ===================== Prophet Fac Table Reading ============================


Comment = pp.restOfLine.setParseAction(lambda toks: toks[0].strip())
Integer = pp.Combine(pp.Optional('-') + pp.Word(pp.nums))\
    .setParseAction(lambda toks: int(toks[0]))
Float = pp.Combine(pp.Word(pp.nums) + '.' + pp.Word(pp.nums))\
    .setParseAction(lambda toks: float(toks[0]))
Number = Float | Integer

TableValue = Number | pp.Word(pp.alphanums + '_') | pp.QuotedString('"')
TableStartAt = Integer.setResultsName('tblStartAt')
ColName = pp.Word(pp.alphanums + '_[]') | pp.QuotedString('"')

EndOfTable = pp.Suppress(pp.Literal(r'\x1a##END##'))
FirstLine = Integer.setResultsName('colNum') \
            + (Comment.setResultsName('tableCaption') | pp.lineEnd)
HeadLine = (pp.Suppress('!') + pp.Optional(TableStartAt) + ','
            + pp.delimitedList(ColName).setResultsName('colNames')) \
    .setResultsName('header')
TableLine = pp.Suppress('*,') + pp.delimitedList(TableValue)\
    .setResultsName('rows', True)
Table = pp.OneOrMore(TableLine).setResultsName('table')

Product = ColName.setResultsName('productCode') \
              + pp.Optional(pp.Suppress('#')
                            + ColName.setResultsName('subProductCode'))

VariableTypesLine = pp.Suppress('VARIABLE_TYPES,') +\
                    pp.Suppress(pp.ZeroOrMore(pp.White())+pp.Optional('T1,'))\
                    + pp.delimitedList(TableValue).setResultsName('varTypes')

RowCountCheckerLine = pp.Suppress('NUMLINES,')\
                      + Integer.setResultsName('rowNum') + pp.lineEnd
OutputFormatLine = pp.Suppress('Output_Format,') + pp.restOfLine\
    .setResultsName('outputFormat')
MPTableName = pp.restOfLine.setResultsName('tableCaption')


class ProphetTableType(enum.Enum):
    GenericTable = 0
    ModelPoint = 1
    Sales = 2
    Parameter = 3
    Probability = 4
    TableOfTable = 5


class ProphetTable:
    """ Class used for **Prophet** table management and reading.

    """

    # Generic Table Parser the parsing result has follow keys:
    #     - tableCaption: the caption and description for the table
    #     - colNum: how many columns does the table have, index not include
    #     - tblStartAt: the first column of data start at where, the value minus
    #       1 is the num of indexes
    #     - colNames: names of indexes and data columns
    GenericTableParser = FirstLine + HeadLine + Table

    ModelPointTableParser = MPTableName + OutputFormatLine \
        + RowCountCheckerLine + VariableTypesLine + HeadLine + Table

    ProbabilityTableParser = FirstLine + HeadLine + pp.OneOrMore(
        pp.Suppress(Integer) + (pp.delimitedList(Number).setResultsName('rows', True) | Number) +
        pp.Suppress(Integer)
    ).setResultsName('table')

    PROPHET_NUMPY_VAR_MAP = {
        'I': np.int,
        'S': np.int,
        'N': np.float,
    }

    # mp special column
    MP_BATCH_COL = 'SPCODE'
    MP_AGE_COL = 'AGE_AT_ENTRY'
    MP_SEX_COL = 'SEX'
    MP_BFT_COL = 'POL_TERM_Y'
    MP_PMT_COL = 'PREM_PAYBL_Y'
    MP_MTH_COL = 'DURATIONIF_M'

    _ALL_TABLES_ = {}  # type: Dict[ProphetTable]

    _BANDED_ATTRIBUTE = {'loc', 'iloc'}

    def __init__(self, tablename, tabletype, dataframe, *, cache=True):
        self.tablename = tablename
        self.tabletype = tabletype
        self.dataframe: pd.DataFrame = dataframe
        if cache and self.tabletype != ProphetTableType.ModelPoint:
            if self.tablename in self._ALL_TABLES_:
                warnings.warn("tablename :{} all ready exists".format(self.tablename),
                              RuntimeWarning)
            self._ALL_TABLES_[self.tablename] = self

    def __getattr__(self, name):
        if name in self._BANDED_ATTRIBUTE:
            raise AttributeError
        rst = getattr(self.dataframe, name)
        if isinstance(rst, pd.DataFrame) or isinstance(rst, pd.Series):
            return ProphetTable("{}.{}".format(self.tablename, name), self.tabletype,
                                rst, cache=False)
        else:
            return rst

    def __getitem__(self, item):
        try:
            rst = self.dataframe.loc[item]
        except KeyError:
            try:
                rst = self.dataframe.iloc[item]
            except TypeError:
                warnings.warn("column selected from ProphetTable: {}".format(self.tablename),
                              RuntimeWarning)
                rst = self.dataframe[item]

        if isinstance(rst, pd.DataFrame) or isinstance(rst, pd.Series):
            return ProphetTable("{}[{}]".format(self.tablename, item), self.tabletype,
                                rst, cache=False)
        elif self.tabletype == ProphetTableType.TableOfTable and isinstance(rst, str) \
                and not self.plain_select:
            return self._ALL_TABLES_.get(rst, rst)
        else:
            return rst

    @classmethod
    def clear_cache(cls):
        cls._ALL_TABLES_ = {}

    @classmethod
    def translate_var_type(cls, prophet_var_types: list):
        return get(prophet_var_types, cls.PROPHET_NUMPY_VAR_MAP)

    @staticmethod
    def guess_tablename(file)->str:
        rst = os.path.split(file)[1].split('.')[0]  # type: str
        return rst

    def __repr__(self):
        if self.tablename is None:
            return repr(self.dataframe)
        return "TABLE_NAME: {}".format(self.tablename) + '\n' \
               "TABLE_TYPE: {}".format(self.tabletype) + '\n' \
               + repr(self.dataframe)

    @classmethod
    def get_table(cls, tablename):
        return cls._ALL_TABLES_[tablename]

    @classmethod
    def read_generic_table(cls, file, tabletype=0, tablename=None):
        """ Read file as *Prophet Generic Table*

        :param Union[str, File] file: path to the file
        :param Optional[str] tablename: if not provided name is guessed from `file`
        :param Optional[ProphetTableType] tabletype: table type
        :rtype: ProphetTable
        """
        if tablename is None:
            tablename = cls.guess_tablename(file)
        with open(file, errors='ignore') as file:
            lines = file.readlines()
            s = "".join(lines)
            try:
                p_rst = cls.GenericTableParser.parseString(s)
            except pp.ParseException as e:
                print(file)
                raise e

            table = cls.parse_result2table(p_rst['rows'])
            col_names = list(p_rst['colNames'])
            n_col = p_rst['colNum']
            index_num = p_rst['tblStartAt'] - 1
            df = pd.DataFrame(data=table, columns=col_names)
            df.set_index(col_names[:index_num], inplace=True)
            assert df.shape[1] == n_col, "column missing"
            return cls(tablename, tabletype, df)

    @staticmethod
    def parse_result2table(p_rst):
        return list((list(r) for r in p_rst))

    @classmethod
    def read_modelpoint_table(cls, file, tablename=None):
        """Read file as *Prophet ModelPoint Table*

        :param Union[str, File] file: path to the file
        :param Optional[str] tablename: if not provided name is guessed from `file`
        :rtype: ProphetTable
        """
        if tablename is None:
            tablename = cls.guess_tablename(file)
        with open(file, errors='ignore') as file:
            lines = file.readlines()
            s = "".join(lines)
            p_rst = cls.ModelPointTableParser.parseString(s)
            table = cls.parse_result2table(p_rst['rows'])
            column_names = list(p_rst['colNames'])
            df = pd.DataFrame(data=table, columns=column_names)
            dtypes = cls.translate_var_type(list(p_rst['varTypes']))
            df.astype(dtype=dict(zip(column_names, dtypes)), copy=False)
            df.set_index(cls.MP_BATCH_COL, inplace=True)
            origin_row_num, real_row_num = p_rst['rowNum'], len(df)
            assert real_row_num == origin_row_num,\
                "The table should have {} points but {} points are read".format(origin_row_num, real_row_num)
            return cls(tablename, ProphetTableType.ModelPoint, df)

    @classmethod
    def read_parameter_table(cls, file, tablename=None):
        """
        A simple reader of *Prophet Parameter Table*

        :param Union[str, File] file: path to the file
        :param Optional[str] tablename: if not provided name is guessed from `file`
        :rtype: ProphetTable
        """
        return cls.read_generic_table(file, ProphetTableType.Parameter, tablename)

    @classmethod
    def read_table_of_table(cls, file, tablename=None):
        """
        A simple reader of *Prophet TableOfTable*

        :param Union[str, File] file: path to the file
        :param Optional[str] tablename: if not provided name is guessed from `file`
        :rtype: ProphetTable
        """
        return cls.read_generic_table(file, ProphetTableType.TableOfTable, tablename)

    @classmethod
    def read_probability_table(cls, m_file, f_file=None, tablename=None):
        """ Read as Probability Table

        :param m_file: path of file of probability of male.
        :param f_file:  path of file of probability of female, Default: `m_file`
        :param tablename: Default: if f_file is not None, then the largest common sub string
            is used as tablename else guessed from `m_file`.
        :rtype: ProphetTable
        """
        if tablename is None:
            m_tablename = cls.guess_tablename(m_file)
            if f_file is not None:
                from difflib import SequenceMatcher
                f_tablename = cls.guess_tablename(f_file)
                match = SequenceMatcher(None, m_tablename, f_tablename)\
                    .find_longest_match(0, len(m_tablename), 0, len(f_tablename))
                tablename = m_tablename[match.a: match.a + match.size].rstrip('_')
                assert tablename, "it seems file names can't match"
            else:
                tablename = m_tablename

        def read_array(file):
            with open(file, errors='ignore') as file:
                lines = file.readlines()
                s = "".join(lines)
                p_rst = cls.ProbabilityTableParser.parseString(s)
                table_ = cls.parse_result2table(p_rst['rows'])
                return np.array(table_, dtype=np.double)

        m_table = read_array(m_file)
        if f_file is not None:
            f_table = read_array(f_file)
            table = np.hstack((m_table, f_table))
        else:
            table = m_table
        if table.shape[1] == 1:
            table = table.repeat(2, axis=1)
        else:
            assert table.shape[1] == 2,\
                "currently fac file with more than one rate column is not supported"

        df = pd.DataFrame(data=table, columns=['m', 'f'])
        return cls(tablename, ProphetTableType.Probability, df)

    def as_modelpoint(self, klass=None, *args_of_klass, **kwargs_of_klass):
        """Convert model point table to model point data set, the result is an instance of klass

        :param klass: class of the data set result, Default :class:`~corona.mp.ModelPointSet`
        :param args_of_klass: additional position arguments provided to `klass`
        :param kwargs_of_klass: additional key word arguments provided to `klass`
        :return: model point data set
        """
        assert self.tabletype == ProphetTableType.ModelPoint,\
            "Only Model Point Table can be convert to ModelPointDataSet"
        from corona.mp import ModelPointSet
        if klass is None:
            klass = ModelPointSet
        else:
            if not issubclass(klass, ModelPointSet):
                warnings.warn("{} is not subclass of {}".format(klass, ModelPointSet))
        return klass(self.dataframe, *args_of_klass, **kwargs_of_klass)

    # noinspection PyArgumentList
    def as_probability(self, kx=None, klass=None, *args_of_klass, **kwargs_of_klass):
        """Convert probability table to Probability, the result is an instance of klass.

        :param klass: class of result, default :class:`~corona.core.prob.Probability`
        :param Union[str, ProphetTable, list, ndarray] kx: for detail see default of `klass`
        :param args_of_klass: additional position arguments provided to `klass`
        :param kwargs_of_klass: additional key word arguments provided to `klass`
        :return: probability
        """
        assert self.tabletype == ProphetTableType.Probability, \
            "Only Probability Table can be convert to Probability"
        qx = self.values.T

        if isinstance(kx, str):
            kx = self.get_table(kx)
            assert kx.tabletype == ProphetTableType.ModelPoint,\
                "Only Model Point Table can be convert to kx"

        if isinstance(kx, ProphetTable):
            kx = kx.values.T

        from corona.core.prob import Probability
        if klass is None:
            klass = Probability
        else:
            if not issubclass(klass, Probability):
                warnings.warn("{} is not subclass of {}".format(klass, Probability))

        return klass(qx, kx, *args_of_klass, name=self.tablename, **kwargs_of_klass)

    # noinspection PyArgumentList
    def as_selection_factor(self, klass=None, *args_of_klass, **kwargs_of_klass):
        """Convert probability table to Selection Factor, the result is an instance of klass.

        :param klass: class of result, default :class:`~corona.core.prob.SelectionFactor`
        :param args_of_klass: additional position arguments provided to `klass`
        :param kwargs_of_klass: additional key word arguments provided to `klass`
        :return: probability
        """
        assert self.tabletype in [ProphetTableType.Probability, ProphetTableType.GenericTable], \
            "Only Probability Table can be convert to SelectionFactor"
        fac = self.values.T
        from corona.mm import LinearSensitivity
        if klass is None:
            klass = LinearSensitivity
        else:
            if not issubclass(klass, LinearSensitivity):
                warnings.warn("{} is not subclass of {}".format(klass, LinearSensitivity))

        return klass(fac, *args_of_klass, name=self.tablename, **kwargs_of_klass)

    @classmethod
    def cache_to_hdf(cls, path):
        """Save all cached tables to hdf file

        :param str path: path to hdf file
        """
        with pd.HDFStore(path) as store:
            for k, v in cls._ALL_TABLES_.items():
                if v.tabletype == ProphetTableType.Parameter:
                    group = 'param/'
                elif v.tabletype == ProphetTableType.GenericTable:
                    group = 'gene/'
                elif v.tabletype == ProphetTableType.TableOfTable:
                    group = 'tot/'
                elif v.tabletype == ProphetTableType.Probability:
                    group = 'prob/'
                else:
                    raise ValueError(v.tabletype)
                try:
                    store.append(group + k, v.dataframe)
                except TypeError:
                    store.append(group + k, v.dataframe.T, min_itemsize={'values': 50})
    
    @classmethod
    def cache_to_sqlite(cls, path):
        with sqlite3.connect(path) as conn:
            for tablename, table in cls._ALL_TABLES_.items():
                table.dataframe.to_sql(tablename, conn)
            pd.DataFrame([(tbl.tablename, tbl.tabletype.name) for tbl in cls._ALL_TABLES_.values()],
                         columns=['name', 'type']).to_sql("table_info", conn)


read_generic_table = ProphetTable.read_generic_table
read_parameter_table = ProphetTable.read_parameter_table
read_probability_table = ProphetTable.read_probability_table
read_modelpoint_table = ProphetTable.read_modelpoint_table
read_table_of_table = ProphetTable.read_table_of_table


def read_assumption_tables(folder, *, tot_pattern=None,
                           param_pattern=None, prob_folder=None, gender_suffix=None,
                           exclude_folder=None,
                           exclude_pattern=None, clear_cache=False):
    """ Read All tables in folder. First exclude_folder and exclude_pattern are
    used to test if the table should be ignored, then prob_folder is used to
    test if the table should be read as probability table, at last tot_pattern is
    used to test if table should be read as a table of tables.
    If all tests failed, the table will be read as a generic table.

    .. note::

        Links are treated as folders, it can lead to infinite recursion if a link points
        to a parent directory of itself.


    >>> read_assumption_tables("./TABLES", prob_folder='MORT',
    ...                        param_pattern=r'PARAMET_.+',
    ...                        tot_pattern='GLOBAL|.*_TABLE_CONFIG',
    ...                        exclude_folder='CROSS_LASTVAL',
    ...                        exclude_pattern='PRICING_AGE_TBL',
    ...                        clear_cache=True)

    :param str folder: path of the folder
    :param str tot_pattern: regular expression of tablename of table of tables
    :param str param_pattern: regular expression of tablename of parameter table
    :param str prob_folder:  name(not path) of sub folder in which all tables are
        recognized as probability table
    :param str exclude_folder: name(not path) of sub folder in which all tables are
        ignored
    :param str exclude_pattern: regular expression of table name that should be ignored
    :param bool clear_cache: if True cached tables will be cleared before reading, default False
    """
    import re
    import os
    get_name = ProphetTable.guess_tablename

    if clear_cache:
        ProphetTable.clear_cache()

    def compile_re(p):
        if isinstance(p, str):
            _p = re.compile(p)
            return lambda file: _p.fullmatch(get_name(file))
        elif p is None:
            return lambda _: False
        else:
            raise TypeError(p)

    tot_pattern = compile_re(tot_pattern)
    param_pattern = compile_re(param_pattern)
    exclude_pattern = compile_re(exclude_pattern)

    if prob_folder is None:
        def prob_folder(_): return False
    elif isinstance(prob_folder, str):
        from pathlib import Path
        _f = prob_folder
        def prob_folder(d): return _f in Path(d).parts
    else:
        raise TypeError(prob_folder)

    if exclude_folder is None:
        def exclude_folder(_): return False
    elif isinstance(exclude_folder, str):
        from pathlib import Path
        _e = exclude_folder
        def exclude_folder(d): return _e in Path(d).parts
    else:
        raise TypeError(exclude_folder)

    rst = {
        'prob': [],
        'tot': [],
        'param': [],
        'gen': []
    }

    if gender_suffix is not None:
        male_suffix, female_suffix = gender_suffix
        def male_prob(name):
            return name.endswith(male_suffix)

        def female_prob(name):
            return name.endswith(female_suffix)
    else:
        male_suffix = female_suffix = ''
        def male_prob(_): return False

        def female_prob(_): return False

    single_gender_prob = {}

    for root, dirs, files in os.walk(folder, followlinks=True):
        for f in files:
            f = os.path.join(root, f)
            if exclude_pattern(f):
                continue
            elif exclude_folder(f):
                continue
            elif prob_folder(f):
                f_name = get_name(f)
                if male_prob(f_name):
                    try:
                        single_gender_prob[f_name[:-(len(male_suffix))]]['m_file'] = f
                    except KeyError:
                        single_gender_prob[f_name[:-(len(male_suffix))]] = {'m_file': f}
                elif female_prob(f_name):
                    try:
                        single_gender_prob[f_name[:-(len(male_suffix))]]['f_file'] = f
                    except KeyError:
                        single_gender_prob[f_name[:-(len(male_suffix))]] = {'f_file': f}
                else:
                    rst['prob'].append(read_probability_table(f))
            elif param_pattern(f):
                rst['param'].append(read_parameter_table(f))
            elif tot_pattern(f):
                rst['tot'].append(read_table_of_table(f))
            else:
                rst['gen'].append(read_generic_table(f))

    for fs in single_gender_prob.values():
        rst['prob'].append(read_probability_table(**fs))

    return rst


def prlife_read(folder, clear_cache=True):
    """ Read All Pear River Life Assumption tables in folder

    >>> prlife_read('./TABLES')

    :param str folder: path of the folder
    :param bool clear_cache: if True, all cached tables before reading default True
    :return: dict of tables with tablename as key
    :rtype: dict
    """
    return read_assumption_tables(folder, prob_folder='MORT',  gender_suffix=('_M', '_F'),
                                  param_pattern=r'PARAMET_.+',
                                  tot_pattern='GLOBAL|.*_TABLE_CONFIG',
                                  exclude_folder='CROSS_LASTVAL',
                                  exclude_pattern='PRICING_AGE_TBL',
                                  clear_cache=clear_cache)
