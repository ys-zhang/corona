"""
utils for Sungard Prophet Format file support for example .fac .RPT file
"""
import pyparsing as pp
import pandas as pd
import numpy as np
from cytoolz import get
import enum
import os
import warnings

__all__ = ['ProphetTable']


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


class ProphetTableType(enum.IntEnum):
    GenericTable = 0
    ModelPoint = 1
    Sales = 2
    Parameter = 3
    Probability = 4


class ProphetTable:
    """ Class used for **Prophet** table management and reading.

    """

    GenericTableParser = FirstLine + HeadLine + Table
    """Generic Table Parser the parsing result has follow keys:
    
        - tableCaption: the caption and description for the table
        - colNum: how many columns does the table have, index not include
        - tblStartAt: the first column of data start at where, the value minus 
          1 is the num of indexes
        - colNames: names of indexes and data columns  
    """

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

    _ALL_TABLES_ = {}

    def __init__(self, tablename, tabletype, dataframe):
        self.tablename = tablename
        self.tabletype = tabletype
        self.dataframe = dataframe
        if self.tabletype != ProphetTableType.ModelPoint:
            if self.tablename in self._ALL_TABLES_:
                warnings.warn(f"tablename :{self.tablename} all ready exists")
            self._ALL_TABLES_[self.tablename] = self

    def __getattr__(self, name):
        return getattr(self.dataframe, name)

    @classmethod
    def translate_var_type(cls, prophet_var_types: list):
        return get(prophet_var_types, cls.PROPHET_NUMPY_VAR_MAP)

    @staticmethod
    def _guess_tablename(file):
        return os.path.split(file)[1].split('.')[0]

    def __repr__(self):
        if self.tablename is None:
            return repr(self.dataframe)
        return self.tablename + '\n' + repr(self.dataframe)

    @classmethod
    def get_table(cls, tablename):
        return cls._ALL_TABLES_[tablename]

    @classmethod
    def read_generic_table(cls, file, tabletype=0, tablename=None):
        """ Read file as *Prophet Generic Table*

        :param Union[str, File] file: path to the file
        :param Optional[str] tablename: if not provided name is guessed from file
        :param Optional[ProphetTableType] tabletype: table type
        :return: dataframe
        """
        if tablename is None:
            tablename = cls._guess_tablename(file)
        with open(file, errors='ignore') as file:
            lines = file.readlines()
            s = "".join(lines)
            p_rst = cls.GenericTableParser.parseString(s)
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
        :param Optional[str] tablename: if not provided name is guessed from file
        :return: dataframe
        """
        if tablename is None:
            tablename = cls._guess_tablename(file)
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
                f"The table should have {origin_row_num} points but {real_row_num} points are read"
            return cls(tablename, ProphetTableType.ModelPoint, df)

    @classmethod
    def read_parameter_table(cls, file, tablename=None):
        """
        A simple reader of *Prophet Parameter Table*

        :param Union[str, File] file: path to the file
        :param Optional[str] tablename: if not provided name is guessed from file
        """
        return cls.read_generic_table(file, ProphetTableType.Parameter, tablename)

    @classmethod
    def read_probability(cls, m_file, f_file=None, tablename=None):
        if tablename is None:
            m_tablename = cls._guess_tablename(m_file)
            if f_file is not None:
                from difflib import SequenceMatcher
                f_tablename = cls._guess_tablename(f_file)
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

        :param klass: class of the data set result
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
                warnings.warn(f"{klass} is not subclass of {ModelPointSet}")
        return klass(self.dataframe, *args_of_klass, **kwargs_of_klass)

    def as_probability(self, kx=None, klass=None, *args_of_klass, **kwargs_of_klass):
        """Convert probability table to Probability, the result is an instance of klass.

        :param klass: class of result, default :class:`corona.core.prob.Probability`
        :param Union['str', ProphetTable, list, ndarray] kx: for detail see default of `klass`
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
                warnings.warn(f"{klass} is not subclass of {Probability}")

        return klass(qx, kx, *args_of_klass, name=self.tablename, **kwargs_of_klass)

    def as_selection_factor(self, klass=None, *args_of_klass, **kwargs_of_klass):
        """Convert probability table to Selection Factor, the result is an instance of klass.

        :param klass: class of result, default :class:`corona.core.prob.SelectionFactor`
        :param args_of_klass: additional position arguments provided to `klass`
        :param kwargs_of_klass: additional key word arguments provided to `klass`
        :return: probability
        """
        assert self.tabletype in [ProphetTableType.Probability, ProphetTableType.GenericTable], \
            "Only Probability Table can be convert to SelectionFactor"
        fac = self.values.T
        from corona.core.prob import SelectionFactor
        if klass is None:
            klass = SelectionFactor
        else:
            if not issubclass(klass, SelectionFactor):
                warnings.warn(f"{klass} is not subclass of {SelectionFactor}")

        return klass(fac, *args_of_klass, name=self.tablename, **kwargs_of_klass)
