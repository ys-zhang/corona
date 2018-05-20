"""
utils for Sungard Prophet Format file support for example .fac .RPT file
"""
import pyparsing as pp
import pandas as pd
import numpy as np
from cytoolz import get
import enum

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
ColName = pp.Word(pp.alphanums + '_')

EndOfTable = r'\x1a##END##'
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

VariableTypesLine = pp.Suppress('VARIABLE_TYPES,')\
                    + pp.delimitedList(TableValue).setResultsName('varTypes')
RowCountCheckerLine = pp.Suppress('NUMLINES,')\
                      + Integer.setResultsName('rowNum') + pp.lineEnd
OutputFormatLine = pp.Suppress('Output_Format,') + pp.restOfLine\
    .setResultsName('outputFormat')
MPTableName = pp.restOfLine.setResultsName('tableCaption')


class ProphetTableType(enum.Enum):
    GenericTable = enum.auto()
    ModelPoint = enum.auto()
    Sales = enum.auto()
    Parameter = enum.auto()


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

    def __init__(self, tablename, tabletype: ProphetTableType, dataframe):
        self.tablename = tablename
        self.tabletype = tabletype
        self.dataframe = dataframe

    @classmethod
    def translate_var_type(cls, prophet_var_types: list):
        return get(prophet_var_types, cls.PROPHET_NUMPY_VAR_MAP)

    @classmethod
    def read_generic_table(cls, file)->pd.DataFrame:
        """ A simple reader of *Prophet Generic Table*

        :param Union[str, File] file:
        :return: dataframe
        """
        try:
            file = file.readlines()
        except AttributeError:
            file = open(file, errors='ignore').readlines()
        s = "".join(file)
        p_rst = cls.GenericTableParser.parseString(s)
        table = cls.parse_result2table(p_rst['rows'])
        col_names = list(p_rst['colNames'])
        n_col = p_rst['colNum']
        index_num = p_rst['tblStartAt'] - 1
        dtypes = cls.translate_var_type(list(p_rst['varTypes']))
        df = pd.DataFrame(data=table, columns=col_names)
        df.astype(dtype=dict(zip(col_names, dtypes)), copy=False)
        df.set_index(col_names[:index_num], inplace=True)
        assert df.shape[1] == n_col, "column missing"
        return df

    @staticmethod
    def parse_result2table(p_rst):
        return list((list(r) for r in p_rst))

    @classmethod
    def read_modelpoint_table(cls, file):
        """
        A simple reader of *Prophet ModelPoint Table*

        :param Union[str, File] file:
        :return: dataframe
        """
        try:
            file = file.readlines()
        except AttributeError:
            file = open(file, errors='ignore').readlines()
        s = "".join(file)
        p_rst = cls.ModelPointTableParser.parseString(s)
        table = cls.parse_result2table(p_rst['rows'])
        column_names = list(p_rst['colNames'])
        df = pd.DataFrame(data=table, columns=column_names)
        df.set_index(cls.MP_BATCH_COL, inplace=True)
        origin_row_num, real_row_num = p_rst['rowNum'], len(df)
        assert real_row_num == origin_row_num,\
            f"The table should have {origin_row_num} points " \
            f"but {real_row_num} points are read"
        return df

    @classmethod
    def read_parameter_table(cls, file):
        """
        A simple reader of *Prophet Parameter Table*

        :param Union[str, File] file:
        :return: dataframe
        """
        return cls.read_generic_table(file)


# ================= MINI LANG =============
TIME = pp.Literal('T') | pp.Literal('t')
ID = pp.Combine(pp.Word(pp.alphas + '_')
                + pp.ZeroOrMore(pp.Word(pp.alphanums + '_')))
NUMBER = pp.Regex(r'-?[0-9]*[.]?[0-9]+').\
    setParseAction(lambda tks: float(tks[0]) if '.' in tks[0] else int(tks[0]))
STR = pp.quotedString("'") | pp.quotedString('"')
LITERAL = NUMBER | STR
PY_ID = pp.Combine('@' + ID + pp.OneOrMore("." + ID))
