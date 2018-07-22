"""Corona is a package aims to provide a structure and tools to help actuaries
 developing well organized models for different kind of insurance products for
 multiple purposes like valuate, profit testing, BP etc.

"""
__version__ = '0.1'

from . import prophet
from . import table
from . import conf
from . import utils
from . import mp
from . import core
from corona.core import contract
from corona.core import prob
from corona.core import discount
from corona.core import result
