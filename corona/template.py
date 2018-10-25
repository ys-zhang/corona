import datetime
import getpass
from typing import List
from weakref import proxy
import contextlib

import jinja2
from corona import contract, prob
from corona.core import mm

__all__ = ['environment',
           'InModule',
           'Model', 'Call',
           'Slot', 'Attribute', 'PrimaryKey', 'Parameter', 'SubModule',
           'TProbability', 'TLinearSensitivity']


environment = jinja2.Environment(
    loader=jinja2.PackageLoader('corona', 'templates'),
    trim_blocks=True,
)


# =============== costume filters ================

def add_filter(func):
    func_name = func.__name__
    environment.filters[func_name] = func
    return func


@add_filter
def info(i: str):
    if i == 'now':
        return str(datetime.datetime.now())
    elif i == 'user':
        return getpass.getuser()
    else:
        raise ValueError(i)


@add_filter
def py_strip(value, strip=None):
    return str(value).strip(strip)


#  ================ Templates ===============


class Slot:
    __slots__ = ('name', '_default')
    NOT_DEFINED_FLAG = '__PLEASE_OH_PLEASE_JERRIE!!!__'

    def __init__(self, name):
        self.name = name
        self._default = self.NOT_DEFINED_FLAG

    def with_default(self, default):
        self._default = default
        return self

    @property
    def default(self):
        if self._default != self.NOT_DEFINED_FLAG:
            return self._default
        else:
            raise ValueError(f"{self.name} is not an Optional Argument")

    @property
    def slot_name(self):
        """ name as a slot in a TVariable Class """
        return self.name

    def __call__(self, value):
        """ convert value to a suitable form which can be pass to a template """
        return str(value)


class Attribute(Slot):
    pass


class PrimaryKey(Attribute):
    pass


class Parameter(Slot):
    pass


class SubModule(Slot):
    __slots__ = ('collection', 't_type')

    def __init__(self, name, collection=False):
        super().__init__(name)
        self.collection = collection
        self.t_type = None

    def as_t_type(self, t_type):
        self.t_type = t_type
        return self

    def __call__(self, value):
        if not self.collection:
            return super().__call__(value)
        else:
            raise NotImplementedError()


class Call:
    __slots__ = ('caller', 'args', 'kwargs')

    def __init__(self, caller, *args, **kwargs):
        self.caller: str = caller
        self.args: list = args
        self.kwargs: dict = kwargs

    def __str__(self):
        args = ",".join(self.args)
        kwargs = ",".join(f"{k}={str(v)}" for k, v in self.kwargs.items())
        if args and kwargs:
            return f"{self.caller}({args + ', ' + kwargs})"
        else:
            return f"{self.caller}({args + kwargs})"


Model = Call


class InModule(contextlib.ContextDecorator):
    _MODULES = {}
    DEFAULT_TEMPLATE = environment.get_template('base.jinja')  # type: jinja2.Template
    _MODULE_STACK = []  # module stack

    def __new__(cls, module, *args, **kwargs):
        try:
            return cls._MODULES[module]
        except KeyError:
            rst = super().__new__(cls, *args, **kwargs)
            cls._MODULES[module] = rst
            return rst

    def __init__(self, module: str, *, user=None, doc=None, alias=None,
                 jinja_template=None):
        """

        :param module: name of the module
        :param user: name of user if left blank then `getpass.getuser()` is used
        :param doc: doc string of the module
        :param alias: alias of the module
        :param jinja_template: jinja template used to generate the module,
            if left blank, the default template is used.
        """
        if module.endswith('.py'):
            module = module[:-3]
        self._module = module
        self._prev_module = None
        self._init_template_var(user, doc)
        if not hasattr(self, '_variables'):
            self._variables = {}
        if not hasattr(self, '_alias'):
            self._alias = alias
        if not hasattr(self, '_template'):
            if jinja_template is not None:
                jinja_template = environment.get_template(jinja_template)
            self._template = jinja_template
        if not hasattr(self, 'dependencies'):
            self.dependencies = set()

    def _init_template_var(self, user, doc):
        if doc is not None:
            if not hasattr(self, 'doc'):
                self.doc = doc
            else:
                self.doc += '\n' + doc
        if not hasattr(self, 'user'):
            if user is None:
                self.user = info('user')
            else:
                self.user = user

    @property
    def template(self)->jinja2.Template:
        if self._template is None:
            return self.DEFAULT_TEMPLATE
        else:
            return self._template

    @property
    def name(self):
        return self._module

    @property
    def alias(self):
        return self.name if self._alias is None else self._alias

    def __enter__(self):
        self._prev_module = TVariable._CURRENT_MODULE
        TVariable._CURRENT_MODULE = self
        self._MODULE_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        TVariable._CURRENT_MODULE = self._prev_module
        self._prev_module = None
        self._MODULE_STACK.pop()

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, InModule):
            return self.name == other.name
        else:
            return False

    def __str__(self):
        return self._module

    @property
    def code(self):
        return self.template.render(module=self)

    @classmethod
    def compile(cls, output_dir='.'):
        raise NotImplementedError()

    def __getitem__(self, item):
        if isinstance(item, tuple):
            item = "_".join(item)
        return self._variables[item]

    def __setitem__(self, key, value):
        self._variables[key] = value

    def __iter__(self):
        yield from self._variables.items()

    @property
    def variables(self):
        yield from self._variables.values()


# ========================== TVariables =======================


class TVariable:
    """ Type of Data that will be pass to a jinja2 template and rendered by
    the macro `pycode`.
    """
    SLOTS: List[Slot] = []
    TARGET: str  # name of the target class

    _CURRENT_MODULE: InModule = None  # don't override this

    def __init_subclass__(cls, target=None, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.LOCAL_INSTANCES = set()

        # slots of the Module
        try:
            slots = cls.SLOTS
        except AttributeError:
            slots = target.SLOTS
        cls.__slots__ = tuple(s[0].slot_name if isinstance(s, list)
                              else s.slot_name for s in slots)

        # TARGET
        if hasattr(cls, "TARGET"):
            pass
        elif target is not None:
            cls.TARGET = target.__name__
        else:
            cls.TARGET = cls.__name__[1:]

    def __init__(self, *args, **kwargs):
        if isinstance(self.SLOTS[0], list):
            setattr(self, self.__slots__[0], args)
            for s in self.SLOTS[1:]:
                setattr(self, s.slot_name, kwargs.get(s.slot_name, s.default))
        else:
            for s, val in zip(self.SLOTS[:len(args)], args):
                setattr(self, s.slot_name, val)
            for s in self.SLOTS[len(args):]:
                setattr(self, s.slot_name, kwargs.get(s.slot_name, s.default))
        self._init_primary_key()
        self._init_module()

    def _init_primary_key(self):
        try:
            self.primary_key = "_".join(getattr(self, pk.name)
                                        for pk in filter(lambda a: isinstance(a, PrimaryKey),
                                                         self.SLOTS))
        except (AttributeError, TypeError):
            self.primary_key = None

    def _init_module(self):
        if self.current_module() is not None:
            self.module = proxy(self.current_module())  # type: InModule
            """ module the variable is defined in """
            if self.primary_key is not None:
                self.module[self.primary_key] = self
                # update module dependencies
                for md in (var.module for var in self.dependencies
                           if var.module is not None and var.module != self.module):
                    self.module.dependencies.add(md)
        else:
            self.module = None

    @classmethod
    def current_module(cls):
        return cls._CURRENT_MODULE

    @property
    def dependencies(self):
        for slot in self.SLOTS:
            if isinstance(slot, list) and isinstance(slot[0], SubModule):
                yield from getattr(self, slot[0].slot_name)
            elif isinstance(slot, SubModule):
                yield getattr(self, slot.slot_name)

    def __hash__(self):
        return hash(self.primary_key)

    def __eq__(self, other):
        try:
            return self.primary_key == other.primary_key and \
                   self.__class__ == other.__class__
        except Attribute:
            return super().__eq__(other)


class TLinearSensitivity(TVariable, target=mm.LinearSensitivity):
    SLOTS = [
        Parameter('weight').with_default(None),
        Parameter('bias').with_default(None),
        Attribute('mm').with_default(False),
        PrimaryKey('name').with_default(None)
    ]


class TProbability(TVariable, target=prob.Probability):
    SLOTS = [
        PrimaryKey('name'),
        Parameter('qx'),
        Parameter('kx').with_default(None),
        SubModule('sens_model').as_t_type(TLinearSensitivity).with_default(None),
        Attribute('sens_type').with_default(None),
    ]


class TClause(TVariable, target=contract.Clause):
    SLOTS = [
        PrimaryKey('name'),
        SubModule('ratio_tables'),
        SubModule('base'),
        Attribute('t_offset'),
        SubModule('mth_converter').with_default(None),
        SubModule('prob_tables').with_default(None),
        Attribute('default_context').with_default('DEFAULT'),
        Attribute('virtual').with_default(False),
        Attribute('contexts_exclude').with_default(()),
    ]


class TAClause(TClause, target=contract.AClause):
    pass


class TClauseGroup(TVariable, target=contract.ClauseGroup):
    pass


class TParallelGroup(TVariable, target=contract.ParallelGroup):
    SLOTS = [
        [SubModule('clause').as_t_type(TClause)],
        PrimaryKey('name').with_default(None),
        Attribute('t_offset'),
    ]


class TSequentialGroup(TVariable, target=contract.SequentialGroup):
    SLOTS = [
        [SubModule('clause').as_t_type(TClause)],
        SubModule('copula').with_default(None),
        PrimaryKey('name').with_default(None),
    ]


class TContract(TVariable, target=contract.Contract):
    SLOTS = [
        PrimaryKey('name'),
        SubModule('clauses')
    ]


if __name__ == '__main__':
    with InModule('prob') as p:
        p1 = TProbability('pad1', [1, 2], [3, 4], TLinearSensitivity([[1, 2], [3, 4]]), 'table')
        p2 = TProbability('p2', [1, 2], sens_model=Model('md1', 'arg1', kwarg1=Model('md1', 'arg1', kwarg1='kwarg1')),
                          sens_type='table')
    print(p.code)
