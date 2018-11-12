from typing import Dict, Optional, Type, List, Union
from collections import namedtuple
from .core import Clause, Contract
from dataclasses import dataclass, field, InitVar, fields, Field, MISSING
from importlib.util import spec_from_file_location, module_from_spec
from enum import Enum, auto
from torch import Tensor
import os
__all__ = ['ClauseTemplate', 'ContractTemplate', "WorkSpace", 'PyModule',
           'template', 'Template', 'field', 'InitVar', 'public_fields',
           'Field', 'MISSING']

template = dataclass


def public_fields(klass: Type[template])->List[Field]:
    return [f for f in fields(klass) if not f.name.startswith('_')]


class Template:
    pass


class ClauseTemplate(Template):

    def __call__(self, *args, **kwargs)->Clause:
        raise NotImplementedError()

    @classmethod
    def post_init(cls, field_name):
        pass


class ContractTemplate(Template):

    def __call__(self, *args, **kwargs) -> Contract:
        raise NotImplementedError()


@dataclass
class PyModule:
    name: str
    clauseTemp: Dict[str, Type[ClauseTemplate]]
    contractTemp: Dict[str, Type[ContractTemplate]]

    def __getattr__(self, item):
        try:
            return self.clauseTemp[item]
        except KeyError:
            return self.contractTemp[item]


class WorkSpace:
    path: str
    tensorStore: Dict[str, Tensor]
    pyModules: Dict[str, PyModule]

    def __init__(self, dir_path):
        self.path = dir_path
        self.tensorStore = {}
        self.pyModules = {}
        self.reload()

    def clearTensorStore(self):
        self.tensorStore.clear()

    def reload(self):
        self.pyModules.update({module.name: module for module in (
            self.load_py_module(file[:-3]) for file in os.listdir(self.path)
            if file.endswith('.py')) if module})
        return self

    def load_module(self, name: str):
        route = name.split(".")
        route[-1] = route[-1] + '.py'
        spec = spec_from_file_location(name, os.path.join(self.path, *route))
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def load_py_module(self, name)->Optional[PyModule]:
        module = self.load_module(name)
        if not hasattr(module, '__all__'):
            return None
        clauses = {name: getattr(module, name)
                   for name in module.__all__
                   if issubclass(getattr(module, name), ClauseTemplate)}
        contracts = {name: getattr(module, name)
                     for name in module.__all__
                     if issubclass(getattr(module, name), ContractTemplate)}
        if clauses or contracts:
            return PyModule(name, clauses, contracts)

    def __getattr__(self, item):
        try:
            return self.pyModules[item]
        except KeyError:
            return self.tensorStore[item]


# ============ utils for generating clauses and contracts from prophet tables
class FieldProphetType(Enum):
    Cell = auto()
    Linker = auto()
    Table = auto()


ProphetReaderMeta = namedtuple(
    'FieldProphetMeta',
    ['type', 'reader', 'misc']
)


@dataclass
class LinkInfo:
    """ meta info, indicating that the field if used as a foreign key to link 2
    prophet table together.

    :attr:`from` tablename of `from` table
    :attr:`to` tablename of `to` table
    """
    fromTableName: str
    toTableName: str


def foreignKey(*, default=MISSING, default_factory=MISSING, init=True, repr=True,
               hash=None, compare=True, from_table: str, to_table: str, metadata=None):
    metadata.update(prophet={
        'type': FieldProphetType.Linker,
        'meta': LinkInfo(from_table, to_table)
    })
    return field(default=default, default_factory=default_factory,
                 init=init, repr=repr, hash=hash, compare=compare,
                 metadata=metadata)


@dataclass
class CellReader:
    table: Union[str, Field]
    rowIdx: Optional[Field] = None




