from typing import Dict
import copy
import dataclasses


@dataclasses.dataclass
class BaseConf:
    def __init_subclass__(cls, **_):
        '''Applies the dataclass decorator to all children of this class.'''
        return dataclasses.dataclass(cls)

    def to_dict(self) -> Dict:
        '''Recursively converts all configuration fields to dictionaries.'''
        return dataclasses.asdict(self)

    def update(self, d: Dict) -> 'BaseConf':
        '''Create a new configuration with updated fields.'''
        return recursive_update(self, d)

    @classmethod
    def from_dict(cls, d: Dict):
        '''Instanciate nested configurations from a dictionary'''
        return dataclass_from_dict(cls, d)


def dataclass_from_dict(cls, d: Dict):
    '''Does not handle collections or mappings of dataclasses.'''
    if dataclasses.is_dataclass(cls):
        fieldtypes = {f.name: f.type for f in dataclasses.fields(cls)}
        args = {}
        for k in d:
            if k not in fieldtypes:
                raise ValueError(f'{k} not in {fieldtypes.keys()}')
            args[k] = dataclass_from_dict(fieldtypes[k], d[k])
        return cls(**args)

    return copy.deepcopy(d)


def recursive_update(conf: BaseConf, d: Dict) -> BaseConf:
    out = copy.deepcopy(conf)
    fieldtypes = {f.name: f.type for f in dataclasses.fields(conf)}
    for k, v in d.items():
        if k not in fieldtypes:
            raise ValueError(f'{k} not in {fieldtypes.keys()}')
        type_ = fieldtypes[k]
        if not isinstance(type_, type):  # try to recover the underlying type
            type_ = type_.__origin__
        if isinstance(type_, type):  # cannot handle typing.Generic (Union, Optional, etc.)
            if issubclass(type_, BaseConf):
                if not isinstance(v, dict):
                    raise ValueError(f'Incorrect type of {k}: {v} vs {dict}')
                v = recursive_update(getattr(out, k), v)
            if not isinstance(v, type_):
                raise ValueError(f'Incorrect type of {k}: {type(v)} vs {type_}')
        setattr(out, k, v)
    return out
