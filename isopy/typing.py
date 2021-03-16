import functools
import typing
from typing import TypeVar, Union, Optional, Any, Literal, Callable, Generic, NewType
import inspect
from collections.abc import Callable as CallableType

__all__ = ['Coerce', 'check_input',
           'array_like', 'isopy_array_like', 'numpy_array_like',
           'scalar', 'keystring', 'keylist']

CHECK_INPUT = True
ISOPYTYPES = {}

#TODO special case for when single value shoudl always return list

def create_type(name, check_function=None):
    """
    Create and register a ``TypeVar``. Optionally a function for evaluating the *value* of the
    parameter can be passed. This function should take *value* and return *value* if it is
    correct type.
    """
    typevar = TypeVar(name)
    ISOPYTYPES[typevar] = check_function
    return typevar

class InputException(Exception):
    def copy(self):
        return self.__class__(self.name, self.message)

class InputTypeError(TypeError, InputException):
    def __init__(self, name, message=None):
        self.name = name
        self.message = message

    def __str__(self):
        return f"parameter '{self.name}': {self.message}"

    def from_types(self, value_type, valid_types):
        if type(valid_types) is type:
            valid_types = (valid_types,)

        vstring = f'{value_type.__name__}'
        tstrings = [f"'{valid_type.__name__}'" for valid_type in valid_types]

        if len(tstrings) == 1:
            tstring = tstrings[0]
        else:
            tstring = f'{", ".join(tstrings[:-1])} or {tstrings[-1]}'

        self.message = f"Got '{vstring}' expected {tstring}"

        return self


class InputValueError(ValueError, InputException):
    def __init__(self, name, message=None):
        self.name = name
        self.message = message

    def __str__(self):
        return f'parameter "{self.name}": {self.message}'

    def from_values(self, value, valid_values):
        if not type(valid_values) is tuple:
            valid_values = (valid_values,)

        vstring = f'{value}'
        tstrings = [f"'{valid_value}'" for valid_value in valid_values]

        if len(tstrings) == 1:
            tstring = tstrings[0]
        else:
            tstring = f'{", ".join(tstrings[:-1])} or {tstrings[-1]}'

        self.message = f"Got '{vstring}' expected {tstring}"

        return self


class IsopyType:
    @staticmethod
    def check_type(name, value, valid_types):
        return value


class Coerce(IsopyType, list):
    """
    Special isopy type that will attempt to coerce the parameter *value* into one of the
    types if the *value* is not already one of these types.

    **Note** Not compatible with other type checkers as it will look like a list.
    """
    @staticmethod
    def check_type(name, value, valid_types):
        for valid_type in valid_types:
            try:
                return check_type(name, value, valid_type)
            except:
                break
        for valid_type in valid_types:
            try:
                return valid_type(value)
            except:
                continue
        else:
            raise InputTypeError(name).from_dtypes(type(value), valid_types)


array_like = create_type('array_like')
isopy_array_like = create_type('isopy_array_like')
numpy_array_like = create_type('numpy_array_like')

keystring = create_type('keystring')
keylist = create_type('keylist')

scalar = create_type('keylist')

def check_input(func):
    """
    Decorator that enables type checking on the input of a function according to the annotation.
    """
    signature = inspect.signature(func)
    pmdict = signature.parameters
    pmlist = [p for p in pmdict.values()]

    @functools.wraps(func)
    def input_type_checker(*args, **kwargs):
        if kwargs.pop('type_check', CHECK_INPUT):
            bound = signature.bind(*args, **kwargs)
            try:
                args = [check_type(pmlist[i].name, arg, pmlist[i].annotation) for i, arg in
                        enumerate(bound.args)]
                kwargs = {key: check_type(pmdict[key].name, value, pmdict[key].annotation) for
                          key, value in bound.kwargs.items()}
            except InputException as input_exception:
                #This removes the superfluous traceback within the type check.
                raise input_exception.copy()


        return func(*args, **kwargs)

    return input_type_checker


def check_type(name, value, annotation):
    if annotation is inspect._empty or annotation is Any:
        return value

    if type(annotation) is type:
        if type(value) is annotation:
            return value
        else:
            raise InputTypeError(name).from_types(type(value), annotation)

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if type(annotation) is TypeVar:
        return check_typevar(name, value, annotation, args)

    if origin is CallableType:
        return check_callable(name, value, args)

    if origin is Union:
        return check_union(name, value, args)

    if origin is list:
        value = check_type(name, value, list)
        return check_list(name, value, args)

    if origin is tuple:
        value = check_type(name, value, tuple)
        return check_tuple(name, value, args)

    if origin is dict:
        value = check_type(name, value, dict)
        return check_dict(name, value, args)

    if origin is Literal:
        return check_literal(name, value, args)

    if issubclass(origin, IsopyType):
        return origin.check_type(name, value, args)

    print('not caught', name, value, annotation, origin, args)
    return value

#TODO generic typevar
def check_typevar(name, value, typevar, args):
    function = ISOPYTYPES.get(typevar, None)
    if function is not None:
        try:
            return function(value)
        except ValueError as err:
            raise InputValueError(name, str(err)) from err
        except TypeError as err:
            raise InputTypeError(name, str(err)) from err
        except Exception as err:
            raise InputValueError(name, f"Got invalid value") from err
    else:
        return value


def check_callable(name, value, args):
    if callable(value):
        return value
    else:
        raise InputTypeError(name, f"Got '{type(value).__name__}' expected callable object")


def check_literal(name, value, valid_values):
    if value in valid_values:
        return value
    raise InputValueError(name).from_values(value, valid_values)

#TODO check key and value types
def check_dict(name, value, valid_types):
    return value


def check_union(name, value, valid_types):
    for valid_type in valid_types:
        try:
            return check_type(name, value, valid_type)
        except Exception:
            continue
    else:
        raise InputTypeError(name).from_types(type(value), valid_types)


def check_list(name, value, valid_types):
    if len(value) == 0:
        return value

    for i, item in enumerate(value):
        for valid_type in valid_types[:]:
            try:
                item = check_type(f'{name}[{i}]', item, valid_type, raise_exception=False)
            except Exception:
                continue
            else:
                args = (valid_type,)
                value[i] = item
                break
        else:
            raise InputTypeError(f"{name}[{i}]").from_types(type(value), valid_types)

    return value


def check_tuple(name, value, valid_types):
    if len(value) != len(valid_types):
        raise ValueError(f'parameter "{name}": Got {len(value)} values expected {len(args)}')

    return tuple(check_type(f'{name}[{i}]', item, valid_type) for i, (item, valid_type) in
                 enumerate(zip(value, valid_types)))