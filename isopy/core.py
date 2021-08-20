import isopy
import numpy as np
import pyperclip as pyperclip
import inspect as inspect
import functools
import itertools
import warnings
import hashlib
import warnings

#optional imports
try:
    import pandas
except:
    pandas = None

try:
    import tables
except:
    tables = None

try:
    import IPython
except:
    IPython = None

from numpy import ndarray, nan, float64, void
from typing import TypeVar, Union, Optional, Any, NoReturn, Generic

class NotGivenType:
    def __bool__(self):
        return False

NotGiven = NotGivenType()

ARRAY_REPR = dict(include_row=True, include_dtype=False, nrows=10, f='{:.5g}')

__all__ = ['MassKeyString', 'ElementKeyString', 'IsotopeKeyString', 'RatioKeyString', 'GeneralKeyString',
           'MassKeyList', 'ElementKeyList', 'IsotopeKeyList', 'RatioKeyList', 'GeneralKeyList', 'MixedKeyList',
           'MassArray', 'ElementArray', 'IsotopeArray', 'RatioArray', 'GeneralArray', 'MixedArray',
           'IsopyDict', 'ScalarDict',
           'keystring', 'askeystring', 'keylist', 'askeylist', 'array', 'asarray', 'asanyarray',
           'ones', 'zeros', 'empty', 'full', 'random',
           'concatenate',
           'iskeystring', 'iskeylist', 'isarray',
           'flavour', 'isflavour', 'allowed_numpy_functions',
           'array_from_csv', 'array_from_clipboard', 'array_from_xlsx']

CACHE_MAXSIZE = 128
CACHES_ENABLED = False
#TODO stackoverflow error that i cannot figure out
def lru_cache(maxsize=128, typed=False, ignore_unhashable=True):
    """
    decorator for functools.lru_cache but with the option to call the original function when an unhashable value
    is encountered.

    """
    def lru_cache_decorator(func):
        cached = functools.lru_cache(maxsize, typed)(func)
        uncached = func
        @functools.wraps(func)
        def lru_cache_wrapper(*args, **kwargs):
            if CACHES_ENABLED:
                try:
                    return cached(*args, **kwargs)
                except TypeError as err:
                    if ignore_unhashable is False or not str(err).startswith('unhashable'):
                        raise err

            return uncached(*args, **kwargs)

        lru_cache_wrapper.__cached__ = cached
        lru_cache_wrapper.__uncached__ = uncached
        lru_cache_wrapper.cache_info = cached.cache_info
        lru_cache_wrapper.clear_cache = cached.cache_clear
        return lru_cache_wrapper
    return lru_cache_decorator

UNCACHED = object()
def cached_property(func):
    cache_name = f'__property_cache_{func.__name__}'
    def cache(obj):
        obj_cache = getattr(obj, cache_name, UNCACHED)
        if obj_cache is UNCACHED:
            obj_cache = func(obj)
            setattr(obj, cache_name, obj_cache)

        return obj_cache
    cache.__doc__ = func.__doc__
    return property(cache, doc=func.__doc__)

def remove_prefix(string, prefix):
    if string[:len(prefix)] == prefix:
        return string[len(prefix):]
    else:
        return string

def updatedoc(ofunc=None, /, **fdict):
    """
    Format the docstring of a function using the keyword arguments. If another
    function is passed as the first argument then the docstring from that function is used.
    """
    def wrap(func):
        if ofunc is None:
            f = func
        else:
            f = ofunc
        func.__doc__original = func.__doc__
        fdoc = getattr(f, '__doc__original', f.__doc__)
        func.__doc__ = fdoc.format(**fdict)
        return func
    return wrap

def get_module_name(item):
    name = getattr(item, '__name__', None)
    if name is None: return str(item)

    module = getattr(item, '__module__', None)
    if module is not None:
        if module == 'numpy': module = 'np'
        name = f'{module}.{name}'

    return name

def get_classname(thing):
    if isinstance(thing, type): return thing.__name__
    else: return thing.__class__.__name__

def append_preset_docstring(func):
    presets_rows = []
    for preset, kwargs in getattr(func, '_presets', []):
        kwargs_doc = ', '.join([f'{key} = {get_module_name(value)}' for key, value in kwargs.items()])
        presets_rows.append(f'* ``{preset.__name__}(*args, **kwargs, {kwargs_doc})``')

    preset_list = '\n    '.join(presets_rows)
    preset_title = f'.. rubric:: Presets\n    The following presets are avaliable for this function:'
    # Has to be rubric because custom headers does not work

    doc = func.__doc__ or ''
    doc += '\n\n    ' + preset_title + '\n    \n    ' + preset_list + '\n    \n\n    '

    func.__doc__ = doc
    return func

def add_preset(name, **kwargs):
    if type(name) is not tuple:
        name = (name,)

    def decorator(func):
        if not hasattr(func, '_presets'):
            func._presets = []

        for n in name:
            preset = functools.partial(func, **kwargs)
            preset.__name__ = f'{func.__name__}.{n}'
            preset.__module__ = func.__module__
            setattr(func, n, preset)
            func._presets.insert(0, (preset, kwargs))
        return func
    return decorator

def set_module(module):
    def decorator(func):
        func.__module__ = module
        return func
    return decorator

def extract_kwargs(kwargs, prefix, keep_prefix=False):
    new_kwargs = {}
    for kwarg in list(kwargs.keys()):
        prefix_ = f'{prefix}_'
        if kwarg.startswith(prefix_):
            if keep_prefix:
                new_kwargs[kwarg] = kwargs.pop(kwarg)
            else:
                new_kwargs[kwarg[len(prefix_):]] = kwargs.pop(kwarg)

    return new_kwargs

def renamed_function(new_func, **renamed_args):
    #decorate the old function which will forward the args and kwargs to the new function
    #The old code will not be executed and can be deleted.
    #Does not work with positional only arguments ('/' in signature)
    def rename_function_old(old_func):
        signature = inspect.signature(old_func)

        @functools.wraps(old_func)
        def wrapper(*args, **kwargs):
            warnings.warn(f'This function/method has been renamed "{new_func.__name__}". Please update code',
                          DeprecationWarning, stacklevel=2)

            #Turn positional arguments into keywords that avoids problems if the
            #Signarure of the new_function is different
            bound = signature.bind(*args, **kwargs)
            new_args = []
            new_kwargs = {}
            for arg, value in bound.arguments.items():
                kind = signature.parameters[arg].kind
                if kind is inspect.Parameter.VAR_KEYWORD:
                    new_kwargs.update(value)
                elif kind is inspect.Parameter.POSITIONAL_ONLY:
                    new_args.append(value)
                else:
                    new_kwargs[renamed_args.get(arg, arg)] = value

            return new_func(*new_args, **new_kwargs)
        return wrapper
    return rename_function_old

def renamed_kwarg(**old_new_name):
    #decorator for functions/methods with renamed kwargs
    def renamed_kwarg_func(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            new_kwargs = kwargs.copy()
            for old_name, new_name in old_new_name.items():
                if old_name in kwargs:
                    warnings.warn(f'Kwargs "{old_name}" has been renamed "{new_name}". Please update code', stacklevel=2)
                    if new_name in kwargs: new_kwargs.pop(old_name)
                    else: new_kwargs[new_name] = new_kwargs.pop(old_name)

            return func(*args, **new_kwargs)
        return wrapper
    return renamed_kwarg_func

def hashstr(string):
    return hashlib.md5(string.encode('UTF-8')).hexdigest()

def parse_keyfilters(**filters):
    """
    Parses key filters into a format that is accepted by KeyString._filter. Allows
    nesting of filter arguments for Isotope and RatioKeyString's.
    """
    #Make sure these are lists
    if 'mz' in filters: filters['mz_eq'] = filters.pop('mz')
    if 'key' in filters: filters['key_eq'] = filters.pop('key')
    for key in ['key_eq', 'key_neq', 'mz_eq', 'mz_neq']:
        if (f:=filters.get(key, None)) is not None and not isinstance(f, (list, tuple)):
            filters[key] = (f,)

    if 'flavour' in filters: filters['flavour_eq'] = filters.pop('flavour')
    for key in ['flavour_eq', 'flavour_neq']:
        if (f:=filters.get(key, None)) is not None:
            if not isinstance(f, (list, tuple)):
                filters[key] = (flavour(f),)
            else:
                filters[key] = tuple(flavour(ff) for ff in f)

    #For isotope keys
    mass_number = _split_filter('mass_number', filters)
    if mass_number:
        filters['mass_number'] = parse_keyfilters(**mass_number)

    element_symbol = _split_filter('element_symbol', filters)
    if element_symbol:
        filters['element_symbol'] = parse_keyfilters(**element_symbol)

    mz = _split_filter('mz', filters)
    if mz:
        filters['mz'] = parse_keyfilters(**mz)

    #For ratiokeys
    numerator = _split_filter('numerator', filters)
    if numerator:
        filters['numerator'] = parse_keyfilters(**numerator)

    denominator = _split_filter('denominator', filters)
    if denominator:
        filters['denominator'] = parse_keyfilters(**denominator)

    return filters

def _split_filter(prefix, filters):
    #Seperates out filters with a specific prefix
    out = {}
    if prefix[-1] != '_': prefix = f'{prefix}_'
    for key in tuple(filters.keys()):
        if key == prefix[:-1]:
            #avoids getting errors
            out['key_eq'] = filters.pop(prefix[:-1])
        elif key.startswith(prefix): #.startswith(prefix):
            filter = filters.pop(key)
            key = remove_prefix(key, prefix)
            if key in ('eq', 'neq', 'lt', 'le', 'ge', 'gt'):
                key = f'key_{key}'
            out[key] = filter
    return out

def isflavour(obj1, obj2):
    """returns True if the favour of the first object is the same as the favour of the second object."""
    return flavour(obj1) is flavour(obj2)

def flavour(obj):
    """returns the flavour type of the object."""
    if type(obj) is type:
        if issubclass(obj, MassFlavour) or obj is MassArray:
            return MassFlavour
        elif issubclass(obj, ElementFlavour) or obj is ElementArray:
            return ElementFlavour
        elif issubclass(obj, IsotopeFlavour) or obj is IsotopeArray:
            return IsotopeFlavour
        elif issubclass(obj, RatioFlavour) or obj is RatioArray:
            return RatioFlavour
        elif issubclass(obj, GeneralFlavour) or obj is GeneralArray:
            return GeneralFlavour
        elif issubclass(obj, MixedFlavour) or obj is MixedArray:
            return MixedFlavour
        else:
            raise TypeError(f'{obj} is not an isopy object')
    else:
        if isinstance(obj, MassFlavour):
            return MassFlavour
        elif isinstance(obj, ElementFlavour):
            return ElementFlavour
        elif isinstance(obj, IsotopeFlavour):
            return IsotopeFlavour
        elif isinstance(obj, RatioFlavour):
            return RatioFlavour
        elif isinstance(obj, GeneralFlavour):
            return GeneralFlavour
        elif isinstance(obj, MixedFlavour):
            return MixedFlavour
        elif isinstance(obj, str):
            if obj.lower() == 'mass':
                return MassFlavour
            elif obj.lower() == 'element':
                return ElementFlavour
            elif obj.lower() == 'isotope':
                return IsotopeFlavour
            elif obj.lower() == 'ratio':
                return RatioFlavour
            elif obj.lower() == 'general':
                return GeneralFlavour
            elif obj.lower() == 'mixed':
                return MixedFlavour

        raise TypeError(f'{type(obj)} is not an isopy object')

def iskeystring(item) -> bool:
    """
    Return ``True`` if *item* is a key string otherwise ``False`` is returned.
    """
    return isinstance(item, IsopyKeyString)

def iskeylist(item) -> bool:
    """
    Return ``True`` if *item* is a key string list otherwise ``False`` is returned.
    """
    return isinstance(item, IsopyKeyList)

def isarray(item) -> bool:
    """
    Return ``True`` if *item* is an isopy array otherwise ``False`` is returned.
    """
    return isinstance(item, IsopyArray)

##################
### Exceptions ###
##################

class KeyParseError(Exception):
    pass


class KeyValueError(KeyParseError, ValueError):
    def __init__(self, cls, string, additional_information = None):
        self.string = string
        self.cls = cls
        self.additional_information = additional_information

    def __str__(self):
        message = f'{get_classname(self.cls)}: unable to parse "{self.string}"'
        if self.additional_information:
            return f'{message}. {self.additional_information}.'
        else:
            return message


class KeyTypeError(KeyParseError, TypeError):
    def __init__(self, cls, obj):
        self.obj = obj
        self.cls = cls

    def __str__(self):
        return f'{get_classname(self.cls)}: cannot convert {type(self.obj)} into \'str\''



################
### Flavours ###
################

class IsopyFlavour:
    @classmethod
    def _Flavour__view(cls, obj):
        if obj.dtype.names:
            if isinstance(obj, void):
                return cls._Flavour__view_void(obj)
            else:
                return cls._Flavour__view_ndarray(obj)
        else:
            return obj.view(ndarray)


class MassFlavour:
    _Flavour__id = 1
    @staticmethod
    def _Flavour__keystring(string, **kwargs):
        return MassKeyString(string, **kwargs)

    @staticmethod
    def _Flavour__keylist(*args, **kwargs):
        return MassKeyList(*args, **kwargs)

    @staticmethod
    def _Flavour__array(*args, **kwargs):
        return MassArray(*args, **kwargs)

    @staticmethod
    def _Flavour__view_void(obj):
        return obj.view((MassVoid, obj.dtype))

    @staticmethod
    def _Flavour__view_ndarray(obj):
        return obj.view(MassNdarray)


class ElementFlavour:
    _Flavour__id = 2
    @staticmethod
    def _Flavour__keystring(string, **kwargs):
        return ElementKeyString(string, **kwargs)

    @staticmethod
    def _Flavour__keylist(*args, **kwargs):
        return ElementKeyList(*args, **kwargs)

    @staticmethod
    def _Flavour__array(*args, **kwargs):
        return ElementArray(*args, **kwargs)

    @staticmethod
    def _Flavour__view_void(obj):
        return obj.view((ElementVoid, obj.dtype))

    @staticmethod
    def _Flavour__view_ndarray(obj):
        return obj.view(ElementNdarray)


class IsotopeFlavour:
    _Flavour__id = 3
    @staticmethod
    def _Flavour__keystring(string, **kwargs):
        return IsotopeKeyString(string, **kwargs)

    @staticmethod
    def _Flavour__keylist(*args, **kwargs):
        return IsotopeKeyList(*args, **kwargs)

    @staticmethod
    def _Flavour__array(*args, **kwargs):
        return IsotopeArray(*args, **kwargs)

    @staticmethod
    def _Flavour__view_void(obj):
        return obj.view((IsotopeVoid, obj.dtype))

    @staticmethod
    def _Flavour__view_ndarray(obj):
        return obj.view(IsotopeNdarray)


class RatioFlavour:
    _Flavour__id = 4
    @staticmethod
    def _Flavour__keystring(string, **kwargs):
        return RatioKeyString(string, **kwargs)

    @staticmethod
    def _Flavour__keylist(*args, **kwargs):
        return RatioKeyList(*args, **kwargs)

    @staticmethod
    def _Flavour__array(*args, **kwargs):
        return RatioArray(*args, **kwargs)

    @staticmethod
    def _Flavour__view_void(obj):
        return obj.view((RatioVoid, obj.dtype))

    @staticmethod
    def _Flavour__view_ndarray(obj):
        return obj.view(RatioNdarray)


class GeneralFlavour:
    _Flavour__id = 5
    @staticmethod
    def _Flavour__keystring(string, **kwargs):
        return GeneralKeyString(string, **kwargs)

    @staticmethod
    def _Flavour__keylist(*args, **kwargs):
        return GeneralKeyList(*args, **kwargs)

    @staticmethod
    def _Flavour__array(*args, **kwargs):
        return GeneralArray(*args, **kwargs)

    @staticmethod
    def _Flavour__view_void(obj):
        return obj.view((GeneralVoid, obj.dtype))

    @staticmethod
    def _Flavour__view_ndarray(obj):
        return obj.view(GeneralNdarray)


class MixedFlavour:
    _Flavour__id = 6
    @staticmethod
    def _Flavour__keystring(string, **kwargs):
        return askeystring(string, **kwargs)

    @staticmethod
    def _Flavour__keylist(*args, **kwargs):
        keys = MixedKeyList(*args, **kwargs)

        if len(flavours:={type(k) for k in keys}) == 1:
            keys =  flavours.pop()._Flavour__keylist(keys)

        return keys

    @staticmethod
    def _Flavour__array(*args, **kwargs):
        return MixedArray(*args, **kwargs)

    @staticmethod
    def _Flavour__view_void(obj):
        return obj.view((MixedVoid, obj.dtype))

    @staticmethod
    def _Flavour__view_ndarray(obj):
        return obj.view(MixedNdArray)

##############
### Key ###
##############
class IsopyKeyString(IsopyFlavour, str):
    def __repr__(self):
        return f"{self.__class__.__name__}('{self}')"

    def __new__(cls, string, basekey = None, **kwargs):
        obj = str.__new__(cls, string)
        if basekey is None: basekey = obj
        object.__setattr__(obj, 'basekey', basekey)

        #object.__setattr__(obj, '_colname', string)
        for name, value in kwargs.items():
            object.__setattr__(obj, name, value)
        return obj

    def __hash__(self):
        return hash( (self.__class__, super(IsopyKeyString, self).__hash__()) )

    def __setattr__(self, key, value):
        raise AttributeError('{} does not allow additional attributes'.format(self.__class__.__name__))

    def __eq__(self, other):
        if not isinstance(other, IsopyKeyString):
            try:
                other = self._Flavour__keystring(other)
            except:
                return False

        return hash(self) == hash(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __truediv__(self, other):
        if isinstance(other, (list, tuple)):
            return askeylist(other).__rtruediv__(self)
        else:
            return RatioFlavour._Flavour__keystring((self, other))

    def __rtruediv__(self, other):
        if isinstance(other, (list, tuple)):
            return askeylist(other).__truediv__(self)
        else:
            return askeystring(other).__truediv__(self)

    def str(self, format = None):
        if format is None: return str(self)
        options = self._str_options()

        if format in options:
            return options[format]
        else:
            return format.format(**options)


class MassKeyString(MassFlavour, IsopyKeyString):
    """
    String representation of a mass number.

    Inherits from :class:`str` and therefore contains all the method that a :class:`str` does.
    Unless specifically noted below these methods will return a :class:`str` rather than a
    :class:`MassKeyString`.


    Parameters
    ----------
    string : str, int
        A string or integer representing an mass number.
    allow_reformatting : bool, Default = True
        If ``True`` the string will be reformatted to the correct format. If ``False`` an exception
        will be raised it the string is not correctly formatted.


    Raises
    ------
    KeyValueError
        Is raised when the supplied string cannot be parsed into the correct format
    KeyTypeError
        Raised when the supplied item cannot be turned into a string


    Examples
    --------
    >>> isopy.MassKeyString('76')
    '76'
    >>> isopy.MassKeyString(76)
    '76'

    Mass key strings also support the ``<,> <=, >=`` operators:

    >>> isopy.MassKeyString('76') > 75
    True
    >>> isopy.MassKeyString('76') <= 75
    False
    """

    @lru_cache(CACHE_MAXSIZE)
    def __new__(cls, string, *, allow_reformatting=True, ignore_charge = False):
        if isinstance(string, cls):
            if ignore_charge: return string.basekey
            else: return string

        if isinstance(string, int) and allow_reformatting:
            string = str(string)

        if not isinstance(string, str):
            raise KeyTypeError(cls, string)

        string.strip()
        if allow_reformatting is True:
            # string = string.removeprefix('_')
            string = remove_prefix(string, 'Mass_') #For backwards compatibility
            string = remove_prefix(string, 'MAS_') #For backwards compatibility

        if len(string) == 0:
            raise KeyValueError(cls, string, 'cannot parse empty string')

        if not string.isdigit():
            raise KeyValueError(cls, string, 'Can only contain numerical characters')

        if int(string) < 0:
            raise KeyValueError(cls, string, 'Must be a positive integer')

        return super(MassKeyString, cls).__new__(cls, string)

    def __ge__(self, item):
        if isinstance(item, str):
            try:
                item = float(item)
            except:
                pass

        return int(self) >= item

    def __le__(self, item):
        if isinstance(item, str):
            try:
                item = float(item)
            except:
                pass

        return int(self) <= item

    def __gt__(self, item):
        if isinstance(item, str):
            try:
                item = float(item)
            except:
                pass

        return int(self) > item

    def __lt__(self, item):
        if isinstance(item, str):
            try:
                item = float(item)
            except:
                pass

        return int(self) < item

    def _filter(self, key_eq=None, key_neq=None, flavour_eq = None, flavour_neq = None,
                *, key_lt=None, key_gt=None, key_le=None, key_ge=None, **invalid):
        if invalid:
            return False
        if key_eq is not None and self not in key_eq:
            return False
        if key_neq is not None and self in key_neq:
            return False
        if flavour_eq is not None and flavour(self) not in flavour_eq:
            return False
        if flavour_neq is not None and flavour(self) in flavour_neq:
            return False
        if key_lt is not None and not self < key_lt:
            return False
        if key_gt is not None and not self > key_gt:
            return False
        if key_le is not None and not self <= key_le:
            return False
        if key_ge is not None and not self >= key_ge:
            return False

        return True

    def _sortkey(self):
        return f'{self:0>4}'

    def str(self, format=None):
        """
        Return a ``str`` object of the key string.

        the optional *format* can either be a string matching one of the format options or a
        string which can be formatted using the format options as key words.

        Format options are:
        * ``"m"`` and ``"key"`` - Same as ``str(keystring)``
        * ``"math"`` -  Key string formatted for latex math mode.
        * ""`latex`"" - Same as above but including $ $ math deliminators.

        Examples
        --------
        >>> key = isopy.MassKeyString('101')
        >>> key.str()
        '101'
        >>> key.str('key is "{m}"')
        'key is "101"'
        """
        return super(MassKeyString, self).str(format)

    def _str_options(self):
        return dict(m = str(self), key = str(self),
                    math = fr'\mathrm{{{self}}}',
                    latex = fr'$\mathrm{{{self}}}$')


class ElementKeyString(ElementFlavour, IsopyKeyString):
    """
    A string representation of an element symbol limited to two letters.

    The first letter is in upper case and subsequent letters are in lower case.

    Inherits from :class:`str` and therefore contains all the method that a :class:`str` does.
    Unless specifically noted below these methods will return a :class:`str` rather than a
    :class:`ElementKeyString`.


    Parameters
    ----------
    string : str
        A one or two letter string representing an element symbol. The letter case is not considered.
    allow_reformatting : bool, Default = True
        If ``True`` the string will be reformatted to the correct format. If ``False`` an exception
        will be raised it the string is not correctly formatted.

    Raises
    ------
    KeyValueError
        Is raised when the supplied string cannot be parsed into the correct format
    KeyTypeError
        Raised when the supplied item cannot be turned into a string


    Examples
    --------
    >>> isopy.ElementKeyString('Pd')
    'Pd'
    >>> isopy.ElementKeyString('pd')
    'pd'
    """

    @lru_cache(CACHE_MAXSIZE)
    def __new__(cls, string, *, charge = NotGiven, allow_reformatting=True, ignore_charge = False):
        if isinstance(string, cls) and charge is NotGiven:
            if ignore_charge:
                return string.basekey
            else:
                return string

        if not isinstance(string, str):
            raise KeyTypeError(cls, string)

        string = string.strip()
        if allow_reformatting is True:
            string = remove_prefix(string, 'Element_') #For backwards compatibility
            string = remove_prefix(string, 'ELE_') #For backwards compatibility

        if len(string) == 0:
            raise KeyValueError(cls, string, 'Cannot parse empty string')

        if charge is NotGiven:
            if '+' in string:
                charge = len(string) - len(string:=string.rstrip('+'))
            elif '-' in string:
                charge = -(len(string) - len(string:=string.rstrip('-')))
            else:
                charge = None
        elif type(charge) is int or charge is None:
            if '+' in string:
                string = string.rstrip('+')
            elif '-' in string:
                string = string.rstrip('-')
            if charge == 0:
                charge = None
        else:
            raise TypeError('charge must be an integer or None')

        if len(string) > 2:
            if allow_reformatting:
                symbol = isopy.refval.element.name_symbol.get(string.capitalize(), None)
            else:
                symbol = None
            if symbol is None:
                raise KeyValueError(cls, string, 'ElementKeyString is limited to two characters')
            else:
                string = symbol

        if not string.isalpha():
            raise KeyValueError(cls, string, 'ElementKeyString is limited to alphabetical characters')

        if allow_reformatting:
            string = string.capitalize()
        elif string[0].isupper() and (len(string) == 1 or string[1].islower()):
            pass
        else:
            raise KeyValueError(cls, string, 'First character must be upper case and second character must be lower case')

        if not ignore_charge and charge is not None:
            basekey = super(ElementKeyString, cls).__new__(cls, string, charge = None)
            string = f'{string}{"+" * charge or "-" * abs(charge)}'
        else:
            basekey = None
            charge = None

        return super(ElementKeyString, cls).__new__(cls, string, basekey, charge = charge)

    def _filter(self, key_eq = None, key_neq = None, flavour_eq = None, flavour_neq = None, **invalid):
        if invalid:
            return False
        if key_eq is not None and self not in key_eq:
            return False
        if key_neq is not None and self in key_neq:
            return False
        if flavour_eq is not None and flavour(self) not in flavour_eq:
            return False
        if flavour_neq is not None and flavour(self) in flavour_neq:
            return False
        return True

    def _sortkey(self):
        z = isopy.refval.element.atomic_number.get(self, self)
        return f'{z:0>4}'

    def str(self, format=None):
        """
        Return a ``str`` object of the key string.

        the optional *format* can either be a string matching one of the format options or a
        string which can be formatted using the format options as key words.

        Format options are:
        * ``"key"`` - Same as ``str(keystring)``.
        * ``"Es"``, ``"es"``, ``"ES"`` - Element symbol capitalised, in lower case and in upper case respectivley.
        * ``"Name"``, ``"name"``, ``"NAME"`` - Full element name capitalised, in lower case and in upper case respectivley.
        * ``"math"`` -  Key string formatted for latex math mode.
        * ""`latex`"" - Same as above but including $ $ math deliminators.

        Examples
        --------
        >>> key = isopy.ElementKeyString('ru')
        >>> key.str()
        'Ru'
        >>> key.str('Name')
        'Ruthenium'
        >>> key.str('Name of "{es}" is {Name}')
        'name of "ru" is Ruthenium'
        """
        return super(ElementKeyString, self).str(format)

    def _str_options(self):
        name: str = isopy.refval.element.symbol_name.get(self.basekey, str(self.basekey))
        if self.charge is not None: name = f'{name}{"+" * self.charge or "-" * abs(self.charge)}'
        return dict(key = str(self),
                    es=self.lower(), ES=self.upper(), Es=str(self),
                    name=name.lower(), NAME=name.upper(), Name=name,
                    math = fr'\mathrm{{{self}}}',
                    latex = fr'$\mathrm{{{self}}}$')

    def set_charge(self, charge):
        return self.__class__(self, charge=charge)


class IsotopeKeyString(IsotopeFlavour, IsopyKeyString):
    """
    A string representation of an isotope consisting of a mass number followed by an element symbol.

    Inherits from :class:`str` and therefore contains all the method that a :class:`str` does.
    Unless specifically noted below these methods will return a :class:`str` rather than a
    :class:`IsotopeKeyString`.


    Parameters
    ----------
    string : str
        Should consist of an mass number and a valid element symbol. The order of the two does
        not matter.
    allow_reformatting : bool, Default = True
        If ``True`` the string will be reformatted to the correct format. If ``False`` an exception
        will be raised it the string is not correctly formatted.


    Raises
    ------
    KeyValueError
        Is raised when the supplied string cannot be parsed into the correct format
    KeyTypeError
        Raised when the supplied item cannot be turned into a string


    Attributes
    ----------
    mass_number : MassKeyString
        The mass number of the isotope
    element_symbol : ElementKeyString
        The element symbol of the isotope


    Examples
    --------
    >>> isopy.IsotopeKeyString('Pd104')
    '104Pd'
    >>> isopy.IsotopeKeyString('104pd')
    '104Pd'
    >>> isopy.IsotopeKeyString('Pd104').mass_W17
    '104'
    >>> isopy.IsotopeKeyString('Pd104').element_symbol
    'Pd'

    ``in`` can be used to test if a string is equal to the mass number or an element symbol of an
    isotope key string.

    >>> 'pd' in isopy.IsotopeKeyString('Pd104')
    True
    >>> 104 in isopy.IsotopeKeyString('Pd104')
    True
    """

    @lru_cache(CACHE_MAXSIZE)
    def __new__(cls, string, *, allow_reformatting=True, ignore_charge = False):
        if isinstance(string, cls):
            if ignore_charge:
                return string.basekey
            else:
                return string

        if not isinstance(string, str):
            raise KeyTypeError(cls, string)

        string = string.strip()
        if len(string) == 0:
            raise KeyValueError(cls, string, 'Cannot parse empty string')

        if allow_reformatting is True:
            string = remove_prefix(string, '_') #For backwards compatibility
            string = remove_prefix(string, 'Isotope_') #For backwards compatibility
            string = remove_prefix(string, 'ISO_') #For backwards compatibility
        if allow_reformatting and '-' in string.rstrip('-') and string.count('-') == 1:
            string = string.replace('-', '') #To allow e.g. 'ru-102'

        # If no digits in string then only Symbol is given.
        if string.isalpha():
            raise KeyValueError(cls, string, 'string does not contain a mass number')

        # If only digits then only mass number
        if string.isdigit():
            raise KeyValueError(cls, string, 'string does not contain an element symbol')

        if allow_reformatting:
            element = string.strip('0123456789')
            mass = string.strip(element)
        else:
            element = string.lstrip('0123456789')
            mass = string.rstrip(element)

        try:
            mass = MassKeyString(mass, allow_reformatting=allow_reformatting, ignore_charge=ignore_charge)
            element = ElementKeyString(element, allow_reformatting=allow_reformatting, ignore_charge=ignore_charge)
        except KeyParseError as err:
            raise KeyValueError(cls, string,
                                'unable to separate string into a mass number and an element symbol') from err

        basekey = super(IsotopeKeyString, cls).__new__(cls, f'{mass.basekey}{element.basekey}',
                                                       mass_number = mass.basekey,
                                                       element_symbol = element.basekey,
                                                       charge = None)

        string = '{}{}'.format(mass, element)

        return super(IsotopeKeyString, cls).__new__(cls, string, basekey,
                                                   mass_number = mass,
                                                   element_symbol = element,
                                                   charge = element.charge)

    def __hash__(self):
        return hash( (self.__class__, hash(self.mass_number), hash(self.element_symbol)) )

    def __contains__(self, string):
        """
        Return **True** if `string` is equal to the IsotopeKeyString's mass number or element symbol
        """
        return self.mass_number == string or self.element_symbol == string

    def _filter(self, key_eq=None, key_neq=None, flavour_eq = None, flavour_neq = None,
                mass_number = {}, element_symbol = {}, mz = {}, **invalid):
        if len(invalid) > 0:
            return False
        if key_eq is not None and self not in key_eq:
            return False
        if key_neq is not None and self in key_neq:
            return False
        if flavour_eq is not None and flavour(self) not in flavour_eq:
            return False
        if flavour_neq is not None and flavour(self) in flavour_neq:
            return False
        if mass_number and not self.mass_number._filter(**mass_number):
            return False
        if element_symbol and not self.element_symbol._filter(**element_symbol):
            return False
        if mz and not self._filter_mz(**mz):
            return False

        return True

    def _filter_mz(self, key_eq=None, key_neq=None, flavour_eq = None, flavour_neq = None,key_lt=None, key_gt=None, key_le=None, key_ge=None,
                   true_mass=False, isotope_masses=None):
        mz = self.mz(true_mass=true_mass, isotope_masses=isotope_masses)
        if key_eq is not None and mz not in key_eq:
            return False
        if key_neq is not None and mz in key_neq:
            return False
        if key_lt is not None and not mz < key_lt:
            return False
        if key_gt is not None and not mz > key_gt:
            return False
        if key_le is not None and not mz <= key_le:
            return False
        if key_ge is not None and not mz >= key_ge:
            return False

        return True

    def _sortkey(self):
        return f'{self.mz():0>8.3f}{self.element_symbol._sortkey()}'

    def str(self, format=None):
        """
        Return a ``str`` object of the key string.

        the optional *format* can either be a string matching one of the format options or a
        string which can be formatted using the format options as key words.

        Format options are:
        * ``"key"`` - Same as ``str(keystring)``.
        * All format options listed for :func:`MassKeyString.str`_
        * All format options listed for :func:`ElementKeyString.str`_
        * All combinations of mass key string format options and the element key string options, e.g. 'esm' or 'mName'.
        * All combinations listed above but with a ``"-"`` between the two format options.
        * ``"math"`` -  Key string formatted for latex math mode.
        * ""`latex`"" - Same as above but including $ $ math deliminators.


        Examples
        --------
        >>> key = isopy.IsotopeKeyString('101ru')
        >>> key.str()
        '101Ru'
        >>> key.str('esm')
        'ru101'
        >>> key.str('esm')
        'ru101'
        >>> key.str('Name-m')
        'Ruthenium-101'
        >>> key.str('Mass {m} of element {Name}')
        'Mass 101 of Ruthenium'
        """
        return super(IsotopeKeyString, self).str(format)
        
    def _str_options(self):
        options = dict()

        mass_options = self.mass_number._str_options()
        element_options = self.element_symbol._str_options()
        options.update(mass_options)
        options.update(element_options)
        options.update(dict(key = str(self),
                       math = fr'^{{{self.mass_number}}}\mathrm{{{self.element_symbol}}}',
                       latex = fr'$^{{{self.mass_number}}}\mathrm{{{self.element_symbol}}}$'))

        product = list(itertools.product(mass_options.items(), element_options.items()))
        options.update({f'{mk}{ek}': f'{mv}{ev}' for (mk, mv), (ek, ev) in product})
        options.update({f'{mk}-{ek}': f'{mv}-{ev}' for (mk, mv), (ek, ev) in product})
        options.update({f'{ek}{mk}': f'{ev}{mv}' for (mk, mv), (ek, ev) in product})
        options.update({f'{ek}-{mk}': f'{ev}-{mv}' for (mk, mv), (ek, ev) in product})

        return options

    def set_charge(self, charge):
        return self.__class__(f'{self.mass_number}{self.element_symbol.set_charge(charge)}')

    def mz(self, true_mass = False, isotope_masses = None):
        charge = abs(self.charge or 1)
        if true_mass:
            if isotope_masses is None: isotope_masses = isopy.refval.isotope.mass
            return isotope_masses[self.basekey] / charge
        else:
            return float(self.mass_number) / charge


class RatioKeyString(RatioFlavour, IsopyKeyString):
    """
    A string representation of a ratio of two keystrings.

    Inherits from :class:`str` and therefore contains all the method that a :class:`str` does.
    Unless specifically noted below these methods will return a :class:`str` rather than a
    :class:`RatioKeyString`.


    Parameters
    ----------
    string : str, tuple[str, str]
        A string with the numerator and denominator seperated by "/" or a
        (numerator, denominator) tuple of strings. The numerator and denominator key
        strings can be of different flavours. Nested ratios can be created
        using a combination of "/", "//", "///" etc upto a maximum of 9 nested ratios.
    allow_reformatting : bool, Default = True
        If ``True`` the string will be reformatted to the correct format. If ``False`` an exception
        will be raised it the string is not correctly formatted.


    Raises
    ------
    KeyValueError
        Is raised when the supplied string cannot be parsed into the correct format
    KeyTypeError
        Raised when the supplied item cannot be turned into a string


    Attributes
    ----------
    numerator : :class:`MassKeyString`, :class:`ElementKeyString`, :class:`IsotopeKeyString`, :class:`RatioKeyString` or :class:`GeneralKeyString`
        The numerator key string
    denominator : :class:`MassKeyString`, :class:`ElementKeyString`, :class:`IsotopeKeyString`, :class:`RatioKeyString` or :class:`GeneralKeyString`
        The denominator key string


    Examples
    --------
    >>> isopy.RatioKeyString('Pd108/105pd')
    '104Pd/108Pd'
    >>> isopy.RatioKeyString(('Pd108', '105pd'))
    '104Pd/108Pd'
    >>> isopy.RatioKeyString('108pd/ge') #You can mix flavours
    '104Pd/Ge'
    >>> isopy.RatioKeyString('108pd/ge').numerator
    '108Pd'
    >>> isopy.RatioKeyString('108pd/ge').denominator
    'Ge'

    Nested arrays

    >>> isopy.RatioKeyString('Pd108/105pd//ge')
    '108Pd/105Pd//Ge
    >>> isopy.RatioKeyString((('Pd108', '105pd'), 'ge'))
    '108Pd/105Pd//Ge
    >>> isopy.RatioKeyString('Pd108/105pd//as/ge')
    '108Pd/105Pd//As/Ge
    >>> isopy.RatioKeyString(('Pd108/105pd', 'as/ge'))
    '108Pd/105Pd//As/Ge
    >>> isopy.RatioKeyString(('Pd108/105pd', 'as/ge')).numerator
    '108Pd/105Pd'
    >>> isopy.RatioKeyString(('Pd108/105pd', 'as/ge')).denominator
    'As/Ge'

    ``in`` can be used to test if a string is equal to the numerator or denominator of the ratio.

    >>> 'pd108' in isopy.RatioKeyString('108Pd/Ge')
    True
    >>> 'as/ge' in isopy.RatioKeyString('Pd108/105pd//as/ge')
    True
    """

    @lru_cache(CACHE_MAXSIZE)
    def __new__(cls, string, *, allow_reformatting=True, ignore_charge = False):
        if isinstance(string, cls):
            if ignore_charge:
                return string.basekey
            else:
                return string

        if isinstance(string, tuple) and len(string) == 2:
            numer, denom = string

        elif not isinstance(string, str):
            raise KeyTypeError(cls, string)

        else:

            string = string.strip()

            # For backwards compatibility
            if string.startswith('Ratio_') and allow_reformatting is True: #.startswith('Ratio_'):
                string = remove_prefix(string, 'Ratio_')
                try:
                    numer, denom = string.split('_', 1)
                except:
                    raise KeyValueError(cls, string,
                                        'unable to split string into numerator and denominator')

            # For backwards compatibility
            elif string.startswith('RAT') and string[3].isdigit() and allow_reformatting is True:
                n = int(string[3])
                string = string[5:]
                try:
                    numer, denom = string.split(f'_OVER{n}_', 1)
                except:
                    raise KeyValueError(cls, string,
                                        'unable to split string into numerator and denominator')

            else:
                for n in range(9,0, -1):
                    divider = '/' * n
                    if string.count(divider) == 1:
                        numer, denom = string.split(divider, 1)
                        break
                else:
                    raise KeyValueError(cls, string,
                                            'unable to split string into numerator and denominator')

        numer = askeystring(numer, allow_reformatting=allow_reformatting, ignore_charge=ignore_charge)
        denom = askeystring(denom, allow_reformatting=allow_reformatting, ignore_charge=ignore_charge)

        for n in range(1, 10):
            divider = '/' * n
            if numer.count(divider) > 0 or denom.count(divider) > 0:
                continue
            else:
                break
        else:
            raise KeyValueError('Limit of nested ratios reached')

        basekey = super(RatioKeyString, cls).__new__(cls, f'{numer.basekey}{divider}{denom.basekey}',
                                                  numerator = numer.basekey, denominator = denom.basekey)
        string = f'{numer}{divider}{denom}'

        #colname = f'RAT{n}_{numer.colname}_OVER{n}_{denom.colname}'

        return super(RatioKeyString, cls).__new__(cls, string, basekey,
                                                  numerator = numer, denominator = denom)

    def __hash__(self):
        return hash( (self.__class__, hash(self.numerator), hash(self.denominator)) )

    def __contains__(self, string):
        """
        Return **True** if `string` is equal to the RatioKeyString's numerator or denominator
        """
        return self.numerator == string or self.denominator == string

    def _filter(self, key_eq=None, key_neq=None, flavour_eq = None, flavour_neq = None,
                numerator = {}, denominator = {}, **invalid):
        if invalid:
            return False
        if key_eq is not None and self not in key_eq:
            return False
        if key_neq is not None and self in key_neq:
            return False
        if flavour_eq is not None and flavour(self) not in flavour_eq:
            return False
        if flavour_neq is not None and flavour(self) in flavour_neq:
            return False
        if numerator and not self.numerator._filter(**numerator):
            return False
        if denominator and not self.denominator._filter(**denominator):
            return False

        return True

    def _sortkey(self):
        return f'{self.denominator._sortkey()}/{self.numerator._sortkey()}'

    def str(self, format = None, nformat=None, dformat = None):
        """
        Return a ``str`` object of the key string.

        the optional *format* can either be a string matching one of the format options or a
        string which can be formatted using the format options as key words.

        Format options are:
        * ``"key"`` - Same as ``str(keystring)``, does consider *nformat* and *dformat*.
        * ``"n"`` - The numerator. *nformat* can be given to specify format of the numerator.
        * ``"d"`` - The denominator. *dformat* can be given to specify format of the denominator.
        * ``"n/d"`` - The ratio including *nformat* and *dformat*. This is the default is *format* is not given.
        * ``"math"`` -  Key string formatted for latex math mode.
        * ""`latex`"" - Same as above but including $ $ math deliminators.

        *format* can be a tuple or a dict which will be unpacked into *format*, *nformat* and *dformat*.
        This is useful for ratios of ratios.

        Examples
        --------
        >>> key = isopy.RatioKeyString('101ru/104ru')
        >>> key.str()
        '101Ru/104Ru'
        >>> key.str(nformat='Name-m', dformat='esm')
        'Ruthenium-101/ru104'
        >>> key.str('Ratio is "{n/d}"')
        'Ratio is "101Ru/104Ru"'
        >>> key.str('numerator is: {n}, denominator is: {d}', dformat='esm')
        'numerator is: 101Ru, denominator is: ru104'
        """
        if type(format) is tuple:
            format, nformat, dformat, *_ = itertools.chain(format, (None, None, None))
        elif isinstance(format, dict):
            nformat = format.get('nformat', None)
            dformat = format.get('dformat', None)
            format = format.get('format', None)

        if format is None and nformat is None and dformat is None: return str(self)

        n = self.numerator.str(nformat)
        d = self.denominator.str(dformat)
        nd = f'{n}/{d}'

        options = self._str_options()
        options['n'] = n
        options['d'] = d
        options['n/d'] = nd

        if format in options:
            return options[format]
        else:
            return format.format(**options)

    def _str_options(self):
        options = dict(key = str(self))

        nmath = self.numerator.str('math')
        if type(self.numerator) is RatioKeyString:
            nmath = fr'\left({nmath}\right)'

        dmath = self.denominator.str('math')
        if type(self.denominator) is RatioKeyString:
            dmath = fr'\left({dmath}\right)'

        options['math'] = fr'\frac{{{nmath}}} {{{dmath}}}'
        options['latex'] = fr'${options["math"]}$'
        return options


class GeneralKeyString(GeneralFlavour, IsopyKeyString):
    """
    A general key string that can hold any string value.

    No formatting is applied to the string.

    Inherits from :class:`str` and therefore contains all the method that a :class:`str` does.
    Unless specifically noted below these methods will return a :class:`str` rather than a
    :class:`GeneralKeyString`.


    Parameters
    ----------
    string : str
        Any string will work as GeneralKeyString
    allow_reformatting : bool, Default = True
        If ``True`` the string will be reformatted to the correct format. If ``False`` an exception
        will be raised it the string is not correctly formatted.


    Raises
    ------
    KeyValueError
        Is raised when the supplied string cannot be parsed into the correct format
    KeyTypeError
        Raised when the supplied item cannot be turned into a string


    Examples
    --------
    >>> isopy.GeneralKeyString('harry')
    'harry'
    >>> isopy.GeneralKeyString('Hermione/Ron')
    'Hermione/Ron'
    """

    @lru_cache(CACHE_MAXSIZE)
    def __new__(cls, string, *, allow_reformatting=True, ignore_charge = False):
        if isinstance(string, cls):
            if ignore_charge:
                return string.basekey
            else:
                return string

        elif isinstance(string, str):
            string = str(string).strip()

        elif allow_reformatting:
            try:
                string = str(string).strip()
            except:
                raise KeyTypeError(cls, string)
        else:
            raise KeyTypeError(cls, string)

        if len(string) == 0:
            raise KeyValueError(cls, string, 'Cannot parse empty string')

        if allow_reformatting is True:
            string = remove_prefix(string, 'GEN_') #For backwards compatibility
            #colname = string.replace('/', '_SLASH_') #For backwards compatibility
            string = string.replace('_SLASH_', '/') #For backwards compatibility
        return super(GeneralKeyString, cls).__new__(cls, string)

    def _filter(self, key_eq=None, key_neq=None, flavour_eq = None, flavour_neq = None, **invalid):
        if invalid:
            return False
        if key_eq is not None and self not in key_eq:
            return False
        if key_neq is not None and self in key_neq:
            return False
        if flavour_eq is not None and flavour(self) not in flavour_eq:
            return False
        if flavour_neq is not None and flavour(self) in flavour_neq:
            return False

        return True

    def _sortkey(self):
        return str(self)

    def str(self, format=None):
        """
        Return a ``str`` object of the key string.

        the optional *format* can either be a string matching one of the format options or a
        string which can be formatted using the format options as key words.

        Format options are:
        * ``"key"`` - Same as ``str(keystring)``
        * ``"math"`` -  Key string formatted for latex math mode.
        * ""`latex`"" - Same as above but including $ $ math deliminators.

        Examples
        --------
        >>> key = isopy.GeneralKeyString('hermione')
        >>> key.str()
        'hermione'
        >>> key.str('{key} is really smart')
        'hermione is really smart'
        """
        return super(GeneralKeyString, self).str(format)

    def _str_options(self):
        return dict(key = str(self),
                    math = fr'\mathrm{{{self}}}',
                    plt = fr'$\mathrm{{{self}}}$',
                    tex = str(self))


############
### List ###
############

#TODO document sorted with exampels in each key string

class IsopyKeyList(IsopyFlavour, tuple):
    """
        When comparing the list against another list the order of items is not considered.
    The ``in`` operator can be used to test whether a string is in the list. To test whether all items in a list are
    present in another list use the ``<=`` operator.

    The ``&``, ``|`` and ``^`` operators can be used in combination with another list for an unordered item to item
    comparison. The and (``&``) operator will return the items that occur in both lists. The or (``|``) operator
    will return all items that appear in at least one of the lists. The xor (``^``) operator will return the items
    that do not appear in both lists. All duplicate items will be removed from the returned lists.
    """

    @lru_cache(CACHE_MAXSIZE)
    def __new__(cls, *keys, ignore_duplicates = False,
                allow_duplicates = True, allow_reformatting = True, ignore_charge=False):
        new_keys = []
        for key in keys:
            if isinstance(key, str):
                new_keys.append(cls._Flavour__keystring(key, allow_reformatting=allow_reformatting,
                                                        ignore_charge=ignore_charge))
            elif isinstance(key, np.dtype) and key.names is not None:
                new_keys.extend([cls._Flavour__keystring(name, allow_reformatting=allow_reformatting,
                                                        ignore_charge=ignore_charge) for name in key.names])
            elif isinstance(key, ndarray) and key.dtype.names is not None:
                new_keys.extend([cls._Flavour__keystring(name, allow_reformatting=allow_reformatting,
                                                        ignore_charge=ignore_charge) for name in key.dtype.names])
            elif hasattr(key, '__iter__'):
                new_keys.extend([cls._Flavour__keystring(k, allow_reformatting=allow_reformatting,
                                                        ignore_charge=ignore_charge) for k in key])
            else:
                new_keys.append(cls._Flavour__keystring(key, allow_reformatting=allow_reformatting,
                                                        ignore_charge=ignore_charge))

        if ignore_duplicates:
            new_keys = list(dict.fromkeys(new_keys).keys())
        elif not allow_duplicates and (len(set(new_keys)) != len(new_keys)):
            raise ValueError(f'duplicate key found in list {new_keys}')

        return super(IsopyKeyList, cls).__new__(cls, new_keys)

    def __call__(self):
        return self

    def __hash__(self):
        return super(IsopyKeyList, self).__hash__()

    def __repr__(self):
        return f'''{self.__class__.__name__}({", ".join([f"'{k}'" for k in self])})'''

    def __eq__(self, other):
        if not isinstance(other, IsopyKeyList):
            try:
                other = self._Flavour__keylist(other)
            except:
                return False
        return hash(self) == hash(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __contains__(self, items):
        """
        Return **True** if `item` is present in the list. Otherwise return **False**.
        """
        if not isinstance(items, (list, tuple)):
            items = (items,)
        for item in items:
            if not isinstance(item, IsopyKeyString):
                try:
                    item = self._Flavour__keystring(item)
                except:
                    return False
            try:
                if super(IsopyKeyList, self).__contains__(item) is False:
                    return False
            except:
                return False

        return True

    def __getitem__(self, index):
        """
        Return the item at `index`. `index` can be int, slice or sequence of int.
        """
        if isinstance(index, slice):
            return self._Flavour__keylist(super(IsopyKeyList, self).__getitem__(index))
        elif hasattr(index, '__iter__'):
                return self._Flavour__keylist(tuple(super(IsopyKeyList, self).__getitem__(i) for i in index))
        else:
            return super(IsopyKeyList, self).__getitem__(index)

    def __truediv__(self, denominator):
        if isinstance(denominator, (tuple, list)):
            if len(denominator) != len(self):
                raise ValueError(f'size of values ({len(self)}) not compatible with size of key list ({len(self)}')
            return RatioFlavour._Flavour__keylist(tuple(n / denominator[i] for i, n in enumerate(self)))
        else:
            return RatioFlavour._Flavour__keylist(tuple(n / denominator for i, n in enumerate(self)))

    def __rtruediv__(self, numerator):
        if isinstance(numerator, (tuple, list)):
            if len(numerator) != len(self):
                raise ValueError(f'size of values ({len(self)}) not compatible with size of key list ({len(self)}')
            return RatioFlavour._Flavour__keylist(tuple(numerator[i] / d for i, d in enumerate(self)))
        else:
            return RatioFlavour._Flavour__keylist(tuple(numerator / d for i, d in enumerate(self)))

    def __add__(self, other):
        other = askeylist(other)
        return askeylist((*self, *other))

    def __radd__(self, other):
        other = askeylist(other)
        return askeylist((*other, *self))

    def __sub__(self, other):
        other = askeylist(other)
        return self._Flavour__keylist((key for key in self if key not in other))

    def __rsub__(self, other):
        return askeylist(other).__sub__(self)

    def count(self, item):
        try:
            item = self._Flavour__keystring(item)
        except:
            return 0
        else:
            return super(IsopyKeyList, self).count(item)

    def index(self, item, *args):
        try:
            return super(IsopyKeyList, self).index(self._Flavour__keystring(item), *args)
        except (KeyValueError, ValueError):
            raise ValueError(f'{item} not in {self.__class__}')

    def has_duplicates(self):
        """
        Returns ``True`` if the list contains duplicates items. Otherwise it returns ``False``
        """
        return len(set(self)) != len(self)

    def strlist(self, format=None):
        """
        Return a list of ``str`` object for each key in the key list.

        Analogous to ``[key.str(format) for key in keylist]``
        """
        return [key.str(format) for key in self]

    def sorted(self):
        """
        Return a sorted key string list.
        """
        return self._Flavour__keylist(sorted(self, key= lambda k: k._sortkey()))

    def reversed(self):
        """
        Return a reversed key string list.
        """
        return self._Flavour__keylist(reversed(self))

    def flatten(self, ignore_duplicates = False, allow_duplicates = True, allow_reformatting = False,
                ignore_charge=False):
        """Has no effect on this key list flavour other than those imposed by the arguments"""
        return self._Flavour__keylist(self, ignore_duplicates=ignore_duplicates, allow_duplicates=allow_duplicates,
                                      allow_reformatting=allow_reformatting, ignore_charge=ignore_charge)

    ##########################################
    ### Cached methods for increased speed ###
    ##########################################

    @lru_cache(CACHE_MAXSIZE)
    def __and__(self, *others):
        this = dict.fromkeys(self)
        for other in others:
            if not isinstance(other, IsopyKeyList):
                other = askeylist(other)
            other = [hash(o) for o in dict.fromkeys(other)]
            this = (t for t in this if hash(t) in other)
        return askeylist(this)

    @lru_cache(CACHE_MAXSIZE)
    def __or__(self, *others):
        this = self
        for other in others:
            if not isinstance(other, IsopyKeyList):
                other = askeylist(other)
            this = tuple(dict.fromkeys((*this, *other)))
        return askeylist(this)

    @lru_cache(CACHE_MAXSIZE)
    def __xor__(self, *others):
        this = dict.fromkeys(self)
        for other in others:
            this_hash = [hash(t) for t in this]
            if not isinstance(other, IsopyKeyList):
                other = askeylist(other)
            other = dict.fromkeys(other)
            other_hash = [hash(o) for o in dict.fromkeys(other)]

            this = (*(t for i, t in enumerate(this) if this_hash[i] not in other_hash),
                                              *(o for i, o in enumerate(other) if other_hash[i] not in this_hash))
        return askeylist(this)

    @lru_cache(CACHE_MAXSIZE)
    def __rand__(self, other):
        return askeylist(other).__and__(self)

    @lru_cache(CACHE_MAXSIZE)
    def __ror__(self, other):
        return askeylist(other).__or__(self)

    @lru_cache(CACHE_MAXSIZE)
    def __rxor__(self, other):
        return askeylist(other).__xor__(self)


class MassKeyList(MassFlavour, IsopyKeyList):
    """
    A tuple consisting exclusively of :class:`MassKeyString` items.

    Behaves just like, and contains all the methods, that a normal tuple does unless otherwise noted.
    Only methods that behave differently from a normal python tuples are documented below.

    Parameters
    ----------
    keys : str, int,  Sequence[(str, int)]
        A string or sequence of strings that can be converted to the correct key string type.
    ignore_duplicates : bool, Default = True
        If ``True`` all duplicate items will be removed from the sequence.
    allow_duplicates : bool, Default  = True
        If ``False`` a ListDuplicateError will be raised if the sequence contains any duplicate items.
    allow_reformatting : bool, Default = True
        If ``True`` the string will be reformatted to the correct format. If ``False`` an exception
        will be raised it the string is not correctly formatted.


    Raises
    ------
    ListDuplicateError
        Raised when a string already exist in the sequence and ``allow_duplicates = False``


    Examples
    --------
    >>> MassKeyList([99, 105, '111'])
    ('99', '105', '111']´)
    >>> MassKeyList('99', 105,'111')
    ('99', '105', '111')
    >>> MassKeyList('99', ['105', 111])
    ('99', '105', '111')
    """

    def filter(self, key_eq=None, key_neq=None, flavour_eq = None, flavour_neq= None,
               *, key_lt=None, key_gt=None, key_le=None, key_ge=None):
        """
        Return a new key string list containing only the key strings that satisfies the specified
        filters.

        Parameters
        ----------
        key_eq : str, int, Sequence[(str, int)]
           Only key strings equal to/found in *key_eq* pass this filter.
        key_neq : str, int, Sequence[(str, int)]
           Only key strings not equal to/found in *key_neq* pass this filter.
        flavour_eq : str, Sequence[str]
            Only key strings of this flavour(s) pass this filter.
        flavour_neq : str, Sequence[str]
            Only key strings not of this flavour(s) pass this filter.
        key_lt : str, int
            Only key strings less than *key_lt* pass this filter.
        key_gt : str, int
            Only key strings greater than *key_gt* pass this filter.
        key_le : str, int
            Only key strings less than or equal to *key_le* pass this filter.
        key_ge : str, int
            Only key strings greater than or equal to *key_ge* pass this filter.


        Returns
        -------
        result : MassKeyList
            Key strings in the sequence that satisfy the specified filters

        Examples
        --------
        >>> keylist = MassKeyList(['99', '105', '111'])
        >>> keylist.copy(key_eq=[105, 107, 111])
        ('105', '111')
        >>> keylist.copy(key_neq='111'])
        ('99', '105')
        >>> keylist.copy(key_gt='99'])
        ('105', '111')
        >>> keylist.copy(key_le=105])
        ('99', '105')
        """

        filters = parse_keyfilters(key_eq=key_eq, key_neq=key_neq,
                                     flavour_eq=flavour_eq, flavour_neq=flavour_neq,
                                     key_lt=key_lt, key_gt=key_gt,
                                     key_le=key_le, key_ge=key_ge)
        return self._Flavour__keylist(key for key in self if key._filter(**filters))


class ElementKeyList(ElementFlavour, IsopyKeyList):
    """
    A tuple consisting exclusively of :class:`ElementKeyString` items.

    Behaves just like, and contains all the methods, that a normal tuple does unless otherwise noted.
    Only methods that behave differently from a normal python tuples are documented below.

    Parameters
    ----------
    keys : str, Sequence[str]
        A string or sequence of strings that can be converted to the correct key string type.
    ignore_duplicates : bool, Default = True
        If ``True`` all duplicate items will be removed from the sequence.
    allow_duplicates : bool, Default  = True
        If ``False`` a ListDuplicateError will be raised if the sequence contains any duplicate items.
    allow_reformatting : bool, Default = True
        If ``True`` the string will be reformatted to the correct format. If ``False`` an exception
        will be raised it the string is not correctly formatted.


    Raises
    ------
    ListDuplicateError
        Raised when a string already exist in the sequence and ``allow_duplicates = False``


    Examples
    --------
    >>> ElementKeyList(['ru', 'pd', 'cd'])
    ('Ru', 'Pd', 'Cd')
    >>> ElementKeyList('ru', 'pd' , 'cd')
    ('Ru', 'Pd', 'Cd')
    >>> ElementKeyList('ru', ['pd' , 'cd'])
    ('Ru', 'Pd', 'Cd')
    """

    def filter(self, key_eq = None, key_neq = None, flavour_eq = None, flavour_neq= None):
        """
        Return a new key string list containing only the key strings that satisfies the specified
        filters.


        Parameters
        ----------
        key_eq : str, Sequence[str], Optional
           Only key strings equal to/found in *key_eq* pass this filter.
        key_neq : str, Sequence[str], Optional
           Only key strings not equal to/found in *key_neq* pass this filter.
        flavour_eq : str, Sequence[str]
            Only key strings of this flavour(s) pass this filter.
        flavour_neq : str, Sequence[str]
            Only key strings not of this flavour(s) pass this filter.


        Returns
        -------
        result : ElementKeyList
            Key strings in the sequence that satisfy the specified filters

        Examples
        --------
        >>> keylist = ElementKeyList(['ru', 'pd', 'cd'])
        >>> keylist.copy(key_eq=['pd','ag', 'cd'])
        ('Pd', 'Cd')
        >>> keylist.copy(key_neq='cd')
        ('Ru', 'Pd')
        """
        filters = parse_keyfilters(key_eq=key_eq, key_neq=key_neq,
                                   flavour_eq=flavour_eq, flavour_neq=flavour_neq)
        return self._Flavour__keylist(key for key in self if key._filter(**filters))

    @cached_property
    def charges(self):
        return tuple(item.charge for item in self)

    def set_charges(self, charges):
        if type(charges) is tuple or type(charges) is list:
            if len(charges) != len(self):
                raise ValueError(f'size of charges ({len(charges)}) does not match size of key list ({len(self)})')
            else:
                return self.__class__(self[i].set_charge(charges[i]) for i in range(len(self)))
        elif type(charges) is int or charges is None:
            return self.__class__(self[i].set_charge(charges) for i in range(len(self)))
        else:
            raise TypeError('charges must be None or an integer or a list of integers')


class IsotopeKeyList(IsotopeFlavour, IsopyKeyList):
    """
    A tuple consisting exclusively of :class:`IsotopeKeyString` items.

    Behaves just like, and contains all the methods, that a normal tuple does unless otherwise noted.
    Only methods that behave differently from a normal python tuples are documented below.

    Parameters
    ----------
    keys : str, Sequence[str]
        A string or sequence of strings that can be converted to the correct key string type.
    ignore_duplicates : bool, Default = True
        If ``True`` all duplicate items will be removed from the sequence.
    allow_duplicates : bool, Default  = True
        If ``False`` a ListDuplicateError will be raised if the sequence contains any duplicate items.
    allow_reformatting : bool, Default = True
        If ``True`` the string will be reformatted to the correct format. If ``False`` an exception
        will be raised it the string is not correctly formatted.


    Raises
    ------
    ListDuplicateError
        Raised when a string already exist in the sequence and ``allow_duplicates = False``


    Examples
    --------
    >>> IsotopeKeyList(['99ru', '105pd', '111cd'])
    ('99Ru', '105Pd', '111Cd')
    >>> IsotopeKeyList('99ru', '105pd' , 'cd111')
    ('99Ru', '105Pd', '111Cd')
    >>> IsotopeKeyList('99ru', ['105pd', 'cd111'])
    ('99Ru', '105Pd', '111Cd')
    """

    @cached_property
    def mass_numbers(self) -> MassKeyList:
        """
        Returns a :class:`MassKeyList` containing the mass number of each item in the list.

        Examples
        --------
        >>> IsotopeKeyList(['99ru', '105pd', '111cd']).mass_numbers()
        ('99', '105', '111')
        """
        return MassFlavour._Flavour__keylist(tuple(x.mass_number for x in self))

    @cached_property
    def element_symbols(self) -> ElementKeyList:
        """
        Returns an :class:`ElementKeyList` containing the element symbol of each item in the list.

        Examples
        --------
        >>> IsotopeKeyList(['99ru', '105pd', '111cd']).element_symbols()
        ('Ru', 'Pd', 'Cd')
        """
        return ElementFlavour._Flavour__keylist(tuple(x.element_symbol for x in self))

    def filter(self, key_eq = None, key_neq = None, flavour_eq = None, flavour_neq= None,
               **mass_number_and_element_symbol_kwargs) -> 'IsotopeKeyList':
        """
        Return a new key string list containing only the key strings that satisfies the specified
        filters.

        Parameters
        ----------
        key_eq : str, Sequence[str], Optional
           Only key strings equal to/found in *key_eq* pass this filter.
        key_neq : str, Sequence[str], Optional
           Only key strings not equal to/found in *key_neq* pass this filter.
        flavour_eq : str, Sequence[str]
            Only key strings of this flavour(s) pass this filter.
        flavour_neq : str, Sequence[str]
            Only key strings not of this flavour(s) pass this filter.
        mass_number_and_element_symbol_kwargs : str, Sequence[str], Optional
            Filter based on the mass number or element symbol of the key strings. Prefix
            :func:`MassKeyList.filter` filters with ``mass_number_ and :func:`Element.filter`
            filters with ``element_symbol``.


        Returns
        -------
        result : IsotopeKeyList
            Key strings in the sequence that satisfy the specified filters

        Examples
        --------
        >>> keylist = IsotopeKeyList(['99ru', '105pd', '111cd'])
        >>> keylist.copy(key_eq=['105pd','107ag', '111cd'])
        ('105Pd', '111Cd)
        >>> keylist.copy(key_neq='111cd')
        ('99Ru', '105Pd')
        >>> keylist.copy(mass_number_key_gt = 100)
        ('105Pd', '111Cd)
        >>> keylist.copy(element_symbol_key_neq='pd'])
        ('99Ru', '111Cd)
        """
        filters = parse_keyfilters(key_eq=key_eq, key_neq=key_neq,
                                   flavour_eq=flavour_eq, flavour_neq=flavour_neq,
                                   **mass_number_and_element_symbol_kwargs)

        return self._Flavour__keylist(key for key in self if key._filter(**filters))

    @cached_property
    def charges(self):
        return tuple(item.charge for item in self)

    def set_charges(self, charges):
        if type(charges) is tuple or type(charges) is list:
            if len(charges) != len(self):
                raise ValueError(f'size of charges ({len(charges)}) does not match size of key list ({len(self)})')
            else:
                return self.__class__(self[i].set_charge(charges[i]) for i in range(len(self)))
        elif type(charges) is int or charges is None:
            return self.__class__(self[i].set_charge(charges) for i in range(len(self)))
        else:
            raise TypeError('charges must be None or an integer or a list of integers')

    def mz(self, true_mass = False, isotope_masses = None):
        return tuple(key.mz(true_mass=true_mass, isotope_masses=isotope_masses) for key in self)


class RatioKeyList(RatioFlavour, IsopyKeyList):
    """
    A tuple consisting exclusively of :class:`RatioKeyString` items.

    Behaves just like, and contains all the methods, that a normal tuple does unless otherwise noted.
    Only methods that behave differently from a normal python tuples are documented below.

    Parameters
    ----------
    keys : str, tuple[str, str], Sequence[str, tuple[str, str]]
        A string or sequence of strings that can be converted to the correct key string type.
    ignore_duplicates : bool, Default = True
        If ``True`` all duplicate items will be removed from the sequence.
    allow_duplicates : bool, Default  = True
        If ``False`` a ListDuplicateError will be raised if the sequence contains any duplicate items.
    allow_reformatting : bool, Default = True
        If ``True`` the string will be reformatted to the correct format. If ``False`` an exception
        will be raised it the string is not correctly formatted.


    Raises
    ------
    ListDuplicateError
        Raised when a string already exist in the sequence and ``allow_duplicates = False``

    Examples
    --------
    >>> RatioKeyList(['99ru/108Pd', '105pd/108Pd', '111cd/108Pd'])
    ('99Ru/108Pd', '105Pd/108Pd', '111Cd/108Pd')
    >>> RatioKeyList('99ru/108Pd', '105pd/108Pd' ,'cd111/108Pd')
    ('99Ru/108Pd', '105Pd/108Pd', '111Cd/108Pd')
    >>> RatioKeyList('99ru/108Pd', ['105pd/108Pd' ,'cd111/108Pd'])
    ('99Ru/108Pd', '105Pd/108Pd', '111Cd/108Pd')
    """

    def flatten(self, ignore_duplicates = False,
                allow_duplicates = True, allow_reformatting = False, ignore_charge=False):
        """
        Return a key list with the numerator and denominators key strings joined into a single list.
        """
        return askeylist((*self.numerators.flatten(), *self.denominators.flatten()),
                         ignore_duplicates=ignore_duplicates, allow_duplicates=allow_duplicates,
                         allow_reformatting=allow_reformatting, ignore_charge=ignore_charge)
    @cached_property
    def numerators(self) -> IsopyKeyList:
        """
        Returns an isopy list containing the numerators for each ration in the sequence.

        Examples
        --------
        >>> RatioKeyList(['99ru/108Pd', '105pd/108Pd', '111cd/108Pd']).numerators
        ('99Ru', '105Pd', '111Cd')
        """
        if len(self) == 0:
            return tuple()
        else:
            return askeylist(tuple(rat.numerator for rat in self))

    @cached_property
    def denominators(self)  -> IsopyKeyList:
        """
        Returns an isopy list containing the numerators for each ration in the sequence.

        Examples
        --------
        >>> RatioKeyList(['99ru/108Pd', '105pd/108Pd', '111cd/108Pd']).numerators
        ('108Pd', '108Pd', '108Pd')
        """
        if len(self) == 0:
            return tuple()
        else:
            return keylist(tuple(rat.denominator for rat in self))

    @cached_property
    def common_denominator(self) -> IsopyKeyString:
        """
        The common demoninator of all ratios in the sequence. Returns ``None`` is there is
        no common denominator.


        Examples
        --------
        >>> RatioKeyList(['99ru/108Pd', '105pd/108Pd', '111cd/108Pd']).common_denominator
        '108Pd'
        """

        if len(set(self.denominators)) == 1:
            return self.denominators[0]
        else:
            return None

    @cached_property
    def _has_common_denominator(self) -> bool:
        """
        ``True`` if all the ratios in the sequence as a common denominator otherwise ``False``.

        Examples
        --------
        >>> RatioKeyList(['99ru/108Pd', '105pd/108Pd', '111cd/108Pd']).has_common_denominator
        True
        >>> RatioKeyList(['99ru/108Pd', '105pd/108Pd', '111cd/108Pd']).has_common_denominator
        False
        """
        return len(set(self.denominators)) == 1

    def filter(self, key_eq = None, key_neq = None, flavour_eq = None, flavour_neq= None, **numerator_and_denominator_kwargs):
        """
        Return a new key string list containing only the key strings that satisfies the specified
        filters.

        Parameters
        ----------
        key_eq : str, Sequence[str], Optional
           Only key strings equal to/found in *key_eq* pass this filter.
        key_neq : str, Sequence[str], Optional
           Only key strings not equal to/found in *key_neq* pass this filter.
        flavour_eq : str, Sequence[str]
            Only key strings of this flavour(s) pass this filter.
        flavour_neq : str, Sequence[str]
            Only key strings not of this flavour(s) pass this filter.
        mass_number_and_element_symbol_kwargs : str, Sequence[str], Optional
            Filter based on the numerator and denominator key strings of the ratio. Prefix
            numerator filters with ``numerator_`` and denominator
            filters with ``denominator_``.


        Returns
        -------
        result : RatioKeyList
            Key strings in the sequence that satisfy the specified filters

        Examples
        --------
        >>> keylist = RatioKeyList(['99ru/108Pd', '105pd/108Pd', '111cd/108Pd'])
        >>> keylist.copy(key_eq=['105pd/108Pd','107ag/108Pd', '111cd/108Pd'])
        ('105Pd/108Pd', '111Cd/108Pd')
        >>> keylist.copy(key_neq='111cd/108Pd')
        ('99Ru/108Pd', '105Pd/108Pd')
        >>> keylist.copy(numerator_isotope_symbol_key_eq = ['pd', 'ag', 'cd'])
        ('105Pd/108Pd', '111Cd/108Pd')
        >>> keylist.copy(numerator_mass_number_key_lt = 100)
        ('99Ru/108Pd', '105Pd/108Pd')
        """
        filters = parse_keyfilters(key_eq=key_eq, key_neq=key_neq,
                                   flavour_eq=flavour_eq, flavour_neq=flavour_neq,
                                   **numerator_and_denominator_kwargs)
        return self._Flavour__keylist(key for key in self if key._filter(**filters))


class GeneralKeyList(GeneralFlavour, IsopyKeyList):
    """
    A tuple consisting exclusively of :class:`GeneralKeyString` items.

    Behaves just like, and contains all the methods, that a normal tuple does unless otherwise noted.
    Only methods that behave differently from a normal python tuples are documented below.

    Parameters
    ----------
    keys : str, Sequence[str]
        A string or sequence of strings that can be converted to the correct key string type.
    ignore_duplicates : bool, Default = True
        If ``True`` all duplicate items will be removed from the sequence.
    allow_duplicates : bool, Default  = True
        If ``False`` a ListDuplicateError will be raised if the sequence contains any duplicate items.
    allow_reformatting : bool, Default = True
        If ``True`` the string will be reformatted to the correct format. If ``False`` an exception
        will be raised it the string is not correctly formatted.


    Raises
    ------
    ListDuplicateError
        Raised when a string already exist in the sequence and ``allow_duplicates = False``

    Examples
    --------
    >>> GeneralKeyList([harry, 'ron', 'hermione'])
    ('harry', 'ron', 'hermione')
    >>> GeneralKeyList(harry, 'ron' ,'hermione')
    ('harry', 'ron', 'hermione')
    >>> GeneralKeyList(harry, ['ron' , 'hermione'])
    ('harry', 'ron', 'hermione')
    """
    def filter(self, key_eq= None, key_neq = None, flavour_eq = None, flavour_neq= None):
        """
        Return a new key string list containing only the key strings that satisfies the specified
        filters.

        Parameters
        ----------
        key_eq : str, Sequence[str], Optional
           Only key strings equal to/found in *key_eq* pass this filter.
        key_neq : str, Sequence[str], Optional
           Only key strings not equal to/found in *key_neq* pass this filter.
        flavour_eq : str, Sequence[str]
            Only key strings of this flavour(s) pass this filter.
        flavour_neq : str, Sequence[str]
            Only key strings not of this flavour(s) pass this filter.

        Returns
        -------
        result : GeneralKeyList
            Key strings in the sequence that satisfy the specified filters

        Examples
        --------
        >>> keylist = GeneralKeyList(['harry', 'ron', 'hermione'])
        >>> keylist.copy(key_eq=['harry', 'ron', 'neville'])
        ('harry', 'ron')
        >>> keylist.copy(key_neq='harry')
        ('ron', 'hermione')
        """

        filters = parse_keyfilters(key_eq=key_eq, key_neq=key_neq,
                                   flavour_eq=flavour_eq, flavour_neq=flavour_neq)
        return self._Flavour__keylist(key for key in self if key._filter(**filters))


class MixedKeyList(MixedFlavour, IsopyKeyList):
    """
    A tuple consisting of mixed isopy key items.

    Behaves just like, and contains all the methods, that a normal tuple does unless otherwise noted.
    Only methods that behave differently from a normal python tuples are documented below.

    Parameters
    ----------
    keys : str, Sequence[str]
        A string or sequence of strings that can be converted to the correct key string type.
    ignore_duplicates : bool, Default = True
        If ``True`` all duplicate items will be removed from the sequence.
    allow_duplicates : bool, Default  = True
        If ``False`` a ListDuplicateError will be raised if the sequence contains any duplicate items.
    allow_reformatting : bool, Default = True
        If ``True`` the string will be reformatted to the correct format. If ``False`` an exception
        will be raised it the string is not correctly formatted.


    Raises
    ------
    ListDuplicateError
        Raised when a string already exist in the sequence and ``allow_duplicates = False``

    Examples
    --------
    >>> MixedKeyList([harry, 'ron', 'hermione'])
    ('harry', 'ron', 'hermione')
    >>> MixedKeyList(harry, 'ron' ,'hermione')
    ('harry', 'ron', 'hermione')
    >>> MixedKeyList(harry, ['ron' , 'hermione'])
    ('harry', 'ron', 'hermione')
    """
    def filter(self, key_eq= None, key_neq = None, flavour_eq = None, flavour_neq= None):
        """
        Return a new key string list containing only the key strings that satisfies the specified
        filters.

        Parameters
        ----------
        key_eq : str, Sequence[str], Optional
           Only key strings equal to/found in *key_eq* pass this filter.
        key_neq : str, Sequence[str], Optional
           Only key strings not equal to/found in *key_neq* pass this filter.
        flavour_eq : str, Sequence[str]
            Only key strings of this flavour(s) pass this filter.
        flavour_neq : str, Sequence[str]
            Only key strings not of this flavour(s) pass this filter.

        Returns
        -------
        result : GeneralKeyList
            Key strings in the sequence that satisfy the specified filters

        Examples
        --------
        >>> keylist = MixedKeyList(['Ru', '105Pd', 'Cd'])
        >>> keylist.copy(flavour_eq='element')
        ElementKeyList('Ru', 'Pd')
        >>> keylist.copy(key_neq='element')
        IsotopeKeyList('105Pd')
        """

        filters = parse_keyfilters(key_eq=key_eq, key_neq=key_neq,
                                   flavour_eq=flavour_eq, flavour_neq=flavour_neq)

        return self._Flavour__keylist(key for key in self if key._filter(**filters))

    def sorted(self):
        """
        Return a sorted key string list.
        """
        return self._Flavour__keylist(sorted(self, key= lambda k: (k._Flavour__id, k._sortkey())))

    def flatten(self, ignore_duplicates = False,
                allow_duplicates = True, allow_reformatting = False, ignore_charge=False):
        """
        Return a key list with the numerator and denominators key strings joined into a single list.
        """
        keys = ()
        for key in self:
            if type(key) is RatioKeyString: keys += (key.numerator, key.denominator)
            else: keys += (key,)
        return askeylist(keys,
                         ignore_duplicates=ignore_duplicates, allow_duplicates=allow_duplicates,
                         allow_reformatting=allow_reformatting, ignore_charge=ignore_charge)

############
### Dict ###
############

class IsopyDict(dict):
    """
    Dictionary where each value is stored by a isopy keystring key.

    Behaves just like, and contains all the methods, that a normal dictionary does unless otherwise
    noted. Only methods that behave differently from a normal dictionary are documented below.

    Parameters
    ----------
    args : dict[str, Any], Optional
        Dictionary(ies) where each key can be converted to a keystring.
    default_value : Any, Default = None
        The default value for a key not present in the dictionary. Should ideally be the same type as
        the value stored in the dictionary
    readonly : bool, Default = False
        If ``True`` the dictionary cannot be edited. This attribute is not inherited by child
        dictionaries.
    kwargs : Any, Optional
        Key, Value pairs to be included in the dictionary

    Examples
    --------
    >>> isopy.IsopyDict({'Pd108': 108, '105Pd': 105, 'pd': 46})
    IsopyDict(default_value = None, readonly = False,
    {"108Pd": 108
    "105Pd": 105
    "Pd": 46})

    >>> isopy.IsopyDict(Pd108 = 108, pd105= 105, pd=46, default_value=0)
    IsopyDict(default_value = 0, readonly = False,
    {"108Pd": 108
    "105Pd": 105
    "Pd": 46})

    """

    def __repr__(self):
        items = '\n'.join([f'"{key}": {value}' for key, value in self.items()])
        return f'{self.__class__.__name__}(default_value = {self.default_value}, readonly = {self.readonly},\n{{{items}}})'

    def __init__(self, *args, default_value = NotGiven, readonly =False, **kwargs):
        super(IsopyDict, self).__init__()
        self._readonly = False
        self._default_value = default_value

        for arg in args:
            if isinstance(arg, IsopyArray):
                self.update(arg.to_dict())
            elif isinstance(arg, dict):
                self.update(arg)
            else:
                raise TypeError('arg must be dict')
        self.update(kwargs)
        self._readonly = readonly

    def __delitem__(self, key):
        if self._readonly is True:
            raise TypeError('this dictionary is readonly. Make a copy to make changes')

        key = askeystring(key)
        super(IsopyDict, self).__delitem__(key)

    def __setitem__(self, key, value):
        if self._readonly is True:
            raise TypeError('this dictionary is readonly. Make a copy to make changes')

        key = askeystring(key)
        super(IsopyDict, self).__setitem__(key, value)

    def __contains__(self, key):
        key = askeystring(key)
        return super(IsopyDict, self).__contains__(key)

    def __getitem__(self, key):
        key = askeystring(key)
        return super(IsopyDict, self).__getitem__(key)

    @property
    def readonly(self) -> bool:
        """
        ``True`` if the dictionary cannot be edited otherwise ``False``.

        This attribute is **not** inherited by derivative dictionaries.
        """
        return self._readonly

    @property
    def default_value(self):
        """
        The default value, if given, for keys not present in the dictionary using the ``get()`` method.

        This attribute in inherited by derivative dictionaries.
        """
        return self._default_value

    def update(self, other):
        """
        Update the dictionary with the key/value pairs from other, overwriting existing keys.

        A TypeError is raised if the dictionary is readonly.
        """
        if self._readonly is True:
            raise TypeError('this dictionary is readonly. Make a copy to make changes')

        if not isinstance(other, dict):
            raise ValueError('other must be a dict')

        for k in other.keys():
            self.__setitem__(k, other[k])

    def pop(self, key, default=NotGiven):
        """
        If *key* is in the dictionary, remove it and return its value, else return *default*. If
        *default* is not given the default value of hte dictionary is used.

        A TypeError is raised if the dictionary is readonly.
        """
        if self._readonly is True:
            raise TypeError('this dictionary is readonly. Make a copy to make changes')

        if default is NotGiven:

            default = self._default_value
        key = askeystring(key)
        if key in self:
            return super(IsopyDict, self).pop(key)
        elif default is not NotGiven:
            return default
        else:
            raise ValueError('No default value given')

    def setdefault(self, key, default=NotGiven):
        """
        If *key* in dictionary, return its value. If not, insert *key* with the default value and
        the default value. If *default* is not given the default value of the dictionary is used.

        A TypeError is raised if the dictionary is readonly and *key* is not in the dictionary.
        """
        key = askeystring(key)
        if default is NotGiven:
            default = self._default_value

        if key not in self:
            if self._readonly is True:
                raise TypeError('this dictionary is readonly. Make a copy to make changes')

            if default is NotGiven:
                raise ValueError('No default value given')

            self.__setitem__(key, default)

        return self.__getitem__(key)

    def copy(self, **key_filters):
        """
        Returns a copy of the current dictionary.

        If key filters are given then only the items whose keys pass the key filter is included in the returned
        dictionary.
        """
        if key_filters:
            key_filters = parse_keyfilters(**key_filters)
            keys = [k for k in self if k._filter(**key_filters)]
            return self.__class__({key: self[key] for key in keys}, default_value=self._default_value)
        else:
            return self.__class__(self, default_value = self._default_value)

    def clear(self):
        """
        Removes all items from the dictionary.

        A TypeError is raised if the dictionary is readonly.
        """
        if self._readonly is True:
            raise TypeError('this dictionary is readonly. Make a copy to make changes')
        super(IsopyDict, self).clear()

    def get(self, key=None, default=NotGiven, **key_filters):
        """
        Return the the value for *key* if present in the dictionary. Otherwise *default* is
        returned.

        If *key* is a sequence of keys a tuple containing the values for each key is returned.

        If *default* is not given the default value of the dictionary is used.

        Examples
        --------
        >>> reference = isopy.IsopyDict({'108Pd': 100, '105Pd': 20, '104Pd': 150})
        >>> reference.get('pd108')
        100
        >>> reference.get('104Pd/105Pd')
        None
        >>> reference.get('104Pd/105Pd', default=np.nan)
        nan


        >>> reference = isopy.IsopyDict({'108Pd': 100, '105Pd': 20, '104Pd': 150}, default_value=np.nan)
        >>> reference.get('104Pd/105Pd')
        nan
        """
        if default is NotGiven:
            default = self._default_value

        if isinstance(key, (str, int)):
            key = askeystring(key)
            try:
                return super(IsopyDict, self).__getitem__(key)
            except KeyError:
                if key != key.basekey:
                    try: return super(IsopyDict, self).__getitem__(key.basekey)
                    except KeyError: pass

                if default is NotGiven:
                    raise ValueError('No default value given')
                else:
                    return default

        if hasattr(key, '__iter__'):
            return tuple(self.get(k, default) for k in key)

        raise TypeError(f'key type {type(key)} not understood')


class ScalarDict(IsopyDict):
    """
    Dictionary where each value is stored as a numpy float by a isopy keystring key.

    Behaves just like, and contains all the methods, that a normal dictionary does unless otherwise
    noted. Only methods that behave differently from a normal dictionary are documented below.

    Parameters
    ----------
    *args : dict[str, scalar], Optional
        Dictionary(ies) where each key can be converted to a keystring.
    default_value : scalar, Default = np.nan
        The default value for a key not present in the dictionary. Should ideally be the same type as
        the value stored in the dictionary
    readonly : bool, Default = True
        If ``True`` the dictionary cannot be edited. This attribute is not inherited by child
        dictionaries.
    kwargs : scalar, Optional
        Key, Value pairs to be included in the dictionary

    Examples
    --------
    >>> isopy.ScalarDict({'Pd108': 108, '105Pd': 105, 'pd': 46})
    ScalarDict(default_value = nan, readonly = False,
    {"108Pd": 108.0
    "105Pd": 105.0
    "Pd": 46.0})

    >>> isopy.ScalarDict(Pd108 = 108, pd105= 105, pd=46, default_value=0)
    ScalarDict(default_value = 0, readonly = False,
    {"108Pd": 108.0
    "105Pd": 105.0
    "Pd": 46.0})

    """

    def __init__(self, *args: dict, default_value=nan,
                 readonly= False, **kwargs):
        try:
            default_value = np.float_(default_value)
        except Exception as err:
            raise ValueError(f'unable to convert default value to float') from err
        super(ScalarDict, self).__init__(*args, default_value=default_value, readonly=readonly, **kwargs)

    def __setitem__(self, key, value):
        try:
            value = np.float_(value)
        except Exception as err:
            raise ValueError(f'key "{key}": unable to convert value {value} to float') from err

        super(ScalarDict, self).__setitem__(key, value)


    def get(self, key = None, default = NotGiven):
        """
        Return the the value for *key* if present in the dictionary.

        If *key* is a sequence of keys then an array in returned containing the value for each key.

        If *key* not in the dictionary
        and *key* is a ratio key string the numerator value divided by the denominator value will
        be returned if both key strings are present in the dictionary. Otherwise *default* is
        returned. If *key* is a sequence of keys a dictionary is returned. If *default* is
        not given the default value of the dictionary is used.

        Examples
        --------
        >>> reference = ScalarDict({'108Pd': 100, '105Pd': 20, '104Pd': 150})
        >>> reference.get('pd108')
        100
        >>> reference.get('104Pd/105Pd')
        7.5
        >>> reference.get('110Pd/105Pd')
        nan
        """
        if default is NotGiven:
            default = self._default_value
        else:
            try:
                default = np.float64(default)
            except Exception as err:
                raise ValueError(f'unable to convert default value to float') from err

        if isinstance(key, (str, int)):
            key = askeystring(key)

            try:
                return super(IsopyDict, self).__getitem__(key)
            except KeyError:
                if key != key.basekey:
                    try: return super(IsopyDict, self).__getitem__(key.basekey)
                    except KeyError: pass
                if type(key) is RatioKeyString:
                    try:
                        result = super(IsopyDict, self).__getitem__(key.numerator)
                        result /= super(IsopyDict, self).__getitem__(key.denominator)
                        return result
                    except KeyError:
                        pass

                return default

        elif hasattr(key, '__iter__'):
            return np.array(super(ScalarDict, self).get(key, default))
        else:
            return super(ScalarDict, self).get(key, default)


    def to_array(self, keys = None, default=NotGiven, **key_filters):
        """
        Returns an isopy array based on values in the dictionary.

        If *keys* are given then the array will contain these keys. If no *keys* are given then the
        array will contain all the values in the dictionary unless *key_filters* are given.

        If key filters are specified then only the items that pass these filters are included in
        the returned array.
        """
        if key_filters:
            d = self.copy(**key_filters)
        else:
            d = self

        if keys is not None:
            d = {k: d.get(k, default=default) for k in askeylist(keys)}

        return array(d)

    @renamed_function(to_array)
    def asarray(self, keys=None, default=NotGiven, **key_filters):
        pass

#############
### Array ###
#############
class IsopyArray(IsopyFlavour):
    def __repr__(self):
        return self.to_text(**ARRAY_REPR)

    def __str__(self):
        return self.to_text(', ', False, False)

    def __eq__(self, other):
        try:
            other = np.asanyarray(other)
        except:
            return False
        if other.shape != self.shape:
            return False
        else:
            return np.all(np.equal(self, other), axis=None)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # if not ufunc.__module__ == 'isopy': isopy has no ufuncs oly array functions
        allowed = ALLOWED_NUMPY_FUNCTIONS.get(ufunc, False)
        if allowed is False:
            warnings.warn(f"The functionality of {ufunc.__name__} has not been tested with isopy arrays.")

        elif allowed is not True:
            # reimplemented function
            return allowed(*inputs, **kwargs)

        if method != '__call__':
            ufunc = getattr(ufunc, method)
            #raise TypeError(f'the {ufunc.__name__} method "{method}" is not supported by isopy arrays')

        return call_array_function(ufunc, *inputs, **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        if not func.__module__ == 'isopy':
            allowed = ALLOWED_NUMPY_FUNCTIONS.get(func, False)
            if allowed is False:
                warnings.warn(f"The functionality of {func.__name__} has not been tested with isopy arrays.")

            elif allowed is not True:
                # reimplemented function
                return allowed(*args, **kwargs)

        return call_array_function(func, *args, **kwargs)

    def get(self, key, default = nan):
        """
        Returns the values of column *key* if present in the array. Otherwise an numpy array
        filled with *default* is returned with the same shape as a column in the array. An
        exception will be raised if *default* cannot be broadcast to the correct shape.
        """
        try:
            key = self._Flavour__keystring(key)
            return self.__getitem__(key)
        except:
            return np.full(self.shape, default)

    def copy(self, **key_filters):
        """
        Returns a copy of the array. If *key_filters* are given then the returned array only
        contains the columns that satisfy the *key_filter* filters.
        """
        if key_filters:
            return self.filter(**key_filters).copy()
        else:
            return super(IsopyArray, self).copy()


    def filter(self, **key_filters):
        """
        Returns a view of the array containing the columns that satisfy the *key_filter* filters.
        """
        if key_filters:
            return self[self.keys.filter(**key_filters)]
        else:
            return self

    def ratio(self, denominator=None, remove_denominator = True):
        """
        Divide all values in the array by the *denominator* column and return a :class:`RatioArray`.

        If not denominator is given the key in the array with the largest value will be used as
        the denominator. If *remove_denominator* is ``True`` the denominator/denominator ratio
        is not included in the returned array.
        """
        if denominator is None:
            denominator = isopy.keymax(self)

        keys = self.keys()
        if denominator not in keys:
            raise ValueError(f'key "{denominator}" not found in keys of the array')

        if remove_denominator:
            keys = keys - denominator

        return RatioArray(self[keys] / self[denominator], keys=keys/denominator)

    def items(self):
        """
        Returns a tuple containing a tuple with the key and the column values for each key in the array

        Equivalent to ``tuple([)(key, array[key]) for key in array.keys)``
        """
        return tuple((key, self[key]) for key in self.keys)

    def values(self):
        """
        Returns a tuple containing the column values for each key in the array

        Equivalent to ``tuple(array[key] for key in array.keys)``
        """
        return tuple(self[key] for key in self.keys)

    def normalise(self, value = 1, key = None):
        """
        Normalise the values in each row so that the the value of *key* is equal to *value*.

        If *key* is a sequence of keys then the sum of those keys will be set equal to *value*. If
        *keys* is not given then the sum of all columns will be used. *key* can also be a callable
        that takes an array and return a key string.

        **Note** returns a copy of the array.

        Examples
        --------
        >>> array = isopy.tb.make_ms_array('pd')
        >>> array
        (row) , 102Pd  , 104Pd  , 105Pd  , 106Pd  , 108Pd  , 110Pd
        None  , 0.0102 , 0.1114 , 0.2233 , 0.2733 , 0.2646 , 0.1172
        >>> array.normalise(10)
        (row) , 102Pd , 104Pd , 105Pd , 106Pd , 108Pd , 110Pd
        None  , 0.102 , 1.114 , 2.233 , 2.733 , 2.646 , 1.172
        >>> array.normalise(1, 'pd102')
        (row) , 102Pd , 104Pd  , 105Pd  , 106Pd  , 108Pd  , 110Pd
        None  , 1     , 10.922 , 21.892 , 26.794 , 25.941 , 11.49
        >>> array.normalise(10, ['pd106', 'pd108'])
        (row) , 102Pd   , 104Pd , 105Pd  , 106Pd  , 108Pd  , 110Pd
        None  , 0.18963 , 2.071 , 4.1513 , 5.0809 , 4.9191 , 2.1788
        >>> array.normalise(10, isopy.keymax)
        (row) , 102Pd   , 104Pd  , 105Pd  , 106Pd , 108Pd  , 110Pd
        None  , 0.37322 , 4.0761 , 8.1705 , 10    , 9.6817 , 4.2883
        """
        if key is None:
            multiplier = value / np.nansum(self, axis = 1)
        elif isinstance(key, (list, tuple)):
            multiplier = value / np.nansum(self[key], axis=1)
        elif callable(key):
            key = key(self)
            return self.normalise(value, key)
        elif isinstance(key, str):
            multiplier = value / self[key]
        else:
            raise TypeError(f'Got unexpected type "{type(key).__name__}" for key ')

        return self * multiplier

    @property
    def keys(self):
        """
        A key string list containing the name of each column in the array.

        ``array.keys()`` is also allowed as calling a key string list will just return a
        pointer to itself.
        """
        return self._Flavour__keylist(self.dtype.names, allow_reformatting=False)

    @property
    def ncols(self):
        """Number of columns in the array"""
        return len(self.dtype.names)

    @property
    def nrows(self):
        """
        Number of rows in the array. If the array is 0-dimensional ``-1`` is returned.
        """
        if self.ndim == 0:
            return -1
        else:
            return self.size

    def __to_text(self, delimiter=', ', include_row = False, include_dtype=False,
                nrows = None, **vformat):

        sdict = {}
        if include_row:
            if self.ndim == 0:
                sdict['(row)'] = ['None']
            else:
                sdict['(row)'] = [str(i) for i in range(self.size)]

        for k in self.keys():
            val = self[k]
            if include_dtype:
                title = f'{k} ({val.dtype.kind}{val.dtype.itemsize})'
            else:
                title = f'{k}'
            if val.ndim == 0:
                sdict[title] = [vformat.get(val.dtype.kind, '{}').format(self[k])]
            else:
                sdict[title] = [vformat.get(val.dtype.kind, '{}').format(self[k][i]) for i in
                                range(self.size)]

        if nrows is not None and nrows > 2 and nrows < self.size:
            first = nrows // 2
            last = self.size - (nrows // 2 + nrows % 2)
            for title, value in sdict.items():
                sdict[title] = value[:first] + ['...'] + value[last:]
            nrows += 1
        else:
            nrows = self.size

        flen = {}
        for k in sdict.keys():
            flen[k] = max([len(x) for x in sdict[k]]) + 1
            if len(k) >= flen[k]: flen[k] = len(k) + 1

        return flen, sdict, nrows

    def to_text(self, delimiter=', ', include_row = False, include_dtype=False,
                nrows = None, **vformat):
        """
        Returns a string containing the contents of the array.

        Parameters
        ----------
        delimiter : str, Default = ', '
            String used to separate columns in each row.
        include_row : bool, Default = False
            If ``True`` a column containing the row index is included. *None* Is given as the
            row index for 0-dimensional arrays.
        include_dtype : bool, Default = False
            If ``True`` the column data type is included in the first row next to the column name.
        nrows : int, Optional
            The number of rows to show.
        vformat : str, Optional
            Format string for different kinds of data. The key denoted the data kind. Common data
            kind strings ara ``"f"`` for floats, ``"i"`` for integers and ``"S"`` for strings.
            Dictionary containing a format string for different kinds of data.  Most common ``"f"``.
            Default format string for each data type is ``'{}'``. A list of all avaliable data kinds
            is avaliable `here <https://numpy.org/doc/stable/reference/arrays.interface.html>`_.
        """

        flen, sdict, nrows = self.__to_text(delimiter, include_row, include_dtype, nrows, **vformat)

        return '{}\n{}'.format(delimiter.join(['{:<{}}'.format(k, flen[k]) for k in sdict.keys()]),
                                   '\n'.join('{}'.format(delimiter.join('{:<{}}'.format(sdict[k][i], flen[k]) for k in sdict.keys()))
                                             for i in range(nrows)))

    def to_table(self, include_row = False, include_dtype=False,
                nrows = None, **vformat):
        """
        Returns a text string of a markdown table containing the contents of the array.

        Parameters
        ----------
        include_row : bool, Default = False
            If ``True`` a column containing the row index is included. *None* Is given as the
            row index for 0-dimensional arrays.
        include_dtype : bool, Default = False
            If ``True`` the column data type is included in the first row next to the column name.
        nrows : int, Optional
            The number of rows to show.
        vformat : str, Optional
            Format string for different kinds of data. The key denoted the data kind. Common data
            kind strings ara ``"f"`` for floats, ``"i"`` for integers and ``"S"`` for strings.
            Dictionary containing a format string for different kinds of data.  Most common ``"f"``.
            Default format string for each data type is ``'{}'``. A list of all avaliable data kinds
            is avaliable `here <https://numpy.org/doc/stable/reference/arrays.interface.html>`_.
        """

        delimiter = '| '

        flen, sdict, nrows = self.__to_text(delimiter, include_row, include_dtype, nrows, **vformat)

        lines = []
        for k in flen.keys():
            if k == '(row)':
                lines.append('-' * flen[k])
            else:
                lines.append('-' * (flen[k]-1) + ':')


        return '{}\n{}\n{}'.format(delimiter.join(['{:<{}}'.format(k, flen[k]) for k in sdict.keys()]),
                                   delimiter.join(lines),
                               '\n'.join('{}'.format(delimiter.join(
                                   '{:<{}}'.format(sdict[k][i], flen[k]) for k in sdict.keys()))
                                         for i in range(nrows)))

    def display_table(self, include_row = False, include_dtype=False,
                nrows = None, **vformat):
        """
       Returns a Markdown display of a table containing the contents of the array. This will render
       a table in an IPython console or a Jupyter cell.

       an error is raised if IPython is not installed.

       Parameters
       ----------
       include_row : bool, Default = False
           If ``True`` a column containing the row index is included. *None* Is given as the
           row index for 0-dimensional arrays.
       include_dtype : bool, Default = False
           If ``True`` the column data type is included in the first row next to the column name.
       nrows : int, Optional
           The number of rows to show.
       vformat : str, Optional
           Format string for different kinds of data. The key denoted the data kind. Common data
           kind strings ara ``"f"`` for floats, ``"i"`` for integers and ``"S"`` for strings.
           Dictionary containing a format string for different kinds of data.  Most common ``"f"``.
           Default format string for each data type is ``'{}'``. A list of all avaliable data kinds
           is avaliable `here <https://numpy.org/doc/stable/reference/arrays.interface.html>`_.
       """
        if IPython is not None:
            return IPython.display.Markdown(self.to_table(include_row, include_dtype, nrows, **vformat))
        else:
            raise TypeError('IPython not installed')


    def to_list(self):
        """
        Return a list containing the data in the array.
        """
        if self.ndim == 0:
            return list(self.tolist())
        else:
            return [list(row) for row in self.tolist()]

    def to_dict(self):
        """
        Return a dictionary containing a list of the data in each column of the array.
        """
        return {str(key): self[key].tolist() for key in self.keys()}

    def to_ndarray(self):
        """Return a copy of the array as a normal numpy ndarray"""
        return self.view(ndarray).copy()

    def to_clipboard(self, delimiter=', ', include_row= False,
                     include_dtype= False, **vformat):
        """
        Copy the string returned from ``array.to_text(*args, **kwargs)`` to the clipboard.
        """
        string = self.to_text(delimiter=delimiter, include_row=include_row, include_dtype=include_dtype, **vformat)
        pyperclip.copy(string)
        return string

    def to_csv(self, filename, comments = None):
        """
        Save array to a cv file.

        If *filename* already exits it will be overwritten. If *comments*
        are given they will be included before the array data.
        """
        isopy.write_csv(filename, self, comments=comments)

    def to_xlsx(self, filename, sheetname = 'sheet1',
                comments = None, append = False):
        """
        Save array to a excel workbook.

        If *sheetname* is not given the array will be saved as
        "sheet1".  If *comments* are given they will be included before the array data. Existing
        files will be overwritten.
        """
        #If *filename* exists and *append* is ``True`` the sheet will be added to the existing workbook. Otherwise the existing file will be overwritten.
        if sheetname is None:
            sheetname = 'sheet1'

        # TODO if sheetname is given and file exits open it and add the sheet, overwrite if nessecary
        isopy.write_xlsx(filename, comments=comments, append=append, **{sheetname: self})

    def to_dataframe(self):
        """
        Convert array to a pandas dataframe. An exception is raised if pandas is not installed.
        """
        if pandas is not None:
            return pandas.DataFrame(self)
        else:
            raise TypeError('Pandas is not installed')

    # ufuncs
    @functools.wraps(np.ndarray.all)
    def all(self, *args, **kwargs):
        return np.all(self, *args, **kwargs)

    @functools.wraps(np.ndarray.any)
    def any(self, *args, **kwargs):
        return np.any(self, *args, **kwargs)

    @functools.wraps(np.ndarray.cumprod)
    def cumprod(self, *args, **kwargs):
        return np.cumprod(self, *args, **kwargs)

    @functools.wraps(np.ndarray.cumsum)
    def cumsum(self, *args, **kwargs):
        return np.cumsum(self, *args, **kwargs)

    @functools.wraps(np.ndarray.max)
    def max(self, *args, **kwargs):
        return np.max(self, *args, **kwargs)

    @functools.wraps(np.ndarray.min)
    def min(self, *args, **kwargs):
        return np.min(self, *args, **kwargs)

    @functools.wraps(np.ndarray.mean)
    def mean(self, *args, **kwargs):
        return np.mean(self, *args, **kwargs)

    @functools.wraps(np.ndarray.prod)
    def prod(self, *args, **kwargs):
        return np.prod(self, *args, **kwargs)

    @functools.wraps(np.ndarray.ptp)
    def ptp(self, *args, **kwargs):
        return np.ptp(self, *args, **kwargs)

    @functools.wraps(np.ndarray.std)
    def std(self, *args, **kwargs):
        return np.std(self, *args, **kwargs)

    @functools.wraps(np.ndarray.sum)
    def sum(self, *args, **kwargs):
        return np.sum(self, *args, **kwargs)

    @functools.wraps(np.ndarray.var)
    def var(self, *args, **kwargs):
        return np.var(self, *args, **kwargs)


class MassArray(IsopyArray):
    """
    An array where data is stored in named columns with :class:`MassKeyString` keys.

    This is a custom subclass of a `structured numpy array <https://docs.scipy.org/doc/numpy/user/basics.rec.html>`_
    and therefore contains all the methods and attributes that a normal numpy ndarray does. However, these
    may **not** work as expected and caution is advised when using attributes/methods not described below or in the
    tutorial.

    Parameters
    ----------
    values : dict, list, numpy_array, isopy_array
        Values can be a dictionary containing values or a sequence of values. A sequence containing
        values or sequences of values. A structured numpy array or a subclass thereof (e.g. pandas
        dataframe).
    keys : Sequence[str], Optional
        Name of each column in the array. Does not need to be given if *values* is a dictionary
        or structured numpy array or if *dtype* is a ``np.dtype`` containing named columns.
    ndim : {-1, 0, 1}, Optional
        Number of dimensions of the returned array. If ``-1`` then the final array will be
        0-dimensional if it has a size of 1 otherwise it will be 1-dimensional.
    dtype : numpy_dtype, Sequence[numpy_dtype], dict[str, numpy_dtype], Optional
        Any data type accepted by numpy. Can also be a sequence of data types in which case the
        first data type in the sequence for which a conversion is possible is used. Data types
        for individual columns can be specified by a dictionary mapping the column name to the
        data type. If not given the data type is inferred from *values* if they already
        have a numpy data type. Otherwise values are converted to ``np.float64`` if possible. If
        conversion fails the default data type from ``np.array(values[column])`` is used.
    """
    def __new__(cls, values, keys=None, dtype=None, ndim=None):
        return MassNdarray(values, keys=keys, ndim=ndim, dtype=dtype)


class ElementArray(IsopyArray):
    """
    An array where data is stored in named columns with :class:`ElementKeyString` keys.

    This is a custom subclass of a `structured numpy array <https://docs.scipy.org/doc/numpy/user/basics.rec.html>`_
    and therefore contains all the methods and attributes that a normal numpy ndarray does. However, these
    may **not** work as expected and caution is advised when using attributes/methods not described below or in the
    tutorial.

    Parameters
    ----------
    values : dict, list, numpy_array, isopy_array
        Values can be a dictionary containing values or a sequence of values. A sequence containing
        values or sequences of values. A structured numpy array or a subclass thereof (e.g. pandas
        dataframe).
    keys : Sequence[str], Optional
        Name of each column in the array. Does not need to be given if *values* is a dictionary
        or structured numpy array or if *dtype* is a ``np.dtype`` containing named columns.
    ndim : {-1, 0, 1}, Optional
        Number of dimensions of the returned array. If ``-1`` then the final array will be
        0-dimensional if it has a size of 1 otherwise it will be 1-dimensional.
    dtype : numpy_dtype, Sequence[numpy_dtype], dict[str, numpy_dtype], Optional
        Any data type accepted by numpy. Can also be a sequence of data types in which case the
        first data type in the sequence for which a conversion is possible is used. Data types
        for individual columns can be specified by a dictionary mapping the column name to the
        data type. If not given the data type is inferred from *values* if they already
        have a numpy data type. Otherwise values are converted to ``np.float64`` if possible. If
        conversion fails the default data type from ``np.array(values[column])`` is used.
    """
    def __new__(cls, values, keys=None, dtype=None, ndim=None):
        return ElementNdarray(values, keys=keys, ndim=ndim, dtype=dtype)


class IsotopeArray(IsopyArray):
    """
    An array where data is stored in named columns with :class:`IsotopeKeyString` keys.

    This is a custom subclass of a `structured numpy array <https://docs.scipy.org/doc/numpy/user/basics.rec.html>`_
    and therefore contains all the methods and attributes that a normal numpy ndarray does. However, these
    may **not** work as expected and caution is advised when using attributes/methods not described below or in the
    tutorial.

    Parameters
    ----------
    values : dict, list, numpy_array, isopy_array
        Values can be a dictionary containing values or a sequence of values. A sequence containing
        values or sequences of values. A structured numpy array or a subclass thereof (e.g. pandas
        dataframe).
    keys : Sequence[str], Optional
        Name of each column in the array. Does not need to be given if *values* is a dictionary
        or structured numpy array or if *dtype* is a ``np.dtype`` containing named columns.
    ndim : {-1, 0, 1}, Optional
        Number of dimensions of the returned array. If ``-1`` then the final array will be
        0-dimensional if it has a size of 1 otherwise it will be 1-dimensional.
    dtype : numpy_dtype, Sequence[numpy_dtype], dict[str, numpy_dtype], Optional
        Any data type accepted by numpy. Can also be a sequence of data types in which case the
        first data type in the sequence for which a conversion is possible is used. Data types
        for individual columns can be specified by a dictionary mapping the column name to the
        data type. If not given the data type is inferred from *values* if they already
        have a numpy data type. Otherwise values are converted to ``np.float64`` if possible. If
        conversion fails the default data type from ``np.array(values[column])`` is used.
    """
    def __new__(cls, values, keys=None, dtype=None, ndim=None):
        return IsotopeNdarray(values, keys=keys, ndim=ndim, dtype=dtype)


class RatioArray(IsopyArray):
    """
    An array where data is stored in named columns with :class:`RatioKeyString` keys.

    This is a custom subclass of a `structured numpy array <https://docs.scipy.org/doc/numpy/user/basics.rec.html>`_
    and therefore contains all the methods and attributes that a normal numpy ndarray does. However, these
    may **not** work as expected and caution is advised when using attributes/methods not described below or in the
    tutorial.

    Parameters
    ----------
    values : dict, list, numpy_array, isopy_array
        Values can be a dictionary containing values or a sequence of values. A sequence containing
        values or sequences of values. A structured numpy array or a subclass thereof (e.g. pandas
        dataframe).
    keys : Sequence[str], Optional
        Name of each column in the array. Does not need to be given if *values* is a dictionary
        or structured numpy array or if *dtype* is a ``np.dtype`` containing named columns.
    ndim : {-1, 0, 1}, Optional
        Number of dimensions of the returned array. If ``-1`` then the final array will be
        0-dimensional if it has a size of 1 otherwise it will be 1-dimensional.
    dtype : numpy_dtype, Sequence[numpy_dtype], dict[str, numpy_dtype], Optional
        Any data type accepted by numpy. Can also be a sequence of data types in which case the
        first data type in the sequence for which a conversion is possible is used. Data types
        for individual columns can be specified by a dictionary mapping the column name to the
        data type. If not given the data type is inferred from *values* if they already
        have a numpy data type. Otherwise values are converted to ``np.float64`` if possible. If
        conversion fails the default data type from ``np.array(values[column])`` is used.
    """
    def __new__(cls, values, keys=None, dtype=None, ndim=None):
        return RatioNdarray(values, keys=keys, ndim=ndim, dtype=dtype)

    def deratio(self, denominator_value=1):
        """
        Return a array with the numerators and the common denominator as columns. Values for the
        numerators will be copied from the original array and the entire array will be multiplied by
        *denominator_value*.

        An exception is raised if the array does not contain a common denominator.
        """

        denominator = self.keys().common_denominator
        if denominator is None:
            raise ValueError('Column keys do not have a common denominator')
        numerators = self.keys().numerators

        out = numerators._Flavour__array(self, numerators)
        if denominator not in out.keys():
            out = concatenate(out, ones(self.nrows, denominator), axis=1)
        return out * denominator_value


class GeneralArray(IsopyArray):
    """
    An array where data is stored in named columns with :class:`GeneralKeyString` keys.

    This is a custom subclass of a `structured numpy array <https://docs.scipy.org/doc/numpy/user/basics.rec.html>`
    and therefore contains all the methods and attributes that a normal numpy ndarray does. However, these
    may **not** work as expected and caution is advised when using attributes/methods not described below or in the
    tutorial.

    Parameters
    ----------
    values : dict, list, numpy_array, isopy_array
        Values can be a dictionary containing values or a sequence of values. A sequence containing
        values or sequences of values. A structured numpy array or a subclass thereof (e.g. pandas
        dataframe).
    keys : Sequence[str], Optional
        Name of each column in the array. Does not need to be given if *values* is a dictionary
        or structured numpy array or if *dtype* is a ``np.dtype`` containing named columns.
    ndim : {-1, 0, 1}, Optional
        Number of dimensions of the returned array. If ``-1`` then the final array will be
        0-dimensional if it has a size of 1 otherwise it will be 1-dimensional.
    dtype : numpy_dtype, Sequence[numpy_dtype], dict[str, numpy_dtype], Optional
        Any data type accepted by numpy. Can also be a sequence of data types in which case the
        first data type in the sequence for which a conversion is possible is used. Data types
        for individual columns can be specified by a dictionary mapping the column name to the
        data type. If not given the data type is inferred from *values* if they already
        have a numpy data type. Otherwise values are converted to ``np.float64`` if possible. If
        conversion fails the default data type from ``np.array(values[column])`` is used.
    """
    def __new__(cls, values, keys=None, dtype=None, ndim=None):
        return GeneralNdarray(values, keys=keys, ndim=ndim, dtype=dtype)


class MixedArray(IsopyArray):
    """
    An array where data is stored in named columns with mixed isopy keys.

    This is a custom subclass of a `structured numpy array <https://docs.scipy.org/doc/numpy/user/basics.rec.html>`
    and therefore contains all the methods and attributes that a normal numpy ndarray does. However, these
    may **not** work as expected and caution is advised when using attributes/methods not described below or in the
    tutorial.

    Parameters
    ----------
    values : dict, list, numpy_array, isopy_array
        Values can be a dictionary containing values or a sequence of values. A sequence containing
        values or sequences of values. A structured numpy array or a subclass thereof (e.g. pandas
        dataframe).
    keys : Sequence[str], Optional
        Name of each column in the array. Does not need to be given if *values* is a dictionary
        or structured numpy array or if *dtype* is a ``np.dtype`` containing named columns.
    ndim : {-1, 0, 1}, Optional
        Number of dimensions of the returned array. If ``-1`` then the final array will be
        0-dimensional if it has a size of 1 otherwise it will be 1-dimensional.
    dtype : numpy_dtype, Sequence[numpy_dtype], dict[str, numpy_dtype], Optional
        Any data type accepted by numpy. Can also be a sequence of data types in which case the
        first data type in the sequence for which a conversion is possible is used. Data types
        for individual columns can be specified by a dictionary mapping the column name to the
        data type. If not given the data type is inferred from *values* if they already
        have a numpy data type. Otherwise values are converted to ``np.float64`` if possible. If
        conversion fails the default data type from ``np.array(values[column])`` is used.
    """
    def __new__(cls, values, keys=None, dtype=None, ndim=None):
        return MixedNdArray(values, keys=keys, ndim=ndim, dtype=dtype)

###############
### Ndarray ###
###############
class IsopyNdarray(IsopyArray, ndarray):
    def __new__(cls, values, keys=None, *, dtype=None, ndim=None):
        if type(values) is cls and keys is None and dtype is None and ndim is None:
            return values.copy()

        #Do this early so no time is wasted if it fails
        if keys is None and (isinstance(dtype, np.dtype) and dtype.names is not None):
            keys = cls._Flavour__keylist(dtype.names, allow_duplicates=False)

        if ndim is not None and (not isinstance(ndim , int) or ndim < -1 or ndim > 1):
            raise ValueError('parameter "ndim" must be -1, 0 or 1')

        if keys is None and type(dtype) is np.dtype and dtype.names is not None:
            keys = list(dtype.names)

        if pandas is not None and isinstance(values, pandas.DataFrame):
            values = values.to_records(index=False)

        if tables is not None and isinstance(values, tables.Table):
            values = values.read()

        if isinstance(values, (ndarray, void)):
            if values.dtype.names is not None:
                if keys is None:
                    keys = list(values.dtype.names)

                if dtype is None:
                    dtype = [(values.dtype[i],) for i in range(len(values.dtype))]

            else:
                if dtype is None:
                    dtype = [(values.dtype,) for i in range(values.shape[-1])]

            values = values.tolist()

        if isinstance(values, (list, tuple)):
            if [type(v) is list or type(v) is tuple or (isinstance(v, np.ndarray) and v.ndim > 0)
                                                    for v in values].count(True) == len(values):
                values = [tuple(v) for v in values]
            else:
                values = tuple(values)

        elif isinstance(values, dict):
            if keys is None:
                keys = list(values.keys())

            if dtype is None:
                dtype = [(v.dtype,) if isinstance(v, np.ndarray) else (float64, None)
                                                                    for v in values.values()]

            values = tuple(values.values())
            if [type(v) is list or type(v) is tuple or (isinstance(v, np.ndarray) and v.ndim > 0)
                                                        for v in values].count(True) == len(values):
                values = list(zip(*values))

        elif isinstance(values, tuple):
            pass

        else:
            raise ValueError(f'unable to convert values with type "{type(values)}" to IsopyArray')

        if keys is None:
            #IF there are no keys at this stage raise an error
            raise ValueError('Keys argument not given and keys not found in values')
        else:
            keys = cls._Flavour__keylist(keys, allow_duplicates=False)

        if isinstance(values, tuple):
            vlen = len(values)
        else:
            try:
                vlen = {len(v) for v in values}
            except:
                raise
            if len(vlen) != 1:
                raise ValueError('All rows in values are not the same size')
            vlen = vlen.pop()

        if vlen != len(keys):
            raise ValueError('size of keys does not match size of values')

        if dtype is None:
            new_dtype = [(float64, None) for k in keys]

        elif isinstance(dtype, list) or (isinstance(dtype, np.dtype) and dtype.names is not None):
            if len(dtype) != vlen:
                raise ValueError(
                    'number of dtypes given does not match number of keys')
            else:
                new_dtype = [(dtype[i], ) if not isinstance(dtype[i], tuple) else dtype[i] for i in range(vlen)]

        elif isinstance(dtype, tuple):
            new_dtype = [dtype for i in range(vlen)]

        else:
            new_dtype = [(dtype,) for i in range(vlen)]

        if type(values) is tuple:
            colvalues = list(values)
        else:
            colvalues = list(zip(*values))

        dtype = []
        for i, v in enumerate(colvalues):
            for dt in new_dtype[i]:
                try: dtype.append(np.asarray(v, dtype=dt).dtype)
                except: pass
                else: break
            else:
                raise ValueError(f'Unable to convert values for {keys[i]} to one of the specified dtypes')

        out = np.array(values, dtype = list(zip(keys.strlist(), dtype)))
        if ndim == -1:
            if out.size == 1: ndim = 0
            else: ndim = 1

        if ndim is not None and ndim != out.ndim:
            if ndim == 1:
                out = out.reshape(-1)
            elif ndim == 0 and out.size != 1:
                raise ValueError(f'Cannot convert array with {out.size} rows to 0-dimensions')
            else:
                out = out.reshape(tuple())

        return keys._Flavour__view_ndarray(out) # So that mixed arrays work if all keys have the same flavour

    def __setitem__(self, key, value):
        if isinstance(key, str):
            try:
                key = str(self._Flavour__keystring(key))
            except:
                pass
        elif isinstance(key, (list,tuple)):
            if len(key) > 0 and isinstance(key[0], str):
                try:
                    key = self._Flavour__keylist(key)
                except:
                    pass
            elif not isinstance(key, list):
                key = list(key)

        if isinstance(value, dict) and not isinstance(value, IsopyDict):
            value = isopy.IsopyDict(value, default_value=np.nan)

        if isinstance(key, (str, IsopyKeyList)):
            call_array_function(np.copyto, self, value, keys=key)

        elif isinstance(value, (IsopyArray, IsopyDict)):
            for k in self.keys:
                self[k][key] = value.get(k)
        else:
            try:
                value = np.asarray(value)
            except:
                pass
            else:
                if value.size == 1 and value.ndim == 1:
                    value = value.reshape(tuple())
            super(IsopyNdarray, self).__setitem__(key, value)

    def __getitem__(self, key):
        if isinstance(key, str):
            try:
                key = str(self._Flavour__keystring(key))
            except KeyParseError:
                pass
            else:
                return super(IsopyNdarray, self).__getitem__(key).view(ndarray)

        elif isinstance(key, (list,tuple)):
            if len(key) == 0:
                return np.array([])
            elif isinstance(key[0], str):
                try:
                    key = self._Flavour__keylist(key)
                except:
                    pass
                else:
                    return key._Flavour__view(super(IsopyNdarray, self).__getitem__(key.strlist()))
            else:
                # sequence of row indexes
                if not isinstance(key, list):
                    # Since we cannot index in other dimensions
                    key = list(key)
                return self._Flavour__view(super(IsopyNdarray, self).__getitem__(key))


        return self._Flavour__view(super(IsopyNdarray, self).__getitem__(key))


class MassNdarray(MassFlavour, IsopyNdarray, MassArray):
    pass


class ElementNdarray(ElementFlavour, IsopyNdarray, ElementArray):
    pass


class IsotopeNdarray(IsotopeFlavour, IsopyNdarray, IsotopeArray):
    pass


class RatioNdarray(RatioFlavour, IsopyNdarray, RatioArray):
    pass


class GeneralNdarray(GeneralFlavour, IsopyNdarray, GeneralArray):
    pass


class MixedNdArray(MixedFlavour, IsopyNdarray, MixedArray):
    pass

############
### Void ###
############

class IsopyVoid(IsopyArray, void):
    def __new__(cls, void):
        return void.view((cls, void.dtype))

    def __len__(self):
        raise TypeError('len() of unsized object')

    def __setitem__(self, key, value):
        if isinstance(key, int):
            raise IndexError('0-dimensional arrays cannot be indexed by row')

        if type(key) is slice and key.start is None and key.stop is None and key.step is None:
            key = self.keys
        else:
            try:
                key = self._Flavour__keylist(key)
            except:
                pass

        if isinstance(value, dict) and not isinstance(value, IsopyDict):
            value = isopy.IsopyDict(value, default_value=np.nan)

        if isinstance(key, IsopyKeyList):
            for k in key:
                if k not in self.keys:
                    raise KeyError(f'key {k} not found in array')
            if isinstance(value, (IsopyArray, IsopyDict)):
                for k in self.keys:
                    v = np.asarray(value.get(k))
                    if v.size == 1 and v.ndim == 1:
                        v = v.reshape(tuple())
                    super(IsopyVoid, self).__setitem__(str(k), v)
            else:
                try:
                    value = np.asarray(value)
                except:
                    pass
                else:
                    if value.size == 1 and value.ndim == 1:
                        value = value.reshape(tuple())
                for k in key:
                    super(IsopyVoid, self).__setitem__(str(k), value)
        else:
            super(IsopyVoid, self).__setitem__(key, value)

    def __getitem__(self, key):
        if isinstance(key, str):
            try:
                key = str(self._Flavour__keystring(key))
            except:
                pass

        elif isinstance(key, (list,tuple)):
            if len(key) == 0:
                return np.array([])
            elif isinstance(key[0], str):
                try:
                    key = self._Flavour__keylist(key)
                except:
                    pass
                else:
                    return key._Flavour__view(super(IsopyVoid, self).__getitem__(key.strlist()))
            else:
                key = list(key)

        if isinstance(key, (int, slice)):
            raise IndexError('0-dimensional arrays cannot be indexed by row')

        return self._Flavour__view(super(IsopyVoid, self).__getitem__(key))

    def reshape(self, shape):
        return self._Flavour__view(super(IsopyVoid, self).reshape(shape))


class MassVoid(MassFlavour, IsopyVoid, MassArray):
    pass


class ElementVoid(ElementFlavour, IsopyVoid, ElementArray):
    pass


class IsotopeVoid(IsotopeFlavour, IsopyVoid, IsotopeArray):
    pass


class RatioVoid(RatioFlavour, IsopyVoid, RatioArray):
    pass


class GeneralVoid(GeneralFlavour, IsopyVoid, GeneralArray):
    pass


class MixedVoid(MixedFlavour, IsopyVoid, MixedArray):
    pass

###############################################
### functions for creating isopy data types ###
###############################################
@lru_cache(CACHE_MAXSIZE)
def keystring(string, *, allow_reformatting=True, ignore_charge = False):
    """
    Convert *string* into a key string and return it.

    Will first attempt to convert *string* into a MassKeyString, ElementKeyString, IsotopeKeyString,
    or RatioKeyString. If that fails a GeneralKeyString is returned.
    """
    try:
        return MassKeyString(string, allow_reformatting=allow_reformatting,
                                                        ignore_charge=ignore_charge)
    except KeyParseError:
        pass

    try:
        return ElementKeyString(string, allow_reformatting=allow_reformatting,
                                                        ignore_charge=ignore_charge)
    except KeyParseError:
        pass

    try:
        return IsotopeKeyString(string, allow_reformatting=allow_reformatting,
                                                        ignore_charge=ignore_charge)
    except KeyParseError:
        pass

    try:
        return RatioKeyString(string, allow_reformatting=allow_reformatting,
                                                        ignore_charge=ignore_charge)
    except KeyParseError:
        pass

    try:
        return GeneralKeyString(string, allow_reformatting=allow_reformatting,
                                                        ignore_charge=ignore_charge)
    except KeyParseError:
        pass

    raise KeyParseError(string, IsopyKeyString, f'unable to parse {type(string)} "{string}" into a key string')


@lru_cache(CACHE_MAXSIZE)
def askeystring(key, *, allow_reformatting=True, ignore_charge = False):
    """
    If *string* is a key string it is returned otherwise convert *string* to a key string and
    return it.

    Will first attempt to convert *string* into a MassKeyString, ElementKeyString, IsotopeKeyString,
    or RatioKeyString. If that fails a GeneralKeyString is returned.
    """
    if isinstance(key, IsopyKeyString):
        if ignore_charge:
            return key.basekey
        else:
            return key
    else:
        return keystring(key, allow_reformatting=allow_reformatting, ignore_charge=ignore_charge)


@lru_cache(CACHE_MAXSIZE)
def keylist(*keys, ignore_duplicates=False, allow_duplicates=True, allow_reformatting=True,
            ignore_charge=False, mix_flavours = True):
    """
    Convert *keys* into a key string list and return it. *keys* can be a string or a sequence of
    string.

    Will attempt to convert *keys* into a MassKeyList, ElementKeyList, IsotopeKeyList, or a
    RatioKeyList. If that fails it will return a MixedKeyList unless *mix_flavours* is ``False`` in
    which case a GeneralKeyList is returned.
    """
    try:
        return MassKeyList(*keys, ignore_duplicates=ignore_duplicates,
                           allow_duplicates=allow_duplicates, allow_reformatting=allow_reformatting,
                           ignore_charge=ignore_charge)
    except KeyParseError:
        pass

    try:
        return ElementKeyList(*keys, ignore_duplicates=ignore_duplicates,
                              allow_duplicates=allow_duplicates, allow_reformatting=allow_reformatting,
                              ignore_charge=ignore_charge)
    except KeyParseError:
        pass

    try:
        return IsotopeKeyList(*keys, ignore_duplicates=ignore_duplicates,
                              allow_duplicates=allow_duplicates, allow_reformatting=allow_reformatting,
                              ignore_charge=ignore_charge)
    except KeyParseError:
        pass

    try:
        return RatioKeyList(*keys, ignore_duplicates=ignore_duplicates,
                            allow_duplicates=allow_duplicates, allow_reformatting=allow_reformatting,
                            ignore_charge=ignore_charge)
    except KeyParseError:
        pass

    if mix_flavours:
        try:
            #Is the flavour version so that if all keys are general keys a general list is returned
            return MixedFlavour._Flavour__keylist(*keys, ignore_duplicates=ignore_duplicates,
                                  allow_duplicates=allow_duplicates,
                                  allow_reformatting=allow_reformatting,
                                  ignore_charge=ignore_charge)
        except KeyParseError:
            pass
    else:
        try:
            return GeneralKeyList(*keys, ignore_duplicates=ignore_duplicates,
                                  allow_duplicates=allow_duplicates, allow_reformatting=allow_reformatting,
                                  ignore_charge=ignore_charge)
        except KeyParseError:
            pass

    raise KeyParseError('unable to parse keys into a key string list')


@lru_cache(CACHE_MAXSIZE)
def askeylist(keys, *, ignore_duplicates=False, allow_duplicates=True, allow_reformatting=True,
              ignore_charge = False, mix_flavours = True):
    """
    If *keys* is a key string list return it otherwise convert *keys* to a key string list and
    return it.

    Will attempt to convert *keys* into a MassKeyList, ElementKeyList, IsotopeKeyList, or a
    RatioKeyList. If that fails it will return a MixedKeyList unless *mix_flavours* is ``False`` in
    which case a GeneralKeyList is returned.
    """
    if isinstance(keys, IsopyArray) or isinstance(keys, dict):
        keys = keys.keys()

    if isinstance(keys, IsopyKeyList):
        if ignore_duplicates or not allow_duplicates or ignore_charge:
            return keys._Flavour__keylist(keys, ignore_duplicates=ignore_duplicates, allow_duplicates=allow_duplicates,
                                          ignore_charge=ignore_charge)
        else:
            return keys

    if isinstance(keys, str):
        keys = (keys,)

    keys = tuple(askeystring(key, allow_reformatting=allow_reformatting,
                             ignore_charge=ignore_charge) for key in keys) #Tuple so cache works
    types = {flavour(key) for key in keys}

    if len(types) == 1:
        return types.pop()._Flavour__keylist(keys, ignore_duplicates=ignore_duplicates,
                                             allow_duplicates=allow_duplicates)
    elif mix_flavours:
        return MixedFlavour._Flavour__keylist(keys, ignore_duplicates=ignore_duplicates,
                              allow_duplicates=allow_duplicates,
                              allow_reformatting=allow_reformatting)
    else:
        return GeneralKeyList(keys, ignore_duplicates=ignore_duplicates,
                              allow_duplicates=allow_duplicates, allow_reformatting=allow_reformatting)


def array(values=None, keys=None, *, dtype=None, ndim=None, mix_flavours=True, **columns):
    """
    Convert the input arguments to a isopy array.

    Will attempt to convert the input arguments into a MassArray, ElementArray, IsotopeArray, or a
    RatioArray. If that fails it will return a MixedArray.
    """
    if values is None and len(columns) == 0:
        raise ValueError('No values were given')
    elif values is not None and len(columns) != 0:
        raise ValueError('values and column kwargs cannot be given together')
    elif values is None:
        values = columns

    if isinstance(array, IsopyArray) and keys is None:
        return array._Flavour__array(array, dtype=dtype, ndim=ndim)

    try: return MassArray(values, keys=keys, dtype=dtype, ndim=ndim)
    except KeyParseError: pass

    try: return ElementArray(values, keys=keys, dtype=dtype, ndim=ndim)
    except KeyParseError: pass

    try: return IsotopeArray(values, keys=keys, dtype=dtype, ndim=ndim)
    except KeyParseError: pass

    try: return RatioArray(values, keys=keys, dtype=dtype, ndim=ndim)
    except KeyParseError: pass

    if mix_flavours:
        try: return MixedArray(values, keys=keys, dtype=dtype, ndim=ndim)
        except KeyParseError: pass
    else:
        try: return GeneralArray(values, keys=keys, dtype=dtype, ndim=ndim)
        except KeyParseError: pass

    raise KeyParseError('Unable to convert input to IsopyArray')


def asarray(a, *, ndim = None, mix_flavours = True):
    """
    If *a* is an isopy array return it otherwise convert *a* into an isopy array and return it. If
    *ndim* is given a view of the array with the specified dimensionality is returned.
    """
    if not isinstance(a, IsopyArray):
        a = array(a, mix_flavours=mix_flavours)

    if ndim is not None:
        if ndim < -1 or ndim > 1:
            raise ValueError('the only accepted ndim values are -1, 0, and 1')
        if ndim <= 0 and a.ndim != 0:
            if a.size == 1:
                a = a.reshape(tuple())
            elif ndim != -1:
                raise ValueError('cannot make array with more than one value zero-dimensional')
        elif ndim > 0 and a.ndim == 0:
            a = a.reshape(-1)

    return a


def asanyarray(a, *, dtype = None, ndim = None, mix_flavours = True):
    """
    Return ``isopy.asarray(a)`` if *a* possible otherwise return ``numpy.asanyarray(a)``.

    The data type and number of dimensions of the returned array can be specified by *dtype and
    *ndim*, respectively.
    """
    if a is None: return None

    if isinstance(a, IsopyArray) and dtype is None:
        return asarray(a, ndim=ndim)

    if isinstance(a, dict) or (isinstance(a, ndarray) and a.dtype.names is not None):
        return array(a, dtype=dtype, mix_flavours = mix_flavours, ndim=ndim)

    else:
        if type(dtype) is not tuple:
            dtype = (dtype, )

        for dt in dtype:
            try:
                a = np.asanyarray(a, dtype=dt)
            except Exception as err:
                pass
            else:
                break
        else:
            raise TypeError(f'Unable to convert array to dtype {dtype}')

        if ndim is not None:
            if ndim < -1 or ndim > 2:
                raise ValueError(f'Unable to convert array to number of dimensions specified {ndim}')
            if ndim == a.ndim:
                pass
            elif ndim <= 0:
                if a.ndim == 0:
                    pass
                elif a.size == 1:
                    a = a.reshape(tuple())
                elif ndim != -1:
                    raise ValueError('cannot make array with more than one value zero-dimensional')
            elif ndim == 1 and a.ndim != ndim:
                a = a.reshape(-1)
            elif ndim == 2 and a.ndim != ndim:
                a = a.reshape(-1, a.size)

        return a


def array_from_csv(filename, *, dtype=None, ndim=None, mix_flavours=True, **read_csv_kwargs):
    """
    Returns an array of values from a csv file.
    """
    data = isopy.read_csv(filename, **read_csv_kwargs)
    return isopy.asanyarray(data, dtype=dtype, ndim=ndim, mix_flavours=mix_flavours)

def array_from_xlsx(filename, sheetname, *, dtype=None, ndim=None, mix_flavours=True, **read_xlsx_kwargs):
    """
    Returns an array of values from *sheet_name* in a xlsx file.
    """
    data = isopy.read_xlsx(filename, sheetname, **read_xlsx_kwargs)
    return isopy.asanyarray(data, dtype=dtype, ndim=ndim, mix_flavours=mix_flavours)

def array_from_clipboard(*, dtype=None, ndim=None, mix_flavours=True, **read_clipboard_kwargs):
    """
    Returns an array of values from the clipboard.
    """
    data = isopy.read_clipboard(**read_clipboard_kwargs)
    return isopy.asanyarray(data, dtype=dtype, ndim=ndim, mix_flavours=mix_flavours)

###########################
### Create empty arrays ###
###########################
def zeros(rows, keys=None, *, ndim=None, dtype=None):
    """
    Create an isopy array filled with zeros.

    If *keys* are not given, and cannot be inferred, then a normal numpy array is returned.

    Parameters
    ----------
    rows : int, None
        Number of rows in the returned array. A value of ``-1`` or ``None``
        will return a 0-dimensional array unless overridden by *ndim*.
    keys : Sequence[str], Optional
        Column names for the returned array. Can also be inferred from *dtype* if *dtype* is a
        named ``np.dtype``.
    ndim : {-1, 0, 1}, Optional
        Dimensions of the final array. A value of ``-1`` will return an
        0-dimensional array if size is 1 otherwise a 1-dimensional array is returned.
        An exception is raised if value is ``0`` and *size* is not ``-1`` or ``1``.
    dtype : numpy_dtype, Sequence[numpy_dtype]
        Data type of returned array. A sequence of data types can given to specify
        different datatypes for different columns in the final array. If not given ``np.float64``
        is used for all columns.
    """
    return _new_empty_array(rows, keys, ndim, dtype, np.zeros)


def ones(rows, keys=None, *, ndim=None, dtype=None):
    """
    Create an isopy array filled with ones.

    If *keys* are not given, and cannot be inferred, then a normal numpy array is returned.

    Parameters
    ----------
    rows : int, None
        Number of rows in the returned array. A value of ``-1`` or ``None``
        will return a 0-dimensional array unless overridden by *ndim*.
    keys : Sequence[str], Optional
        Column names for the returned array. Can also be inferred from *dtype* if *dtype* is a
        named ``np.dtype``.
    ndim : {-1, 0, 1}, Optional
        Dimensions of the final array. A value of ``-1`` will return an
        0-dimensional array if size is 1 otherwise a 1-dimensional array is returned.
        An exception is raised if value is ``0`` and *size* is not ``-1`` or ``1``.
    dtype : numpy_dtype, Sequence[numpy_dtype]
        Data type of returned array. A sequence of data types can given to specify
        different datatypes for different columns in the final array. If not given ``np.float64``
        is used for all columns.
    """
    return _new_empty_array(rows, keys, ndim, dtype, np.ones)


def empty(rows, keys=None, *, ndim=None, dtype=None):
    """
    Create an isopy array without initalising entries.

    If *keys* are not given, and cannot be inferred, then a normal numpy array is returned.

    Parameters
    ----------
    rows : int, None
        Number of rows in the returned array. A value of ``-1`` or ``None``
        will return a 0-dimensional array unless overridden by *ndim*.
    keys : Sequence[str], Optional
        Column names for the returned array. Can also be inferred from *dtype* if *dtype* is a
        named ``np.dtype``.
    ndim : {-1, 0, 1}, Optional
        Dimensions of the final array. A value of ``-1`` will return an
        0-dimensional array if size is 1 otherwise a 1-dimensional array is returned.
        An exception is raised if value is ``0`` and *size* is not ``-1`` or ``1``.
    dtype : numpy_dtype, Sequence[numpy_dtype]
        Data type of returned array. A sequence of data types can given to specify
        different datatypes for different columns in the final array. If not given ``np.float64``
        is used for all columns.
    """
    return _new_empty_array(rows, keys, ndim, dtype, np.empty)


def full(rows, fill_value, keys=None, *, ndim = None, dtype=None):
    """
    Create an isopy array filled with *fill_value*.

    If *keys* are not given, and cannot be inferred, then a normal numpy array is returned.

    Parameters
    ----------
    rows : int, None
        Number of rows in the returned array. A value of ``-1`` or ``None``
        will return a 0-dimensional array unless overridden by *ndim*.
    fill_value
        the value the array will be filled with.
    keys : Sequence[str], Optional
        Column names for the returned array. Can also be inferred from *dtype* if *dtype* is a
        named ``np.dtype``.
    ndim : {-1, 0, 1}, Optional
        Dimensions of the final array. A value of ``-1`` will return an
        0-dimensional array if size is 1 otherwise a 1-dimensional array is returned.
        An exception is raised if value is ``0`` and *size* is not ``-1`` or ``1``.
    dtype : numpy_dtype, Sequence[numpy_dtype]
        Data type of returned array. A sequence of data types can given to specify
        different datatypes for different columns in the final array. If not given ``np.float64``
        is used for all columns.
    """
    out = _new_empty_array(rows, keys, ndim, dtype, np.empty)
    np.copyto(out, fill_value)
    return out


def random(rows, random_args = None, keys=None, *, distribution='normal', seed=None, ndim = None, dtype=None):
    """
    Creates an an isopy array filled with random numbers.

    If *keys* are not given, and cannot be inferred, then a normal numpy array is returned.

    Parameters
    ----------
    rows : int, None
        Number of rows in the returned array. A value of ``-1`` or ``None``
        will return a 0-dimensional array unless overridden by *ndim*.
    random_args
        Arguments to be passed to the random generator when creating the random numbers for each
        column. Can be a single value or a tuple containing several values. If you want different
        arguments for different columns pass a list of values/tuples, one for each column.
    keys : Sequence[str], Optional
        Column names for the returned array. Can also be inferred from *dtype* if *dtype* is a
        named ``np.dtype``.
    distribution
        The name of the distribution to be used when calculating the
        random numbers. See `here <https://numpy.org/doc/stable/reference/random/generator.html?highlight=default_rng#distributions>`_ for
        a list of avaliable distributions.
    seed
        The seed passed to the random generator.
    fill_value
        the value the array will be filled with.

    ndim : {-1, 0, 1}, Optional
        Dimensions of the final array. A value of ``-1`` will return an
        0-dimensional array if size is 1 otherwise a 1-dimensional array is returned.
        An exception is raised if value is ``0`` and *size* is not ``-1`` or ``1``.
    dtype : numpy_dtype, Sequence[numpy_dtype]
        Data type of returned array. A sequence of data types can given to specify
        different datatypes for different columns in the final array. If not given ``np.float64``
        is used for all columns.

    Examples
    --------
    >>> array = isopy.random(100, keys=('ru', 'pd', 'cd'))
    >>> array
    (row) , Ru       , Pd       , Cd
    0     , -0.30167 , -0.82244 , -0.91288
    1     , -0.51438 , -0.87501 , -0.10230
    2     , -0.72600 , -0.43822 , 1.17180
    3     , 0.55762  , -0.52775 , -1.21364
    4     , -0.64446 , 0.42803  , -0.42528
    ...   , ...      , ...      , ...
    95    , -0.51426 , 0.13598  , 1.82878
    96    , 0.45020  , -0.70594 , -1.04865
    97    , 1.79499  , 0.24688  , -0.18669
    98    , 0.57716  , 0.57589  , -0.66426
    99    , -0.25646 , -1.20771 , -0.01936
    >>> np.mean(array)
    (row) , Ru      , Pd       , Cd
    None  , 0.01832 , -0.07294 , -0.00178
    >>> isopy.sd(array)
    (row) , Ru      , Pd      , Cd
    None  , 0.89535 , 1.05086 , 0.91490

    >>> array = isopy.random(100, [(0, 1), (1, 0.1), (-1, 10)], keys=('ru', 'pd', 'cd'))
    >>> array
    (row) , Ru       , Pd      , Cd
    0     , -0.99121 , 1.03848 , -10.71260
    1     , 0.93820  , 1.12808 , 33.88074
    2     , -0.22853 , 1.06643 , 2.65216
    3     , -0.05776 , 1.03931 , -7.55531
    4     , -0.58707 , 1.03019 , 0.06148
    ...   , ...      , ...     , ...
    95    , 0.51169  , 1.10513 , 17.36456
    96    , 0.21135  , 1.04240 , -8.05624
    97    , -0.79133 , 1.08202 , -13.74861
    98    , 1.07542  , 0.86911 , -5.70063
    99    , 1.20108  , 0.78890 , -12.57918
    >>> np.mean(array)
    (row) , Ru      , Pd      , Cd
    None  , 0.00008 , 1.00373 , -0.10169
    >>> isopy.sd(array)
    (row) , Ru      , Pd      , Cd
    None  , 0.86765 , 0.09708 , 10.36754
    """
    array = empty(rows, keys=keys, ndim=ndim, dtype=dtype)

    rng = np.random.default_rng(seed=seed)
    dist = getattr(rng, distribution)

    if isinstance(array, IsopyArray):
        if random_args is None:
            random_args = [tuple() for k in array.keys]
        elif type(random_args) is tuple:
            random_args = [random_args for k in array.keys]
        elif type(random_args) is list:
            if len(random_args) != array.ncols:
                raise ValueError('size of "random_args" does not match the number of keys"')
            random_args = [d if type(d) is tuple else (d,) for d in random_args]
        else:
            random_args = [(random_args,) for k in array.keys]

        for key, args in zip(array.keys, random_args):
            array[key] = dist(*args, size=array.shape)
    else:
        if random_args is None:
            random_args = tuple()
        elif type(random_args) is not tuple:
            random_args = (random_args, )
        return dist(*random_args, size=array.shape)

    return array


def _new_empty_array(rows, keys, ndim, dtype, func):
    if isinstance(rows, IsopyArray):
        if keys is None: keys = rows.keys
        if dtype is None: dtype = rows.dtype
        rows = rows.nrows

    if isinstance(keys, IsopyArray):
        if dtype is None: dtype = keys.dtype
        keys = keys.keys

    if isinstance(dtype, IsopyArray):
        dtype = dtype.dtype

    if dtype is None:
        dtype = float64

    if rows is None: rows = -1
    elif rows == tuple(): rows = -1
    elif rows == -1: pass
    elif rows < 1:
        raise ValueError('parameter "rows" must be -1 or a positive integer')

    if keys is not None:
        keys = askeylist(keys, allow_duplicates=False)

    if ndim is None:
        if rows == -1: shape = None
        else: shape = rows
    elif ndim == -1:
        if abs(rows) == 1: shape = None
        else: shape = rows
    elif ndim == 1:
        shape = abs(rows)
    elif ndim == 0:
        if abs(rows)  == 1:
            shape = None
        else:
            raise TypeError('cannot create an zero-dimensional array with more than one row')
    elif ndim < -1 or ndim > 1:
        raise ValueError('accepted values for "ndim" is -1, 0,  or 1')

    if shape is None: shape = ()

    if isinstance(dtype, np.dtype) and dtype.names is not None:
        if keys is None:
            keys = keylist(dtype.names, allow_duplicates=False)
        elif len(keys) != len(dtype):
            raise ValueError('size of dtype does not match size of keys')
        dtype = [dtype[i] for i in range(len(dtype))]

    elif hasattr(dtype, '__iter__'):
        if keys is None:
            raise ValueError('dtype is an iterable but no keys were given')
        elif (len(keys) != len(dtype)):
            raise ValueError('size of dtype does not match size of keys')

    elif keys is not None:
        dtype = [dtype for i in range(len(keys))]

    if keys is not None:
        dtype = list(zip(keys.strlist(), dtype))
        return keys._Flavour__view_ndarray(func(shape, dtype=dtype))
    else:
        return func(shape, dtype=dtype)


######################################################
### reimplementationz of exisiting numpy functions ###
######################################################
def concatenate(*arrays, axis=0, default_value=nan):
    """
    Join arrays together along specified axis.

    Same function as ``np.concatenate`` with the additional option of specifying the default value for columns not
    present in all arrays.

    Normal numpy broadcasting rules apply so when concatenating columns the shape of the arrays must be compatible.
    When array with a size of 1 is concatenated to an array of a different size it will be copied into every row
    of the new shape.

    Parameters
    ----------
    arrays
        Arrays to be joined together.
    axis
        If 0 then the rows of each array in *arrays* will be concatenated. If 1 then the columns of each array will
        be concatenated. If *axis* is 1 an error will be raised if a column key occurs in more than one array.
    default_value : float, optional
        The default value for any columns that are missing when concatenating rows. Default value is ``np.nan``.

    Returns
    -------
    IsopyArray
        Array containing all the row data and all the columns keys found in *arrays*.
    """
    if len(arrays) == 1 and isinstance(arrays[0], (list, tuple)):
        arrays = arrays[0]
    arrays = [asanyarray(a) for a in arrays]

    if True not in (tlist:=[isinstance(a, IsopyArray) for a in arrays if a is not None]):
        return np.concatenate(arrays)
    if False in tlist:
        raise ValueError('Cannot concatenate Isopy arrays with normal arrays.')

    if axis == 0 or axis is None: #extend rows
        keys = keylist(*(a.dtype.names for a in arrays if a is not None), ignore_duplicates=True)
        arrays = [a.reshape(1) if (a is not None and a.ndim == 0) else a for a in arrays]
        if None in arrays:
            if len(size:={a.size for a in arrays if a is not None}) == 1:
                default_values = np.full(size.pop(), default_value)
                arrays = [_getter(default_values) if a is None else a for a in arrays]
            else:
                raise ValueError('Cannot determine size for None based on other input')

        result = [np.concatenate([a.get(key, default_value) for a in arrays]) for key in keys]
        dtype = [(key, result[i].dtype) for i, key in enumerate(keys.strlist())]
        return keys._Flavour__view_ndarray(np.fromiter(zip(*result), dtype=dtype))

    elif axis == 1: #extend columns
        arrays = [a for a in arrays if a is not None] #Ignore None values
        size = {a.size for a in arrays}
        ndim = {a.ndim for a in arrays}
        if not (len(size) == 1 or (1 in size and len(size) == 2)):
            raise ValueError('all arrays must have the same size concatenating in axis 1')
        if len(ndim) != 1:
            arrays = [array.reshape(1) if array.ndim == 0 else array for array in arrays]

        if len(arrays) == 1:
            return arrays[0].copy()

        keys = keylist(*(a.dtype.names for a in arrays), allow_duplicates=False)

        result = {}
        for a in arrays:
            for key in a.keys():
                result[key] = a.get(key, default_value)


        return isopy.full(max(size) * max(ndim) or -1, result, keys, dtype=[v.dtype for v in result.values()])

    else:
        raise np.AxisError(axis, 1, 'isopy.concatenate only accepts axis values of 0 or 1')


ALLOWED_NUMPY_FUNCTIONS = {np.concatenate: concatenate}
afnp_elementwise = [np.sin, np.cos, np.tan, np.arcsin, np.arccos, np.arctan, np.degrees, np.isnan,
                    np.radians, np.deg2rad, np.rad2deg, np.sinh, np.cosh, np.tanh, np.arcsinh,
                    np.arccosh, np.arctanh,
                    np.rint, np.floor, np.ceil, np.trunc, np.exp, np.expm1, np.exp2, np.log,
                    np.log10, np.log2,
                    np.log1p, np.reciprocal, np.positive, np.negative, np.sqrt, np.cbrt, np.square,
                    np.fabs, np.sign,
                    np.absolute, np.abs]
afnp_cumulative = [np.cumprod, np.cumsum, np.nancumprod, np.nancumsum]
afnp_reducing = [np.prod, np.sum, np.nanprod, np.nansum, np.cumprod, np.cumsum, np.nancumprod, np.nancumsum,
                 np.amin, np.amax, np.nanmin, np.nanmax, np.ptp, np.median, np.average, np.mean, np.average,
                 np.std, np.var, np.nanmedian, np.nanmean, np.nanstd, np.nanvar, np.max, np.nanmax, np.min, np.nanmin,
                 np.all, np.any]
afnp_special = [np.copyto, np.average]
afnp_dual = [np.add, np.subtract, np.divide, np.multiply, np.power]

for func in afnp_elementwise: ALLOWED_NUMPY_FUNCTIONS[func] = True
for func in afnp_cumulative: ALLOWED_NUMPY_FUNCTIONS[func] = True
for func in afnp_reducing: ALLOWED_NUMPY_FUNCTIONS[func] = True
for func in afnp_special: ALLOWED_NUMPY_FUNCTIONS[func] = True
for func in afnp_dual: ALLOWED_NUMPY_FUNCTIONS[func] = True

def allowed_numpy_functions(format = 'name', delimiter = ', '):
    """
    Return a string containing the names of all the numpy functions supported by isopy arrays.

     With *format* you can specify the format of the string for each function. Avaliable keywords
     are ``{name}`` and ``{link}`` for the name and a link the numpy documentation web page for a
     function. Avaliable presets are ``"name"``, ``"link"``, ``"rst"`` and ``"markdown"`` for just the name,
     just the link, reStructured text hyper referenced link or a markdown hyper referenced link.

     You can specify the delimiter used to seperate items in the list using the *delimiter* argument.
    If *delimiter* is ``None`` a python list is returned.
    """
    if format == 'name': format = '{name}'
    if format == 'link': format = '{link}'
    if format == 'rst': format = '`{name} <{link}>`_'
    if format == 'markdown': format = '[{name}]({link})'

    strings = []
    for func in isopy.core.ALLOWED_NUMPY_FUNCTIONS:
        name = func.__name__
        link = f'https://numpy.org/doc/stable/reference/generated/numpy.{name}.html'
        strings.append(format.format(name = name, link=link))
    if delimiter is None:
        return strings
    else:
        return delimiter.join(strings)

###############################################
### numpy function overrides via            ###
###__array_ufunc__ and __array_function__   ###
###############################################
@functools.lru_cache(maxsize=CACHE_MAXSIZE)
def _function_signature(func):
    #returns the signature and number of input arguments for a function
    #An input argument is given as any argument that does not have a default value
    try:
        parameters = inspect.signature(func).parameters
    except:
        #No signature avaliable. Should be a rare occurance but applies to for example
        # concatenate (which wont call this anyway)
        #This return assumes all args are inputs
        return None, None
    else:
        return tuple(parameters.keys()), \
            len([p for p in parameters.values() if (p.default is p.empty)])


class _getter:
    """
    Returns the same array for all calls to get. Used for array function
     inputs that are not isopy arrays or dicts.
    """
    __slots__ = ['value']
    def __init__(self, value):
        self.value = value

    def get(self, *_):
        return self.value

# This is for functions that dont have a return value
def call_array_function(func, *inputs, axis=0, default_value= nan, keys=None, **kwargs):
    """
    Used to call numpy ufuncs/functions on isopy arrays.

    This function produces the expected result for the majority, but not all, numpy functions.
    With a few exceptions all calls to numpy functions on isopy arrays will be sent to this function
     via the __array_ufunc__ and __array_function__ interface.

    If axis == 0 then the function is called on each column present in the isopy arrays given as
    input. Isopy arrays given as kwargs are not used taken into consideration when determining the
    returned array. Analogous to ``isopy.array({key: func(array.get(key, default_value), **kwargs)}
    for key in array.keys())`` for a single isopy array input.

    If all isopy arrays in inputs are of the same flavour then the returned array will have the
    same flavour. If the inputs are of different flavours then the returned array will be a
    GeneralArray

    If axis == 1 or None the function is called once with all isopy arrays are converted to lists.
    Only isopy arrays given as input are used to determine the keys used when converting isopy
    arrays to lists. Analogous to ``func([array.get(key, default_value) for key
    in array.keys()], axis, **kwargs)`` for a single isopy array input.


    Function must accept an axis keyword argument if axis == 1 or None. No axis keyword is passed if
    axis == 1.

    If inputs contain no isopy arrays then the function is called on the inputs and kwargs as is.
    Analogous to ``func(*inputs, **kwargs)``.


    Parameters
    ----------
    func : callable
        Function that will be called on input.
    *inputs : IsopyArray, dict, array_like
        Isopy arrays given here will determine the flavour and the keys of the returned array.
        All inputs that are not isopy arrays or dicts will be passed as it to each function call.
    axis : {0, 1, None}, default = 0
        The axis on which the function should be called. If 0 the function is called on each column
        present in the input. If 1 or None the function is called once with all isopy arrays in
        inputs and kwargs turned into lists. If axis is 1 or None no axis argument is passed to the
        function.
    default_value : int, float, default = np.nan
        Default value used for a column if it is not present in an array. If a scalar is given
        it will be used to represent the value for every row in the column.
    **kwargs
        Keyword arguments to be supplied to the function. Keyword arguments that are isopy arrays or
        dicts will behave the same as those in inputs but will not be used to .

    Returns
    -------
    IsopyArray, np.ndarray

    Notes
    -----
    An attempt will be made to convert any structured numpy array in inputs to a isopy array. This
    is *not* attempted for any structured numpy arrays given in kwargs.

    See Also
    --------
    concancate, append
    """

    # Try to sort input from optional arguments.
    fargs, nin = _function_signature(func)
    if fargs and len(inputs) != nin:
        kwargs.update(zip(fargs, inputs))
        try:
            inputs = tuple(kwargs.pop(fargs[i]) for i in range(nin))
        except:
            # I dont think this would ever happen as it would be caught by the
            # array function dispatcher.
            raise ValueError(
                f'"{func.__name__}" expects {nin} input arguments but only got {len(inputs)}')

    if fargs and 'axis' in fargs and axis == 0:
        kwargs['axis'] = None

    if type(default_value) is not tuple:
        default_value = tuple(default_value for i in range(len(inputs)))
    elif len(default_value) != len(inputs):
        raise ValueError('size of default value not the same as the number of inputs')

    new_inputs = []
    new_default_values = []
    new_keys = []
    for i, arg in enumerate(inputs):
        if isinstance(arg, IsopyArray):
            new_inputs.append(arg)
            new_default_values.append(default_value[i])
            new_keys.append(arg.keys())
        elif isinstance(arg, ndarray) and len(arg.dtype) > 1:
            try:
                new_inputs.append(asarray(arg))
            except:
                new_inputs.append(arg)
            else:
                new_keys.append(new_inputs[-1].keys())
            new_default_values.append(default_value[i])
        elif isinstance(arg, IsopyDict):
            new_inputs.append(arg)
            new_default_values.append(arg.default_value)
        elif isinstance(arg, dict):
            new_inputs.append(IsopyDict(arg))
            new_default_values.append(default_value[i])
        else:
            new_inputs.append(_getter(arg))
            new_default_values.append(default_value[i])

    if keys is None:
        if len(new_keys) == 0:
            return func(*inputs, **kwargs)
        else:
            new_keys = new_keys[0].__or__(*new_keys[1:])
    else:
        new_keys = isopy.askeylist(keys)

    if axis == 0:
        out = kwargs.get('out', None)
        keykwargs = {kwk: kwargs.pop(kwk) for kwk, kwv in tuple(kwargs.items())
                                if isinstance(kwv, IsopyArray)}

        result = [func(*(input.get(key, default_value) for input, default_value in zip(new_inputs, new_default_values)), **kwargs,
                       **{k: v.get(key) for k, v in keykwargs.items()}) for key in new_keys]

        #There is no return from the function so dont return an array
        if False not in [r is None for r in result]:
            return None

        if out is None:
            dtype = [(str(key), getattr(result[i], 'dtype', float64)) for i, key in enumerate(new_keys)]
            if getattr(result[0], 'ndim', 0) == 0:
                return new_keys._Flavour__view_ndarray(np.array(tuple(result), dtype=dtype))
            else:
                return new_keys._Flavour__view_ndarray(np.fromiter(zip(*result), dtype=dtype))
        else:
            return out

    else:
        for kwk, kwv in kwargs.items():
            if isinstance(kwv, (IsopyArray, dict)):
                kwargs['kwk'] = np.transpose([kwv.get(key, default_value) for key in new_keys])

        new_inputs = [np.transpose([input.get(key, default_value) for key in new_keys]) if
                      not isinstance(input, _getter) else input.get() for input, default_value in zip(new_inputs, new_default_values)]

        if len(nd :={inp.ndim for inp in new_inputs}) == 1 and nd.pop() == 1:
            #So that 0-dimensional arrays get do not raise axis out of bounds error
            axis = 0

        return func(*new_inputs, axis = axis, **kwargs)