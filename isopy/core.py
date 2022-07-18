import isopy
import numpy as np
import pyperclip as pyperclip
import inspect as inspect
import functools
import itertools
import hashlib
import warnings
import collections.abc as abc
import io
import operator

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


class NotGivenType:
    def __repr__(self):
        return 'N/A'

    def __bool__(self):
        return False

NotGiven = NotGivenType()

ARRAY_REPR = dict(include_row=True, include_dtype=False, nrows=10, f='{:.5g}')

__all__ = ['iskeystring', 'iskeylist', 'isarray', 'isdict', 'isrefval',
           'asflavour',
           'keystring', 'askeystring',
           'keylist', 'askeylist',
           'array', 'asarray', 'asanyarray',
           'asdict', 'asrefval', 'IsopyDict', 'RefValDict', 'ScalarDict',
           'ones', 'zeros', 'empty', 'full', 'random']

CACHE_MAXSIZE = 128
CACHES_ENABLED = True

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
                    if ignore_unhashable is False or not startswith(str(err), 'unhashable'):
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
            preset = partial_func(func, f'{func.__name__}.{n}', **kwargs)
            #preset = functools.partial(func, **kwargs)
            #preset.__name__ = f'{func.__name__}.{n}'
            #preset.__module__ = func.__module__
            setattr(func, n, preset)
            func._presets.insert(0, (preset, kwargs))
        return func
    return decorator

def partial_func(func, name = None, doc = None, /, **func_kwargs):
    new_func = functools.partial(func, **func_kwargs)
    new_func.__name__ = name or func.__name__
    new_func.__doc__ = doc or func.__doc__
    return new_func

def startswith(string, prefix):
    return string[:len(prefix)] == prefix

def extract_kwargs(kwargs, prefix, keep_prefix=False):
    new_kwargs = {}
    for kwarg in list(kwargs.keys()):
        prefix_ = f'{prefix}_'
        if startswith(kwarg, prefix_):
            if keep_prefix:
                new_kwargs[kwarg] = kwargs.pop(kwarg)
            else:
                new_kwargs[kwarg[len(prefix_):]] = kwargs.pop(kwarg)

    return new_kwargs

#TODO better error messages if unknown arguments or keywords are given
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

def deprecrated_function(message, stacklevel = 1):
    def wrap_func(func):
        def func_wrapper(*args, **kwargs):
            warnings.warn(message, stacklevel=stacklevel)
            return func(*args,**kwargs)
        return func_wrapper
    return wrap_func

def hashstr(string):
    return hashlib.md5(string.encode('UTF-8')).hexdigest()

####################
### type testers ###
####################

def iskeystring(item, *, flavour = None, flavour_in = None) -> bool:
    """
    Returns ``True`` if *item* is a key string otherwise returns ``False``.
    """
    if flavour is not None:
        return isinstance(item, IsopyKeyString) and item.flavour == asflavour(flavour)
    elif flavour_in is not None:
        return isinstance(item, IsopyKeyString) and item.flavour in asflavour(flavour_in)
    else:
        return isinstance(item, IsopyKeyString)

def iskeylist(item, *, flavour = None, flavour_in = None) -> bool:
    """
    Returns ``True`` if *item* is a key string list otherwise returns ``False``.
    """
    if flavour is not None:
        return isinstance(item, IsopyKeyList) and item.flavour == asflavour(flavour)
    elif flavour_in is not None:
        return isinstance(item, IsopyKeyList) and item.flavour in asflavour(flavour_in)
    else:
        return isinstance(item, IsopyKeyList)

def isarray(item, *, flavour = None, flavour_in = None) -> bool:
    """
    Returns ``True`` if *item* is an isopy array otherwise returns ``False``.
    """
    if flavour is not None:
        return isinstance(item, IsopyArray) and item.flavour == asflavour(flavour)
    elif flavour_in is not None:
        return isinstance(item, IsopyArray) and item.flavour in asflavour(flavour_in)
    else:
        return isinstance(item, IsopyArray)

def isdict(item):
    """
    Returns ``True`` if *item* is an isopy dict or refval dict otherwise returns ``False``.
    """
    return isinstance(item, IsopyDict)

def isrefval(item):
    """
    Returns ``True`` if *item* is a refval dict otherwise returns ``False``.
    """
    return type(item) is RefValDict

################
### Flavours ###
################
class IsopyKeyFlavour:
    def __repr__(self):
        return self.__string__

    def __init__(self):
        self.__string__ = f'{self.__flavour_name__}'

    def __hash__(self):
        return hash((self.__class__, self.__string__))

    def __eq__(self, other):
        try:
            other_flavours = ListFlavour(other).__string__
        except:
            return False

        return self.__string__ == other_flavours or self.__flavour_name__ == str(other).lower()

    def __contains__(self, other):
        try:
            other = ListFlavour(other)
        except:
            return False

        for flavour in other._flavours:
            if not self._contains_(flavour):
                return False
        else:
            return True

    def _sortkey_(self):
        return str(self.__flavour_id__)

    def _contains_(self, item):
        return type(item) is self.__class__

    @classmethod
    def _new_(cls, subflavours):
        if subflavours is None:
            return cls()
        else:
            raise ValueError(f'{cls.__name__} does not contain subflavour(s)')


class MassFlavour(IsopyKeyFlavour):
    __flavour_name__ = 'mass'
    __flavour_id__ = 1

    def _keystring_(self, string, **kwargs):
        return MassKeyString(string, **kwargs)


class ElementFlavour(IsopyKeyFlavour):
    __flavour_name__ = 'element'
    __flavour_id__ = 2

    def _keystring_(self, string, **kwargs):
        return ElementKeyString(string, **kwargs)


class IsotopeFlavour(IsopyKeyFlavour):
    __flavour_name__ = 'isotope'
    __flavour_id__ = 3

    def _keystring_(self, string, **kwargs):
        return IsotopeKeyString(string, **kwargs)


class MoleculeFlavour(IsopyKeyFlavour):
    __flavour_name__ = 'molecule'
    __flavour_id__ = 4

    def __init__(self, component_flavour = 'any'):
        if component_flavour == 'any':
            self.component_flavour = 'any'
        else:
            self.component_flavour = asflavour(component_flavour)
        self.__string__ = f'{self.__flavour_name__}[{self.component_flavour}]'

    def _sortkey_(self):
        if type(self.component_flavour) is ListFlavour:
            return f'{self.__flavour_id__}{self.component_flavour._sortkey_()}'
        else:
            return super(MoleculeFlavour, self)._sortkey_()

    def _contains_(self, item):
        if type(item) is not self.__class__:
            return False

        if type(self.component_flavour) is ListFlavour and type(item.component_flavour) is ListFlavour:
            return item.component_flavour in self.component_flavour
        else:
            #Component flavour is any. So always returns  true
            return True

    @classmethod
    def _new_(cls, subflavours):
        if subflavours is None:
            return cls()
        elif len(subflavours) == 1:
            return cls(subflavours[0])
        else:
            raise ValueError(f'RatioFlavour takes 1 subflavours. Got {len(subflavours)}')

    def _keystring_(self, string, **kwargs):
        return MoleculeKeyString(string, component_flavour = self.component_flavour, **kwargs)


class RatioFlavour(IsopyKeyFlavour):
    __flavour_name__ = 'ratio'
    __flavour_id__ = 5

    def __init__(self, numerator_flavour = 'any', denominator_flavour = 'any'):
        if numerator_flavour == 'any' and denominator_flavour == 'any':
            self.numerator_flavour = 'any'
            self.denominator_flavour = 'any'
        else:
            self.numerator_flavour = asflavour(numerator_flavour)
            self.denominator_flavour = asflavour(denominator_flavour)

        self.__string__ = f'{self.__flavour_name__}[{self.numerator_flavour}, {self.denominator_flavour}]'


    @classmethod
    def _new_(cls, subflavours):
        if subflavours is None:
            return cls()
        elif len(subflavours) == 1:
            return cls(subflavours[0], subflavours[0])
        elif len(subflavours) == 2:
            return cls(subflavours[0], subflavours[1])
        else:
            raise ValueError(f'RatioFlavour takes 1 or 2 subflavours. Got {len(subflavours)}')

    def _sortkey_(self):
        if type(self.numerator_flavour) is ListFlavour:
            return f'{self.__flavour_id__}{self.numerator_flavour._sortkey_()}{self.denominator_flavour._sortkey_()}'
        else:
            return super(RatioFlavour, self)._sortkey_()

    def _contains_(self, item):
        if type(item) is not self.__class__:
            return False

        if type(self.numerator_flavour) is ListFlavour and type(item.numerator_flavour) is ListFlavour:
            return (item.numerator_flavour in self.numerator_flavour and
                    item.denominator_flavour in self.denominator_flavour)
        else:
            return True

    def _keystring_(self, string, **kwargs):
        return RatioKeyString(string,
                              numerator_flavour = self.numerator_flavour,
                              denominator_flavour = self.denominator_flavour,
                              **kwargs)


class GeneralFlavour(IsopyKeyFlavour):
    __flavour_name__ = 'general'
    __flavour_id__ = 6

    def _keystring_(self, string, **kwargs):
        return GeneralKeyString(string, **kwargs)


class ListFlavour:
    def __repr__(self):
        if self.__string__ == ANY_FLAVOUR.__string__:
            return 'any'
        else:
            return self.__string__

    def _sortkey_(self):
        return ''.join([f._sortkey_() for f in self._flavours])

    def __hash__(self):
        return hash((self.__class__, self.__string__))

    def __new__(cls, flavours):
        if type(flavours) is cls:
            return flavours
        elif flavours is None or flavours == 'any':
            return ANY_FLAVOUR

        def parse_flavour(flavour):
            if isinstance(flavour, IsopyKeyFlavour):
                return [flavour]
            elif type(flavour) is str:
                # If there is more than one here it means they were seperated by a ,. Which
                sout = cls._parse_string_(flavour.lower())
                if len(sout) > 1:
                    raise ValueError('flavours must be seperated by "|" not ","')
                else:
                    return sout[0]
            else:
                raise TypeError('flavour must be a string or flavour object or a tuple of thereof')

        if isinstance(flavours, (list, tuple)):
            out = []
            for flavour in flavours:
                out.extend(parse_flavour(flavour))
        else:
            out = parse_flavour(flavours)

        if len(out) == 0:
            out = ANY_FLAVOUR

        obj = super(ListFlavour, cls).__new__(cls)
        obj._flavours = tuple(sorted(set(out), key=lambda f: f._sortkey_()))
        obj.__string__ = '|'.join([str(f) for f in obj._flavours])

        return obj

    @classmethod
    def _parse_string_(cls, string):
        out = [[]]
        def new_flavour(flavour, subflavours):
            if flavour == '':
                pass
            elif flavour == 'any':
                return out[-1].extend(ANY_FLAVOUR._flavours)
            else:
                try:
                    out[-1].append(FLAVOURS[flavour]._new_(subflavours))
                except KeyError:
                    raise ValueError(f'flavour "{flavour}" not recognised')
            return ''

        flavour = ''
        i = 0
        while i < len(string):
            char = string[i]

            if char == '|':
                flavour = new_flavour(flavour, None)

            elif char == '[':
                open_brackets = 1
                for j, jc in enumerate(string[i + 1:]):
                    if jc == '[':
                        open_brackets += 1
                    elif jc == ']':
                        open_brackets -= 1

                    if open_brackets == 0:
                        j = i + j + 1
                        break
                else:
                    raise KeyValueError(cls, string, f'Unmatched "[" at index {i}')

                subflavours = cls._parse_string_(string[i + 1: j])
                flavour = new_flavour(flavour, subflavours)
                i = j

            elif char == ',':
                flavour = new_flavour(flavour, None)
                out.append( [] )

            elif char != ' ':
                flavour += char

            i += 1

        new_flavour(flavour, None)
        return out

    @classmethod
    def _string_flavour_(self, flavour, subflavours = None):
        try:
            return FLAVOURS[flavour]._new_(subflavours)
        except KeyError:
            raise ValueError(f'flavour "{flavour}" not recognised')

    def __add__(self, other):
        other = self.__class__(other)
        return self.__class__(self._flavours + other._flavours)

    def __radd__(self, other):
        # Order doesnt matter since its sorted
        return self.__add__(other)

    def __eq__(self, other):
        try:
            other_flavours = ListFlavour(other).__string__
        except:
            return False

        return (self.__string__ == other_flavours or
                False not in [f.__flavour_name__ == str(other).lower() for f in self._flavours])

    def __contains__(self, other):
        try:
            other = self.__class__(other)
        except:
            return False

        for flavour in other._flavours:
            if True not in [f._contains_(flavour) for f in self._flavours]:
                return False
        else:
            return True

    def __len__(self):
        return len(self._flavours)

    def __iter__(self):
        return self._flavours.__iter__()
    
    def _keystring_(self, string, **kwargs):
        return askeystring(string, flavour = self, **kwargs)


@lru_cache(CACHE_MAXSIZE)
def asflavour(flavour):
    return ListFlavour(flavour)

FLAVOURS = {f.__flavour_name__: f for f in [MassFlavour, ElementFlavour,
                                            IsotopeFlavour, MoleculeFlavour,
                                            RatioFlavour, GeneralFlavour]}

ANY_FLAVOUR = ListFlavour(tuple(f() for f in FLAVOURS.values()))

###################
### Key strings ###
###################
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
            return f'{message}. {self.additional_information}'
        else:
            return message


class KeyTypeError(KeyParseError, TypeError):
    def __init__(self, cls, obj):
        self.obj = obj
        self.cls = cls

    def __str__(self):
        return f'{get_classname(self.cls)}: cannot convert {type(self.obj)} into \'str\''


class IsopyKeyString(str):
    def __repr__(self):
        return f"{self.__class__.__name__}('{self}')"

    def _repr_latex_(self):
        return fr'$${self.str("math")}$$'

    def __new__(cls, string, **kwargs):
        obj = str.__new__(cls, string)

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
                other = askeystring(other, flavour = self.flavour)
            except:
                return False

        return hash(self) == hash(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __truediv__(self, other):
        if isinstance(other, (list, tuple)):
            return askeylist(other).__rtruediv__(self)
        else:
            return RatioKeyString((self, other))

    def __rtruediv__(self, other):
        if isinstance(other, (list, tuple)):
            return askeylist(other).__truediv__(self)
        else:
            return askeystring(other).__truediv__(self)
        
    def _view_array_(self, a):
        return a.view(ndarray)

    def _str_(self):
        return str(self)

    def str(self, format = None):
        """
        Return a ``str`` object of the key string.

        the optional *format* can either be a string matching one of the format options or a
        string which can be formatted using the format options as key words.

        Format options are:
        * ``"m"`` and ``"key"`` - Same as ``str(keystring)``
        * ``"math"`` -  Key string formatted for latex math mode.
        * ""`latex`"" - Same as above but including $ $ math deliminators.
        """
        if format is None: return str(self)
        options = self._str_options_()

        if format in options:
            return options[format]
        else:
            return format.format(**options)

    def _flatten_(self):
        return (self,)

    def _filter_(self, **filters):
        for f, v in filters.items():
            try:
                attr, comparison = f.rsplit('_', 1)
            except ValueError:
                return False

            if attr == 'key':
                attr = self
            else:
                attr = getattr(self, attr, NotImplemented)

            if attr is NotImplemented or comparison not in ('eq', 'neq', 'lt', 'gt', 'le', 'ge'):
                return False

            try:
                if comparison == 'eq' and attr not in v:
                    return False
                elif comparison == 'neq' and attr in v:
                    return False
                elif comparison == 'lt' and not attr < v:
                    return False
                elif comparison == 'gt' and not attr > v:
                    return False
                elif comparison == 'le' and not attr <= v:
                    return False
                elif comparison == 'ge' and not attr >= v:
                    return False
            except:
                return False
        return True


class MassKeyString(IsopyKeyString):
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


    Attributes
    ----------
    flavour
        The flavour of the key string.
    mass_number : MassKeyString
        A reference to itself.


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

    def __new__(cls, string, *, allow_reformatting=True):
        if isinstance(string, cls):
            return string

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
            if string[0] == '-' and string[1:].isdigit():
                raise KeyValueError(cls, string, 'Must be a positive integer')
            else:
                raise KeyValueError(cls, string, 'Can only contain numerical characters')

        return super(MassKeyString, cls).__new__(cls, string, flavour = MassFlavour())

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

    def _sortkey_(self):
        return f'A{self:0>4}'

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

    def _str_options_(self):
        return dict(m = str(self), key = str(self),
                    math = fr'\mathrm{{{self}}}',
                    latex = fr'$\mathrm{{{self}}}$')


class ElementKeyString(IsopyKeyString):
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

    Attributes
    ----------
    flavour
        The flavour of the key string.
    element_symbol : ElementKeyString
        A reference to itself.
    isotopes : IsotopeKeyList
        A key list of all the isotopes of this element.


    Examples
    --------
    >>> isopy.ElementKeyString('Pd')
    'Pd'
    >>> isopy.ElementKeyString('pd')
    'pd'
    """

    def __new__(cls, string, *, allow_reformatting=True):
        if isinstance(string, cls):
            return string

        if not isinstance(string, str):
            raise KeyTypeError(cls, string)

        string = string.strip()
        if allow_reformatting is True:
            string = remove_prefix(string, 'Element_') #For backwards compatibility
            string = remove_prefix(string, 'ELE_') #For backwards compatibility

        if len(string) == 0:
            raise KeyValueError(cls, string, 'Cannot parse empty string')

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

        key = super(ElementKeyString, cls).__new__(cls, string, flavour = ElementFlavour())
        object.__setattr__(key, 'element_symbol', key)
        return key

    def _z_(self):
        return isopy.refval.element.atomic_number.get(self, 0)

    def _sortkey_(self):
        return f'B{self._z_():0>4}'

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

    def _str_options_(self):
        name = isopy.refval.element.symbol_name.get(self, str(self))
        return dict(key = str(self),
                    es=self.lower(), ES=self.upper(), Es=str(self),
                    name=name.lower(), NAME=name.upper(), Name=name,
                    math = fr'\mathrm{{{self}}}',
                    latex = fr'$\mathrm{{{self}}}$')

    @property
    def isotopes(self, isotopes = None):
        if isotopes is None:
            isotopes = isopy.refval.element.isotopes

        return askeylist(isotopes.get(self, []))


class IsotopeKeyString(IsopyKeyString):
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


    Attributes
    ----------
    flavour
        The flavour of the key string.
    mass_number : MassKeyString
        The mass number of this isotope.
    element_symbol : ElementKeyString
        The element symbol of this isotope.
    isotopes : IsotopeKeyList
        A reference to itself.
    mz : tuple[float]
        The mass to charge ratio of this isotope on the basis of the mass number.


    Examples
    --------
    >>> isopy.IsotopeKeyString('Pd104')
    '104Pd'
    >>> isopy.IsotopeKeyString('104pd')
    '104Pd'
    >>> isopy.IsotopeKeyString('Pd104').mass
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

    def __new__(cls, string, *, allow_reformatting=True):
        if isinstance(string, cls):
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
            mass = MassKeyString(mass, allow_reformatting=allow_reformatting)
            element = ElementKeyString(element, allow_reformatting=allow_reformatting)
        except KeyParseError as err:
            raise KeyValueError(cls, string,
                                'unable to separate string into a mass number and an element symbol') from err

        string = '{}{}'.format(mass, element)

        return super(IsotopeKeyString, cls).__new__(cls, string,
                                                   mass_number = mass,
                                                   element_symbol = element,
                                                   mz = float(mass),
                                                   flavour = IsotopeFlavour())

    def __hash__(self):
        return hash( (self.__class__, hash(self.mass_number), hash(self.element_symbol)) )

    def __contains__(self, string):
        """
        Return **True** if `string` is equal to the IsotopeKeyString's mass number or element symbol
        """
        return self.mass_number == string or self.element_symbol == string

    def _filter_(self, mass_number = {}, element_symbol = {}, **filters):
        if filters and not super(IsotopeKeyString, self)._filter_(**filters):
            return False
        if mass_number and not self.mass_number._filter_(**mass_number):
            return False
        if element_symbol and not self.element_symbol._filter_(**element_symbol):
            return False
        return True

    def _sortkey_(self):
        return f'C{self.mz:0>8.3f}{self._z_():0>4}'

    def _z_(self):
        return self.element_symbol._z_()

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
        
    def _str_options_(self):
        options = dict()

        mass_options = self.mass_number._str_options_()
        element_options = self.element_symbol._str_options_()
        options.update(mass_options)
        options.update(element_options)
        options.update(dict(key = str(self),
                       math = fr'{{}}^{{{self.mass_number}}}\mathrm{{{self.element_symbol}}}',
                       latex = fr'$^{{{self.mass_number}}}\mathrm{{{self.element_symbol}}}$'))

        product = list(itertools.product(mass_options.items(), element_options.items()))
        options.update({f'{mk}{ek}': f'{mv}{ev}' for (mk, mv), (ek, ev) in product})
        options.update({f'{mk}-{ek}': f'{mv}-{ev}' for (mk, mv), (ek, ev) in product})
        options.update({f'{ek}{mk}': f'{ev}{mv}' for (mk, mv), (ek, ev) in product})
        options.update({f'{ek}-{mk}': f'{ev}-{mv}' for (mk, mv), (ek, ev) in product})

        return options

    @property
    def isotopes(self):
        """
        Return the key as a single item key list.

        The *isotopes* argument is ignored.

        """
        return askeylist(self)


class MoleculeKeyString(IsopyKeyString):
    """
    A string representation of an molecue consisting of a element and/or isotope key strings.

    Inherits from :class:`str` and therefore contains all the method that a :class:`str` does.
    Unless specifically noted below these methods will return a :class:`str` rather than a
    :class:`MoleculeKeyString`.

    Parameters
    ----------
    string : str
        Mass numbers must be before the element symbol. Any number after the element symbol is
        assumed to be a multiple. Capital letters signify a new element symbol and must be used
        when listing succesive element symbols. Parenthesis can be used to group elements or to
        seperate mass numbers from multipliers.
    allow_reformatting : bool, Default = True
        If ``True`` the string will be reformatted to the correct format. If ``False`` an exception
        will be raised it the string is not correctly formatted.

    Attributes
    ----------
    flavour
        The flavour of the key string.
    element_symbol : MoleculeKeyList
        A molecule key string containing the element formula for this molecule.
    isotopes : MoleculeKeyList
        A molecule key string containing all the isotopes for this molecule.
    mz : tuple[float]
        A tuple of the mass to charge ratio for each molecule in the list on the basis of the mass number.
        Negative charges will always return a positive number.
    components : tuple
        The components of this molecule.
    n : int
        The multiplier of this molecule.
    charge : None, int
        The charge of this molecule.

    Examples
    --------
    >>> isopy.MoleculeKeyString('H2O')
    'H2O'
    >>> isopy.MoleculeKeyString('(1H)2(16O)')
    '(1H)2(16O)'

    >>> isopy.MoleculeKeyString('OH')
    'OH'
    >>> isopy.MoleculeKeyString('oh') # Becomes element symbol Oh
    'Oh'
    """

    def __new__(cls, components, *, allow_reformatting=True, component_flavour='element|isotope'):
        component_flavour = asflavour(component_flavour)
        if type(components) is cls and components.flavour in component_flavour:
            return components

        parsed_components = cls._parse_components(components)
        if type(parsed_components) is MoleculeKeyString:
            new = parsed_components
        else:
            new = cls._new(parsed_components)

        if new.flavour.component_flavour not in component_flavour:
            raise KeyValueError(cls, new, f'Key flavour {new.flavour.component_flavour} not compatible. '
                                          f'Got {new.flavour.component_flavour} expected {component_flavour}')

        if (allow_reformatting is False and isinstance(components, str) and
                str(components) != str(new)):
            raise KeyValueError(cls, str(components), f'Final string "{new}" does not match input')

        return new

    @classmethod
    def _new(cls, components, n=1, charge=None):
        if type(components) is list:
            components = tuple(components)
        elif type(components) is not tuple:
            components = (components,)

        string = cls._make_string_(components, n, charge, bracket=False, square=True)
        component_flavour = cls._find_flavour_(components)
        return super(MoleculeKeyString, cls).__new__(cls, string,
                                                     components=components,
                                                     n=n,
                                                     charge=charge,
                                                     flavour = MoleculeFlavour(component_flavour))

    @classmethod
    def _parse_components(cls, components, ignore_charge = False):
        if not isinstance(components, (list, tuple)):
            components = [components]

        items = []

        for component in components:
            if isinstance(component, (ElementKeyString, IsotopeKeyString, MoleculeKeyString)):
                items.append(component)
            elif isinstance(component, str):
                items += cls._parse_string(component, ignore_charge=ignore_charge)
            elif type(component) is int:
                if component > 0:
                    items.append(component)
                else:
                    raise KeyValueError(cls, component, 'Negative value encountered')
            elif isinstance(component, (list, tuple)):
                items.append(cls._parse_components(component, ignore_charge=ignore_charge))
            else:
                raise KeyTypeError(cls, component)

        if len(items) == 0:
            raise KeyValueError(cls, components, 'No molecule components found')

        out = []
        for i, item in enumerate(items):
            if type(item) is str and not item.isdigit():
                if item.isalpha():
                    out.append(ElementKeyString(item))
                elif item.isalnum():
                    out.append(IsotopeKeyString(item))
                elif not ignore_charge:
                    charge = item.count('+') or item.count('-') * -1
                    if len(out) != 0:
                        if type(out[-1]) is cls and out[-1].charge is None:
                            out[-1] = cls._new(out[-1].components, n=out[-1].n, charge=charge)
                        else:
                            out[-1] = cls._new(out[-1], charge=charge)
                    else:
                        raise KeyValueError(cls, item, f'Unattached charge')

            elif type(item) is int or (type(item) is str and item.isdigit()):
                n = int(item)
                if len(out) != 0:
                    if type(out[-1]) is cls and out[-1].n == 1:
                        out[-1] = cls._new(out[-1].components, n=n, charge=out[-1].charge)
                    else:
                        out[-1] = cls._new(out[-1], n=n)
                else:
                    raise KeyValueError(cls, item, f'Unattached number')

            elif type(item) is list:
                out.append(cls._new(item))

            else:
                # Key string
                out.append(item)

        if len(out) == 1:
            return out[0]
        else:
            return out

    @classmethod
    def _parse_string(cls, string, ignore_charge=False):
        # A molecule contains at least one subcomponent
        components = []

        # This is the variables we are extracting from the string
        number = ''
        text = ''

        i = 0
        while i < len(string):
            char = string[i]

            if char.isdigit():
                if text:
                    components.append(number + text)
                    number, text = '', ''

                number += char

            elif char.isalpha():
                if text and (char.isupper() or not text.isalpha()):
                    components.append(number + text)
                    number, text = '', ''

                if number and len(components) != 0:
                    components.append(number + text)
                    number, text = '', ''

                text += char

            elif char == '+' or char == '-':
                if text.count(char) != len(text) or number:
                    components.append(number + text)
                    number, text = '', ''

                text += char

            elif char == '(':
                if number or text:
                    components.append(number + text)
                    number, text = '', ''

                open_brackets = 1
                for j, jc in enumerate(string[i + 1:]):
                    if jc == '(':
                        open_brackets += 1
                    elif jc == ')':
                        open_brackets -= 1

                    if open_brackets == 0:
                        j = i + j + 1
                        break
                else:
                    raise KeyValueError(cls, string, f'Unmatched "(" at index {i}')

                components.append(cls._parse_components(string[i + 1:j], ignore_charge=ignore_charge))
                i = j

            elif char == '[':
                if number or text:
                    components.append(number + text)
                    number, text = '', ''

                open_brackets = 1
                for j, jc in enumerate(string[i + 1:]):
                    if jc == '[':
                        open_brackets += 1
                    elif jc == ']':
                        open_brackets -= 1

                    if open_brackets == 0:
                        j = i + j + 1
                        break
                else:
                    raise KeyValueError(cls, string, f'Unmatched "[" at index {i}')

                components.append(cls._parse_components(string[i + 1:j], ignore_charge=ignore_charge))
                i = j

            else:
                raise KeyValueError(cls, string, f'Invalid character "{char}" found.')

            # Move to next char
            i += 1

        components.append(number + text)
        while components.count('') > 0:
            components.remove('')

        return components

    @classmethod
    def _find_flavour_(cls, components):
        contains_flavours = tuple()
        for component in components:
            if type(component) is MoleculeKeyString:
                contains_flavours += component.flavour.component_flavour._flavours
            else:
                contains_flavours += (component.flavour, )

        return ListFlavour(contains_flavours)

    @classmethod
    def _make_string_(cls, components, n=None, charge=None, format=None, bracket=True, square = False):
        if format == 'math':
            l = fr'\left('
            r = fr'\right)'
        else:
            l = '('
            r = ')'

        ncharge = ''
        if n is not None and n != 1:
            if format == 'math':
                ncharge += f'_{{{n}}}'
            else:
                ncharge += str(n)

        if charge is not None:
            c = ('+' * charge or '-' * abs(charge))
            if format == 'math':
                ncharge += f'^{{{c}}}'
            else:
                ncharge += c

        if isinstance(components, (list, tuple)) and len(components) == 1:
            components = components[0]

        if type(components) == ElementKeyString:
            return  f'{components.str(format)}{ncharge}'

        elif type(components) == IsotopeKeyString:
            if format == 'math' or not bracket:
                return f'{components.str(format)}{ncharge}'
            else:
                return f'[{components.str(format)}]{ncharge}'

        elif type(components) == MoleculeKeyString:
            if ncharge:
                return f'{l}{components.str(format)}{r}{ncharge}'
            else:
                return cls._make_string_(components.components, components.n, components.charge, format, bracket)

        else:
            string = ''.join([cls._make_string_(c, format=format) for c in components])

            if ncharge:
                string = f'{l}{string}{r}{ncharge}'

            if bracket:
                return f'{l}{string}{r}'
            elif square:
                return f'[{string}]'
            else:
                return string

    def _z_(self):
        z = 0
        for sc in self.components:
            z += (sc._z_() * self.n)

        return z

    def _sortkey_(self):
        if self.flavour == 'molecule[isotope]':
            return f'D{self.mz:0>8.3f}{self._z_():0>4}'
        else:
            return f'D{0:0>8.3f}{self._z_():0>4}'

    # TODO examples and more options
    def str(self, format=None):
        """
        Return a ``str`` object of the key string.

        the optional *format* can either be a string matching one of the format options or a
        string which can be formatted using the format options as key words.

        Format options are:
        * ``"key"`` - Same as ``str(keystring)``.
        * ``"math"`` -  Key string formatted for latex math mode.
        * ""`latex`"" - Same as above but including $ $ math deliminators.
        """
        return super(MoleculeKeyString, self).str(format)

    def _str_options_(self):
        options = dict(key = str(self))
        options['math'] = self._make_string_(self, format='math', bracket=False)
        options['latex'] = fr'${self._make_string_(self, format="math", bracket=False)}$'
        return options

    @property
    def mz(self):
        # Has to be a property as it looks up the value
        return isopy.refval.isotope.mass_number.get(self)

    @property
    def element_symbol(self):
        components = []
        for c in self.components:
            components.append(c.element_symbol)

        return self._new(components, self.n, self.charge)

    def _isotopes_(self, isotopes):
        if isotopes is None:
            isotopes = isopy.refval.element.isotopes

        all = [[]]

        for sc in self.components:
            # Should save a bit of time
            if type(sc) is MoleculeKeyString:
                molecules = sc._isotopes_(isotopes)

            for i in range(self.n):
                if type(sc) is IsotopeKeyString:
                    all = [a + [sc] for a in all]
                elif type(sc) is ElementKeyString:
                    element_isotopes = isotopes[sc]
                    alliso = []
                    for element_isotope in element_isotopes:
                        alliso.extend([a + [element_isotope] for a in all])
                    all = alliso
                else:
                    alliso = []
                    for molecule in molecules:
                        alliso.extend([a + [molecule] for a in all])
                    all = alliso

        return [self._new(mol, 1, self.charge) for mol in all]

    @property
    def isotopes(self, isotopes = None):
        return askeylist(self._isotopes_(isotopes))


class RatioKeyString(IsopyKeyString):
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


    Attributes
    ----------
    flavour
        The flavour of this key string.
    numerator : keystring
        The numerator of this ratio.
    denominator : keystring
        The denominator of this ratio.


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

    def __new__(cls, string, *, allow_reformatting=True, numerator_flavour = 'any', denominator_flavour = 'any'):
        if isinstance(string, cls):
            return string

        if isinstance(string, tuple) and len(string) == 2:
            numer, denom = string

        elif not isinstance(string, str):
            raise KeyTypeError(cls, string)
        else:
            string = string.strip()

            # For backwards compatibility
            if startswith(string, 'Ratio_') and allow_reformatting is True: #.startswith('Ratio_'):
                string = remove_prefix(string, 'Ratio_')
                try:
                    numer, denom = string.split('_', 1)
                except:
                    raise KeyValueError(cls, string,
                                        'unable to split string into numerator and denominator')

            # For backwards compatibility
            elif startswith(string, 'RAT') and string[3].isdigit() and allow_reformatting is True:
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

        numer = askeystring(numer, allow_reformatting=allow_reformatting, flavour = numerator_flavour)
        denom = askeystring(denom, allow_reformatting=allow_reformatting, flavour = denominator_flavour)

        for n in range(1, 10):
            divider = '/' * n
            if numer.count(divider) > 0 or denom.count(divider) > 0:
                continue
            else:
                break
        else:
            raise KeyValueError('Limit of nested ratios reached')

        string = f'{numer}{divider}{denom}'

        return super(RatioKeyString, cls).__new__(cls, string,
                                                  numerator = numer, denominator = denom,
                                                  flavour = RatioFlavour(numer.flavour, denom.flavour))

    def __hash__(self):
        return hash( (self.__class__, hash(self.numerator), hash(self.denominator)) )

    def __contains__(self, string):
        """
        Return **True** if `string` is equal to the RatioKeyString's numerator or denominator
        """
        return self.numerator == string or self.denominator == string

    def _filter_(self, numerator = {}, denominator = {}, **filters):
        if filters and not super(RatioKeyString, self)._filter_(**filters):
            return False
        if numerator and not self.numerator._filter_(**numerator):
            return False
        if denominator and not self.denominator._filter_(**denominator):
            return False
        return True

    def _sortkey_(self):
        return f'E{self.denominator._sortkey_()}/{self.numerator._sortkey_()}'

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

        options = self._str_options_()
        options['n'] = n
        options['d'] = d
        options['n/d'] = nd

        if format in options:
            return options[format]
        else:
            return format.format(**options)

    def _str_options_(self):
        options = dict(key = str(self))

        nmath = self.numerator.str('math')
        if type(self.numerator) is RatioKeyString:
            nmath = fr'\left({nmath}\right)'

        dmath = self.denominator.str('math')
        if type(self.denominator) is RatioKeyString:
            dmath = fr'\left({dmath}\right)'

        options['math'] = fr'\cfrac{{{nmath}}} {{{dmath}}}'
        options['latex'] = fr'${options["math"]}$'
        return options

    def _flatten_(self):
        return self.numerator._flatten_() + self.denominator._flatten_()


class GeneralKeyString(IsopyKeyString):
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


    Attributes
    ----------
    flavour
        The flavour of this key string.


    Examples
    --------
    >>> isopy.GeneralKeyString('harry')
    'harry'
    >>> isopy.GeneralKeyString('pd')
    'pd'
    """

    def __new__(cls, string, *, allow_reformatting=True):
        if isinstance(string, cls):
            return string

        elif isinstance(string, (str, int, float)):
            string = str(string).strip()
        else:
            raise KeyTypeError(cls, string)

        if len(string) == 0:
            raise KeyValueError(cls, string, 'Cannot parse empty string')

        if allow_reformatting is True:
            string = remove_prefix(string, 'GEN_') #For backwards compatibility
            #colname = string.replace('/', '_SLASH_') #For backwards compatibility
            string = string.replace('_SLASH_', '/') #For backwards compatibility
        return super(GeneralKeyString, cls).__new__(cls, string, flavour = GeneralFlavour())

    def _sortkey_(self):
        return f'F{self}'

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

    def _str_options_(self):
        return dict(key = str(self),
                    math = fr'\mathrm{{{self}}}',
                    plt = fr'$\mathrm{{{self}}}$',
                    tex = str(self))


@lru_cache(CACHE_MAXSIZE)
def keystring(key, *, allow_reformatting=True, flavour='any'):
    """
    Returns an key string with the highest priority compatible flavour.

    Parameters
    ----------
    key
        A string to be converted into a key string.
    allow_reformatting : bool, Default = True
        If ``True`` the string can be reformatted to get the correct format. If ``False`` only a string that already
        has the correct format is considered.
    flavour
        The possible flavour(s) of the key string.

    Returns
    -------
    IsopyKeyString
    """
    flavours = asflavour(flavour)

    for flavour in flavours:
        try:
            return flavour._keystring_(key, allow_reformatting=allow_reformatting)
        except KeyParseError as err:
            pass

    raise KeyValueError(key, IsopyKeyString.__name__,
                        f'unable to parse {type(key).__name__} "{key}" into {flavours}')


@lru_cache(CACHE_MAXSIZE)
def askeystring(key, *, allow_reformatting=True, flavour ='any'):
    """
    Returns a key string preserving the flavour if valid.

    If *key* is already a key string with a flavour in *flavours* no attempt is made to convert it to a
    flavour of higher priority.

    Parameters
    ----------
    key
        A string to be converted into a key string.
    allow_reformatting : bool, Default = True
        If ``True`` the string can be reformatted to get the correct format. If ``False`` only a string that already
        has the correct format is considered.
    flavour
        The possible flavour(s) of the key string.

    Returns
    -------
    IsopyKeyString
    """
    flavour = asflavour(flavour)

    if isinstance(key, IsopyKeyString) and key.flavour in flavour:
        return key
    else:
        return keystring(key, allow_reformatting=allow_reformatting, flavour=flavour)

################
### Key List ###
################

def parse_keyfilters(**filters):
    """
    Parses key filters into a format that is accepted by KeyString._filter. Allows
    nesting of filter arguments for RatioKeyString's.
    """
    parsed_filters = {}
    for key, v in list(filters.items()):
        if v is None:
            continue

        try:
            attr, comparison = key.rsplit('_', 1)
        except ValueError:
            attr = key
            comparison = 'eq'

        if comparison not in ('eq', 'neq', 'lt', 'le', 'ge', 'gt'):
            attr = f'{attr}_{comparison}'
            comparison = 'eq'

        if comparison in ('eq', 'neq') and not isinstance(v, (list, tuple)):
            v = (v, )

        parsed_filters[f'{attr}_{comparison}'] = v

    #For ratio keys
    numerator = _split_filter('numerator', parsed_filters)
    if numerator:
        parsed_filters['numerator'] = parse_keyfilters(**numerator)

    denominator = _split_filter('denominator', parsed_filters)
    if denominator:
        parsed_filters['denominator'] = parse_keyfilters(**denominator)

    return parsed_filters

def _split_filter(prefix, filters):
    #Seperates out filters with a specific prefix
    out = {}
    if prefix[-1] != '_': prefix = f'{prefix}_'
    for key in tuple(filters.keys()):
        if key == prefix[:-1]:
            #avoids getting errors
            out['key_eq'] = filters.pop(prefix[:-1])
        elif startswith(key, prefix): #.startswith(prefix):
            parsed_key = remove_prefix(key, prefix)
            if parsed_key in ('eq', 'neq', 'lt', 'le', 'ge', 'gt'):
                parsed_key = f'key_{parsed_key}'
            out[parsed_key] = filters.pop(key)

    return out

def combine_keys_func(func):
    @functools.wraps(func)
    def combine(*args, **kwargs):
        keys = tuple()

        for arg in args:
            if isinstance(arg, str):
                keys += (arg,)
            elif type(arg) is IsopyKeyList:
                keys += tuple(arg)
            elif isinstance(arg, np.dtype) and arg.names is not None:
                keys += tuple(name for name in arg.names)
            elif isinstance(arg, ndarray) and arg.dtype.names is not None:
                keys += tuple(name for name in arg.dtype.names)
            elif isinstance(arg, dict):
                keys += tuple(arg.keys())
            elif isinstance(arg, abc.Iterable):
                keys += tuple(a for a in arg)
            else:
                keys += (arg,)
        return func(keys, **kwargs)

    return combine

def list_keyattr(attr, keylist = True):
    if keylist is True:
        outname = 'IsopyKeyList'
    else:
        outname = tuple.__name__

    def func(self):
        """
        Returns a sequence containing the ``{attr}`` attribute of each key in the list.

        ``None`` is returned if one or more of the keys is missing the ``{attr}`` attribute.

        Returns
        -------
        {outname} | None
        """
        try:
            out = tuple(getattr(key, attr) for key in self)
        except AttributeError:
            return None
        else:
            if keylist is True:
                return askeylist(*out)
            else:
                return out

    return property(func, doc=func.__doc__.format(attr=attr, outname=outname))


class IsopyKeyList(tuple):
    """
    A sequence of key strings.

    Key lists can be created using ``keylist`` and ``askeylist``.

    Is a subclass of tuple and contains all the methods that a normal tuple does unless otherwise noted.
    Only methods that behave differently from a normal tuple are documented below.

    Attributes
    ----------
    flavour : ListFlavour
        The flavour of the keys in the key list.
    flavours : tuple
        A tuple containing the flavour of each key in the list.
    mass_numbers : IsopyKeyList
        A key list containing the mass number of each key in the list.
        ``None`` if one or more of the keys is missing the ``mass_number`` attribute.
    element_symbols : IsopyKeyList
        A key list containing the element symbol of each key in the list.
        ``None`` if one or more of the keys is missing the ``element_symbol`` attribute.
    isotopes : IsopyKeyList
        A key list containing the isotopes of each key in the list.
        ``None`` if one or more of the keys is missing the ``isotopes`` attribute.
    mz : tuple
        A key list containing the mass to charge ratio of each key in the list.
        ``None`` if one or more of the keys is missing the ``mz`` attribute.
    numerators : IsopyKeyList
        A key list containing the ratio numerator of each key in the list.
        ``None`` if one or more of the keys is missing the ``numerator`` attribute.
    denominators : IsopyKeyList
        A key list containing the ratio denominator of each key in the list.
        ``None`` if one or more of the keys is missing the ``denominator`` attribute.
    common_denominator : IsopyKeyString, None
        The common demoninator of all ratio key strings in the sequence.
        ``None`` if there is no common denominator or the list contains non-ratio keys.
    """
    def __new__(cls, keys, flavour, ignore_duplicates = False, allow_duplicates = True):
        if ignore_duplicates:
            keys = list(dict.fromkeys(keys).keys())
        elif not allow_duplicates and (len(set(keys)) != len(keys)):
            raise ValueError(f'duplicate key found in list {keys}')

        flavour = asflavour(tuple(k.flavour for k in keys))

        obj = super(IsopyKeyList, cls).__new__(cls, keys)
        obj.flavour = flavour
        return obj

    def _repr_latex_(self):
        return '$$\left[' + r', '.join([k.str("math") for k in self]) + r'\right]$$'

    def __repr__(self):
        return f"""{self.__class__.__name__}({", ".join([fr"'{str(k)}'" for k in self])}, flavour='{str(self.flavour)}')"""

    def __call__(self):
        return self

    def __hash__(self):
        return hash( (self.__class__, super(IsopyKeyList, self).__hash__()) )

    def __eq__(self, other):
        if not isinstance(other, IsopyKeyList):
            try:
                other = askeylist(other, flavour=self.flavour)
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
                    item = askeystring(item, flavour=self.flavour)
                except:
                    return False

            if super(IsopyKeyList, self).__contains__(item) is False:
                return False

        return True

    def __getitem__(self, index):
        """
        Return the item at `index`. `index` can be int, slice or sequence of int.
        """
        if isinstance(index, slice):
            return askeylist(super(IsopyKeyList, self).__getitem__(index))
        elif isinstance(index, abc.Iterable):
            return askeylist(tuple(super(IsopyKeyList, self).__getitem__(i) for i in index))
        else:
            return super(IsopyKeyList, self).__getitem__(index)

    def __truediv__(self, denominator):
        if isinstance(denominator, (tuple, list)):
            if len(denominator) != len(self):
                raise ValueError(f'size of values ({len(self)}) not compatible with size of key list ({len(self)}')
            return askeylist(tuple(n / denominator[i] for i, n in enumerate(self)))
        else:
            return askeylist(tuple(n / denominator for i, n in enumerate(self)))

    def __rtruediv__(self, numerator):
        if isinstance(numerator, (tuple, list)):
            if len(numerator) != len(self):
                raise ValueError(f'size of values ({len(self)}) not compatible with size of key list ({len(self)}')
            return askeylist(tuple(numerator[i] / d for i, d in enumerate(self)))
        else:
            return askeylist(tuple(numerator / d for i, d in enumerate(self)))

    def __add__(self, other):
        other = askeylist(other)
        return askeylist((*self, *other))

    def __radd__(self, other):
        other = askeylist(other)
        return askeylist(other, self)

    def __sub__(self, other):
        other = askeylist(other)
        return askeylist((key for key in self if key not in other))

    def __rsub__(self, other):
        return askeylist(other).__sub__(self)

    def __and__(self, other):
        this = dict.fromkeys(self)

        if not isinstance(other, IsopyKeyList):
            other = askeylist(other)

        other = [hash(o) for o in dict.fromkeys(other)]
        this = (t for t in this if hash(t) in other)

        return askeylist(tuple(this))

    def __or__(self, other):
        this = self

        if not isinstance(other, IsopyKeyList):
            other = askeylist(other)

        this = tuple(dict.fromkeys((*this, *other)))

        return askeylist(this)

    def __xor__(self, other):
        this = dict.fromkeys(self)

        if not isinstance(other, IsopyKeyList):
            other = askeylist(other)
        other = dict.fromkeys(other)

        this_hash = [hash(t) for t in this]
        other_hash = [hash(o) for o in dict.fromkeys(other)]

        this = (*(t for i, t in enumerate(this) if this_hash[i] not in other_hash),
                *(o for i, o in enumerate(other) if other_hash[i] not in this_hash))

        return askeylist(this)

    def __rand__(self, other):
        return askeylist(other).__and__(self)

    def __ror__(self, other):
        return askeylist(other).__or__(self)

    def __rxor__(self, other):
        return askeylist(other).__xor__(self)

    def _view_array_(self, a):
        if isinstance(a, void):
            view = a.view((IsopyVoid, a.dtype))
        else:
            view =  a.view(IsopyNdarray)

        view.flavour = self.flavour
        view.keys = self
        return view


    def filter(self, key_eq= None, key_neq = None, **filters):
        """
        Returns a new key list containing the keys that satify **all** the filter arguments given.

        Parameters
        ----------
        key_eq : str, Sequence[str], Optional
           Only key strings equal to/found in *key_eq* pass this filter.
        key_neq : str, Sequence[str], Optional
           Only key strings not equal to/found in *key_neq* pass this filter.
        **filters
            A filter consists of the attribute name followed by the comparison type speperated by a ``_``.
            Avaliable comparison types are:
             * ``eq`` for ``==`` for a single value or ``in`` for multiple values
             * ``neq`` for ``!=`` for a single value or ``not in`` for multiple values
             * ``lt`` for ``<``
             * ``gt`` for ``>``
             * ``le`` for ``<=``
             * ``ge`` for ``>=``

            Any filter preceded by ``numerator_`` or ``denominator_``
            will be forwarded to the numerator and denominator keys of ratio key strings.

        Examples
        --------
        >>> keylist = isopy.askeylist(['101Ru', '105Pd', '111Cd'])
        >>> keylist.filter(key_neq = '105pd')
        ('101Ru', '111Cd')
        >>> keylist.filter(element_symbol_eq=['pd', 'cd'])
        ('105Pd', '111Cd')
        >>> keylist.filter(mass_number_le=105)
        ('101Ru', '105Pd')

        >>> keylist = isopy.askeylist(['101Ru/102Ru', '108Pd/105Pd', '111Cd/110Cd'])
        >>> keylist.filter(numerator_element_symbol_neq='pd')
        ('101Ru/102Ru', '111Cd/110Cd')
        """

        filters = parse_keyfilters(key_eq=key_eq, key_neq=key_neq, **filters)
        return askeylist(key for key in self if key._filter_(**filters))

    @functools.wraps(tuple.count)
    def count(self, item):
        try:
            askeystring(item, flavour=self.flavour)
        except:
            return 0
        else:
            return super(IsopyKeyList, self).count(item)

    @functools.wraps(tuple.index)
    def index(self, item, *args):
        try:
            item = askeystring(item, flavour=self.flavour)
            return super(IsopyKeyList, self).index(item, *args)
        except (KeyValueError, ValueError):
            raise ValueError(f'{item} not in {self.__class__}')

    def _str_(self):
        return [str(key) for key in self]
    
    def strlist(self, format=None):
        """
        Return a list of ``str`` object for each key in the key list.

        Analogous to ``[key.str(format) for key in keylist]``
        """
        return [key.str(format) for key in self]

    def sorted(self):
        """
        Returns a sorted copy of the list.
        """
        return askeylist(sorted(self, key= lambda k: k._sortkey_()))

    def reversed(self):
        """
        Returns a reversed copy of the list.
        """
        return askeylist(reversed(self))

    def flatten(self, ignore_duplicates = False, allow_duplicates = True):
        """
        Returns a flattened copy of the list. Only ratio key string can be flattened all other key strings will
        remain the same.

        Parameters
        ----------
        ignore_duplicates : bool, Default = True
            If ``True`` all duplicate items will be removed from the sequence.
        allow_duplicates : bool, Default  = True
            If ``False`` a ListDuplicateError will be raised if the sequence contains any duplicate items.

        Examples
        --------
        >>> keylist = isopy.keylist(['103rh', 'ru/pd', 'ag', '111cd/105pd'])
        >>> keylist.flatten()
        ('103Rh', 'Ru', 'Pd', Ag', '111Cd', '105Pd')
        """
        keys = tuple()
        for k in self:
            keys += k._flatten_()
        return askeylist(keys, ignore_duplicates=ignore_duplicates, allow_duplicates=allow_duplicates)

    ##################
    ### Attributes ###
    ##################

    flavours = list_keyattr('flavour', False)
    mass_numbers = list_keyattr('mass_number', True)
    element_symbols = list_keyattr('element_symbol', True)
    isotopes = list_keyattr('isotopes', True)
    mz = list_keyattr('mz', False)
    numerators = list_keyattr('numerator', True)
    denominators = list_keyattr('denominator', True)

    @property
    def common_denominator(self):
        denominators = self.denominators
        if denominators is not None and len(set(denominators)) == 1:
            return denominators[0]
        else:
            return None


@combine_keys_func
@lru_cache(CACHE_MAXSIZE)
def keylist(*keys, ignore_duplicates=False, allow_duplicates=True, allow_reformatting=True, flavour ='any'):
    """
    Returns a key list with the highest priority flavour compatible with each key string.

    *keys* can consist of single strings, sequences of strings, dictionaries, isopy arrays and numpy arrays.
    For dictionaries and arrays the keys or dtype.name values are used as keys.

    Parameters
    ----------
    *keys
        Keys to be included in the list.
    ignore_duplicates : bool, Default = True
        If ``True`` all duplicate items will be removed from the sequence.
    allow_duplicates : bool, Default  = True
        If ``False`` a ListDuplicateError will be raised if the sequence contains any duplicate items.
    allow_reformatting : bool, Default = True
        If ``True`` the string can be reformatted to get the correct format. If ``False`` only strings that already
        have the correct format are considered.
    flavour
        The possible flavour(s) of key strings in the key list.

    Returns
    -------
    IsopyKeyList
    """
    flavour = asflavour(flavour)

    keys = [keystring(k, flavour=flavour, allow_reformatting=allow_reformatting) for k in keys[0]]

    return IsopyKeyList(keys, flavour, ignore_duplicates=ignore_duplicates,
                       allow_duplicates=allow_duplicates)

@combine_keys_func
@lru_cache(CACHE_MAXSIZE)
def askeylist(*keys, ignore_duplicates=False, allow_duplicates=True, allow_reformatting=True, flavour ='any'):
    """
    Returns a key list preserving the flavour of each key string if it has a valid flavour.

    If a key is already a key string with a flavour in *flavour* no attempt is made to convert it to another
    flavour with higher priority.

    *keys* can consist of single strings, sequences of strings, dictionaries, isopy arrays and numpy arrays.
    For dictionaries and arrays the keys or dtype.name values are used as keys.

    Parameters
    ----------
    *keys
        Keys to be included in the list.
    ignore_duplicates : bool, Default = True
        If ``True`` all duplicate items will be removed from the sequence.
    allow_duplicates : bool, Default  = True
        If ``False`` a ListDuplicateError will be raised if the sequence contains any duplicate items.
    allow_reformatting : bool, Default = True
        If ``True`` the string can be reformatted to get the correct format. If ``False`` only strings that already
        have the correct format are considered.
    flavour
        The possible flavour(s) of key strings in the key list.

    Returns
    -------
    IsopyKeyList
    """
    flavour = asflavour(flavour)

    keys = [askeystring(k, allow_reformatting=allow_reformatting, flavour=flavour) for k in keys[0]]

    return IsopyKeyList(keys, flavour, ignore_duplicates=ignore_duplicates,
                        allow_duplicates=allow_duplicates)

###################################
### Mixins for Arrays and dicts ###
###################################
class ArrayFuncMixin:
    # For IsopyArray and RefValDict
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc not in APPROVED_FUNCTIONS:
            warnings.warn(f"The functionality of {ufunc.__name__} has not been tested with isopy arrays.")

        if 'out' in kwargs and ufunc.nout == 1:
            kwargs['out'] = kwargs.pop('out')[0]

        if method != '__call__':
            ufunc = getattr(ufunc, method)
            #raise TypeError(f'the {ufunc.__name__} method "{method}" is not supported by isopy arrays')

        return call_array_function(ufunc, *inputs, **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        if func not in APPROVED_FUNCTIONS:
            warnings.warn(f"The functionality of {func.__name__} has not been tested with isopy arrays.")

        return call_array_function(func, *args, **kwargs)


class ToTextMixin:
    # For IsopyArray and RefValDict
    def __to_text(self, delimiter=', ', include_row = False, include_dtype=False,
                nrows = None, row_names = None, **vformat):

        sdict = {}
        if include_row or row_names is not None:
            if row_names is None:
                if self.ndim == 0:
                    row_names = ['None']
                else:
                    row_names = [str(i) for i in range(self.size)]

            elif isinstance(row_names, str):
                row_names = [row_names]

            if len(row_names) != self.size:
                raise ValueError(f'Size of row names ({len(row_names)} does not match size of array ({self.size})')

            if self.ndim == 0:
                sdict['(row)'] = ['None']
            else:
                sdict['(row)'] = row_names

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
                nrows = None, row_names = None, **vformat):
        """
        Convert the array/dictionary to a text string.

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
        row_names
            A list of the name for each row in the table. If not given the index of the row will be used.
        vformat : str, Optional
            Format string for different kinds of data. The key denoted the data kind. Common data
            kind strings ara ``"f"`` for floats, ``"i"`` for integers and ``"S"`` for strings.
            Dictionary containing a format string for different kinds of data.  Most common ``"f"``.
            Default format string for each data type is ``'{}'``. A list of all avaliable data kinds
            is avaliable `here <https://numpy.org/doc/stable/reference/arrays.interface.html>`_.
        """
        flen, sdict, nrows = self.__to_text(delimiter, include_row, include_dtype, nrows, row_names, **vformat)

        return '{}\n{}'.format(delimiter.join(['{:<{}}'.format(k, flen[k]) for k in sdict.keys()]),
                                   '\n'.join('{}'.format(delimiter.join('{:<{}}'.format(sdict[k][i], flen[k]) for k in sdict.keys()))
                                             for i in range(nrows)))

    def to_table(self, include_row = False, include_dtype=False,
                nrows = None, row_names = None, **vformat):
        """
        Convert the array/dictionary to a markdown table text string.

        Parameters
        ----------
        include_row : bool, Default = False
            If ``True`` a column containing the row index is included. *None* Is given as the
            row index for 0-dimensional arrays.
        include_dtype : bool, Default = False
            If ``True`` the column data type is included in the first row next to the column name.
        nrows : int, Optional
            The number of rows to show.
        row_names
            A list of the name for each row in the table. If not given the index of the row will be used.
        vformat : str, Optional
            Format string for different kinds of data. The key denoted the data kind. Common data
            kind strings ara ``"f"`` for floats, ``"i"`` for integers and ``"S"`` for strings.
            Dictionary containing a format string for different kinds of data.  Most common ``"f"``.
            Default format string for each data type is ``'{}'``. A list of all avaliable data kinds
            is avaliable `here <https://numpy.org/doc/stable/reference/arrays.interface.html>`_.
        """

        delimiter = '| '

        flen, sdict, nrows = self.__to_text(delimiter, include_row, include_dtype, nrows, row_names, **vformat)
        flen = {k: f if f > 4 else 4 for k, f in flen.items()}

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
                nrows = None, row_names = None, **vformat):
        """
       Convert the array/dictionary to a IPython markdown table. This will render
       a table in an IPython console or a Jupyter cell.

       An exception is raised if IPython is not installed.

       Parameters
       ----------
       include_row : bool, Default = False
           If ``True`` a column containing the row index is included. *None* Is given as the
           row index for 0-dimensional arrays.
       include_dtype : bool, Default = False
           If ``True`` the column data type is included in the first row next to the column name.
       nrows : int, Optional
           The number of rows to show.
       row_names
            A list of the name for each row in the table. If not given the index of the row will be used.
       vformat : str, Optional
           Format string for different kinds of data. The key denoted the data kind. Common data
           kind strings ara ``"f"`` for floats, ``"i"`` for integers and ``"S"`` for strings.
           Dictionary containing a format string for different kinds of data.  Most common ``"f"``.
           Default format string for each data type is ``'{}'``. A list of all avaliable data kinds
           is avaliable `here <https://numpy.org/doc/stable/reference/arrays.interface.html>`_.
       """
        if IPython is not None:
            return IPython.display.Markdown(self.to_table(include_row, include_dtype, nrows, row_names, **vformat))
        else:
            raise TypeError('IPython not installed')


class ToFromFileMixin:
    # For IsopyArray and RefValDict
    def to_csv(self, filename, comments = None, keys_in_first='r',
              dialect = 'excel', comment_symbol = '#'):
        """
        Save the array/dictionary to a csv file.

        Parameters
        ----------
        filename : str, StringIO, BytesIO
            Path/name of the csv file to be created. Any existing file with the same path/name will be
            over written. Also accepts file like objects.
        comments : str, Sequence[str], Optional
            Comments to be included at the top of the file
        keys_in_first : {'c', 'r'}
            Only used if the input has keys. Give 'r' if the keys should be in the first row and 'c' if the
            keys should be in the first column.
        comment_symbol : str, Default = '#'
            This string will precede any comments at the beginning of the file.
        dialect
            The CSV dialect used to save the file. Default to 'excel' which is a ', ' seperated file.
        """
        isopy.write_csv(filename, self, comments=comments, keys_in_first=keys_in_first,
                        dialect=dialect, comment_symbol=comment_symbol)

    def to_xlsx(self, filename, sheetname = 'sheet1', comments = None,
               keys_in_first= 'r', comment_symbol= '#', start_at ="A1", append = False, clear = True):
        """
        Save the array/dictionary to an excel workbook.

        Parameters
        ----------
        filename : str, BytesIO
            Path/name of the excel file to be created. Any existing file with the same path/name
            will be overwritten. Also accepts file like objects.
        sheetname : isopy_array_like, numpy_array_like
            Data will be saved in a sheet with this name.
        comments : str, Sequence[str], Optional
            Comments to be included at the top of the file
        comment_symbol : str, Default = '#'
            This string will precede any comments at the beginning of the file
        keys_in_first : {'c', 'r'}
            Only used if the input has keys. Give 'r' if the keys should be in the first row and 'c' if the
            keys should be in the first column.
        start_at: str, (int, int)
            The first cell where the data is written. Can either be a excel style cell reference or a (row, column)
            tuple of integers.
        append : bool, Default = False
            If ``True`` and *filename* exists it will append the data to this workbook. An exception
            is raised if *filename* is not a valid excel workbook.
        clear : bool, Default = True
            If ``True`` any preexisting sheets are cleared before any new data is written to it.
        """
        isopy.write_xlsx(filename, comments=comments, keys_in_first=keys_in_first, comment_symbol=comment_symbol,
                        start_at=start_at, append=append, clear=clear, **{sheetname: self})

    def to_clipboard(self, comments=None, keys_in_first='r', dialect = 'excel', comment_symbol = '#'):
        """
        Copy the array/dictionary to the clipboard.

        Parameters
        ----------
        comments : str, Sequence[str], Optional
            Comments to be included
        keys_in_first : {'c', 'r'}
            Only used if the input has keys. Give 'r' if the keys should be in the first row and 'c' if the
            keys should be in the first column.
        dialect
            The CSV dialect used to copy the data to the clipboard. Default to 'excel' which is a ', ' seperated file.
        comment_symbol : str, Default = '#'
            This string will precede any comments.
        """
        isopy.write_clipboard(self, comments=comments,  keys_in_first=keys_in_first,
                        dialect=dialect, comment_symbol=comment_symbol)

    def to_dataframe(self):
        """
        Convert the array/dictionary to a pandas dataframe
        """
        if pandas is not None:
            return pandas.DataFrame(self)
        else:
            raise TypeError('Pandas is not installed')

    # FROM functions
    @classmethod
    def from_csv(cls, filename, comment_symbol ='#', keys_in_first=None, encoding = None, dialect = None, **kwargs):
        """
        Convert the data in csv file to IsopyArray/RefValDict.

        Parameters
        ----------
        filename : str, bytes, StringIO, BytesIO
            Path for file to be opened. Alternatively a file like byte string can be supplied.
            Also accepts file like objects.
        has_keys : bool, None
            If True or *keys_in_first* is not None a dictionary will always be returned. If False an nexted list is
            always returned. If None it will return a nested list of all values in the first row and column
            can be converted to/is a float.
        keys_in_first : {'c', 'r', None}
            Where the keys are found. Give 'r' if the keys are found in the first row and 'c' if the
            keys are in first column. If None it will analyse the data to determine where the keys are. If *has_keys*
            is not False an exception will be raised if it cant determine where the keys are.
        comment_symbol : str, Default = '#'
            Rows starting with this string will be ignored.
        encoding : str
            Encoding of the file. If None the encoding will be guessed from the file.
        dialect
            Dialect of the csv file. If None the dialect will be guessed from the file.
        **kwargs
            Arguments pass when initialing the array/dictionary.
        """
        data = isopy.read_csv(filename, comment_symbol=comment_symbol, has_keys=True, keys_in_first=keys_in_first,
                                  encoding=encoding, dialect=dialect)
        return cls(data, **kwargs)

    @classmethod
    def from_xlsx(cls, filename, sheetname, keys_in_first=None,
                  comment_symbol='#', start_at='A1', **kwargs):
        """
        Convert data in excel worksheet to IsopyArray/RefValDict
        Parameters
        ----------
        filename : str, bytes, BytesIO
            Path for file to be opened. Also accepts file like objects.
        sheetname : str, int, Optional
            To load data from a single sheet in the workbook pass either the name of the sheet or
            the position of the sheet. If nothing is specified all the data for all sheets in the
            workbook is returned.
        has_keys : bool, None
            If True or *keys_in_first* is not None a dictionary will always be returned. If False an nexted list is
            always returned. If None it will return a nested list of all values in the first row and column
            can be converted to/is a float.
        keys_in_first : {'c', 'r', None}
            Where the keys are found. Give 'r' if the keys are found in the first row and 'c' if the
            keys are in first column. If None it will analyse the data to determine where the keys are. If *has_keys*
            is not False an exception will be raised if it cant determine where the keys are.
        comment_symbol : str, Default = '#'
            Rows starting with this string will be ignored.
        start_at : str, (int, int)
            Start scanning at this cell. Can either be a excel style cell reference or a (row, column) tuple of integers.
        **kwargs
            Arguments pass when initialing the array/dictionary.
        """
        data = isopy.read_xlsx(filename, sheetname, has_keys=True, keys_in_first=keys_in_first,
                               comment_symbol=comment_symbol, start_at=start_at)
        return cls(data, **kwargs)

    @classmethod
    def from_clipboard(cls, comment_symbol ='#', keys_in_first=None, dialect = None, **kwargs):
        """
        Convert clipboard data to IsopyArray/RefValDict.

        Parameters
        ----------
        comment_symbol : str, Default = '#'
            Rows starting with this string will be ignored.
        has_keys : bool, None
            If True or *keys_in_first* is not None a dictionary will always be returned. If False an nexted list is
            always returned. If None it will return a nested list of all values in the first row and column
            can be converted to/is a float.
        keys_in_first : {'c', 'r', None}
            Where the keys are found. Give 'r' if the keys are found in the first row and 'c' if the
            keys are in first column. If None it will analyse the data to determine where the keys are. If *has_keys*
            is not False an exception will be raised if it cant determine where the keys are.
        dialect
            Dialect of the values in the clipboard. If None the dialect will be guessed from the values.
        **kwargs
            Arguments pass when initialing the array/dictionary.
        """
        data = isopy.read_clipboard(comment_symbol=comment_symbol, has_keys=True, keys_in_first=keys_in_first,
                                    dialect=dialect)
        return cls(data, **kwargs)

    @classmethod
    def from_dataframe(cls, dataframe, **kwargs):
        """
        Convert a pandas dataframe to IsopyArray/RefValDict.

        Parameters
        ----------
        dataframe
            Pandas dataframe to be converted
        **kwargs
            Arguments pass when initialing the array/dictionary.
        """
        return cls(dataframe, **kwargs)


class UFuncMixin:
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

############
### Dict ###
############
def readonly_method(func):
    def decorator(self, *args, **kwargs):
        if self._readonly:
            raise TypeError('This dictionary is readonly. Make a copy to make changes')

        return func(self, *args, **kwargs)
    return decorator


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
    key_flavour
        Will attempt to convert each key into an *key_flavour* key string. If *key_flavour* is a sequence of
        flavours then the first successful conversion is used.  If *key_flavour* is 'any' the flavours
        tried are ``['mass', 'element', 'isotope', 'ratio', 'molecule', 'general']``.
    kwargs : Any, Optional
        Key, Value pairs to be included in the dictionary

    Attributes
    ----------
    readonly
        True if the dictionary is readonly. Otherwise False. Readonly Attribute.
    key_flavour
        The possible flavours of the keys in this dictionary. Readonly attribute.
    default_value
        The default value for the dictionary. An exception will be raised if you try to change the value
        while readonly is true.

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
        d = f'default_value = {self.default_value}'
        r = f'readonly = {self.readonly}'
        k = f'key_flavour = {self.key_flavour}'
        items = '\n'.join([f'"{key}": {value}' for key, value in self.items()])
        return f'{self.__class__.__name__}({d}, {r},\n{k},\n{{{items}}})'

    def __init__(self, *args, default_value = NotGiven, readonly =False, key_flavour = 'any', **kwargs):
        super(IsopyDict, self).__init__()
        self._readonly = False
        self.default_value = default_value
        if key_flavour is NotGiven:
            key_flavour = 'any'
        self._key_flavour = asflavour(key_flavour)

        for arg in args:
            if isinstance(arg, IsopyArray):
                self.update(arg.to_dict())
            if pandas is not None and isinstance(arg, pandas.DataFrame):
                arg = arg.to_dict('list')
                self.update(arg)
            elif isinstance(arg, dict):
                self.update(arg)
            elif isinstance(arg, abc.Iterable):
                arg = dict(arg)
                self.update(arg)
            else:
                raise TypeError(f'arg must be dict not {type(arg)}')

        self.update(kwargs)
        self._readonly = readonly

    @readonly_method
    def __delitem__(self, key):
        key = askeystring(key, flavour=self._key_flavour)
        super(IsopyDict, self).__delitem__(key)

    @readonly_method
    def __setitem__(self, key, value):
        key = askeystring(key, flavour=self._key_flavour)
        value = self._make_value(value, key)
        super(IsopyDict, self).__setitem__(key, value)

    def __contains__(self, key):
        key = askeystring(key, flavour=self._key_flavour)
        return super(IsopyDict, self).__contains__(key)

    def __getitem__(self, key):
        key = askeystring(key, flavour=self._key_flavour)
        return super(IsopyDict, self).__getitem__(key)

    @property
    def keys(self):
        """
        Return a key list with the keys in the dictionary
        """
        return askeylist(super(IsopyDict, self).keys(), flavour=self._key_flavour)

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

    @default_value.setter
    @readonly_method
    def default_value(self, value):
        self._default_value = self._make_default_value(value)

    def _make_value(self, value, key):
        return value

    def _make_default_value(self, value):
        return value

    @property
    def key_flavour(self):
        return self._key_flavour

    @property
    def keylist(self):
        warnings.warn('This attribute has been deprecated. Use keys directly')
        return self.keys

    @readonly_method
    def update(self, other):
        """
        Update the dictionary with the key/value pairs from other, overwriting existing keys.

        A TypeError is raised if the dictionary is readonly.
        """
        if not isinstance(other, dict):
            raise TypeError('other must be a dict')

        for k in other.keys():
            self.__setitem__(k, other[k])

    @readonly_method
    def pop(self, key, default=NotGiven):
        """
        If *key* is in the dictionary, remove it and return its value, else return *default*. If
        *default* is not given the default value of hte dictionary is used.

        A TypeError is raised if the dictionary is readonly.
        """
        if default is NotGiven:
            default = self._default_value

        key = askeystring(key, flavour=self._key_flavour)
        if key in self:
            return super(IsopyDict, self).pop(key)
        elif default is not NotGiven:
            return default
        else:
            raise ValueError('No default value given')

    @readonly_method
    def setdefault(self, key, default=NotGiven):
        """
        If *key* in dictionary, return its value. If not, insert *key* with the default value and
        the default value. If *default* is not given the default value of the dictionary is used.

        A TypeError is raised if the dictionary is readonly and *key* is not in the dictionary.
        """
        key = askeystring(key, flavour=self._key_flavour)
        if default is NotGiven:
            default = self._default_value

        if key not in self:
            if default is NotGiven:
                raise ValueError('No default value given')

            self.__setitem__(key, default)

        return self.__getitem__(key)

    def _copy(self, data, default_value):
        return self.__class__(data,
                              default_value=default_value,
                              key_flavour=self._key_flavour)

    def copy(self, **key_filters):
        """
        Returns a copy of the current dictionary.

        If key filters are given then only the items whose keys pass the key filter is included in the returned
        dictionary.
        """
        if key_filters:
            key_filters = parse_keyfilters(**key_filters)
            keys = [k for k in self if k._filter_(**key_filters)]
            data = {key: self[key] for key in keys}
        else:
            data = self

        return self._copy(data, self._default_value)

    @readonly_method
    def clear(self):
        super(IsopyDict, self).clear()

    def get(self, key=None, default=NotGiven):
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
            key = askeystring(key, flavour=self._key_flavour)
            try:
                return super(IsopyDict, self).__getitem__(key)
            except KeyError:
                if default is NotGiven:
                    raise ValueError('No default value given')
                else:
                    return default

        if isinstance(key, abc.Sequence):
            return tuple(self.get(k, default) for k in key)

        raise TypeError(f'key type {type(key)} not understood')

    def to_dict(self):
        """
        Convert the dictionary to a normal python dictionary.
        """
        return {str(key): self[key] for key in self.keys}


class RefValDict(IsopyDict, ArrayFuncMixin, ToTextMixin, ToFromFileMixin):
    """
    Dictionary where each value is stored as an array of floats by a isopy keystring key.

    Each value in the dictionary has the same ndim and size. If the dictionary has a size of 1 ndim will always be 0.

    Behaves just like, and contains all the methods, that a normal dictionary does unless otherwise
    noted. Only methods that behave differently from a normal dictionary are documented below.

    Parameters
    ----------
    *args : dict[str, scalar], Optional
        Dictionary(ies) where each key can be converted to a keystring.
    default_value : scalar, Default = np.nan
        The default value for a key not present in the dictionary.
    readonly : bool, Default = True
        If ``True`` the dictionary cannot be edited. This attribute is not inherited by child
        dictionaries.
    key_flavour
        Will attempt to convert each key into an *flavour* key string. If *flavour* is a sequence of
        flavours then the first successful conversion is used.  If *flavour* is 'any' the flavours
        tried are ``['mass', 'element', 'isotope', 'ratio', 'molecule', 'general']``.
    ratio_func : callable
        The function that should be used to calculate the value of a missing ratio key string from the data present
        in the array. If None then no attempt is made to calculate the missing value. ``'divide'`` is an
        alias for ``np.divide``.
    molecule_funcs : None or (callable, callable, callable or None)
        A tuple of three functions that should be used to calculate the value of a missing molecule key string from
        the data present in the array. The first function is used to calculate the value for the components,
        the second function for the ``n`` and the final function for the ``charge``. If the third item in the tuple
        is ``None`` then the charge is ignored. If None then no attempt is made to calculate the missing value.
        ``'fraction'`` is an alias for ``(np.multiply, np.multiply, None)``, ``'abundance'`` is an alias for
        ``(np.add, np.multiply, None)`` and ``'mass'`` is an alias for
        ``(np.add, np.multiply, lambda value, charge: np.multiply(value, np.abs(charge)))``
    kwargs : scalar, Optional
        Key, Value pairs to be included in the dictionary

    Attributes
    ----------
    readonly
        True if the dictionary is readonly. Otherwise False. Readonly Attribute.
    key_flavour
        The possible flavours of the keys in this dictionary. Readonly attribute.
    default_value
        The default value for the dictionary. An exception will be raised if you try to change the value
        while readonly is true.
    ndim
        The number of dimensions that each value array in the dictionary.
    size
        The size of each value array in the dictionary.
    ratio_func
        The function used to calculate the value of a missing ratio key string from the data present in the array. If
        None then no attempt is made to calculate the missing value.
    molecule_funcs
        A tuple of three functions used to calculate the value of a missing molecule key string from the data present
        in the array. The first function is used to calculate the value for the components, the second function for
        the ``n`` and the final function for the ``charge``. If the third item in the tuple is None then the charge is
        ignored. If None then no attempt is made to calculate the missing value.

    Examples
    --------
    >>> isopy.RefValDict({'Pd108': 108, '105Pd': 105, 'pd': 46})
    RefValDict(default_value = nan, readonly = False,
    {"108Pd": 108.0
    "105Pd": 105.0
    "Pd": 46.0})

    >>> isopy.RefValDict(Pd108 = 108, pd105= 105, pd=46, default_value=0)
    RefValDict(default_value = 0, readonly = False,
    {"108Pd": 108.0
    "105Pd": 105.0
    "Pd": 46.0})

    """

    def __repr__(self):
        d = f'default_value = {self.default_value}'
        r = f'readonly = {self.readonly}'
        k = f'key_flavour = {self.key_flavour}'
        items = self.to_text(**ARRAY_REPR)
        return f'{self.__class__.__name__}({d}, {r},\n{k}),\n{items}'

    def _repr_markdown_(self):
        return f'{self.to_table()}\n\n**RefValDict**'

    def __init__(self, *args: dict, default_value=nan,
                 readonly= False, key_flavour = 'any', ratio_func = None,
                 molecule_funcs = None, **kwargs):
        # For asrefval
        if default_value is NotGiven:
            default_value = np.nan

        self._size = 1
        self._ndim = 0

        self._ratio_func = ratio_func
        self._molecule_funcs = molecule_funcs

        super(RefValDict, self).__init__(*args, default_value=default_value,
                                         readonly=readonly,
                                         key_flavour= key_flavour,
                                         **kwargs)

    def __getitem__(self, key):
        if isinstance(key, str):
            key = askeystring(key, flavour=self._key_flavour)
            return super(IsopyDict, self).__getitem__(key)
        elif isinstance(key, (int, slice)):
            data = {k: v[key] for k, v in self.items()}
            return self._copy(data, self._default_value[key])
        elif isinstance(key, (list,tuple)):
            if len(key) == 0:
                return self.copy(key_eq=[])
            elif False not in {isinstance(k, str) for k in key}:
                return self.copy(key_eq=key)
            else:
                data = {k: v[key] for k, v in self.items()}
                return self._copy(data, self._default_value[key])
        else:
            return super(IsopyDict, self).__getitem__(key)


    def __add__(self, other):
        return np.add(self, other)

    def __radd__(self, other):
        return np.add(other, self)

    def __sub__(self, other):
        return np.subtract(self, other)

    def __rsub__(self, other):
        return np.subtract(other, self)

    def __mul__(self, other):
        return np.multiply(self, other)

    def __rmul__(self, other):
        return np.multiply(other, self)

    def __truediv__(self, other):
        return np.divide(self, other)
    
    def __rtruediv__(self, other):
        return np.divide(other, self)
    
    def __floordiv__(self, other):
        return np.floor_divide(self, other)
    
    def __rfloordiv__(self, other):
        return np.floor_divide(other, self)
    
    def __pow__(self, power, modulo=None):
        return np.power(self, power)
    
    def __rpow__(self, other):
        return np.power(other, self)

    def _copy(self, data, default_value):
        return self.__class__(data,
                              default_value=default_value,
                              key_flavour=self._key_flavour,
                              ratio_func = self._ratio_func,
                              molecule_funcs=self._molecule_funcs)

    def _make_value(self, value, key, resize = True, default_value = False):
        try:
            value = np.array(value, dtype=np.float64, ndmin=1)
        except Exception as err:
            raise ValueError(f'Key "{key}": Unable to convert value(s) to float') from err

        if value.size == 0:
            raise ValueError(f'Key "{key}": Value has size 0')
        if value.ndim > 1:
            raise ValueError(f'Key "{key}": Value has to many dimensions')

        if self._ndim == 0:
            if value.size > 1:
                if resize:
                    self._ndim = 1
                    self._size = value.size
                    for k, v in tuple(self.items()):
                        super(RefValDict, self).__setitem__(k, np.full(self._size, v))

                    if not default_value:
                        self._default_value = np.full(self._size, self._default_value)
                        self._default_value.setflags(write=False)
                else:
                    raise ValueError(f'Key "{key}": Size of value does not match other values in dictionary')
            else:
                value = value[0]
        else:
            if value.size != 1 and value.size != self._size:
                raise ValueError(f'Key "{key}": Size of value does not match other values in dictionary')
            else:
                value = np.full(self._size, value)

        value.setflags(write=False)
        return value

    def _make_default_value(self, value):
        return self._make_value(value, 'default_value', default_value=True)

    def get(self, key = None, default = NotGiven):
        """
        Return the the value for *key* if present in the dictionary. If *default* is
        not given the default value of the dictionary is used.

        If *key* is a sequence of keys then an array in returned containing the value for each key.

        If *key* is a RatioKeyString and not in the dictionary and the dictionary has a ratio function set
        then that function is used to calculate the

        If *key* is a MoleculeKeyString and not in the dictionary and the dictionary has molecule functions set
        then those used to calculate the value of the molecule.

        Examples
        --------
        >>> reference = RefValDict({'108Pd': 100, '105Pd': 20, '104Pd': 150},
                                    ratio_func=isopy.divide, molecule_funcs=(isopy.multiply, isopy.multiply, None))
        >>> reference.get('pd108')
        100
        >>> reference.get('104Pd/105Pd') # Automatically divides the values
        7.5
        >>> reference.get('(105Pd)2') # Return the product of all the components multiplied by n ignoring any charge
        40
        """
        if isinstance(key, (str, int)):
            key = askeystring(key, flavour=self._key_flavour)
            try:
                return super(IsopyDict, self).__getitem__(key)
            except KeyError:
                if type(key) is RatioKeyString and self._ratio_func is not None:
                    return self._ratio_func(self.get(key.numerator, default), self.get(key.denominator, default))

                if type(key) is MoleculeKeyString and self._molecule_funcs is not None:
                    m = functools.reduce(self._molecule_funcs[0], [self.get(c, default) for c in key.components])
                    m = self._molecule_funcs[1](m, key.n)

                    if  key.charge is not None and self._molecule_funcs[2] is not None:
                        m = self._molecule_funcs[2](m, key.charge)

                    return m

                if default is NotGiven:
                    return self._default_value
                else:
                    return self._make_value(default, 'default', False)

        elif isinstance(key, abc.Sequence):
            keys = askeylist(key, flavour=self._key_flavour)
            return isopy.array([self.get(k, default) for k in keys], keys)
        else:
            return super(RefValDict, self).get(key, default)

    @property
    def ndim(self):
        """
        The number of dimensions for each item in the dictionary.
        """
        return self._ndim

    @property
    def size(self):
        """
        The number of values for each item in the dictionary.
        """
        return self._size

    @property
    def ratio_func(self):
        return self._ratio_func

    @ratio_func.setter
    @readonly_method
    def ratio_func(self, ratio_func):
        if ratio_func == 'divide':
            self._ratio_func = np.divide
        elif ratio_func is None or callable(ratio_func):
            self._ratio_func = ratio_func
        else:
            raise ValueError('ratio_func must be None or a callable')

    @property
    def molecule_funcs(self):
        return self._molecule_funcs

    @molecule_funcs.setter
    @readonly_method
    def molecule_funcs(self, molecule_funcs):
        if molecule_funcs == 'abundance':
            self._molecule_funcs = (np.add, np.multiply, None)
        elif molecule_funcs == 'fraction':
            self._molecule_funcs = (np.multiply, np.multiply, None)
        elif molecule_funcs == 'mass':
            self._molecule_funcs = (np.add, np.multiply, lambda v, c: np.multiply(v, np.abs(c)))
        elif type(molecule_funcs) is tuple(molecule_funcs) and len(molecule_funcs) == 3:
            if not callable(molecule_funcs[0]):
                raise ValueError('molecule_funcs[0] must be callable')
            if not callable(molecule_funcs[1]):
                raise ValueError('molecule_funcs[1] must be callable')
            if not callable(molecule_funcs[2]) and molecule_funcs[2] is not None:
                raise ValueError('molecule_funcs[2] must be callable or None')
            else:
                self._molecule_funcs = molecule_funcs
        elif molecule_funcs is None:
            self._molecule_funcs = None
        else:
            raise ValueError('molecule_funcs must be None or a tuple of callable items')

    def to_list(self):
        """
        Convert the dictionary to a list.
        """
        if self.ndim == 0:
            return list(self.values())
        else:
            return [list(r) for r in zip(*self.values())]

    def to_dict(self):
        """
        Convert the dictionary to a normal python dictionary
        """
        return {str(key): self[key].tolist() for key in self.keys}

    def to_array(self, keys = None, default=NotGiven, **key_filters):
        """
        Convert the dictionary to an IsopyArray.
        If *keys* are given then the array will only contain these keys. If no *keys* are given then the
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

    def to_ndarray(self):
        """
        Convert the dictionary to a normal numpy ndarray.
        """
        return self.to_array().to_ndarray()

    @renamed_function(to_array)
    def asarray(self, keys=None, default=NotGiven, **key_filters):
        pass


ScalarDict = RefValDict # For legacy reasons

def asdict(d, default_value = NotGiven, key_flavour = NotGiven):
    """
    Return *d* if it is an IsopyDict otherwise convert *d* into one and return it.

    The returned IsopyDict will have the specified default value and key flavour(s) if these are given.
    """
    if key_flavour is not NotGiven: key_flavour = asflavour(key_flavour)

    if type(d) == str:
        d = isopy.refval(d)

    if (isopy.isdict(d)
            and (default_value is NotGiven or d.default_value == default_value)
            and (key_flavour is NotGiven or d.key_flavour == key_flavour)):
        return d
    else:
        return IsopyDict(d, default_value = default_value, key_flavour = key_flavour)


def asrefval(d, default_value = NotGiven, key_flavour = NotGiven, ratio_func = None, molecule_funcs = None):
    """
    Return *d* if it is an RefValDict otherwise convert *d* into one and return it.

    The returned RefValDict will have the specified default value and key flavour(s) if these are given.
    """
    if key_flavour is not NotGiven:
        key_flavour = asflavour(key_flavour)

    if type(d) == str:
        d = isopy.refval(d)

    if (isopy.isrefval(d)
            and (default_value is NotGiven or d.default_value == default_value)
            and (key_flavour is NotGiven or d.key_flavour == key_flavour)):
        return d
    else:
        return RefValDict(d, default_value = default_value, key_flavour = key_flavour)


#############
### Array ###
#############
class IsopyArray(ArrayFuncMixin, ToTextMixin, ToFromFileMixin, UFuncMixin):
    """
    An array where data is stored rows and columns of isopy key strings.

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

    __default__ = nan

    def __new__(cls, values, keys=None, *, dtype=None, ndim=None, flavour = None):
        flavour = asflavour(flavour)

        if type(values) is IsopyNdarray and keys is None and dtype is None and ndim is None:
            return values.copy()

        # Do this early so no time is wasted if it fails
        if keys is None and (isinstance(dtype, np.dtype) and dtype.names is not None):
            keys = askeylist(dtype.names, allow_duplicates=False, flavour=flavour)

        if isinstance(keys, np.dtype):
            if not keys.names:
                raise ValueError('dtype does not contain named fields')

            if dtype is None: dtype = keys
            keys = askeylist(keys.names, allow_duplicates=False, flavour=flavour)

        if ndim is not None and (not isinstance(ndim, int) or ndim < -1 or ndim > 1):
            raise ValueError('parameter "ndim" must be -1, 0 or 1')

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
                    if values.ndim == 0:
                        dtype = [(values.dtype,)]
                    else:
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
            # Leave as is
            pass

        else:
            raise ValueError(f'unable to convert values with type "{type(values)}" to IsopyArray')

        if keys is None:
            # IF there are no keys at this stage raise an error
            raise ValueError('Keys argument not given and keys not found in values')
        else:
            keys = askeylist(keys, allow_duplicates=False, flavour=flavour)

        if isinstance(values, tuple):
            vlen = len(values)
        else:
            try:
                vlen = {len(v) for v in values}
            except:
                raise
            if len(vlen) == 0:
                raise ValueError('Cannot create an empty array')
            if len(vlen) != 1:
                raise ValueError('All rows in values are not the same size')
            vlen = vlen.pop()

        if vlen != len(keys):
            raise ValueError('size of keys does not match size of values')

        if dtype is None:
            new_dtype = [(float64, None) for k in keys]

        elif isinstance(dtype, list) or (isinstance(dtype, np.dtype) and dtype.names is not None):
            if len(dtype) != vlen:
                raise ValueError('number of dtypes given does not match number of keys')
            else:
                new_dtype = [(dtype[i],) if not isinstance(dtype[i], tuple) else dtype[i] for i in range(vlen)]

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
                try:
                    dtype.append(np.asarray(v, dtype=dt).dtype)
                except:
                    pass
                else:
                    break
            else:
                raise ValueError(f'Unable to convert values for {keys[i]} to one of the specified dtypes')

        out = np.array(values, dtype=list(zip(keys.strlist(), dtype)))
        if ndim == -1:
            if out.size == 1:
                ndim = 0
            else:
                ndim = 1

        if ndim is not None and ndim != out.ndim:
            if ndim == 1:
                out = out.reshape(-1)
            elif ndim == 0 and out.size != 1:
                raise ValueError(f'Cannot convert array with {out.size} rows to 0-dimensions')
            else:
                out = out.reshape(tuple())

        return keys._view_array_(out)

    def __repr__(self):
        return self.to_text(**ARRAY_REPR)

    def _repr_markdown_(self):
        return self.to_table(include_row=True)

    def __str__(self):
        return self.to_text(', ', False, False)

    def __eq__(self, other):
        other = np.asanyarray(other)

        if self.shape == other.shape:
            return np.all(np.array_equal(self, other, equal_nan=True), axis=None)
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def _parse_index_(self, index):
        if isinstance(index, IsopyKeyString) or type(index) is IsopyKeyList:
            return index, None

        elif isinstance(index, str):
            return askeystring(index, flavour = self.flavour), None

        elif type(index) is slice and index.start is None and index.stop is None and index.step is None:
            # So that array[:] works even for voids
            return None, None

        elif self.ndim == 0 and (type(index) is int or type(index) is slice):
            raise IndexError('0-dimensional arrays cannot be indexed by row')

        elif isinstance(index, (list, tuple)) and len(index) > 0:
            if False not in {isinstance(k, str) for k in index}:
                return askeylist(index, flavour = self.flavour), None

            elif self.ndim == 0 and False not in {type(k) is int for k in index}:
                raise IndexError('0-dimensional arrays cannot be indexed by row')

        if type(index) is tuple:
            if (klen :=len(index)) == 0:
                return self.keys, None

            elif klen == 1:
                keyi = self._parse_index_(index[0])
                if keyi[1] is not None:
                    raise IndexError(f'Invalid index for keys {index[0]}')
                else:
                    return keyi[0], None

            elif klen == 2:
                keyi = self._parse_index_(index[0])
                if keyi[1] is not None:
                    raise IndexError(f'Invalid index for keys {index[0]}')

                rowi = self._parse_index_(index[1])
                if rowi[0] is not None:
                    raise IndexError(f'Invalid index for rows {index[1]}')

                return keyi[0], rowi[1]

            else:
                raise IndexError('')

        return None, index

    def __getitem__(self, index):
        keyi, rowi = self._parse_index_(index)

        if keyi is None:
            a = self
        else:
            a = keyi._view_array_(super(IsopyArray, self).__getitem__(keyi._str_()))

        if rowi is None:
            return a
        elif rowi == []:
            return np.array([]) #Isopy array cannot contain 0 rows
        elif isinstance(a, IsopyArray):
            return a.keys._view_array_(super(IsopyArray, a).__getitem__(rowi))
        else:
            return a.keys._view_array_(a[rowi])

    def __setitem__(self, index, value):
        keyi, rowi = self._parse_index_(index)

        if isinstance(value, dict) and not isinstance(value, IsopyDict):
            value = isopy.asdict(value, default_value=np.nan)

        if not isinstance(value, (IsopyArray, IsopyDict)):
            value = np.asarray(value)
            if self.ndim == 0 and value.ndim != 0 and value.size == 1:
                value = value.reshape(tuple())
            value = KeylessValue(value)

        if keyi is None:
            a = self
        elif type(keystr:=keyi._str_()) is list:
            a = keyi._view_array_(super(IsopyArray, self).__getitem__(keystr))
        elif rowi is None: # True of key index is a single key and no rows index was given
            super(IsopyArray, self).__setitem__(keystr, value.get(keyi))
            return
        else:
            super(IsopyArray, self).__getitem__(keystr).view(ndarray)[rowi] = value.get(keyi)
            return

        if rowi is None:
            for key in a.keys:
                super(IsopyArray, a).__setitem__(str(key), value.get(key))
            return
        elif isinstance(rowi, IsopyArray):
            for k in rowi.keys:
                a.get(k)[rowi[k]] = value.get(k)
        else:
            for key in a.keys:
                super(IsopyArray, a).__getitem__(str(key)).view(ndarray)[rowi] = value.get(key)
            return
        
    def __len__(self):
        if self.ndim == 0: #For voids
            raise TypeError('len() of unsized object')
        return super(IsopyArray, self).__len__()

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

    def values(self):
        """
        Returns a tuple containing the column values for each key in the array

        Equivalent to ``tuple(array[key] for key in array.keys)``
        """
        return tuple(self[key] for key in self.keys)

    def items(self):
        """
        Returns a tuple containing a tuple with the key and the column values for each key in the array

        Equivalent to ``tuple([)(key, array[key]) for key in array.keys)``
        """
        return tuple((key, self[key]) for key in self.keys)

    def get(self, key, default = NotGiven):
        """
        Returns the values of column *key* if present in the array. Otherwise an numpy array
        filled with *default* is returned with the same shape as a column in the array. An
        exception will be raised if *default* cannot be broadcast to the correct shape.

        If *default* is not given np.nan is used.
        """
        if default is NotGiven:
            default = self.__default__ #This is always nan unless the .default() has been called.

        try:
            key = askeystring(key, flavour=self.flavour)
            return self.__getitem__(key)
        except:
            return np.full(self.shape, default)

    def copy(self, **key_filters):
        """
        Returns a copy of the array. If *key_filters* are given then the returned array only
        contains the columns that satisfy the *key_filter* filters.
        """
        if key_filters:
            copy =  self.filter(**key_filters)
            return copy.keys._view_array_(copy.copy())
        else:
            copy =  super(IsopyArray, self).copy()
            return self.keys._view_array_(copy)

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

        return IsopyArray(self[keys] / self[denominator], keys=keys/denominator, flavour='ratio')

    def deratio(self, denominator_value=1):
        """
        Return a array with the numerators and the common denominator as columns. Values for the
        numerators will be copied from the original array and the entire array will be multiplied by
        *denominator_value*.

        An exception is raised if the array flavour is not 'ratio' or if the array does not
        contain a common denominator.
        """
        if self.flavour != 'ratio':
            raise TypeError('This method only works when all column keys are ratio key strings')

        denominator = self.keys().common_denominator
        if denominator is None:
            raise ValueError('Column keys do not have a common denominator')
        numerators = self.keys().numerators

        out = IsopyArray(self, numerators, flavour=numerators.flavour)
        if denominator not in out.keys():
            out = isopy.cstack(out, ones(self.nrows, denominator))
        return out * denominator_value

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

    def reshape(self, shape):
        return self.keys._view_array_(super(IsopyArray, self).reshape(shape))

    def default(self, value):
        """
        Return a view of the array with a **temporary** default value.

        This should only be used directly in conjuction with mathematical expressions. Any new view created
        will not inherit the default value.

        Examples
        --------
        >>> a1 = isopy.array([1,2,3], 'ru pd cd'.split())
        >>> a2 = isopy.array([20, 10], 'ru cd'.split())
        >>> a1 + a2 # Default value is np.nan
        (row) , Ru , Pd  , Cd
        None  , 21 , nan , 13

        >>> a1 + a2.default(0) # a2 has a temporary default value of 0
        (row) , Ru , Pd , Cd
        None  , 21 , 2  , 13
        """
        temp_view = self.keys._view_array_(self)
        temp_view.__default__ = value
        return temp_view

    def to_list(self):
        """
        Convert the array to a list of values
        """
        if self.ndim == 0:
            return list(self.tolist())
        else:
            return [list(row) for row in self.tolist()]

    def to_dict(self):
        """
        Convert the array to a normal python dictionary.
        """
        return {str(key): self[key].tolist() for key in self.keys()}

    def to_refval(self, default_value=NotGiven):
        """
        Convert the array to a RefValDict.
        """
        if default_value is NotGiven:
            default_value = self.__default__

        return RefValDict(self, default_value=default_value)

    def to_ndarray(self):
        """
        Convert the array to a numpy ndarray.
        """
        if isinstance(self, void):
            view = self.view((np.void, self.dtype))
        else:
            view = self.view(ndarray)
        return view.copy()


class IsopyNdarray(IsopyArray, ndarray):
    pass


class IsopyVoid(IsopyArray, void):
    pass


def array(values=None, keys=None, *, dtype=None, ndim=None, flavour='any', **columns_or_read_kwargs):
    """
    Convert the input arguments to a isopy array.

    If *values* is a string it assumes it is a filename and will load the contents of the file together
    with *columns_or_read_kwargs*. If *values* is 'clipboard' it will read values from the clipboard.
    If *columns_or_read_kwargs* in *read_kwargs* it assumes *a* is an excel file. Otherwise it
    assumes *values* is a CSV file.

    Will attempt to convert the input into an *flavour* array. If *flavour* is a sequence of
    flavours then the first successful conversion is returned.  If *flavour* is 'any' the flavours
    tried are ``['mass', 'element', 'isotope', 'ratio', 'mixed', 'molecule', 'general']``.
    """
    if isinstance(values, (str, bytes, io.StringIO, io.BytesIO)):
        if values == 'clipboard':
            values = isopy.read_clipboard(**columns_or_read_kwargs)
        elif 'sheetname' in columns_or_read_kwargs:
            values = isopy.read_xlsx(values, **columns_or_read_kwargs)
        else:
            values = isopy.read_csv(values, **columns_or_read_kwargs)
        columns_or_read_kwargs = dict()

    if values is None and len(columns_or_read_kwargs) == 0:
        raise ValueError('No values were given')
    elif values is not None and len(columns_or_read_kwargs) != 0:
        raise ValueError('values and column kwargs cannot be given together')
    elif values is None:
        values = columns_or_read_kwargs
    
    return IsopyArray(values, keys=keys, dtype=dtype, ndim=ndim, flavour=flavour)


def asarray(a, *, ndim = None, flavour = None, **read_kwargs):
    """
    If *a* is an isopy array return it otherwise convert *a* into an isopy array and return it. If
    *ndim* is given a view of the array with the specified dimensionality is returned.

    If *a* is a string it assumes it is a filename and will load the contents of the file together with *read_kwargs*.
    If *a* is 'clipboard' it will read values from the clipboard. If *sheetname* in *read_kwargs* it assumes *a*
    is an excel file. Otherwise it assumes *a* is a CSV file.

    Will attempt to convert the input into an *flavour* array. If *flavour* is a sequence of
    flavours then the first successful conversion is returned.  If *flavour* is 'any' the flavours
    tried are ``['mass', 'element', 'isotope', 'ratio', 'mixed', 'molecule', 'general']``.
    """
    flavour = asflavour(flavour)

    if isinstance(a, (str, bytes, io.StringIO, io.BytesIO)):
        if a == 'clipboard':
            a = isopy.read_clipboard(**read_kwargs)
        elif 'sheetname' in read_kwargs:
            a = isopy.read_xlsx(a, **read_kwargs)
        else:
            a = isopy.read_csv(a, **read_kwargs)

    if not isinstance(a, IsopyArray) or a.flavour not in flavour:
        a = array(a, flavour=flavour)

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


def asanyarray(a, *, dtype = None, ndim = None, flavour=None, **read_kwargs):
    """
    Return ``isopy.asarray(a)`` if *a* possible otherwise return ``numpy.asanyarray(a)``.

    The data type and number of dimensions of the returned array can be specified by *dtype and
    *ndim*, respectively.

    If *a* is a string it assumes it is a filename and will load the contents of the file together with *read_kwargs*.
    If *a* is 'clipboard' it will read values from the clipboard. If *sheetname* in *read_kwargs* it assumes *a*
    is an excel file. Otherwise it assumes *a* is a CSV file.

    Will attempt to convert the input into an *flavour* array. If *flavour* is a sequence of
    flavours then the first successful conversion is returned.  If *flavour* is 'any' the flavours
    tried are ``['mass', 'element', 'isotope', 'ratio', 'mixed', 'molecule', 'general']``.
    """
    if isinstance(a, (str, bytes, io.StringIO, io.BytesIO)):
        if a == 'clipboard':
            a = isopy.read_clipboard(**read_kwargs)
        elif 'sheetname' in read_kwargs:
            a = isopy.read_xlsx(a, **read_kwargs)
        else:
            a = isopy.read_csv(a, **read_kwargs)

    if isinstance(a, IsopyArray) and dtype is None:
        return asarray(a, ndim=ndim, flavour=flavour)

    if isinstance(a, dict) or (isinstance(a, ndarray) and a.dtype.names is not None):
        return array(a, dtype=dtype, ndim=ndim, flavour=flavour)

    else:
        if not (dtype is None and isinstance(a, np.ndarray)):
            if dtype is None:
                dtype = (float64, None)
            elif type(dtype) is not tuple:
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

###########################
### Create empty arrays ###
###########################
def zeros(rows, keys=None, *, ndim=None, dtype=None, flavour = 'any'):
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
    return new_empty_array(rows, keys, ndim, dtype, np.zeros, flavour)


def ones(rows, keys=None, *, ndim=None, dtype=None, flavour = 'any'):
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
    return new_empty_array(rows, keys, ndim, dtype, np.ones, flavour)


def empty(rows, keys=None, *, ndim=None, dtype=None, flavour = 'any'):
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
    return new_empty_array(rows, keys, ndim, dtype, np.empty, flavour)


def full(rows, fill_value, keys=None, *, ndim = None, dtype=None, flavour = 'any'):
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
    if keys is None and isinstance(fill_value, IsopyArray):
        keys = fill_value.keys()

    out = new_empty_array(rows, keys, ndim, dtype, np.empty, flavour)
    np.copyto(out, fill_value)
    return out


def random(rows, random_args = None, keys=None, *, distribution='normal', seed=None, ndim = None,
           dtype=None, flavour = 'any'):
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
    array = empty(rows, keys=keys, ndim=ndim, dtype=dtype, flavour = flavour)

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


def new_empty_array(rows, keys, ndim, dtype, func, flavour):
    if isinstance(rows, IsopyArray):
        if keys is None:
            keys = rows.keys
        if dtype is None:
            dtype = rows.dtype
        rows = rows.nrows

    if isinstance(keys, IsopyArray):
        if dtype is None:
            dtype = keys.dtype
        keys = keys.keys

    if isinstance(dtype, IsopyArray):
        dtype = dtype.dtype

    if dtype is None:
        dtype = float64

    if rows is None:
        rows = -1
    elif rows == tuple():
        rows = -1
    elif rows == -1:
        pass
    elif rows < 1:
        raise ValueError('parameter "rows" must be -1 or a positive integer')

    if keys is not None:
        keys = askeylist(keys, allow_duplicates=False, flavour = flavour)

    if ndim is None:
        if rows == -1:
            shape = None
        else:
            shape = rows
    elif ndim == -1:
        if abs(rows) == 1:
            shape = None
        else:
            shape = rows
    elif ndim == 1:
        shape = abs(rows)
    elif ndim == 0:
        if abs(rows)  == 1:
            shape = None
        else:
            raise TypeError('cannot create an zero-dimensional array with more than one row')
    elif ndim < -1 or ndim > 1:
        raise ValueError('accepted values for "ndim" is -1, 0,  or 1')

    if shape is None:
        shape = ()

    if isinstance(dtype, np.dtype) and dtype.names is not None:
        if keys is None:
            keys = keylist(dtype.names, allow_duplicates=False)
        elif len(keys) != len(dtype):
            raise ValueError('size of dtype does not match size of keys')
        dtype = [dtype[i] for i in range(len(dtype))]

    elif isinstance(dtype, abc.Iterable):
        if keys is None:
            raise ValueError('dtype is an iterable but no keys were given')
        elif (len(keys) != len(dtype)):
            raise ValueError('size of dtype does not match size of keys')

    elif keys is not None:
        dtype = [dtype for i in range(len(keys))]

    if keys is not None:
        dtype = list(zip(keys.strlist(), dtype))
        return keys._view_array_(func(shape, dtype=dtype))
    else:
        return func(shape, dtype=dtype)


###############################################
### numpy function overrides via            ###
###__array_ufunc__ and __array_function__   ###
###############################################
APPROVED_FUNCTIONS = []

@functools.lru_cache(maxsize=CACHE_MAXSIZE)
def function_signature(func):
    #returns the signature and number of input arguments for a function
    #An input argument is given as any argument that does not have a default value
    try:
        parameters = inspect.signature(func).parameters
    except:
        #No signature avaliable. Should be a rare occurance but applies to for example
        # np.concatenate (which doesnt work with isopy arrays)
        #This return assumes all args are inputs
        return None, None
    else:
        return tuple(parameters.keys()), \
            len([p for p in parameters.values() if (p.default is p.empty)])

class KeylessValue:
    """
    Returns the same array for all calls to get. Used for array function
     inputs that are not isopy arrays or dicts.
    """
    __slots__ = ['value']
    def __init__(self, value):
        self.value = value

    def get(self, *_):
        return self.value

@functools.lru_cache(maxsize=CACHE_MAXSIZE)
def find_keys(*keys):
    new_keys = keys[0]
    for key in keys[1:]:
        new_keys = new_keys | key

    return new_keys

# Simplify to that there is one where we know there is only one input and one where we know there is
# More than two inputs and this one for unknown cases.
# This is for functions that dont have a return value
def call_array_function(func, *inputs, axis=0, keys=None, **kwargs):
    """
    Used to call numpy ufuncs/functions on isopy arrays and/or scalar dicts.

    This function produces the expected result for the majority, but not all, numpy functions.
    With a few exceptions all calls to numpy functions on isopy arrays will be sent to this function
     via the __array_ufunc__ and __array_function__ interface.

    If axis == 0 then the function is called on each column present in the isopy arrays given as
    input. Isopy arrays given as kwargs are not used taken into consideration when determining the
    returned array. Analogous to ``isopy.array({key: func(array.get(key, default_value), **kwargs)}
    for key in array.keys())`` for a single isopy array input.

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
    """

    # Try to sort input from optional arguments.
    fargs, nin = function_signature(func)
    if fargs and len(inputs) != nin:
        kwargs.update(zip(fargs, inputs))
        try:
            inputs = tuple(kwargs.pop(fargs[i]) for i in range(nin))
        except:
            # I dont think this would ever happen as it would be caught by the
            # array function dispatcher.
            raise ValueError(
                f'"{func.__name__}" expects {nin} input arguments but only got {len(inputs)}')

    if fargs and 'axis' in fargs:
        if 'axis' in kwargs:
            axis = kwargs.pop('axis')
        if axis == 0:
            kwargs['axis'] = None

    new_inputs = []
    new_keys_a = []
    new_keys_d = []
    d_input = []
    for i, arg in enumerate(inputs):
        if isinstance(arg, IsopyArray):
            new_inputs.append(arg)
            new_keys_a.append(arg.keys())
        elif isinstance(arg, ndarray) and len(arg.dtype) > 1:
            new_inputs.append(asarray(arg))
            new_keys_a.append(new_inputs[-1].keys())
        elif isinstance(arg, RefValDict):
            d_input.append(arg)
            new_inputs.append(arg)
            new_keys_d.append(arg.keys())
        elif isinstance(arg, dict):
            new_inputs.append(RefValDict(arg))
            new_keys_d.append(new_inputs[-1].keys())
        else:
            new_inputs.append(KeylessValue(np.asarray(arg)))

    if keys is None:
        if len(new_keys_a) != 0:
            new_keys = find_keys(*new_keys_a)
        elif len(new_keys_d) != 0:
            new_keys = find_keys(*new_keys_d)
        else:
            return func(*inputs, **kwargs)
    else:
        new_keys = isopy.askeylist(keys)

    if axis == 0:
        out = kwargs.get('out', None)
        keykwargs = {kwk: kwargs.pop(kwk) for kwk, kwv in tuple(kwargs.items())
                                if isinstance(kwv, (IsopyArray, IsopyDict))}

        result = [func(*(input.get(key) for input in new_inputs), **kwargs,
                       **{k: v.get(key) for k, v in keykwargs.items()}) for key in new_keys]

        #There is no return from the function so dont return an array
        if False not in [r is None for r in result]:
            return None

        if out is None:
            if len(new_keys_a) != 0:
                dtype = [(str(key), getattr(result[i], 'dtype', float64)) for i, key in enumerate(new_keys)]
                if getattr(result[0], 'ndim', 0) == 0:
                    out = np.array(tuple(result), dtype=dtype)
                else:
                    out = np.fromiter(zip(*result), dtype=dtype)
                return new_keys._view_array_(out)
            else:
                if len(d_input) == 1:
                    dv = d_input[0].default_value
                    if dv.size > 1 and result[0].size != dv.size:
                        if np.all(dv==dv[0]):
                            dv = dv[0]
                        else:
                            dv = np.nan
                    # The return dictionary inherits the properties of the original dictionary
                    return d_input[0]._copy(zip(new_keys, result), dv)
                else:
                    return RefValDict(zip(new_keys, result))
        else:
            return out

    else:
        for kwk, kwv in list(kwargs.items()):
            if isinstance(kwv, (IsopyArray, IsopyDict)):
                kwargs[kwk] = np.transpose([kwv.get(key) for key in new_keys])

        new_inputs = [np.transpose([input.get(key) for key in new_keys]) if
                      not isinstance(input, KeylessValue) else input.get() for input in new_inputs]

        if len(nd :={inp.ndim for inp in new_inputs}) == 1 and nd.pop() == 1:
            #So that 0-dimensional arrays get do not raise axis out of bounds error
            axis = 0

        return func(*new_inputs, axis = axis, **kwargs)

