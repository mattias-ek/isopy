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
from tabulate import tabulate as tabulate_

#optional imports
try:
    import pandas
except:
    pandas = None

try:
    import IPython
except:
    IPython = None

from numpy import ndarray, nan, float64, void


class NotGivenType:
    def __repr__(self):
        return 'Optional' # So that it looks like a None in the docs

    def __bool__(self):
        return False

NotGiven = NotGivenType()

ARRAY_REPR = dict(include_row=True, include_dtype=True, nrows=10, floatfmt='.5f', include_objinfo = True)
ARRAY_STR = dict()
IPYTHON_REPR = True


__all__ = ['iskeystring', 'iskeylist', 'isarray', 'isdict', 'isrefval',
           'asflavour',
           'keystring', 'askeystring',
           'keylist', 'askeylist',
           'array', 'asarray', 'asanyarray',
           'asdict', 'asrefval', 'IsopyDict', 'RefValDict', 'ScalarDict',
           'ones', 'zeros', 'empty', 'full', 'random']

CACHE_MAXSIZE = 128
CACHES_ENABLED = True

def lru_cache(maxsize=128):
    """
    decorator for functools.lru_cache but will call the uncached function when an unhashable value
    is encountered.
    """
    def lru_cache_decorator(func):
        cached = functools.lru_cache(maxsize)(func)
        uncached = func
        @functools.wraps(func)
        def lru_cache_wrapper(*args, **kwargs):
            if CACHES_ENABLED:
                try:
                    return cached(*args, **kwargs)
                except TypeError as err:
                    if not startswith(str(err), 'unhashable'):
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
    if name is None:
        return str(item)

    module = getattr(item, '__module__', None)
    if module is not None:
        if module == 'numpy':
            module = 'np'
        name = f'{module}.{name}'

    return name

def get_classname(thing):
    if isinstance(thing, type):
        return thing.__name__
    else:
        return thing.__class__.__name__

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
    new_func.__repr__ = lambda: f'<function {new_func.__name__}>'
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

    def __add__(self, other):
        return ListFlavour((self, other))

    def __radd__(self, other):
        # Order doesnt matter since its sorted
        return self.__add__(other)

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
            elif isinstance(flavour, cls):
                return flavour._flavours
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

    def __add__(self, other):
        return self.__class__((self, other))

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
    """
    Convert *flavour* into a flavour object.

    Each flavour is represented by a lower case string: mass, element, isotope, molecule, ratio, and general.

    You can specify multiple flavours using ``|`` e.g ``element|isotope``.

    You can specify the allowed flavour of components in molecule key string using ``molecule[<flavour>]``. Only
    ``element`` and ``isotope`` are valid sub-flavours of molecules. If the square brackets are omitted it defaults
    to ``any``.

    You can specify the flavour of the numerator and denominator for a ratio using
    ``ratio[<numerator_flavour>,<denominator_flavour>]``. If you specify only one flavour inside the brackets it
    will be used for both the numerator and the denominator. If the square brackets are omitted then
    the numerator and denominator flavour defaults to ``any``.
    """
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

class KeyFlavourError(KeyParseError, TypeError):
    def __init__(self, key, flavour):
        self.cls = key.__class__
        self.flavour = flavour

    def __str__(self):
        return f'{self.cls}: Key not compatible with flavour {self.flavour}'

KEY_COMPARISONS = ('eq', 'neq', 'lt', 'gt', 'le', 'ge')
class IsopyKeyString(str):
    """
    Key strings should be created using :py:func:`isopy.keystring` and :py:func:`isopy.askeystring` functions and not
    by invoking the key string objects directly.

    Key strings are a subclass of :class:`str` and therefore contains all the method that a :class:`str` does.
    Unless specifically noted below these methods will return a :class:`str` rather than a key string.
    """
    def __repr__(self):
        return f"{self.__class__.__name__}('{self}')"

    def _repr_latex_(self):
        if IPYTHON_REPR:
            return fr'$${self.str("math")}$$'
        else:
            return None

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

            if attr is NotImplemented or comparison not in KEY_COMPARISONS:
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
    MassKeyString()

    String representation of a mass number.

    Attributes
    ----------
    flavour
        The flavour of the key string.
    mass_number : MassKeyString
        A reference to itself.


    Examples
    --------
    >>> isopy.keystring('76')
    '76'
    >>> isopy.keystring(76)
    '76'

    Mass key strings also support the ``<,> <=, >=`` operators:

    >>> isopy.keystring('76') > 75
    True
    >>> isopy.keystring('76') <= 75
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

        key = super(MassKeyString, cls).__new__(cls, string, flavour = MassFlavour())
        object.__setattr__(key, 'mass_number', key)
        return key

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
        >>> key = isopy.keystring('101')
        >>> key.str()
        '101'
        >>> key.str('key is "{m}"')
        'key is "101"'
        """
        return super(MassKeyString, self).str(format)

    def _str_options_(self):
        return dict(m = str(self), key = str(self),
                    math = fr'{self}',
                    latex = fr'${self}$')


class ElementKeyString(IsopyKeyString):
    """
    ElementKeyString()

    A string representation of an element symbol limited to two letters.

    The first letter is in upper case and subsequent letters are in lower case.

    Attributes
    ----------
    flavour
        The flavour of the key string.
    element_symbol : ElementKeyString
        A reference to itself.
    isotopes : IsotopeKeyList
        A key list of all the present day naturally occuring isotopes of this element.


    Examples
    --------
    >>> isopy.keystring('Pd')
    'Pd'
    >>> isopy.keystring('pd')
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
        return askeylist(isopy.refval.element.isotopes.get(self, []))


class IsotopeKeyString(IsopyKeyString):
    """
    IsotopeKeyString()

    A string representation of an isotope consisting of a mass number followed by an element symbol.

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
    mz : float
        The mass to charge ratio of this isotope on the basis of the mass number.

    Examples
    --------
    >>> isopy.keystring('Pd104')
    '104Pd'
    >>> isopy.keystring('104pd')
    '104Pd'

    ``in`` can be used to test if a string is equal to the mass number or an element symbol of an
    isotope key string.

    >>> 'pd' in isopy.keystring('Pd104')
    True
    >>> 104 in isopy.keystring('Pd104')
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
        >>> key = isopy.keystring('101ru')
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
                       latex = fr'${{}}^{{{self.mass_number}}}\mathrm{{{self.element_symbol}}}$'))

        product = list(itertools.product(mass_options.items(), element_options.items()))
        options.update({f'{mk}{ek}': f'{mv}{ev}' for (mk, mv), (ek, ev) in product})
        options.update({f'{mk}-{ek}': f'{mv}-{ev}' for (mk, mv), (ek, ev) in product})
        options.update({f'{ek}{mk}': f'{ev}{mv}' for (mk, mv), (ek, ev) in product})
        options.update({f'{ek}-{mk}': f'{ev}-{mv}' for (mk, mv), (ek, ev) in product})

        return options

    @property
    def isotopes(self):
        return askeylist(self)


class MoleculeKeyString(IsopyKeyString):
    """
    MoleculeKeyString()

    A string representation of an molecue consisting of a element and/or isotope key strings.

    Mass numbers must be before the element symbols. Any number after the element symbol is
    assumed to be a multiple. Capital letters signify a new element symbol and must be used
    when listing succesive element symbols. Parenthesis, or square brackets, can be used to group elements or to
    seperate mass numbers from multipliers. Multiple + and - signs are used to signify

    Molecule keys strings with more than one component is enclosed in square brackets.
    Isotope molecules are enclosed in square brackets if there is more than one component in the molecule.

    Attributes
    ----------
    flavour
        The flavour of the key string.
    element_symbol : MoleculeKeyList
        A molecule key string containing the element formula for this molecule.
    isotopes : MoleculeKeyList
        A molecule key string containing all the isotopes for this molecule.
    mz : float
        The mass to charge ratio for each molecule in the list on the basis of the mass number.
        Negative charges will return a positive number.
    components : tuple
        The components of this molecule.
    n : int
        The multiplier of this molecule.
    charge : None, int
        The charge of this molecule.

    Examples
    --------
    >>> isopy.keystring('H2O')
    '[H2O]'
    >>> isopy.keystring('(1H)2(16O)')
    '[[1H]2[16O]]'
    >>> isopy.keystring('137Ba++')
    '137Ba++'
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
            return f'C{self.mz:0>8.3f}{self._z_():0>4}'
        else:
            return f'C{0:0>8.3f}{self._z_():0>4}'

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

        return tuple(self._new(mol, 1, self.charge) for mol in all)

    @property
    def isotopes(self):
        return askeylist(self._isotopes_(None))


class RatioKeyString(IsopyKeyString):
    """
    RatioKeyString()

    A string representation of a ratio of two key strings.

    A string must consist of a numerator and denominator seperated by "/". The numerator and denominator
    key strings can be of different flavours. Nested ratios can be created using a combination of
    "/", "//", "///" etc upto a maximum of 9 nested ratios.

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
    >>> isopy.keystring('Pd108/105pd')
    '104Pd/108Pd'
    >>> isopy.keystring('Pd108/105pd//ge')
    '108Pd/105Pd//Ge

    ``in`` can be used to test if a string is equal to the numerator or denominator of the ratio.

    >>> 'pd108' in isopy.keystring('108Pd/Ge')
    True
    >>> 'as/ge' in isopy.keystring('Pd108/105pd//as/ge')
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
            raise KeyValueError(cls, string, 'Limit of nested ratios reached')

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

        if format is None and nformat is None and dformat is None:
            return str(self)

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

        options['math'] = fr'\cfrac{{{nmath}}}{{{dmath}}}'
        options['latex'] = fr'${options["math"]}$'
        return options

    def _flatten_(self):
        return self.numerator._flatten_() + self.denominator._flatten_()


class GeneralKeyString(IsopyKeyString):
    """
    GeneralKeyString()

    A general key string that can hold any string value.

    No formatting is applied to the string.

    Attributes
    ----------
    flavour
        The flavour of this key string.


    Examples
    --------
    >>> isopy.keystring('harry')
    'harry'
    >>> isopy.keystring('pd', flavour='general')
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
                    math = fr'\mathrm{{{self}}}'.replace(' ', '\ '),
                    latex = fr'$\mathrm{{{self}}}$'.replace(' ', '\ '))


def iskeystring(item, *, flavour = None, flavour_in = None) -> bool:
    """
    Returns ``True`` if the supplied string is an key string otherwise returns ``False``.

    Parameters
    ----------
    item
        The string to be verified.
    flavour :  flavour_like, Optional
        If given then ``True`` is returned if the flavour of *item* is equal to *flavour*.
    flavour_in : flavour_like Optional
        If given then then ``True`` is returned if the flavour of *item* is found in *flavour_in*.

    Examples
    --------
    >>> isopy.iskeystring('Pd')
    False

    >>> key = isopy.keystring('pd')
    >>> isopy.iskeystring(key)
    True
    >>> isopy.iskeystring(key, flavour='isotope')
    False
    >>> isopy.iskeystring(key, flavour_in='element|isotope')
    True
    """
    if flavour is not None:
        return isinstance(item, IsopyKeyString) and item.flavour == asflavour(flavour)
    elif flavour_in is not None:
        return isinstance(item, IsopyKeyString) and item.flavour in asflavour(flavour_in)
    else:
        return isinstance(item, IsopyKeyString)

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

    Examples
    --------
    >>> isopy.keystring('pd')
    ElementKeyString('Pd')
    >>> isopy.keystring('pd', allow_reformatting=False)
    GeneralKeyString('pd')

    >>> key = isopy.keystring('pd', flavour = 'general'); key
    GeneralKeyString('pd')
    >>>  isopy.keystring(key) # The element flavour has a higher priority
    ElementKeyString('Pd')

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

    If *key* is a key string with a flavour not in *flavour* an exception is raised. If the key flavour is in *flavour*
    it will not be converted to a flavour of higher priority.

    Parameters
    ----------
    key
        A string to be converted into a key string.
    allow_reformatting : bool, Default = True
        If ``True`` the string can be reformatted to get the correct format. If ``False`` only a string that already
        has the correct format is considered.
    flavour
        The possible flavour(s) of the key string.

    Examples
    --------
    >>> isopy.askeystring('pd')
    ElementKeyString('Pd')
    >>> isopy.askeystring('pd', allow_reformatting=False)
    GeneralKeyString('pd')

    >>> key = isopy.askeystring('pd', flavour = 'general'); key
    GeneralKeyString('pd')
    >>>  isopy.askeystring(key) # Preserves the flavour
    GeneralKeyString('pd')

    Returns
    -------
    IsopyKeyString
    """
    flavour = asflavour(flavour)

    if isinstance(key, IsopyKeyString):
        if key.flavour in flavour:
            return key
        else:
            raise KeyFlavourError(key, flavour)
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
        if startswith(key, prefix): #.startswith(prefix):
            parsed_key = remove_prefix(key, prefix)
            if parsed_key in KEY_COMPARISONS:
                parsed_key = f'key_{parsed_key}'
            out[parsed_key] = filters.pop(key)

    return out

def combine_keys_func(func):
    @functools.wraps(func)
    def combine(*args, **kwargs):
        keys = tuple()

        for arg in args:
            if type(arg) is str:
                keys += tuple(arg.strip().split())
            elif isinstance(arg, IsopyKeyString):
                keys += (arg, )
            elif type(arg) is IsopyKeyList:
                keys += tuple(arg)
            elif isinstance(arg, IsopyArray):
                keys += arg.keys
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

def filter_keys(item, keys, key_filters):
    if keys is None:
        keys = item.keys
    else:
        keys = askeylist(keys, flavour=item._key_flavour_)

    if key_filters:
        keys = keys.filter(**key_filters)

    return keys

# Key lists can be created using :py:func:`isopy.keylist()` and :py:func:`isopy.askeylist()`.
class IsopyKeyList(tuple):
    """
    IsopyKeyList()

    A sequence of key strings.

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
    def __new__(cls, keys, flavour, ignore_duplicates = False, allow_duplicates = True, sort = False):
        if ignore_duplicates:
            keys = list(dict.fromkeys(keys).keys())
        elif not allow_duplicates and (len(set(keys)) != len(keys)):
            raise ValueError(f'duplicate key found in list {keys}')

        flavour = asflavour(tuple(k.flavour for k in keys))

        if sort:
            keys = sorted(keys, key= lambda k: k._sortkey_())

        obj = super(IsopyKeyList, cls).__new__(cls, keys)
        obj.flavour = flavour
        return obj

    def _repr_latex_(self):
        if IPYTHON_REPR:
            return fr'$${self.str("math")}$$'
        else:
            return None

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
        view._key_flavour_ = view.flavour
        view.keys = self
        return view

    def filter(self, key_eq= None, key_neq = None, **filters):
        """
        Returns a new key list containing the keys that satify all the filter arguments given.

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

    def str(self, format = None):
        """
        Return a string of the list formatted according to the format.
        """
        if format == 'math':
            keys = ', '.join(self.strlist('math'))
            return fr'\left[{keys}\right]'
        elif format == 'latex':
            keys = ', '.join(self.strlist('math'))
            return fr'$\left[{keys}\right]$'
        else:
            keys = ', '.join(self.strlist(format))
            return f'[{keys}]'

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


def iskeylist(item, *, flavour=None, flavour_in=None) -> bool:
    """
    Returns ``True`` if *item* is a key string list otherwise returns ``False``.

    Parameters
    ----------
    item
        A sequence of strings to be verified.
    flavour :  flavour_like, Optional
        If given then ``True`` is returned if the flavour of *item* is equal to *flavour*.
    flavour_in : flavour_like Optional
        If given then then ``True`` is returned if the flavour of *item* is found in *flavour_in*.

    Examples
    --------
    >>> isopy.iskeylist(['Ru', 'Pd', 'Cd'])
    False

    >>> keys = isopy.keylist('ru pd cd')
    >>> isopy.iskeylist(keys)
    True
    >>> isopy.iskeylist(keys, flavour='element|isotope')
    False
    >>> isopy.iskeylist(keys, flavour_in='element|isotope')
    True
    """

    if flavour is not None:
        return isinstance(item, IsopyKeyList) and item.flavour == asflavour(flavour)
    elif flavour_in is not None:
        return isinstance(item, IsopyKeyList) and item.flavour in asflavour(flavour_in)
    else:
        return isinstance(item, IsopyKeyList)

@combine_keys_func
@lru_cache(CACHE_MAXSIZE)
def keylist(*keys, ignore_duplicates=False, allow_duplicates=True, allow_reformatting=True, sort = False, flavour ='any'):
    """
    Returns a key list with the highest priority flavour compatible with each key string.

    *keys* can consist of single strings, sequences of strings, dictionaries, isopy arrays and numpy arrays.
    For dictionaries and arrays the keys or dtype.name values are used as keys. Strings containing whitespace will
    be split into multiple strings. This only applied to strings given directly as a key,
    not for string contained within other object.

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
    sort : bool
        If ``True`` the keys in will be sorted
    flavour
        The possible flavour(s) of key strings in the key list.

    Examples
    --------
    >>> isopy.keylist(['ru', 'pd', 'cd'])
    IsopyKeyList('Ru', 'Pd', 'Cd', flavour='element')
    >>> isopy.keylist('ru pd cd') # Split into multiple keys
    IsopyKeyList('Ru', 'Pd', 'Cd', flavour='element')
    >>> isopy.keylist(['ru pd cd']) # Strings in other objects are left as is
    IsopyKeyList('ru pd cd', flavour='general')

    >>> d = dict(ru=1, pd=2, cd=3)
    >>> isopy.keylist(d)
    IsopyKeyList('Ru', 'Pd', 'Cd', flavour='element')
    >>> a = isopy.array(d, flavour = 'general')
    >>> isopy.keylist(a)
    IsopyKeyList('Ru', 'Pd', 'Cd', flavour='element')


    Returns
    -------
    IsopyKeyList
    """
    flavour = asflavour(flavour)

    keys = [keystring(k, flavour=flavour, allow_reformatting=allow_reformatting) for k in keys[0]]

    return IsopyKeyList(keys, flavour, ignore_duplicates=ignore_duplicates,
                       allow_duplicates=allow_duplicates, sort=sort)

@combine_keys_func
@lru_cache(CACHE_MAXSIZE)
def askeylist(*keys, ignore_duplicates=False, allow_duplicates=True, allow_reformatting=True, sort=False, flavour ='any'):
    """
    Returns a key list preserving the flavour of each key string if it has a valid flavour.

    If a key is a key string with a flavour not in *flavour* an exception is raised. If the key flavour is in *flavour*
    it will not be converted to a flavour of higher priority.

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
    sort : bool
        If ``True`` the keys in will be sorted
    flavour
        The possible flavour(s) of key strings in the key list.

    Examples
    --------
    >>> isopy.askeylist(['ru', 'pd', 'cd'])
    IsopyKeyList('Ru', 'Pd', 'Cd', flavour='element')
    >>> isopy.askeylist('ru pd cd') # Split into multiple keys
    IsopyKeyList('Ru', 'Pd', 'Cd', flavour='element')
    >>> isopy.askeylist(['ru pd cd']) # Strings in other objects are left as is
    IsopyKeyList('ru pd cd', flavour='general')

    >>> d = dict(ru=1, pd=2, cd=3)
    >>> isopy.askeylist(d)
    IsopyKeyList('Ru', 'Pd', 'Cd', flavour='element')
    >>> a = isopy.array(d, flavour = 'general')
    >>> isopy.askeylist(a)
    IsopyKeyList('ru', 'pd', 'cd', flavour='general')

    Returns
    -------
    IsopyKeyList
    """
    flavour = asflavour(flavour)

    keys = [askeystring(k, allow_reformatting=allow_reformatting, flavour=flavour) for k in keys[0]]

    return IsopyKeyList(keys, flavour, ignore_duplicates=ignore_duplicates,
                        allow_duplicates=allow_duplicates, sort=sort)

###################################
### Mixins for Array and RefVal ###
###################################
def cls_arrayfunc_wrapper(func, name = None):
    if name is None: name = func.__name__

    def call_arrayfunc(self, *args, **kwargs):
        return isopy.arrayfunc(func, self, *args, **kwargs)

    call_arrayfunc.__name__ = name
    return call_arrayfunc

def add_array_functions(cls):
    # These are the ones included with ndarrays
    for func in [np.all, np.any, np.cumprod, np.cumsum,
                 np.mean, np.prod, np.ptp, np.std, np.sum, np.var]:
        setattr(cls, func.__name__, cls_arrayfunc_wrapper(func))

    # these functions are aliases for amin and amax
    setattr(cls, 'min', cls_arrayfunc_wrapper(np.min, 'min'))
    setattr(cls, 'max', cls_arrayfunc_wrapper(np.max, 'max'))

    return cls

@add_array_functions
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




class TableStr(str):
    def __new__(cls, string, *, latex=None, markdown=None, html=None):
        cls._latex = latex
        cls._markdown = markdown
        cls._html = html
        cls._string = string

        return super(TableStr, cls).__new__(cls, string)

    def __repr__(self):
        return self._string

    def __str__(self):
        return self._string

    def _repr_latex_(self):
        return self._latex if IPYTHON_REPR else None

    def _repr_markdown_(self):
        return self._markdown if IPYTHON_REPR else None

    def _repr_html_(self):
        return self._html if IPYTHON_REPR else None

    def copy(self):
        """
        Copy the string to the clipboard.
        """
        pyperclip.copy(self._string)
        return self


class TabulateMixin:
    def tabulate(self, tablefmt='default', *,
                 include_row = False,
                 row_names = None,
                 nrows=None,
                 include_dtype = False,
                 include_objinfo = False,
                 keyfmt = None,
                 floatfmt = None,
                 intfmt = None,
                 keys = None,
                 **key_filters):
        """
        Turn the contents of the array/dictionary to a table.

        Uses `tabulate <https://github.com/astanin/python-tabulate#table-format>`_ to turn the object
        into a table. Markdown, Latex and HTML table formats will render in jupyter notebooks. The
        default table style will render as HTML in jupyter notebooks but the text itself uses the
        "simple" table format.


        Parameters
        ----------
        tablefmt : str
            Format of the table. See the `tabulate documentation <https://github.com/astanin/python-tabulate#table-format>`_.
            for a list of option.
        include_row : bool
            If ``True`` a column with the row number will be included in the table.
        row_names
            The name for each row in the table. Will replace the row number in the row column. If given
            the row column is always included regardless of the value given for *include_row*.
        nrows : int | None
            The maximum number of rows shown in the table. If the number of rows exceeds *nrows*
            rows in the middle of the table will be omitted and replaced with a single row of ``...``.
        include_dtype : bool | None
            If ``True`` then the dtype of each column will be included in the column title.
        include_objinfo : bool | None
            If ``True`` then the object info will be included at the end of the table.
        keyfmt : str | None
            The format used for the column titles.
        floatfmt : str | None
            The format for float values in the table, e.g. ``".2f"`` for 2 decimal places.
        intfmt : str | None
            The format for integer values in the table.
        keys
            The keys the table should contain. If not given all the keys in the item is shown.
        **key_filters
            Key filters to decide which keys should be included in the table.

        Returns
        -------
        TableStr
            A subclass of str that will render the table in jupyter notebooks if the
            table format is markdown, latex or html. Contains one custom method ``.copy()`` that will
            copy the text to the clipboard.
        """
        if len(self.keys) == 0:
            return 'Empty'

        if tablefmt == 'markdown':
            tablefmt = 'pipe'

        if tablefmt == 'default':
            tablefmt = 'simple'
            html_repr = True
        else:
            html_repr = False

        if floatfmt is None:
            floatfmt = '{}'
        elif '{' not in floatfmt:
            floatfmt = '{:' + floatfmt + '}'

        if intfmt is None:
            intfmt = '{}'
        elif '{' not in intfmt:
            intfmt = '{:' + intfmt + '}'

        kwargs = dict(disable_numparse=True, headers = 'keys')

        if keys:
            keys = askeylist(keys)
        else:
            keys = self.keys

        if key_filters:
            keys = keys.filter(**key_filters)


        colalign = []
        table = {}
        if include_row or row_names is not None:
            if row_names is None:
                row_names = [str(i) for i in range(self.size)] if self.ndim == 1 else ['None']

            elif isinstance(row_names, str):
                row_names = [row_names]

            if len(row_names) != self.size:
                raise ValueError(f'Size of row names ({len(row_names)} does not match size of array ({self.size})')
            else:
                row_names = [rn for rn in row_names]

            colalign.append('left')
            table['(row)'] = row_names

        for k in keys:
            dtype = self.get(k).dtype
            title = f'{k.str(keyfmt)} ({dtype.kind}{dtype.itemsize})' if include_dtype else f'{k.str(keyfmt)}'

            val = self.get(k).tolist() if self.ndim == 1 else [self.get(k).tolist()]

            if dtype.kind == 'f':
                colalign.append('right')
                val = [floatfmt.format(v) for v in val]
            elif dtype.kind == 'i' or dtype.kind == 'u':
                colalign.append('right')
                val = [intfmt.format(v) for v in val]
            else:
                colalign.append('left')

            table[title] = val

        if nrows is not None and nrows > 2 and nrows < self.size:
            first = nrows // 2
            last = self.size - (nrows // 2 + nrows % 2)
            for title, values in table.items():
                table[title] = values[:first] + ['...'] + values[last:]

        string = tabulate_(table, tablefmt = tablefmt, colalign = colalign, **kwargs)

        if tablefmt in ['pipe', 'github']:
            if include_objinfo:
                string = f'{string}\n\n{self._description_("**", "**")}'

            return TableStr(string, markdown = string)

        elif tablefmt in ['html', 'usesafehtml']:
            if include_objinfo:
                string = f'{string}\n{self._description_("<b>", "</b>")}'

            return TableStr(string, html = string)

        elif tablefmt in ['latex', 'latex_raw', 'latex_booktabs', 'latex_longtable']:
            if include_objinfo:
                descr = self._description_(r"\textbf{", "}")
                string = f'{string}\n\n{descr}'

            return TableStr(string, latex = string)

        else:
            if include_objinfo:
                string = f'{string}\n{self._description_()}'


            if html_repr:
                html_kwargs = dict(tablefmt = 'html',
                                   include_row = include_row,
                                   row_names=row_names,
                                   nrows = nrows,
                                   include_dtype=include_dtype,
                                   include_objinfo = include_objinfo,
                                   keyfmt = 'latex',
                                   floatfmt = floatfmt,
                                   intfmt = intfmt,
                                   keys = keys)

                html_string = self.tabulate(**html_kwargs)._html
                return TableStr(string, html = html_string)
            else:
                return TableStr(string)

    def __repr__(self):
        return self.tabulate(**ARRAY_REPR)

    def __str__(self):
        return self.tabulate(**ARRAY_STR)

    def _repr_html_(self):
        if IPYTHON_REPR:
            return self.tabulate(tablefmt ='html', keyfmt = 'latex', **ARRAY_REPR)
        else:
            return None


class ToTypeFileMixin:
    def to_list(self, keys = None, default=NotGiven, **key_filters):
        """
        Convert the object to a list.

        If *keys* are given then the array will only these keys will be used/considered for the output.
        If *keys* is not given then all the keys in the array/dictionary are used/considered. If key filters are
        specified then only the keys that pass these filters are included in the output.
        """
        keys = filter_keys(self, keys, key_filters)

        values = [self.get(key, default).tolist() for key in keys]

        if self.ndim == 0:
            return list(values)
        else:
            return [list(r) for r in zip(*values)]

    def to_dict(self, keys = None, default=NotGiven, **key_filters):
        """
        Convert the object to a normal python dictionary.

        If *keys* are given then the array will only these keys will be used/considered for the output.
        If *keys* is not given then all the keys in the array/dictionary are used/considered. If key filters are
        specified then only the keys that pass these filters are included in the output.
        """
        keys = filter_keys(self, keys, key_filters)

        return {key.str(): self.get(key, default).tolist() for key in keys}

    def to_array(self, keys = None, default=NotGiven, **key_filters):
        """
        Convert the object to an IsopyArray.

        If *keys* are given then the array will only these keys will be used/considered for the output.
        If *keys* is not given then all the keys in the array/dictionary are used/considered. If key filters are
        specified then only the keys that pass these filters are included in the output.
        """
        keys = filter_keys(self, keys, key_filters)

        return array({k: self.get(k, default=default) for k in keys})

    def to_refval(self, keys = None, default=NotGiven, *, default_value=NotGiven, ratio_function=NotGiven,
                  molecule_functions=NotGiven, **key_filters):
        """
        Convert the object to a RefValDict.

        If *keys* are given then the array will only these keys will be used/considered for the output.
        If *keys* is not given then all the keys in the array/dictionary are used/considered. If key filters are
        specified then only the keys that pass these filters are included in the output.
        """
        keys = filter_keys(self, keys, key_filters)

        d = {k: self.get(k, default=default) for k in keys}

        if default_value is NotGiven:
            default_value = self.__default__

        if type(self) is RefValDict:
            if ratio_function is NotGiven:
                ratio_function = self.ratio_function
            if molecule_functions is NotGiven:
                molecule_functions = self.molecule_functions

        return RefValDict(d, default_value=default_value, ratio_function=ratio_function,
                          molecule_functions=molecule_functions)

    def to_ndarray(self, keys = None, default=NotGiven, **key_filters):
        """
        Convert the object to a numpy ndarray.

        If *keys* are given then the array will only these keys will be used/considered for the output.
        If *keys* is not given then all the keys in the array/dictionary are used/considered. If key filters are
        specified then only the keys that pass these filters are included in the output.
        """
        keys = filter_keys(self, keys, key_filters)

        a = isopy.array({k: self.get(k, default=default) for k in keys})

        return a.view(ndarray) # Current implementation alwayrs creates a new array.copy()

    def to_dataframe(self, keys = None, default=NotGiven, **key_filters):
        """
        Convert the object to a pandas dataframe.

        If *keys* are given then the array will only these keys will be used/considered for the output.
        If *keys* is not given then all the keys in the array/dictionary are used/considered. If key filters are
        specified then only the keys that pass these filters are included in the output.
        """
        if pandas is not None:
            keys = filter_keys(self, keys, key_filters)

            d = {k.str(): self.get(k, default=default) for k in keys}

            if self.ndim == 0:
                index = [0]
            else:
                index = None
            return pandas.DataFrame(d, copy=True, index=index)
        else:
            raise TypeError('Pandas not installed')

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

# Merge with ArrayFuncMixin

############
### Dict ###
############
NAMED_RATIO_FUNCTION = dict(divide=np.divide)
NAMED_MOLECULE_FUNCTIONS = dict(abundance=(np.add, np.multiply, None),
                                fraction=(np.multiply, np.power, None),
                                mass=(np.add, np.multiply, np.divide))

def inv_named_function(value, named_function):
    for name, func in named_function.items():
        if value == func or value is func:
            return f"'{name}'"
    return value

def readonly_method(func):
    def decorator(self, *args, **kwargs):
        if self._readonly_:
            raise TypeError('This dictionary is readonly. Make a copy to make changes')

        return func(self, *args, **kwargs)
    return decorator


class IsopyDict(dict):
    """
    IsopyDict()

    Dictionary where each value is stored by a isopy keystring key.

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
        descr = f"{self.__class__.__name__}(readonly={self.readonly}, key_flavour='{self.key_flavour}'"
        if self.__default__ is not NotGiven:
            descr = f'{descr}, default_value={self.__default__}'
        return f'{descr})\n{super(IsopyDict, self).__repr__()}'

    def __init__(self, *args, default_value = NotGiven, readonly =False, key_flavour = 'any', **kwargs):
        super(IsopyDict, self).__init__()
        self._readonly_ = False
        self.default_value = default_value
        if key_flavour is NotGiven:
            if len(args) == 1 and isinstance(args[0], IsopyDict):
                key_flavour = args[0]._key_flavour_
            else:
                key_flavour = 'any'
        self._key_flavour_ = asflavour(key_flavour)

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
        self._readonly_ = readonly

    @readonly_method
    def __delitem__(self, key):
        key = askeystring(key, flavour=self._key_flavour_)
        super(IsopyDict, self).__delitem__(key)

    @readonly_method
    def __setitem__(self, key, value):
        key = askeystring(key, flavour=self._key_flavour_)
        value = self._make_value(value, key)
        super(IsopyDict, self).__setitem__(key, value)

    def __contains__(self, key):
        key = askeystring(key, flavour=self._key_flavour_)
        return super(IsopyDict, self).__contains__(key)

    def __getitem__(self, key):
        key = askeystring(key, flavour=self._key_flavour_)
        return super(IsopyDict, self).__getitem__(key)

    @property
    def keys(self):
        return askeylist(super(IsopyDict, self).keys(), flavour=self._key_flavour_)

    @property
    def readonly(self) -> bool:
        return self._readonly_

    @property
    def default_value(self):
        return self.__default__

    @default_value.setter
    @readonly_method
    def default_value(self, value):
        self.__default__ = self._make_default_value(value)

    def _make_value(self, value, key, *_):
        return value

    def _make_default_value(self, value):
        return value

    @property
    def key_flavour(self):
        return self._key_flavour_

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
        *default* is not given the default value of the dictionary is used.

        A TypeError is raised if the dictionary is readonly.
        """
        if default is NotGiven:
            default = self.__default__

        try:
            key = askeystring(key, flavour=self._key_flavour_)
        except KeyParseError:
            pass
        else:
            if key in self:
                return super(IsopyDict, self).pop(key)

        if default is not NotGiven:
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
        key = askeystring(key, flavour=self._key_flavour_)
        if default is NotGiven:
            default = self.__default__

        if key not in self:
            if default is NotGiven:
                raise ValueError('No default value given')

            self.__setitem__(key, default)

        return self.__getitem__(key)

    def _copy(self, data, default_value):
        return self.__class__(data,
                              default_value=default_value,
                              key_flavour=self._key_flavour_)

    def copy(self):
        return self._copy(self, self.__default__)

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
        if isinstance(key, (str, int)):
            try:
                key = askeystring(key, flavour=self._key_flavour_)
            except KeyParseError:
                pass
            else:
                try:
                    return super(IsopyDict, self).__getitem__(key)
                except KeyError:
                    pass

        if isinstance(key, abc.Sequence) is False:
            return tuple(self.get(k, default) for k in key)

        # Only gets here if key not in dict
        if default is NotGiven:
            default = self.__default__

        if default is NotGiven:
            raise ValueError('No default value given')

        if isinstance(default, dict):
            default = isopy.asdict(default)
            default_value = default.get(key, self.__default__)
        elif isinstance(default, IsopyArray):
            default_value = default.get(key, self.__default__)
        else:
            default_value = default

        return default_value

    def to_dict(self):
        return {key.str(): self.get(key) for key in self.keys}


class RefValDict(ArrayFuncMixin, ToTypeFileMixin, TabulateMixin, IsopyDict):
    """
    RefValDict()

    Dictionary where each value is stored as an array of floats by a isopy keystring key.

    Each value in the dictionary has the same ndim and size. If the dictionary has a size of 1 ndim will always be 0.

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
    ratio_function : callable
        The function that should be used to calculate the value of a missing ratio key string from the data present
        in the array. If None then no attempt is made to calculate the missing value. ``'divide'`` is an
        alias for ``np.divide``.
    molecule_functions : None or (callable, callable, callable or None)
        A tuple of three functions that should be used to calculate the value of a missing molecule key string from
        the data present in the array. The first function is used to calculate the value for the components,
        the second function for the ``n`` and the final function for the ``charge``. If the third item in the tuple
        is ``None`` then the charge is ignored. If None then no attempt is made to calculate the missing value.
        ``'fraction'`` is an alias for ``(np.multiply, np.power, None)``, ``'abundance'`` is an alias for
        ``(np.add, np.multiply, None)`` and ``'mass'`` is an alias for
        ``(np.add, np.multiply, np.divide)``
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
    ratio_function
        The function used to calculate the value of a missing ratio key string from the data present in the array. If
        None then no attempt is made to calculate the missing value.
    molecule_functions
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

    def _description_(self, boldstart = '', boldend = ''):
        descr = f"{boldstart}{self.__class__.__name__}{boldend}({self.size}, readonly={self.readonly}, key_flavour='{self.key_flavour}'"
        descr = f'{descr}, ratio_function={inv_named_function(self._ratio_func, NAMED_RATIO_FUNCTION)}'
        descr = f'{descr}, molecule_functions={inv_named_function(self._molecule_funcs, NAMED_MOLECULE_FUNCTIONS)}'
        if self.ndim == 0 or False in [np.array_equal(self.__default__[0], dv, equal_nan=True) for dv in self.__default__]:
            default_value = self.__default__
        else:
            default_value = self.__default__[0]
        descr = f'{descr}, default_value={default_value})'
        return descr

    def __init__(self, *args: dict, default_value=nan,
                 readonly= False, key_flavour = 'any', ratio_function = None,
                 molecule_functions = None, **kwargs):

        if len(args) == 1 and type(args[0]) is RefValDict:
            if default_value is NotGiven:
                default_value = args[0].default_value
            if ratio_function is NotGiven:
                ratio_function = args[0].ratio_function
            if molecule_functions is NotGiven:
                molecule_functions = args[0].molecule_functions
        else:
            if default_value is NotGiven:
                default_value = nan
            if ratio_function is NotGiven:
                ratio_function = None
            if molecule_functions is NotGiven:
                molecule_functions = None

        self._readonly_ = False
        self._size = 1
        self._ndim = 0

        self.ratio_function = ratio_function
        self.molecule_functions = molecule_functions

        super(RefValDict, self).__init__(*args, default_value=default_value,
                                         readonly=readonly,
                                         key_flavour= key_flavour,
                                         **kwargs)

    def __getitem__(self, key):
        if isinstance(key, str):
            key = askeystring(key, flavour=self._key_flavour_)
            return super(IsopyDict, self).__getitem__(key)
        elif isinstance(key, (int, slice)):
            if self.ndim == 0 and (key == 0 or key == slice(None)):
                data = {k: v for k, v in self.items()}
                default = self.__default__
            else:
                data = {k: v[key] for k, v in self.items()}
                default = self.__default__[key]
            return self._copy(data, default)
        elif isinstance(key, (list, tuple)):
            if len(key) == 0:
                return self.to_refval(key_eq=[])
            elif False not in {isinstance(k, str) for k in key}:
                return self.to_refval(key_eq=key)
            else:
                data = {k: v[key] for k, v in self.items()}
                return self._copy(data, self.__default__[key])
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
                              key_flavour=self._key_flavour_,
                              ratio_function = self._ratio_func,
                              molecule_functions=self._molecule_funcs)

    def _make_value(self, value, key, resize = True, default_value = False):
        try:
            value = np.array(value, dtype=np.float64, ndmin=1)
        except Exception as err:
            raise ValueError(f'Key "{key}": Unable to convert value(s) {value} to float') from err

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
                        self.__default__ = np.full(self._size, self.__default__)
                        self.__default__.setflags(write=False)
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
            try:
                key = askeystring(key, flavour=self._key_flavour_)
            except KeyParseError:
                pass
            else:
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
            return self.__default__

        if isinstance(default, dict):
            default = isopy.asdict(default)
            default_value = default.get(key, self.__default__)
        elif isinstance(default, IsopyArray):
            default_value = default.get(key, self.__default__)
        else:
            default_value = default

        return self._make_value(default_value, 'default', False)

    @property
    def ndim(self):
        return self._ndim

    @property
    def size(self):
        return self._size

    @property
    def ratio_function(self):
        return self._ratio_func

    @ratio_function.setter
    @readonly_method
    def ratio_function(self, ratio_func):
        ratio_func = NAMED_RATIO_FUNCTION.get(ratio_func, ratio_func)

        if ratio_func is None or callable(ratio_func):
            self._ratio_func = ratio_func
        else:
            raise ValueError('ratio_functions must be None or a callable')

    @property
    def molecule_functions(self):
        return self._molecule_funcs

    @molecule_functions.setter
    @readonly_method
    def molecule_functions(self, molecule_funcs):
        molecule_funcs = NAMED_MOLECULE_FUNCTIONS.get(molecule_funcs, molecule_funcs)

        if type(molecule_funcs) is tuple and len(molecule_funcs) == 3:
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
            raise ValueError('molecule_functions must be None or a tuple of callable items')


ScalarDict = RefValDict # For legacy reasons

def isdict(item, key_flavour=NotGiven, default_value=NotGiven):
    """
    Returns ``True`` if *item* is an isopy dict or refval dict otherwise returns ``False``.
    """
    if type(item) is RefValDict:
        return isrefval(item, key_flavour=key_flavour, default_value=default_value)
    elif type(item) is not IsopyDict:
        return False

    if key_flavour is not NotGiven:
        if item.key_flavour != key_flavour:
            return False

    if default_value is not NotGiven:
        #This should work for both None and np.nan too
        if item.default_value != default_value and item.default_value is not default_value:
            return False

    return True

def isrefval(item, key_flavour=NotGiven, default_value=NotGiven, ratio_function=NotGiven, molecule_functions=NotGiven):
    """
    Returns ``True`` if *item* is a refval dict otherwise returns ``False``.
    """
    if not type(item) is RefValDict:
        return False

    if key_flavour is not NotGiven:
        if item.key_flavour != key_flavour:
            return False

    if default_value is not NotGiven:
        try:
            default_value = item._make_value(default_value, 'default_value', False)
        except:
            return False
        else:
            if not np.array_equal(item.default_value, default_value, equal_nan=True):
                return False

    if ratio_function is not NotGiven:
        if (item._ratio_func is not ratio_function and
                item._ratio_func != NAMED_RATIO_FUNCTION.get(ratio_function, ratio_function)):
            return False

    if molecule_functions is not NotGiven:
        if (item._molecule_funcs is not molecule_functions and
                item._molecule_funcs != NAMED_MOLECULE_FUNCTIONS.get(molecule_functions, molecule_functions)):
            return False

    return True

def asdict(d, default_value = NotGiven, key_flavour = NotGiven):
    """
    Return *d* if it is an IsopyDict otherwise convert *d* into one and return it.

    The returned IsopyDict will have the specified default value and key flavour(s) if these are given.
    """
    if type(d) == str:
        d = isopy.refval(d)

    if isopy.isdict(d, key_flavour=key_flavour, default_value=default_value):
        return d
    else:
        return IsopyDict(d, default_value = default_value, key_flavour = key_flavour)

def asrefval(d, default_value = NotGiven, key_flavour = NotGiven, ratio_function = NotGiven, molecule_functions = NotGiven):
    """
    Return *d* if it is an RefValDict otherwise convert *d* into one and return it.

    The returned RefValDict will have the specified default value and key flavour(s) if these are given.
    """
    if key_flavour is not NotGiven:
        key_flavour = asflavour(key_flavour)

    if type(d) == str:
        d = isopy.refval(d)

    if isopy.isrefval(d, key_flavour=key_flavour, default_value=default_value,
                      ratio_function=ratio_function, molecule_functions=molecule_functions):
        return d
    else:
        return RefValDict(d, default_value = default_value, key_flavour = key_flavour,
                      molecule_functions=molecule_functions, ratio_function=ratio_function)


#############
### Array ###
#############
class IsopyArray(ArrayFuncMixin, ToTypeFileMixin, TabulateMixin):
    """
    IsopyArray()

    An array where data is stored rows and columns of isopy key strings.

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

    Attributes
    ----------
    ncols
        The number of columns in the array
    nrows
        The number of rows in the array. If the array is 0-dimensional *nrows* is ``-1``.
    ndim
        The number of dimensions of the data in the array.
    size
        The number of rows in the array. If the array is 0-dimensional *size* is ``1``.
    datatypes
        The data type for each column in the array.
    keys
        The column key strings
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

        if isinstance(values, (ndarray, void)):
            if values.dtype.names is not None:
                if keys is None:
                    keys = list(values.dtype.names)

                if dtype is None:
                    dtype = [(values.dtype[i],) for i in range(len(values.dtype))]

            else:
                if dtype is None:
                    if True:
                        dtype = (values.dtype,)
                    elif values.ndim == 0:
                        dtype = [(values.dtype,)]
                    elif values.size > 1:
                        dtype = [(values.dtype,) for i in range(values.shape[-1])]

            values = values.tolist()

        if isinstance(values, (list, tuple)):
            if False not in [(type(v) is list or type(v) is tuple or (isinstance(v, np.ndarray) and v.ndim > 0))
                            for v in values]:
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

        else:
            raise ValueError(f'unable to convert values with type "{type(values)}" to IsopyArray')

        if keys is None:
            # IF there are no keys at this stage raise an error
            raise ValueError('Keys argument not given and keys not found in values')
        else:
            keys = askeylist(keys, allow_duplicates=False, flavour=flavour)
            klen = len(keys)

        if isinstance(values, tuple):
            vlen = len(values)
        else:
            vlen = {len(v) for v in values}

            if len(vlen) == 1:
                vlen = vlen.pop()
            elif len(vlen) == 0:
                vlen = None
            else:
                raise ValueError('All rows in values are not the same size')

        if vlen is not None and vlen != klen:
            raise ValueError('size of values does not match the number of keys')

        if dtype is None:
            new_dtype = [(float64, None) for k in keys]

        elif isinstance(dtype, list) or (isinstance(dtype, np.dtype) and dtype.names is not None):
            if len(dtype) != klen:
                raise ValueError('number of dtypes given does not match number of keys')
            else:
                new_dtype = [(dtype[i],) if not isinstance(dtype[i], tuple) else dtype[i] for i in range(klen)]

        elif isinstance(dtype, tuple):
            new_dtype = [dtype for i in range(klen)]
        else:
            new_dtype = [(dtype,) for i in range(klen)]

        if len(values) == 0:
            colvalues = [[] for k in keys]
        elif type(values) is tuple:
            colvalues = values
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

    def _description_(self, boldstart = '', boldend = ''):
        return f"{boldstart}{self.__class__.__name__}{boldend}({self.nrows}, flavour='{self.flavour}', default_value={self.__default__})"

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

        elif type(index) is slice:
            if index == slice(None):
                return None, None
            elif self.ndim == 0:
                raise IndexError('0-dimensional arrays cannot be indexed by row')

        elif self.ndim == 0 and type(index) is int:
            if index == 0:
                return None, None
            else:
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
                raise IndexError('too many indices for array')

        return None, index

    def __getitem__(self, index):
        keyi, rowi = self._parse_index_(index)

        if keyi is None:
            a = self
        else:
            a = keyi._view_array_(super(IsopyArray, self).__getitem__(keyi._str_()))

        if rowi is None:
            return a
        elif isinstance(rowi, IsopyArray):
            raise IndexError('Isopy arrays can only be used as an index when setting arrays')
        elif isinstance(a, IsopyArray):
            return a.keys._view_array_(super(IsopyArray, a).__getitem__(rowi))
        else:
            return a[rowi]

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

    @property
    def datatypes(self):
        return tuple(self.dtype[i] for i in range(len(self.dtype.names)))

    def values(self):
        """
        Returns a tuple containing the column values for each key in the array

        Equivalent to ``tuple(array[key] for key in array.keys)``
        """
        return tuple(self[key] for key in self.keys)

    def items(self):
        """
        Returns a tuple containing a tuple with the key and the column values for each key in the array

        Equivalent to ``tuple((key, array[key]) for key in array.keys)``
        """
        return tuple((key, self[key]) for key in self.keys)

    def get(self, key, default = NotGiven):
        """
        Returns the values of column *key* if present in the array. Otherwise an numpy array
        filled with *default* is returned with the same shape as a column in the array. An
        exception will be raised if *default* cannot be broadcast to the correct shape.

        If *default* is not given np.nan is used.
        """
        try:
            key = askeystring(key, flavour=self.flavour)
            return self.__getitem__(key)
        except:

            if default is NotGiven:
                default_value = self.__default__ #This is always nan unless the .default() has been called.
            elif isinstance(default, dict):
                default = isopy.asdict(default)
                default_value = default.get(key, self.__default__)
            elif isinstance(default, IsopyArray):
                default_value = default.get(key, self.__default__)
            else:
                default_value = default

            if not isinstance(default_value, ndarray):
                try:
                    return np.full(self.shape, default_value, dtype=float64)
                except ValueError:
                    pass

            return np.full(self.shape, default_value)

    def copy(self):
        """
        Returns a copy of the array.
        """
        copy =  super(IsopyArray, self).copy()
        return self.keys._view_array_(copy)

    def filter(self, **key_filters):
        """
        Returns a view of the array containing the keys that satisfy the *key_filters*.
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

    def deratio(self, denominator_value=1, sort_keys=True):
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
            out = isopy.cstack(out, ones(self.nrows, denominator), sort_keys=sort_keys)
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
        (row)      102Pd (f8)    104Pd (f8)    105Pd (f8)    106Pd (f8)    108Pd (f8)    110Pd (f8)
        -------  ------------  ------------  ------------  ------------  ------------  ------------
        None          0.01020       0.11140       0.22330       0.27330       0.26460       0.11720
        IsopyNdarray(-1, flavour='isotope', default_value=nan)
        >>> array.normalise(10)
        (row)      102Pd (f8)    104Pd (f8)    105Pd (f8)    106Pd (f8)    108Pd (f8)    110Pd (f8)
        -------  ------------  ------------  ------------  ------------  ------------  ------------
        None          0.10200       1.11400       2.23300       2.73300       2.64600       1.17200
        IsopyNdarray(-1, flavour='isotope', default_value=nan)
        >>> array.normalise(1, 'pd102')
        (row)      102Pd (f8)    104Pd (f8)    105Pd (f8)    106Pd (f8)    108Pd (f8)    110Pd (f8)
        -------  ------------  ------------  ------------  ------------  ------------  ------------
        None          1.00000      10.92157      21.89216      26.79412      25.94118      11.49020
        IsopyNdarray(-1, flavour='isotope', default_value=nan)
        >>> array.normalise(10, ['pd106', 'pd108'])
        (row)      102Pd (f8)    104Pd (f8)    105Pd (f8)    106Pd (f8)    108Pd (f8)    110Pd (f8)
        -------  ------------  ------------  ------------  ------------  ------------  ------------
        None          0.18963       2.07102       4.15133       5.08087       4.91913       2.17884
        IsopyNdarray(-1, flavour='isotope', default_value=nan)
        >>> array.normalise(10, isopy.keymax)
        (row)      102Pd (f8)    104Pd (f8)    105Pd (f8)    106Pd (f8)    108Pd (f8)    110Pd (f8)
        -------  ------------  ------------  ------------  ------------  ------------  ------------
        None          0.37322       4.07611       8.17051      10.00000       9.68167       4.28833
        IsopyNdarray(-1, flavour='isotope', default_value=nan)
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
        >>> a1 = isopy.array([1,2,3], 'ru pd cd')
        >>> a2 = isopy.array([20, 10], 'ru cd')
        >>> a1 + a2 # Default value is np.nan
        (row)      Ru (f8)    Pd (f8)    Cd (f8)
        -------  ---------  ---------  ---------
        None      21.00000        nan   13.00000
        IsopyNdarray(-1, flavour='element', default_value=nan)
        >>> a1 + a2.default(0) # a2 has a temporary default value of 0
        (row)      Ru (f8)    Pd (f8)    Cd (f8)
        -------  ---------  ---------  ---------
        None      21.00000    2.00000   13.00000
        IsopyNdarray(-1, flavour='element', default_value=nan)
        """
        temp_view = self.keys._view_array_(self)
        temp_view.__default__ = value
        return temp_view


class IsopyNdarray(IsopyArray, ndarray):
    pass


class IsopyVoid(IsopyArray, void):
    pass


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
    (row)      Ru (f8)    Pd (f8)    Cd (f8)
    -------  ---------  ---------  ---------
    0         -0.73359    0.11496   -0.00119
    1         -0.59661   -0.14210    0.52218
    2         -0.62663   -1.32210    0.71435
    3          1.69478   -0.60308   -0.31961
    4          0.99229    0.42969   -0.36984
                ...        ...        ...
    95         1.29482   -1.49722    0.00716
    96        -1.32433    0.99887   -0.02710
    97        -0.34908    0.39324   -1.46929
    98        -0.47520    0.39947   -0.16034
    99         0.32749    0.53820   -0.23848
    IsopyNdarray(100, flavour='element', default_value=nan)
    >>> isopy.mean(array)
    (row)      Ru (f8)    Pd (f8)    Cd (f8)
    -------  ---------  ---------  ---------
    None      -0.04493    0.07397   -0.06447
    IsopyNdarray(-1, flavour='element', default_value=nan)

    >>> isopy.sd(array)
    (row)      Ru (f8)    Pd (f8)    Cd (f8)
    -------  ---------  ---------  ---------
    None       1.06521    1.03131    1.03173
    IsopyNdarray(-1, flavour='element', default_value=nan)

    >>> array = isopy.random(100, [(0, 1), (1, 0.1), (-1, 10)], keys=('ru', 'pd', 'cd'))
    >>> array
    (row)      Ru (f8)    Pd (f8)    Cd (f8)
    -------  ---------  ---------  ---------
    0          0.82868    0.97201    2.53042
    1         -0.87905    1.04721   17.23299
    2          0.04199    0.88000  -11.31050
    3          0.11860    1.02957  -15.47807
    4          0.02590    1.14512   -4.99726
                ...        ...        ...
    95        -0.27775    1.04640   -9.34926
    96        -0.09882    1.09960    7.67782
    97        -0.22307    1.03733    9.68606
    98         0.24518    1.04231   -4.08202
    99        -0.75468    0.92260    0.70036
    IsopyNdarray(100, flavour='element', default_value=nan)
    >>> np.mean(array)
    (row)      Ru (f8)    Pd (f8)    Cd (f8)
    -------  ---------  ---------  ---------
    None      -0.04996    0.99352   -1.02721
    IsopyNdarray(-1, flavour='element', default_value=nan)

    >>> isopy.sd(array)
    (row)      Ru (f8)    Pd (f8)    Cd (f8)
    -------  ---------  ---------  ---------
    None       0.85324    0.09870    8.70021
    IsopyNdarray(-1, flavour='element', default_value=nan)
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

# Simplify to that there is one where we know there is only one input and one where we know there is
# More than two inputs and this one for unknown cases.
# This is for functions that dont have a return value
def call_array_function(func, *inputs, axis=NotGiven, keys=None, key_filters = None, **kwargs):
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
        axis_kwarg = True
        if 'axis' in kwargs:
            axis = kwargs['axis']
    else:
        axis_kwarg = False

    if axis is NotGiven:
        axis = 0
    else:
        kwargs['axis'] = axis

    new_inputs = []
    a_input = []
    d_input = []
    for i, arg in enumerate(inputs):
        if isinstance(arg, IsopyArray):
            new_inputs.append(arg)
            a_input.append(arg)

        elif isinstance(arg, ndarray) and len(arg.dtype) > 1:
            new_inputs.append(asarray(arg))
            a_input.append(new_inputs[-1])

        elif isinstance(arg, RefValDict):
            new_inputs.append(arg)
            d_input.append(arg)

        elif isinstance(arg, dict):
            new_inputs.append(RefValDict(arg))
            d_input.append(new_inputs[-1])

        else:
            new_inputs.append(KeylessValue(np.asarray(arg)))

    if keys:
        return_array = True
        keys = askeylist(keys)

    if a_input:
        return_array = True
        if not keys:
            keys = askeylist(*(input.keys for input in a_input), sort=(len(a_input)>1), ignore_duplicates=True)
    elif d_input:
        return_array = False
        if not keys:
            keys = askeylist(*(input.keys for input in d_input), sort=(len(d_input)>1), ignore_duplicates=True)
    elif not keys:
        return func(*inputs, **kwargs)

    if key_filters:
        keys = keys.filter(**key_filters)

    if axis == 0:
        if axis_kwarg:
            kwargs['axis'] = None
        out = kwargs.get('out', None)
        keykwargs = {kwk: kwargs.pop(kwk) for kwk, kwv in tuple(kwargs.items())
                                if isinstance(kwv, (IsopyArray, IsopyDict))}

        result = [func(*(input.get(key) for input in new_inputs), **kwargs,
                       **{k: v.get(key) for k, v in keykwargs.items()}) for key in keys]

        #There is no return from the function so dont return an array
        if False not in [r is None for r in result]:
            return None

        if out is None:
            if return_array:
                dtype = [(str(key), getattr(result[i], 'dtype', float64)) for i, key in enumerate(keys)]
                if getattr(result[0], 'ndim', 0) == 0:
                    out = np.array(tuple(result), dtype=dtype)
                else:
                    out = np.fromiter(zip(*result), dtype=dtype)
                return keys._view_array_(out)
            else:
                if len(d_input) == 1:
                    dv = d_input[0].default_value
                    if dv.size > 1 and result[0].size != dv.size:
                        if np.all(dv==dv[0]):
                            dv = dv[0]
                        else:
                            dv = np.nan
                    # The return dictionary inherits the properties of the original dictionary
                    return d_input[0]._copy(zip(keys, result), dv)
                else:
                    return RefValDict(zip(keys, result))
        else:
            return out

    else:
        for kwk, kwv in list(kwargs.items()):
            if isinstance(kwv, (IsopyArray, IsopyDict)):
                kwargs[kwk] = np.transpose([kwv.get(key) for key in keys])

        new_inputs = [np.transpose([input.get(key) for key in keys]) if
                      not isinstance(input, KeylessValue) else input.get() for input in new_inputs]

        if axis == 1 and len(nd :={inp.ndim for inp in new_inputs}) == 1 and nd.pop() == 1:
            #So that 0-dimensional arrays get do not raise axis out of bounds error
            kwargs['axis'] = 0

        return func(*new_inputs, **kwargs)

