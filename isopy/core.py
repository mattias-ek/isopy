import isopy
import numpy as np
import pyperclip as pyperclip
import inspect as inspect
import functools

#optional imports
try:
    import pandas
except:
    pandas = None

try:
    import tables
except:
    tables = None

from numpy import ndarray, nan, float64, void
from typing import TypeVar, Union, Optional, Any, NoReturn, Generic

class NotGivenType: pass
NotGiven = NotGivenType()

ALLOW_ALL_NUMPY_FUNCTIONS = False

ARRAY_REPR = dict(include_row=True, include_dtype=False, nrows=10, f='{:.5f}')

__all__ = ['MassKeyString', 'ElementKeyString', 'IsotopeKeyString', 'RatioKeyString', 'GeneralKeyString',
           'MassKeyList', 'ElementKeyList', 'IsotopeKeyList', 'RatioKeyList', 'GeneralKeyList',
           'MassArray', 'ElementArray', 'IsotopeArray', 'RatioArray', 'GeneralArray',
           'IsopyDict', 'ScalarDict',
           'keystring', 'askeystring', 'keylist', 'askeylist', 'array', 'asarray', 'asanyarray',
           'ones', 'zeros', 'empty', 'full',
           'concatenate',
           'iskeystring', 'iskeylist', 'isarray',
           'flavour', 'isflavour']

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

##################
### Exceptions ###
##################
def _classname(thing):
    if isinstance(thing, type): return thing.__name__
    else: return thing.__class__.__name__


class IsopyException(Exception):
    pass


class KeyStringException(IsopyException):
    pass


class KeyParseError(KeyStringException):
    pass


class KeyValueError(KeyParseError, ValueError):
    def __init__(self, cls, string, additional_information = None):
        self.string = string
        self.cls = cls
        self.additional_information = additional_information

    def __str__(self):
        message = f'{_classname(self.cls)}: unable to parse "{self.string}"'
        if self.additional_information:
            return f'{message}. {self.additional_information}.'
        else:
            return message


class KeyTypeError(KeyParseError, TypeError):
    def __init__(self, cls, obj):
        self.obj = obj
        self.cls = cls

    def __str__(self):
        return f'{_classname(self.cls)}: cannot convert {type(self.obj)} into \'str\''


class KeyListException(IsopyException):
    pass


class ListDuplicateError(KeyListException, ValueError):
    def __init__(self, listobj, list_):
        self.listobj = listobj
        self.list_ = list_

    def __str__(self):
        return f'{_classname(self.listobj)}: duplicate key found in list {self.list_}'


class ListSizeError(KeyListException, ValueError):
    def __init__(self, listobj, other):
        self.other = other
        self.listobj = listobj

    def __str__(self):
        return f'{_classname(self.listobj)}: size of \'{_classname(self.other)}\' ({len(self.other)}) does not match current list ({len(self.listobj)})'


class ListTypeError(KeyListException, TypeError):
    def __init__(self, other, expected, listobj):
        self.other = other
        self.expected = expected
        self.listobj = listobj

    def __str__(self):
        return '{}: Item must be a {} not {}'.format(_classname(self.listobj), _classname(self.listobj),
                                                     _classname(self.expected))


class NoCommomDenominator(KeyListException, TypeError):
    def __init__(self, listobj):
        self.listobj = listobj

    def __str__(self):
        return f'{_classname(self.listobj)}: list does not have a common denominator'


class NoKeysError(IsopyException, ValueError):
    pass

class NDimError(IsopyException, ValueError):
    pass


################
### Flavours ###
################

class IsopyFlavour:
    @staticmethod
    def _Flavour__keystring(string, **kwargs):
        raise TypeError('IsopyType not specified')

    @staticmethod
    def _Flavour__keylist(*args, allow_reformatting=True, **kwargs):
        raise TypeError('IsopyType not specified')

    @staticmethod
    def _Flavour__array(*args, allow_reformatting=True, **kwargs):
        raise TypeError('IsopyType not specified')

    @staticmethod
    def _Flavour__view_ndarray(obj):
        raise TypeError('IsopyType not specified')

    @staticmethod
    def _Flavour__view_void(obj):
        raise TypeError('IsopyType not specified')


class MassFlavour:
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


def isflavour(obj1, obj2):
    """returns True if the favour of the first object is the same as the favour of the second object."""
    return flavour(obj1) is flavour(obj2)


def flavour(obj):
    """returns the flavour type of the object."""
    if type(obj) is type:
        if issubclass(obj, MassFlavour):
            return MassFlavour
        elif issubclass(obj, ElementFlavour):
            return ElementFlavour
        elif issubclass(obj, IsotopeFlavour):
            return IsotopeFlavour
        elif issubclass(obj, RatioFlavour):
            return RatioFlavour
        elif issubclass(obj, GeneralFlavour):
            return GeneralFlavour
        else:
            raise TypeError(f'{type(obj)} is not an isopy object')
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
        else:
            raise TypeError(f'{type(obj)} is not an isopy object')

##############
### Key ###
##############
class IsopyKeyString(IsopyFlavour, str):
    def __repr__(self):
        return f"{self.__class__.__name__}('{self}')"

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
    def __new__(cls, string, *, allow_reformatting=True):
        if isinstance(string, cls):
            return string

        if isinstance(string, int) and allow_reformatting:
            string = str(string)

        if not isinstance(string, str):
            raise KeyTypeError(cls, string)

        string.strip()
        # string = string.removeprefix('_')
        string = string.removeprefix('Mass_') #For backwards compatibility
        string = string.removeprefix('MAS_') #For backwards compatibility

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

    def _filter(self, key_eq=None, key_neq=None, *, key_lt=None, key_gt=None,
                key_le=None, key_ge=None, **invalid):
        if invalid:
            return False
        if key_eq is not None and self not in key_eq:
            return False
        if key_neq is not None and self in key_neq:
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
    def __new__(cls, string, *, allow_reformatting=True):
        if isinstance(string, cls):
            return string

        if not isinstance(string, str):
            raise KeyTypeError(cls, string)

        string = string.strip()
        string = string.removeprefix('Element_') #For backwards compatibility
        string = string.removeprefix('ELE_') #For backwards compatibility

        if len(string) == 0:
            raise KeyValueError(cls, string, 'Cannot parse empty string')

        if len(string) > 2:
            raise KeyValueError(cls, string, 'ElementKeyString is limited to two characters')

        if not string.isalpha():
            raise KeyValueError(cls, string, 'ElementKeyString is limited to alphabetical characters')

        if allow_reformatting:
            string = string.capitalize()
        elif string[0].isupper() and (len(string) == 1 or string[1].islower()):
            pass
        else:
            raise KeyValueError(cls, string, 'First character must be upper case and second character must be lower case')

        return super(ElementKeyString, cls).__new__(cls, string)

    def _filter(self, key_eq = None, key_neq = None, **invalid):
        if invalid:
            return False
        if key_eq is not None and self not in key_eq:
            return False
        if key_neq is not None and self in key_neq:
            return False
        return True

    def _sortkey(self):
        z = isopy.refval.element.atomic_number.get(self, self)
        return f'{z:0>4}'


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
    >>> isopy.IsotopeKeyString('Pd104').mass_number
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

        string = string.removeprefix('_') #For backwards compatibility
        string = string.removeprefix('Isotope_') #For backwards compatibility
        string = string.removeprefix('ISO_') #For backwards compatibility

        # If no digits in string then only Symbol is given.
        if string.isalpha():
            raise KeyValueError(cls, string, 'string does not contain a mass number')

        # If only digits then only mass number
        if string.isdigit():
            raise KeyValueError(cls, string, 'string does not contain an element symbol')

        # Loop through to split
        l = len(string)
        for i in range(1, l):
            a = string[:i]
            b = string[i:]

            if ((mass:=a).isdigit() and (element:=b).isalpha()) or ((mass:=b).isdigit() and (element:=a).isalpha()):
                mass = MassKeyString(mass, allow_reformatting=allow_reformatting)
                element = ElementKeyString(element, allow_reformatting=allow_reformatting)

                string = '{}{}'.format(mass, element)
                return super(IsotopeKeyString, cls).__new__(cls, string,
                                                           mass_number = mass,
                                                           element_symbol = element)

        raise KeyValueError(cls, string, 'unable to separate string into a mass number and an element symbol')

    def __hash__(self):
        return hash( (self.__class__, hash(self.mass_number), hash(self.element_symbol)) )

    def __contains__(self, string):
        """
        Return **True** if `string` is equal to the IsotopeKeyString's mass number or element symbol
        """
        return self.mass_number == string or self.element_symbol == string

    def _filter(self, key_eq=None, key_neq=None, mass_number = {}, element_symbol = {}, **invalid):
        if len(invalid) > 0:
            return False
        if key_eq is not None and self not in key_eq:
            return False
        if key_neq is not None and self in key_neq:
            return False
        if mass_number and not self.mass_number._filter(**mass_number):
            return False
        if element_symbol and not self.element_symbol._filter(**element_symbol):
            return False

        return True

    def _sortkey(self):
        return f'{self.mass_number._sortkey()}{self.element_symbol._sortkey()}'


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
    def __new__(cls, string, *, allow_reformatting=True):
        if isinstance(string, cls):
            return string

        if isinstance(string, tuple) and len(string) == 2:
            numer, denom = string

        elif not isinstance(string, str):
            raise KeyTypeError(cls, string)

        else:

            string = string.strip()

            # For backwards compatibility
            if string.startswith('Ratio_'):
                string = string.removeprefix('Ratio_')
                try:
                    numer, denom = string.split('_', 1)
                except:
                    raise KeyValueError(cls, string,
                                        'unable to split string into numerator and denominator')

            # For backwards compatibility
            elif string.startswith('RAT') and string[3].isdigit():
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

        numer = askeystring(numer, allow_reformatting=allow_reformatting)
        denom = askeystring(denom, allow_reformatting=allow_reformatting)

        for n in range(1, 10):
            divider = '/' * n
            if numer.count(divider) > 0 or denom.count(divider) > 0:
                continue
            else:
                break
        else:
            raise KeyValueError('Limit of nested ratios reached')

        string = f'{numer}{divider}{denom}'

        #colname = f'RAT{n}_{numer.colname}_OVER{n}_{denom.colname}'

        return super(RatioKeyString, cls).__new__(cls, string,
                                                  numerator = numer, denominator = denom)

    def __hash__(self):
        return hash( (self.__class__, hash(self.numerator), hash(self.denominator)) )

    def __contains__(self, string):
        """
        Return **True** if `string` is equal to the RatioKeyString's numerator or denominator
        """
        return self.numerator == string or self.denominator == string

    def _filter(self, key_eq=None, key_neq=None, numerator = {}, denominator = {}, **invalid):
        if invalid:
            return False
        if key_eq is not None and self not in key_eq:
            return False
        if key_neq is not None and self in key_neq:
            return False
        if numerator and not self.numerator._filter(**numerator):
            return False
        if denominator and not self.denominator._filter(**denominator):
            return False

        return True

    def _sortkey(self):
        return f'{self.denominator._sortkey()}/{self.numerator._sortkey()}'


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
    def __new__(cls, string, *, allow_reformatting=True):

        if isinstance(string, cls):
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

        string = string.removeprefix('GEN_') #For backwards compatibility
        #colname = string.replace('/', '_SLASH_') #For backwards compatibility
        string = string.replace('_SLASH_', '/') #For backwards compatibility
        return super(GeneralKeyString, cls).__new__(cls, string)

    def _filter(self, key_eq=None, key_neq=None, **invalid):
        if invalid:
            return False
        if key_eq is not None and self not in key_eq:
            return False
        if key_neq is not None and self in key_neq:
            return False

        return True

    def _sortkey(self):
        return str(self)


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
    def __new__(cls, *keys, skip_duplicates = False,
                allow_duplicates = True, allow_reformatting = True):
        new_keys = []
        for key in keys:
            if isinstance(key, str):
                new_keys.append(cls._Flavour__keystring(key, allow_reformatting=allow_reformatting))
            elif isinstance(key, np.dtype) and key.names is not None:
                new_keys.extend([cls._Flavour__keystring(name, allow_reformatting=allow_reformatting) for name in key.names])
            elif isinstance(key, ndarray) and key.dtype.names is not None:
                new_keys.extend([cls._Flavour__keystring(name, allow_reformatting=allow_reformatting) for name in key.dtype.names])
            elif hasattr(key, '__iter__'):
                new_keys.extend([cls._Flavour__keystring(k, allow_reformatting=allow_reformatting) for k in key])
            else:
                new_keys.append(cls._Flavour__keystring(key, allow_reformatting=allow_reformatting))

        if skip_duplicates:
            new_keys = list(dict.fromkeys(new_keys).keys())
        elif not allow_duplicates and (len(set(new_keys)) != len(new_keys)):
            raise ListDuplicateError(cls, new_keys)

        return super(IsopyKeyList, cls).__new__(cls, new_keys)

    def __call__(self):
        return self

    def __hash__(self):
        return super(IsopyKeyList, self).__hash__()
        return tuple(k for k in self).__hash__()

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

    def __le(self, other):
        if not isinstance(other, IsopyKeyList):
            try:
                other = self._Flavour__keylist(other)
            except:
                return False

        for key in self:
            if key not in other: return False
            if self.count(key) > other.count(key): return False
        return True

    def __lt(self, other):
        if not isinstance(other, IsopyKeyList):
            try:
                other = self._Flavour__keylist(other)
            except:
                return False

        return self.__le__(other) and (len(self) != len(other))

    def __ge(self, other):
        if not isinstance(other, IsopyKeyList):
            try:
                other = self._Flavour__keylist(other)
            except:
                return False
        return other.__le__(self)

    def __gt(self, other):
        if not isinstance(other, IsopyKeyList):
            try:
                other = self._Flavour__keylist(other)
            except:
                return False
        return other.__lt__(self)

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
            return self._Flavour__keylist(*super(IsopyKeyList, self).__getitem__(index))
        elif hasattr(index, '__iter__'):
                return self._Flavour__keylist(*(super(IsopyKeyList, self).__getitem__(i) for i in index))
        else:
            return super(IsopyKeyList, self).__getitem__(index)

    def __truediv__(self, denominator):
        if isinstance(denominator, (tuple, list)):
            if len(denominator) != len(self): raise ListSizeError(self, denominator)
            return RatioFlavour._Flavour__keylist(*(n / denominator[i] for i, n in enumerate(self)))
        else:
            return RatioFlavour._Flavour__keylist(*(n / denominator for i, n in enumerate(self)))

    def __rtruediv__(self, numerator):
        if isinstance(numerator, (tuple, list)):
            if len(numerator) != len(self): raise ListSizeError(self, numerator)
            return RatioFlavour._Flavour__keylist(*(numerator[i] / d for i, d in enumerate(self)))
        else:
            return RatioFlavour._Flavour__keylist(*(numerator / d for i, d in enumerate(self)))

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            other = self._Flavour__keylist(other)
        return self._Flavour__keylist(*super(IsopyKeyList, self).__add__(other))

    def __radd__(self, other):
        if not isinstance(other, self.__class__):
            other = self._Flavour__keylist(other)
        return other.__add__(self)

    def __sub__(self, other):
        if not isinstance(other, self.__class__):
            other = self._Flavour__keylist(other)
        out = list(self)
        for key in other:
            if key in out: out.remove(key)
        return self._Flavour__keylist(*out)

    def __rsub__(self, other):
        if not isinstance(other, self.__class__):
            other = self._Flavour__keylist(other)
        return other.__sub__(self)

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

    def strlist(self):
        """
        Return a list of ``str`` object for each key in the key list.

        Analogous to ``[str(key) for key in keylist]``
        """
        return [str(key) for key in self]

    def sorted(self):
        """
        Return a sorted key string list.
        """
        return self._Flavour__keylist(sorted(self, key= lambda k: k._sortkey()))

    def reversed(self):
        """
        Return a reversed key string list.
        """
        return self._Flavour__keylist(self.sorted())

    ##########################################
    ### Cached methods for increased speed ###
    ##########################################

    @lru_cache(CACHE_MAXSIZE)
    def __and__(self, *others):
        result = self
        for other in others:
            if type(other) is not type(result):
                try:
                    other = result._Flavour__keylist(other)
                except KeyParseError:
                    result = GeneralFlavour._Flavour__keylist(result)
                    other = GeneralFlavour._Flavour__keylist(other)
            this = list(dict.fromkeys(result))
            other = list(dict.fromkeys(other))
            other_hash = [hash(o) for o in other]
            result = result._Flavour__keylist(*(t for t in this if hash(t) in other_hash))
        return result

    @lru_cache(CACHE_MAXSIZE)
    def __or__(self, *others):
        result = self
        for other in others:
            if type(other) is not type(result):
                try:
                    other = result._Flavour__keylist(other)
                except KeyParseError:
                    result = GeneralFlavour._Flavour__keylist(result)
                    other = GeneralFlavour._Flavour__keylist(other)
            result = result._Flavour__keylist(*dict.fromkeys((*result, *other)))
        return result

    @lru_cache(CACHE_MAXSIZE)
    def __xor__(self, *others):
        result = self
        for other in others:
            if type(other) is not type(result):
                try:
                    other = result._Flavour__keylist(other)
                except KeyParseError:
                    result = GeneralFlavour._Flavour__keylist(result)
                    other = GeneralFlavour._Flavour__keylist(other)

            this = list(dict.fromkeys(result))
            other = list(dict.fromkeys(other))
            this_hash = [hash(t) for t in this]
            other_hash = [hash(o) for o in other]
            result = result._Flavour__keylist(*(t for t in this if hash(t) not in other_hash),
                                              *(o for o in other if hash(o) not in this_hash))
        return result

    @lru_cache(CACHE_MAXSIZE)
    def __rand__(self, other):
        if type(other) is not type(self):
            try:
                other = self._Flavour__keylist(other)
            except:
                return GeneralFlavour._Flavour__keylist(self).__rand__(other)
        return other.__and__(self)

    @lru_cache(CACHE_MAXSIZE)
    def __ror__(self, other):
        if type(other) is not type(self):
            try:
                other = self._Flavour__keylist(other)
            except:
                return GeneralFlavour._Flavour__keylist(self).__ror__(other)
        return other.__or__(self)

    @lru_cache(CACHE_MAXSIZE)
    def __rxor__(self, other):
        if type(other) is not type(self):
            try:
                other = self._Flavour__keylist(other)
            except:
                return GeneralFlavour._Flavour__keylist(self).__rxor__(other)
        return other.__xor__(self)


class MassKeyList(MassFlavour, IsopyKeyList):
    """
    A tuple consisting exclusively of :class:`MassKeyString` items.

    Behaves just like, and contains all the methods, that a normal tuple does unless otherwise noted.
    Only methods that behave differently from a normal python tuples are documented below.

    Parameters
    ----------
    keys : str, int,  Sequence[(str, int)]
        A string or sequence of strings that can be converted to the correct key string type.
    skip_duplicates : bool, Default = True
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

    def filter(self, key_eq=None, key_neq=None, *, key_lt=None,
               key_gt=None, key_le=None, key_ge=None):
        """
        Return a new key string list containing only the key strings that satisfies the specified
        filters.

        Parameters
        ----------
        key_eq : str, int, Sequence[(str, int)]
           Only key strings equal to/found in *key_eq* pass this filter.
        key_neq : str, int, Sequence[(str, int)]
           Only key strings not equal to/found in *key_neq* pass this filter.
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
        >>> keylist.filter(key_eq=[105, 107, 111])
        ('105', '111')
        >>> keylist.filter(key_neq='111'])
        ('99', '105')
        >>> keylist.filter(key_gt='99'])
        ('105', '111')
        >>> keylist.filter(key_le=105])
        ('99', '105')
        """

        filters = parse_keyfilters(key_eq=key_eq, key_neq=key_neq,
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
    skip_duplicates : bool, Default = True
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

    def filter(self, key_eq = None, key_neq = None):
        """
        Return a new key string list containing only the key strings that satisfies the specified
        filters.


        Parameters
        ----------
        key_eq : str, Sequence[str], Optional
           Only key strings equal to/found in *key_eq* pass this filter.
        key_neq : str, Sequence[str], Optional
           Only key strings not equal to/found in *key_neq* pass this filter.


        Returns
        -------
        result : ElementKeyList
            Key strings in the sequence that satisfy the specified filters

        Examples
        --------
        >>> keylist = ElementKeyList(['ru', 'pd', 'cd'])
        >>> keylist.filter(key_eq=['pd','ag', 'cd'])
        ('Pd', 'Cd')
        >>> keylist.filter(key_neq='cd')
        ('Ru', 'Pd')
        """
        filters = parse_keyfilters(key_eq=key_eq, key_neq=key_neq)
        return self._Flavour__keylist(key for key in self if key._filter(**filters))


class IsotopeKeyList(IsotopeFlavour, IsopyKeyList):
    """
    A tuple consisting exclusively of :class:`IsotopeKeyString` items.

    Behaves just like, and contains all the methods, that a normal tuple does unless otherwise noted.
    Only methods that behave differently from a normal python tuples are documented below.

    Parameters
    ----------
    keys : str, Sequence[str]
        A string or sequence of strings that can be converted to the correct key string type.
    skip_duplicates : bool, Default = True
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

    @property
    @functools.cache
    def mass_numbers(self) -> MassKeyList:
        """
        Returns a :class:`MassKeyList` containing the mass number of each item in the list.

        Examples
        --------
        >>> IsotopeKeyList(['99ru', '105pd', '111cd']).mass_numbers()
        ('99', '105', '111')
        """
        return MassFlavour._Flavour__keylist(*(x.mass_number for x in self))

    @property
    @functools.cache
    def element_symbols(self) -> ElementKeyList:
        """
        Returns an :class:`ElementKeyList` containing the element symbol of each item in the list.

        Examples
        --------
        >>> IsotopeKeyList(['99ru', '105pd', '111cd']).element_symbols()
        ('Ru', 'Pd', 'Cd')
        """
        return ElementFlavour._Flavour__keylist(*(x.element_symbol for x in self))

    def filter(self, key_eq = None,
               key_neq = None,
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
        >>> keylist.filter(key_eq=['105pd','107ag', '111cd'])
        ('105Pd', '111Cd)
        >>> keylist.filter(key_neq='111cd')
        ('99Ru', '105Pd')
        >>> keylist.filter(mass_number_key_gt = 100)
        ('105Pd', '111Cd)
        >>> keylist.filter(element_symbol_key_neq='pd'])
        ('99Ru', '111Cd)
        """
        filters = parse_keyfilters(key_eq=key_eq, key_neq=key_neq,
                                   **mass_number_and_element_symbol_kwargs)

        return self._Flavour__keylist(key for key in self if key._filter(**filters))


class RatioKeyList(RatioFlavour, IsopyKeyList):
    """
    A tuple consisting exclusively of :class:`RatioKeyString` items.

    Behaves just like, and contains all the methods, that a normal tuple does unless otherwise noted.
    Only methods that behave differently from a normal python tuples are documented below.

    Parameters
    ----------
    keys : str, tuple[str, str], Sequence[str, tuple[str, str]]
        A string or sequence of strings that can be converted to the correct key string type.
    skip_duplicates : bool, Default = True
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

    @property
    @functools.cache
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

    @property
    @functools.cache
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

    @property
    @functools.cache
    def common_denominator(self) -> IsopyKeyString:
        """
        The common demoninator of all ratios in the sequence.

        Raises
        ------
        NoCommonDenominator
            Raised if there is no common denominator in the sequence

        Examples
        --------
        >>> RatioKeyList(['99ru/108Pd', '105pd/108Pd', '111cd/108Pd']).common_denominator
        '108Pd'
        """

        if (l:=len(set(self.denominators))) == 0:
            raise NoCommomDenominator('RatioKeyList is empty')
        elif l == 1:
            return self.denominators[0]
        else:
            raise NoCommomDenominator('RatioKeyList does not have a common denominator')

    @property
    @functools.cache
    def has_common_denominator(self) -> bool:
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

    def filter(self, key_eq = None, key_neq = None, **numerator_and_denominator_kwargs):
        """
        Return a new key string list containing only the key strings that satisfies the specified
        filters.

        Parameters
        ----------
        key_eq : str, Sequence[str], Optional
           Only key strings equal to/found in *key_eq* pass this filter.
        key_neq : str, Sequence[str], Optional
           Only key strings not equal to/found in *key_neq* pass this filter.
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
        >>> keylist.filter(key_eq=['105pd/108Pd','107ag/108Pd', '111cd/108Pd'])
        ('105Pd/108Pd', '111Cd/108Pd')
        >>> keylist.filter(key_neq='111cd/108Pd')
        ('99Ru/108Pd', '105Pd/108Pd')
        >>> keylist.filter(numerator_isotope_symbol_key_eq = ['pd', 'ag', 'cd'])
        ('105Pd/108Pd', '111Cd/108Pd')
        >>> keylist.filter(numerator_mass_number_key_lt = 100)
        ('99Ru/108Pd', '105Pd/108Pd')
        """
        filters = parse_keyfilters(key_eq=key_eq, key_neq=key_neq,
                                   **numerator_and_denominator_kwargs)
        return self._Flavour__keylist(key for key in self if key._filter(**filters))


class GeneralKeyList(GeneralFlavour, IsopyKeyList):
    """
    A tuple consisting exclusively of :class:`GeneralKeyList` items.

    Behaves just like, and contains all the methods, that a normal tuple does unless otherwise noted.
    Only methods that behave differently from a normal python tuples are documented below.

    Parameters
    ----------
    keys : str, Sequence[str]
        A string or sequence of strings that can be converted to the correct key string type.
    skip_duplicates : bool, Default = True
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
    def filter(self, key_eq= None, key_neq = None):
        """
        Return a new key string list containing only the key strings that satisfies the specified
        filters.

        Parameters
        ----------
        key_eq : str, Sequence[str], Optional
           Only key strings equal to/found in *key_eq* pass this filter.
        key_neq : str, Sequence[str], Optional
           Only key strings not equal to/found in *key_neq* pass this filter.

        Returns
        -------
        result : GeneralKeyList
            Key strings in the sequence that satisfy the specified filters

        Examples
        --------
        >>> keylist = GeneralKeyList(['harry', 'ron', 'hermione'])
        >>> keylist.filter(key_eq=['harry', 'ron', 'neville'])
        ('harry', 'ron')
        >>> keylist.filter(key_neq='harry')
        ('ron', 'hermione')
        """

        filters = parse_keyfilters(key_eq=key_eq, key_neq=key_neq)
        return self._Flavour__keylist(key for key in self if key._filter(**filters))

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

    def __init__(self, *args, default_value = None, readonly =False, **kwargs):
        super(IsopyDict, self).__init__()
        self._readonly = False
        self._default_value = default_value

        for arg in args:
            if isinstance(arg, dict):
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

    def get(self, key=None, default=NotGiven, **key_filters):
        """
        Return the the value for *key* if present in the dictionary. Otherwise *default* is
        returned. If *key* is a sequence of keys a dictionary is returned. If *default* is
        not given the default value of the dictionary is used.

        *key_filters* will only be processed if *key* is not given. A new dictionary will be returned
        containing only they keys in the current dictonrary that passes these filters.

        Examples
        --------
        >>> reference = isopy.IsopyDict({'108Pd': 100, '105Pd': 20, '104Pd': 150})
        >>> reference.get('pd108')
        100
        >>> reference.get('104Pd/105Pd')
        None
        >>> reference.get('104Pd/105Pd', default=np.nan)
        nan
        >>> reference.get(mass_number_lt = 106)
        {IsotopeKeyString('105Pd'): 20, IsotopeKeyString('104Pd'): 150}


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
                return default

        if hasattr(key, '__iter__'):
            return self.__class__({k: self.get(k, default) for k in key},
                                  default_value=default)

        if len(key_filters) > 0:
            key_filters = parse_keyfilters(**key_filters)
            keys = [k for k in self if k._filter(**key_filters)]
            return self.get(keys)

        if key is None and len(key_filters) == 0:
            raise KeyError('a key or at least one key filter must be given')
        else:
            raise TypeError(f'key type {type(key)} not understood')

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
        if self._readonly is True:
            raise TypeError('this dictionary is readonly. Make a copy to make changes')

        key = askeystring(key)
        if default is not NotGiven:
            default = self._default_value
        if key not in self:
            if default is not NotGiven:
                raise ValueError('No default value given')
            else:
                self.__setitem__(key, default)

        return self.__getitem__(key)

    def copy(self):
        """Returns a copy of the current dictionary."""
        return self.__class__(self, default_value = self._default_value, get_divide=self.get_divide)

    def clear(self):
        """
        Removes all items from the dictionary.

        A TypeError is raised if the dictionary is readonly.
        """
        if self._readonly is True:
            raise TypeError('this dictionary is readonly. Make a copy to make changes')
        super(IsopyDict, self).clear()


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
        super(ScalarDict, self).__init__(*args, default_value=default_value, readonly=readonly, **kwargs)

    def __setitem__(self, key, value):
        try:
            value = np.float_(value)
        except Exception as err:
            raise ValueError(f'unable to convert value for key "{key}" to float') from err

        super(ScalarDict, self).__setitem__(key, value)

    def get(self, key = None,
            default = NotGiven,
            **key_filters: Any
            ):
        """
        Return the the value for *key* if present in the dictionary. If *key* not in the dictonary
        and *key* is a ratio key string the numerator value divided by the denominator value will
        be returned if both key strings are present in the dictionary. Otherwise *default* is
        returned. If *key* is a sequence of keys a dictionary is returned. If *default* is
        not given the default value of the dictionary is used.

        *key_filters* will only be processed if *key* is not given. A new dictionary will be returned
        containing only they keys in the current dictonrary that passes these filters.

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

        if isinstance(key, (str, int)):
            key = askeystring(key)

            try:
                return super(IsopyDict, self).__getitem__(key)
            except KeyError:
                if type(key) is RatioKeyString:
                    try:
                        result = super(IsopyDict, self).__getitem__(key.numerator)
                        result /= super(IsopyDict, self).__getitem__(key.denominator)
                        return result
                    except KeyError:
                        pass

                return default
        else:
            return super(ScalarDict, self).get(key, default, **key_filters)

def parse_keyfilters(**filters):
    """
    Parses key filters into a format that is accepted by KeyString._filter. Allows
    nesting of filter arguments for Isotope and RatioKeyString's.
    """
    #Make sure these are lists
    for key in ['key_eq', 'key_neq']:
        if (f:=filters.get(key, None)) is not None and not isinstance(f, (list, tuple)):
            filters[key] = (f,)

    #For isotope keys
    mass_number = _split_filter('mass_number', filters)
    if mass_number:
        filters['mass_number'] = parse_keyfilters(**mass_number)

    element_symbol = _split_filter('element_symbol', filters)
    if element_symbol:
        filters['element_symbol'] = parse_keyfilters(**element_symbol)

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
        elif key.startswith(prefix):
            filter = filters.pop(key)
            key = key.removeprefix(prefix)
            if key in ('eq', 'neq', 'lt', 'le', 'ge', 'gt'):
                key = f'key_{key}'
            out[key] = filter
    return out


#############
### Array ###
#############
class IsopyArray(IsopyFlavour):
    def __repr__(self):
        return self.to_text(**ARRAY_REPR)
        if self.ndim == 0:
            out = '({})'.format(', '.join([f'{self[k]}' for k in self.keys()]))
        else:
            out = '[{}]'.format(',\n'.join(['({})'.format(', '.join(
                [f'{self[k][i]}' for k in self.keys()])) for i in range(self.size)]))
        return f'{self.__class__.__name__}(\n{out},\n{self.dtype!r})'

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
            return np.all(np.equal(self, other))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != '__call__':
            raise ValueError('only the ufunc method "__call__" is supported by isopy arrays')

        if ufunc in reimplmented_functions:

            return reimplmented_functions[ufunc](*inputs, **kwargs)

        else:

            return array_function(ufunc, *inputs, **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        if func in reimplmented_functions:

            return reimplmented_functions[func](*args, **kwargs)

        else:
            # Incase input arguments were given as keyword arguments
            fargs, nin = _function_signature(func)
            if fargs and len(args) != nin:
                kwargs.update(zip(fargs, args))
                try:
                    args = tuple(kwargs.pop(fargs[i]) for i in range(nin))
                except:
                    #I dont think this would ever happen as it would be caught by the
                    #array function dispatcher.
                    raise ValueError(f'"{func.__name__}" expects {nin} input arguments but only got {len(args)}')

            #Some scipy stat functions sets
            if 'axis' in fargs and 'axis' not in kwargs:
                kwargs['axis'] = NotGiven

            return array_function(func, *args, **kwargs)

    def _view(self, obj):
        if obj.dtype.names:
            if isinstance(obj, void):
                return self._Flavour__view_void(obj)
            else:
                return self._Flavour__view_ndarray(obj)
        else:
            return obj.view(ndarray)

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

    def copy(self, **key_filter):
        """
        Returns a copy of the array. If *key_filters* are given then the returned array only
        contains the columns that satisfy the *key_filter* filters.
        """
        if key_filter:
            out_keys = self.keys().filter(**key_filter)
            return self[out_keys].copy()
        else:
            return super(IsopyArray, self).copy()

    def ratio(self, denominator, remove_denominator = True):
        """
        Divide all values in the array by the *denominator* column and return a :class:`RatioArray`.
        If *remove_denominator* is ``True`` the denominator/denominator ratio is not included in
        the returned array.
        """
        keys = self.keys()
        if denominator not in keys:
            raise ValueError(f'key "{denominator}" not found in keys of the array')
        elif remove_denominator:
            keys = keys - denominator

        return RatioArray(self[keys] / self[denominator], keys=keys/denominator)

    @property
    def keys(self):
        """
        A key string list containing the name of each column in the array.

        ``array.keys()`` is also allowed as calling a key string list will just return a
        pointer to itself.
        """
        return self._Flavour__keylist(*self.dtype.names, allow_reformatting=False)

    @property
    def ncols(self) -> int:
        """Number of columns in the array"""
        return len(self.dtype.names)

    @property
    def nrows(self)-> int:
        """
        Number of rows in the array. If the array is 0-dimensional ``-1`` is returned.
        """
        if self.ndim == 0:
            return -1
        else:
            return self.size

    def to_text(self, delimiter=', ', include_row = False, include_dtype=False,
                nrows = None, **vformat) -> str:
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
                sdict[title] = [vformat.get(val.dtype.kind, '{}').format(self[k][i]) for i in range(self.size)]

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

        return '{}\n{}'.format(delimiter.join(['{:<{}}'.format(k, flen[k]) for k in sdict.keys()]),
                                   '\n'.join('{}'.format(delimiter.join('{:<{}}'.format(sdict[k][i], flen[k]) for k in sdict.keys()))
                                             for i in range(nrows)))

    def to_list(self) -> Union[ list, list[list]]:
        """
        Return a list containing the data in the array.
        """
        if self.ndim == 0:
            return list(self.tolist())
        else:
            return [list(row) for row in self.tolist()]

    def to_dict(self) -> dict[str, Union[Any, list]]:
        """
        Return a dictionary containing the data in the array.
        """
        return {str(key): self[key].tolist() for key in self.keys()}

    def to_clipboard(self, delimiter: str=', ', include_row: bool = False,
                     include_dtype: bool = False, vformat: Optional[ dict[str, str] ] = None
                     ) -> None:
        """
        Copy the string returned from ``array.pformat(*args, **kwargs)`` to the clipboard.
        """
        string = self.to_text(delimiter=delimiter, include_row=include_row, include_dtype=include_dtype, vformat=vformat)
        pyperclip.copy(string)
        return string

    def to_ndarray(self) -> np.ndarray:
        """Return a copy of the array as a normal numpy ndarray"""
        return self.view(ndarray).copy()

    def to_csv(self, filename: str, comments: Optional[Union[ str, list[str] ]] = None) -> None:
        """
        Save array to a cv file. If *filename* already exits it will be overwritten. If *comments*
        are given they will be included before the array data.
        """
        isopy.write_csv(filename, self, comments=comments)

    def to_xlsx(self, filename: str, sheetname: Optional[str] = None,
                comments:Optional[Union[ str, list[str] ]] = None, append: bool = False) -> None:
        """
        Save array to a excel workbook. If *sheetname* is not given the array will be saved as
        "sheet1". If *filename* exists and *append* is ``True`` the sheet will be added to the
        existing workbook. Otherwie the existing file will be overwritten. If *comments* are given
        they will be included before the array data.
        """
        if sheetname is None:
            sheetname = 'sheet1'

        # TODO if sheetname is given and file exits open it and add the sheet, overwrite if nessecary
        isopy.write_xlsx(filename, comments=comments,  **{sheetname: self})

    def to_dataframe(self):
        """
        Convert array to a pandas dataframe. An exception is raised if pandas is not installed.
        """
        if pandas is not None:
            return pandas.DataFrame(self.to_dict())
        else:
            raise TypeError('Pandas is not installed')

    @classmethod
    def from_xlsx(cls, filename: str, sheetname: Union[ str, int ]):
        """
        Load array from *sheetname* in  workbook with *filename* and return as an array of this type
        """
        return cls._Flavour__array(isopy.read_xlsx(filename, sheetname))

    @classmethod
    def from_csv(cls, filename):
        """
        Load array from csv file with *filename* and return as an array of this type
        """
        return cls._Flavour__array(isopy.read_csv(filename))


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


###############
### Ndarray ###
###############
class IsopyNdarray(IsopyArray, ndarray):
    def __new__(cls, values, keys=None, *, dtype=None, ndim=None):
        if type(values) is cls and keys is None and dtype is None and ndim is None:
            return values.copy()

        #Do this early so no time is wasted if it fails
        if keys is not None:
            keys = cls._Flavour__keylist(keys, allow_duplicates=False)

        if ndim is not None and (not isinstance(ndim , int) or ndim < -1 or ndim > 1):
            raise ValueError('parameter "ndim" must be -1, 0 or 1')

        if keys is None and type(dtype) is np.dtype and dtype.names is not None:
            keys = list(dtype.names)

        if pandas is not None and isinstance(values, pandas.DataFrame):
            values = values.to_records()

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
            raise NoKeysError('Keys argument not given and keys not found in values')
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
                raise NDimError(f'Cannot convert array with {out.size} rows to 0-dimensions')
            else:
                out = out.reshape(tuple())

        return out.view(cls)

    def __setitem__(self, key, value):
        if isinstance(key, str):
            try:
                key = str(self._Flavour__keystring(key))
            except:
                pass
        elif isinstance(key, list):
            try:
                key = self._Flavour__keylist(key).strlist()
            except:
                pass
        super(IsopyNdarray, self).__setitem__(key, value)

    def __getitem__(self, key):
        if isinstance(key, str):
            try:
                key = str(self._Flavour__keystring(key))
            except KeyParseError:
                pass
            else:
                return super(IsopyNdarray, self).__getitem__(key).view(ndarray)
            raise KeyError(f'No column with the key "{key}" in array')

        elif isinstance(key, (list,tuple)):
            if len(key) == 0:
                return np.array([])
            elif isinstance(key[0], str):
                try:
                    key = self._Flavour__keylist(key).strlist()
                except:
                    pass
                else:
                  return self._view(super(IsopyNdarray, self).__getitem__(key))
            else:
                return self._view(super(IsopyNdarray, self).__getitem__(key))

        else:
            return self._Flavour__view_void(super(IsopyNdarray, self).__getitem__(key))


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
            raise IndexError('{} cannot be indexed by position'.format(self.__class__.__name__))
        elif isinstance(key, str):
            try:
                key = str(self._Flavour__keystring(key))
            except:
                pass
        super(IsopyArray, self).__setitem__(key, value)

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
                    key = self._Flavour__keylist(key).strlist()
                except:
                    pass
            else:
                key = list(key)

        if isinstance(key, (int, slice)):
            raise IndexError('0-dimensional arrays cannot be indexed by row'.format(self.__class__.__name__))
        return super(IsopyVoid, self).__getitem__(key)

    def reshape(self, shape):
        return self._view(self.reshape(shape))


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

###############################################
### functions for creating isopy data types ###
###############################################
@lru_cache(CACHE_MAXSIZE)
def keystring(string, *, allow_reformatting=True):
    """
    Convert *string* into a key string and return it.

    Will attempt to convert *string* into a MassKeyString, ElementKeyString, IsotopeKeyString,
    RatioKeyString and finally a GeneralKeyString. The first successfully conversion is returned.
    """
    try:
        return MassKeyString(string, allow_reformatting=allow_reformatting)
    except KeyStringException:
        pass

    try:
        return ElementKeyString(string, allow_reformatting=allow_reformatting)
    except KeyStringException:
        pass

    try:
        return IsotopeKeyString(string, allow_reformatting=allow_reformatting)
    except KeyStringException:
        pass

    try:
        return RatioKeyString(string, allow_reformatting=allow_reformatting)
    except KeyStringException:
        pass

    try:
        return GeneralKeyString(string, allow_reformatting=allow_reformatting)
    except KeyStringException:
        pass

    raise KeyParseError(string, IsopyKeyString, f'unable to parse {type(string)} "{string}" into a key string')


@lru_cache(CACHE_MAXSIZE)
def askeystring(key, *, allow_reformatting=True):
    """
    If *string* is a key string it is returned otherwise convert *string* to a key string and
    return it.

    Will attempt to convert *string* into a MassKeyString, ElementKeyString, IsotopeKeyString,
    RatioKeyString and finally a GeneralKeyString. The first successfully conversion is returned.
    """
    if isinstance(key, IsopyKeyString):
        return key
    else:
        return keystring(key, allow_reformatting=allow_reformatting)


@lru_cache(CACHE_MAXSIZE)
def keylist(*keys, skip_duplicates=False, allow_duplicates=True, allow_reformatting=True):
    """
    Convert *keys* into a key string list and return it. *keys* can be a string or a sequence of
    string.

    Will attempt to convert *keys* into a MassKeyList, ElementKeyList, IsotopeKeyList,
    RatioKeyList and finally a GeneralKeyList. The first successfully conversion is returned.
    """
    try:
        return MassKeyList(*keys, skip_duplicates=skip_duplicates,
                           allow_duplicates=allow_duplicates, allow_reformatting=allow_reformatting)
    except KeyStringException:
        pass

    try:
        return ElementKeyList(*keys, skip_duplicates=skip_duplicates,
                              allow_duplicates=allow_duplicates, allow_reformatting=allow_reformatting)
    except KeyStringException:
        pass

    try:
        return IsotopeKeyList(*keys, skip_duplicates=skip_duplicates,
                              allow_duplicates=allow_duplicates, allow_reformatting=allow_reformatting)
    except KeyStringException:
        pass

    try:
        return RatioKeyList(*keys, skip_duplicates=skip_duplicates,
                           allow_duplicates=allow_duplicates, allow_reformatting=allow_reformatting)
    except KeyStringException:
        pass

    try:
        return GeneralKeyList(*keys, skip_duplicates=skip_duplicates,
                              allow_duplicates=allow_duplicates, allow_reformatting=allow_reformatting)
    except KeyStringException:
        pass

    raise KeyParseError('unable to parse keys into a key string list')


@lru_cache(CACHE_MAXSIZE)
def askeylist(keys, allow_reformatting=True):
    """
    If *keys* is a key string list return it otherwise convert *keys* to a key string list and
    return it.

    Will attempt to convert *keys* into a MassKeyList, ElementKeyList, IsotopeKeyList,
    RatioKeyList and finally a GeneralKeyList. The first successfully conversion is returned.
    """
    if isinstance(keys, IsopyKeyList):
        return keys

    keys = [askeystring(key, allow_reformatting=allow_reformatting) for key in keys]
    types = {flavour(key) for key in keys}

    if len(types) == 1:
        return types.pop()._Flavour__keylist(keys)
    else:
        return GeneralKeyList(keys, allow_reformatting=allow_reformatting)


def array(values=None, keys=None, *, dtype=None, ndim=None, **columns):
    """
    Convert the input arguments to a isopy array.

    Will attempt to convert the input arguments into a MassArray, ElementArray, IsotopeArray,
    RatioArray and finally a GeneralArray. The first successfully conversion is returned.
    """
    if values is None and len(columns) == 0:
        raise ValueError('No values were given')
    elif values is not None and len(columns) != 0:
        raise ValueError('values and column kwargs cannot be given together')
    elif values is None:
        values = columns

    try: return MassArray(values, keys=keys, dtype=dtype, ndim=ndim)
    except KeyStringException: pass

    try: return ElementArray(values, keys=keys, dtype=dtype, ndim=ndim)
    except KeyStringException: pass

    try: return IsotopeArray(values, keys=keys, dtype=dtype, ndim=ndim)
    except KeyStringException: pass

    try: return RatioArray(values, keys=keys, dtype=dtype, ndim=ndim)
    except KeyStringException: pass

    try: return GeneralArray(values, keys=keys, dtype=dtype, ndim=ndim)
    except KeyStringException: pass

    raise KeyParseError('Unable to convert input to IsopyArray')


def asarray(a, *, ndim = None):
    """
    If *a* is an isopy array return it otherwise convert *a* into an isopy array and return it. If
    *ndim* is given a view of the array with the specified dimensionality is returned.
    """
    if not isinstance(a, IsopyArray):
        a = array(a)

    if ndim is not None:
        if ndim <= 0 and a.ndim != 0:
            if a.size == 1:
                a = a.reshape(tuple())
            elif ndim != -1:
                raise ValueError('cannot make array with more than one value zero-dimensional')
        elif ndim > 0 and a.ndim == 0:
            a = a.reshape(-1)
        elif ndim < -1 or ndim > 1:
            raise ValueError('the only accepted ndim values are -1, 0, and 1')

    return a


def asanyarray(a, *, dtype = None, ndim = None):
    """
    Return ``isopy.asarray(a)`` if *a* possible otherwise return ``numpy.asanyarray(a)``.

    The data type and number of dimensions of the returned array can be specified by *dtype and
    *ndim*, respectively.
    """
    if a is None: return None

    if isinstance(a, IsopyArray) and dtype is None:
        pass

    elif isinstance(a, dict) or (isinstance(a, ndarray) and a.dtype.names is not None):
        try:
            a=array(a, dtype=dtype)
        except:
            pass

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
            raise err

    if ndim is not None:
        if ndim == a.ndim:
            pass
        elif ndim <= 0 and a.ndim != 0:
            if a.size == 1:
                a = a.reshape(tuple())
            elif ndim != -1:
                raise ValueError('cannot make array with more than one value zero-dimensional')
        elif ndim == 1 and a.ndim != ndim:
            a = a.reshape(-1)
        elif not isinstance(a, IsopyArray) and ndim == 2 and a.ndim != ndim:
            a = a.reshape(-1, a.size)
        else:
            raise ValueError(f'Unable to convert array to number of dimensions specified {ndim}')

    return a


##################
### type tests ###
##################
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


###########################
### Create empty arrays ###
###########################
def zeros(rows, keys=None, *, ndim=None, dtype=float64):
    """
    Create an isopy array filled with zeros.

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
        different datatypes for different columns in the final array.
    """
    return _new_empty_array(rows, keys, ndim, dtype, np.zeros)


def ones(rows, keys=None, *, ndim=None, dtype=float64):
    """
    Create an isopy array filled with ones.


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
        different datatypes for different columns in the final array.
    """
    return _new_empty_array(rows, keys, ndim, dtype, np.ones)


def empty(rows, keys=None, *, ndim=None, dtype=float64):
    """
    Create an isopy array without initalising entries.


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
        different datatypes for different columns in the final array.
    """
    return _new_empty_array(rows, keys, ndim, dtype, np.empty)


def full(rows, fill_value, *, keys=None, ndim = None, dtype=float64):
    """
    Create an isopy array filled with *fill_value*.


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
        different datatypes for different columns in the final array.
    """
    out = _new_empty_array(rows, keys, ndim, dtype, np.empty)
    return np.copyto(out, fill_value)


def _new_empty_array(rows, keys, ndim, dtype, func):
    if rows is None: rows = -1
    elif rows == tuple(): rows = -1
    elif rows == -1: pass
    elif rows < 1:
        raise ValueError('parameter "rows" must be -1 or a positive integer')

    if keys is not None:
        keys = keylist(keys, allow_duplicates=False)

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
            raise ValueError('cannot create an zero-dimensional array with more than one row')
    elif ndim < -1 or ndim > 1:
        raise ValueError('accepted values for "ndim" is -1, 0,  or 1')

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


def zeros_like(a, *, dtype=None):
    """
    Create and return an isopy array like *a* filled with zeros. If given, *dtype* will override
    the data type of the returned array. *dtype* can be a single data type of a sequence of
    data types, one for each column in *a*.
    """
    return _new_like_array(a, dtype, np.zeros_like)


def ones_like(a, *, dtype=None):
    """
    Create and return an isopy array like *a* filled with ones. If given, *dtype* will override
    the data type of the returned array. *dtype* can be a single data type of a sequence of
    data types, one for each column in *a*.
    """
    return _new_like_array(a, dtype, np.ones_like)


def empty_like(a, *, dtype=None):
    """
    Create and return an isopy array like *a* without initalised values. If given, *dtype* will override
    the data type of the returned array. *dtype* can be a single data type of a sequence of
    data types, one for each column in *a*.
    """
    return _new_like_array(a, dtype, np.empty_like)


def full_like(a, fill_value, *, dtype=None):
    """
    Create and return an isopy array like *a* filled with *fill_value*. If given, *dtype* will override
    the data type of the returned array. *dtype* can be a single data type of a sequence of
    data types, one for each column in *a*.
    """
    out = _new_like_array(a, dtype, np.empty_like)
    return np.copyto(out, fill_value)


def _new_like_array(a, dtype, func):
    if isinstance(a, ndarray) and a.dtype.names is not None:
        keys = keylist(a).strlist()
        if hasattr(dtype, '__iter__'):
            if len(dtype) != len(keys):
                raise ValueError('size of dtype does not match size of keys in array')
            else:
                dtype = list(zip(keys, dtype))
        elif isinstance(dtype, np.dtype) and len(dtype) > 1:
            if len(dtype) != len(keys):
                raise ValueError('size of dtype does not match size of keys in array')
            else:
                dtype = list(zip(keys, [dtype[i] for i in range(len(dtype))]))
        else:
            dtype = list(zip(keys, [dtype for i in range(len(a.dtype))]))
    return func(a, dtype)

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
    if len(arrays) == 1 and isinstance(arrays, (list, tuple)): arrays = arrays[0]
    arrays = [asarray(a) for a in arrays]
    arrays = [array.reshape(1) if array.ndim==0 else array for array in arrays]

    types = {type(a) for a in arrays}
    if len(types) != 1 or not isinstance((atype:=types.pop()), IsopyArray) :
        _list = keylist
    else:
        _list = atype._Flavour__keylist

    if axis == 0 or axis is None: #extend rows
        keys = atype._Flavour__keylist(*(a.dtype.names for a in arrays), skip_duplicates=True)
        #return atype({key: np.concatenate([a.get(key, default_value) for a in arrays]) for key in keys})

        result = [np.concatenate([a.get(key, default_value) for a in arrays]) for key in keys]
        dtype = [(key, result[i].dtype) for i, key in enumerate(keys.strlist())]
        return atype._Flavour__view_ndarray(np.fromiter(zip(*result), dtype=dtype))

    elif axis == 1: #extend columns
        size = {a.size for a in arrays}
        if len(size) != 1:
            raise ValueError('all arrays must have the same size concatenating in axis 1')

        keys = _list(*(a.dtype.names for a in arrays), allow_duplicates=False)

        result = []
        for a in arrays:
            for key in a.keys(): result.append(a.get(key, default_value))

        dtype = [(key, result[i].dtype) for i, key in enumerate(keys.strlist())]
        if result[0].ndim == 0:
            return keys._Flavour__view_ndarray(np.array(tuple(result), dtype=dtype))
        else:
            return keys._Flavour__view_ndarray(np.fromiter(zip(*result), dtype=dtype))

    else:
        raise np.AxisError(axis, 1, 'isopy.concatenate only accepts axis values of 0 or 1')


def argfunc(func, *args, **kwargs):
    parameters = _function_signature(func)
    kwargs.update(zip(parameters, args))
    axis = kwargs.pop('axis', 0)
    array = kwargs.pop(parameters[0])
    keys = array.keys()

    if axis == 0:
        result = [func(array.get(key), **kwargs) for key in keys]
        dtype = [(key, result[i].dtype) for i, key in enumerate(keys)]
        return keys._Flavour__view_ndarray(np.array(tuple(result), dtype=dtype))
    else:
        result = func([array.get(key) for key in keys], axis=axis, **kwargs)

        if axis == 1:
            return array._Flavour__keylist(*(keys[r] for r in result))
        elif axis is None:
            return array._Flavour__array({keys[int(result / array.size)]: result % array.size})
        else:
            raise ValueError(f'axis value "{axis}" unsupported')


reimplmented_functions = {np.concatenate: concatenate, np.append: concatenate}

def sortkeys(item):
    a = askeylist(item)
    return a._Flavour__keylist(sorted(a, key = [_sortval(key) for key in a]))


@lru_cache(128)
def _sortval(key):
    if isinstance(key, MassFlavour):
        return key
    elif isinstance(key, ElementFlavour):
        return str(isopy.refval.element.atomic_number.get(key, key))
    elif isinstance(key, IsotopeFlavour):
        return f'{int(key.mass_number)*1000}{isopy.refval.element.atomic_number.get(key, key)}'
    elif isinstance(key, RatioFlavour):
        return f'{_sortval(key.denominator)}/{_sortval(key.numerator)}'
    else:
        return key
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


def array_function(func, *inputs, axis=0, default_value= nan, **kwargs):
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

    new_inputs = []
    default_values = []
    keys = []
    for arg in inputs:
        if isinstance(arg, IsopyArray):
            new_inputs.append(arg)
            default_values.append(default_value)
            keys.append(arg.keys())
        elif isinstance(arg, ndarray) and len(arg.dtype) > 1:
            try:
                new_inputs.append(asarray(arg))
            except:
                new_inputs.append(arg)
            else:
                keys.append(new_inputs[-1].keys())
            default_values.append(default_value)
        elif isinstance(arg, IsopyDict):
            new_inputs.append(arg)
            default_values.append(arg.default_value)
        elif isinstance(arg, dict):
            new_inputs.append(IsopyDict(arg))
            default_values.append(default_value)
        else:
            new_inputs.append(_getter(arg))
            default_values.append(default_value)

    if len(keys) == 0:
        return func(*inputs, **kwargs)
    else:
        keys = keys[0].__or__(*keys[1:])

    if axis is NotGiven:
        kwargs['axis'] = None
        axis = 0

    if axis == 0:
        out = kwargs.get('out', None)
        keykwargs = {kw: kwargs.pop(kw) for kw in tuple(kwargs.keys())
                                if isinstance(kwargs[kw], IsopyArray)}

        result = [func(*(input.get(key, default_value) for input, default_value in zip(new_inputs, default_values)), **kwargs,
                       **{k: v.get(key) for k, v in keykwargs.items()}) for key in keys]
        if out is None:
            dtype = [(str(key), getattr(result[i], 'dtype', float64)) for i, key in enumerate(keys)]
            if getattr(result[0], 'ndim', 0) == 0:
                return keys._Flavour__view_ndarray(np.array(tuple(result), dtype=dtype))
            else:
                return keys._Flavour__view_ndarray(np.fromiter(zip(*result), dtype=dtype))
        else:
            return out

    else:
        if axis == 1: axis = 0
        for kwk, kwv in kwargs.items():
            if isinstance(kwv, (IsopyArray, dict)):
                kwargs['kwk'] = np.array([kwv.get(key, default_value) for key in keys])

        new_inputs = [[input.get(key, default_value) for key in keys] if
                      not isinstance(input, _getter) else input.get() for input in new_inputs]
        return func(*new_inputs, axis = axis, **kwargs)