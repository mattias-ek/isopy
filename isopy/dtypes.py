import numpy as np
import pyperclip
from tables import Table
import inspect
import isopy.functions as ipf
import warnings

pythonlist = list

__all__ = ['MassString', 'ElementString', 'IsotopeString', 'RatioString',
           'MassList', 'ElementList', 'IsotopeList', 'RatioList',
           'MassArray', 'ElementArray', 'IsotopeArray', 'RatioArray',
           'IsopyString', 'IsopyList', 'IsopyArray', 'ReferenceDict',
           'string', 'asstring', 'slist', 'asslist', 'array', 'asarray', 'asanyarray']

def string(item):
    try:
        return MassString(item)
    except:
        try:
            return ElementString(item)
        except:
            try:
                return IsotopeString(item)
            except:
                try:
                    return RatioString(item)
                except:
                    raise ValueError('Unable to parse "{}" into an IsopyString'.format(item))

def asstring(item):
    if isinstance(item, IsopyString):
        return item
    else:
        return string(item)

def slist(items, skip_duplicates=False, allow_duplicates=True):
    try:
        return MassList(items, skip_duplicates, allow_duplicates)
    except:
        try:
            return ElementList(items, skip_duplicates, allow_duplicates)
        except:
            try:
                return IsotopeList(items, skip_duplicates, allow_duplicates)
            except:
                try:
                    return RatioList(items, skip_duplicates, allow_duplicates)
                except:
                    raise ValueError('Unable to parse "{}" into an IsopyList'.format(items))

def asslist(items, skip_duplicates=False, allow_duplicates=True):
    if isinstance(items, IsopyList):
        if skip_duplicates and items.has_duplicates():
            items = items.copy()
            items.remove_duplicates()
            return items
        elif not allow_duplicates and items.has_duplicated():
            raise ValueError('list contains duplicate items')
        else:
            return items
    else:
        return slist(items)

def array(values=None, *, keys=None, ndim=None, dtype=None):
    try:
        return MassArray(values, keys=keys, ndim=ndim, dtype=dtype)
    except:
        try:
            return ElementArray(values, keys=keys, ndim=ndim, dtype=dtype)
        except:
            try:
                return IsotopeArray(values, keys=keys, ndim=ndim, dtype=dtype)
            except:
                try:
                    return RatioArray(values, keys=keys, ndim=ndim, dtype=dtype)
                except:
                    raise ValueError('Unable to convert input to IsopyArray')

def asarray(values=None, *, keys=None, ndim=None, dtype=None):
    if isinstance(values, IsopyArray):
        if keys is not None and values.keys() != keys:
            return array(values, keys=keys, ndim=ndim, dtype=dtype)
        elif ndim is not None and values.ndim != ndim:
            return array(values, keys=keys, ndim=ndim, dtype=dtype)
        elif dtype is not None:
            return array(values, keys=keys, ndim=ndim, dtype=dtype)
        else:
            return values
    else:
        return array(values, keys=keys, ndim=ndim, dtype=dtype)

def asanyarray(values=None, *, dtype=None):
    try:
        return asarray(values, dtype=dtype)
    except:
        return np.asarray(values, dtype=dtype)

##################
### Exceptions ###
##################

def classname(obj):
    if isinstance(obj, type): return obj.__name__
    else: return obj.__class__.__name__

class IsopyException(Exception):
    pass

class StringParseError(IsopyException):
    def __init__(self, string, cls, additional_information = None):
        self.string = string
        self.cls = cls
        self.additional_information = additional_information

    def __str__(self):
        message = '{}: Unable to parse "{}".'.format(classname(self.cls), self.string)
        if self.additional_information is not None:
            return '{} {}.'.format(message, self.additional_information)
        else:
            return message

class StringTypeError(IsopyException):
    def __init__(self, obj, cls):
        self.obj = obj
        self.cls = cls

    def __str__(self):
        return '{}: Cannot convert {} into <class \'str\'>.'.format(classname(self.cls), type(self.obj))

class StringDuplicateError(IsopyException):
    def __init__(self, string, listobj):
        self.string = string
        self.listobj = listobj

    def __str__(self):
        message = '{}: Duplicate of "{}" found in {}'.format(classname(self.listobj), self.string, self.listobj)
        return message

class ListSizeError(IsopyException):
    def __init__(self, other, listobj):
        self.other = other
        self.listobj = listobj

    def __str__(self):
        return '{}: Length of \'{}\' does not match current list ({}, {})'.format(classname(self.listobj),
                                                    classname(self.other), len(self.other), len(self.listobj))

class ListTypeError(IsopyException):
    def __init__(self, other, expected, listobj):
        self.other = other
        self.expected = expected
        self.listobj = listobj

    def __str__(self):
        return '{}: Item must be a {} not {}'.format(classname(self.listobj), classname(self.listobj),
                                                     classname(self.expected))

class NumeratorTypeError(ListTypeError):
    def __str__(self):
        return '{}: Numerator must be a {} not {}'.format(classname(self.listobj), classname(self.listobj),
                                                     classname(self.expected))

class DenominatorTypeError(ListTypeError):
    def __str__(self):
        return '{}: Denominator must be a {} not {}'.format(classname(self.listobj), classname(self.listobj),
                                                     classname(self.expected))

class NoCommomDenominator(IsopyException):
    def __init__(self, listobj):
        self.listobj = listobj

    def __str__(self):
        return '{}: List does not have a common denominator'.format(classname(self.listobj))

#############
### Types ###
#############
class IsopyType:
    @classmethod
    def _string(cls, string):
        raise TypeError('IsopyType not specified')

    @classmethod
    def _list(cls, strings):
        raise TypeError('IsopyType not specified')

    @classmethod
    def _array(cls, *args, **kwargs):
        raise TypeError('IsopyType not specified')

    @classmethod
    def _ndarray(cls, obj):
        raise TypeError('IsopyType not specified')

    @classmethod
    def _void(cls, obj):
        raise TypeError('IsopyType not specified')


class MassType:
    @classmethod
    def _string(cls, string):
        return MassString(string)

    @classmethod
    def _list(cls, strings):
        return MassList(strings)

    @classmethod
    def _array(cls, *args, **kwargs):
        return MassArray(*args, **kwargs)

    @classmethod
    def _void(cls, obj):
        return obj.view((MassVoid, obj.dtype))

    @classmethod
    def _ndarray(cls, obj):
        return obj.view(MassNdarray)


class ElementType:
    @classmethod
    def _string(cls, string):
        return ElementString(string)

    @classmethod
    def _list(cls, strings):
        return ElementList(strings)

    @classmethod
    def _array(cls, *args, **kwargs):
        return ElementArray(*args, **kwargs)

    @classmethod
    def _void(cls, obj):
        return obj.view((ElementVoid, obj.dtype))

    @classmethod
    def _ndarray(cls, obj):
        return obj.view(ElementNdarray)


class IsotopeType:
    @classmethod
    def _string(cls, string):
        return IsotopeString(string)

    @classmethod
    def _list(cls, strings):
        return IsotopeList(strings)

    @classmethod
    def _array(cls, *args, **kwargs):
        return IsotopeArray(*args, **kwargs)

    @classmethod
    def _void(cls, obj):
        return obj.view((IsotopeVoid, obj.dtype))

    @classmethod
    def _ndarray(cls, obj):
        return obj.view(IsotopeNdarray)


class RatioType:
    @classmethod
    def _string(cls, string):
        return RatioString(string)

    @classmethod
    def _list(cls, strings):
        return RatioList(strings)

    @classmethod
    def _array(cls, *args, **kwargs):
        return RatioArray(*args, **kwargs)

    @classmethod
    def _void(cls, obj):
        return obj.view((RatioVoid, obj.dtype))

    @classmethod
    def _ndarray(cls, obj):
        return obj.view(RatioNdarray)


##############
### String ###
##############
class IsopyString(IsopyType, str):
    def __new__(cls, item):
        warnings.warn('please use isopy.string instead', DeprecationWarning)
        return string(item)

    def __eq__(self, other):
        if self.__hash__() == other:
            return True
        else:
            try:
                return self.__hash__() == self.__class__(other).__hash__()
            except:
                return False

    def __ne__(self, other):
        return not self.__eq__(other)
            
    def __hash__(self):
        return super(IsopyString, self).__hash__()

    def varname(self):
        #This function should return a string that can safely be used as variable name
        raise NotImplementedError()


class MassString(MassType, IsopyString):
    """
    MassString(string)

    String representation of an integer representing a mass number.

    The supplied string must consist only of digits. :class:`MassString` can also be initialised with an integer.

    Behaves just like, and contains all the methods that, a normal python string does. However, any method that
    returns a string will return a normal python string and *not* an :class:`MassString` unless specifically noted below.
    Only methods that behave differently from a normal string are documented below.

    When compared to another item that item will be converted to an :class:`MassString` before comparison.

    The MassString will behave as an integer when used with the ``+``, ``-``, ``*``, ``<``, ``<=``, ``>`` and ``>=``
    operators and will therefore not return a :class:`MassString`. The ``/`` operator when used in conjunction with
    another string will return a  :class:`RatioString`. For any other type of value is will behave the same
    as the other operators.

    Raises
    ------
    StringParseError
        Is raised when the supplied string cannot be parsed into the correct format
    StringTypeError
        Raised when the supplied item cannot be turned into a string


    Examples
    --------
    >>> isopy.MassString('76')
    '76'
    >>> isopy.MassString(70)
    '70'

    Comparisons

    >>> isopy.MassString('76') == 76
    True
    >>> isopy.MassString('76') != 76
    False


    Using ``+``, ``-``, ``*``, ``/``, ``<``, ``<=``, ``>`` and ``>=``

    >>> isopy.MassString('76') + 2
    78
    >>> isopy.MassString('76') / 2
    38
    >>> isopy.MassString('76') / '70'
    '76/70'
    >>> isopy.MassString('76') > 70
    True
    >>> isopy.MassString('76') <= '76'
    True
    """
    def __new__(cls, string):
        if isinstance(string, cls):
            return string
        elif not isinstance(string, str):
            try:
                string = str(string)
            except:
                raise StringTypeError(string, cls)

        string.strip()
        if string[:1] == '_':
            string = string[1:]
        elif string[:5] == 'Mass_':
            string = string[5:]
        if not string.isdigit():
            raise StringParseError(string, cls, 'Can only contain numerical characters')
        if int(string) < 0:
            raise StringParseError(string, cls, 'Must be a positive integer')
        return str.__new__(cls, string)

    def varname(self):
        """
        Returns description of the string that can be used as a variable name.

        The returned string can be parsed back into an :class:`MassString` the same as any other string.

        Examples
        --------
        >>> isopy.MassString(76).varname()
        'Mass_76'
        >>> isopy.MassString('Mass_76')
        '76'

        Returns
        -------
        str
            A string that can be used as a variable name
        """
        return 'Mass_{}'.format(self)

    def __eq__(self, other):
        if isinstance(other, int):
            return super(MassString, self).__eq__(str(other))
        else:
            return super(MassString, self).__eq__(other)
    
    def __hash__(self):
        return super(MassString, self).__hash__()

    def __ge__(self, item):
        if isinstance(item, MassString):
            item = int(item)
        else:
            return int(self).__ge__(item)

    def __le__(self, item):
        if isinstance(item, MassString):
            item = int(item)
        try:
            return int(self).__le__(item)
        except:
            raise TypeError('MassString only supports "<=" operator with other MassStrings or integers')

    def __gt__(self, item):
        if isinstance(item, MassString):
            item = int(item)
        else:
            return int(self).__gt__(item)

    def __lt__(self, item):
        if isinstance(item, MassString):
            item = int(item)
        else:
            return int(self).__lt__(item)

    def __add__(self, other):
        return int(self).__add__(other)

    def __truediv__(self, other):
        if isinstance(other, str):
            return RatioString('{}/{}'.format(self, other))
        else:
            return int(self).__truediv__(other)

    def __mul__(self, other):
        return int(self).__mul__(other)

    def __sub__(self, other):
        return int(self).__sub__(other)


class ElementString(ElementType, IsopyString):
    """
    ElementString(string)

    String representation of an element symbol.

    The supplied string will automatically be formatted such that the first letter is always in upper case
    and, if present, the second letter will always be in lower case. :class:`ElementString` is limited to two characters.

    Behaves just like, and contains all the methods that, a normal python string does. However, any method that
    return a string will return a normal python string and *not* an :class:`ElementString` unless specifically noted below.
    Only methods that behave differently from a normal string are documented below.

    When compared to another item that item will, if possible, be converted to an :class:`ElementString` for comparison.

    Can be made into an :class:`RatioString` using the ``/`` operator.

    Raises
    ------
    StringParseError
        Is raised when the supplied string cannot be parsed into the correct format
    StringTypeError
        Raised when the supplied item cannot be turned into a string


    Examples
    --------
    >>> isopy.ElementString('ge')
    'Ge'

    Comparisons

    >>> isopy.ElementString('Ge') == 'ge'
    True
    >>> isopy.ElementString('Ge') != 'ge'
    False

    Using ``/`` to create a :class:`RatioString`

    >>> isopy.ElementString('Ge') / 'Pd'
    'Ge/Pd'
    """
    def __new__(cls, string):
        if isinstance(string, cls):
            return string
        elif isinstance(string, str):
            string = string.strip()
            if string[:8] == 'Element_':
                string = string[8:]
            if len(string) == 0:
                raise ValueError('ElementString empty')
            if len(string) > 2:
                raise ValueError('ElementString is limited to two characters')
            if not string.isalpha(): raise ValueError(
                'ElementString "{}" can only contain alphabetical characters'.format(string))
            return str.__new__(cls, string.capitalize())
        else:
            raise TypeError("ElementString must be initialised with a <class 'str'> ({})".format(type(string)))

    def varname(self):
        """
        Returns description of the string that can be used as a variable name.

        The returned string can be parsed back into an :class:`ElementString` the same as any other string.

        Examples
        --------
        >>> isopy.ElementString('Ge').varname()
        'Element_Ge'
        >>> isopy.Elementstring('Element_Ge')
        'Ge'

        Returns
        -------
        str
            A string that can be used as a variable name
        """
        return 'Element_{}'.format(self)

    def __truediv__(self, other):
        return RatioString('{}/{}'.format(self, other))


class IsotopeString(IsotopeType, IsopyString):
    """
    IsotopeString(string)

    String representation of an isotope containing a mass number followed by an element symbol.

    The supplied string should consist of a mass number and an element symbol that will be parsed into a :class:`MassString` and
    an :class:`ElementString`, respectively. The mass number can occur before or after the element symbol.

    Behaves just like, and contains all the methods that, a normal python string does. However, any method that
    return a string will return a normal python string and *not* an :class:`IsotopeString` unless specifically noted below.
    Only methods that behave differently from a normal string are documented below.'

    When compared to another item that item will, if possible, be converted to an :class:`IsotopeString` for comparison.
    The ``in`` operator will check if the given item is equal to either the mass number or element symbol

    Can be made into an :class:`RatioString` using the ``/`` operator.

    Raises
    ------
    StringParseError
        Is raised when the supplied string cannot be parsed into the correct format
    StringTypeError
        Raised when the supplied item cannot be turned into a string

    Attributes
    ----------
    mass_number : MassString
        The mass number of the :class:`IsotopeString`
    element_symbol : ElementString
        The element symbol of the :class:`IsotopeString`

    Examples
    --------
    >>> isopy.IsotopeString('ge76')
    '76Ge'
    >>> isopy.IsotopeString('ge76').mass_number
    '76'
    >>> isopy.IsotopeString('ge76').element_symbol
    'Ge'

    Comparisons

    >>> isopy.IsotopeString('76Ge') == 'ge76'
    True
    >>> isopy.IsotopeString('76Ge') != 'ge76'
    False

    Using ``in``

    >>> 'ge' in isopy.IsotopeString('ge76')
    True
    >>> '70' in isopy.IsotopeString('ge76')
    False

    Using ``/`` to create a :class:`RatioString`

    >>> isopy.IsotopeString('Ge76') / '70ge'
    '76Ge/70Ge'
    """
    def __new__(cls, string):
        if isinstance(string, cls):
            return string
        elif isinstance(string, str):
            string = string.strip()

            if string[:1] == '_': string = string[1:]
            elif string[:8] == 'Isotope_': string = string[8:]

            # If no digits in string then only Symbol is given.
            if string.isalpha():
                raise ValueError('"{}" does not contain a mass number'.format(string))

            # If only digits then only mass number
            if string.isdigit():
                raise ValueError('"{}" does not contain an element symbol'.format(string))

            # Loop through to split
            l = len(string)
            for i in range(1, l):
                a = string[:i]
                b = string[i:]

                # a = mass number and b = Symbol
                if a.isdigit() and b.isalpha():
                    mass = MassString(a)
                    element = ElementString(b)
                    obj = str.__new__(cls, '{}{}'.format(mass, element))
                    obj.mass_number = mass
                    obj.element_symbol = element
                    return obj

                # b = mass number and a = Symbol
                elif a.isalpha() and b.isdigit():
                    mass = MassString(b)
                    element = ElementString(a)
                    obj = str.__new__(cls, '{}{}'.format(mass, element))
                    obj.mass_number = mass
                    obj.element_symbol = element
                    return obj
            raise ValueError('unable to parse "{}" into IsotopeString'.format(string))
        else:
            raise TypeError("ElementString must be initialised with a <class 'str'> ({})".format(type(string)))

    def varname(self):
        """
        Returns description of the string that can be used as a variable name.

        The returned string can be parsed back into an :class:`IsotopeString` the same as any other string.

        Examples
        --------
        >>> isopy.IsotopeString('Ge76').varname()
        'Isotope_76Ge'
        >>> isopy.IsotopeString('Isotope_76Ge')
        '76Ge'

        Returns
        -------
        str
            A string that can be used as a variable name
        """
        return 'Isotope_{}'.format(self)

    def __contains__(self, string):
        """
        Return **True** if `string` is equal to the IsotopeString's mass number or element symbol
        """       
        return self.mass_number == string or self.element_symbol == string

    def __truediv__(self, other):
        return RatioString('{}/{}'.format(self, other))


class RatioString(RatioType, IsopyString):
    """
    RatioString(string)

    String representation of a ratio consisting of a numerator and a denominator.


    The supplied string should consist of a numerator and a denominator separated by a '/'. The numerator and
    denominator strings must be able to be parsed into a :class:`MassString`, :class:`ElementString` or a
     :class:`IsotopeString`. The numerator and denominator do not have to be the same type.

    Behaves just like, and contains all the methods that, a normal python string does. However, any method that
    return a string will return a normal python string and *not* an :class:`RatioString` unless specifically noted below.
    Only methods that behave differently from a normal string are documented below.

    When compared to another item that item will, if possible, be converted to an :class:`RatioString` for comparison.
    The ``in`` operator will check if the given item is equal to either the numerator or denominator.

    Raises
    ------
    StringParseError
        Is raised when the supplied string cannot be parsed into the correct format
    StringTypeError
        Raised when the supplied item cannot be turned into a string

    Attributes
    ----------
    numerator : MassString, ElementString or IsotopeString
        The numerator string in the :class:`RatioString`. **Readonly**
    denominator : MassString, ElementString or IsotopeString
        The denominator string in the :class:`RatioString`. **Readonly**

    Examples
    --------
    >>> isopy.RatioString('ge76/ge70')
    '76Ge/70Ge'
    >>> isopy.RatioString('ge76').numerator
    '76Ge'
    >>> isopy.RatioString('ge76').denominator
    '70Ge'

    Comparisons

    >>> isopy.RatioString('ge76/70ge') == 'ge76/70ge'
    True
    >>> isopy.RatioString('ge76/70ge') != 'ge76/70ge'
    False

    Using ``in``

    >>> '76ge' in isopy.RatioString('ge76/70ge')
    True
    >>> '75ge' isopy.RatioString('ge76/70ge')
    False
    """
    def __new__(cls, string):
        if isinstance(string, cls):
            return string
        elif isinstance(string, str):
            string = string.strip()
            if string[:6] == 'Ratio_':
                string = string[6:]
                string = string.replace('_', '/')
            else:
                string = string.replace('_OVER_', '/')
            if '/' not in string:
                raise ValueError('unable to split string into numerator and denominator')
            numer, denom = string.split('/', 1)

            try:
                numer = MassString(numer)
            except ValueError:
                try:
                    numer = ElementString(numer)
                except ValueError:
                    try:
                        numer = IsotopeString(numer)
                    except ValueError:
                        raise ValueError('Unable to parse numerator: "{}"'.format(numer))
            try:
                denom = MassString(denom)
            except ValueError:
                try:
                    denom = ElementString(denom)
                except ValueError:
                    try:
                        denom = IsotopeString(denom)
                    except ValueError:
                        raise ValueError('Unable to parse denominator: "{}"'.format(denom))
            obj = str.__new__(cls, '{}/{}'.format(numer, denom))
            obj.numerator = numer
            obj.denominator = denom
            return obj
        else:
            raise TypeError("RatioString must be initialised with a <class 'str'> ({})".format(type(string)))

    def varname(self):
        """
        Returns description of the string that can be used as a variable name.

        The returned string can be parsed back into an :class:`RatioString` the same as any other string.

        Examples
        --------
        >>> isopy.RatioString('Ge76/Ge70').varname()
        'Ratio_76Ge_70Ge'
        >>> isopy.RatioString('Ratio_76Ge_70Ge')
        '76Ge/70Ge'

        Returns
        -------
        str
        """
        return 'Ratio_{}_{}'.format(self.numerator, self.denominator)

    def __contains__(self, string):
        """
        Return **True** if `string` is equal to the RatioString's numerator or denominator
        """
        return self.numerator == string or self.denominator == string


############
### List ###
############
class IsopyList(IsopyType, list):
    def __new__(cls, items, skip_duplicates = False, allow_duplicates=True):
        warnings.warn('please use isopy.slist instead', DeprecationWarning)
        return slist(items, skip_duplicates=skip_duplicates, allow_duplicates=allow_duplicates)

    def __init__(self, items, skip_duplicates = False, allow_duplicates=True):
        if isinstance(items, str): items = items.split(',')
        elif not hasattr(items, '__iter__'): items = [items]

        items = [self._string(item) for item in items]
        if skip_duplicates:
            for item in items[1:]:
                while items.count(item) > 1: items.remove(item)
        elif not allow_duplicates:
            for item in items[1:]:
                if self.count(item) > 1:
                    raise StringDuplicateError(item, self)

        super().__init__(items)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, super(IsopyList, self).__repr__())
    
    def __eq__(self, other):
        """
        Return **True** if all items in `other` is the same all items in the list. Order is not important
        """
        if not isinstance(other, self.__class__):
            try: other = self.__class__(other)
            except:
                return False
        if len(other) != len(self): return False
        for key in self:
            if key not in other: return False
            if self.count(key) != other.count(key): return False
        return True
    
    def __contains__(self, item):
        """
        Return **True** if `item` is present in the list. If `item` is a list then return **True**
        if all items in that list are present in the list. Otherwise return **False**.
        """
        if isinstance(item, list):
            for i in item:
                if not self.__contains__(i): return False
            return True

        try:
            return super(IsopyList, self).__contains__(self._string(item))
        except:
            return False

    def __getitem__(self, index):
        """
        Return the item at `index`. `index` can be int, slice or sequence of int.
        """
        if isinstance(index,slice):
            return self.__class__(super(IsopyList, self).__getitem__(index))
        elif hasattr(index, '__iter__'):
                return self.__class__([super(IsopyList, self).__getitem__(i) for i in index])
        else:
            return super(IsopyList, self).__getitem__(index)

    def __setitem__(self, key, value):
        super(IsopyList, self).__setitem__(key, self._string(value))

    def __and__(self, other):
        other = self._list(other)
        return self._list([item for item in self if item in other], skip_duplicates=True)

    def __or__(self, other):
        other = self._list(other)
        return self._list(super(IsopyList, self).__add__(self, [item for item in other if item not in self]),
                          skip_duplicates=True)

    def __xor__(self, other):
        other = self._list(other)
        return self._list(
            super(IsopyList, self).__add__([item for item in self if item not in other],
                                           [item for item in other if item not in self]),
                                            skip_duplicates=True)

    def __add__(self, other):
        other = self._list(other)
        return self._list(super(IsopyList, self).__add__(other))

    def count(self, item):
        try:
            item = self._string(item)
        except:
            return 0
        else:
            super(IsopyList, self).count(item)

    def index(self, item, *args):
        try:
            return super(IsopyList, self).index(self._string(item), *args)
        except StringParseError:
            raise ValueError('\'{}\' is not in list'.format(item))

    def insert(self, index, item):
        return super(IsopyList, self).insert(index, self._string(item))

    def remove(self, item):
        try:
            return super(IsopyList, self).remove(self._string(item))
        except StringParseError:
            raise ValueError('\'{}\' is not in list'.format(item))

    def copy(self):
        return self.__class__(super(IsopyList, self).copy())

    def append(self, item, skip_duplicates=False, allow_duplicates=True):
        """
        Add an item to the end of the list. If ``skip_duplicates = True`` the item will only be added if it is not
        already in the list. If ``allow_duplicated = False`` a StringDuplucateError will be raised if item is already
        present in the list.

        ``skip_duplicates`` and ``allow_duplicate`` options does not apply to items already present in the list.

        Raises
        ------
        StringDuplicateError
            Raised when a string already exist in the list and ``allow_duplicates = False``
        """
        item = self._string(item)
        if skip_duplicates and item in self:
            pass
        elif not allow_duplicates and item in self:
            raise StringDuplicateError(item, self)
        else:
            super(IsopyList, self).append(item)

    def extend(self, items, skip_duplicates=False, allow_duplicates=True):
        """
        Extend the list by appending all the items. If ``skip_duplicates = True`` only those items not already present in
        the list will be added. If ``allow_duplicated = False`` a StringDuplucateError will be raised if any item is already
        present in the list.

        ``skip_duplicates`` and ``allow_duplicate`` options do not apply for items already present in the list.

        Raises
        ------
        StringDuplicateError
            Raised when a string already exist in the list and ``allow_duplicates = False``
        """
        if not hasattr(items, '__iter__'): raise TypeError(
            '\'{}\' object is not iterable'.format(items.__class__.__name__))

        items = self._list(items, skip_duplicates, allow_duplicates)
        if skip_duplicates:
            items = [item for item in items if item not in self]
        elif not allow_duplicates:
            for item in items:
                if item in self:
                    raise StringDuplicateError(item, self)

        super(IsopyList, self).extend(items)

    def varnames(self):
        """
        Returns a list of variable names for each item in the list. Equivalent to ``[item.varname() for item in self]``
        """
        return [s.varname() for s in self]

    def has_duplicates(self):
        """
        Returns ``True`` if the list contains duplicates items. Otherwise it returns ``False``
        """
        if len(set(self)) != len(self): return True
        else: return False

    def remove_duplicates(self):
        """
        Removes all duplicate item in the list.
        """
        for item in self[1:]:
            while self.count(item) > 1: self.remove(item)


class MassList(MassType, IsopyList):
    """
    MassList(items, skip_duplicates=False, allow_duplicates=True)

    A list consisting exclusively of :class:`MassString` items.

    Behaves just like, and contains all the methods, that a normal python list does unless otherwise noted. All methods
    that would normally return a list will instead return a :class:`MassList`. Only methods that behave differently from
    a normal python list are documented below.

    When comparing the list against another list the order of items is not considered.
    The ``in`` operator can be used both with a single item or a list. If it is a list it will return ``True`` if
    all item in the other list is present in the list.

    Get item can be an int or a list of integers. However, set item only accepts a single integer.

    The ``&``, ``|`` and ``^`` operators can be used in combination with another list for an unordered item to item
    comparison. The and (``&``) operator will return the items that occur in both lists. The or (``|``) operator
    will return all items that appear in at least one of the lists. The xor (``^``) operator will return the items
    that do not appear in both lists. All duplicate items will be removed from the returned lists.


    Parameters
    ----------
    items : str, MassArray, iterable
        Each item in the iterable will be :class:`MassString`. If items is a string it will be split into a list using the
        ',' seperator.  If items is an :class:`MassArray` it will use the keys from that array to make the list.
    skip_duplicates : bool
        If ``True`` all duplicate items will be removed from the list. This only applies during initialisation of the list.
        Default value is ``False``.
    allow_duplicates : bool
        If ``False`` a StringDuplicateError will be raised if the list contains any duplicate items. This only applies
        during initialisation of the list. Default Value is ``True``.

    Raises
    ------
    StringDuplicateError
        Raised when a string already exist in the list and ``allow_duplicates = False``
    ListLengthError
        Raised when there is a size mismatch between this list and another list


    Examples
    --------
    >>> MassList([99, 105, '111'])
    ['99', '105', '111']
    >>> MassList('99, 105,111')
    ['99', '105', '111']

    Comparisons and ``in``
    >>> MassList([99, 105, '111']) == [111, 105, 99]
    True
    >>> '105' in MassList([99, 105, '111'])
    True
    >>> ['105', 111] in MassList([99, 105, '111'])
    True
    >>> ['105', 107] in MassList([99, 105, '111'])
    False

    Indexing

    >>> MassList([99, 105, '111'])[[0,2]
    ['99', '111']

    Using ``&``, ``|`` and ``^``

    >>> MassList([99, 105, '111']) & [105, 107, '111']
    ['105', '111']
    >>> MassList([99, 105, '111']) | [105, 107, '111']
    ['99', '105', '111', '107']
    >>> MassList([99, 105, '111']) ^ [105, 107, '111']
    ['99', '107']

    Using ``/`` to make a :class:`RatioList`

    >>> MassList([99, 105, '111']) / 108
    ['99/108', '105/108', '111/108']
    >>> MassList([99, 105, '111']) / [108, 109, '110']
    ['99/108', '105/109', '111/110']
    """

    def __new__(cls, items, skip_duplicates = False, allow_duplicates=True):
        obj = list.__new__(cls)
        obj.__init__(items, skip_duplicates, allow_duplicates)
        return obj

    def __truediv__(self, denominator):
        if isinstance(denominator, list):
            if len(denominator) != len(self): raise ValueError('Length of supplied list is not the same as subject list')
            return RatioList(['{}/{}'.format(self[i], denominator[i]) for i in range(len(self))])

        if isinstance(denominator, str):
            return RatioList(['{}/{}'.format(x, denominator) for x in self])

        raise ValueError('unable to parse {}'.format(denominator))

    def _filter_indexes(self, index_list, *, mass_number=None, mass_number_not=None, mass_number_lt=None,
                        mass_number_gt=None, mass_number_le=None, mass_number_ge=None, **_unused_kwargs):
        """
        Checks each index in *index_list* against the keyword arguments and return a a list with the indexes that
        satisfy the keyword arguments.

        This is called by the *copy* method and potentially by the copy method of other lists.

        Returns
        -------
            list of int
                A list of the indexes that satisfy the keyword arguments.
        """
        for k in _unused_kwargs.keys():
            raise ValueError('"{}" not a valid filter for MassList'.format(k))

        if mass_number is not None and not isinstance(mass_number, MassList):
            mass_number = MassList(mass_number)
        if mass_number_not is not None and not isinstance(mass_number_not, MassList):
            mass_number_not = MassList(mass_number_not)

        out = []
        for i in index_list:
            iso = self[i]
            if mass_number is not None:
                if iso not in mass_number: continue
            if mass_number_not is not None:
                if iso in mass_number_not: continue

            if mass_number_lt is not None:
                if iso >= mass_number_lt: continue
            if mass_number_gt is not None:
                if iso <= mass_number_gt: continue
            if mass_number_le is not None:
                if iso > mass_number_le: continue
            if mass_number_ge is not None:
                if iso < mass_number_ge: continue

            # The item satisfies all the keyword arguments given.
            out.append(i)

        return out


    def copy(self, *, mass_number=None, mass_number_not=None, mass_number_lt=None, mass_number_gt=None,
               mass_number_le=None, mass_number_ge=None):
        """
        copy(*, element_symbol = None, element_symbol_not = None)

        Returns a copy of the list.

        If keyword arguments are given then only those items in the list that satisfy these arguments are copied.

        Parameters
        ----------
        mass_number : MassString, MassList, optional
           Only items in the list matching/found in ``mass_number`` are copied.
        mass_number_not : MassString, MassList, optional
           Only items in the list not matching/not found in ``mass_number`` are copied.
        mass_number_lt : MassString, optional
            Only items in the list a mass number less than this value are copied.
        mass_number_gt : MassString, optional
            Only items in the list a mass number greater than this value are copied.
        mass_number_le : MassString, optional
            Only items in the list a mass number less than or equal to this value are copied.
        mass_number_ge : MassString, optional
            Only items in the list a mass number greater than or equal to this value are copied.


        Returns
        -------
        MassList
            A copy of the items in the list that satisfy the keyword arguments or all items in the list if none are
            given.

        Examples
        --------
        >>> MassList(['99', '105', '111']).copy(mass_number=[105, 107, 111])
        ['105', '111']
        >>> MassList(['99', '105', '111']).copy(mass_number_not='111'])
        ['99', '105']
        >>> MassList(['99', '105', '111']).copy(mass_number_gt='99'])
        ['105', '111']
        >>> MassList(['99', '105', '111']).copy(mass_number_le=105])
        ['99', '105']
        """

        """
        copy(*,mass_number = None,  mass_number_not = None, mass_number_lt=None, mass_number_gt=None, mass_number_le=None, mass_number_ge=None)

        Returns a copy of the current list.  *DESCRIPTION*


        Parameters
        ----------
        mass_number : MassInteger, MassList, optional
            Only MassIntegers matching this string/found in this list will pass.
        mass_number_not : MassInteger, MassList, optional
            Only MassIntegers not matching this string/not found in this list will pass.
        


        Returns
        -------
        MassList
            A new MassList with each MassInteger in the list that passed the filters.

        """

        index_list = [i for i in range(len(self))]
        index = self._filter_indexes(index_list, mass_number=mass_number, mass_number_not=mass_number_not,
                                     mass_number_lt=mass_number_lt, mass_number_gt=mass_number_gt,
                                     mass_number_le=mass_number_le, mass_number_ge=mass_number_ge)
        return self[index]


class ElementList(ElementType, IsopyList):
    """
    ElementList(items, skip_duplicates=False, allow_duplicates=True)

    A list consisting exclusively of :class:`ElementString` items.

    Behaves just like, and contains all the methods, that a normal python list does unless otherwise noted. All methods
    that would normally return a list will instead return a :class:`ElementList`. Only methods that behave differently from
    a normal python list are documented below.

    When comparing the list against another list the order of items is not considered.
    The ``in`` operator can be used both with a single item or a list. If it is a list it will return ``True`` if
    all item in the other list is present in the list.

    Get item can be an int or a list of integers. However, set item only accepts a single integer.

    The ``&``, ``|`` and ``^`` operators can be used in combination with another list for an unordered item to item
    comparison. The and (``&``) operator will return the items that occur in both lists. The or (``|``) operator
    will return all items that appear in at least one of the lists. The xor (``^``) operator will return the items
    that do not appear in both lists. All duplicate items will be removed from the returned lists.


    Parameters
    ----------
    items : str, ElementArray, iterable
        Each item in the iterable will be :class:`ElementString`. If items is a string it will be split into a list using the
        ',' seperator.  If items is an :class:`ElementArray` it will use the keys from that array to make the list.
    skip_duplicates : bool
        If ``True`` all duplicate items will be removed from the list. This only applies during initialisation of the list.
        Default value is ``False``.
    allow_duplicates : bool
        If ``False`` a StringDuplicateError will be raised if the list contains any duplicate items. This only applies
        during initialisation of the list. Default Value is ``True``.

    Raises
    ------
    StringDuplicateError
        Raised when a string already exist in the list and ``allow_duplicates = False``
    ListLengthError
        Raised when there is a size mismatch between this list and another list


    Examples
    --------
    >>> ElementList(['ru', 'pd', 'cd'])
    ['Ru', 'Pd', 'Cd']
    >>> ElementList('ru, pd,cd')
    ['Ru', 'Pd', 'Cd']

    Comparisons and ``in``
    >>> ElementList(['ru', 'pd', 'cd']) == ['cd', 'pd', 'ru']
    True
    >>> 'pd' in ElementList(['ru', 'pd', 'cd'])
    True
    >>> ['pd', 'cd'] in ElementList(['ru', 'pd', 'cd'])
    True
    >>> ['pd', 'ag'] in ElementList(['ru', 'pd', 'cd'])
    False

    Indexing

    >>> ElementList(['ru', 'pd', 'cd'])[[0,2]
    ['Ru', 'Cd']

    Using ``&``, ``|`` and ``^``

    >>> ElementList(['ru', 'pd', 'cd']) & ['pd', 'ag', 'cd']
    ['Pd', 'Cd']
    >>> ElementList(['ru', 'pd', 'cd']) | ['pd', 'ag', 'cd']
    ['Ru', 'Pd', 'Cd', 'Ag']
    >>> ElementList(['ru', 'pd', 'cd']) ^ ['pd', 'ag', 'cd']
    ['Ru', 'Ag']

    Using ``/`` to make a :class:`RatioList`

    >>> ElementList(['ru', 'pd', 'cd']) / 'pd'
    ['Ru/Pd', 'Pd/Pd', 'Cd/Pd']
    >>> ElementList(['ru', 'pd', 'cd']) / ['108pd', '109Pd', '110Pd']
    ['Ru/108Pd', 'Pd/109Pd', 'Cd/110Pd']
    """

    def __new__(cls, items, skip_duplicates=False, allow_duplicates=True):
        obj = list.__new__(cls)
        obj.__init__(items, skip_duplicates, allow_duplicates)
        return obj

    def __truediv__(self, denominator):
        if hasattr(denominator, '__iter__'):
            if len(denominator) != len(self): raise ListSizeError(denominator, self)
            return RatioList(['{}/{}'.format(self[i], denominator[i]) for i in range(len(self))])
        else:
            denominator = self._string(denominator)
            return RatioList(['{}/{}'.format(x, denominator) for x in self])

    def _filter_indexes(self, index_list, element_symbol = None, element_symbol_not = None, **_unused_kwargs):
        for k in _unused_kwargs.keys():
            raise ValueError('"{}" not a valid filter for ElementList'.format(k))

        if element_symbol is not None and not isinstance(element_symbol, ElementList):
            element_symbol = ElementList(element_symbol)

        if element_symbol_not is not None and not isinstance(element_symbol_not, ElementList):
            element_symbol_not = ElementList(element_symbol_not)

        out = []
        for i in index_list:
            ele = self[i]
            if element_symbol is not None:
                if ele not in element_symbol: continue
            if element_symbol_not is not None:
                if ele in element_symbol_not: continue

            out.append(i)

        return out

    def copy(self, *, element_symbol = None, element_symbol_not = None):
        """
        copy(*, element_symbol = None, element_symbol_not = None)

        Returns a copy of the list.

        If keyword arguments are given then only those items in the list that satisfy these arguments are copied.

        Parameters
        ----------
        element_symbol : ElementString, ElementList, optional
           Only items in the list matching/found in ``element_symbol`` are copied.
        element_symbol_not : ElementString, ElementList, optional
           Only items in the list not matching/not found in ``element_symbol_not`` are copied.


        Returns
        -------
        ElementList
            A copy of the items in the list that satisfy the keyword arguments or all items in the list if none are
            given.

        Examples
        --------
        >>> ElementList(['ru', 'pd', 'cd']).copy(element_symbol=['pd','ag', 'cd'])
        ['Pd', 'Cd]
        >>> ElementList(['ru', 'pd', 'cd']).copy(element_symbol_not='cd')
        ['Ru', 'Pd']
        """
        index_list = [i for i in range(len(self))]
        index = self._filter_indexes(index_list, element_symbol=element_symbol, element_symbol_not=element_symbol_not)
        return self[index]


class IsotopeList(IsotopeType, IsopyList):
    """
    IsotopeList(items, skip_duplicates=False, allow_duplicates=True)

    A list consisting exclusively of :class:`IsotopeString` items.

    Behaves just like, and contains all the methods, that a normal python list does unless otherwise noted. All methods
    that would normally return a list will instead return a :class:`IsotopeList`. Only methods that behave differently from
    a normal python list are documented below.

    When comparing the list against another list the order of items is not considered.
    The ``in`` operator can be used both with a single item or a list. If it is a list it will return ``True`` if
    all item in the other list is present in the list.

    Get item can be an int or a list of integers. However, set item only accepts a single integer.

    The ``&``, ``|`` and ``^`` operators can be used in combination with another list for an unordered item to item
    comparison. The and (``&``) operator will return the items that occur in both lists. The or (``|``) operator
    will return all items that appear in at least one of the lists. The xor (``^``) operator will return the items
    that do not appear in both lists. All duplicate items will be removed from the returned lists.


    Parameters
    ----------
    items : str, IsotopeArray, iterable
        Each item in the iterable will be :class:`IsotopeString`. If items is a string it will be split into a list using the
        ',' seperator.  If items is an :class:`IsotopeArray` it will use the keys from that array to make the list.
    skip_duplicates : bool
        If ``True`` all duplicate items will be removed from the list. This only applies during initialisation of the list.
        Default value is ``False``.
    allow_duplicates : bool
        If ``False`` a StringDuplicateError will be raised if the list contains any duplicate items. This only applies
        during initialisation of the list. Default Value is ``True``.

    Raises
    ------
    StringDuplicateError
        Raised when a string already exist in the list and ``allow_duplicates = False``
    ListLengthError
        Raised when there is a size mismatch between this list and another list


    Examples
    --------
    >>> IsotopeList(['99ru', '105pd', '111cd'])
    ['99Ru', '105Pd', '111Cd']
    >>> IsotopeList('99ru, 105pd,cd111')
    ['99Ru', '105Pd', '111Cd']

    Comparisons and ``in``

    >>> IsotopeList(['99ru', '105pd', '111cd']) == ['111cd', '105pd',  '99ru', ]
    True
    >>> '105pd' in IsotopeList(['99ru', '105pd', '111cd'])
    True
    >>> ['105pd', '111cd'] in IsotopeList(['99ru', '105pd', '111cd'])
    True
    >>> ['105pd', '107ag'] in IsotopeList(['99ru', '105pd', '111cd'])
    False

    Indexing

    >>> IsotopeList(['99ru', '105pd', '111cd'])[[0,2]
    ['99Ru', '111Cd']

    Using ``&``, ``|`` and ``^``

    >>> IsotopeList(['99ru', '105pd', '111cd']) & ['105pd', '107ag', '111cd']
    ['105Pd', '111Cd']
    >>> IsotopeList(['99ru', '105pd', '111cd']) | ['105pd', '107ag', '111cd']
    ['99Ru', '105Pd', '111Cd', '107Ag']
    >>> IsotopeList(['99ru', '105pd', '111cd']) ^ ['105pd', '107ag', '111cd']
    ['99Ru', '107Ag']

    Using ``/``  to make a :class:`RatioList`

    >>> ElementList(['99ru', '105pd', '111cd']) / '108pd'
    ['99Ru/108Pd', '105Pd/108Pd', '111Cd/108Pd']
    >>> ElementList(['99ru', '105pd', '111cd']) / ['108pd', '109Pd', '110Pd']
    ['99Ru/108Pd', '105Pd/109Pd', '111Cd/110Pd']
    """

    def __new__(cls, items, skip_duplicates=False, allow_duplicates=True):
        obj = list.__new__(cls)
        obj.__init__(items, skip_duplicates, allow_duplicates)
        return obj

    def __truediv__(self, denominator):
        """
        Returns a `RatioList`_ where the numerator of each `RatioString`_ is equal to the IsotopeString in the list and
        the denominator is equal to `denominator`.


        Parameters
        ---------
        denominator : str, list of str
            If a list then it must have the same length as the list.

        Examples
        --------
        >>> a = IsotopeList(['104Pd', '105Pd', '106Pd'])
        >>> a/'108Pd'
        ['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd']
        >>> a/['Ru', 'Pd', 'Cd']
        ['104Pd/Ru', '105Pd/Pd', '106Pd/Cd']
        """
        if isinstance(denominator, list):
            if len(denominator) != len(self):
                raise ValueError('Length of supplied list is not the same as subject list')
            else:
                return RatioList(['{}/{}'.format(self[i], denominator[i]) for i in range(len(self))])

        if isinstance(denominator, str):
            return RatioList(['{}/{}'.format(x, denominator) for x in self])

    @property
    def mass_numbers(self):
        """
        A :class:`MassList` containing the mass number of each item in the list. *Readonly*

        Examples
        --------
        >>> IsotopeList(['99ru', '105pd', '111cd']).mass_numbers
        ['99', '105', '111']
        """
        return MassList([x.mass_number for x in self])

    @property
    def element_symbols(self):
        """
        An :class:`ElementList` containing the element symbol of each item in the list. *Readonly*

        Examples
        --------
        >>> IsotopeList(['99ru', '105pd', '111cd']).element_symbols
        ['Ru', 'Pd', 'Cd']
        """
        return ElementList([x.element_symbol for x in self])

    def _filter_indexes(self, index_list, isotope = None, isotope_not = None, **mass_number_and_element_symbol_filters):
        if isotope is not None and not isinstance(isotope, IsotopeList):
            isotope = IsotopeList(isotope)

        if isotope_not is not None and not isinstance(isotope_not, IsotopeList):
            isotope_not = IsotopeList(isotope_not)

        element_filters = {}
        mass_filters = {}
        for k in mass_number_and_element_symbol_filters.keys():

            if 'mass_number' in k:
                mass_filters[k] = mass_number_and_element_symbol_filters[k]
            elif 'element_symbol' in k:
                element_filters[k] = mass_number_and_element_symbol_filters[k]
            else:
                raise ValueError('"{}" not a valid filter for IsotopeList'.format(k))

        out = []
        for i in index_list:
            iso = self[i]
            if isotope is not None:
                if iso not in isotope: continue
            if isotope_not is not None:
                if iso in isotope_not: continue

            # It has passed all the tests above
            out.append(i)

        if len(element_filters) > 0:
            out = self.element_symbols._filter_indexes(out, **element_filters)
        if len(mass_filters) > 0:
            out = self.mass_numbers._filter_indexes(out, **mass_filters)

        return out

    def copy(self, *, isotope = None, isotope_not = None, **mass_number_and_element_symbol_kwargs):
        """
        copy(*, isotope = None, isotope_not = None, **mass_number_and_element_symbol_kwargs)

        Returns a copy of the list.

        If keyword arguments are given then only those items in the list that satisfy these arguments are copied.

        Parameters
        ----------
        isotope : IsotopeString, IsotopeList, optional
           Only items in the list matching/found in ``isotope`` are copied.
        isotope_not : IsotopeString, IsotopeList, optional
           Only items in the list not matching/not found in ``isotope_not`` are copied.
        **mass_number_and_element_symbol_kwargs
            See :func:`MassList.copy` and :func:`Element.copy` for additional keyword arguments.


        Returns
        -------
        IsotopeList
            A copy of the items in the list that satisfy the keyword arguments or all items in the list if none are
            given.

        Examples
        --------
        >>> IsotopeList(['99ru', '105pd', '111cd']).copy(isotope=['105pd','107ag', '111cd'])
        ['105Pd', '111Cd]
        >>> IsotopeList(['99ru', '105pd', '111cd']).copy(isotope_not='111cd')
        ['99Ru', '105Pd']
        >>> IsotopeList(['99ru', '105pd', '111cd']).copy(mass_number_gt = 100)
        ['105Pd', '111Cd]
        >>> IsotopeList(['99ru', '105pd', '111cd']).copy(element_not='pd'])
        ['99Ru', '111Cd]
        """
        index_list = [i for i in range(len(self))]
        index = self._filter_indexes(index_list, isotope=isotope, isotope_not=isotope_not,
                                     **mass_number_and_element_symbol_kwargs)
        return self[index]


class RatioList(RatioType, IsopyList):
    """
    RatioList(items, skip_duplicates=False, allow_duplicates=True)

    A list consisting exclusively of :class:`RatioString` items.

    The numerator of all items in the list must be the same type of item. The same is true for the denominator. However,
    the numerators and the denominators do not need to be of the same type.

    Behaves just like, and contains all the methods, that a normal python list does unless otherwise noted. All methods
    that would normally return a list will instead return a :class:`IsotopeString`. Only methods that behave differently from
    a normal python list are documented below.

    When comparing the list against another list the order of items is not considered.
    The ``in`` operator can be used both with a single item or a list. If it is a list it will return ``True`` if
    all item in the other list is present in the list.

    Get item can be an int or a list of integers. However, set item only accepts a single integer.

    The ``&``, ``|`` and ``^`` operators can be used in combination with another list for an unordered item to item
    comparison. The and (``&``) operator will return the items that occur in both lists. The or (``|``) operator
    will return all items that appear in at least one of the lists. The xor (``^``) operator will return the items
    that do not appear in both lists. All duplicate items will be removed from the returned lists.

    Parameters
    ----------
    items : str, RatioArray, iterable
        Each item in the iterable will be :class:`RatioString`. If items is a string it will be split into a list using the
        ',' seperator.  If items is an :class:`RatioArray` it will use the keys from that array to make the list.
    skip_duplicates : bool
        If ``True`` all duplicate items will be removed from the list. This only applies during initialisation of the list.
        Default value is ``False``.
    allow_duplicates : bool
        If ``False`` a StringDuplicateError will be raised if the list contains any duplicate items. This only applies
        during initialisation of the list. Default Value is ``True``.

    Raises
    ------
    StringDuplicateError
        Raised when a string already exist in the list and ``allow_duplicates = False``
    NumeratorTypeError
        Raised when the numerator of a :class:`RatioString` does is not the same type as the
        other numerators in the list.
    DenominatorTypeError
        Raised when the denominator of a :class:`RatioString` does is not the same type as the
        other denominators in the list.

    Examples
    --------
    >>> RatioList(['99ru/108Pd', '105pd/108Pd', '111cd/108Pd'])
    ['99Ru/108Pd', '105Pd/108Pd', '111Cd/108Pd']
    >>> RatioList('99ru/108Pd, 105pd/108Pd,cd111/108Pd')
    ['99Ru/108Pd', '105Pd/108Pd', '111Cd/108Pd']

    Comparisons and ``in``

    >>> RatioList(['99ru/108Pd', '105pd/108Pd', '111cd/108Pd']) == ['111cd/108Pd', '105pd/108Pd',  '99ru/108Pd']
    True
    >>> '105pd' in RatioList(['99ru/108Pd', '105pd/108Pd', '111cd/108Pd'])
    True
    >>> ['105pd/108Pd', '111cd/108Pd'] in RatioList(['99ru/108Pd', '105pd/108Pd', '111cd/108Pd'])
    True
    >>> ['105pd/108Pd', '107ag/108Pd'] in RatioList(['99ru/108Pd', '105pd/108Pd', '111cd/108Pd'])
    False

    Indexing

    >>> RatioList(['99ru/108Pd', '105pd/108Pd', '111cd/108Pd'])[[0,2]
    ['99Ru/108Pd', '111Cd/108Pd']

    Using ``&``, ``|`` and ``^``

    >>> RatioList(['99ru/108Pd', '105pd/108Pd', '111cd/108Pd']) & ['105pd/108Pd', '107ag/108Pd', '111cd/108Pd']
    ['105Pd/108Pd', '111Cd/108Pd']
    >>> RatioList(['99ru/108Pd', '105pd/108Pd', '111cd/108Pd']) | ['105pd/108Pd', '107ag/108Pd', '111cd/108Pd']
    ['99Ru/108Pd', '105Pd/108Pd', '111Cd/108Pd', '107Ag/108Pd']
    >>> RatioList(['99ru/108Pd', '105pd/108Pd', '111cd/108Pd']) ^ ['105pd/108Pd', '107ag/108Pd', '111cd/108Pd']
    ['99Ru/108Pd', '107Ag/108Pd']
    """

    def __new__(cls, items, skip_duplicates=False, allow_duplicates=True):
        obj = list.__new__(cls)
        obj.__init__(items, skip_duplicates, allow_duplicates)
        return obj

    def __init__(self, items, skip_duplicates=False, allow_duplicates=True):
        self._numerator_type = None
        self._denominator_type = None
        super(RatioList, self).__init__(items)

    def _string(self, string):
        string = RatioString(string)
        if len(self) == 0:
            self._numerator_type = string.numerator.__class__
            self._denominator_type = string.denominator.__class__
            return string
        else:
            if not isinstance(string.numerator, self._numerator_type):
                raise NumeratorTypeError(string.numerator, self._numerator_type, self)
            elif not isinstance(string.denominator, self._denominator_type):
                raise DenominatorTypeError(string.denominator, self._denominator_type, self)
            return string

    @property
    def numerators(self):
        """
        Either a :class:`MassList`, :class:`ElementList` or a :class:`RatioList` containing the numerator of each
        item in the list. **Readonly**

        Examples
        --------
        >>> RatioList(['99ru/108Pd', '105pd/108Pd', '111cd/108Pd']).numerators
        ['99Ru', '105Pd', '111Cd']
        """
        if len(self) == 0:
            return []
        else:
            return self._numerator_type._list([rat.numerator for rat in self])

    @property
    def denominators(self):
        """
        Either a :class:`MassList`, :class:`ElementList` or a :class:`RatioList` containing the denominator of each
        item in the list. **Readonly**

        Examples
        --------
        >>> RatioList(['99ru/108Pd', '105pd/108Pd', '111cd/108Pd']).numerators
        ['108Pd', '108Pd', '108Pd']
        """
        if len(self) == 0:
            return []
        else:
            return self._denominator_type._list([rat.denominator for rat in self])

    def has_common_denominator(self, denominator = None):
        """
        Returns ``True`` if the all the items in the list have the same denominator. Otherwise returns ``False``.

        If the optional ``denominator`` argument is given then ``True`` will only be returned if
        the list has a common denominator equal to this string.

        Parameters
        ----------
        denominator : MassString, ElementString, IsotopeString, optional
            Should be given to check if the list has a specific common denominator

        Examples
        --------
        >>> RatioList(['99ru/108Pd', '105pd/108Pd', '111cd/108Pd']).has_common_denominator()
        True
        >>> RatioList(['99ru/108Pd', '105pd/108Pd', '111cd/108Pd']).has_common_denominator('105Pd')
        False
        """
        try:
            denom = self.get_common_denominator()
        except ValueError:
            return False
        else:
            if denominator is None:
                return True
            elif denom == denominator:
                return True
            else:
                return False

    def get_common_denominator(self):
        """
        Returns the common denominator string for each item in the list.

        Raises
        ------
        NoCommonDenominator
            Raised if the list does not contain a common denominator

        Examples
        --------
        >>> RatioList(['99ru/108Pd', '105pd/108Pd', '111cd/108Pd']).get_common_denominator()
        '108Pd'
        """
        denom = self.denominators
        if len(denom) == 0:
            raise ValueError('RatioList is empty')
        elif len(denom) == 1:
            return denom[0]
        else:
            for d in denom[1:]:
                if denom[0] != d:
                    raise ValueError('RatioList does not have a common denominator')
            return denom[0]

    def _filter_indexes(self, index_list, ratio = None, ratio_not = None, **numerator_and_denominator_filters):

        if ratio is not None and not isinstance(ratio, RatioList):
            ratio = RatioList(ratio)

        if ratio_not is not None and not isinstance(ratio_not, RatioList):
            ratio_not = RatioList(ratio_not)

        numerator_filters = {}
        denominator_filters = {}
        for k in numerator_and_denominator_filters:
            ks = k.split('_', 1)
            if ks[0] == 'numerator' or ks[0] == 'n':
                try:
                    numerator_filters[ks[1]] = numerator_and_denominator_filters[k]
                except IndexError:
                    raise ValueError('numerator filter parameter not specified for value "{}"'.format(numerator_and_denominator_filters[k]))
            elif ks[0] == 'denominator' or ks[0] == 'd':
                try:
                    denominator_filters[ks[1]] = numerator_and_denominator_filters[k]
                except IndexError:
                    raise ValueError('denominator filter not specified for value "{}"'.format(numerator_and_denominator_filters[k]))
            else: raise ValueError('"{}" not a valid filter for RatioList'.format(k))

        out = []
        for i in index_list:
            rat = self[i]
            if ratio is not None:
                if rat not in ratio: continue
            if ratio_not is not None:
                if rat in ratio_not: continue
            out.append(i)

        if len(numerator_filters) > 0:
            out = self.numerators._filter_indexes(out, **numerator_filters)

        if len(denominator_filters) > 0:
            out = self.denominators._filter_indexes(out, **denominator_filters)

        return out

    def copy(self, *, ratio = None, ratio_not = None, **numerator_and_denominator_kwargs):
        """
        copy(*, ratio = None, ratio_not = None, **numerator_and_denominator_kwargs)

        Returns a copy of the list.

        If keyword arguments are given then only those items in the list that satisfy these arguments are copied.

        Parameters
        ----------
        ratio : RatioString, RatioList, optional
           Only items in the list matching/found in ``ratio`` are copied.
        ratio_not : RatioString, RatioList, optional
           Only items in the list not matching/not found in ``ratio_not`` are copied.
        **numerator_and_denominator_kwargs
            See :func:`MassList.copy`, :func:`Element.copy` and :func:`IsotopeList.copy` for additional keyword
            arguments. These arguments must be preceded by either 'numerator\_' or 'denominator\_' to specify where the
            keywords should be used.


        Returns
        -------
        RatioList
            A copy of the items in the list that satisfy the keyword arguments or all items in the list if none are
            given.

        Examples
        --------
        >>> RatioList(['99ru/108Pd', '105pd/108Pd', '111cd/108Pd']).copy(ratio=['105pd/108Pd','107ag/108Pd', '111cd/108Pd'])
        ['105Pd/108Pd', '111Cd/108Pd']
        >>> RatioList(['99ru/108Pd', '105pd/108Pd', '111cd/108Pd']).copy(ratio_not='111cd/108Pd')
        ['99Ru/108Pd', '105Pd/108Pd']
        >>> RatioList(['99ru/108Pd', '105pd/108Pd', '111cd/108Pd']).copy(numerator_isotope = ['pd', 'ag', 'cd'])
        ['105Pd/108Pd', '111Cd/108Pd']
        >>> RatioList(['99ru/108Pd', '105pd/108Pd', '111cd/108Pd']).copy(numerator_mass_number_lt = 100)
        ['99Ru/108Pd', '105Pd/108Pd']
        """
        index_list = [i for i in range(len(self))]
        index = self._filter_indexes(index_list, ratio=ratio, ratio_not=ratio_not,
                                     **numerator_and_denominator_kwargs)
        return self[index]


############
### Dict ###
############
class ReferenceDict(dict):
    """
    ReferenceDict(dictionary = None)

    Custom dictionary where every item stores a single value as float64 with an :ref:`IsopyString <isopystring>` key.

    All values are stored in a zero-dimensional numpy array as a 64-bit floating-point number.

    Behaves just like, and contains all the methods, that a normal dictionary does unless otherwise noted. All methods
    that takes a key argument will raise a StringParseError if the key cannot be converted to an
    :ref:`IsopyString <isopystring>`.  Only methods that behave differently from a normal dictionary are documented below.

    Parameters
    ----------
    dictionary : dict
        Must only contain keys which can be parsed into an :ref:`IsopyString <isopystring>` that hold a single value
        that can be converted to a float point value.

    Examples
    --------
    >>> ReferenceDict({'Pd108': 108, '105Pd': 105, 'pd': 46}).keys()
    dict_keys(['108Pd', '105Pd', 'Pd'])
    >>> ReferenceDict({'Pd108': 108, '105Pd': 105, 'pd': 46}).values()
    dict_values([array(108.), array(105.), array(46.)])
    """

    def __init__(self, dictionary = None):
        super(ReferenceDict, self).__init__()
        if dictionary is None: pass
        elif isinstance(dictionary, dict):
            self.update(dictionary)
        else:
            raise TypeError('Reference dict can only be initialised with another dict')

    def __setitem__(self, key, value):
        key = asstring(key)
        value = np.asarray(value, dtype='f8')

        if value.size != 1:
            raise ValueError('ReferenceDict can only store a single value per entry')
        elif value.ndim != 0:
            value = value.reshape(())

        super(ReferenceDict, self).__setitem__(key, value.copy())

    def __contains__(self, key):
        try:
            key = asstring(key)
        except StringParseError:
            return False

        if super(ReferenceDict, self).__contains__(key):
            return True
        elif isinstance(key, RatioString):
            return (super(ReferenceDict, self).__contains__(key.numerator) and
                    super(ReferenceDict, self).__contains__(key.denominator))
        else:
            return False

    def __getitem__(self, key):
        key = asstring(key)
        return super(ReferenceDict, self).__getitem__(key)

    def get(self, key, default = np.nan):
        """
        Return the value for *key* if key is in the dictionary. If *key* is not in the dictionary it will return
        the *default* value for :class:`MassString`, :class:`ElementString` and :class:`IsotopeString` keys.
        For :class:`RatioString` keys not in the dictionary it will return the :func:`RatioString.numerator` key value
        divided by the :func:`RatioString.denominator` key value if both are present in the dictionary, otherwise
        it will return the *default* value.

        If *key* is a list then a list with the value for each key in *key* is returned. If *key* is an
        :ref:`IsopyList <isopylist>` then the corresponding :ref:`IsopyArray <isopyarray>` is returned.

        Parameters
        ----------
        key : :ref:`IsopyString <isopystring>`, :ref:`IsopyList <isopylist>`
            Can be either a string or a list of strings.
        default : optional
            If not given the *default* value is np.nan.

        Examples
        --------
        >>> reference = ReferenceDict({'108Pd': 100, '105Pd': 20, '104Pd': 150, '104Pd/105Pd': 10})
        >>> reference.get('108Pd/105Pd')
        5.0
        >>> reference.get('104Pd/105Pd')
        10.0
        >>> reference.get('104Pd') / reference.get('105Pd')
        7.5
        >>> reference.get('102Pd')
        nan
        """
        if isinstance(key, list):
            out = [self.get(k, default) for k in key]
            if isinstance(key, ElementList):
                return ElementArray(out, keys=key)
            elif isinstance(key, IsotopeList):
                return IsotopeArray(out, keys=key)
            elif isinstance(key, RatioList):
                return RatioArray(out, keys=key)
            else:
                return out

        key = asstring(key)
        try:
            return super(ReferenceDict, self).__getitem__(key)
        except KeyError:
            if isinstance(key, RatioString):
                try:
                    return super(ReferenceDict, self).__getitem__(key.numerator) / \
                           super(ReferenceDict, self).__getitem__(key.denominator)
                except KeyError: pass

            return default

    def update(self, other):
        if not isinstance(other, dict):
            raise ValueError('other must be a dict')

        for k in other.keys():
            self.__setitem__(k, other[k])

    def pop(self, key, default=np.nan):
        key = asstring(key)
        return super(ReferenceDict, self).pop(key, default)

    def setdefault(self, key, default=np.nan):
        key = asstring(key)

        try: return self.__getitem__(key)
        except KeyError:
            self.__setitem__(key, default)
            return self.__getitem__(key)

    def mass_keys(self):
        """
        Returns a :class:`MassList` with all the :class:`MassString` keys in the dict.
        """
        return MassList([k for k in self.keys() if isinstance(k, MassString)])

    def element_keys(self):
        """
        Returns a :class:`ElementList` with all the :class:`ElementString` keys in the dict.
        """
        return ElementList([k for k in self.keys() if isinstance(k, ElementString)])

    def isotope_keys(self):
        """
        Returns a :class:`IsotopeList` with all the :class:`IsotopeString` keys in the dict.
        """
        return IsotopeList([k for k in self.keys() if isinstance(k, IsotopeString)])

    def ratio_keys(self):
        """
        Returns a :class:`RatioList` with all the :class:`RatioString` keys in the dict.
        """
        return RatioList([k for k in self.keys() if isinstance(k, RatioString)])

    def copy(self):
        return ReferenceDict(self)

#############
### Array ###
#############
def check_unsupported_parameters(func_name, unsupported_param, given_param):
    unsupported_used = ', '.join([param for param in unsupported_param if param in given_param])
    if len(unsupported_used) > 0:
        raise TypeError('Use of "{}" parameter(s) is currently not supported for "{}" with isopy arrays'.format(
                unsupported_used, func_name))

class input_array_ndarray:
    """
    This is for 0 or 1 dim arrays where the stored value is the same for each key
    """
    def __init__(self, array):
        self._array = array

    def get(self, key, default_value):
        return self._array


class IsopyArray(IsopyType):
    def __str__(self, delimiter=', '):
        sdict = {}
        if self.ndim == 0:
            for k in self.keys(): sdict[k] = ['{}'.format(self[k])]
        if self.ndim == 1:
            for k in self.keys(): sdict[k] = ['{}'.format(self[k][i]) for i in range(self.size)]

        flen = {}
        for k in self.keys():
            flen[k] = max([len(x) for x in sdict[k]]) + 1
            if len(k) >= flen[k]: flen[k] = len(k) + 1

        return '{}\n{}'.format(delimiter.join(['{:<{}}'.format(k, flen[k]) for k in self.keys()]),
                                   '\n'.join('{}'.format(delimiter.join('{:<{}}'.format(sdict[k][i], flen[k]) for k in self.keys()))
                                             for i in range(self.size)))

    def __repr__(self):
        if self.ndim == 0:
            return '{}({}, dtype={})'.format(self.__class__.__name__, '({})'.format(', '.join(
                [self[k].__repr__() for k in self.keys()])), self.dtype.__repr__())
        else:
            return '{}([{}], dtype={})'.format(self.__class__.__name__, ', '.join(
                ['({})'.format(', '.join(
                    [self[k][i].__repr__() for k in self.keys()])) for i in range(self.size)]), self.dtype.__repr__())

    def __new__(cls, values=None, *, keys=None, ndim=None, dtype=None):
        warnings.warn('please use isopy.array instead', DeprecationWarning)
        return array(values, keys=keys, ndim=ndim, dtype=dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        default_value = kwargs.pop('default_value', np.nan)
        if len(inputs) == 1 and ufunc in [np.sin, np.cos, np.tan, np.arcsin, np.arccos, np.arctan, np.degrees,
                    np.radians, np.deg2rad, np.rad2deg, np.sinh, np.cosh, np.tanh, np.arcsinh, np.arccosh, np.arctanh,
                    np.rint, np.floor, np.ceil, np.trunc, np.exp, np.expm1, np.exp2, np.log, np.log10, np.log2,
                    np.log1p, np.reciprocal, np.positive, np.negative, np.sqrt, np.cbrt, np.square, np.fabs, np.sign,
                    np.absolute, np.abs, np.isnan]: pass
        elif len(inputs) == 2 and ufunc in [np.add, np.multiply, np.divide, np.power, np.subtract, np.true_divide,
                    np.true_divide, np.floor_divide, np.float_power, np.fmod, np.mod, np.remainder,
                    np.greater, np.less, np.greater_equal, np.less_equal, np.equal, np.not_equal, np.maximum,
                    np.minimum]: pass
        else:
            raise TypeError('Function "{}" is not supported by isopy arrays'.format(ufunc.__name__))
        if method != '__call__':
            raise TypeError('method "{}" is currently not supported'.format(method))
        check_unsupported_parameters(ufunc.__name__, ['out', 'where', 'axes', 'axis', 'casting', 'order', 'subok',
                                                     'signature', 'extobj', 'dtype', 'copy'], kwargs.keys())

        is_input_isopy = [isinstance(inp, IsopyArray) for inp in inputs]
        if is_input_isopy.count(True) == 2:
            keys = inputs[0].keys()
            try:
                keys.append(inputs[1].keys(), skip_duplicates = True)
            except ValueError:
                raise KeyError('Both input arrays must be of the same isopy type')
            #inputs = [input_array_isopyarray(inputs[0]), input_array_isopyarray(inputs[1])]
        elif is_input_isopy.count(True) == 1:
            inputs = list(inputs)

            isopy_i = is_input_isopy.index(True)
            keys = inputs[isopy_i].keys()

            try:
                other_i = is_input_isopy.index(False)
            except ValueError:
                #Are used as is
                pass
            else:
                if not isinstance(inputs[other_i], dict):
                    try:
                        other = np.asarray(inputs[other_i])
                    except:
                        raise TypeError('Failed to convert input "{}" to compatible array'.format(inputs[other_i]))

                    if other.ndim > 1:
                        raise TypeError('input "{}" has to many dimensions'.format(inputs[other_i]))

                    if other.dtype.names is not None:
                        raise TypeError('structured ndarrays are currently not supported as inputs')

                    inputs[other_i] = input_array_ndarray(inputs[other_i])
        else:
            raise TypeError('At least one input must be an isopy array')

        output = {}
        for key in keys:
            try:
                output[key] = ufunc(*[inp.get(key, default_value) for inp in inputs], **kwargs)
            except KeyError:
                raise KeyError('Key "{}" not found in all inputs'.format(key))
            except:
                raise

        return self._array(output)

    def __array_function__(self, func, types, args, kwargs):
        default_value = kwargs.pop('default_value', np.nan)
        if func in [np.prod, np.sum, np.nanprod, np.nansum, np.cumprod, np.cumsum, np.nancumprod, np.nancumsum,
                    np.amin, np.amax, np.nanmin, np.nanmax, np.fix, np.ptp, np.percentile, np.nanpercentile, np.quantile,
                    np.nanquantile, np.median, np.average, np.mean, np.std, np.var, np.nanmedian, np.nanmean,
                    np.nanstd, np.nanvar, np.nan_to_num, ipf.mad, ipf.nanmad, ipf.se, ipf.nanse, ipf.sd, ipf.nansd,
                    np.any, np.all]:
            sig = inspect.signature(func)
            bind = sig.bind(*args, **kwargs)
            func_parameters = list(sig.parameters.keys())
            parsed_kwargs = bind.arguments
            check_unsupported_parameters(func.__name__, ['out', 'where', 'overwrite_input'], parsed_kwargs)
            axis = parsed_kwargs.pop('axis', 0)
            array = parsed_kwargs.pop(func_parameters[0])
            if not isinstance(array, IsopyArray):
                raise TypeError('Array must be an isopy array')
            if axis == 0:
                return self._array({key: func(array.get(key, default_value), **parsed_kwargs) for key in array.keys()})
            else:
                return func(np.transpose([array.get(key, default_value) for key in array.keys()]), axis = axis, **parsed_kwargs)

        elif func in [np.around, np.round]:
            sig = inspect.signature(func)
            bind = sig.bind(*args, **kwargs)
            parsed_kwargs = bind.arguments
            check_unsupported_parameters(func.__name__, ['out'], parsed_kwargs)
            array = parsed_kwargs.pop('a')
            if not isinstance(array, IsopyArray):
                raise TypeError('Array must be an isopy array')
            return self._array({key: func(array.get(key, default_value), **parsed_kwargs) for key in array.keys()})

        elif func is np.concatenate:
            sig = inspect.Signature([inspect.Parameter('arr', inspect.Parameter.POSITIONAL_ONLY),
                                     inspect.Parameter('axis', inspect.Parameter.POSITIONAL_OR_KEYWORD, default=0),
                                     inspect.Parameter('out', inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None)])
            bind = sig.bind(*args, **kwargs)
            func_parameters = list(sig.parameters.keys())
            parsed_kwargs = bind.arguments
            check_unsupported_parameters(func.__name__, ['out'], parsed_kwargs)
            axis = parsed_kwargs.pop('axis', 0)
            arr = parsed_kwargs.pop(func_parameters[0])
            if not isinstance(arr[0], IsopyArray) or False in [isinstance(a, arr[0].__class__) for a in arr[1:]]:
                #Fails if one is void
                raise TypeError('All arrays must be Isopy Arrays of the same type')

            if axis == 0:
                try:
                    keys = IsopyList([a.keys() for a in arr], skip_duplicates=True)
                except:
                    raise ValueError('Unable to parse keys into a single isopy list')

                return IsopyArray({key: func([a.get(key, default_value) for a in arr], axis=0) for key in keys})
            elif axis == 1:
                if False in [a.size==arr[0].size for a in arr[1:]]:
                    raise ValueError('All arrays must have the same size')
                try: keys = IsopyList([a.keys() for a in arr], allow_duplicates=False)
                except: raise ValueError('Unable to parse keys into a single isopy list with no duplicates')

                arrd = {}
                for a in arr: arrd.update({key: a.get(key, default_value) for key in a.keys()})

                return IsopyArray(arrd)
            else:
                raise np.AxisError(axis, 2)

        elif func is np.append:
            sig = inspect.signature(func)
            bind = sig.bind(*args, **kwargs)
            parsed_kwargs = bind.arguments
            array1 = parsed_kwargs.pop('arr')
            array2 = parsed_kwargs.pop('values')
            axis = parsed_kwargs.pop('axis', 0)
            return np.concatenate((array1, array2), axis = axis, default_value=default_value)

        else:
            raise TypeError('The use of {} is not supported by isopy arrays'.format(func.__name__))

    def _view(self, obj):
        if obj.dtype.names:
            if isinstance(obj, np.core.numerictypes.void):
                return self._void(obj)
            else:
                return self._ndarray(obj)
        else:
            return obj.view(np.ndarray)

    def copy_to_clipboard(self, delimiter=', '):
        """
        Uses the pyperclip package to copy the array to the clipboard.

        Parameters
        ----------
        delimiter: std, optional
            The delimiter for columns in the data. Default is ', '.
        """
        pyperclip.copy(self.__str__(delimiter))

    def keys(self):
        return self._list(self.dtype.names)

    @property
    def nkeys(self):
        """
        The number of keys in the array. Analogous to ``len(array.keys())``
        """
        return len(self.dtype.names)

    @property
    def nrows(self):
        """
        The number of rows in the array. If the array is zero-dimensional ``-1`` is returned.

        Differs from ``array.size`` which will always return a positive integer. So ``array.size`` is equal
        to ``abs(array.nrows)``.
        """
        if self.ndim == 0: return -1
        else: self.size

    def get(self, key, default=np.nan):
        try:
            return self[key]
        except:
            default = np.asarray(default)
            if default.ndim == 0 and self.ndim == 0: return default.copy()
            elif default.ndim == 0:
                out = np.empty(self.size)
                out[:] = default
                return out
            elif default.size != self.size: #default.shape?
                raise ValueError('default_value must be zero-dimensional or the same size as the array')
            else:
                return default.copy()


class MassArray(IsopyArray):
    def __new__(cls, values=None, *, keys=None, ndim=None, dtype=None):
        return MassNdarray(values, keys=keys, ndim=ndim, dtype=dtype)

    def copy(self, **mass_number_kwargs):
        """
        Return a copy of the array.

        If keyword arguments are given then only the keys in the array that satisfy these arguments is copied.

        Parameters
        ----------
        **mass_number_kwargs
            See :func:`MassList.copy` for a description of the avaliable keyword arguments.
        """
        if mass_number_kwargs:
            keys = self.keys().copy(**mass_number_kwargs)
            return super(MassArray, self[keys]).copy()
        else:
            return super(MassArray, self).copy()

    def keys(self):
        """
        Returns a :class:`MassList` containing the keys in the array.
        """
        return super(MassArray, self).keys()

    def get(self, key, default=np.nan):
        """
        Return the data for *key* if in the array. If missing then an array consisting of the *default* value with
        the same size as the current array is returned.

        Parameters
        ----------
        key : :class:`MassString`
            The key to be returned.
        default
            Must either be an zero-dimensional value or a array of values the same size as the array.

        Raises
        ------
        ValueError
            Raised if the size of *default* is not the same as the array.

        Returns
        -------
        ndarray
            An array with the data associated with *key*.
        """
        return super(ElementArray, self).get(key, default)


class ElementArray(IsopyArray):
    """
    An array storing data with :class:`ElementString` keys.

    Bit more description.

    This is a custom subclass of a `structured numpy array <https://docs.scipy.org/doc/numpy/user/basics.rec.html>`
    and therefore contains all the methods and attributes that a normal numpy ndarray does.
    """
    def __new__(cls, values=None, *, keys=None, ndim=None, dtype=None):
        return ElementNdarray(values, keys=keys, ndim=ndim, dtype=dtype)

    def copy(self, **element_symbol_kwargs):
        """
        Return a copy of the array.

        If keyword arguments are given then only the keys in the array that satisfy these arguments is copied.

        Parameters
        ----------
        **element_symbol_kwargs
            See :func:`ElementList.copy` for a description of the avaliable keyword arguments.
        """
        if element_symbol_kwargs:
            keys = self.keys().copy(**element_symbol_kwargs)
            return super(ElementArray, self[keys]).copy()
        else:
            return super(ElementArray, self).copy()

    def keys(self):
        """
        Returns a :class:`ElementList` containing the keys in the array.
        """
        return super(ElementArray, self).keys()

    def get(self, key, default=np.nan):
        """
        Return the data for *key* if in the array. If missing then an array consisting of the *default* value with
        the same size as the current array is returned.

        Parameters
        ----------
        key : :class:`ElementString`
            The key to be returned.
        default
            Must either be an zero-dimensional value or a array of values the same size as the array.

        Raises
        ------
        ValueError
            Raised if the size of *default* is not the same as the array.

        Returns
        -------
        ndarray
            An array with the data associated with *key*.
        """
        return super(IsotopeArray, self).get(key, default)


class IsotopeArray(IsopyArray):
    def __new__(cls, values=None, *, keys=None, ndim=None, dtype=None):
        return IsotopeNdarray(values, keys=keys, ndim=ndim, dtype=dtype)

    def copy(self, **isotope_kwargs):
        """
        Return a copy of the array.

        If keyword arguments are given then only the keys in the array that satisfy these arguments is copied.

        Parameters
        ----------
        **isotope_kwargs
            See :func:`IsotopeList.copy` for a description of the avaliable keyword arguments.
        """
        if isotope_kwargs:
            keys = self.keys().copy(**isotope_kwargs)
            return super(IsotopeArray, self[keys]).copy()
        else:
            return super(IsotopeArray, self).copy()

    def keys(self):
        """
        Returns a :class:`IsotopeList` containing the keys in the array.
        """
        return super(IsotopeArray, self).keys()

    def get(self, key, default=np.nan):
        """
        Return the data for *key* if in the array. If missing then an array consisting of the *default* value with
        the same size as the current array is returned.

        Parameters
        ----------
        key : :class:`IsotopeString`
            The key to be returned.
        default
            Must either be an zero-dimensional value or a array of values the same size as the array.

        Raises
        ------
        ValueError
            Raised if the size of *default* is not the same as the array.

        Returns
        -------
        ndarray
            An array with the data associated with *key*.
        """
        return super(RatioArray, self).get(key, default)


class RatioArray(IsopyArray):
    def __new__(cls, values=None, *, keys=None, ndim=None, dtype=None):
        return RatioNdarray(values, keys=keys, ndim=ndim, dtype=dtype)

    def copy(self, **ratio_kwargs):
        """
        Return a copy of the array.

        If keyword arguments are given then only the keys in the array that satisfy these arguments is copied.

        Parameters
        ----------
        **ratio_kwargs
            See :func:`RatioList.copy` for a description of the avaliable keyword arguments.
        """
        if ratio_kwargs:
            keys = self.keys().copy(**ratio_kwargs)
            return super(RatioArray, self[keys]).copy()
        else:
            return super(RatioArray, self).copy()

    def keys(self):
        """
        Returns a :class:`RatioList` containing the keys in the array.
        """
        return super(RatioArray, self).keys()

    def get(self, key, default=np.nan):
        """
        Return the data for *key* if in the array. If missing then an array consisting of the *default* value with
        the same size as the current array is returned.

        Parameters
        ----------
        key : :class:`RatioString`
            The key to be returned.
        default
            Must either be an zero-dimensional value or a array of values the same size as the array.

        Raises
        ------
        ValueError
            Raised if the size of *default* is not the same as the array.

        Returns
        -------
        ndarray
            An array with the data associated with *key*.
        """
        return super(MassArray, self).get(key, default)


###############
### Ndarray ###
###############

def _new_dtype(dtype, keys=None):
    if isinstance(dtype, (np.ndarray, np.void)):
        if dtype.dtype.names:
            dtypedict = dtype.dtype.fields
            dtype = [dtypedict[name][0] for name in dtype.dtype.names]
        else:
            dtype = dtype.dtype

    if isinstance(dtype, list):
        if keys is None:
            raise ValueError('Expected dtype got list')
        elif len(dtype) != len(keys):
            raise ValueError('Length of "dtype" argument does not match the number of keys')
        else:
            return [(keys[i].varname(), _new_dtype(dtype[i])) for i in range(len(keys))]
    else:
        if keys is None:
            try:
                return np.dtype(dtype)
            except:
                raise ValueError('Could not convert "{}" to numpy dtype'.format(dtype))
        else:
            return [(keys[i].varname(), _new_dtype(dtype)) for i in range(len(keys))]

def _new_keys(cls, old_keys, new_keys):
    if isinstance(old_keys, (np.ndarray, np.void)):
        old_keys = list(old_keys.dtype.names)
    if isinstance(old_keys, dict):
        old_keys = list(old_keys.keys())

    if new_keys is None:
        return cls._list(old_keys)
    elif isinstance(new_keys, list):
        if len(old_keys) != len(new_keys):
            raise ValueError('Length of "keys" argument does not equal the number of keys in the array')
        return cls._list(new_keys)
    elif isinstance(new_keys, dict):
        raise NotImplemented()
    else:
        raise TypeError('"keys" argument has an invalid type {}'.format(type(new_keys)))

class IsopyNdarray(IsopyArray, np.ndarray):
    def __new__(cls, values, *, keys=None, ndim=None, dtype=None):
        if isinstance(values, int):
            if values < 1: raise ValueError('Cannot create empty array: "values" must be a value larger than 1')

            if keys is None: raise ValueError('Cannot create empty array: "keys" argument missing')
            fkeys = cls._list(keys)

            if dtype is None: dtype = 'f8'
            dtype = _new_dtype(dtype, fkeys)

            if ndim == 0 and values != 1:
                raise ValueError('Cannot create a zero dimensional array with a size other than 1')
            elif ndim == 0 or (ndim == -1 and values == 1):
                values = None

            obj = np.zeros(values, dtype = dtype)
            return cls._ndarray(obj)

        if isinstance(values, IsopyNdarray) and keys is None and ndim is None and dtype is None:
            return values.copy()

        if isinstance(values, np.void):
            fkeys = _new_keys(cls, values, keys)
            if dtype is None: dtype = _new_dtype(values, fkeys)
            else: dtype = _new_dtype(dtype, fkeys)

            if ndim == 1:
                obj = np.zeros(1, dtype)
            else:
                obj = np.zeros(None, dtype)

            for i in range(len(fkeys)):
                obj[fkeys[i].varname()] = values[values.dtype.names[i]]
            return cls._ndarray(obj)

        # TODO do seperatley so that each key cna have a different dtype
        if isinstance(values, Table):
            values = values.read()
            #values = {k: values.col(k) for k in values.colnames}

        if isinstance(values, list):
            values = np.asarray(values)

        if isinstance(values, dict):
            dkeys = list(values.keys())
            values = np.asarray([values[key] for key in dkeys])
            if keys is None: keys = dkeys
            else:
                #TODO implement when keys can be a dict
                raise ValueError('"keys" argument can not be given together with a dict')

        if isinstance(values, np.ndarray):
            if values.dtype == object:
                raise ValueError('dtype "object" not supported')

            if values.dtype.names:
                if values.ndim > 1:
                    raise ValueError('"values" argument has to many dimensions ({})'.format(values.ndim))

                fkeys = _new_keys(cls, values, keys)
                if dtype is None: dtype = _new_dtype(values, fkeys)
                else: dtype = _new_dtype(dtype, fkeys)

                if ndim is None: ndim = values.ndim
                if ndim == 0 and values.size != 1:
                    raise ValueError('Cannot create a zero dimensional array with a size other than 1')
                if ndim == 0 or (ndim == -1 and values.size == 1):
                    obj = np.zeros(None, dtype)
                else:
                    obj = np.zeros(values.size, dtype)

                if obj.ndim == 0 and values.ndim == 1:
                    for i in range(len(fkeys)):
                        obj[fkeys[i].varname()] = values[values.dtype.names[i]][0]
                else:
                    for i in range(len(fkeys)):
                        obj[fkeys[i].varname()] = values[values.dtype.names[i]]

                return cls._ndarray(obj)
            else:
                if values.ndim > 2:
                    raise ValueError('"values" argument has to many dimensions ({})'.format(values.ndim))
                if keys is None:
                    raise ValueError('"keys" argument missing')
                else:
                    fkeys = cls._list(keys)

                if dtype is None: dtype = _new_dtype(values, fkeys)
                else: dtype = _new_dtype(dtype, fkeys)

                if values.ndim == 1:
                    nkeys = values.size
                    if ndim == 1:
                        nvalues = 1
                    else:
                        nvalues = None
                else:
                    nkeys = values.shape[0]
                    nvalues = values.shape[1]
                    if ndim == 0 and nvalues != 1:
                        raise ValueError('Cannot create a zero dimensional array with a size other than 1')
                    elif ndim == 0 or (ndim == -1 and nvalues == 1):
                        nvalues = None

                if nkeys != len(fkeys):
                    raise ValueError('Size of the first dimension of "values" argument does not match the number of keys given')

                obj = np.zeros(nvalues, dtype)

                if obj.ndim == 0 and values.ndim == 2:
                    for i in range(len(fkeys)):
                        obj[fkeys[i].varname()] = values[i][0]
                else:
                    for i in range(len(fkeys)):
                        obj[fkeys[i].varname()] = values[i]

                return cls._ndarray(obj)

        raise ValueError('Unable to create array with value type {}'.format(type(values)))

    def __setitem__(self, key, value):
        if isinstance(key, str):
            try:
                key = self._string(key).varname()
            except:
                pass
        super(IsopyNdarray, self).__setitem__(key, value)

    def __getitem__(self, key):
        if isinstance(key, str):
            try:
                key = self._string(key).varname()
            except:
                pass
        elif isinstance(key, list):
            if len(key) == 0:
                return np.array([])
            elif isinstance(key[0], str):
                # Can be smarter
                try:
                    key = self._list(key).varnames()
                except:
                    pass
        return self._view(super(IsopyNdarray, self).__getitem__(key))

    def __array_finalize__(self, obj):
        if obj is None: return
        if obj.dtype.names is None: raise ValueError('Only structured arrays can be used with this view')
        try:
            self._list(obj.dtype.names)
        except:
            raise
        return obj


class MassNdarray(MassType, IsopyNdarray, MassArray):
    pass


class ElementNdarray(ElementType, IsopyNdarray, ElementArray):
    pass


class IsotopeNdarray(IsotopeType, IsopyNdarray, IsotopeArray):
    pass


class RatioNdarray(RatioType, IsopyNdarray, RatioArray):
    pass

############
### Void ###
############

class IsopyVoid(IsopyArray, np.void):
    def __new__(cls, void):
        return void.view((cls, void.dtype))

    def __len__(self):
        # Should raise same error as ndarray
        raise NotImplemented()
        #pass

    def __setitem__(self, key, value):
        if isinstance(key, int):
            raise IndexError('{} cannot be indexed by position'.format(self.__class__.__name__))
        elif isinstance(key, str):
            try:
                key = self._string(key).varname()
            except:
                pass
        super(IsopyArray, self).__setitem__(key, value)

    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            raise IndexError('{} cannot be indexed by position'.format(self.__class__.__name__))
        elif isinstance(key, str):
            try:
                key = self._string(key).varname()
            except:
                pass
        elif isinstance(key, list):
            if len(key) == 0:
                return np.array([])
            elif isinstance(key[0], str):
                # Can be smarter
                try:
                    key = [k.varname() for k in self._list(key)]
                except:
                    pass
        # TODO check if it return IsopyVoid
        return super(IsopyVoid, self).__getitem__(key)


class MassVoid(MassType, IsopyVoid, MassArray):
    pass


class ElementVoid(ElementType, IsopyVoid, ElementArray):
    pass


class IsotopeVoid(IsotopeType, IsopyVoid, IsotopeArray):
    pass


class RatioVoid(RatioType, IsopyVoid, RatioArray):
    pass









