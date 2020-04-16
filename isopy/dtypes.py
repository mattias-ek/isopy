import numpy as np
import pyperclip
from tables import Table
import inspect
import isopy.functions as ipf
import warnings

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

    def safe_format(self):
        return self


class MassString(MassType, IsopyString):
    def __new__(cls, string):
        if isinstance(string, cls):
            return string
        elif isinstance(string, int):
            if string < 0: raise ValueError('MassString must be a positive integer ({})'.format(string))
            return str.__new__(cls, str(string))
        elif isinstance(string, str):
            string.strip()
            if string[:1] == '_':
                string = string[1:]
            if not string.isdigit():
                raise ValueError('MassString can only contain numerical characters ({})'.format(string))
            if int(string) < 0:
                raise ValueError('MassString must be a positive integer ({})'.format(string))
            return str.__new__(cls, string)
        else:
            raise TypeError("ElementString must be initialised with a <class 'str'> or <class 'int'> ({})".format(type(string)))

    def safe_format(self):
        return '_{}'.format(self)

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
        if isinstance(item, int):
            return int(self).__ge__(item)
        else:
            raise TypeError('MassString only supports ">=" operator with other MassStrings or integers')

    def __le__(self, item):
        if isinstance(item, MassString):
            item = int(item)
        if isinstance(item, int):
            return int(self).__le__(item)
        else:
            raise TypeError('MassString only supports "<=" operator with other MassStrings or integers')

    def __gt__(self, item):
        if isinstance(item, MassString):
            item = int(item)
        if isinstance(item, int):
            return int(self).__gt__(item)
        else:
            raise TypeError('MassString only supports ">" operator with other MassStrings or integers')

    def __lt__(self, item):
        if isinstance(item, MassString):
            item = int(item)
        if isinstance(item, int):
            return int(self).__lt__(item)
        else:
            raise TypeError('MassString only supports "<" operator with other MassStrings or integers')

    def __add__(self, other):
        if isinstance(other, int):
            return int(self).__add__(other)
        else:
            raise TypeError('MassString only supports "+" operator with integers')

    def __truediv__(self, other):
        if isinstance(other, int):
            return int(self).__truediv__(other)
        elif isinstance(other, str):
            return RatioString('{}/{}'.format(self, other))
        else:
            raise TypeError('MassString only supports "/" operator with with other MassStrings or integers')

    def __mul__(self, other):
        if isinstance(other, int):
            return int(self).__mul__(other)
        else:
            raise TypeError('MassString only supports "*" operator with integers')

    def __sub__(self, other):
        if isinstance(other, int):
            return int(self).__sub__(other)
        else:
            raise TypeError('MassString only supports "-" operator with integers')


class ElementString(ElementType, IsopyString):
    def __new__(cls, string):
        if isinstance(string, cls):
            return string
        elif isinstance(string, str):
            string = string.strip()
            if len(string) == 0:
                raise ValueError('ElementString empty')
            if len(string) > 2:
                raise ValueError('ElementString is limited to two characters')
            if not string.isalpha(): raise ValueError(
                'ElementString "{}" can only contain alphabetical characters'.format(string))
            return str.__new__(cls, string.capitalize())
        else:
            raise TypeError("ElementString must be initialised with a <class 'str'> ({})".format(type(string)))

    def __truediv__(self, other):
        return RatioString('{}/{}'.format(self, other))


class IsotopeString(IsotopeType, IsopyString):
    def __new__(cls, string):
        if isinstance(string, cls):
            return string
        elif isinstance(string, str):
            string = string.strip()

            # If no digits in string then only Symbol is given.
            if string.isalpha():
                raise ValueError('"{}" does not contain a mass number'.format(string))

            # If only digits then only mass number
            if string.isdigit():
                raise ValueError('"{}" does not contain an element symbol'.format(string))

            #safe string starts with an _
            if string[:1] == '_': string = string[1:]

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

    def safe_format(self):
        return '_{}'.format(self)

    def __contains__(self, string):
        """
        Return **True** if `string` is equal to the IsotopeString's mass number or element symbol
        """       
        return self.mass_number == string or self.element_symbol == string

    def __truediv__(self, other):
        return RatioString('{}/{}'.format(self, other))


class RatioString(RatioType, IsopyString):
    def __new__(cls, string):
        if isinstance(string, cls):
            return string
        elif isinstance(string, str):
            string = string.strip()
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

    def safe_format(self):
        return '{}_OVER_{}'.format(self.numerator.safe_format(), self.denominator)

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
                    raise ValueError('item {} already in list'.format(item))

        super().__init__(items)

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
        elif isinstance(index, list):
                return self.__class__([super(IsopyList, self).__getitem__(i) for i in index])
        else:
            return super(IsopyList, self).__getitem__(index)

    def __setitem__(self, key, value):
        """
        Not allowed. Raises NotImplementedError
        """
        raise NotImplementedError()

    def count(self):
        return self.__len__()

    def safe_format(self):
        return [s.safe_format() for s in self]

    def has_duplicates(self):
        for item in self[1:]:
            if self.count(item) > 1: return True

        return False

    def remove_duplicates(self):
        for item in self[1:]:
            while self.count(item) > 1: self.remove(item)

    def append(self, item, skip_duplicates = False, allow_duplicates = True):
        """
        Append item to the end of the list
        """
        if not isinstance(item, str) and hasattr(item, '__iter__'):
            raise TypeError('iterables other than strings cannot be added to IsopyLists')

        k = self._string(item)
        if skip_duplicates and k in self: pass
        elif not allow_duplicates and k in self:
            raise ValueError('item "{}" already in list'.format(item))
        else:
            super(IsopyList, self).append(k)

    def extend(self, items, skip_duplicates=False, allow_duplicates=True):
        if isinstance(items, str): items = items.split((','))
        if not hasattr(items, '__iter__'): raise TypeError('items must be iterable')

        items = self._list(items, skip_duplicates, allow_duplicates)
        if skip_duplicates: super(IsopyList, self).extend([item for item in items if item not in self])
        elif not allow_duplicates:
            for item in items:
                if item in self:
                    raise ValueError('item {} already in list'.format(item))
        else:
            super(IsopyList, self).extend(items)

    def index(self, item, *args):
        """
        index(x[, start[, end]])

        Return first index of value.

        Raises ValueError if the value is not present.
        """
        return super(IsopyList, self).index(self._string(item), *args)

    def insert(self, index, item):
        """
        Insert item before index

        """
        return super(IsopyList, self).insert(index, self._string(item))

    def remove(self, item):
        """
        Remove first occurrence of item.

        Raises ValueError if the item is not present.
        """
        return super(IsopyList, self).remove(self._string(item))

    def copy(self):
        return self.__class__(super(IsopyList, self).copy())


class MassList(MassType, IsopyList):
    """
    MassList(items)

    A list for storing `MassInteger`_ items.


    Parameters
    ----------
    items : MassInteger, MassList, or MassArray, optional
        If an integer then return [items]. If a list then return a copy of the list. If an array then return a copy of
        the array keys. If not given then return an empty list.


    Examples
    --------
    >>> MassList(['104', '105', '106'])
    ['104', '105', '106']
    >>> MassList('105')
    ['105']
    >>> MassList(MassArray(keys = ['104', '105', '106'], size = 1))
    ['104', '105', '106']
    """

    def __new__(cls, items, skip_duplicates = False, allow_duplicates=True):
        obj = list.__new__(cls)
        obj.__init__(items, skip_duplicates, allow_duplicates)
        return obj

    def __div__(self, denominator):
        """
        Same as __truediv__(denominator).
        """
        return self.__truediv__(denominator)

    def __truediv__(self, denominator):
        """
        Returns a `RatioList`_ where the numerator of each `RatioString`_ is equal to the MassInteger in the list and
        the denominator is equal to `denominator`.


        Parameters
        ---------
        denominator : str, list of str
            If a list then it must have the same length as the list.

        Examples
        --------
        >>> a = MassList([101, 105, 111])
        >>> a/'108Pd'
        ['101/108Pd', '105/108Pd', '111/108Pd']
        >>> a/['Ru', 'Pd', 'Cd']
        ['101/Ru', '105/Pd', '111/Cd']
        """
        if isinstance(denominator, list):
            if len(denominator) != len(self): raise ValueError('Length of supplied list is not the same as subject list')
            return RatioList(['{}/{}'.format(self[i], denominator[i]) for i in range(len(self))])

        if isinstance(denominator, str):
            return RatioList(['{}/{}'.format(x, denominator) for x in self])

        raise ValueError('unable to parse {}'.format(denominator))

    def _filter_indexes(self, index_list, *, mass_number=None, mass_number_not=None, mass_number_lt=None,
                        mass_number_gt=None, mass_number_le=None, mass_number_ge=None, **_unused_kwargs):

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

            # It has passed all the tests above
            out.append(i)

        return out


    def copy(self, *, mass_number=None, mass_number_not=None, mass_number_lt=None, mass_number_gt=None,
               mass_number_le=None, mass_number_ge=None):
        """
        copy(*,mass_number = None,  mass_number_not = None, mass_number_lt=None, mass_number_gt=None, mass_number_le=None, mass_number_ge=None)

        Returns a copy of the current list.  *DESCRIPTION*


        Parameters
        ----------
        mass_number : MassInteger, MassList, optional
            Only MassIntegers matching this string/found in this list will pass.
        mass_number_not : MassInteger, MassList, optional
            Only MassIntegers not matching this string/not found in this list will pass.
        mass_number_lt=None : MassInteger, optional
            Only MassIntegers less than this value will pass.
        mass_number_gt=None : MassInteger, optional
            Only MassIntegers greater than this value will pass.
        mass_number_le=None : MassInteger, optional
            Only MassIntegers less than or equal to this value will pass.
        mass_number_ge=None : MassInteger, optional
            Only MassIntegers greater than or equal to this value will pass.


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
        ElementList(items)

        A list for storing `ElementString`_ items.


        Parameters
        ----------
        items : ElementString, ElementList, or ElementArray, optional
            If an integer then return [items]. If a list then return a copy of the list. If an array then return a copy of
            the array keys. If not given then return an empty list.


        Examples
        --------
        >>> ElementList(['Ru', 'Pd', 'Cd'])
        ['Ru', 'Pd', 'Cd']
        >>> ElementList(['ru', 'pd', 'cd'])
        ['Ru', 'Pd', 'Cd']
        >>> ElementList('pd')
        ['Pd']
        >>> ElementList(ElementArray(keys = ['Ru', 'Pd', 'Cd'], size = 1))
        ['Ru', 'Pd', 'Cd']
        """

    def __new__(cls, items, skip_duplicates=False, allow_duplicates=True):
        obj = list.__new__(cls)
        obj.__init__(items, skip_duplicates, allow_duplicates)
        return obj

    def __div__(self, denominator):
        """
        Same as __truediv__(denominator).
        """
        return self.__truediv__(denominator)

    def __truediv__(self, denominator):
        """
        Returns a `RatioList`_ where the numerator of each `RatioString`_ is equal to the ElementString in the list and
        the denominator is equal to `denominator`.


        Parameters
        ---------
        denominator : str, list of str
            If a list then it must have the same length as the list.

        Examples
        --------
        >>> a = ElementList(['Ru', 'Pd', 'Cd'])
        >>> a/'108Pd'
        ['Ru/108Pd', 'Pd/108Pd', 'Cd/108Pd']
        >>> a/['Ru', 'Pd', 'Cd']
        ['Ru/Ru', 'Pd/Pd', 'Cd/Cd']
        """
        if isinstance(denominator, list):
            if len(denominator) != len(self): raise ValueError('Length of supplied list is not the same as subject list')
            return RatioList(['{}/{}'.format(self[i], denominator[i]) for i in range(len(self))])

        if isinstance(denominator, str):
            return RatioList(['{}/{}'.format(x, denominator) for x in self])

        raise ValueError('unable to parse {}'.format(denominator))

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

        Returns a copy of the current list where

        Parameters
        ----------
        element_symbol : ElementString, ElementList
           Only ElementStrings matching this string/found in this list will pass.
        element_symbol_not : ElementString, ElementList
           Only ElementStrings not matching this string/not found in this list will pass.


        Returns
        -------
        ElementList
           A new ElementList with each ElementString in the list that passed the filters.

        """
        index_list = [i for i in range(len(self))]
        index = self._filter_indexes(index_list, element_symbol=element_symbol, element_symbol_not=element_symbol_not)
        return self[index]


class IsotopeList(IsotopeType, IsopyList):
    """
    IsotopeList(items)

    A list for storing IsotopeString items.


    Parameters
    ----------
    items : IsotopeString, IsotopeList, IsotopeArray, optional
        If a string then return [items]. If a list then return a copy of the list. If an array then return a copy of
        the array keys. If not given then return an empty list.

    Attributes
    ----------
    mass_numbers : MassList
        Return a list of the mass number for each IsotopeString in the list
    element_symbols : ElementList
        Return a list of the element symbol for each IsotopeString in the list


    Examples
    --------
    >>> IsotopeList(['104Pd', '105Pd', '106Pd'])
    ['104Pd', '105Pd', '106Pd']
    >>> IsotopeList(['104pd', 'pd105', 'Pd106'])
    ['104Pd', '105Pd', '106Pd']

    >>> a = IsotopeList(['104Pd', '105Pd', '106Pd'])
    >>> a.mass_numbers
    [104, 105, 106]
    >>> a.element_symbols
    ['Pd', 'Pd', 'Pd']
    """

    def __new__(cls, items, skip_duplicates=False, allow_duplicates=True):
        obj = list.__new__(cls)
        obj.__init__(items, skip_duplicates, allow_duplicates)
        return obj

    def __div__(self, denominator):
        """
        Same as __truediv__(denominator).
        """
        return self.__truediv__(denominator)

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
        return MassList([x.mass_number for x in self])

    @property
    def element_symbols(self):
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

    def copy(self, *, isotope = None, isotope_not = None, **mass_number_and_element_symbol_filters):
        """
        filter(isotope = None, *, isotope_not = None, **mass_number_and_element_symbol_filters)

        Returns a copy of the current list.

        Parameters
        ----------
        isotope : IsotopeString, IsotopeList
            Only IsotopeStrings matching this string/found in this list will pass.
        isotope_not : IsotopeString, IsotopeList
            Only IsotopeStrings not matching this string/not found in this list will pass.
        mass_number_and_element_symbol_filters
            See :func:`MassList.copy()<isopy.dtypes.MassList.filter>` and
            :func:`ElementList.copy()<isopy.dtypes.ElementList.filter>` for a list of available filters and their descriptions.


        Returns
        -------
        IsotopeList
            A new IsotopeList with each IsotopeString in the list that passed the filters.

        """
        index_list = [i for i in range(len(self))]
        index = self._filter_indexes(index_list, isotope=isotope, isotope_not=isotope_not,
                                     **mass_number_and_element_symbol_filters)
        return self[index]


class RatioList(RatioType, IsopyList):
    """
    RatioList(items)

    A list for storing RatioString items.


    Parameters
    ----------
    items : RatioString, RatioList, RatioArray, optional
        If a string then return [items]. If a list then return a copy of the list. If an array then return a copy of
        the array keys. If not given then return an empty list.


    Attributes
    ----------
    numerators : MassList, ElementList or IsotopeList
        Return a list of the numerator for each RatioString in the list.
    denominator : MassList, ElementList or IsotopeList
        Return a list of the denominator for each RatioString in the list.


    Examples
    --------
    >>> RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd'])
    ['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd']
    >>> RatioList(['104Pd/pd', '105Pd/pd', '106Pd/pd'])
    ['104Pd/Pd', '105Pd/Pd', '106Pd/Pd']

    >>> a = RatioList(['104Pd/Pd', '105Pd/Pd', '106Pd/Pd'])
    >>> a.numerators
    ['104Pd', '105Pd', '106Pd']
    >>> a.denominators
    ['Pd', 'Pd', 'Pd']
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
                raise TypeError('numerator must be an {} (not {}'.format(self._numerator_type, type(string.numerator)))
            elif not isinstance(string.denominator, self._denominator_type):
                raise TypeError('denominator must be an {} (not {}'.format(self._denominator_type, type(string.denominator)))
            return string

    @property
    def numerators(self):
        if len(self) == 0:
            return []
        else:
            return self._numerator_type._list([rat.numerator for rat in self])

    @property
    def denominators(self):
        if len(self) == 0:
            return []
        else:
            return self._denominator_type._list([rat.denominator for rat in self])

    def has_common_denominator(self, denominator = None):
        """
        Return **True** if each RatioString in the list has the same denominator. Otherwise return **False**
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
        Return the common denominator for each item in the list.

        Raise ValueError is no common denominator exists.
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

    def copy(self, *, ratio = None, ratio_not = None, **numerator_and_denominator_filters):
        """
        Returns a copy of the current list.


        Parameters
        ----------
        ratio : RatioString, RatioList, optional
            Only RatioStrings matching this string/found in this list will pass.
        ratio_not : RatioString, RatioList, optional
            Only RatioStrings not matching this string/not found in this list will pass.
        numerator_and_denominator_filters
            Use *numerator_* and *denominator_* prefix to specify filters for the numerators and the denominators.
            See :func:`MassList.copy()<isopy.dtypes.MassList.filter>`,
            :func:`ElementList.copy()<isopy.dtypes.ElementList.filter>` and
            :func:`IsotopeList.copy()<isopy.dtypes.IsotopeList.filter>` for a list of available filters and their
            descriptions.

        Returns
        -------
        RatioList
            A new RatioList with each RatioString in the list that passed the filters.
        """
        index_list = [i for i in range(len(self))]
        index = self._filter_indexes(index_list, ratio=ratio, ratio_not=ratio_not,
                                     **numerator_and_denominator_filters)
        return self[index]


############
### Dict ###
############
class ReferenceDict(dict):
    """
    ReferenceDict(values = None, *, keys = None, dtype = None, filepath = None, **file_kwargs)

    A custom dict where a keys are stored as isopy strings.

    All keys will, if they can, be converted into an MassInteger, ElementString, IsotopeString
    or RatioString when added to the dict. All values will be converted to a 'dtype' numpy arrays. Otherwise ReferenceDict
    behaves like a normal python dict.


    Parameters
    ----------
    values : tuple, list, ndarray, optional
        Values must be compatible with the `dtype` given.
    keys : tuple, list, optional
        Keys for the dict. Keys that cannot be converted to an isopy type will be used as given. If values has keys
        then keys must be a dict mapping the old key to a new key. Keys not found in this dict will be default to the old key.
    dtype : str, optional
        Accepts any numpy compatible data type. If **None** then data fill be stored as it is given. Defaults to
        64-bit float ('f8')
    ndim : int, optional
        number of dimensions of values. If -1 then arrays will be made dimensionless if possible.
         Defaults to -1
    filepath : str, optional
        Path of file on disk to read data from.
    file_kwargs
        Additional arguments for reading a file. See isopy.read.file() for list of available options.
    """
    def __init__(self, dictionary = None):
        super(ReferenceDict, self).__init__()
        if dictionary is None: pass
        elif isinstance(dictionary, dict):
            self.update(dictionary)
        else:
            raise TypeError('Reference dict can only be initialised with another dict')

    def __getitem__(self, key):
        return self.get(key, np.nan)

    def __setitem__(self, key, value):
        key = IsopyString(key)
        value = np.asarray(value, dtype='f8')

        if value.size != 1:
            raise ValueError('ReferenceDict can only store a single value per entry')
        else:
            value = value.reshape(())

        super(ReferenceDict, self).__setitem__(key, value.copy())

    def __contains__(self, key):
        key = IsotopeString(key)

        if super(ReferenceDict, self).__contains__(key):
            return True
        elif isinstance(key, RatioString):
            return (super(ReferenceDict, self).__contains__(key.numerator) and
                    super(ReferenceDict, self).__contains__(key.denominator))
        else:
            return False

    def update(self, other):
        if not isinstance(other, dict):
            raise ValueError('other must be a dict')

        for k in other.keys():
            self.__setitem__(k, other[k])

    def pop(self, key, default=np.nan):
        key = IsopyString(key)
        return super(ReferenceDict, self).pop(key, default)

    def setdefault(self, key, default=np.nan):
        key = IsopyString(key)

        try: return self.__getitem__(key)
        except KeyError:
            self.__setitem__(key, default)
            return self.__getitem__(key)

    def get(self, key, default = np.nan):
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

        key = IsopyString(key)
        if isinstance(key, RatioString) and not super(ReferenceDict, self).__contains__(key) and\
                key.numerator in self and key.denominator in self:
            return super(ReferenceDict, self).__getitem__(key.numerator) / super(ReferenceDict, self).__getitem__(
                                                                                                        key.denominator)
        else:
            return super(ReferenceDict, self).get(key, default)

    def mass_keys(self):
       return MassList([k for k in self.keys() if isinstance(k, MassString)])

    def element_keys(self):
        return ElementList([k for k in self.keys() if isinstance(k, ElementString)])

    def isotope_keys(self):
        return IsotopeList([k for k in self.keys() if isinstance(k, IsotopeString)])

    def ratio_keys(self):
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

    def keycount(self):
        return len(self.dtype.names)

    def get(self, key, default_value=np.nan):
        try:
            return self[key]
        except:
            default_value = np.asarray(default_value)
            if default_value.ndim == 0 and self.ndim == 0: return default_value.copy()
            elif default_value.ndim == 0:
                out = np.empty(self.size)
                out[:] = default_value
                return out
            elif default_value.size != self.size:
                raise ValueError('default_value must be dimensionless or the same size as the array')
            else:
                return default_value.copy()


class MassArray(IsopyArray):
    def __new__(cls, values=None, *, keys=None, ndim=None, dtype=None):
        return MassNdarray(values, keys=keys, ndim=ndim, dtype=dtype)

    def copy(self, *, mass_number=None, mass_number_not=None, mass_number_lt=None, mass_number_gt=None,
               mass_number_le=None, mass_number_ge=None):
        """
        Return a new IsotopeArray containing only the keys of the array pass each of the supplied filters.


        Parameters
        ----------
        isotope : IsotopeString, IsotopeArray, optional
            Only keys matching this string/found in this list will pass.
        copy : bool
            If **True** then a copy of the array is returned. If **False** then a view on the array is returned.
            Default value is **True**
        isotope_not : IsotopeString, IsotopeList, optional
            Only keys not matching this string/not found in this list will pass.
        mass_number_and_element_symbol_filters
            See :func:`MassArray.copy()<isopy.dtypes.MassArray.filter>` and
            :func:`ElementArray.copy()<isopy.dtypes.ElementArray.filter>` for a list of available filters and their descriptions.


        Returns
        -------
        IsotopeArray
            New array containing only the isotope keys that match the specified filter parameters.


        See :func:`IsotopeList.copy()<isopy.dtypes.IsotopeList.filter>` for examples.
        """

        if mass_number is None and mass_number_not is None and mass_number_lt is None and mass_number_gt is None\
                and mass_number_le is None and mass_number_ge is None:
            return super(MassArray, self).copy()
        else:
            keys = self.keys().copy(mass_number=mass_number, mass_number_not=mass_number_not,
                                    mass_number_lt=mass_number_lt, mass_number_gt=mass_number_gt,
                                    mass_number_le=mass_number_le, mass_number_ge=mass_number_ge)
            return super(MassArray, self[keys]).copy()


class ElementArray(IsopyArray):
    def __new__(cls, values=None, *, keys=None, ndim=None, dtype=None):
        return ElementNdarray(values, keys=keys, ndim=ndim, dtype=dtype)

    def copy(self, *, element_symbol = None, element_symbol_not = None):
        if element_symbol is None and element_symbol_not is None:
            return super(ElementArray, self).copy()
        else:
            keys = self.keys().copy(element_symbol=element_symbol, element_symbol_not=element_symbol_not)
            return super(ElementArray, self[keys]).copy()


class IsotopeArray(IsopyArray):
    def __new__(cls, values=None, *, keys=None, ndim=None, dtype=None):
        return IsotopeNdarray(values, keys=keys, ndim=ndim, dtype=dtype)

    def copy(self, *, isotope=None, isotope_not=None, **mass_number_and_element_symbol_filters):
        """
        Return a new IsotopeArray containing only the keys of the array pass each of the supplied filters.


        Parameters
        ----------
        isotope : IsotopeString, IsotopeArray, optional
            Only keys matching this string/found in this list will pass.
        copy : bool
            If **True** then a copy of the array is returned. If **False** then a view on the array is returned.
            Default value is **True**
        isotope_not : IsotopeString, IsotopeList, optional
            Only keys not matching this string/not found in this list will pass.
        mass_number_and_element_symbol_filters
            See :func:`MassArray.copy()<isopy.dtypes.MassArray.filter>` and
            :func:`ElementArray.copy()<isopy.dtypes.ElementArray.filter>` for a list of available filters and their descriptions.


        Returns
        -------
        IsotopeArray
            New array containing only the isotope keys that match the specified filter parameters.


        See :func:`IsotopeList.copy()<isopy.dtypes.IsotopeList.filter>` for examples.
        """

        if isotope is None and isotope_not is None and len(mass_number_and_element_symbol_filters) == 0:
            return super(IsotopeArray, self).copy()
        else:
            keys = self.keys().copy(isotope=isotope, isotope_not=isotope_not,
                                      **mass_number_and_element_symbol_filters)
            return super(IsotopeArray, self[keys]).copy()


class RatioArray(IsopyArray):
    def __new__(cls, values=None, *, keys=None, ndim=None, dtype=None):
        return RatioNdarray(values, keys=keys, ndim=ndim, dtype=dtype)

    def copy(self, *, ratio = None, ratio_not = None, **numerator_and_denominator_filters):
        if ratio is None and ratio_not is None and len(numerator_and_denominator_filters) == 0:
            return super(RatioArray, self).copy()
        else:
            keys = self.keys().copy(ratio=ratio, ratio_not=ratio_not, **numerator_and_denominator_filters)
            return super(RatioArray, self[keys]).copy()

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
            return [(keys[i].safe_format(), _new_dtype(dtype[i])) for i in range(len(keys))]
    else:
        if keys is None:
            try:
                return np.dtype(dtype)
            except:
                raise ValueError('Could not convert "{}" to numpy dtype'.format(dtype))
        else:
            return [(keys[i].safe_format(), _new_dtype(dtype)) for i in range(len(keys))]

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
                obj[fkeys[i].safe_format()] = values[values.dtype.names[i]]
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
                        obj[fkeys[i].safe_format()] = values[values.dtype.names[i]][0]
                else:
                    for i in range(len(fkeys)):
                        obj[fkeys[i].safe_format()] = values[values.dtype.names[i]]

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
                        obj[fkeys[i].safe_format()] = values[i][0]
                else:
                    for i in range(len(fkeys)):
                        obj[fkeys[i].safe_format()] = values[i]

                return cls._ndarray(obj)

        raise ValueError('Unable to create array with value type {}'.format(type(values)))

    def __setitem__(self, key, value):
        if isinstance(key, str):
            try:
                key = self._string(key).safe_format()
            except:
                pass
        super(IsopyNdarray, self).__setitem__(key, value)

    def __getitem__(self, key):
        if isinstance(key, str):
            try:
                key = self._string(key).safe_format()
            except:
                pass
        elif isinstance(key, list):
            if len(key) == 0:
                return np.array([])
            elif isinstance(key[0], str):
                # Can be smarter
                try:
                    key = self._list(key).safe_format()
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
                key = self._string(key).safe_format()
            except:
                pass
        super(IsopyArray, self).__setitem__(key, value)

    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            raise IndexError('{} cannot be indexed by position'.format(self.__class__.__name__))
        elif isinstance(key, str):
            try:
                key = self._string(key).safe_format()
            except:
                pass
        elif isinstance(key, list):
            if len(key) == 0:
                return np.array([])
            elif isinstance(key[0], str):
                # Can be smarter
                try:
                    key = [k.safe_format() for k in self._list(key)]
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









