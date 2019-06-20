import numpy as np
import pyperclip
import csv as _csv

##############
### String ###
##############
class MassInteger(int):
    """
    MassInteger(integer)

    Same as a normal integer


    Parameters
    ----------
    integer : int, str
        Anything that can be converted to an integer


    Examples
    --------
    >>> MassInteger(105)
    105
    >>> MassInteger('105')
    105
    """
    __name__ = 'MassInteger'

class ElementString(str):
    """
    ElementString(string)
    
    A string where the first letter is in uppercase and the remaining letters are in lowercase.
       
    
    Parameters
    ----------
    string : str
        Can only contain alphabetical characters.
    
    
    Examples
    --------
    >>> ElementString('Pd')
    'Pd'
    >>> ElementString('pd')
    'Pd
    """
    __name__ = 'ElementString'

    def __new__(cls, string):
        if not isinstance(string, str): raise ValueError('string must be a str type')
        string = string.strip()
        if not string.isalpha(): raise ValueError('ElementString "{}" can only contain alphabetical characters'.format(string))
        if len(string) == 0: raise ValueError('ElementString empty')

        string = string.capitalize()

        obj = super(ElementString, cls).__new__(cls, string)
        return obj

class IsotopeString(str):
    """
    IsotopeString(string)

    A string consisting of the mass number followed by the element symbol of an isotope.

    
    Parameters
    ----------
    string : str
        Must contain a mass number and an element symbol
    

    Attributes
    ----------
    mass_number : MassInteger
        The mass number of the isotope
    element_symbol : ElementString
        The element symbol of the isotope


    Examples
    --------
    >>> IsotopeString('105Pd')
    '105Pd'
    >>> IsotopeString('pd105')
    '105Pd'
    """
    __name__ = 'IsotopeString'

    def __new__(cls, string):
        string = string.strip()
        mass, element = cls._parse(None, string)
        obj = super(IsotopeString, cls).__new__(cls, '{}{}'.format(mass, element))
        obj.mass_number = mass
        obj.element_symbol = element
        return obj

    def _parse(self, string):
        if not isinstance(string, str):
            try: string = str(string)
            except: raise ValueError('unable to parse type {} into an IsotopeString'.format(type(string)))

        if isinstance(string, IsotopeString):
            return string.mass_number, string.element

        elif isinstance(string, str):
            string = string.strip()

            # If no digits in string then only Symbol is given.
            if string.isalpha():
                raise ValueError('"{}" does not contain a nucleon number'.format(string))

            # If only digits then only mass number
                raise ValueError('"{}" does not contain an element'.format(string))

            # Loop through to split
            l = len(string)
            for i in range(1, l):
                a = string[:i]
                b = string[i:]

                # a = mass number and b = Symbol
                if a.isdigit() and b.isalpha():
                    return MassInteger(a), ElementString(b)

                # b = mass number and a = Symbol
                if a.isalpha() and b.isdigit():
                    return MassInteger(b), ElementString(a)

        # Unable to parse input
        raise ValueError('unable to parse "{}" into IsotopeString'.format(string))

    def __contains__(self, item):
        """
        Return **True** if `item` is equal to the mass number or the element symbol of the string. Otherwise returns **False**
        """
        if isinstance(item, str):
            if item.isdigit():
                try: return self.mass_number == MassInteger(item)
                except: return False
            elif item.isalpha():
                try: return self.element_symbol == ElementString(item)
                except: return False
        elif isinstance(item, int):
            return self.mass_number == item
        return False

class RatioString(str):
    """
        RatioString(string)

        A string consisting a numerator string and a denominator string separated by a "/".


        Parameters
        ----------
        string : str
            Two string seperated by a '/' than can each be parsed into either an `ElementString`_, `MassInteger`_ or a
            `RatioString`_.


        Attributes
        ----------
        numerator : ElementString, MassInteger, or IsotopeString
            The numerator string
        denominator : ElementString, MassInteger, or IsotopeString
            The denominator string


        Examples
        --------
        >>> RatioString('105Pd/108Pd')
        '105Pd/108Pd'
        >>> RatioString('pd105/pd')
        '105Pd/Pd'
        """
    __name__ = 'RatioString'

    def __new__(cls, string):
        string = string.strip()
        if isinstance(string, RatioString):
            numer = string.numerator
            denom = string.denominator
        elif isinstance(string, str):
            try:
                numer, denom = string.split("/", 1)
            except:
                raise ValueError('no "/" found in string')
        else:
            raise ValueError('unable to parse "{}"'.format(string))


        try: numer = MassInteger(numer)
        except ValueError:
            try: numer = ElementString(numer)
            except ValueError:
                try: numer = IsotopeString(numer)
                except ValueError: raise ValueError('Unable to parse numerator: "{}"'.format(numer))

        try: denom = ElementString(denom)
        except ValueError:
            try: denom = MassInteger(denom)
            except ValueError:
                try: denom = IsotopeString(denom)
                except ValueError: raise ValueError('Unable to parse denominator: "{}"'.format(denom))

        obj = super(RatioString, cls).__new__(cls, '{}/{}'.format(numer, denom))
        obj.numerator = numer
        obj.denominator = denom
        return obj

    def __contains__(self, item):
        """
        Return **True** if `item` is equal to the numerator or the denominator of the string. Otherwise returns **False**.
        """
        if isinstance(item, str):
            try: item = ElementString(item)
            except: pass
            try: item = MassInteger(item)
            except: pass
            try: item = IsotopeString(item)
            except: pass
        if isinstance(item, (ElementString, MassInteger, IsotopeString)):
            return self.numerator == item or self.denominator == item
        return False

def any_string(string):
    """
        Shortcut function that will return an `ElementString`_, `MassInteger`_, `IsotopeString`_ or a `RatioString`_
        depending on the supplied string.
        
        Parameters
        ----------
        string : str
            A string that can be parsed into either an ElementString, MassInteger, IsotopeString or a RatioString
    """
    try: return ElementString(string)
    except ValueError: pass
    try: return MassInteger(string)
    except ValueError: pass
    try: return IsotopeString(string)
    except ValueError: pass
    try: return RatioString(string)
    except ValueError:
        raise ValueError('Unable to parse item into ElementString, MassInteger, IsotopeList or RatioList')

############
### List ###
############
class _IsopyList(list):
    def __init__(self, items):
        super().__init__([])
        if items is None: items = []
        if isinstance(items, str): items = items.split(',')
        if isinstance(items, int): items = [items]
        elif isinstance(items, _IsopyArray): items = items.keys()
        for item in items: self.append(item)

    def __eq__(self, other):
        """
        Return **True** if all items in `other` is the same all items in the list. Order is not important.
        Otherwise return **False**.
        """
        if not isinstance(other, self.__class__):
            try: other = self.__class__(other)
            except:
                return False
        if len(other) != len(self): return False
        for key in self:
            if key not in other: return False
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

        if not isinstance(item, self._string_class):
            try: item = self._string_class(item)
            except: return False
        
        return super(_IsopyList, self).__contains__(item)

    def __getitem__(self, index):
        """
        Return the item at `index`. `index` can be int, slice or sequence of int.
        """
        if isinstance(index,slice):
            return self.__class__(super(_IsopyList, self).__getitem__(index))
        elif isinstance(index, list):
                return self.__class__([super(_IsopyList, self).__getitem__(i) for i in index])
        else:
            return super(_IsopyList, self).__getitem__(index)

    def __setitem__(self, key, value):
        """
        Not allowed. Raises NotImplementedError
        """
        raise NotImplementedError

    def append(self, item):
        """
        Append item to the end of the list
        """
        return super(_IsopyList, self).append(self._string_class(item))

    def index(self, item, *args):
        """
        index(x[, start[, end]])

        Return first index of value.

        Raises ValueError if the value is not present.
        """
        return super(_IsopyList, self).index(self._string_class(item), *args)

    def insert(self, index, item):
        """
        Insert item before index

        """
        return super(_IsopyList, self).insert(index, self._string_class(item))

    def remove(self, item):
        """
        Remove first occurrence of item.

        Raises ValueError if the item is not present.
        """
        return super(_IsopyList, self).remove(self._string_class(item))

    def copy(self):
        return self.__class__(super(_IsopyList, self).copy())

class MassList(_IsopyList):
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
    _string_class = MassInteger
    __name__ = 'MassList'

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

    def filter(self, mass_number=None, *, mass_number_not = None, mass_number_lt=None, mass_number_gt=None,
               mass_number_le=None, mass_number_ge=None, _return_index_list = False, _index_list = None, **_unused_kwargs):
        """
        filter(mass_number=None, *, mass_number_not = None, mass_number_lt=None, mass_number_gt=None, mass_number_le=None, mass_number_ge=None)

        Checks each MassInteger in the list against the supplied filters and return a list of the ones that pass.


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


        Examples
        --------
        >>> a = MassList([101, 104, 105, 111])
        >>> a.filter([100, 101, 104])
        [101, 104]
        >>> a.filter(mass_number_not = [100, 101, 104])
        [105, 111]
        >>> a.filter(mass_number_gt = 104)
        [105, 111]
        >>> a.filter(mass_number_le = 104)
        [101, 104]
        """

        if _index_list is None: _index_list = [i for i in range(len(self))]

        if mass_number is not None and not isinstance(mass_number, MassList):
            mass_number = MassList(mass_number)
        if mass_number_not is not None and not isinstance(mass_number_not, MassList):
            mass_number_not = MassList(mass_number_not)

        if mass_number_lt is not None and not isinstance(mass_number_lt, MassInteger):
            mass_number_lt = MassInteger(mass_number_lt)
        if mass_number_gt is not None and not isinstance(mass_number_gt, MassInteger):
            mass_number_gt = MassInteger(mass_number_gt)
        if mass_number_le is not None and not isinstance(mass_number_le, MassInteger):
            mass_number_le = MassInteger(mass_number_le)
        if mass_number_ge is not None and not isinstance(mass_number_ge, MassInteger):
            mass_number_ge = MassInteger(mass_number_ge)

        out = []
        for i in _index_list:
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

        if _return_index_list:
            return out
        else:
            return self[out]

class ElementList(_IsopyList):
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
    _string_class = ElementString
    __name__ = 'ElementList'

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

    def filter(self, element_symbol = None, *, element_symbol_not = None,
               _return_index_list = False, _index_list = None, **_unused_kwargs):
        """
        filter(element_symbol = None, *, element_symbol_not = None)

        Checks each ElementString in the list against the supplied filters and returns a list of the ones that pass.

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


        Examples
        --------
        >>> a = ElementList(['Ru', 'Pd', 'Cd'])
        >>> a.filter(['Zr', 'Ru', 'Cd'])
        ['Ru','Cd']
        >>> a.filter(element_symbol_not = ['Zr', 'Ru', 'Cd'])
        ['Pd']
        """

        if _index_list is None: _index_list = [i for i in range(len(self))]

        if element_symbol is not None and not isinstance(element_symbol, ElementList):
            element_symbol = ElementList(element_symbol)

        if element_symbol_not is not None and not isinstance(element_symbol_not, ElementList):
            element_symbol_not = ElementList(element_symbol_not)

        out = []
        for i in _index_list:
            ele = self[i]
            if element_symbol is not None:
                if ele not in element_symbol: continue
            if element_symbol_not is not None:
                if ele in element_symbol_not: continue

            out.append(i)

        if _return_index_list:
            return out
        else:
            return self[out]

class IsotopeList(_IsopyList):
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
    _string_class = IsotopeString
    __name__ = 'IsotopeList'

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
            if len(denominator) != len(self): raise ValueError(
                'Length of supplied list is not the same as subject list')
            return RatioList(['{}/{}'.format(self[i], denominator[i]) for i in range(len(self))])

        if isinstance(denominator, str):
            return RatioList(['{}/{}'.format(x, denominator) for x in self])

    @property
    def mass_numbers(self):
        return self.get_mass_numbers()

    @property
    def element_symbols(self):
        return self.get_element_symbols()

    def get_mass_numbers(self):
        """
        Return an `MassList`_ with the mass number of each `IsotopeString`_ in the list
        """
        return MassList([x.mass_number for x in self])

    def get_element_symbols(self):
        """
        Return an `ElementList`_ with the element symbol of each `IsotopeString`_ in the list
        """
        return ElementList([x.element_symbol for x in self])

    def filter(self, isotope = None, *, isotope_not = None, _return_index_list = False, _index_list = None,
               **mass_number_and_element_symbol_filters):
        """
        filter(isotope = None, *, isotope_not = None, **mass_number_and_element_symbol_filters)

        Checks each IsotopeString in the list against the supplied filters and returns a list of the ones that pass.

        Parameters
        ----------
        isotope : IsotopeString, IsotopeList
            Only IsotopeStrings matching this string/found in this list will pass.
        isotope_not : IsotopeString, IsotopeList
            Only IsotopeStrings not matching this string/not found in this list will pass.
        mass_number_and_element_symbol_filters
            See :func:`MassList.filter()<isopy.dtypes.MassList.filter>` and
            :func:`ElementList.filter()<isopy.dtypes.ElementList.filter>` for a list of available filters and their descriptions.


        Returns
        -------
        IsotopeList
            A new IsotopeList with each IsotopeString in the list that passed the filters.


        Examples
        --------
        >>> a = IsotopeList(['104Ru', '104Pd', '105Pd', '106Pd', '111Cd'])
        >>> a.filter(['102Ru', '104Ru', '110Cd', '111Cd'])
        ['104Ru','111Cd']
        >>> a.filter(isotope_not = ['102Ru', '104Ru', '110Cd', '111Cd'])
        ['104Pd', '105Pd', '106Pd']
        >>> a.filter(element_symbol = 'Pd', mass_number_gt = 104)
        ['105Pd', '106Pd']
        """
        if _index_list is None: _index_list = [i for i in range(len(self))]

        if isotope is not None and not isinstance(isotope, IsotopeList):
            isotope = IsotopeList(isotope)

        if isotope_not is not None and not isinstance(isotope_not, IsotopeList):
            isotope_not = IsotopeList(isotope_not)

        out = []
        for i in _index_list:
            iso = self[i]
            if isotope is not None:
                if iso not in isotope: continue
            if isotope_not is not None:
                if iso in isotope_not: continue

            # It has passed all the tests above
            out.append(i)

        out = self.get_element_symbols().filter(_return_index_list=True, _index_list=out,
                                                **mass_number_and_element_symbol_filters)
        out = self.get_mass_numbers().filter(_return_index_list=True, _index_list=out,
                                             **mass_number_and_element_symbol_filters)

        if _return_index_list:
            return out
        else:
            return self[out]

class RatioList(_IsopyList):
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

    _string_class = RatioString
    __name__ = 'RatioList'

    def __init__(self, items):
        self._numerator_string_class = None
        self._denominator_string_class = None
        super(RatioList, self).__init__(items)

    def _check_new_item_type(self, item):
        """
        Make sure than numerator and denominator types is the same for all item in list.
        """
        if self._numerator_string_class is None: self._numerator_string_class = item.numerator.__class__
        elif not isinstance(item.numerator, self._numerator_string_class):
            raise ValueError('ratio numerator set to "{}" not "{}"'.format(self._numerator_string_class.__name__,
                                                                           item.numerator.__class__.__name__))
        if self._denominator_string_class is None: self._denominator_string_class = item.denominator.__class__
        elif not isinstance(item.denominator, self._denominator_string_class):
            raise ValueError('ratio denominator set to "{}" not "{}"'.format(self._denominator_string_class.__name__,
                                                                             item.denominator.__class__.__name__))
        return item

    def append(self, item):
        item = self._check_new_item_type(self._string_class(item))
        super(RatioList, self).append(item)

    def insert(self, index, item):
        item = self._check_new_item_type(self._string_class(item))
        super(RatioList, self).insert(index, item)

    @property
    def numerators(self):
        return self.get_numerators()

    @property
    def denominators(self):
        return self.get_denominators()

    def get_numerators(self):
        """
        Return an list with the numerator of each `RatioString`_ in the list.
        """
        if self._numerator_string_class is ElementString:
            return ElementList([rat.numerator for rat in self])
        if self._numerator_string_class is IsotopeString:
            return IsotopeList([rat.numerator for rat in self])
        raise ValueError('numerator class unrecognized')

    def get_denominators(self):
        """
        Return an list with the denominator of each `RatioString`_ in the list.
        """
        if self._denominator_string_class is ElementString:
            return ElementList([rat.denominator for rat in self])
        if self._denominator_string_class is IsotopeString:
            return IsotopeList([rat.denominator for rat in self])
        raise ValueError('denominator class unrecognized')

    def has_common_denominator(self, denominator = None):
        """
        Return **True** if each RatioString in the list has the same denominator. Otherwise return **False**
        """
        denom = self.get_denominators()
        if len(denom) == 0:
            if denominator is not None: return False
            else: return True
        elif len(denom) > 0:
            if denom[0] != denominator and denominator is not None:
                return False
            for d in denom[1:]:
                if denom[0] != d:
                    return False
            return True

    def get_common_denominator(self):
        """
        Return the common denominator for each item in the list.

        Raise ValueError is no common denominator exists.
        """
        if not self.has_common_denominator(): raise ValueError('list does not have a common denominator')
        elif len(self.get_denominators()) == 0: raise ValueError('list is empty')
        else: return self.get_denominators()[0]

    def filter(self, ratio = None, *, ratio_not = None, _return_index_list = False, _index_list = None,
               **numerator_and_denominator_filters):
        """
        filter(ratio = None, *, ratio_not = None, numerator = None, denominator = None, **numerator_and_denominator_filters)

        Checks each RatioString in the list against the supplied filters and returns a list of the ones that pass.


        Parameters
        ----------
        ratio : RatioString, RatioList, optional
            Only RatioStrings matching this string/found in this list will pass.
        ratio_not : RatioString, RatioList, optional
            Only RatioStrings not matching this string/not found in this list will pass.
        numerator_and_denominator_filters
            Use *numerator_* and *denominator_* prefix to specify filters for the numerators and the denominators.
            See :func:`MassList.filter()<isopy.dtypes.MassList.filter>`,
            :func:`ElementList.filter()<isopy.dtypes.ElementList.filter>` and
            :func:`IsotopeList.filter()<isopy.dtypes.IsotopeList.filter>` for a list of available filters and their
            descriptions.


        Returns
        -------
        RatioList
            A new RatioList with each RatioString in the list that passed the filters.


        Examples
        --------
        >>> a = RatioList(['101Ru/Pd', '105Pd/Pd', '106Pd/Pd', '111Cd/Pd'])
        >>> a.filter(['102Pd/Pd', '105Pd/Pd', '106Pd/Pd'])
        ['105Pd/Pd', '106Pd/Pd']
        >>> a.filter(numerator_element_symbol = 'Pd')
        ['105Pd/Pd', '106Pd/Pd']
        >>> a.filter(numerator_mass_number_le = 105)
        ['101Ru/Pd', '105Pd/Pd']
        >>> a.filter(['102Pd/Pd', '105Pd/Pd', '106Pd/Pd'], numerator_mass_number_le = 105)
        '105Pd/Pd'
        """
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
                    raise ValueError('numerator filter not specified for value "{}"'.format(numerator_and_denominator_filters[k]))
            elif ks[0] == 'denominator' or ks[0] == 'd':
                try:
                    denominator_filters[ks[1]] = numerator_and_denominator_filters[k]
                except IndexError:
                    raise ValueError('denominator filter not specified for value "{}"'.format(numerator_and_denominator_filters[k]))
            else: raise ValueError('nd_filter prefix "{}" unknown'.format(ks[0]))

        if _index_list is None: _index_list = [i for i in range(len(self))]

        out = []
        for i in _index_list:
            rat = self[i]
            if ratio is not None:
                if rat not in ratio: continue
            if ratio_not is not None:
                if rat in ratio_not: continue
            out.append(i)

        if len(numerator_filters) > 0:
            out = self.get_numerators().filter(_index_list= out, _return_index_list= True, **numerator_filters)

        if len(denominator_filters) > 0:
            out = self.get_denominators().filter(_index_list= out, _return_index_list= True, **denominator_filters)

        if _return_index_list:
            return out
        else:
            return self[out]

def any_list(items):
    """
    Return a list matching the type of item supplied
    """
    try: return MassList(items)
    except ValueError: pass
    try: return ElementList(items)
    except ValueError: pass
    try: return IsotopeList(items)
    except ValueError: pass
    try: return RatioList(items)
    except ValueError:
        raise ValueError('Unable to parse items into MassList, ElementList, IsotopeList or RatioList')

############
### Dict ###
############
class IsopyDict(dict):
    """
    IsopyDict(values = None, *, keys = None, dtype = None, filepath = None, **file_kwargs)

    A custom dict where a keys are stored as isopy strings.

    All keys will, if they can, be converted into an MassInteger, ElementString, IsotopeString
    or RatioString when added to the dict. All values will be converted to a 'dtype' numpy arrays. Otherwise IsopyDict
    behaves like a normal python dict.


    Parameters
    ----------
    values : tuple, list, ndarray, optional
        Values must be compatible with the `dtype` given.
    keys : tuple, list, optional
        Keys for the dict. Keys that cannot be converted to an isopy type will be used as given.
    dtype : str, optional
        Accepts any numpy compatible data type. If **None** then data fill be stored as it is given. Defaults to
        64-bit float ('f8')
    filepath : str, optional
        Path of file on disk to read data from.
    file_kwargs
        Additional arguments for reading a file. See isopy.read.file() for list of available options.
    """
    def __init__(self, values = None, *, keys = None, dtype = 'f8', filepath = None, **file_kwargs):
        if isinstance(keys, str): keys = keys.split(',')
        if not isinstance(keys, list) and keys is not None:
            raise ValueError('"keys" must be a list')
        if dtype is not None and isinstance(dtype, str):
            raise ValueError('"dtype" must be a str')

        if filepath is not None:
            keys, values = _read_file(filepath)

        new = {}
        if isinstance(values, (dict, list, tuple, np.ndarray)):
            if isinstance(values, (dict, _IsopyArray)):
                vkeys = values.keys()
            else:
                values = np.asarray(values)
                vkeys = values.dtype.names
            if vkeys is not None:
                if keys is None: keys = vkeys
                for k in keys:
                    try:
                        newk = any_string(k)
                    except ValueError:
                        newk = k
                    if dtype is None: new[newk] = values[k]
                    try:
                        new[newk] = np.array(values[k], dtype)
                    except ValueError:
                        new[k] = np.nan
            else:
                if keys is None: raise ValueError('"keys" missing in "values" and input arguments')
                if len(values) != len(keys): raise ValueError('size of "keys" does not match size of "values"')
                for i in range(len(keys)):
                    try:
                        newk = any_string(keys[i])
                    except ValueError:
                        newk = keys[i]
                    if dtype is None: new[newk] = values[i]
                    try:
                        new[newk] = np.array(values[i], dtype)
                    except ValueError:
                        new[i] = np.nan

                for k in values.dtype.names:
                    try:
                        key = any_string(k)
                    except ValueError:
                        key = k
                    if dtype is None: new[key] = values[k]
                    try:
                        new[key] = np.array(values[k], dtype)
                    except ValueError:
                        new[k] = np.nan
        else:
            raise ValueError('"values" invalid type')

        super(IsopyDict, self).__init__(**new)

    def __getitem__(self, key):
        if isinstance(key, list):
            output = [self.__getitem__(k) for k in key]
            if isinstance(key, ElementList): return ElementArray(output, keys=key, size=-1)
            elif isinstance(key, IsotopeList): return IsotopeArray(output, keys=key, size=-1)
            elif isinstance(key, RatioList): return RatioArray(output, keys=key, size=-1)
            else: return output

        #TODO check if unformaed key exists
        try: key = any_string(key)
        except: pass

        try: return super(IsopyDict, self).__getitem__(key)
        except KeyError as err:
            if isinstance(key, RatioString):
                try: return super(IsopyDict, self).__getitem__(key.numerator) / super(IsopyDict, self).__getitem__(key.denominator)
                except: raise err
            else: raise err

    def __setitem__(self, key, value):
        try: key = any_string(key)
        except: pass

        return super(IsopyDict, self).__setitem__(key, value)

    def __contains__(self, key):
        try: key = any_string(key)
        except: pass

        return super(IsopyDict, self).__contains__(key)

    def pop(self, key, default=None):
        try: key = any_string(key)
        except: pass

        return super(IsopyDict, self).pop(key, default)

    def setdefault(self, key, default=None):
        try: key = any_string(key)
        except: pass

        return super(IsopyDict, self).setdefault(key, default)

    def get(self, key, default = np.nan):
        """
        Return key value.

        Parameters
        ----------
        key : str, list
            If a list then a list of values will be returned
        default
            Default value if key not in dict. Default is `numpy.nan`
        """
        if isinstance(key, list):
            out = []
            for k in key:
                try: out.append(self[k])
                except KeyError: out.append(default)
            return out
        try: return self.__getitem__(key)
        except KeyError: return default

    def keys(self, mass = False, element = False, isotope = False, ratio = False):
        """
        Return all keys in dict or only keys with a specific type.

        If no options are given then all keys are returned.

        Parameters
        ----------
        mass : bool
            If **True** then MassInteger keys are returned.
        element : bool
            If **True** then ElementString keys are returned.
        isotope : bool
            If **True** then IsotopeString keys are returned.
        ratio : bool
            If **True** then RatioString keys are returned.

        Returns
        -------
        list
            A list of keys in the dict
        """
        if not mass and not element and not isotope and not ratio and not other:
            return super(IsopyDict, self).keys()

        out = []
        for k in super(IsopyDict, self).keys():
            if mass and isinstance(k, MassInteger): out.append(k)
            elif element and isinstance(k, ElementString): out.append(k)
            elif isotope and isinstance(k, IsotopeString): out.append(k)
            elif ratio and isinstance(k, RatioString): out.append(k)

        return out


#############
### Array ###
#############
class _IsopyArray(np.ndarray):
    def __str__(self, delimiter = ', '):
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
                                     '\n'.join(delimiter.join('{:<{}}'.format(sdict[k][i], flen[k]) for k in self.keys())
                                                                      for i in range(self.size)))

    def __new__(cls, values = None, *, keys = None, size = None, ndim = None, filepath = None, dtype = 'f8'):
        if size is not None and not isinstance(size, int):
            raise ValueError('size must be an integer')

        if ndim is not None:
            if not isinstance(ndim, int):
                raise ValueError('ndim must be an integer')
            elif ndim != -1 and ndim != 0 and ndim != 1:
                raise ValueError('ndim must be -1, 0 or 1')

        if dtype is not None and not isinstance(dtype, str):
            raise ValueError('"dtype" must be a str')

        if keys is not None and not isinstance(keys, list) and not isinstance(keys, dict):
            raise ValueError('keys must be either a list or a dict')

        if filepath is not None:
            keys, values = _readfile(filepath)

        if isinstance(values, _IsopyArray):
            if keys is None and size is None and ndim is None and dtype is 'f8':
                if isinstance(values, cls):
                    return values.copy()
                else:
                    raise ValueError('{} can not be converted to a {}'.format(values.__class__, cls.__class__))
            if keys is not None:
                if isinstance(keys, list):
                    if len(keys) != len(values.keys()): raise ValueError('number of new keys does not match number of'
                                                                         ' old keys')
                    keys = {values.keys()[i]: keys[i] for i in range(len(keys))}

        if isinstance(values, (list, tuple)):
            if keys is None: raise ValueError('no keys given for list/tuple')
            if not isinstance(keys, list): raise ValueError(
                'keys must be a list when given together with a list of values')
            if len(keys) != len(values): raise ValueError(
                'number of keys does not match number of values given')
            values = {keys[i]: values[i] for i in range(len(keys))}
            keys = None

        if isinstance(values, np.ndarray):
            if values.dtype.names is not None:
                values = {values.dtype.names[i]: values[values.dtype.names[i]] for i in range(len(values.dtype.names))}
            else:
                if keys is None: raise ValueError('no keys given for ndarray')
                if not isinstance(keys, list): raise ValueError(
                    'keys must be a list when given together with a ndarray')
                if len(keys) != len(values): raise ValueError('number of keys does not match number of values given')
                values = {keys[i]: values[i] for i in range(len(keys))}
                keys = None

        if isinstance(values, dict):
            if keys is not None:
                if not isinstance(values, dict): raise ValueError('keys must be a dict to rename old keys')
                values = {keys.get(k, k): np.asarray(values[k], dtype=dtype) for k in values.keys()}
            else:
                values = {k: np.asarray(values[k], dtype=dtype) for k in values.keys()}

            keys = cls._list_class(values.keys())

            for k in keys:
                if size is None: size = values[k].size
                elif size != values[k].size: raise ValueError('size of keys in dict is not consistent or different from'
                                                              'the given size input')
                if ndim is None and values[k].ndim != 0: ndim = 1
                elif ndim == -1 and values[k].size != 1: ndim = 1
                elif ndim == 0 and values[k].size != 1: raise ValueError('size of key "{}" must be 1 for a 0-dim array'
                                                                      'not {}'.format(k, values[k].size))
                values[k] = values[k].flatten()

            if ndim == 1:
                obj = np.zeros(size, dtype=[(k, dtype) for k in keys])
            else:
                obj = np.zeros(None, dtype=[(k, dtype) for k in keys])

            for k in keys:
                if ndim == 1:
                    obj[k] = values[k]
                else:
                    obj[k] = values[k][0]

            return obj.view(cls)

        if values is not None:
            raise ValueError('Unable to parse values')

        if keys is not None and size is not None:
            fkeys = cls._list_class(keys)
            if ndim == 0 and size != 1: raise ValueError('size of a 0-ndim array can only be 1')
            elif ndim == -1 and size == 1: ndim = 0
            elif ndim is None: ndim = 1

            if ndim == 0:
                obj = np.zeros(None, dtype=[(k, dtype) for k in fkeys])
            else:
                obj = np.zeros(size, dtype=[(k, dtype) for k in fkeys])
            return obj.view(cls)

        raise ValueError('Unable to create {} with given options'.format(cls.__class__))

    def __getitem__(self, key):
        if not isinstance(key, self._string_class):
            try: key = self._string_class(key)
            except: pass

        arr = super(_IsopyArray, self).__getitem__(key)
        if arr.dtype.names is None: return arr.view(np.ndarray)
        else: return arr.view(self.__class__)
        
    def __setitem__(self, key, value):
        if not isinstance(key, self._string_class):
            try: key = self._string_class(key)
            except: pass
            
        super(_IsopyArray, self).__setitem__(key, value)

    def __array_finalize__(self, obj):
        if obj is None: return
        if obj.dtype.names is None: raise ValueError('Only structured arrays can be used with this view')
        try:
           self._list_class(obj.dtype.names)
        except:
            raise
        return obj

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        #Is called e.g. when multiplying arrays
        skip_missing_keys = kwargs.pop('skip_missing_keys', False)

        #TODO what if input is IsoRatDict
        #TODO key not in out
        #keys dont match


        #TODO dlen
        keys = None
        dlen = -1
        for i in inputs:
            if isinstance(i, _IsopyArray):
                if keys is None:
                    keys = i.keys()
                else:
                    if skip_missing_keys: keys = keys.filter(i.keys())
                    elif keys != i.keys(): raise ValueError('keys of {} dont match.'.format(self.__name__))
            if dlen is -1 and isinstance(i, (tuple,list, np.ndarray)):
                try: dlen = len(i)
                except: pass


        if len(keys) == 0: return np.array([])

        if 'out' in kwargs: out = kwargs.pop('out')
        else: out = self.__class__(size = dlen, keys=keys)

        #ufunc with only one input and operates on each item
        if ufunc.__name__ in ['log', 'log10', 'isnan']:
            for key in keys:
                kwargs['out'] = out[key]
                super(_IsopyArray, self).__array_ufunc__(ufunc, method, inputs[0][key], **kwargs)
            return out

        #ufuncs that uses two inputs
        if ufunc.__name__ in ['add', 'subtract', 'multiply', 'true_divide', 'floor_divide', 'power', 'sqrt', 'square']:
            for key in keys:
                try: kwargs['out'] = out[key]
                except: raise IndexError('key: "{}" not found in output array'.format(key))

                key_inputs = []
                for i in range(len(inputs)):
                    try: key_inputs.append(inputs[i][key])
                    except (ValueError, KeyError): raise ValueError('key "{}" not found in input {}'.format(key, i+1))
                    except: key_inputs.append(inputs[i])

                super(_IsopyArray, self).__array_ufunc__(ufunc, method, *key_inputs, **kwargs)
            return out

        raise NotImplementedError('ufunc "{}" is not supported for {}'.format(ufunc.__name__, self.__name__))

    def copy_to_clipboard(self, delimiter = ', '):
        """
        Uses the pyperclip package to copy the array to the clipboard.

        Parameters
        ----------
        delimiter: std, optional
            The delimiter for columns in the data. Default is ', '.
        """
        pyperclip.copy(self.__str__(delimiter))

    def save_to_file(self, filepath, first_row_comment = None):
        _savefile(self, filepath, first_row_comment)

    def _array_function_(self, func, **kwargs):
        # TODO dtype. Need to be fixed in creaton so list is accepted.
        axis = kwargs.pop('axis', 1)

        if axis == 1:
            out = kwargs.pop('out', None)
            if out is None:
                return self.__class__({key: func(self[key], **kwargs) for key in self.keys()})
            else:
                for key in self.keys():
                    #TODO catch key error here
                    func(self[key], out = out[key], **kwargs)
                    return out
        else:
            return func([self[x] for x in self.keys()], axis=axis, **kwargs)


        #never
        axis = kwargs.get('axis', None)

        if axis == 0 or axis is None:
            out = kwargs.get('out', None)
            if out is None:
                if kwargs.get('keepdims', np._NoValue) == True: ndim = 1
                else: ndim = 0
                out = self.__class__(size = 1, keys = self.keys(), ndim=ndim)

            for key in self.keys():
                out[key] = func(self[key], **kwargs)
            return out
        if axis == -1:
            return func([self[x] for x in self.keys()], **kwargs)

        if axis == 1:
            return func([self[x] for x in self.keys()], axis=0, **kwargs)

        raise ValueError('axis {} is out of bounds'.format(axis))

    def keys(self):
        return self._list_class(self.dtype.names)

class MassArray(_IsopyArray):
    """
    MassArray(values, * keys = None, size = None, filepath = None, dtype = 'f8')

    A custom numpy ndarray storing data with a set of MassInteger keys.

    Behaves much like a structured numpy array with a few useful exceptions described in the
    **Working with isopy arrays** section. Each array consists of a set of MassInteger keys and a number of data
    records. Each record contains a single data value for each key in the array.

    See **Creating isopy arrays** for a description on the possible ways to create isopy arrays.

    Parameters
    ----------
    values : tuple, list, ndarray, MassArray, optional
        Values must be compatible with `dtype` given.
    keys : MassList, optional
        Keys for the new array.
    size : int, optional
        Number of records in an empty array.
    filepath : str, optional
        Path of MassArray stored on disk.
    dtype : str, optional
        Accepts any numpy compatible data type. Defaults to 64-bit float ('f8')
    """
    _list_class = MassList
    _string_class = MassInteger
    __name__ = 'MassArray'

    def filter(self, mass_number, *, copy = True, mass_number_not = None, mass_number_lt=None, mass_number_gt=None,
               mass_number_le=None, mass_number_ge=None, return_index_list = False):
        """
        Return a new MassArray containing only the keys of the array pass each of the supplied filters.


        Parameters
        ----------
        mass_number : MassInteger, MassList optional
            Only keys matching this string/found in this list will pass.

        mass_number : MassInteger, MassList, optional
            Only MassIntegers matching this string/found in this list will pass.
        copy : bool
            If **True** then a copy of the array is returned. If **False** then a view on the array is returned.
            Default value is **True**
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
        MassArray
            New array containing only the isotope keys that match the specified filter parameters.


        See :func:`MassList.filter()<isopy.dtypes.MassList.filter>` for examples.
        """
        keys = self.keys().filter(mass_number, mass_number_not=mass_number_not, mass_number_lt=mass_number_lt,
                    mass_number_gt=mass_number_gt, mass_number_ge=mass_number_ge, mass_number_le=mass_number_le)

        if copy: return self[keys].copy()
        else: return self[keys]

    def convert_to_ratio_array(self, denominator, keep_as_numerator = False):
        """
        Return the current array divided by the specified `denominator' key.


        Parameters
        ----------
        denominator : MassInteger
            Must be present in the keys of the current array.
        keep_as_numerator : bool
            If **True** then the denominator will be present as a numerator in the new keys. If **False** then the
            denominator key will not be in included as a numerator in the new keys. Defaults to **False**.


        Returns
        -------
        RatioArray
            A copy of the current array divided by the `denominator` key and with updated keys.
        """
        denominator = self._string_class(denominator)
        keys = self.keys()
        if denominator not in keys(): raise ValueError('denominator "{}" not found in array keys'.format(denominator))

        if not keep_as_numerator: keys.remove(denominator)
        new_keys = keys / denominator

        if self.ndim == 0:
            output_array = RatioArray(keys=new_keys, size=-1)
        else:
            output_array = RatioArray(keys=new_keys, size=self.size)

        for nk in new_keys:
            output_array[nk] = self[nk.numerator] / self[nk.denominator]

        return output_array

class ElementArray(_IsopyArray):
    """
    ElementArray(values, * keys = None, size = None, filepath = None, dtype = 'f8')

    A custom numpy ndarray storing data with a set of ElementString keys.

    Behaves much like a structured numpy array with a few useful exceptions described in the
    **Working with isopy arrays** section. Each array consists of a set of IsotopeString keys and a number of data
    records. Each record contains a single data value for each key in the array.

    See **Creating isopy arrays** for a description on the possible ways to create isopy arrays.

    Parameters
    ----------
    values : tuple, list, ndarray, ElementArray, optional
        Values must be compatible with `dtype` given.
    keys : ElementList, optional
        Keys for the new array.
    size : int, optional
        Number of records in an empty array.
    filepath : str, optional
        Path of ElementArray stored on disk.
    dtype : str, optional
        Accepts any numpy compatible data type. Defaults to 64-bit float ('f8')
    """
    _list_class = ElementList
    _string_class = ElementString
    __name__ = 'ElementArray'

    def filter(self, element_symbol, *, copy = True, element_symbol_not = None):
        """
        Return a new ElementArray containing only the keys of the array pass each of the supplied filters.


        Parameters
        ----------
        element_symbol : ElementString, ElementList optional
            Only keys matching this string/found in this list will pass.
        copy : bool
            If **True** then a copy of the array is returned. If **False** then a view on the array is returned.
            Default value is **True**
        element_symbol_not : ElementString, ElementList, optional
            Only keys not matching this string/not found in this list will pass.

        Returns
        -------
        ElementArray
            New array containing only the isotope keys that match the specified filter parameters.


        See :func:`ElementList.filter()<isopy.dtypes.ElementList.filter>` for examples.
        """
        keys = self.keys().filter(element_symbol, element_symbol_not=element_symbol_not)
        if copy: return self[keys].copy()
        else: return self[keys]

    def convert_to_ratio_array(self, denominator, keep_as_numerator = False):
        """
        Return the current array divided by the specified `denominator' key.


        Parameters
        ----------
        denominator : ElementString
            Must be present in the keys of the current array.
        keep_as_numerator : bool
            If **True** then the denominator will be present as a numerator in the new keys. If **False** then the
            denominator key will not be in included as a numerator in the new keys. Defaults to **False**.


        Returns
        -------
        RatioArray
            A copy of the current array divided by the `denominator` key and with updated keys.
        """
        denominator = self._string_class(denominator)
        keys = self.keys()
        if denominator not in keys(): raise ValueError('denominator "{}" not found in array keys'.format(denominator))

        if not keep_as_numerator: keys.remove(denominator)
        new_keys = keys / denominator

        if self.ndim == 0:
            output_array = RatioArray(keys=new_keys, size=-1)
        else:
            output_array = RatioArray(keys=new_keys, size=self.size)

        for nk in new_keys:
            output_array[nk] = self[nk.numerator] / self[nk.denominator]

        return output_array

class IsotopeArray(_IsopyArray):
    """
    IsotopeArray(values, * keys = None, size = None, filepath = None, dtype = 'f8')

    A custom numpy ndarray storing data with a set of IsotopeString keys.

    Behaves much like a structured numpy array with a few useful exceptions described in the
    **Working with isopy arrays** section. Each array consists of a set of IsotopeString keys and a number of data
    records. Each record contains a single data value for each key in the array.

    See **Creating isopy arrays** for a description on the possible ways to create isopy arrays.

    Parameters
    ----------
    values : tuple, list, ndarray, IsotopeArray, optional
        Values must be compatible with the `dtype` given.
    keys : IsotopeList, optional
        Keys for the new array.
    size : int, optional
        Number of records in an empty array.
    filepath : str, optional
        Path of IsotopeArray stored on disk.
    dtype : str, optional
        Accepts any numpy compatible data type. Defaults to 64-bit float ('f8')
    """
    _list_class = IsotopeList
    _string_class = IsotopeString
    __name__ = 'IsotopeArray'

    def filter(self, isotope = None, *, copy = True, isotope_not = None, **mass_number_and_element_symbol_filters):
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
            See :func:`MassArray.filter()<isopy.dtypes.MassArray.filter>` and
            :func:`ElementArray.filter()<isopy.dtypes.ElementArray.filter>` for a list of available filters and their descriptions.


        Returns
        -------
        IsotopeArray
            New array containing only the isotope keys that match the specified filter parameters.


        See :func:`IsotopeList.filter()<isopy.dtypes.IsotopeList.filter>` for examples.
        """

        keys = self.keys().filter(isotope, isotope_not, **mass_number_and_element_symbol_filters)

        if copy: return self[keys].copy()
        else: return self[keys]

    def convert_to_ratio_array(self, denominator, keep_as_numerator = False):
        """
        Return the current array divided by the specified `denominator' key.

        Parameters
        ----------
        denominator : IsotopeString
            Must be present in the keys of the current array.
        keep_as_numerator : bool
            If **True** then the denominator will be present as a numerator in the new keys. If **False** then the
            denominator key will not be in included as a numerator in the new keys. Defaults to **False**.


        Returns
        -------
        RatioArray
            A copy of the current array divided by the `denominator` key and with updated keys.
        """
        denominator = self._string_class(denominator)
        keys = self.keys()
        if denominator not in keys(): raise ValueError('denominator "{}" not found in array keys'.format(denominator))

        if not keep_as_numerator: keys.remove(denominator)
        new_keys = keys/denominator

        if self.ndim == 0:
            output_array = RatioArray(keys=new_keys, size=-1)
        else:
            output_array = RatioArray(keys=new_keys, size=self.size)

        for nk in new_keys:
            output_array[nk] = self[nk.numerator]/self[nk.denominator]

        return output_array

class RatioArray(_IsopyArray):
    """
        RatioArray(values, * keys = None, size = None, filepath = None, dtype = 'f8')

        A custom numpy ndarray storing data with a set of RatioString keys.

        Behaves much like a structured numpy array with a few useful exceptions described in the
        **Working with isopy arrays** section. Each array consists of a set of RatioString keys and a number of data
        records. Each record contains a single data value for each key in the array.

        See **Creating isopy arrays** for a description on the possible ways to create isopy arrays.

        Parameters
        ----------
        values : tuple, list, ndarray, RatioArray, optional
            Values must be compatible with `dtype` given.
        keys : RatioList, optional
            Keys for the new array.
        size : int, optional
            Number of records in an empty array.
        filepath : str, optional
            Path of RatioArray stored on disk.
        dtype : str, optional
            Accepts any numpy compatible data type. Defaults to 64-bit float ('f8')
        """
    _list_class = RatioList
    _string_class = RatioString
    __name__ = 'RatioArray'

    def filter(self, ratio = None, *, copy = True, ratio_not = None, **numerator_and_denominator_filters):
        """
        Return a new RatioArray containing only the keys of the array pass each of the supplied filters.


        Parameters
        ----------
        isotope : IsotopeString, IsotopeArray, optional
            Only keys matching this string/found in this list will pass.
        copy : bool
            If **True** then a copy of the array is returned. If **False** then a view on the array is returned.
            Default value is **True**
        isotope_not : IsotopeString, IsotopeList, optional
            Only keys not matching this string/not found in this list will pass.
        numerator_and_denominator_filters
            Use *numerator_* and *denominator_* prefix to specify filters for the numerators and the denominators.
            See :func:`MassArray.filter()<isopy.dtypes.MassArray.filter>`,
            :func:`ElementArray.filter()<isopy.dtypes.ElementArray.filter>` and
            :func:`IsotopeArray.filter()<isopy.dtypes.IsotopeArray.filter>` for a list of available filters and their
            descriptions.

        Returns
        -------
        RatioArray
            New array containing only the isotope keys that match the specified filter parameters.


        See :func:`RatioList.filter()<isopy.dtypes.RatioList.filter>` for examples.
        """
        keys = self.keys().filter(ratio, ratio_not=ratio_not, **numerator_and_denominator_filters)
        return self[keys]

    def convert_to_numerator_array(self, denominator_value = 1, denominator_as_key = True):
        """
        Return the current array multiplied by the denominator value.

        Raises **ValueError** if array does not have a common denominator or if the numerators and denominators do not
        have the same data type.

        Parameters
        ----------
        denominator_value: int, float, optional
            The assumed value of the denominator. Defaults to 1.0
        denominator_as_key : bool
            If **True** then the denominator will be present in the keys of the new array. If **False** then the
            denominator will not be in included in the keys of the new array. Defaults to **True**.


        Returns
        -------
        RatioArray
            A copy of hte current array multiplied by the `denominator_value` and with updated keys.
        """

        keys = self.keys()
        if not keys.has_common_denominator(): raise ValueError('array must have a common denominator')
        numerators = keys.get_numerators()
        denominator = keys.get_common_denominator()
        if type(numerators) != type(denominator): raise ValueError('numerators and denominators must have the same datatype')

        if denominator_as_key:
            if denominator in numerators: pass
            else: numerators.append(denominator)
        else:
            if denominator in numerators:
                numerators.remove(denominator)

        if self.ndim == 0: output_array = any_array(size = -1, keys = numerators)
        else: output_array = numerators._array_type(size = self.size, keys = numerators)

        for k in self.keys():
            if k.numerator in numerators: output_array[k.numerator] = self[k]

        if denominator_as_key: output_array[denominator] = 1
        output_array = output_array * denominator_value
        return output_array

def any_array(values = None, *, keys = None, size = None, filepath = None, dtype = 'f8'):
    if isinstance(values, _IsopyArray): return values.__class__(values, keys=keys, size=size,filepath=filepath, dtype=dtype)
    if isinstance(keys, MassList): return MassArray(values, keys=keys, size=size,filepath=filepath, dtype=dtype)
    if isinstance(keys, ElementList): return ElementArray(values, keys=keys, size=size, filepath=filepath, dtype=dtype)
    if isinstance(keys, IsotopeList): return IsotopeArray(values, keys=keys, size=size, filepath=filepath, dtype=dtype)
    if isinstance(keys, RatioList): return RatioArray(values, keys=keys, size=size, filepath=filepath, dtype=dtype)
    raise ValueError('Unable to create array')

def _readfile(filepath, comments ='#', delimiter = ','):
    keys = []
    values = []
    key_len = None

    #First row keys
    #data after that

    #TODO propblems opening file etc

    with open(filepath, 'r', newline='') as file:
        reader = _csv.reader(file, delimiter=delimiter)

        for row in reader:
            if comments is not None:
                try:
                    if row[0].strip()[0] == comments: continue
                except:
                    pass

            if key_len is None:
                keys = [x.strip() for x in row]
                key_len = len(keys)
                values = [list([]) for x in keys]
            else:
                if len(row) < key_len: raise ValueError('Row does not have a value for every key')
                for i in range(key_len):
                    values[i].append(row[i].strip())

    return keys, values

def _savefile(array, filepath, first_row_comment = None, delimiter = ','):
    with open(filepath, mode='w', newline='') as file:
        writer = _csv.writer(file, delimiter=',')

        if first_row_comment is not None:
            writer.writerow(['#{}'.format(first_row_comment)])

        writer.writerow([k for k in array.keys()])

        if array.ndim == 0:
            writer.writerow(['{}'.format(array[k]) for k in array.keys()])
        else:
            keys = array.keys()
            for i in range(array.size):
                writer.writerow(['{}'.format(array[k][i]) for k in keys])