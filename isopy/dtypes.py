import numpy as np

DTYPE = 'f8'

##############
### String ###
##############
class ElementString(str):
    """
    ElementString(string, reformat = True)
    
    A string representation of an element.
     
    The first letter of an ``ElementString``  is in uppercase and the remaining letters are in lowercase.
       
    
    Parameters
    ----------
    string : str
        Can only contain alphabetical characters.
    reformat : bool, optional
        If ``reformat == True`` then string will be parsed to correct format. If ``reformat == False`` then a ValueError will be
        thrown if string is not correctly formated.
    
    
    Examples
    --------
    Default initalisation, ``reformat == True``
    
    >>> ElementString('Pd')
    'Pd'
    >>> ElementString('pd')
    'Pd
    >>> ElementString('105Pd')
    Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
    ValueError: ElementString "105Pd" can only contain alphabetical characters
    
    Initialisation when ``reformat == False``
    
    >>> ElementString('Pd', False)
    'Pd'
    >>> ElementString('pd', False)
    Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
    ValueError: ElementString "pd" incorrectly formatted: First letter must be uppercase

    """
    __name__ = 'ElementString'

    def __new__(cls, string, reformat = True):
        if not isinstance(string, str): raise ValueError('string must be a str type')
        string = string.strip()
        if not string.isalpha(): raise ValueError('ElementString "{}" can only contain alphabetical characters'.format(string))
        if len(string) == 0: raise ValueError('ElementString empty')
        if not reformat:
            if not string[0].isupper(): raise ValueError('ElementString "{}" incorrectly formatted: First character must be'
                                                   ' uppercase'.format(string))
            if len(string) > 1:
                if not string[1:].islower(): raise ValueError('ElementString "{}" incorrectly formatted: All but'
                                                                    ' first character must be lowercase'.format(string))

        string = string.capitalize()

        obj = super(ElementString, cls).__new__(cls, string)
        return obj

class IsotopeString(str):
    """
    IsotopeString(string, reformat = True)
    
    A string representation of an isotope.
    
    A string consisting of the mass number followed by the element symbol of an isotope.

    
    Parameters
    ----------
    string : str
        Must contain nucleon number and element string
    reformat : bool, optional
        If ``reformat == True`` then string will be parsed to correct format. If ``reformat == False`` then a ValueError will be
        thrown if `string` is not correctly formated.
    
    
    Attributes
    ----------
    mass_number : int
        The mass number of the isotope
    element_symbol : ElementString
        The element symbol of the isotope


    Examples
    --------
    Default initalisation, ``reformat == True``
    
    >>> IsotopeString('105Pd')
    '105Pd'
    >>> IsotopeString('pd105')
    '105Pd'
    >>> IsotopeString('Pd')
    Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
    ValueError: "Pd" does not contain a nucleon number
    
    Initialisation when ``reformat == False``
    
    >>> IsotopeString('105Pd', False)
    '105Pd'
    >>> IsotopeString('Pd105', False)
    Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
    ValueError: In string "Pd105" element appears before nucleon number (A)
    >>> IsotopeString('105pd', False)
    Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
    ValueError: "ElementString "pd" incorrectly formatted: First character must be uppercase"

    Accessing attributes

    >>> IsotopeString('105Pd').A
    105
    >>> IsotopeString('105Pd').element
    'Pd'
    
    Contains
    
    >>> 'Pd' in IsotopeString('105Pd')
    True
    >>> 'pd' in IsotopeString('105Pd')
    True
    >>> 'Ru' in IsotopeString('105Pd')
    False
    >>> 105 in IsotopeString('105Pd')
    True
    >>> 106 in IsotopeString('105Pd')
    False
    
    """
    __name__ = 'IsotopeString'

    def __new__(cls, string, reformat = True):
        string = string.strip()
        mass, element = cls._parse(None, string, reformat)
        obj = super(IsotopeString, cls).__new__(cls, '{}{}'.format(mass, element))
        obj.mass_number = mass
        obj.element_symbol = element
        return obj

    def _parse(self, string, reformat=True):
        if not isinstance(string, str):
            try: string = str(string)
            except: raise ValueError('unable to parse type {}'.format(type(string)))

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

                # a = nucleon number and b = Symbol
                if a.isdigit() and b.isalpha():
                    return int(a), ElementString(b, reformat)

                # b = nucleon number and a = Symbol
                if a.isalpha() and b.isdigit():
                    if not reformat:
                        raise ValueError('In string "{}" element appears before nucleon number (A)'.format(string))
                    else:
                        return int(b), ElementString(a, reformat)

        # Unable to parse input
        raise ValueError('unable to parse "{}" into IsotopeString'.format(string))

    def __contains__(self, item):
        if isinstance(item, str):
            if item.isdigit():
                return self.mass_number == int(item)
            elif item.isalpha():
                try: return self.element_symbol == ElementString(item)
                except: return False
        elif isinstance(item, int):
            return self.mass_number == item
        return False

class RatioString(str):
    """
        RatioString(string, reformat = True)

        A string representation of a ratio.

        A string consisting the numerator ElementString/IsotopeString followed by the denominator ElementString/IsotopeString seperated by a "/".

        Parameters
        ----------
        string : str
            Must contain two ElementString/IsotopeString seperated by a "/"
        reformat : bool, optional
            If ``reformat == True`` then string will be parsed to correct format. If ``reformat == False`` then a ValueError will be
            thrown if string is not already correctly formatted.


        Attributes
        ----------
        numerator : ElementString or IsotopeString
            The numerator string
        denominator : ElementString or IsotopeString
            The denominator string


        Examples
        --------
        Default initalisation, ``reformat == True``

        >>> RatioString('105Pd/108Pd')
        '105Pd/108Pd'
        >>> RatioString('pd105/pd')
        '105Pd/Pd'
        >>> RatioString('105Pd')  
        Traceback (most recent call last):
            File "<stdin>", line 1, in <module>
        ValueError: no "/" found in string

        Initialisation when ``reformat == False``

        >>> RatioString('105Pd/108Pd', True)
        '105Pd/108Pd'
        >>> RatioString('pd105/108pd', True)
        Traceback (most recent call last):
            File "<stdin>", line 1, in <module>
        ValueError: Unable to parse numerator: "In string "pd105" element appears before nucleon number (A)"
        >>> RatioString('105Pd/108pd', True)
        Traceback (most recent call last):
            File "<stdin>", line 1, in <module>
        ValueError: Unable to parse denominator: "ElementString "pd" incorrectly formatted: First character must be uppercase"

        Accessing attributes

        >>> RatioString('105Pd/108Pd').numerator
        '105Pd'
        >>> RatioString('105Pd/108Pd').denominator
        '108Pd'
        
        Contains
        
        >>> '105Pd' in RatioString('105Pd/108Pd')
        True
        >>> 'pd108' in RatioString('105Pd/108Pd')
        True
        >>> '105Ru' in RatioString('105Pd/108Pd')
        False
        
        """
    __name__ = 'RatioString'

    def __new__(cls, string, reformat = True):
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

        try: numer = ElementString(numer, reformat)
        except ValueError:
            try: numer = IsotopeString(numer, reformat)
            except ValueError as err: raise ValueError('Unable to parse numerator: "{}"'.format(err))

        try: denom = ElementString(denom, reformat)
        except ValueError:
            try: denom = IsotopeString(denom, reformat)
            except ValueError as err: raise ValueError('Unable to parse denominator: "{}"'.format(err))

        obj = super(RatioString, cls).__new__(cls, '{}/{}'.format(numer, denom))
        obj.numerator = numer
        obj.denominator = denom
        return obj

    def __contains__(self, item, reformat = True):
        if isinstance(item, str):
            try: item = ElementString(item, reformat)
            except: pass
            try: item = IsotopeString(item, reformat)
            except: pass
        if isinstance(item, (IsotopeString, ElementString)):
            return self.numerator == item or self.denominator == item
        return False

def any_string(string, reformat = True):
    """
        Shortcut function that will return an IsotopeString or a RatioString depending on the supplied string.
        
        Parameters
        ----------
        string : str
            A string that can be parsed into either an IsotopeSting or a RatioSting
        reformat : bool, optional
            If ``reformat == True`` then string will be parsed to correct format. If ``reformat == False`` then a ValueError will be
            thrown if string is not already in the correct format.
    
        Returns
        -------
        IsotopeString or RatioString
        
        Examples
        --------
        >>> type(any_string('105Pd'))
        IsotopeString
        >>> Type(any_string('105Pd/106Pd'))
        RatioString

        
    """
    try: return ElementString(string, reformat)
    except ValueError: pass
    try: return IsotopeString(string, reformat)
    except ValueError: pass
    try: return RatioString(string, reformat)
    except ValueError: raise #ValueError('Unable to parse items into ElementString, IsotopeList or RatioList')

############
### List ###
############
#TODO disallow duplicates
class _IsopyList(list):
    def __init__(self, items, reformat = True):
        super().__init__([])
        if items is None: items = []
        if isinstance(items, str): items = items.split(',')
        elif isinstance(items, _IsopyArray): items = items.keys()
        for item in items: self.append(item, reformat)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            try: other = self.__class__(other)
            except:
                return False
        if len(other) != len(self): return False
        for key in self:
            if key not in other: return False
        return True
    
    def __contains__(self, item):
        if isinstance(item, list):
            for i in item:
                if not self.__contains__(i): return False
            return True

        if not isinstance(item, self._string_class):
            try: item = self._string_class(item)
            except: return False
        
        return super(_IsopyList, self).__contains__(item)

    def __getitem__(self, index):
        if isinstance(index,slice):
            return self.__class__(super(_IsopyList, self).__getitem__(index))
        elif isinstance(index, list):
                return self.__class__([super(_IsopyList, self).__getitem__(i) for i in index])
        else:
            return super(_IsopyList, self).__getitem__(index)

    def __setitem__(self, key, value):
        raise NotImplementedError

    def _new_item(self, item, reformat = True):
        return self._string_class(item, reformat)

    def append(self, item, reformat = True):
        item = self._new_item(item, reformat)
        return super().append(item)

    def insert(self, index, item, reformat = True):
        item = self._new_item(item, reformat)
        return super().insert(index, item)

    def remove(self, item, reformat = True):
        return super().remove(self._new_item(item, reformat))

    def copy(self):
        return self.__class__([self._new_item(x) for x in self])

class ElementList(_IsopyList):
    _string_class = ElementString
    __name__ = 'ElementList'

    def __div__(self, other):
        if isinstance(other, list):
            if len(other) != len(self): raise ValueError('Length of supplied list is not the same as subject list')
            return RatioList(['{}/{}'.format(self[i], other[i]) for i in range(len(self))])

        if isinstance(other, str):
            return RatioList(['{}/{}'.format(x, other) for x in self])

    def __truediv__(self, other):
        return self.__div__(other)

    def filter(self, element_symbol = None, *, element_symbol_not = None,
               return_index_list = False, index_list = None):

        if index_list is None: index_list = [i for i in range(len(self))]

        if element_symbol is None or isinstance(element_symbol, ElementList):
            pass
        else:
            element_symbol = ElementList(element_symbol)

        if element_symbol_not is None or isinstance(element_symbol_not, ElementList):
            pass
        else:
            element_symbol_not = ElementList(element_symbol_not)

        out = []
        for i in index_list:
            ele = self[i]
            if element_symbol is not None:
                if ele not in element_symbol: continue
            if element_symbol_not is not None:
                if ele in element_symbol_not: continue

            out.append(i)

        if return_index_list:
            return out
        else:
            return self[out]

class IsotopeList(_IsopyList):
    """
        IsotopeList(items, reformat = True)

        A list for storing IsotopeString items.

        Parameters
        ----------
        items : IsotopeString, IsotopeList, IsotopeArray
            A string will be converted to a list of one item. A copy of a given list will be used for the newly
            created list. The key of a IsotopeArray will be used to create the list.
        
        reformat : bool, optional
            If ``reformat == True`` then strings will be parsed to correct format. If ``reformat == False`` then all items
            must already be correctly formatted or an exception will be thrown.


        
        Examples
        --------
        Default initialisation, ``reformat == True``

        >>> IsotopeList(['104Pd', '105Pd', '106Pd'])
        ['104Pd', '105Pd', '106Pd']
        >>> IsotopeList(['104pd', 'pd105', 'Pd106'])
        ['104Pd', '105Pd', '106Pd']

        Initialisation when ``reformat == False``

        >>> IsotopeList(['104Pd', '105Pd', '106Pd'], True)
        ['104Pd', '105Pd', '106Pd']
        >>> IsotopeList(['104pd', 'pd105', 'Pd106'], True)
        Traceback (most recent call last):
            File "<stdin>", line 1, in <module>
        ValueError: First letter of ElementString must be upper case while the remaning letters are in lower case
        
        Methods
        -------
        __contains__(item)
            Returns ``True`` if list contains `item`. Otherwise returns ``False``.
            
            Parameters
            ----------
            item : IsotopeString, IsotopeList
                If a list then all items must be present in the curent list
                 
            Returns
            -------
            bool
            
            Examples
            --------
            >>> 'pd105' in IsotopeList(['104Pd', '105Pd', '106Pd'])
            True
            >>> '108Pd' in IsotopeList(['104Pd', '105Pd', '106Pd'])
            False
            >>> ['pd105', '106pd'] in IsotopeList(['104Pd', '105Pd', '106Pd'])
            True
            >>> ['pd105', '108pd'] in IsotopeList(['104Pd', '105Pd', '106Pd'])
            False
            
        __eq__(other)
            Return ``True`` if list equals `other`. Otherwise returns ``False``. Order of items in either list is not
            considerd.
            
            Parameters
            ----------
            other : IsotopeList
            
            Return
            ------
            bool
        
            Examples
            --------
            >>> ['104pd', '105pd', '106pd'] == IsotopeList(['104Pd', '105Pd', '106Pd'])
            True
            >>> ['105pd', '104pd', '106pd'] == IsotopeList(['104Pd', '105Pd', '106Pd'])
            True

        __div__(denominator)
            Used to create a RatioList from the current list.

            Parameters
            ----------
            denominator : ElementString, IsotopeString, ElementList, IsotopeList
                Lists must be same length as current list.

            Returns
            -------
            RatioList


            Examples
            --------
            >>>  IsotopeList(['104Pd', '105Pd', '106Pd']) / '108pd'
            ['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd']

            >>>  IsotopeList(['104Pd', '105Pd', '106Pd']) / ['Ru', 'Pd', 'Cd']
            ['104Pd/Ru', '105Pd/Pd', '106Pd/Cd']

        
        append(item, reformat = True)
            Add item to the end of the list.
            
            Parameters
            ----------
            item : IsotopeString
                String to be appended to list
            reformat : bool, optional
                If ``reformat == True`` then strings will be parsed to correct format. If ``reformat == False`` then all items
                must already be correctly formatted or an exception will be thrown.
    
            Examples
            --------      
            >>> a = IsotopeList(['104Pd', '105Pd', '106Pd'])
            >>> a.append('108pd')
            >>> a
            ['104Pd', '105Pd', '106Pd', '108Pd']

        
        insert(index, item, stict = False)
            Insert item at list index.
        
            Parameters
            ----------
            index : int
                Position where item will be added
            item : IsotopeString
                String to be inserted into the current list
            reformat : bool, optional
                If ``reformat == True`` then strings will be parsed to correct format. If ``reformat == False`` then all items
                must already be correctly formatted or an exception will be thrown.
    
            Examples
            --------
            >>> a = IsotopeList(['104Pd', '105Pd', '106Pd'])
            >>> a.insert(0,'102pd')
            >>> a
            ['102Pd', '104Pd', '105Pd', '106Pd']

        
        remove(item, reformat = True)
            Remove item from list
        
            Parameters
            ----------
            item : IsotopeString
                The first occurrence of this string will be removed from the list
            reformat : bool, optional
                If ``reformat == True`` then strings will be parsed to correct format. If ``reformat == False`` then all items
                must already be correctly formatted or an exception will be thrown.
    
            Examples
            --------
            >>> a = IsotopeList(['104Pd', '105Pd', '106Pd'])
            >>> a.remove(0,'105pd')
            >>> a
            ['104Pd', '106Pd']
        
        copy()
            Returns a copy of the current list. Equalivent to IsotopeList[:]
        
            Returns
            -------
            IsotopeList
            
            Examples
            --------
            >>> a = IsotopeList(['104Pd', '105Pd', '106Pd'])
            >>> b = a.copy()
            >>> b
            ['104Pd', '105Pd', '106Pd']
            >>> b == a
            True
            >>> b == a
            False
        
        """
    _string_class = IsotopeString
    __name__ = 'IsotopeList'

    def __div__(self, denominator):
        if isinstance(denominator, list):
            if len(denominator) != len(self): raise ValueError('Length of supplied list is not the same as subject list')
            return RatioList(['{}/{}'.format(self[i], denominator[i]) for i in range(len(self))])

        if isinstance(denominator, str):
            return RatioList(['{}/{}'.format(x, denominator) for x in self])

    def __truediv__(self, other):
        return self.__div__(other)

    @property
    def mass_numbers(self):
        return self.get_mass_numbers()

    @property
    def element_symbols(self):
        return self.get_element_symbols()

    def get_mass_numbers(self):
        return [x.mass_number for x in self]

    def get_element_symbols(self):
        """
        Return a list of the element property for all IsotopeStrings in the current list

        Returns
        -------
        ElementList

        Examples
        --------
        >>> a = IsotopeList(['101Ru', '105Pd', '111Cd'])
        >>> a.get_elements()
        ['Ru', 'Pd', 'Cd]
        """
        return ElementList([x.element_symbol for x in self])

    def filter(self, isotope = None, *, isotope_not = None, element_symbol=None, element_symbol_not = None,
               mass_number=None, mass_number_not = None, mass_number_lt=None, mass_number_gt=None,
               return_index_list = False, index_list = None):
        """
        Return a list of the index of all items in list that match the given filter restrictions.

        Parameters
        ----------
        isotope : IsotopeList
            Isotope must be present in ´isotope´
        isotope : str
            pass
        isotope : list
            pass
        element : str, optional
            Isotope must have this element symbol
        mass_number : int, optional
            Isotope must have this nucleon number (A)
        mass_number_not:
            pass
        mass_number_lt : int, optional
            Isotope must have a nucleon number (A) less than this value
        mass_number_gt : int, optional
            Isotope must have a nucleon number (A) greater than this value
        index : list
            List of integers. Only check these indexes. If not given then check all items in list.

        Returns
        -------
        list
            List of indexes of all items matches the filter.
        
        Examples
        --------
        >>> a = IsotopeList(['104Ru', '104Pd', '105Pd', '106Pd', '111Cd'])
        >>> a.filter(element_symbol = 'Pd')
        [1,2,3]
        >>> a.filter(mass_number = 104)
        [0,1]
        >>> a.filter(mass_number_gt = 104)
        [2,3,4]
        >>> a.filter(['102Ru', '104Ru', '110Cd', '111Cd'])
        [0,4]
        >>> a.filter(['102Ru', '104Ru', '110Cd', '111Cd'], mass_number_lt = 105)
        [0]
        """
        if index_list is None: index_list = [i for i in range(len(self))]

        if isotope is None or isinstance(isotope, IsotopeList):
            pass
        else:
            isotope = IsotopeList(isotope)

        if isotope_not is None or isinstance(isotope_not, IsotopeList):
            pass
        else:
            isotope_not = IsotopeList(isotope_not)

        if element_symbol is None or isinstance(element_symbol, ElementList):
            pass
        else:
            element_symbol = ElementList(element_symbol)

        if element_symbol_not is None or isinstance(element_symbol_not, ElementList):
            pass
        else:
            element_symbol_not = ElementList(element_symbol_not)

        if mass_number is None or isinstance(mass_number, list):
            pass
        else:
            mass_number = [mass_number]

        if mass_number_not is None or isinstance(mass_number_not, list):
            pass
        else:
            mass_number_not = [mass_number_not]

        out = []
        for i in index_list:
            iso = self[i]
            if isotope is not None:
                if iso not in isotope: continue
            if isotope_not is not None:
                if iso in isotope_not: continue

            if element_symbol is not None:
                if iso.element_symbol not in element_symbol: continue
            if element_symbol_not is not None:
                if iso.element_symbol in element_symbol_not: continue

            if mass_number is not None:
                if iso.mass_number not in mass_number: continue
            if mass_number_not is not None:
                if iso.mass_number in mass_number_not: continue

            if mass_number_lt is not None:
                if iso.mass_number >= mass_number_lt: continue
            if mass_number_gt is not None:
                if iso.mass_number <= mass_number_gt: continue

            # It has passed all the tests above
            out.append(i)

        if return_index_list:
            return out
        else:
            return self[out]

class RatioList(_IsopyList):
    """
    RatioList(items = None, reformat = True)

    A subclass of list specifically for list of RatioString items.

    Functions much like a normal list with the exception that all objects must be/are converted to an RatioString. 

    Parameters
    ----------
    items : list of strings or RatioArray, optional
        If ``None`` an empty list will be created. If ``list`` then all items must be strings that can be converted
        into an RatioString. If ``RatioArray`` then the the keys of the array will used to populate the RatioList.

    reformat : bool, optional
        If ``reformat == True`` then strings will be parsed to correct format. If ``reformat == False`` then a ValueError
        will be thrown if strings are not already in the correct format.

    Examples
    --------
    Default initalisation, ``reformat == True``

    >>> RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd'])
    ['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd']
    >>> RatioList(['104pd/108pd', 'pd105/108pd', 'Pd106/Pd108'])
    ['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd']
    >>> RatioList(['104pd', 'pd105', 'Pd106'])
    Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
    ValueError: no "/" found in string

    Initialisation when ``reformat == False``

    >>> RatioList(['104Pd/108Pd', '105Pd108Pd8', '106Pd/108Pd'], True)
    ['104Pd/Pd108', '105Pd/108Pd', '106Pd/108Pd']
    >>> RatioList(['104pd/108pd', 'pd105/108pd', 'Pd106/Pd108'], True)
    Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
    ValueError: First letter of ElementString must be upper case while the remaning letters are in lower case

    Methods
    -------
    __contains__(item)
        Returns ``True`` if list contains `item`. Otherwise returns ``False``.

        Parameters
        ----------
        item : str
            String representation of an isotope ratio

        Returns
        -------
        bool

        Examples
        --------
        >>> '105Pd/108Pd' in RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd'])
        True
        >>> 'pd105/pd108' in RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd'])
        True
        >>> '105Pd/106Pd' in RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd'])
        False
        >>> '105Pd' in RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd'])
        False

    __eq__(other)
        Return ``True`` if items in list are the same as items in `other`. Otherwise returns ``False``.

        Parameters
        ----------
        other : list of strings or RatioList
            List of strings representing isotope ratios

        Return
        ------
        bool

        Examples
        --------
        >>> RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd']) == RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd'])
        True
        >>> RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd']) == ['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd']
        True
        >>> RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd']) == RatioList(['106Pd/108Pd', '105Pd/108Pd', '104Pd/108Pd'])
        True
        >>> RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd']) == RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd', '110Pd/108Pd'])
        False

    append(item, reformat = True)
        Add item to the end of the list.

        Parameters
        ----------
        item : str
            String representation of an isotope ratio
        reformat : bool, optional
            If ``reformat == True`` then item will be parsed to correct format. If ``reformat == False`` then a ValueError will be
            thrown if item is not already in the correct format.

        Examples
        --------      
        >>> a = RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd'])
        >>> a.append('110pd/108pd')
        >>> a
        ['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd', '110Pd/108Pd]

        >>> a = RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd'])
        >>> a.append('108pd/110pd', reformat = False)
        Traceback (most recent call last):
            File "<stdin>", line 1, in <module>
        ValueError: First letter of ElementString must be upper case while the remaning letters are in lower case

    insert(index, item, stict = False)
        Insert item at list index.

        Parameters
        ----------
        index : int
            Postion where item will be added
        item : str
            String representation of an isotope ratio
        reformat : bool, optional
            If ``reformat == True`` then item will be parsed to correct format. If ``reformat == False`` then a ValueError will be
            thrown if item is not already in the correct format.

        Examples
        --------
        >>> a = RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd'])
        >>> a.insert(0,'110pd/108pd')
        >>> a
        ['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd', '110Pd/108Pd']

        >>> a = RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd'])
        >>> a.insert(0,'110pd/108pd', reformat = False)
        Traceback (most recent call last):
            File "<stdin>", line 1, in <module>
        ValueError: First letter of ElementString must be upper case while the remaning letters are in lower case

    remove(item, reformat = True)
        Remove item from list

        Parameters
        ----------
        item : str
            String representation of an isotope
        reformat : bool, optional
            If ``reformat == True`` then item will be parsed to correct format. If ``reformat == False`` then a ValueError will be
            thrown if item is not already in the correct format.

        Examples
        --------
        >>> a = RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd'])
        >>> a.remove(0,'105pd/108pd')
        >>> a
        ['104Pd/108Pd', '106Pd/108Pd']

        >>> a = RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd'])
        >>> a.remove('105pd/108pd', reformat = False)
        Traceback (most recent call last):
            File "<stdin>", line 1, in <module>
        ValueError: First letter of ElementString must be upper case while the remaning letters are in lower case

    copy()
        Returns a copy of the current list.

        Returns
        -------
        RatioList

        Examples
        --------
        >>> a = RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd'])
        >>> b = a.copy()
        >>> b
        ['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd']
        >>> a.append('110Pd/108Pd')
        >>> a
        ['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd', '110Pd/108Pd']
        >>> b
        ['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd']
    """
    _string_class = RatioString
    __name__ = 'RatioList'
    _numerator_string_class = None
    _denominator_string_class = None

    def _new_item(self, item, reformat = True):
        item = super()._new_item(item, reformat)
        if self._numerator_string_class is None: self._numerator_string_class = item.numerator.__class__
        elif not isinstance(item.numerator, self._numerator_string_class):
            raise ValueError('ratio numerator set to "{}" not "{}"'.format(self._numerator_string_class.__name__,
                                                                           item.numerator.__class__.__name__))
        if self._denominator_string_class is None: self._denominator_string_class = item.denominator.__class__
        elif not isinstance(item.denominator, self._denominator_string_class):
            raise ValueError('ratio denominator set to "{}" not "{}"'.format(self._denominator_string_class.__name__,
                                                                             item.denominator.__class__.__name__))
        return item

    @property
    def numerators(self):
        return self.get_numerators()

    @property
    def denominators(self):
        return self.get_denominators()

    def get_numerators(self):
        """
        Returns a list with the numerators for all items in the list.
        
        Returns
        -------
        IsotopeList
            A list of the numerator ``IsotopeString`` for each ``RatioString`` item in list
            
        Examples
        --------
        >>> a = RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd'])
        >>> a.get_numerators()
        ['104Pd', '105Pd', '106Pd']
        """
        if self._numerator_string_class is ElementString:
            return ElementList([rat.numerator for rat in self])
        if self._numerator_string_class is IsotopeString:
            return IsotopeList([rat.numerator for rat in self])
        raise ValueError('numerator class unrecognized')

    def get_denominators(self):
        """
        Returns a list with the denominators for all items in the list.

        Returns
        -------
        IsotopeList
            A list of the denominator ``IsotopeString`` for each ``RatioString`` item in list
            
        Examples
        --------
        >>> a = RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd'])
        >>> a.get_denominators()
        ['108Pd', '108Pd', '108Pd']
        """
        if self._denominator_string_class is ElementString:
            return ElementList([rat.denominator for rat in self])
        if self._denominator_string_class is IsotopeString:
            return IsotopeList([rat.denominator for rat in self])
        raise ValueError('denominator class unrecognized')

    def has_common_denominator(self, denominator = None):
        """
        Returns ``True`` if all isotope ratios in list have a common denominator.
        
        Parameters
        ----------
        denominator : IsotopeString, optional
            Function only returns ``True`` if the common denominator matches ´denominator´.

        Returns
        -------
        bool
        
        Examples
        --------
        >>> a = RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd'])
        >>> a.has_common_denominator()
        True
        
        >>> a = RatioList(['104Pd/102Pd', '105Pd/108Pd', '106Pd/110Pd'])
        >>> a.has_common_denominator()
        False
        
        >>> a = RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd'])
        >>> a.has_common_denominator('110Pd')
        False
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
        Returns the common denominator for each item in the list. Error will be thrown if no common denominator exists.
        
        Returns
        -------
        IsotopeString
            The common denominator for each item in the list
            
        Examples
        --------
        >>> a = RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd'])
        >>> a.get_common_denominator()
        '108Pd'
        
        >>> a = RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd', '101Ru/102Ru'])
        >>> a.get_common_denominator()
        Traceback (most recent call last):
            File "<stdin>", line 1, in <module>
        ValueError: list does not have a common denominator

        """
        if not self.has_common_denominator(): raise ValueError('list does not have a common denominator')
        elif len(self.get_denominators()) == 0: raise ValueError('list is empty')
        else: return self.get_denominators()[0]

    def filter(self, ratio = None, *, ratio_not = None, numerator= None, denominator = None,
               return_index_list = False, index_list = None, **nd_filters):
        """
        Return a list of the index of all items in list that match the given filter restrictions.
        
        Parameters
        ----------
        ratio_list : RatioList
            Ratio must be present in ´ratio_list´.
        numerator : IsotopeString, IsotopeList, optional
            If ``IsotopeString`` the numerator for each item in list must match 'numerator'. If ``dict`` then
            this dict is used to filter a ``IsotopeList`` of each items numerator. See IsotopeList.filter_index
        denominator : IsotopeString, IsotopeList, optional
            If ``IsotopeString`` the denominator for each item in list must match 'numerator'. If ``dict`` then
            this dict is used to filter a ``IsotopeList`` of each items denominator. See IsotopeList.filter_index
        index : list of integers, optional
            List of integers. Only check these indexes. If not given then check all items in list.
        
        Returns
        -------
        list
            List of indexes of all items matches the filter.
            
        Examples
        --------
        >>> a = RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd', '101Ru/102Ru']])
        >>> a.filter(['102Pd/108Pd', '104Pd/108Pd', '106Pd/108Pd'])
        [0,2]
        
        >>> a = RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd', '101Ru/102Ru'])
        >>> a.filter(denominator_filter = '108Pd')
        [0,1,2]
        
        >>> a = RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd', '101Ru/102Ru'])
        >>> a.filter(numerator_filter = {'A_lt': 105})
        [0,3]
        
        >>> a = RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd', '101Ru/102Ru'])
        >>> a.filter(numerator_filter = {'A_lt': 105}, denominator_filter = '102Ru')
        [3]
        """


        if ratio is None or isinstance(ratio, RatioList):
            pass
        else:
            ratio = RatioList(ratio)

        if ratio_not is None or isinstance(ratio_not, RatioList):
            pass
        else:
            ratio_not = RatioList(ratio_not)

        numerator_filters = {}
        denominator_filters = {}
        for k in nd_filters:
            ks = k.split('_', 1)
            if ks[0] == 'numerator' or ks[0] == 'n':
                numerator_filters[ks[1]] = nd_filters[k]
            elif ks[0] == 'denominator' or ks[0] == 'd':
                denominator_filters[ks[1]] = nd_filters[k]
            else: raise ValueError('nd_filter prefix "{}" unknown'.format(ks[0]))

        if index_list is None: index_list = [i for i in range(len(self))]

        out = []
        for i in index_list:
            rat = self[i]
            if ratio is not None:
                if rat not in ratio: continue
            if ratio_not is not None:
                if rat in ratio_not: continue
            out.append(i)

        if numerator is not None or len(numerator_filters) > 0:
            out = self.get_numerators().filter(numerator, index_list= out, return_index_list = True, **numerator_filters)

        if denominator is not None or len(denominator_filters) > 0:
            out = self.get_denominators().filter(denominator, index_list= out, return_index_list = True, **denominator_filters)

        if return_index_list:
            return out
        else:
            return self[out]

def any_list(items, reformat = True):
    """
    
    Parameters
    ----------
    items : (str or int or Isotope or Ratio)
        items to be parsed
    Returns
    -------
    (IsotopeList or RatioList)
    """
    try: return IsotopeList(items, reformat)
    except ValueError: pass
    try: return RatioList(items, reformat)
    except ValueError: raise ValueError('Unable to parse items into IsotopeList or RatioList')

############
### Dict ###
############
#TODO Elementstring, IsotopeString, RatioString keys
class IsopyDict(dict):
    def __init__(self, dictionary=None, values = None, keys = None, float_or_nan = True, **kwargs):
        self._float_or_nan = float_or_nan
        if dictionary is None: dictionary = kwargs
        else:
            dictionary = dictionary.copy()
            dictionary.update(kwargs)
        if values is not None and keys is not None:
            if len(values) != len(keys): raise ValueError('keys and values must have the same length')

            dictionary.update({*zip(keys, values)})

        new = {}
        for k in dictionary:
            try: key = any_string(k)
            except: key = k

            try: new[key] = np.float64(dictionary[k])
            except ValueError:
                if float_or_nan: new[key] = np.nan
                else: new[key] = dictionary[k]

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
        if isinstance(key, list):
            out = {}
            for k in key:
                try: out[k] = self.__getitem__(k)
                except KeyError: out[k] = default
            return out
        try: return self.__getitem__(key)
        except KeyError: return default

    def keys(self, element = False, isotope = False, ratio = False, other = False):
        if not element and not isotope and not ratio and not other:
            return super(IsopyDict, self).keys()

        if isotope and not (element and ratio and other): out = IsotopeList()
        elif ratio and not (element and isotope and other): out = RatioList()
        else: out = []

        for k in super(IsopyDict, self).keys():
            if element and isinstance(k, ElementString): out.append(k)
            elif isotope and isinstance(k, IsotopeString): out.append(k)
            elif ratio and isinstance(k, RatioString): out.append(k)
            elif other: out.append(k)

        return out

#############
### Array ###
#############
class _IsopyArray(np.ndarray):
    def __new__(cls, values = None, keys = None, size = None, **key_data):
        if keys is not None: keys = cls._list_class(keys)
        if values is None and len(key_data) != 0: values = key_data

        if isinstance(values, cls.__class__):
            #Data is already an IsoRatArray so just return a copy
            return values.copy()

        elif isinstance(values, np.ndarray):
            if values.dtype.names is None:
                if keys is None: raise ValueError('keys must be given if data is an numpy array')
                if len(values) != len(keys): raise ValueError('data array and keys must have the same length')

                if values.ndim == 1: dlen = 1
                elif values.ndim == 2: dlen = values.shape[-1]
                else: raise ValueError('data array contains to many dimensions')

                obj = np.zeros(dlen, dtype=[(k, DTYPE) for k in keys])
                obj = obj.view(cls)
                for i in range(len(keys)):
                    obj[keys[i]] = values[i]
            else:
                if keys is None: keys = cls._list_class(values.dtype.names)
                elif len(keys) != len(values.dtype.names): raise ValueError('Number of keys supplied ({}) does not match number'
                    'of keys in data supplied ({})'.format(len(keys), len(values.dtype.names)))
                try: dlen = len(values)
                except: dlen = None
                obj = np.zeros(dlen, dtype=[(k, DTYPE) for k in keys])
                obj = obj.view(cls)
                for i in range(len(keys)):
                    obj[keys[i]] = values[values.dtype.names[i]]

        elif isinstance(values, list):
            if keys is None: raise ValueError('keys must be given if data is a list')
            if len(values) != len(keys): raise ValueError('data list and keys must have the same length')

            dlen = None
            for d in values:
                if dlen is None:
                    try: dlen = len(d)
                    except: pass
                else:
                    try:
                        if dlen != len(d): raise ValueError('not all items in data list have the same length')
                    except:
                        if dlen != 1: raise ValueError('not all items in data list have the same length')

            obj = np.zeros(dlen, dtype=[(k, DTYPE) for k in keys])
            obj = obj.view(cls)
            for i in range(len(keys)):
                obj[keys[i]] = values[i]

        elif isinstance(values, dict):
            dlen = None
            keys = list(values.keys())
            new_keys = cls._list_class(keys)
            for d in values:
                if dlen is None:
                    try:
                        dlen = len(values[d])
                    except:
                        pass
                else:
                    try:
                        if dlen != len(values[d]): raise ValueError('not all items in data list have the same length')
                    except:
                        if dlen != 1: raise ValueError('not all items in data list have the same length')

            obj = np.zeros(dlen, dtype=[(k, DTYPE) for k in new_keys])
            obj = obj.view(cls)
            for i in range(len(keys)):
                try: obj[new_keys[i]] = values[keys[i]]
                except ValueError: raise ValueError('could not convert string to float key: {}'.format(keys[i]))

        elif isinstance(size, int):
            if size == -1: size = None
            if keys is None: raise ValueError('keys not given')
            obj = np.zeros(size, dtype=[(k, DTYPE) for k in keys])
            obj = obj.view(cls)
            if values is not None:
                for k in obj.keys(): obj[k] = values

        else:
            raise ValueError('Unable to create {}'.format(cls.__name__))

        return obj

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
        skip_missing_keys = kwargs.pop('skip_missing_keys', False)
        #
        #TODO what if input is IsoRatDict
        #TODO key not in out
        #keys dont match


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

        if 'out' in kwargs:

            out = kwargs.pop('out')
        else:
            out = self.__class__(size = dlen, keys=keys)

        #ufunc with only one input and operates on each item
        if ufunc.__name__ in ['log', 'log10']:
            for key in keys:
                kwargs['out'] = out[key]
                super(_IsopyArray, self).__array_ufunc__(ufunc, method, inputs[0][key], **kwargs)
            return out

        #ufuncs that uses two inputs
        if ufunc.__name__ in ['add', 'subtract','multiply', 'true_divide', 'floor_divide', 'power', 'sqrt', 'square']:
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

    def mean(self, axis = 0, dtype = None, out = None, **unused_kwargs):
        if dtype is None: dtype = DTYPE
        return self._ado_function(np.mean, axis, out, dtype=dtype)

    def std(self, axis = 0, dtype = None, out = None, ddof = 0, **unused_kwargs):
        if dtype is None: dtype = DTYPE
        return self._ado_function(np.std, axis, out, dtype=dtype, ddof = ddof)

    def sum(self, axis=0, dtype = None, out=None, **unused_kwargs):
        if dtype is None: dtype = DTYPE
        return self._ado_function(np.sum, axis, out, dtype=dtype)

    def max(self, axis=0, out=None, **unused_kwargs):
        return self._ado_function(np.max, axis, out)

    def min(self, axis=0, out=None, **unused_kwargs):
        return self._ado_function(np.min, axis, out)

    def _ado_function(self, func, axis, out, **kwargs):
        #function with axis, dtype, out options e.g. sum, mean std
        #if dtype is None: dtype = DTYPE

        if axis == 0 or axis is None:
            if out is None: out = self.__class__(size = -1, keys = self.keys())
            for key in self.keys():
                func(self[key], out=out[key], **kwargs)
            return out
        if axis == -1:
            if out is None:
                return func([self[x] for x in self.keys()], **kwargs)
            else:
                func([self[x] for x in self.keys()], out=out, **kwargs)
                return out
        if axis == 1:
            if out is None:
                return func([self[x] for x in self.keys()], axis=0, **kwargs)
            else:
                func([self[x] for x in self.keys()], axis=0, out=out, **kwargs)
                return out
        raise ValueError('axis {} is out of bounds'.format(axis))

    def keys(self):
        return self._list_class(self.dtype.names)

    def len(self):
        try: return self.__len__()
        except TypeError: return -1

class ElementArray(_IsopyArray):
    _list_class = ElementList
    _string_class = ElementString
    __name__ = 'ElementArray'

    def filter(self, element_symbol, **filter):
        keys = self.keys().filter(element_symbol, **filter)
        return self[keys]

    def convert_to_ratio_array(self, denominator):
        denominator = ElementString(denominator)
        keys = self.keys()
        if denominator not in keys(): raise ValueError('denominator "{}" not found in array keys'.format(denominator))
        keys.remove(denominator)
        new_keys = keys/denominator
        output_array = RatioArray(values = 1, keys = new_keys, size = self.len())
        for nk in new_keys:
            output_array[nk] = self[nk.numerator]/self[nk.denominator]

class IsotopeArray(_IsopyArray):
    """
    IsotopeArray(data, keys = None)

    A custom view of a structured numpy array storing data with IsotopeString keys.

    Functions much like a normal structured numpy array with a few exceptions.

    Parameters
    ----------
    data : numpy.ndarray
        test 1
    data : list
        test 2
    keys : list of str
        A list of keys to be used with the data supplied.

    Examples
    --------
    Init examples


    Methods
    -------
    keys()
        Return an IsotopeList containing the array keys.

        Returns
        -------
        IsotopeList
            List containing the IsotopeString keys stored in the array.

        Examples
        --------
        >>> a = IsotopeArray(5, keys = ['pd105', '106Pd', '108pd'])
        >>> type(a.keys())
        IsotopeList
        >>> a.keys()
        ['105Pd', '106Pd', 108Pd']


    """
    _list_class = IsotopeList
    _string_class = IsotopeString
    __name__ = 'IsotopeArray'

    def filter(self, isotope = None, **filter):
        """
        Return new IsotopeArray with isotopes of current array that passes matches the supplied filter parameters.

        See IsotopeList.filter() for description of available filter parameters.

        Returns
        -------
        IsotopeArray
            New array containing only the isotope keys that match the specified filter parameters.
        """

        keys = self.keys().filter(isotope, **filter)
        return self[keys]

    def convert_to_ratio_array(self, denominator):
        denominator = IsotopeString(denominator)
        keys = self.keys()
        if denominator not in keys(): raise ValueError('denominator "{}" not found in array keys'.format(denominator))
        keys.remove(denominator)
        new_keys = keys/denominator
        output_array = RatioArray(values = 1, keys = new_keys, size = self.len())
        for nk in new_keys:
            output_array[nk] = self[nk.numerator]/self[nk.denominator]

class RatioArray(_IsopyArray):
    _list_class = RatioList
    _string_class = RatioString
    __name__ = 'RatioArray'

    def filter(self, ratio = None, **filter):
        """
        Return new RatioArray with isotopes of current array that passes matches the supplied filter parameters.

        See :ref:`RatioList.filter() <ratio-filter>` for description of available filter parameters.

        Returns
        -------
        RatioArray
            New array containing only the isotope keys that match the specified filter parameters.
        """

        keys = self.keys().filter(ratio, **filter)
        return self[keys]

    def convert_to_isotope_array(self, denominator_value = 1):
        keys = self.keys()
        if not keys.has_common_denominator(): raise ValueError('array does not have a common denominator')
        numerators = keys.get_numerators()
        if not isinstance(numerators, IsotopeList): raise ValueError('array numerators are not isotope type')
        denominator = keys.get_common_denominator()
        if not isinstance(denominator, IsotopeString): raise ValueError('array denominator are not isotope type')

        numerators.append(denominator)
        output_array = IsotopeArray(values = 1, size = self.len(), keys = numerators)
        for i in range(len(keys)):
            output_array[numerators[i]] = self[keys[i]]
        output_array = output_array * denominator_value
        return output_array

    def convert_to_element_array(self, denominator_value = 1):
        keys = self.keys()
        if not keys.has_common_denominator(): raise ValueError('array does not have a common denominator')
        numerators = keys.get_numerators()
        if not isinstance(numerators, ElementList): raise ValueError('array numerators are not isotope type')
        denominator = keys.get_common_denominator()
        if not isinstance(denominator, ElementString): raise ValueError('array denominator are not isotope type')

        numerators.append(denominator)
        output_array = ElementArray(values = 1, size=self.len(), keys=numerators)
        for i in range(len(keys)):
            output_array[numerators[i]] = self[keys[i]]
        output_array = output_array * denominator_value
        return output_array
