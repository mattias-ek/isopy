import numpy as np

DTYPE = 'f8'

##############
### String ###
##############
class ElementString(str):
    """
    ElementString(string, strict = False)
    
    String representation of an element.
     
    The first letter of an ``ElementString``  is in uppercase and the remaining letters are in lowercase.
       
    
    Parameters
    ----------
    string : str
        Can only contain alphabetical characters.
    strict : bool, optional
        If ``strict == False`` then string will be parsed to correct format. If ``strict == True`` then a ValueError will be
        thrown if string is not already in the correct format.
    
    
    Examples
    --------
    Default initalisation, ``strict == False``
    
    >>> ElementString('Pd')
    'Pd'
    >>> ElementString('pd')
    'Pd
    >>> ElementString('Pd105')
    Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
    ValueError: ElementString "Pd105"can only contain alphabetical characters
    
    Initialisation when ``strict == True``
    
    >>> ElementString('Pd', True)
    'Pd'
    >>> ElementString('pd', True)
    Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
    ValueError: ElementString "pd" incorrectly formatted: First letter must be uppercase

    """
    __name__ = 'ElementString'

    def __new__(cls, string, strict = False):
        if not string.isalpha(): raise ValueError('ElementString "{}" can only contain alphabetical characters'.format(string))
        if len(string) == 0: raise ValueError('ElementString empty')
        if strict:
            if not string[0].isupper(): ValueError('ElementString "{}" incorrectly formatted: First character must be'
                                                   ' uppercase'.format(string))
            if len(string) > 1 and string[1:].islower(): ValueError('ElementString "{}" incorrectly formatted: All but'
                                                                    ' first character must be lowercase'.format(string))

        string = '{}{}'.format(string[0].upper(), string[1:].lower())

        obj = super(ElementString, cls).__new__(cls, string)
        return obj


class IsotopeString(str):
    """
    IsotopeString(string, strict = False)
    
    A subclass of ``str`` for a storing a string representation of an isotope.
    
    A string consisting of the nucleon (mass) number followed by the element of an isotope.

    
    Parameters
    ----------
    string : str
        Must contain nucleon number and element string
    strict : bool, optional
        If ``strict == False`` then string will be parsed to correct format. If ``strict == True`` then a ValueError will be
        thrown if string is not already in the correct format.
    
    
    Attributes
    ----------
    A : int
        The nucleon (mass) number of the isotope
    symbol : ElementString
        The element of the isotope


    Examples
    --------
    Default initalisation, ``strict == False``
    
    >>> IsotopeString('105Pd')
    '105Pd'
    >>> ElementString('pd105')
    '105Pd'
    >>> ElementString('Pd')  
    Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
    ValueError: "Pd" does not contain a nucleon number
    
    Initialisation when ``strict == True``
    
    >>> ElementString('105Pd', True)
    'Pd'
    >>> ElementString('Pd105', True)
    Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
    ValueError: In string "Pd105" element appears before nucleon number (A)
    >>> ElementString('105pd', True)
    Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
    ValueError: First letter of ElementString must be upper case and following letters in lower case
    
    Accessing attibutes
    
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

    def __new__(cls, string, strict = False):
        A, element = cls._parse(string, strict)
        obj = super(IsotopeString, cls).__new__(cls, '{}{}'.format(A, element))
        obj.A = A
        obj.element = element
        return obj

    def _parse(string, strict=False):
        if not isinstance(string, str):
            try: string = str(string)
            except: raise ValueError('unable to parse type {}'.format(type(string)))

        if isinstance(string, IsotopeString):
            return string.A, string.element

        elif isinstance(string, str):
            string = string.strip()

            # If no digits in string then only Symbol is given.
            if string.isalpha():
                raise ValueError('"{}" does not contain a nucleon number'.format(string))


            # If only digits then only nucleon number is given.
            if string.isdigit():
                raise ValueError('"{}" does not contain an element'.format(string))

            # Loop through to split
            l = len(string)
            for i in range(1, l):
                a = string[:i]
                b = string[i:]

                # a = nucleon number and b = Symbol
                if a.isdigit() and b.isalpha():
                    return int(a), ElementString(b, strict)

                # b = nucleon number and a = Symbol
                if a.isalpha() and b.isdigit():
                    if strict:
                        raise ValueError('In string "{}" element appears before nucleon number (A)'.format(string))
                    else:
                        return int(b), ElementString(a, strict)

        # Unable to parse input
        raise ValueError('unable to parse "{}" into IsotopeString'.format(string))

    def __contains__(self, item):
        if isinstance(item, str):
            if item.isdigit():
                return self.A == int(item)
            elif item.isalpha():
                try: return self.symbol == ElementString(item)
                except: return False
        elif isinstance(item, int):
            return self.A == item
        return False


class RatioString(str):
    """
        RatioString(string, strict = False)

        A subclass of ``str`` for a storing an isotope ratio.

        A string consisting the numerator IsotopeString followed by the denominator IsotopeString seperated by a "/".

        Parameters
        ----------
        string : str
            Must contain two IsotopeString seperated by a "/"
        strict : bool, optional
            If ``strict == False`` then string will be parsed to correct format. If ``strict == True`` then a ValueError will be
            thrown if string is not already in the correct format.


        Attributes
        ----------
        numerator : IsotopeString
            The numerator isotope in the ratio
        denominator : IsotopeString
            The denominator isotope in the ratio


        Examples
        --------
        Default initalisation, ``strict == False``

        >>> RatioString('105Pd/108Pd')
        '105Pd/108Pd'
        >>> RatioString('pd105/108pd')
        '105Pd/108Pd'
        >>> RatioString('105Pd')  
        Traceback (most recent call last):
            File "<stdin>", line 1, in <module>
        ValueError: no "/" found in string

        Initialisation when ``strict == True``

        >>> RatioString('105Pd/108Pd', True)
        '105Pd/108Pd'
        >>> RatioString('pd105/108pd', True)
        Traceback (most recent call last):
            File "<stdin>", line 1, in <module>
        ValueError: Unable to parse numerator: "pd105"
        >>> RatioString('105Pd/108pd', True)
        Traceback (most recent call last):
            File "<stdin>", line 1, in <module>
        ValueError: Unable to parse denominator: "108pd"

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

    def __new__(cls, string, strict = False):
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


        try: numer = ElementString(numer, strict)
        except ValueError:
            try: numer = IsotopeString(numer, strict)
            except ValueError: ValueError('Unable to parse numerator: "{}"'.format(numer))

        try: denom = ElementString(denom, strict)
        except ValueError:
            try: denom = IsotopeString(denom, strict)
            except: raise ValueError('Unable to parse denominator: "{}"'.format(denom))

        obj =  super(RatioString, cls).__new__(cls, '{}/{}'.format(numer, denom))
        obj.numerator = numer
        obj.denominator = denom
        return obj

    def __contains__(self, item, strict = False):
        if isinstance(item, str):
            try: item = ElementString(item, strict)
            except: pass
            try: item = IsotopeString(item, strict)
            except: pass
        if isinstance(item, (IsotopeString, ElementString)):
            return self.numerator == item or self.denominator == item
        return False


def any_string(string, strict = False):
    """
        Shortcut function that will return an IsotopeString or a RatioString depending on the supplied string.
        
        Parameters
        ----------
        string : str
            A string that can be parsed into either an IsotopeSting or a RatioSting
        strict : bool, optional
            If ``strict == False`` then string will be parsed to correct format. If ``strict == True`` then a ValueError will be
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
    try: return ElementString(string, strict)
    except ValueError: pass
    try: return IsotopeString(string, strict)
    except ValueError: pass
    try: return RatioString(string, strict)
    except ValueError: raise #ValueError('Unable to parse items into ElementString, IsotopeList or RatioList')

############
### List ###
############
class _IsopyList(list):
    def __init__(self, items = None, strict = False):
        if items is None: items = []
        elif isinstance(items, np.ndarray): items = items.dtype.names
        try: super().__init__([self._new_item(item, strict) for item in items])
        except: raise

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
        if not isinstance(item, self._string_class):
            try: item = self._string_class(item)
            except: return None
        
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

    def _new_item(self, item, strict = False):
        return self._string_class(item, strict)

    def append(self, item, strict = False):
        return super().append(self._new_item(item, strict))

    def insert(self, index, item, strict = False):
        return super().insert(index, self._new_item(item, strict))

    def remove(self, item, strict = False):
        return super().remove(self._new_item(item, strict))

    def copy(self):
        return self.__class__([self._new_item(x) for x in self])


class ElementList(_IsopyList):
    _string_class = ElementString
    __name__ = 'ElementList'

    def filter_index(self, element = None, *, index = None):
        if index is None: index = [i for i in range(len(self))]

        if element is None or isinstance(element, (ElementList, IsopyDict)):
            pass
        elif isinstance(element, list):
            element = ElementList(element)
        else:
            return self.index(ElementString(element))

        out = []
        for i in index:
            ele = self[i]
            if element is not None:
                if ele not in element: continue

            out.append(i)
        return out

    def filter(self, element = None, *, index = None):
        return self[self.filter_index(element, index = index)]


class IsotopeList(_IsopyList):
    """
        IsotopeList(items = None, strict = False)

        A subclass of list specifically for list of IsotopeString items.

        Functions much like a normal list with the exception that all objects must be/are converted to an IsotopeString. 

        Parameters
        ----------
        items : list of strings or IsotopeArray, optional
            If ``None`` an empty list will be created. If ``list`` then all items must be strings that can be converted
            into an IsotopeString. If ``IsotopeArray`` then the the keys of the array will used to populate the IsotopeList.
        
        strict : bool, optional
            If ``strict == False`` then strings will be parsed to correct format. If ``strict == True`` then a ValueError will be
            thrown if strings are not already in the correct format.
        
        Examples
        --------
        Default initalisation, ``strict == False``

        >>> IsotopeList(['104Pd', '105Pd', '106Pd'])
        ['104Pd', '105Pd', '106Pd']
        >>> IsotopeList(['104pd', 'pd105', 'Pd106'])
        ['104Pd', '105Pd', '106Pd']
        >>> IsotopeList(['104', 'pd105', 'Pd106'])
        Traceback (most recent call last):
            File "<stdin>", line 1, in <module>
        ValueError: "104" does not contain an element

        Initialisation when ``strict == True``

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
            item : str
                String representation of an isotope
                 
            Returns
            -------
            bool
            
            Examples
            --------
            >>> '105Pd' in IsotopeList(['104Pd', '105Pd', '106Pd'])
            True
            >>> 'pd105' in IsotopeList(['104Pd', '105Pd', '106Pd'])
            True
            >>> '108Pd' in IsotopeList(['104Pd', '105Pd', '106Pd'])
            False
            
        __eq__(other)
            Return ``True`` if list equals `other`. Otherwise returns ``False``.
            
            Parameters
            ----------
            other : list of strings or IsotopeList
                List of strings representing isotopes
            
            Return
            ------
            bool
        
            Examples
            --------
            >>> IsotopeList(['104Pd', '105Pd', '106Pd']) == IsotopeList(['104Pd', '105Pd', '106Pd'])
            True
            >>> IsotopeList(['104Pd', '105Pd', '106Pd']) == ['104Pd', '105Pd', '106Pd']
            False
            >>> IsotopeList(['106Pd', '105Pd', '104Pd']) == IsotopeList(['104Pd', '105Pd', '106Pd'])
            True
            >>> IsotopeList(['104Pd', '105Pd', '108Pd', '108Pd']) == IsotopeList(['104Pd', '105Pd', '106Pd'])
            False
        
        append(item, strict = False)
            Add item to the end of the list.
            
            Parameters
            ----------
            item : str
                String representation of an isotope
            strict : bool, optional
                If ``strict == False`` then item will be parsed to correct format. If ``strict == True`` then a ValueError will be
                thrown if item is not already in the correct format.
    
            Examples
            --------      
            >>> a = IsotopeList(['104Pd', '105Pd', '106Pd'])
            >>> a.append('108pd')
            >>> a
            ['104Pd', '105Pd', '106Pd', '108Pd']
            
            >>> a = IsotopeList(['104Pd', '105Pd', '106Pd'])
            >>> a.append('108pd', strict = True)
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
                String representation of an isotope
            strict : bool, optional
                If ``strict == False`` then item will be parsed to correct format. If ``strict == True`` then a ValueError will be
                thrown if item is not already in the correct format.
    
            Examples
            --------
            >>> a = IsotopeList(['104Pd', '105Pd', '106Pd'])
            >>> a.insert(0,'102pd')
            >>> a
            ['102Pd', '104Pd', '105Pd', '106Pd']
            
            >>> a = IsotopeList(['104Pd', '105Pd', '106Pd'])
            >>> a.insert(0,'102pd', strict = True)
            Traceback (most recent call last):
                File "<stdin>", line 1, in <module>
            ValueError: First letter of ElementString must be upper case while the remaning letters are in lower case
        
        remove(item, strict = False)
            Remove item from list
        
            Parameters
            ----------
            item : str
                String representation of an isotope
            strict : bool, optional
                If ``strict == False`` then item will be parsed to correct format. If ``strict == True`` then a ValueError will be
                thrown if item is not already in the correct format.
    
            Examples
            --------
            >>> a = IsotopeList(['104Pd', '105Pd', '106Pd'])
            >>> a.remove(0,'105pd')
            >>> a
            ['104Pd', '106Pd']
            
            >>> a = IsotopeList(['104Pd', '105Pd', '106Pd'])
            >>> a.remove('105pd', strict = True)
            Traceback (most recent call last):
                File "<stdin>", line 1, in <module>
            ValueError: First letter of ElementString must be upper case while the remaning letters are in lower case
        
        copy()
            Returns a copy of the current list.
        
            Returns
            -------
            IsotopeList
            
            Examples
            --------
            >>> a = IsotopeList(['104Pd', '105Pd', '106Pd'])
            >>> b = a.copy()
            >>> b
            ['104Pd', '105Pd', '106Pd']
            >>> a.append('108Pd')
            >>> a
            ['104Pd', '105Pd', '106Pd', 108Pd]
            >>> b
            ['104Pd', '105Pd', '106Pd']
        
        """
    _string_class = IsotopeString
    __name__ = 'IsotopeList'

    def filter_index(self, isotope = None, *, element=None, A=None, A_lt=None, A_gt=None, index = None):
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
        A : int, optional
            Isotope must have this nucleon number (A)
        A_lt : int, optional
            Isotope must have a nucleon number (A) less than this value
        A_gt : int, optional
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
        >>> a.filter(element = 'Pd')
        [1,2,3]
        >>> a.filter(A = 104)
        [0,1]
        >>> a.filter(A_gt = 104)
        [2,3,4]
        >>> a.filter(['102Ru', '104Ru', '110Cd', '111Cd'])
        [0,4]
        >>> a.filter(['102Ru', '104Ru', '110Cd', '111Cd'], A_lt = 105)
        [0]
        """
        if index is None: index = [i for i in range(len(self))]

        if isotope is None or isinstance(isotope, (IsotopeList, IsopyDict)):
            pass
        elif isinstance(isotope, (list, IsotopeArray)):
            isotope = IsotopeList(isotope)
        else:
            isotope = IsotopeList([isotope])

        if element is None or isinstance(element, ElementList):
            pass
        elif isinstance(element, list):
            element = ElementList(element)
        else:
            element = ElementList([element])

        out = []
        for i in index:
            iso = self[i]
            # Do not include iso if it fails any of these tests
            if isotope is not None:
                if iso not in isotope: continue
            if element is not None:
                if iso.symbol not in element: continue
            if A is not None:
                if iso.A != A: continue
            if A_lt is not None:
                if iso.A > A_lt: continue
            if A_gt is not None:
                if iso.A < A_gt: continue

            # It has passed all the tests above
            out.append(i)
        return out

    def filter(self, isotope_list = None, *, element=None, A=None, A_lt=None, A_gt=None, index = None):
        """
        Return only isotopes in list that matches the filters specified.

        Parameters
        ----------
        isotope_list : IsotopeList
            Keep only isotopes appearing in self and the supplied IsotopeList
        element : str, optional
            Isotope must have this element symbol
        A : int, optional
            Isotope must have this nucleon number (A)
        A_lt : int, optional
            Isotope must have a nucleon number (A) less than this value
        A_gt : int, optional
            Isotope must have a nucleon number (A) greater than this value

        Returns
        -------
        IsotopeList
            List consisting of the isotopes that comply with the filter specified.
            
        Examples
        --------
        >>> a = IsotopeList(['104Ru', '104Pd', '105Pd', '106Pd', '111Cd'])
        >>> a.filter(element = 'Pd')
        ['104Pd', '105Pd', '106Pd']
        >>> a.filter(A = 104)
        ['104Ru', '104Pd']
        >>> a.filter(A_gt = 104)
        ['105Pd', '106Pd', '111Cd']
        >>> a.filter(['102Ru', '104Ru', '110Cd', '111Cd'])
        ['104Ru', '111Cd']
        >>> a.filter(['102Ru', '104Ru', '110Cd', '111Cd'], A_lt = 105)
        ['104Ru']
        """
        return self[self.filter_index(isotope_list, element=element, A=A, A_lt=A_lt, A_gt=A_gt, index = index)]


class RatioList(_IsopyList):
    """
    RatioList(items = None, strict = False)

    A subclass of list specifically for list of RatioString items.

    Functions much like a normal list with the exception that all objects must be/are converted to an RatioString. 

    Parameters
    ----------
    items : list of strings or RatioArray, optional
        If ``None`` an empty list will be created. If ``list`` then all items must be strings that can be converted
        into an RatioString. If ``RatioArray`` then the the keys of the array will used to populate the RatioList.

    strict : bool, optional
        If ``strict == False`` then strings will be parsed to correct format. If ``strict == True`` then a ValueError
        will be thrown if strings are not already in the correct format.

    Examples
    --------
    Default initalisation, ``strict == False``

    >>> RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd'])
    ['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd']
    >>> RatioList(['104pd/108pd', 'pd105/108pd', 'Pd106/Pd108'])
    ['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd']
    >>> RatioList(['104pd', 'pd105', 'Pd106'])
    Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
    ValueError: no "/" found in string

    Initialisation when ``strict == True``

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

    append(item, strict = False)
        Add item to the end of the list.

        Parameters
        ----------
        item : str
            String representation of an isotope ratio
        strict : bool, optional
            If ``strict == False`` then item will be parsed to correct format. If ``strict == True`` then a ValueError will be
            thrown if item is not already in the correct format.

        Examples
        --------      
        >>> a = RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd'])
        >>> a.append('110pd/108pd')
        >>> a
        ['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd', '110Pd/108Pd]

        >>> a = RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd'])
        >>> a.append('108pd/110pd', strict = True)
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
        strict : bool, optional
            If ``strict == False`` then item will be parsed to correct format. If ``strict == True`` then a ValueError will be
            thrown if item is not already in the correct format.

        Examples
        --------
        >>> a = RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd'])
        >>> a.insert(0,'110pd/108pd')
        >>> a
        ['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd', '110Pd/108Pd']

        >>> a = RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd'])
        >>> a.insert(0,'110pd/108pd', strict = True)
        Traceback (most recent call last):
            File "<stdin>", line 1, in <module>
        ValueError: First letter of ElementString must be upper case while the remaning letters are in lower case

    remove(item, strict = False)
        Remove item from list

        Parameters
        ----------
        item : str
            String representation of an isotope
        strict : bool, optional
            If ``strict == False`` then item will be parsed to correct format. If ``strict == True`` then a ValueError will be
            thrown if item is not already in the correct format.

        Examples
        --------
        >>> a = RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd'])
        >>> a.remove(0,'105pd/108pd')
        >>> a
        ['104Pd/108Pd', '106Pd/108Pd']

        >>> a = RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd'])
        >>> a.remove('105pd/108pd', strict = True)
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

    def _new_item(self, item, strict = False):
        item = super()._new_item(item, strict)
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
        
        >>> a = RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd', '101Ru/102Ru'])
        >>> a.has_common_denominator()
        False
        
        >>> a = RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd'])
        >>> a.has_common_denominator('110Pd')
        False
        """
        denom = self.get_denominators()
        if len(denom) == 0:
            return True
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

    def filter_index(self, ratio = None, *, numerator = None, denominator = None, index = None):
        """
        Return a list of the index of all items in list that match the given filter restrictions.
        
        Parameters
        ----------
        ratio_list : RatioList
            Ratio must be present in ´ratio_list´.
        numerator : IsotopeString or dict, optional
            If ``IsotopeString`` the numerator for each item in list must match 'numerator'. If ``dict`` then
            this dict is used to filter a ``IsotopeList`` of each items numerator. See IsotopeList.filter_index
        denominator : IsotopeString or dict, optional
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
        >>> a.filter_index(['102Pd/108Pd', '104Pd/108Pd', '106Pd/108Pd'])
        [0,2]
        
        >>> a = RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd', '101Ru/102Ru'])
        >>> a.filter_index(denominator = '108Pd')
        [0,1,2]
        
        >>> a = RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd', '101Ru/102Ru'])
        >>> a.filter_index(numerator = {'A_lt': 105})
        [0,3]
        
        >>> a = RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd', '101Ru/102Ru'])
        >>> a.filter_index(numerator = {'A_lt': 105}, denominator = '102Ru')
        [3]
        """
        if ratio is None or isinstance(ratio, RatioList):
            pass
        elif isinstance(ratio, (list, RatioArray)):
            ratio = RatioList(ratio)
        else:
            ratio = RatioList([ratio])

        if not isinstance(numerator, dict): raise ValueError('numberator must be a dict')
        if not isinstance(denominator, dict): raise ValueError('numberator must be a dict')

        if index is None: index = [i for i in range(len(self))]

        out = []
        if ratio is not None:
            for i in index:
                if self[i] in ratio: out.append(i)
            index = out

        if numerator is not None:
            numerator['index'] = index
            out = self.get_numerators().filter_index(**numerator)
            index = out

        if denominator is not None:
            denominator['index'] = index
            out = self.get_denominators().filter_index(**denominator)
            index = out

        return out

    def filter(self, ratio_list = None, *, numerator = None, denominator = None, index = None):
        """
        Return a list items in list that match the given filter restrictions.

        Parameters
        ----------
        ratio_list : RatioList
            Ratio must be present in ´ratio_list´.
        numerator : IsotopeString or dict, optional
            If ``IsotopeString`` the numerator for each item in list must match 'numerator'. If ``dict`` then
            this dict is used to filter a ``IsotopeList`` of each items numerator. See IsotopeList.filter_index
        denominator : IsotopeString or dict, optional
            If ``IsotopeString`` the denominator for each item in list must match 'numerator'. If ``dict`` then
            this dict is used to filter a ``IsotopeList`` of each items denominator. See IsotopeList.filter_index
        index : list of integers, optional
            List of integers. Only check these indexes. If not given then check all items in list.

        Returns
        -------
        RatioList
            List of all items matches the filter.

        Examples
        --------
        >>> a = RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd', '101Ru/102Ru']])
        >>> a.filter_index(['102Pd/108Pd', '104Pd/108Pd', '106Pd/108Pd'])
        ['104Pd/108Pd', '106Pd/108Pd']

        >>> a = RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd', '101Ru/102Ru'])
        >>> a.filter_index(denominator = '108Pd')
        ['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd']

        >>> a = RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd', '101Ru/102Ru'])
        >>> a.filter_index(numerator = {'A_lt': 105})
        ['104Pd/108Pd', '101Ru/102Ru']

        >>> a = RatioList(['104Pd/108Pd', '105Pd/108Pd', '106Pd/108Pd', '101Ru/102Ru'])
        >>> a.filter_index(numerator = {'A_lt': 105}, denominator = '102Ru')
        ['101Ru/102Ru']
        """
        return self[self.filter_index(ratio_list, numerator = numerator, denominator = denominator, index = index)]


def any_list(items, strict = False):
    """
    
    Parameters
    ----------
    items : (str or int or Isotope or Ratio)
        items to be parsed
    Returns
    -------
    (IsotopeList or RatioList)
    """
    try: return IsotopeList(items, strict)
    except ValueError: pass
    try: return RatioList(items, strict)
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
        if isinstance(key, list): return [self.__getitem__(k) for k in key]

        try: key = any_string(key)
        except: pass

        try: return super(IsopyDict, self).__getitem__(key)
        except KeyError as err:
            if isinstance(key, RatioString):
                try: return super(IsopyDict, self).__getitem__(key.numerator) / super(IsopyDict, self).__getitem__(key.denominator)
                except: raise err
            else: raise

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
    def __new__(cls, data, keys = None):
        if keys is not None: keys = cls._list_class(keys)

        if isinstance(data, cls.__class__):
            #Data is already an IsoRatArray so just return a copy
            return data.copy()

        elif isinstance(data, np.ndarray):
            if data.dtype.names is None:
                if keys is None: raise ValueError('keys must be given if data an numpy array')
                if len(data) != len(keys): raise ValueError('data array and keys must have the same length')

                if data.ndim == 1: dlen = 1
                elif data.ndim == 2: dlen = data.shape[-1]
                else: raise ValueError('data array contains to many dimensions')

                obj = np.zeros(dlen, dtype=[(k, DTYPE) for k in keys])
                obj = obj.view(cls)
                for i in range(len(keys)):
                    obj[keys[i]] = data[i]
            else:
                if keys is None: keys = cls._list_class(data.dtype.names)
                elif len(keys) != data.dtype.names: raise ValueError('Number of keys supplied ({}) does not match number'
                    'of keys in data supplied ({}'.format(len(keys), len(data.dtype.names)))
                try: dlen = len(data)
                except: dlen = None
                obj = np.zeros(dlen, dtype=[(k, DTYPE) for k in keys])
                obj = obj.view(cls)
                for i in range(len(keys)):
                    obj[keys[i]] = data[data.dtype.names[i]]

        elif isinstance(data, list):
            if keys is None: raise ValueError('keys must be given if data is a list')
            if len(data) != len(keys): raise ValueError('data list and keys must have the same length')

            dlen = None
            for d in data:
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
                obj[keys[i]] = data[i]

        elif isinstance(data, dict):
            dlen = None
            keys = list(data.keys())
            new_keys = cls._list_class(keys)
            for d in data:
                if dlen is None:
                    try:
                        dlen = len(data[d])
                    except:
                        pass
                else:
                    try:
                        if dlen != len(data[d]): raise ValueError('not all items in data list have the same length')
                    except:
                        if dlen != 1: raise ValueError('not all items in data list have the same length')

            obj = np.zeros(dlen, dtype=[(k, DTYPE) for k in new_keys])
            obj = obj.view(cls)
            for i in range(len(keys)):
                try: obj[new_keys[i]] = data[keys[i]]
                except ValueError: raise ValueError('could not convert string to float key: {}'.format(keys[i]))

        elif isinstance(data, int):
            if keys is None: raise ValueError('keys not given')
            obj = np.zeros(data, dtype=[(k, DTYPE) for k in keys])
            obj = obj.view(cls)
        elif data is None:
            if keys is None: raise ValueError('keys not given')
            obj = np.zeros(None, dtype=[(k, DTYPE) for k in keys])
            obj = obj.view(cls)
        else:
            raise ValueError('data type "{}" cannot be converted into IsoRatArray'.format(type(data)))

        return obj

    def __getitem__(self, key):
        if not isinstance(key, self._string_class):
            try: key = self._string_class(key)
            except: pass

        arr = super(_IsopyArray, self).__getitem__(key)
        if arr.dtype.names is None: return arr.view(np.ndarray)
        else: return arr.view(self.__class__)

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
        dlen = None
        for i in inputs:
            if isinstance(i, _IsopyArray):
                if keys is None:
                    keys = i.keys()
                else:
                    if skip_missing_keys: keys = keys.filter(i.keys())
                    elif keys != i.keys(): raise ValueError('keys of {} dont match.'.format(self.__name__))
            if dlen is None and isinstance(i, (list, tuple, np.ndarray)):
                try: dlen = len(i)
                except: pass


        if len(keys) == 0: return np.array([])

        if 'out' in kwargs:
            out = kwargs.pop('out')
        else:
            out = self.__class__(dlen, keys=keys)

        #ufunc with only one input and operates on each item
        if ufunc.__name__ in ['log']:
            for key in keys:
                kwargs['out'] = out[key]
                super(_IsopyArray, self).__array_ufunc__(ufunc, method, inputs[0][key], **kwargs)
            return out

        #ufuncs that uses two inputs
        if ufunc.__name__ in ['add', 'subtract','multiply', 'true_divide', 'floor_divide', 'power']:
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

    def mean(self, out = None, **unused_kwargs):
        #TODO warning for unused kwargs
        if out is None: out = self.__class__(1, keys = self.keys())
        for key in self.keys():
            out[key] = np.mean(self[key])
        return out

    def std(self, out = None, ddof = 0, **unused_kwargs):
        if out is None: out = self.__class__(1, keys = self.keys())
        for key in self.keys():
            out[key] = np.std(self[key], ddof = ddof)
        return out

    def keys(self):
        return self._list_class(self.dtype.names)

class ElementArray(_IsopyArray):
    _list_class = ElementList
    _string_class = ElementString
    __name__ = 'ElementArray'

    def filter(self, element, **filter):
        index = self.keys().filter(element, **filter)
        return self[index].copy()

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

        index = self.keys().filter(isotope, **filter)
        return self[index].copy()

class RatioArray(_IsopyArray):
    _list_class = RatioList
    _string_class = RatioString
    __name__ = 'RatioArray'

    def filter(self, ratio = None, **filter):
        """
        Return new RatioArray with isotopes of current array that passes matches the supplied filter parameters.

        See RatioList.filter() for description of available filter parameters.

        Returns
        -------
        RatioArray
            New array containing only the isotope keys that match the specified filter parameters.
        """

        index = self.keys().filter(ratio, **filter)
        return self[index].copy()
