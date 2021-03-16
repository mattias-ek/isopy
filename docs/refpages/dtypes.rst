Data types
**********
.. currentmodule:: isopy

Documented here are the custom data types implemented by isopy for working with and manipulating different types of
geochemical data.

Flavours
--------
Isopy key strings, key lists and isopy arrays come in 5 different flavours, each meant to represent a different kind of geochemical data. The flavour of
each data type is determined by the key string(s) that it contains. The 5 different flavours are:

* The ``Mass``  flavour represent data described by a mass number. Therefore key strings are restricted to integer numbers, e.g. ``"105"``.

* The ``Element``  flavour represents elemental data using the element symbol. Key string are restricted to one or two characters. The first character is always
  in upper case and the second character, if present, is always in lower case, e.g. ``"Pd"``

* The ``Isotope``  flavour represent isotope data and the key string consists of a ``Mass`` key string followed by an ``Element``
  key string, e.g ``"105Pd"``.

* The ``Ratio``  flavour represents a ratio between two sets of data. It consists of a numerator key string and a denominator key string, e.g. ``"108Pd/105Pd"``.

* The ``General``  flavour represent data that cannot be described by any of the other flavours. The key string can be any string e.g. ``"hermione"``.


.. _keystring:

Key strings
-----------
Key strings are what determine the flavour of key lists and isopy arrays. Each key string has a
strict string format but this is in almost all instances applied internally by isopy. Therefore
any string that returns ``True`` when compared to a key string is a valid representation of that
key string:

>>> isopy.keystring('105pd') == 'pd105'
True
>>> isopy.keystring('pd105') == 'PD105'
True
>>> isopy.keystring('pd105') == '105pD'
True

Any key string can be turned in to a ratio key string using ``/``:

>>> isopy.keystring('pd108') / 'pd105'
RatioKeyString('108Pd/105Pd')
>>> 'pd108' / isopy.keystring('pd105') #Works in both directions
RatioKeyString('108Pd/105Pd')


**Note** Comparing two key string with different flavours will always return
``False`` even if the string itself is the same:

>>> elekey = isopy.ElementKeyString('Pd')
>>> genkey = isopy.GeneralKeyString('Pd')
>>> elekey == genkey, str(elekey) == str(genkey)
(False, True)

Comparing the hash of a key string to the hash of a normal string will also always return
``False`` even when a normal comparison would return ``True``:

>>> key = isopy.keystring('Pd')
>>> hash(key) == hash('Pd'), key == 'Pd'
(False, True)

Therefore it is not advised to use key strings as *keys* in a normal ``dict``.
Use one of the isopy dictionaries instead which will automatically convert *key* strings into isopy
key strings before looking up the value.

The following classes/functions can be used to create isopy key strings:

.. autosummary::
    :toctree: Strings

    keystring
    askeystring

    :template: string_template.rst

    MassKeyString
    ElementKeyString
    IsotopeKeyString
    RatioKeyString
    GeneralKeyString

.. _keylist:

Key lists
---------
Key list store contain a sequence of one or more key strings. They can only contain key string of
the same flavour. Key lists are immutable as they inherit from ``tuple``, however, ``+`` and ``-`` can
be used to add/remove items and return a new key list:

>>> isopy.keylist('ru', 'pd', 'cd') + 'ag'
ElementKeyList('Ru', 'Pd', 'Cd', 'Ag')
>>> ['ag', 'rh'] + isopy.keylist('ru', 'pd', 'cd')
ElementKeyList('Ag', 'Rh', 'Ru', 'Pd', 'Cd')

>>> isopy.keylist('ru', 'pd', 'cd') - 'cd'
ElementKeyList('Ru', 'Pd')
>>> ['ru', 'pd', 'rh', 'ag', 'cd'] - isopy.keylist('ru', 'pd', 'cd')
ElementKeyList('Rh', 'Ag')

In addition new key lists can be created using ``&``, ``|`` and ``^``
for *and*, *or* and *xor* operations:

>>> isopy.keylist('ru', 'pd', 'cd') & ['pd','ag', 'rh', 'cd'] #key strings present in both lists
ElementKeyList('Pd', 'Cd')
>>> ['pd','ag', 'rh', 'cd'] | isopy.keylist('ru', 'pd', 'cd') #key strings present in either of the lists
ElementKeyList('Pd', 'Ag', 'Rh', 'Cd', 'Ru')
>>> isopy.keylist('ru', 'pd', 'cd') ^ ['pd','ag', 'rh', 'cd'] #key strings present in only one of the lists
ElementKeyList('Ru', 'Ag', 'Rh')

You can compare two list using ``==`` and  you can test membership using ``in``:

>>> isopy.keylist('ru', 'pd', 'cd') == ['ru', 'pd', 'cd']
True
>>> 'pd' in isopy.keylist('ru', 'pd', 'cd')
True
>>> ['pd', 'ru'] in isopy.keylist('ru', 'pd', 'cd')
True

You can turn any key list into a ratio key list using ``/``:

>>> isopy.keylist('ru', 'pd', 'cd') / 'pd' #Same denominator for all numerators
RatioKeyList('Ru/Pd', 'Pd/Pd', 'Cd/Pd')
>>> ['pd', 'rh', 'ag'] / isopy.keylist('ru', 'pd', 'cd')
RatioKeyList('Pd/Ru', 'Rh/Pd', 'Ag/Cd')


The following classes/functions can be used to create isopy key lists:

.. autosummary::
    :toctree: Lists

    keylist
    askeylist

    :template: list_template.rst

    MassKeyList
    ElementKeyList
    IsotopeKeyList
    RatioKeyList
    GeneralKeyList

.. _isopyarray:

Isopy arrays
------------
Isopy arrays are a custom view of a structured numpy array where each column in the array
is represented by a key string. Isopy arrays can be 0-dimensional, where each column only
holds a single value, or 1-dimensional, where the array contains one or more rows each holding
a single value for each column. Each column has its own data type. If the data type is not
specified upon creation it is inherited from the input, if the input is a numpy ndarray object,
otherwise it defaults to ``np.float64``.

>>> isopy.array([1, 2, 3], ('ru', 'pd', 'cd')) #0-dimensional, no rows
(row) , Ru      , Pd      , Cd
None  , 1.00000 , 2.00000 , 3.00000

>>> isopy.array([[1, 2, 3]], ('ru', 'pd', 'cd')) #1-dimensional, 1 row
(row) , Ru      , Pd      , Cd
0     , 1.00000 , 2.00000 , 3.00000

>>> isopy.array([[1, 2, 3], [11, 12, 13], [21, 22, 23]], ('ru', 'pd', 'cd')) #1-dimensional, 3 rows
(row) , Ru       , Pd       , Cd
0     , 1.00000  , 2.00000  , 3.00000
1     , 11.00000 , 12.00000 , 13.00000
2     , 21.00000 , 22.00000 , 23.00000

>>> isopy.array([1, 2, 3], ('ru', 'pd', 'cd'), dtype=int) #integer data type
(row) , Ru , Pd , Cd
None  , 1  , 2  , 3

>>> isopy.array(np.array([1, 2, 3], dtype=int), ('ru', 'pd', 'cd')) #data type inherited from numpy object
(row) , Ru , Pd , Cd
None  , 1  , 2  , 3

>>> isopy.array([1, 2, 3], ('ru', 'pd', 'cd'), dtype=[int, float, int])
(row) , Ru , Pd      , Cd
None  , 1  , 2.00000 , 3

Isopy arrays can also be created from a number of other data types, e.g:

>>> isopy.array({'ru': [1, 11, 12], 'pd': [2, 12, 22], 'cd':[3, 13, 23]}) #Keys taken from dictionary
(row) , Ru       , Pd       , Cd
0     , 1.00000  , 2.00000  , 3.00000
1     , 11.00000 , 12.00000 , 13.00000
2     , 12.00000 , 22.00000 , 23.00000

>>> isopy.array({'ru': [1, 11, 12], 'pd': [2, 12, 22], 'cd':[3, 13, 23]}, ['101ru', '105pd', '111cd']) #Overrides keys
(row) , 101Ru    , 105Pd    , 111Cd
0     , 1.00000  , 2.00000  , 3.00000
1     , 11.00000 , 12.00000 , 13.00000
2     , 12.00000 , 22.00000 , 23.00000

>>> dataframe = pandas.DataFrame({'ru': [1, 11, 12], 'pd': [2, 12, 22], 'cd':[3, 13, 23]})
>>> isopy.array(dataframe) #inherits data type from the dataframe
(row) , index , ru , pd , cd
0     , 0     , 1  , 2  , 3
1     , 1     , 11 , 12 , 13
2     , 2     , 12 , 22 , 23

Using ``+``, ``-``, ``*``, ``/`` and ``**`` with isopy arrays with scalars or numpy like arrays
will perform the operation on each column in turn.

>>> array = isopy.array([1, 2, 3], ('ru', 'pd', 'cd'))
>>> array * 2
(row) , Ru      , Pd      , Cd
None  , 2.00000 , 4.00000 , 6.00000
>>> array * [1,2,3]
(row) , Ru      , Pd      , Cd
0     , 1.00000 , 2.00000 , 3.00000
1     , 2.00000 , 4.00000 , 6.00000
2     , 3.00000 , 6.00000 , 9.00000

>>> array = isopy.array([[1, 2, 3], [11, 12, 13], [21, 22, 23]], ('ru', 'pd', 'cd'))
>>> array * 2
(row) , Ru       , Pd       , Cd
0     , 2.00000  , 4.00000  , 6.00000
1     , 22.00000 , 24.00000 , 26.00000
2     , 42.00000 , 44.00000 , 46.00000
>>> array * [1,2,3]
(row) , Ru       , Pd       , Cd
0     , 1.00000  , 2.00000  , 3.00000
1     , 22.00000 , 24.00000 , 26.00000
2     , 63.00000 , 66.00000 , 69.00000

Operations involving two isopy arrays will perform the operation per column.

>>> array1 = isopy.array([1, 2, 3], ('ru', 'pd', 'cd'))
>>> array2 = isopy.array([0.5, 1, 1.5], ('ru', 'pd', 'cd'))
>>> array1 * array2
(row) , Ru      , Pd      , Cd
None  , 0.50000 , 2.00000 , 4.50000
>>> array3 = isopy.array([[1, 2, 3], [11, 12, 13], [21, 22, 23]], ('ru', 'pd', 'cd'))
>>> array2 + array3
(row) , Ru       , Pd       , Cd
0     , 1.50000  , 3.00000  , 4.50000
1     , 11.50000 , 13.00000 , 14.50000
2     , 21.50000 , 23.00000 , 24.50000

For columns only present in one of the arrays a default value of ``np.nan`` will be used.
However, isopy comes with functions that allow you to change the default value used.

>>> array1 = isopy.array([1, 2, 3], ('ru', 'pd', 'cd'))
>>> array2 = isopy.array([0.5, 1, 1.5], ('ru', 'pd', 'ag'))
>>> array1 + array2
(row) , Ru      , Pd      , Cd  , Ag
None  , 1.50000 , 3.00000 , nan , nan
>>> isopy.add(array1, array2, 0) #Changes the default value to 0
(row) , Ru      , Pd      , Cd      , Ag
None  , 1.50000 , 3.00000 , 3.00000 , 1.50000

You can also combine isopy arrays with dictionaries. In this case the column *key*
will be looked up in the dictionary. The default value for missing keys is ``np.nan``
for normal dictionaries but for isopy dictionaries their associated default value.

>>> array1 = isopy.array([1, 2, 3], ('ru', 'pd', 'cd'))
>>> dict2 = {'ru': 0.5, 'rh': 0.75, 'pd': 1, 'ag': 1.25, 'cd': 1.5} #Keys are automatically reformatted
>>> array1 * dict2
(row) , Ru      , Pd      , Cd
None  , 0.50000 , 2.00000 , 4.50000

>>> array1 = isopy.array([1, 2, 3], ('ru', 'pd', 'cd'))
>>> dict2 = {'ru': 0.5, 'rh': 0.75, 'pd': 1, 'ag': 1.25}
array1 * dict2
(row) , Ru      , Pd      , Cd
None  , 0.50000 , 2.00000 , nan

>>> array1 = isopy.array([1, 2, 3], ('ru', 'pd', 'cd'))
>>> dict2 = isopy.ScalarDict({'ru': 0.5, 'rh': 0.75, 'pd': 1, 'ag': 1.25}, default_value=1)
array1 * dict2
(row) , Ru      , Pd      , Cd
None  , 0.50000 , 2.00000 , 3.00000

The following classes/functions can be used to create isopy arrays:

.. autosummary::
    :toctree: Arrays

    array
    asarray
    asanyarray

    :template: array_template.rst

    MassArray
    ElementArray
    IsotopeArray
    RatioArray
    GeneralArray

    :template: autosummary/function.rst

    empty
    zeros
    ones
    full

Isopy dictionaries
------------------
Isopy dictionaries store values using isopy key strings. They also allow you to make
dictionaries readonly and to predefine the default value to be used for absent keys.

>>> isopy.IsopyDict({'ru': 0.5, 'rh': 0.75, 'pd': 1, 'ag': 1.25}, default_value=0)
IsopyDict(default_value = 0, readonly = False,
{"Ru": 0.5
"Rh": 0.75
"Pd": 1
"Ag": 1.25})

The ``get()`` method of the :class:`ScalarDict`_ will automatically calculate the value
for an absent ratio key string if both the numerator and the denominator keys are present.

>>> refval = isopy.ScalarDict({'ru': 0.5, 'rh': 0.75, 'pd': 1, 'ag': 1.25}, default_value=0)
>>> refval.get('pd/ru')
2.0

The following classes/functions can be used to create isopy dictionaries:

.. autosummary::
    :toctree: Dicts
    :template: dict_template.rst

    IsopyDict
    ScalarDict