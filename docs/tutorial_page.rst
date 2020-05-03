Tutorial
********

.. currentmodule:: isopy

Introduction
============
The main feature of isopy are :ref:`IsopyArrays` which can be though of as a table containing rows and columns.
Each column is represented by a *key* which identifies the information stored in the column. These come in four
different flavours, each designed to represent a specific kind of data.

In :class:`MassArray` 's and :class:`ElementArrays` 's each column key represented by a mass number and an element
symbol, respectively. The column key for :class:`IsotopeArray` 's contains both a mass number and an
element symbol. Finally we have :class:`RatioArray` 's where the column key consists of a numerator and a denominator.
These can be any of the aforementioned column keys but each numerator must be of the same flavour,
as must each denominator. However, the flavour of the numerators do not have be the same denominators. So you could have
a ratio array consisting of isotope numerators and element symbol denominators.

Each column key is represented by an :ref:`IsopyStrings` of the same flavour as the array. Mass number keys are
represented by :class:`MassString` 's which is string an integer. Element symbols keys are
represented by :class:`ElementString` 's which are limited to two alphabetical characters where the first character
must be upper case and the following character, if given, in lower case. Isotope keys are represented by
:class:`IsotopeString` 's which consists of a :class:`MassString` followed by a :class:`ElementString`. Ratio keys are
represented by :class:`RatioString` 's  which consists of two of the aforementioned strings seperated by a "/".

Isopy will automatically convert strings into the correct format which means that to access a column with an element
symbol key any string that can be converted to the correct format will work. For example if we want to access a
column with the key 'Pd' we can use 'Pd', 'pd', 'PD' or 'pD'. For isotope keys the order of the mass number and the
element symbol can be swapped so instead of '105Pd' you could use 'pd105'.

:ref:`IsopyLists` come in the same 4 flavours, :class:`MassList`, :class:`ElementList`, :class:`IsotopeList` and
:class:`RatioList`. These behave just like normal lists but with the restriction that it can only contain their
:ref:`IsopyStrings` counterpart. These lists are returned when calling ``IsopyArray.keys()`` and contain a few
additional features that are useful for processing data.

Below we will discuss some of the functionality of these different data types. For a complete description of the
functionality of these different data types please see their respective documentation.

Creating arrays
===============

We can create an empty array, where all values are 0, by just passing an integer and the desired keys to
:func:`array`

>>> a = isopy.array(2, '99ru, 105pd, 111cd')
>>> print(a)
99Ru, 105Pd, 111Cd
0.0 , 0.0  , 0.0
0.0 , 0.0  , 0.0

Note that when isopy expects an :ref:`IsopyLists` and gets a string it will turn the string into a list by splitting
it with the ',' delimiter.

If we want already know which flavour of array we want we can create it directly using the :ref:`IsopyArrays` class,
e.g. ``a = isopy.IsotopeArray(5, '99ru, 105pd, 111cd')``. This can be useful if you need to make sure your data is
a certain flavour.

From dicts
----------
We can create an array from a dict by just passing the dict

>>> a = isopy.array({'99ru': [1,2], '105pd': [11,12], '111Cd': [21,22]})
>>> print(a)
99Ru, 105Pd, 111Cd
1.0 , 11.0 , 21.0
2.0 , 12.0 , 22.0

If the first argument is not a numpy ndarray isopy will automatically convert the data values to float. If we want
the values to have a different type we have to pass the *dtype* keyword with the type we want.

>>> a = isopy.array({'99ru': [1, 2], '105pd': [11, 12], '111Cd': [21, 22]}, dtype=int)
>>> print(a)
99Ru, 105Pd, 111Cd
1   , 11   , 21
2   , 12   , 22

From lists/ndarrays
-------------------
If we want to create an array from a list of values we also have to specify the column keys of the final array

>>> a = isopy.array([[1, 11, 21], [2, 12, 22]], ['99ru', '105pd', '111cd'])
>>> print(a)
99Ru, 105Pd, 111Cd
1.0 , 11.0 , 21.0
2.0 , 12.0 , 22.0


If *values* is a numpy ndarray and *dtype* is not specified it will take the dtype of the ndarray

>>> a = isopy.array(np.array([[1, 11, 21], [2, 12, 22]]), ['99ru', '105pd', '111cd'])
>>> print(a)
99Ru, 105Pd, 111Cd
1   , 11   , 21
2   , 12   , 22


For lists or arrays with more than one dimension the first dimension is always the row and the second dimension
is always the column.

If we just pass a single list then we will create a dimensionless array. We can see this because the ``a.ndim`` will
return 0 and ``a.nrows`` will return -1. Also ``len(a)`` will raise an error for dimensionless arrays. However,
``a.size`` will return 1 both for dimensionless arrays and 1 dimensional arrays with a single row.

>>> a = isopy.array([[1, 11, 21]], ['99ru', '105pd', '111cd'])
>>> print(a.ndim, a.nrows, a.size)
1, 1, 1
>>> a = isopy.array([1, 11, 21], ['99ru', '105pd', '111cd'])
>>> print(a.ndim, a.nrows, a.size)
0, -1, 1

You can use the *ndim* keyword to specify if you want your array to have 0 or 1 dimension. If you pass ``ndim=0`` and
*values* contains more than one row of data an error will be raised. This can be avoided by using ``ndim=-1`` which
will return a dimensionless array only if *values* only contains one row of data.

>>> >>> a = isopy.array([[1, 11, 21]], ['99ru', '105pd', '111cd'], ndim=0)
>>> print(a.ndim, a.nrows, a.size)
0, -1, 1
>>> a = isopy.array([1, 11, 21], ['99ru', '105pd', '111cd'], ndim=1)
>>> print(a.ndim, a.nrows, a.size)
1, 1, 1

From other IsopyArrays
----------------------
We can also pass another :ref:`IsopyArrays` as the first argument which will return a copy of the array. If we just
want to make sure the data is an isopy array and do not need a copy we can use ``isopy.asarray``. This function behaves
just like ``np.array`` with the exception that if the first argument already is an isopy array it will just return the
array without copying it (If no other arguments are given).

>>> a == isopy.array(a)
False
>>> a == isopt.asarray(a)
True

We can copy the data in an isopy array with a new set of keys by simply passing the array and the new keys

>>> b = np.array(a, a.keys().element_symbols())
>>> print(b)
Ru  , Pd   , Cd
1.0 , 11.0 , 21.0
2.0 , 12.0 , 22.0


Here we copied array *a* and made the new keys the element symbols of the old keys.


From csv/excel files
-------------------------------------
Isopy provides a function for reading data from csv files. Assuming we have a file that looks like this::

    #Lines at the beginning of the file
    # starting with '#' will be ignored
    ru99, pd105, cd111
    1, 11, 21
    2, 12, 22


We can use :func:`read_csv` to import the data in this file as a dictionary where the first row is the key
and the remaining rows will be appended to a list for each column. We can then pass this dictionary to
:func:`array`  to create an isopy array.

>>> data = isopy.read_csv('ourfile.csv')
>>> a= isopy.array(data)
>>> print(data)
99Ru, 105Pd, 111Cd
1.0 , 11.0 , 21.0
2.0 , 12.0 , 22.0


We can also read data from excel workbooks using :func:`read_excel`. If sheetname is
an integer then it will return the data for the sheet with this index. If sheetname is a string then it will
return the data for the shett with this name. If sheetname is omitted, or None, it will return a dictionary
containing the data for each sheet in the workbook.

Assuming sheet 'Sheet1' contains the same data as our csv file above we can do

>>> data = isopy.read_excel('ourfile.xlsx', 'Sheet1')
>>> a = isopy.array(data)
>>> print(data)
99Ru, 105Pd, 111Cd
1.0 , 11.0 , 21.0
2.0 , 12.0 , 22.0

From reference data
-------------------

Isopy comes with a number of reference data sets. These can be retived using the :func:`get_reference_values` function.
If we call :meth:`ReferenceDict.get` with a list it will automatically return an :ref:`IsopyArray <isopyarray>`.

>>> rel_abu = isopy.get_reference_values('best isotope fraction')
>>> a = ref_val.get(['102Pd', '104Pd', '105Pd', '106Pd', '108Pd', '110Pd'])
>>> print(a)
102Pd  , 104Pd  , 105Pd  , 106Pd  , 108Pd  , 110Pd
0.0102 , 0.1114 , 0.2233 , 0.2733 , 0.2646 , 0.1172


Working with arrays
===================
Isopy arrays can be used much like normal arrays can be

>>> a = isopy.array({'99ru': [1,2,3,4,5], '105Pd': [2,4,8,10,12], '111Cd': [3,6,9,12,15]})
>>> print( a / 5 )
99Ru , 105Pd , 111Cd
0.2  , 0.4   , 0.6
0.4  , 0.8   , 1.2
0.6  , 1.6   , 1.8
0.8  , 2.0   , 2.4
1.0  , 2.4   , 3.0

If we attempt an operation with another array, that is not an isopy array, the operation will always be
performed for each column. This means the size of the other array has to be compatible with the number of rows in the
array.

>>> print (a + [0.1, 0.2, 0.3, 0.4, 0.5])
99Ru , 105Pd , 111Cd
1.1  , 2.1   , 3.1
2.2  , 4.2   , 6.2
3.3  , 8.3   , 9.3
4.4  , 10.4  , 12.4
5.5  , 12.5  , 15.5

If the array has only one row it will be expanded to the dimensions of the other array

>>> b = isopy.array(1, ['99ru', '105pd', '111cd']) #All values are 0
>>> print (b + [0.1, 0.2, 0.3, 0.4, 0.5])
99Ru , 105Pd , 111Cd
0.1  , 0.1   , 0.1
0.2  , 0.2   , 0.2
0.3  , 0.3   , 0.3
0.4  , 0.4   , 0.4
0.5  , 0.5   , 0.5

IsopyArrays and IsopyArrays
---------------------------
The same rules applies when we attempt an operation with another isopy array

>>> a = isopy.array({'99ru': [1,2,3,4,5], '105Pd': [2,4,8,10,12], '111Cd': [3,6,9,12,15]})
>>> b = isopy.array(1, ['99ru', '105pd', '111cd']) #All values are 0
>>> b2 = b + 0.5 #Set all values in b to 0.5
>>> print ( a + b2 )
99Ru , 105Pd , 111Cd
1.5  , 2.5   , 3.5
2.5  , 4.5   , 6.5
3.5  , 8.5   , 9.5
4.5  , 10.5  , 12.5
5.5  , 12.5  , 15.5
>>> b2 = b + [0.1, 0.2, 0.3, 0.4, 0.5]
99Ru , 105Pd , 111Cd
1.1  , 2.1   , 3.1
2.2  , 4.2   , 6.2
3.3  , 8.3   , 9.3
4.4  , 10.4  , 12.4
5.5  , 12.5  , 15.5

The returned array will always perform the operation on all keys present in at least one of the inputs. If a column is
not present in one array it will perform the operation using a default value of ``np.nan`` for that array. This will
in most instances mean that the this column in the returned array  will have the value ``np.nan``.

>>> c = isopy.array(1, ['99ru', '111cd']) + 0.5
>>> print (a + c)
99Ru , 105Pd , 111Cd
1.5  , nan   , 3.5
2.5  , nan   , 6.5
3.5  , nan   , 9.5
4.5  , nan   , 12.5
5.5  , nan   , 15.5

We can change the default value by calling the numpy function for the operation and passing an additional
*default_value* argument to the function.

>>> print (np.add(a, c, default_value=0)
99Ru , 105Pd , 111Cd
1.5  , 2.0   , 3.5
2.5  , 4.0   , 6.5
3.5  , 8.0   , 9.5
4.5  , 10.0  , 12.5
5.5  , 12.0  , 15.5

Passing additional keyword arguments is unfortunately not supported by some numpy functions, notably ``np.append()`` and
``np.concatenate()``.

IsopyArrays and dictionaries
----------------------------
One special feature of isopy arrays is that if we attempt an operation with a dictionary, such as that returned by
:func:`get_refeence_values` it will lookup the value for each column key in the dictionary

>>> rel_abu = isopy.get_reference_values('best isotope fraction')
>>> pd = isopy.array(1, ['102Pd', '104Pd', '105Pd', '106Pd', '108Pd', '110Pd'])
>>> print( pd ) #Array is automatically filled with 0 values
102Pd , 104Pd , 105Pd , 106Pd , 108Pd , 110Pd
0.0   , 0.0   , 0.0   , 0.0   , 0.0   , 0.0
>>> pd = pd + rel_abu
>>> print ( pd )
102Pd  , 104Pd  , 105Pd  , 106Pd  , 108Pd  , 110Pd
0.0102 , 0.1114 , 0.2233 , 0.2733 , 0.2646 , 0.1172

:class:`ReferenceDict`, returned by :func:`get_refeence_values`, has a special feature that if you try to lookup a
ratio value using :meth:`ReferenceDict.get` and that ratio is not present in the dictionary it will calculate the
value if the numerator and denominator values are present in the dictionary.

>>> '108Pd/105Pd' in rel_abu
False
>>> rel_abu.get('108Pd/105Pd')
1.1849529780564263

This would also work for the example above

>>> pd_rat = isopy.array(1, ['102Pd/105Pd', '104Pd/105Pd', '105Pd/105Pd', '106Pd/105Pd', '108Pd/105Pd', '110Pd/105Pd'])
>>> print( pd_rat + rel_abu)
102Pd/105Pd          , 104Pd/105Pd         , 105Pd/105Pd , 106Pd/105Pd        , 108Pd/105Pd        , 110Pd/105Pd
0.045678459471562925 , 0.49888042991491266 , 1.0         , 1.2239140170174652 , 1.1849529780564263 , 0.5248544558889386

Numpy functions
---------------
Most numpy functions will work on :ref:`IsopyArrays` out of the box

>>> a = isopy.array({'99ru': [1,2,3,4,5], '105Pd': [2,4,8,10,12], '111Cd': [3,6,9,12,15]})
>>> print( np.sqrt(a) )
99Ru               , 105Pd              , 111Cd
1.0                , 1.4142135623730951 , 1.7320508075688772
1.4142135623730951 , 2.0                , 2.449489742783178
1.7320508075688772 , 2.8284271247461903 , 3.0
2.0                , 3.1622776601683795 , 3.4641016151377544
2.23606797749979   , 3.4641016151377544 , 3.872983346207417
>>> print( np.log(a) )
99Ru               , 105Pd              , 111Cd
0.0                , 0.6931471805599453 , 1.0986122886681098
0.6931471805599453 , 1.3862943611198906 , 1.791759469228055
1.0986122886681098 , 2.0794415416798357 , 2.1972245773362196
1.3862943611198906 , 2.302585092994046  , 2.4849066497880004
1.6094379124341003 , 2.4849066497880004 , 2.70805020110221

It is common to use ``import numpy as np`` hence the abbreviation used above and hence forth.

Functions that have the *axis* parameter will behave differently for :ref:`IsopyArrays` than for other types input. If
the *axis* parameter is not given it will default to 0 (As opposed to None). This means the operation will be performed
on each column in the array. If you wish the perform the operation over the entire array then you need to explicitly
pass ``axis=None`` to the function. If you wish to perform the operation on each row in the table you must
pass ``axis=1`` to the function.

>>> print( np.mean(a) ) # same as np.mean(a, axis=0)
99Ru , 105Pd , 111Cd
3.0  , 7.2   , 9.0
>>> print( np.mean(a, axis=None) )
6.4
>>> print( np.mean(a, axis=1) )
[ 2.          4.          6.66666667  8.66666667 10.66666667]


Lets start by creating an array containing the relative isotope abundance of each Pd isotope using the
reference data set *best isotope fraction*





>>> a = isopy.array([0.1, 0.1, 1.0, 10], 'Ge70, ge74, pd105, pd108')
>>> b = isopy.array([5.0, 2.0, 1.0, 1.0], 'Ge70, ge74, pd105, pd108')

>>> key/ key = rat

>>> steps = np.linspace(0,1,10)

>>> c = a * steps + b * (1 - steps)


TODO np.mean, isopy.sd, isopy.se, isopy.median

TODO isobari

>>> rel_val = isopy.get_reference_values('best isotope fraction')
>>> pd = ref_val.get(ref_val.isotope_keys(element_symbol = 'pd'))
>>> ru = ref_val.get(ref_val.isotope_keys(element_symbol = 'ru'))

>>> pd2 = np.array(pd, pd.keys().mass_numbers())
>>> ru2 = np.array(ru, ru.keys().mass_numbers())

>>> all = pd2 + ru2

>>> all = np.add(pd2, ru2, default_value=0)



Lets start by creating an array of Pd isotopes. Isopy comes with
Interferences

multipliing etc
copying parts
adding two arrays together that contain isotopes to do a mixing

add mass isobaric interferences

lets do this ione next


Toolboxes
=========
Isopy contains a number of toolboxes that contain a couple of useful functions for processing data.


