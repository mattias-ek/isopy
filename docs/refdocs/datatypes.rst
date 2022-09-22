Isopy Data Types
================

.. currentmodule:: isopy

Key strings
-----------

Key strings should be created using :py:func:`isopy.keystring` and :py:func:`isopy.askeystring` functions and not
by invoking the key string objects directly.

Key strings are a subclass of :py:class:`str` and therefore contains all the method that a python string does.
Unless specifically noted below these methods will return a :class:`str` rather than a key string.

keystring
+++++++++
.. autofunction:: keystring

askeystring
+++++++++++
.. autofunction:: askeystring

iskeystring
+++++++++++
.. autofunction:: iskeystring


MassKeyString
+++++++++++++
.. autoclass:: isopy.core.MassKeyString
    :member-order: bysource
    :members: str


ElementKeyString
++++++++++++++++
.. autoclass:: isopy.core.ElementKeyString
    :member-order: bysource
    :members: str

IsotopeKeyString
++++++++++++++++
.. autoclass:: isopy.core.IsotopeKeyString
    :member-order: bysource
    :members: str

MoleculeKeyString
+++++++++++++++++
.. autoclass:: isopy.core.MoleculeKeyString
    :member-order: bysource
    :members: str

RatioKeyString
++++++++++++++
.. autoclass:: isopy.core.RatioKeyString
    :member-order: bysource
    :members: str

GeneralKeyString
++++++++++++++++
.. autoclass:: isopy.core.GeneralKeyString
    :member-order: bysource
    :members: str

Key list
--------

Key lists should be created using :py:func:`isopy.keylist` and :py:func:`isopy.askeylist` functions and not
by invoking the key string objects directly.

Key strings are a subclass of :py:class:`tuple` and therefore contains all the method that a normal tuple does.
Only those method which behave differently from :py:class:`tuple` are documented here.

keylist
+++++++
.. autofunction:: keylist

askeylist
+++++++++
.. autofunction:: askeylist

iskeylist
+++++++++
.. autofunction:: iskeylist

IsopyKeyList
++++++++++++
.. autoclass:: isopy.core.IsopyKeyList
    :member-order: bysource
    :members: filter, sorted, reversed, flatten, strlist, str

Arrays
------

Array must be created using one of the functions below. They can not be created by initialising
:py:class:`isopy.core.IsopyArray` directly.

Isopy arrays are subclass of a :py:class:`numpy.ndarray` and therefore contains all the methods and attributes that a
normal numpy ndarray does. However, these may **not** work as expected and caution is advised when
using attributes/methods not described in :ref:`IsopyArray`.


array
+++++
.. autofunction:: array

asarray
+++++++
.. autofunction:: asarray

asanyarray
++++++++++
.. autofunction:: asanyarray

zeros
+++++
.. autofunction:: zeros

ones
++++
.. autofunction:: ones

empty
+++++
.. autofunction:: empty

full
++++
.. autofunction:: full

random
++++++
.. autofunction:: random

isarray
+++++++
.. autofunction:: isarray

IsopyArray
++++++++++
.. autoclass:: isopy.core.IsopyArray
    :member-order: bysource
    :members: values, items, get, copy, filter, ratio, deratio, normalise, default, tabulate, to_array, to_refval, to_ndarray, to_dict, to_list, to_dataframe, to_clipboard, to_csv, to_xlsx




Dictionaries
------------
Isopy dictionaries must be created using one of the functions below. They can should not created by initialising
:py:class:`isopy.core.IsopyDict` or :py:class:`isopy.core.RefValDict` directly.

Is a subclass of :py:class:`dict` and therfore and contains all the methods that a normal dictionary does unless
otherwise noted. Only methods that behave differently from a normal dictionary are documented for :ref:`IsopyDict` and
:ref:`RefValDict`.

asdict
++++++
.. autofunction:: asdict

asrefval
++++++++
.. autofunction:: asrefval

isdict
++++++
.. autofunction:: isdict

isrefval
++++++++
.. autofunction:: isrefval

IsopyDict
+++++++++
.. autoclass:: isopy.core.IsopyDict
    :member-order: bysource
    :members: get, to_dict

RefValDict
++++++++++
.. autoclass:: isopy.core.RefValDict
    :member-order: bysource
    :members: get, tabulate, to_array, to_refval, to_ndarray, to_dict, to_list, to_dataframe, to_clipboard, to_csv, to_xlsx


Flavour
-------

asflavour
+++++++++
.. autofunction:: asflavour