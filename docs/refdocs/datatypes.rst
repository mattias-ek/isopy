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

ElementKeyString
++++++++++++++++
.. autoclass:: isopy.core.ElementKeyString
    :member-order: bysource

IsotopeKeyString
++++++++++++++++
.. autoclass:: isopy.core.IsotopeKeyString
    :member-order: bysource

MoleculeKeyString
+++++++++++++++++
.. autoclass:: isopy.core.MoleculeKeyString
    :member-order: bysource

RatioKeyString
++++++++++++++
.. autoclass:: isopy.core.RatioKeyString
    :member-order: bysource

GeneralKeyString
++++++++++++++++
.. autoclass:: isopy.core.GeneralKeyString
    :member-order: bysource

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
    :inherited-members:
    :member-order: bysource



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

RefValDict
++++++++++
.. autoclass:: isopy.core.RefValDict
    :inherited-members: dict
    :member-order: bysource


Flavour
-------

asflavour
+++++++++
.. autofunction:: asflavour