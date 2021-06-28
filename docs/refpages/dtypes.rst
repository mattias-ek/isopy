Data types
**********
.. currentmodule:: isopy

Documented here are the custom data types implemented by isopy for working with and manipulating different types of
geochemical data.

Key strings
-----------
See the tutorial for an introduction to key string. The following
classes/functions can be used to create isopy key strings:

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

Key lists
---------
See the tutorial for an introduction to key lists. The following
classes/functions can be used to create isopy key lists:

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
    MixedKeyList


Isopy arrays
------------
See the tutorial for an introduction to isopy arrays. The following
classes/functions can be used to create isopy arrays:

.. autosummary::
    :toctree: Arrays

    array
    asarray
    asanyarray
    array_from_csv
    array_from_xlsx
    array_from_clipboard

    :template: array_template.rst

    MassArray
    ElementArray
    IsotopeArray
    RatioArray
    GeneralArray
    MixedArray

    :template: autosummary/function.rst

    empty
    zeros
    ones
    full
    random

Isopy dictionaries
------------------
See the tutorial for an introduction to isopy dictionaries.
The following classes/functions can be used to create isopy dictionaries:

.. autosummary::
    :toctree: Dicts
    :template: dict_template.rst

    IsopyDict
    ScalarDict