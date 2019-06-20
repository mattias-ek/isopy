isopy data types
****************

Some description.

IsopyItems
##########

.. autoclass:: isopy.MassInteger

.. autoclass:: isopy.ElementString

.. autoclass:: isopy.IsotopeString
    :members: __contains__

.. autoclass:: isopy.RatioString
    :members: __contains__

IsopyList's
#####

.. autoclass:: isopy.MassList
    :members: __eq__, __contains__, __getitem__, append, index, insert, remove, copy, filter

.. autoclass:: isopy.ElementList
    :members: __eq__, __contains__, __getitem__, append, index, insert, remove, copy, filter

.. autoclass:: isopy.IsotopeList
    :members: __eq__, __contains__, __getitem__, append, index, insert, remove, copy, get_element_symbols, get_mass_numbers, filter

.. autoclass:: isopy.RatioList
    :members: __eq__, __contains__, __getitem__, append, index, insert, remove, copy, get_numerators, get_denominators, has_common_denominator, get_common_denominator, filter

IsopyArray's
######

.. autoclass:: isopy.MassArray
    :members: filter

.. autoclass:: isopy.ElementArray
    :members: filter

.. autoclass:: isopy.IsotopeArray
    :members: filter

.. autoclass:: isopy.RatioArray
    :members: filter

IsopyDict
#########
.. autoclass:: isopy.IsopyDict
    :members: get, keys