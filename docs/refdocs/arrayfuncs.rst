Array Functions
===============

Isopy Functions
---------------
.. currentmodule:: isopy

sd
++
.. autofunction:: sd

nansd
+++++
.. autofunction:: nansd

se
++
.. autofunction:: se

nanse
+++++
.. autofunction:: nanse

mad
+++
.. autofunction:: mad

nanmad
++++++
.. autofunction:: nanmad

nancount
++++++++
.. autofunction:: nancount

add
+++
.. autofunction:: add

subtract
++++++++
.. autofunction:: subtract

multiply
++++++++
.. autofunction:: multiply

divide
++++++
.. autofunction:: divide

power
+++++
.. autofunction:: power

is_outlier
++++++++++
.. autofunction:: is_outlier

not_outlier
+++++++++++
.. autofunction:: not_outlier

lower_limit
+++++++++++
.. autofunction:: lower_limit

upper_limit
+++++++++++
.. autofunction:: upper_limit

arrayfunc
+++++++++
.. autofunction:: arrayfunc

rstack
++++++
.. autofunction:: rstack

cstack
++++++
.. autofunction:: cstack

concatenate
+++++++++++
.. autofunction:: concatenate

Numpy Functions
---------------

Isopy arrays, and reference value dictionaries, are compatible with a host of numpy functions. The functions
that have been tested with isopy arrays are avaliable within the isopy namespace. Other numpy functions **may**
still work with isopy arrays but should be used with caution. A warning will be raised the first time an unvetted
function is used.

The numpy functions have been tested and are avaliable within the isopy namespace are: :py:data:`sin()<numpy.sin>`,
:py:data:`cos()<numpy.cos>`, :py:data:`tan()<numpy.tan>`, :py:data:`arcsin()<numpy.arcsin>`,
:py:data:`arccos()<numpy.arccos>`, :py:data:`arctan()<numpy.arctan>`, :py:data:`degrees()<numpy.degrees>`,
:py:data:`isnan()<numpy.isnan>`, :py:data:`radians()<numpy.radians>`, :py:data:`deg2rad()<numpy.deg2rad>`,
:py:data:`rad2deg()<numpy.rad2deg>`, :py:data:`sinh()<numpy.sinh>`, :py:data:`cosh()<numpy.cosh>`,
:py:data:`tanh()<numpy.tanh>`, :py:data:`arcsinh()<numpy.arcsinh>`, :py:func:`arccosh()<numpy.arccosh>`,
:py:data:`rint()<numpy.rint>`, :py:data:`floor()<numpy.floor>`, :py:data:`ceil()<numpy.ceil>`,
:py:data:`trunc()<numpy.trunc>`, :py:data:`exp()<numpy.exp>`, :py:data:`expm1()<numpy.expm1>`,
:py:data:`exp2()<numpy.exp2>`, :py:data:`log()<numpy.log>`, :py:data:`log10()<numpy.log10>`,
:py:data:`log2()<numpy.log2>`, :py:data:`log1p()<numpy.log1p>`, :py:data:`reciprocal()<numpy.reciprocal>`,
:py:data:`positive()<numpy.positive>`, :py:data:`negative()<numpy.negative>`, :py:data:`sqrt()<numpy.sqrt>`,
:py:data:`cbrt()<numpy.cbrt>`, :py:data:`square()<numpy.square>`, :py:data:`fabs()<numpy.fabs>`,
:py:data:`sign()<numpy.sign>`, :py:data:`absolute()<numpy.absolute>`, :py:data:`abs()<numpy.absolute>`,
:py:func:`cumprod()<numpy.cumprod>`, :py:func:`cumsum()<numpy.cumsum>`, :py:func:`nancumprod()<numpy.nancumprod>`,
:py:func:`nancumsum()<numpy.nancumsum>`, :py:func:`prod()<numpy.prod>`, :py:func:`sum()<numpy.sum>`,
:py:func:`nanprod()<numpy.nanprod>`, :py:func:`nansum()<numpy.nansum>`, :py:func:`cumprod()<numpy.cumprod>`,
:py:func:`cumsum()<numpy.cumsum>`, :py:func:`nancumprod()<numpy.nancumprod>`, :py:func:`nancumsum()<numpy.nancumsum>`,
:py:func:`amin()<numpy.amin>`, :py:func:`amax()<numpy.amax>`, :py:func:`min()<numpy.amin>`, :py:func:`max()<numpy.amax>`,
:py:func:`nanmin()<numpy.nanmin>`, :py:func:`nanmax()<numpy.nanmax>`, :py:func:`ptp()<numpy.ptp>`,
:py:func:`median()<numpy.median>`, :py:func:`average()<numpy.average>`, :py:func:`mean()<numpy.mean>`,
:py:func:`std()<numpy.std>`, :py:func:`var()<numpy.var>`, :py:func:`nanmedian()<numpy.nanmedian>`,
:py:func:`nanmean()<numpy.nanmean>`, :py:func:`nanstd()<numpy.nanstd>`, :py:func:`nanvar()<numpy.nanvar>`,
:py:func:`nanmax()<numpy.nanmax>`, :py:func:`nanmin()<numpy.nanmin>`, :py:func:`all()<numpy.all>`,
:py:func:`any()<numpy.any>`, :py:func:`copyto()<numpy.copyto>`,

These functions in the isopy namespace have the additional functionality that you can include key filter arguments
to only perform the function over a subset of the data. E.g.

>>> a = isopy.array(ru = -1, pd = -2, cd = -3)
>>> isopy.abs(a)
(row)      Ru (f8)    Pd (f8)    Cd (f8)
-------  ---------  ---------  ---------
None       1.00000    2.00000    3.00000
IsopyNdarray(-1, flavour='element', default_value=nan)

>>> isopy.abs(a, key_eq = ['ru', 'pd'])
(row)      Ru (f8)    Pd (f8)
-------  ---------  ---------
None       1.00000    2.00000
IsopyNdarray(-1, flavour='element', default_value=nan)

>>> isopy.sum(a, axis=0, key_eq = ['ru', 'pd'])
-3

.. note:: You must specify the comparison type (eq, neq, lt, gt, le, ge) for the attribute you wish to apply
the filter to.