Functions
*********

The following functions work the same as their NumPy counterparts but will also work on IsopyArray's. When the
``axis == 1`` the returned item will be an IsopyArray with the same keys as the input array. When used on IsopyArrays the axis option will default to 1.

Here is an example,

>>> array = isopy.IsotopeArray({'104Pd': [1,2,3], '105Pd': [4,5,6], '106Pd': [7,8,9]})
>>> print(array)
104Pd , 105Pd , 106Pd
1.0   , 4.0   , 7.0
2.0   , 5.0   , 8.0
3.0   , 6.0   , 9.0

When used on IsopyArrays the axis option will default to 1.

>>> print(isopy.min(array))
104Pd , 105Pd , 106Pd
1.0   , 4.0   , 7.0
>>> print(isopy.min(array, axis = 1))
104Pd , 105Pd , 106Pd
1.0   , 4.0   , 7.0

However, it will work with other axis values too,

>>> print(isopy.min(array, axis = None))
1.0
>>> print(isopy.min(array, axis = 0))
[1. 2. 3.]


Order statistics
----------------
.. automodule:: isopy
    :members: min, amax, amin, amax, nanmin, nanmax, ptp, percentile, nanpercentile, quantile, nanquantile
    :member-order: bysource

Averages and variances
----------------------
.. automodule:: isopy
    :members: median, average, mean, std, var, nanmedian, nanmean, nanstd, nanvar
    :member-order: bysource