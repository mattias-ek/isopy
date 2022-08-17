import numpy as np
from numpy.lib.function_base import array_function_dispatch
from scipy import stats
import functools
from . import core
import warnings

__all__ = ['sd', 'nansd', 'se', 'nanse', 'mad', 'nanmad', 'nancount',
           'rstack', 'cstack', 'concatenate',
           'arrayfunc', 'keymax', 'keymin',
           'add', 'subtract', 'power', 'multiply', 'divide',
           'approved_numpy_functions']

__all__ += 'sd2 sd3 sd95 sd99 nansd2 nansd3 nansd95 nansd99'.split()
__all__ += 'se2 se3 se95 se99 nanse2 nanse3 nanse95 nanse99'.split()
__all__ += 'mad2 mad3 mad95 mad99 nanmad2 nanmad3 nanmad95 nanmad99'.split()

_min = min
_max = max
def calculate_ci(ci, df=None):
    if df is None:
        return stats.norm.ppf(0.5 + ci / 2)
    else:
        return stats.t.ppf(0.5 + ci / 2, df)

##########################
### Dispatch functions ###
##########################
def _sd_dispatcher(a, axis=None, *, ci = None, zscore=None): #, where=None):
    return (a,)

@array_function_dispatch(_sd_dispatcher)
def sd(a, axis = None, *, ci = None, zscore=None): #, where = core.NotGiven):
    """
    Compute the standard deviation along the specified axis for N-1 degrees of freedom.

    Shortcut for ``np.std(a, ddof = 1, axis=axis)``.

    If *ci* is given the result will be multiplied by the confidence interval *ci* of a
    t-distribution with N-1 degrees of freedom.

    If *zscore* is given then the result is multiplied by *zscore*.

    Functions ``sd2``, ``sd3``, ``sd95`` ``sd99`` are also avaliable for z-score
    values of 2 and 3 and for 95% and 99% confidence intervals respectivley.

    Examples
    --------
    >>> array = isopy.array(ru = [np.nan, 1.1, 2.2, 1.8],
                            pd = [3.1, 3.8, 2.9, 4.2],
                            cd = [6.1, 5.8, 4.7, 8.1])
    >>> isopy.sd(array)
    (row) , Ru  , Pd      , Cd
    None  , nan , 0.60553 , 1.41745
    >>> isopy.sd(array, axis=1)
    array([       nan, 2.35867194, 1.28970281, 3.17962262])
    >>> isopy.sd2(array) #same as sd(array, zscore=2)
    (row) , Ru  , Pd      , Cd
    None  , nan , 1.21106 , 2.83490
    >>> isopy.sd95(array) #same as sd(array, ci=0.95)
    (row) , Ru  , Pd      , Cd
    None  , nan , 1.9271 , 4.511

    See Also
    --------
    :func:`nansd`, `np.std <https://numpy.org/devdocs/reference/generated/numpy.std.html#numpy.std>`_
    """
    result = np.std(a, ddof=1, axis=axis)
    if ci is not None:
        df = np.count_nonzero(np.invert(np.isnan(a)), axis=axis) - 1
        return result * calculate_ci(ci, df)

    if zscore is not None:
        return result * zscore

    return result


@array_function_dispatch(_sd_dispatcher)
def nansd(a, axis = None, *, ci = None, zscore=None): #, where = core.NotGiven):
    """
    Compute the standard deviation along the specified axis for N-1 degrees of freedom, while ignoring NaNs.

    Shortcut for ``np.nanstd(a, ddof = 1, axis=axis)``.

    If *ci* is given the result will be multiplied by the confidence interval *ci* of a
    t-distribution with N-1 degrees of freedom.

    If *zscore* is given then the result is multiplied by *zscore*.

    Functions ``nansd2``, ``nansd3``, ``nansd95`` ``nansd99`` are also avaliable for z-score
    values of 2 and 3 and for 95% and 99% confidence intervals respectivley.


    Examples
    --------
    >>> array = isopy.array(ru = [np.nan, 1.1, 2.2, 1.8],
                            pd = [3.1, 3.8, 2.9, 4.2],
                            cd = [6.1, 5.8, 4.7, 8.1])
    >>> isopy.nansd(array)
    (row) , Ru      , Pd      , Cd
    None  , 0.55678 , 0.60553 , 1.41745
    >>> isopy.nansd(array, axis=1)
    array([2.12132034, 2.35867194, 1.28970281, 3.17962262])
    >>> isopy.nansd2(array) #same as nansd(array, zscore=2)
    (row) , Ru      , Pd      , Cd
    None  , 1.11355 , 1.21106 , 2.83490
    >>> isopy.nansd95(array) #same as nansd(array, ci=0.95)
    (row) , Ru      , Pd      , Cd
    None  , 2.3956 , 1.9271 , 4.511

    See Also
    --------
    :func:`sd`, `numpy.nanstd <https://numpy.org/devdocs/reference/generated/numpy.nanstd.html#numpy.nanstd>`_
    """
    result = np.nanstd(a, ddof=1, axis=axis)

    if ci is not None:
        df = np.count_nonzero(np.invert(np.isnan(a)), axis=axis) - 1
        return result * calculate_ci(ci, df)

    if zscore is not None:
        return result * zscore

    return result

def _se_dispatcher(a, axis=None, *, ci = None, zscore=None): #, where = None):
    return (a,)

@array_function_dispatch(_se_dispatcher)
def se(a, axis=None, *, ci = None, zscore=None): #, where = core.NotGiven):
    """
    Compute the standard error along the specified axis for N-1 degrees of freedom.

    Shortcut for ``scipy.stats.sem(a, axis=axis, nan_policy='propagate')``.

    If *ci* is given the result will be multiplied by the confidence interval *ci* of a
    t-distribution with N-1 degrees of freedom.

    If *zscore* is given then the result is multiplied by *zscore*.

    Functions ``se2``, ``se3``, ``se95`` ``se99`` are also avaliable for z-score
    values of 2 and 3 and for 95% and 99% confidence intervals respectivley.

    Examples
    --------
    >>> array = isopy.array(ru = [np.nan, 1.1, 2.2, 1.8],
                            pd = [3.1, 3.8, 2.9, 4.2],
                            cd = [6.1, 5.8, 4.7, 8.1])
    >>> isopy.se(array)
    (row) , Ru  , Pd      , Cd
    None  , nan , 0.30277 , 0.70873
    >>> isopy.se(array, axis=1)
    array([       nan, 1.36177988, 0.74461026, 1.83575598])
    >>> isopy.se2(array) #same as se(array, zscore=2)
    (row) , Ru  , Pd      , Cd
    None  , nan , 0.60553 , 1.41745
    >>> isopy.se95(array) #same as se(array, ci=0.95)
    (row) , Ru  , Pd      , Cd
    None  , nan , 0.96353 , 2.2555

    See Also
    --------
    :func:`nanse`, `scipy.stats.sem <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.sem.html?highlight=sem#scipy.stats.sem>`_
    """
    result =  stats.sem(a, nan_policy='propagate', axis=axis)

    if ci is not None:
        df = np.count_nonzero(np.invert(np.isnan(a)), axis=axis) - 1
        return result * calculate_ci(ci, df)

    if zscore is not None:
        return result * zscore

    return result

@array_function_dispatch(_se_dispatcher)
def nanse(a, axis=None, *, ci = None, zscore=None): #, where = core.NotGiven):
    """
    Compute the standard error along the specified axis for N-1 degrees of freedom, while ignoring
    NaNs.

    Shortcut for ``scipy.stats.sem(a, axis=axis, nan_policy='omit')``.

    If *ci* is given the result will be multiplied by the confidence interval *ci* of a
    t-distribution with N-1 degrees of freedom.

    If *zscore* is given then the result is multiplied by *zscore*.

    Functions ``nanse2``, ``nanse3``, ``nanse5`` ``nanse99`` are also avaliable for z-score
    values of 2 and 3 and for 95% and 99% confidence intervals respectivley.

    Examples
    --------
    >>> array = isopy.array(ru = [np.nan, 1.1, 2.2, 1.8],
                            pd = [3.1, 3.8, 2.9, 4.2],
                            cd = [6.1, 5.8, 4.7, 8.1])
    >>> isopy.nanse(array)
    (row) , Ru      , Pd      , Cd
    None  , 0.32146 , 0.30277 , 0.70873
    >>> isopy.nanse(array, axis=1) #returns a masked array
    masked_array(data=[1.4999999999999996, 1.3617798810543664,
                   0.7446102634562892, 1.835755975068582],
             mask=[False, False, False, False],
       fill_value=1e+20)
    >>> isopy.nanse2(array) #same as nanse(array, zscore=2)
    (row) , Ru      , Pd      , Cd
    None  , 0.64291 , 0.60553 , 1.41745
    >>> isopy.nanse95(array) #same as nanse(array, ci=0.95)
    (row) , Ru      , Pd      , Cd
    None  , 1.3831 , 0.96353 , 2.2555

    See Also
    --------
    :func:`se`, `scipy.stats.sem <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.sem.html?highlight=sem#scipy.stats.sem>`_
    """
    result = stats.sem(a, nan_policy='omit', axis=axis)

    if ci is not None:
        df = np.count_nonzero(np.invert(np.isnan(a)), axis=axis) - 1
        return result * calculate_ci(ci, df)

    if zscore is not None:
        return result * zscore

    return result

def _mad_dispatcher(a, axis=None, scale=None, *, ci = None, zscore=None): #, where = None):
    return (a,)

@array_function_dispatch(_mad_dispatcher)
def mad(a, axis=None, scale= 'normal', *, ci = None, zscore=None): #,  where = core.NotGiven):
    """
    Compute the median absolute deviation of the data along the given axis.

    Shortcut for ``scipy.stats.median_abs_deviation(a, axis, scale=scale, nan_policy='propagate')``.

    If *ci* is given the result will be multiplied by the confidence interval *ci* of a
    t-distribution with N-1 degrees of freedom.

    If *zscore* is given then the result is multiplied by *zscore*.

    Functions ``mad2``, ``mad3``, ``mad95`` ``mad99`` are also avaliable for z-score
    values of 2 and 3 and for 95% and 99% confidence intervals respectivley.

    Examples
    --------
    >>> array = isopy.array(ru = [np.nan, 1.1, 2.2, 1.8],
                            pd = [3.1, 3.8, 2.9, 4.2],
                            cd = [6.1, 5.8, 4.7, 8.1])
    >>> isopy.mad(array)
    (row) , Ru  , Pd      , Cd
    None  , nan , 0.66717 , 1.03782
    >>> isopy.mad(array, axis=1)
    array([       nan, 2.96520444, 1.03782155, 3.55824532])
    >>> isopy.mad2(array) #same as mad(array, zscore=2)
    (row) , Ru  , Pd      , Cd
    None  , nan , 1.33434 , 2.07564
    >>> isopy.mad95(array) #same as mad(array, ci=0.95)
    (row) , Ru  , Pd      , Cd
    None  , nan , 2.1232 , 3.3028

    See Also
    --------
    :func:`nanmad`, `scipy.stats.median_abs_deviation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_abs_deviation.html#scipy.stats.median_abs_deviation>`_
    """
    # function sometimes return a float value
    result = np.asarray( stats.median_abs_deviation(a, scale=scale, nan_policy='propagate', axis=axis) )

    if ci is not None:
        df = np.count_nonzero(np.invert(np.isnan(a)), axis=axis) - 1
        return result * calculate_ci(ci, df)

    if zscore is not None:
        return result * zscore

    # result is not always an array
    return result

@array_function_dispatch(_mad_dispatcher)
def nanmad(a, axis=None, scale = 'normal', *, ci = None, zscore=None): #, where = core.NotGiven):
    """
    Compute the median absolute deviation along the specified axis, while ignoring NaNs.

    Shortcut for ``scipy.stats.median_abs_deviation(a, axis, scale=scale, nan_policy='omit')``.

    If *ci* is given the result will be multiplied by the confidence interval *ci* of a
    t-distribution with N-1 degrees of freedom.

    If *zscore* is given then the result is multiplied by *zscore*.

    Functions ``nanmad2``, ``nanmad3``, ``nanmad95`` ``nanmad99`` are also avaliable for z-score
    values of 2 and 3 and for 95% and 99% confidence intervals respectivley.

    Examples
    --------
    >>> array = isopy.array(ru = [np.nan, 1.1, 2.2, 1.8],
                            pd = [3.1, 3.8, 2.9, 4.2],
                            cd = [6.1, 5.8, 4.7, 8.1])
    >>> isopy.nanmad(array)
    (row) , Ru      , Pd      , Cd
    None  , 0.59304 , 0.66717 , 1.03782
    >>> isopy.nanmad(array, axis=1)
    array([2.22390333, 2.96520444, 1.03782155, 3.55824532])
    >>> isopy.nanmad2(array) #same as nanmad(array, zscore=2)
    (row) , Ru      , Pd      , Cd
    None  , 1.18608 , 1.33434 , 2.07564
    >>> isopy.nanmad95(array) #same as nanmad(array, ci=0.95)
    (row) , Ru      , Pd      , Cd
    None  , 2.5516 , 2.1232 , 3.3028

    See Also
    --------
    :func:`mad`, `scipy.stats.median_abs_deviation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_abs_deviation.html#scipy.stats.median_abs_deviation>`_
    """
    # function sometimes return a float value
    result = np.asarray( stats.median_abs_deviation(a, scale=scale, nan_policy = 'omit', axis=axis) )

    if ci is not None:
        df = np.count_nonzero(np.invert(np.isnan(a)), axis=axis) - 1
        return result * calculate_ci(ci, df)

    if zscore is not None:
        return result * zscore

    return result

def _count_dispatcher(a, axis=None): #, where = None):
    return (a,)

@array_function_dispatch(_count_dispatcher)
def nancount(a, axis=None): #, where = core.NotGiven):
    """
    Count all values in array that are not NaN or Inf along the specified axis.

    Shortcut for ``np.count_nonzero(np.isfinite(a), axis=axis)``.

    Examples
    --------
    >>> array = isopy.array(ru = [np.nan, 1.1, 2.2, 1.8],
                            pd = [3.1, 3.8, 2.9, 4.2],
                            cd = [6.1, 5.8, 4.7, 8.1])
    >>> isopy.nancount(array)
    (row) , Ru      , Pd      , Cd
    None  , 3.00000 , 4.00000 , 4.00000
    >>> isopy.nancount(array, axis=1)
    array([2, 3, 3, 3], dtype=int64)
    """
    #if where is not core.NotGiven:
    #    a = a[where]

    return np.count_nonzero(np.isfinite(a), axis=axis)

for func in [sd, nansd, se, nanse, mad, nanmad, nancount]:
    core.APPROVED_FUNCTIONS.append(func)

### preset functions
# These dont need to be approved
sd2 = core.partial_func(sd, 'sd2', zscore=2)
sd3 = core.partial_func(sd, 'sd3', zscore=3)
sd95 = core.partial_func(sd, 'sd95', ci=0.95)
sd99 = core.partial_func(sd, 'sd99', ci=0.99)
nansd2 = core.partial_func(nansd, 'nansd2', zscore=2)
nansd3 = core.partial_func(nansd, 'nansd3', zscore=3)
nansd95 = core.partial_func(nansd, 'nansd95', ci=0.95)
nansd99 = core.partial_func(nansd, 'nansd99', ci=0.99)

se2 = core.partial_func(se, 'se2', zscore=2)
se3 = core.partial_func(se, 'se3', zscore=3)
se95 = core.partial_func(se, 'se95', ci=0.95)
se99 = core.partial_func(se, 'se99', ci=0.99)
nanse2 = core.partial_func(nanse, 'nanse2', zscore=2)
nanse3 = core.partial_func(nanse, 'nanse3', zscore=3)
nanse95 = core.partial_func(nanse, 'nanse95', ci=0.95)
nanse99 = core.partial_func(nanse, 'nanse99', ci=0.99)

mad2 = core.partial_func(mad, 'mad2', zscore=2)
mad3 = core.partial_func(mad, 'mad3', zscore=3)
mad95 = core.partial_func(mad, 'mad95', ci=0.95)
mad99 = core.partial_func(mad, 'mad99', ci=0.99)
nanmad2 = core.partial_func(nanmad, 'nanmad2', zscore=2)
nanmad3 = core.partial_func(nanmad, 'nanmad3', zscore=3)
nanmad95 = core.partial_func(nanmad, 'nanmad95', ci=0.95)
nanmad99 = core.partial_func(nanmad, 'nanmad99', ci=0.99)

#####################################
### Functions without dispatchers ###
#####################################
def arrayfunc(func, *inputs, keys=None, **kwargs):
    """
    Call a function *func* on each column in the *inputs*.

    Useful for functions that does not support the numpy __array_function__ interface, e.g. scipy functions.

    If *keys* are given then the operation if only performed for those
    keys.

    Examples
    --------
    >>> isopy.arrayfunc(scipy.stats.sem, array)
    (row) , Ru      , Pd      , Cd
    None  , 0.09129 , 0.09430 , 0.09021

    >>> array = isopy.random(100, keys=('ru', 'pd', 'cd')
    >>> isopy.arrayfunc(np.std, array, keys=('ru', 'cd'))
    (row) , Ru      , Cd
    None  , 0.90836 , 0.89753
    """
    return core.call_array_function(func, *inputs, keys=keys, **kwargs)

def rstack(*arrays, sort_keys=False):
    """
    Stack the rows of multiple arrays.

    If scalar value(s) are given then these values will be appended to each column in the returned array.

    Parameters
    ----------
    arrays
        Arrays to be joined together.

    Returns
    -------
    IsopyArray
        Array containing all the row data and all the columns keys found in *arrays*.

    See Also
    --------
    rstack, cstack
    """
    arrays = [core.asanyarray(a) for a in arrays]

    keys = core.keylist(*(a.dtype.names for a in arrays if isinstance(a, core.IsopyArray)),
                        ignore_duplicates=True, sort=sort_keys)
    arrays = [a.reshape(1) if a.ndim == 0 else a for a in arrays]

    for i, a in enumerate(arrays):
        if not isinstance(a, core.IsopyArray):
            arrays[i] = core.full(a.size, a, keys)

    result = [np.concatenate([a.get(key) for a in arrays]) for key in keys]
    dtype = [(key, result[i].dtype) for i, key in enumerate(keys.strlist())]
    return keys._view_array_(np.fromiter(zip(*result), dtype=dtype))

def cstack(*arrays, sort_keys=False):
    """
    Stack the columns of multiple arrays.

    An error will thrown if a column occurs in more than one array.

    Normal numpy broadcasting rules apply so when concatenating columns the shape of the arrays must be compatible.
    When array with a size of 1 is concatenated to an array of a different size it will be copied into every row
    of the new shape.

    Parameters
    ----------
    arrays
        Arrays to be joined together.

    Returns
    -------
    IsopyArray
        Array containing all the row data and all the columns keys found in *arrays*.

    See Also
    --------
    rstack, cstack
    """
    arrays = [core.asarray(a) for a in arrays]
    size = {a.size for a in arrays}
    ndim = {a.ndim for a in arrays}

    if not (len(size) == 1 or (1 in size and len(size) == 2)):
        raise ValueError('Arrays have incompatible sizes and cannot be stacked')
    if len(ndim) != 1:
        arrays = [array.reshape(1) if array.ndim == 0 else array for array in arrays]

    keys = core.keylist(*(a.keys for a in arrays), allow_duplicates=False, sort=sort_keys)

    result = {}
    size = _max(size) * _max(ndim) or None
    for key in keys:
        for a in arrays:
            if key in a.keys():
                result[key] = np.full(size, a.get(key))

    return core.array(result)

def concatenate(*arrays, axis=0):
    """
    Join arrays together along specified axis.

    If *arrays* contains at least one isopy array the input is forwarded to *rstack*/*cstack*. Otherwise the
    input is forwarded to *np.concatenate*.

    Parameters
    ----------
    arrays
        Arrays to be joined together.
    axis
        If 0 *rstack* is used. If 1 *cstack* is used.

    Returns
    -------
    IsopyArray
        Array containing all the row data and all the columns keys found in *arrays*.

    See Also
    --------
    rstack, cstack
    """
    arrays = [core.asanyarray(a) for a in arrays]

    if True not in [isinstance(a, core.IsopyArray) for a in arrays]:
        return np.concatenate([a for a in arrays if a is not None], axis=axis)

    if axis == 0 or axis is None:
        return rstack(*arrays)
    elif axis == 1:
        return cstack(*arrays)
    else:
        raise np.AxisError(axis, 1, 'concatenate only accepts axis values of 0 or 1')

# TODO dont deprecate. Inlude error propagation
def dual_arrayfunc(func):
    # With the new array.default() feature these functions are not necessary any more. In the future
    # the numpy functions will be used directly.
    def func_wrapper(a1, a2, default_value=None, keys=None):
        if default_value is not None:
            if type(default_value) is not tuple:
                default_value = (default_value, default_value)

            if isinstance(a1, core.IsopyArray):
                a1 = a1.default(default_value[0])
            elif isinstance(a1, dict) and type(a1) is not core.RefValDict:
                a1 = core.RefValDict(a1, default_value = default_value[0])

            if isinstance(a2, core.IsopyArray):
                a2 = a2.default(default_value[1])
            elif isinstance(a2, dict) and type(a2) is not core.RefValDict:
                a2 = core.RefValDict(a2, default_value = default_value[1])

        return arrayfunc(func, a1, a2, keys=keys)
    return func_wrapper

add = dual_arrayfunc(np.add)
subtract = dual_arrayfunc(np.subtract)
multiply = dual_arrayfunc(np.multiply)
divide = dual_arrayfunc(np.divide)
power = dual_arrayfunc(np.power)

@core.append_preset_docstring
@core.add_preset('abs', abs = True)
def keymax(a, evalfunc = np.nanmedian, abs=False):
    """
    Return the name of the column where the largest value returned by *evalfunc* is found.

    If *abs* is ``True`` the absolute value is used.

    This function requires that *a* is an isopy array.

    Examples
    --------
    >>> array = isopy.tb.make_ms_array('pd')
    >>> isopy.keymax(array)
    IsotopeKeyString('106Pd')
    """
    array = core.asarray(a)
    if abs:
        key_index = np.nanargmax([evalfunc(np.abs(v)) for v in array.values()])
    else:
        key_index = np.nanargmax([evalfunc(v) for v in array.values()])
    return array.keys[key_index]

@core.append_preset_docstring
@core.add_preset('abs', abs = True)
def keymin(a, evalfunc= np.nanmedian, abs = False):
    """
    Return the name of the column where the smallest value returned by *evalfunc* is found.

    If *abs* is ``True`` the absolute value is used.

    This function requires that *a* is an isopy array.

    Examples
    --------
    >>> array = isopy.tb.make_ms_array('pd')
    >>> isopy.keymin(array)
    IsotopeKeyString('102Pd')
    """
    array = core.asarray(a)
    if abs:
        key_index = np.nanargmin([evalfunc(np.abs(v)) for v in array.values()])
    else:
        key_index = np.nanargmin([evalfunc(v) for v in array.values()])
    return array.keys[key_index]

################################
### Approved numpy functions ###
################################

np_elementwise = [np.sin, np.cos, np.tan, np.arcsin, np.arccos, np.arctan, np.degrees, np.isnan,
                  np.radians, np.deg2rad, np.rad2deg, np.sinh, np.cosh, np.tanh, np.arcsinh,
                  np.arccosh, np.arctanh,
                  np.rint, np.floor, np.ceil, np.trunc, np.exp, np.expm1, np.exp2, np.log,
                  np.log10, np.log2,
                  np.log1p, np.reciprocal, np.positive, np.negative, np.sqrt, np.cbrt, np.square,
                  np.fabs, np.sign,
                  np.absolute, np.abs]
np_cumulative = [np.cumprod, np.cumsum, np.nancumprod, np.nancumsum]
np_reducing = [np.prod, np.sum, np.nanprod, np.nansum, np.cumprod, np.cumsum, np.nancumprod, np.nancumsum,
               np.amin, np.amax, np.nanmin, np.nanmax, np.ptp, np.median, np.average, np.mean, np.average,
               np.std, np.var, np.nanmedian, np.nanmean, np.nanstd, np.nanvar, np.nanmax, np.nanmin,
               np.all, np.any]
np_special = [np.copyto, np.average]
np_dual = [np.add, np.subtract, np.divide, np.multiply, np.power]


for functions in [np_elementwise, np_cumulative, np_reducing, np_special, np_dual]:
    for func in functions:
        core.APPROVED_FUNCTIONS.append(func)
        if func.__name__ not in __all__:
            __all__.append(func.__name__)
            globals()[func.__name__] = func
        if func.__name__ == 'amin':
            __all__.append('min')
            globals()['min'] = func
        if func.__name__ == 'amax':
            __all__.append('max')
            globals()['max'] = func

def approved_numpy_functions(format ='name', delimiter =', '):
    """
    Return a string containing the names of all the numpy functions supported by isopy arrays.

    With *format* you can specify the format of the string for each function. Avaliable keywords
    are ``{name}`` and ``{link}`` for the name and a link the numpy documentation web page for a
    function. Avaliable presets are ``"name"``, ``"link"``, ``"rst"`` and ``"markdown"`` for just the name,
    just the link, reStructured text hyper referenced link or a markdown hyper referenced link.

    You can specify the delimiter used to seperate items in the list using the *delimiter* argument.
    If *delimiter* is ``None`` a python list is returned.
    """
    if format == 'name': format = '{name}'
    if format == 'link': format = '{link}'
    if format == 'rst': format = '`{name} <{link}>`_'
    if format == 'markdown': format = '[{name}]({link})'

    strings = []
    for functions in [np_elementwise, np_cumulative, np_reducing, np_special, np_dual]:
        for func in functions:
            name = func.__name__
            link = f'https://numpy.org/doc/stable/reference/generated/numpy.{name}.html'
            strings.append(format.format(name = name, link=link))
    if delimiter is None:
        return strings
    else:
        return delimiter.join(strings)