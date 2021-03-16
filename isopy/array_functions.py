import numpy as np
from numpy.lib.function_base import array_function_dispatch
from scipy import stats
import functools
from typing import Optional, Union, Literal
from . import core

__all__ = ['sd', 'nansd', 'se', 'nanse', 'mad', 'nanmad',
           'count_finite',
           'add', 'subtract', 'divide', 'multiply', 'power',
           'argmaxkey', 'argminkey']

##########################
### Dispatch functions ###
##########################
def _sd_dispatcher(a, axis=None, level = None):
    return (a,)

@array_function_dispatch(_sd_dispatcher)
def sd(a, axis = core.NotGiven, level = 1):
    """
    Compute the standard deviation along the specified axis for N-1 degrees of freedom.

    Shortcut for ``np.std(a, ddof = 1, axis=axis) * level``.

    If *level* is less than 1 it assumes *level* represents a percentage point
    and recalculates *level* as ``scipy.stats.norm.ppf(0.5 + level/2)``.

    Functions ``sd2`` to ``sd5`` are also avaliable where the last digits represents the
    *level* multiplication factor of the function. ``sd95`` is also exists which
    represents a *level* value of 0.95.

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
    >>> isopy.sd2(array) #same as sd(array, level=2)
    (row) , Ru  , Pd      , Cd
    None  , nan , 1.21106 , 2.83490
    >>> isopy.sd95(array) #same as sd(array, level=0.95)
    (row) , Ru  , Pd      , Cd
    None  , nan , 1.18682 , 2.77815

    See Also
    --------
    :func:`nansd`, `np.std <https://numpy.org/devdocs/reference/generated/numpy.std.html#numpy.std>`_
    """
    if level < 1:
        level = stats.norm.ppf(0.5 + level/2)

    if axis is core.NotGiven:
        return np.std(a, ddof=1) * level
    else:
        return np.std(a, ddof=1, axis=axis) * level

@array_function_dispatch(_sd_dispatcher)
def nansd(a, axis =core.NotGiven, level = 1):
    """
    Compute the standard deviation along the specified axis for N-1 degrees of freedom, while ignoring NaNs.

    Shortcut for ``np.nanstd(a, ddof = 1, axis=axis) * level``.

    If *level* is less than 1 it assumes *level* represents a percentage point
    and recalculates *level* as ``scipy.stats.norm.ppf(0.5 + level/2)``

    Functions ``nansd2`` to ``nansd5`` are also avaliable where the last digits represents the
    *level* multiplication factor of the function. ``nansd95`` is also exists which
    represents a *level* value of 0.95.

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
    >>> isopy.nansd2(array) #same as nansd(array, level=2)
    (row) , Ru      , Pd      , Cd
    None  , 1.11355 , 1.21106 , 2.83490
    >>> isopy.nansd95(array) #same as nansd(array, level=0.95)
    (row) , Ru      , Pd      , Cd
    None  , 1.09126 , 1.18682 , 2.77815

    See Also
    --------
    :func:`sd`, `numpy.nanstd <https://numpy.org/devdocs/reference/generated/numpy.nanstd.html#numpy.nanstd>`_
    """
    if level < 1:
        level = stats.norm.ppf(0.5 + level/2)

    if axis is core.NotGiven:
        return np.nanstd(a, ddof=1) * level
    else:
        return np.nanstd(a, ddof=1, axis=axis) * level

def _se_dispatcher(a, axis=None, level = None):
    return (a,)

@array_function_dispatch(_se_dispatcher)
def se(a, axis=core.NotGiven, level = 1):
    """
    Compute the standard error along the specified axis for N-1 degrees of freedom.

    Shortcut for ``scipy.stats.sem(a, axis=axis, nan_policy='propagate') * level``.

    If *level* is less than 1 it assumes *level* represents a percentage point
    and recalculates *level* as ``scipy.stats.norm.ppf(0.5 + level/2)``.

    Functions ``se2`` to ``se5`` are also avaliable where the last digits represents the
    *level* multiplication factor of the function. ``se95`` is also exists which
    represents a *level* value of 0.95.

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
    >>> isopy.se2(array) #same as se(array, level=2)
    (row) , Ru  , Pd      , Cd
    None  , nan , 0.60553 , 1.41745
    >>> isopy.se95(array) #same as se(array, level=0.95)
    (row) , Ru  , Pd      , Cd
    None  , nan , 0.59341 , 1.38908

    See Also
    --------
    :func:`nanse`, `scipy.stats.sem <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.sem.html?highlight=sem#scipy.stats.sem>`_
    """
    if level < 1:
        level = stats.norm.ppf(0.5 + level/2)

    if axis is core.NotGiven:
        return stats.sem(a, nan_policy='propagate') * level
    else:
        return stats.sem(a, axis=axis, nan_policy='propagate') * level

@array_function_dispatch(_se_dispatcher)
def nanse(a, axis=core.NotGiven, level = 1):
    """
    Compute the standard error along the specified axis for N-1 degrees of freedom, while ignoring
    NaNs.

    Shortcut for ``scipy.stats.sem(a, axis=axis, nan_policy='omit') * level``.

    If *level* is less than 1 it assumes *level* represents a percentage point
    and recalculates *level* as ``scipy.stats.norm.ppf(0.5 + level/2)``.

    Functions ``nanse2`` to ``nanse5`` are also avaliable where the last digits represents the
    *level* multiplication factor of the function. ``nanse95`` is also exists which
    represents a *level* value of 0.95.

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
    >>> isopy.nanse2(array) #same as nanse(array, level=2)
    (row) , Ru      , Pd      , Cd
    None  , 0.64291 , 0.60553 , 1.41745
    >>> isopy.nanse95(array) #same as nanse(array, level=0.95)
    (row) , Ru      , Pd      , Cd
    None  , 0.63004 , 0.59341 , 1.38908

    See Also
    --------
    :func:`se`, `scipy.stats.sem <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.sem.html?highlight=sem#scipy.stats.sem>`_
    """
    if level < 1:
        level = stats.norm.ppf(0.5 + level/2)

    if axis is core.NotGiven:
        return stats.sem(a, nan_policy='omit') * level
    else:
        return stats.sem(a, axis=axis, nan_policy='omit') *level

def _mad_dispatcher(a, axis=None, scale=None, level = None):
    return (a,)

@array_function_dispatch(_mad_dispatcher)
def mad(a, axis=core.NotGiven, scale= 'normal', level = 1):
    """
    Compute the median absolute deviation of the data along the given axis.

    Shortcut for ``scipy.stats.median_abs_deviation(a, axis, scale=scale, nan_policy='propagate') * level``.

    If *level* is less than 1 it assumes *level* represents a percentage point
    and recalculates *level* as ``scipy.stats.norm.ppf(0.5 + level/2)``.

    Functions ``mad2`` to ``mad5`` are also avaliable where the last digits represents the
    *level* multiplication factor of the function. ``mad95`` is also exists which
    represents a *level* value of 0.95.

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
    >>> isopy.mad2(array) #same as mad(array, level=2)
    (row) , Ru  , Pd      , Cd
    None  , nan , 1.33434 , 2.07564
    >>> isopy.mad95(array) #same as mad(array, level=0.95)
    (row) , Ru  , Pd      , Cd
    None  , nan , 1.30763 , 2.03409

    See Also
    --------
    :func:`nanmad`, `scipy.stats.median_abs_deviation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_abs_deviation.html#scipy.stats.median_abs_deviation>`_
    """
    if level < 1:
        level = stats.norm.ppf(0.5 + level/2)

    if axis is core.NotGiven:
        return stats.median_abs_deviation(a, scale=scale, nan_policy='propagate') * level
    else:
        return stats.median_abs_deviation(a, axis=axis, scale=scale, nan_policy='propagate') * level

@array_function_dispatch(_mad_dispatcher)
def nanmad(a, axis=core.NotGiven, scale = 'normal', level = 1):
    """
    Compute the median absolute deviation along the specified axis, while ignoring NaNs.

    Shortcut for ``scipy.stats.median_abs_deviation(a, axis, scale=scale, nan_policy='omit')``.

    If *level* is less than 1 it assumes *level* represents a percentage point
    and recalculates *level* as ``scipy.stats.norm.ppf(0.5 + level/2)``.

    Functions ``nanmad2`` to ``nanmad5`` are also avaliable where the last digits represents the
    *level* multiplication factor of the function. ``nanmad95`` is also exists which
    represents a *level* value of 0.95.

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
    >>> isopy.nanmad2(array) #same as nanmad(array, level=2)
    (row) , Ru      , Pd      , Cd
    None  , 1.18608 , 1.33434 , 2.07564
    >>> isopy.nanmad95(array) #same as nanmad(array, level=0.95)
    (row) , Ru      , Pd      , Cd
    None  , 1.16234 , 1.30763 , 2.03409

    See Also
    --------
    :func:`mad`, `scipy.stats.median_abs_deviation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_abs_deviation.html#scipy.stats.median_abs_deviation>`_
    """
    if level < 1:
        level = stats.norm.ppf(0.5 + level/2)

    if axis is core.NotGiven:
        return stats.median_abs_deviation(a, scale=scale, nan_policy='omit') * level
    else:
        return stats.median_abs_deviation(a, axis=axis, scale=scale, nan_policy = 'omit') * level

### Add multipliers

def leveller(func, level, name = None):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, level=level, **kwargs)

    if name is None:
        wrapper.__name__ = f'{wrapper.__name__}{level}'
    else:
        wrapper.__name__ = f'{wrapper.__name__}{name}'

    __all__.append(wrapper.__name__)

    return wrapper

sd2 = leveller(sd, 2)
sd3 = leveller(sd, 3)
sd4 = leveller(sd, 4)
sd5 = leveller(sd, 5)
sd95 = leveller(sd, 0.95, '95')
nansd2 = leveller(nansd, 2)
nansd3 = leveller(nansd, 3)
nansd4 = leveller(nansd, 4)
nansd5 = leveller(nansd, 5)
nansd95 = leveller(nansd, 0.95, '95')

se2 = leveller(se, 2)
se3 = leveller(se, 3)
se4 = leveller(se, 4)
se5 = leveller(se, 5)
se95 = leveller(se, 0.95, '95')
nanse2 = leveller(nanse, 2)
nanse3 = leveller(nanse, 3)
nanse4 = leveller(nanse, 4)
nanse5 = leveller(nanse, 5)
nanse95 = leveller(nanse, 0.95, '95')

mad2 = leveller(mad, 2)
mad3 = leveller(mad, 3)
mad4 = leveller(mad, 4)
mad5 = leveller(mad, 5)
mad95 = leveller(mad, 0.95, '95')
nanmad2 = leveller(nanmad, 2)
nanmad3 = leveller(nanmad, 3)
nanmad4 = leveller(nanmad, 4)
nanmad5 = leveller(nanmad, 5)
nanmad95 = leveller(nanmad, 0.95, '95')

def _count_dispatcher(a, axis=None):
    return (a,)

@array_function_dispatch(_count_dispatcher)
def count_finite(a, axis=core.NotGiven):
    """
    Count all values in array that are not NaN or Inf along the specified axis.

    Shortcut for ``np.count_nonzero(np.isfinite(a), axis=axis)``.

    Examples
    --------
    >>> array = isopy.array(ru = [np.nan, 1.1, 2.2, 1.8],
                            pd = [3.1, 3.8, 2.9, 4.2],
                            cd = [6.1, 5.8, 4.7, 8.1])
    >>> isopy.count_finite(array)
    (row) , Ru      , Pd      , Cd
    None  , 3.00000 , 4.00000 , 4.00000
    >>> isopy.count_finite(array, axis=1)
    array([2, 3, 3, 3], dtype=int64)
    """
    if axis is core.NotGiven:
        return np.count_nonzero(np.isfinite(a))
    else:
        return np.count_nonzero(np.isfinite(a), axis=axis)


#####################################
### Functions without dispatchers ###
#####################################
# These should  only be used with isopy arrays
def add(x1, x2, default_value=np.nan):
    """
    Add *x2* to *x1* substituting *default_value* for absent columns.

    This function requires that *x1* and/or *x2* are isopy arrays.

    Examples
    --------
    >>> array1 = isopy.ones(1, ('ru', 'pd'))
    >>> array2 = isopy.ones(1, ('pd', 'cd'))
    >>> isopy.add(array1, array2) #Same as array1 + array2
    (row) , Ru  , Pd      , Cd
    0     , nan , 2.00000 , nan
    >>> isopy.add(array1, array2, 0)
    (row) , Ru      , Pd      , Cd
    0     , 1.00000 , 2.00000 , 1.00000
    """
    return core.array_function(np.add, x1, x2, default_value=default_value)

def subtract(x1, x2, default_value=np.nan):
    """
    Subtract *x2* from *x1* substituting *default_value* for absent columns.

    This function requires that *x1* and/or *x2* are isopy arrays.

    Examples
    --------
    >>> array1 = isopy.ones(1, ('ru', 'pd'))
    >>> array2 = isopy.ones(1, ('pd', 'cd'))
    >>> isopy.subtract(array1, array2) #Same as array1 - array2
    (row) , Ru  , Pd      , Cd
    0     , nan , 0.00000 , nan
    >>> isopy.subtract(array1, array2, 0)
    (row) , Ru      , Pd      , Cd
    0     , 1.00000 , 0.00000 , -1.00000
    """
    return core.array_function(np.subtract, x1, x2, default_value=default_value)

def multiply(x1, x2, default_value=np.nan):
    """
    Multiply *x1* by *x2* substituting *default_value* for absent columns.

    This function requires that *x1* and/or *x2* are isopy arrays.

    Examples
    --------
    >>> array1 = isopy.ones(1, ('ru', 'pd')) * 2
    >>> array2 = isopy.ones(1, ('pd', 'cd')) * 5
    >>> isopy.multiply(array1, array2) #Same as array1 * array2
    (row) , Ru  , Pd       , Cd
    0     , nan , 10.00000 , nan
    >>> isopy.multiply(array1, array2, 1)
    (row) , Ru      , Pd       , Cd
    0     , 2.00000 , 10.00000 , 5.00000
    """
    return core.array_function(np.multiply, x1, x2, default_value=default_value)

def divide(x1, x2, default_value=np.nan):
    """
    Divide *x1* by *x2* substituting *default_value* for absent columns.

    This function requires that *x1* and/or *x2* are isopy arrays.

    Examples
    --------
    >>> array1 = isopy.ones(1, ('ru', 'pd')) * 2
    >>> array2 = isopy.ones(1, ('pd', 'cd')) * 5
    >>> isopy.divide(array1, array2) #Same as array1 / array2
    (row) , Ru  , Pd      , Cd
    0     , nan , 0.40000 , nan
    >>> isopy.divide(array1, array2, 1)
    (row) , Ru      , Pd      , Cd
    0     , 2.00000 , 0.40000 , 0.20000
    """
    return core.array_function(np.divide, x1, x2, default_value=default_value)

def power(x1, x2, default_value=np.nan):
    """
    Raise *x1* to the power of *x2* substituting *default_value* for absent columns.

    This function requires that *x1* and/or *x2* are isopy arrays.

    Examples
    --------
    >>> array1 = isopy.ones(1, ('ru', 'pd')) * 2
    >>> array2 = isopy.ones(1, ('pd', 'cd')) * 5
    >>> isopy.power(array1, array2) #Same as array1 ** array2
    (row) , Ru  , Pd       , Cd
    0     , nan , 32.00000 , nan
    >>> isopy.power(array1, array2, 1)
    (row) , Ru      , Pd       , Cd
    0     , 2.00000 , 32.00000 , 1.00000
    """
    return core.array_function(np.power, x1, x2, default_value=default_value)

def argmaxkey(a):
    """
    Return the name of the column where the largest value of the array is found.

    This function requires that *x1* and/or *x2* are isopy arrays.

    Examples
    --------
    >>> array = isopy.tb.make_ms_array('pd')
    >>> isopy.argmaxkey(array)
    IsotopeKeyString('106Pd')
    """
    array = core.asarray(a)
    return array.keys[np.argmax([np.max(array[key]) for key in array.keys()])]

def argminkey(a):
    """
    Return the name of the column where the smallest value of the array is found.

    This function requires that *a* is an isopy array.

    Examples
    --------
    >>> array = isopy.tb.make_ms_array('pd')
    >>> isopy.argminkey(array)
    IsotopeKeyString('102Pd')
    """
    array = core.asarray(a)
    return array.keys[np.argmin([np.min(array[key]) for key in array.keys()])]


