import numpy as np
from numpy.lib.function_base import array_function_dispatch
from scipy import stats
import functools
from . import core

__all__ = ['sd', 'nansd', 'se', 'nanse', 'mad', 'nanmad',
           'nancount',
           'add', 'subtract', 'divide', 'multiply', 'power', 'arrayfunc',
           'keymax', 'keymin']

# TODO homoscedastic_sd(*args)

##########################
### Dispatch functions ###
##########################
def _sd_dispatcher(a, axis=None, level = None): #, where=None):
    return (a,)

@core.set_module('isopy')
@array_function_dispatch(_sd_dispatcher)
def sd(a, axis = None, level = 1): #, where = core.NotGiven):
    """
    Compute the standard deviation along the specified axis for N-1 degrees of freedom.

    Shortcut for ``np.std(a, ddof = 1, axis=axis) * level``.

    If *level* is less than 1 it assumes *level* represents a t-distribution percentage point
    and recalculates *level* using ``scipy.stats.t.ppf``.

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
    None  , nan , 1.9271 , 4.511

    See Also
    --------
    :func:`nansd`, `np.std <https://numpy.org/devdocs/reference/generated/numpy.std.html#numpy.std>`_
    """
    if level < 1:
        df = np.size(a, axis = axis) - 1
        level = stats.t.ppf(0.5 + level/2, df)

    return np.std(a, ddof=1, axis=axis) * level

#TODO
def homoscedastic_sd(*a, axis=1):
    raise NotImplementedError()
    ddof = len(a)
    a = np.concatenate(a) #wont work if a is not 1 dimensional
    return np.std(a, ddof = ddof)


@core.set_module('isopy')
@array_function_dispatch(_sd_dispatcher)
def nansd(a, axis = None, level = 1): #, where = core.NotGiven):
    """
    Compute the standard deviation along the specified axis for N-1 degrees of freedom, while ignoring NaNs.

    Shortcut for ``np.nanstd(a, ddof = 1, axis=axis) * level``.

    If *level* is less than 1 it assumes *level* represents a t-distribution percentage point
    and recalculates *level* using ``scipy.stats.t.ppf``.

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
    None  , 2.3956 , 1.9271 , 4.511

    See Also
    --------
    :func:`sd`, `numpy.nanstd <https://numpy.org/devdocs/reference/generated/numpy.nanstd.html#numpy.nanstd>`_
    """
    if level < 1:
        df = np.count_nonzero(np.invert(np.isnan(a)), axis=axis) - 1
        level = stats.t.ppf(0.5 + level / 2, df)

    return np.nanstd(a, ddof=1, axis=axis) * level

def _se_dispatcher(a, axis=None, level = None): #, where = None):
    return (a,)

@core.set_module('isopy')
@array_function_dispatch(_se_dispatcher)
def se(a, axis=None, level = 1): #, where = core.NotGiven):
    """
    Compute the standard error along the specified axis for N-1 degrees of freedom.

    Shortcut for ``scipy.stats.sem(a, axis=axis, nan_policy='propagate') * level``.

    If *level* is less than 1 it assumes *level* represents a t-distribution percentage point
    and recalculates *level* using ``scipy.stats.t.ppf``.

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
    None  , nan , 0.96353 , 2.2555

    See Also
    --------
    :func:`nanse`, `scipy.stats.sem <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.sem.html?highlight=sem#scipy.stats.sem>`_
    """
    if level < 1:
        df = np.size(a, axis=axis) - 1
        level = stats.t.ppf(0.5 + level / 2, df)

    return stats.sem(a, nan_policy='propagate', axis=axis) * level

@core.set_module('isopy')
@array_function_dispatch(_se_dispatcher)
def nanse(a, axis=None, level = 1): #, where = core.NotGiven):
    """
    Compute the standard error along the specified axis for N-1 degrees of freedom, while ignoring
    NaNs.

    Shortcut for ``scipy.stats.sem(a, axis=axis, nan_policy='omit') * level``.

    If *level* is less than 1 it assumes *level* represents a t-distribution percentage point
    and recalculates *level* using ``scipy.stats.t.ppf``.

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
    None  , 1.3831 , 0.96353 , 2.2555

    See Also
    --------
    :func:`se`, `scipy.stats.sem <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.sem.html?highlight=sem#scipy.stats.sem>`_
    """
    if level < 1:
        df = np.count_nonzero(np.invert(np.isnan(a)), axis=axis) - 1
        level = stats.t.ppf(0.5 + level / 2, df)

    return stats.sem(a, nan_policy='omit', axis=axis) * level

def _mad_dispatcher(a, axis=None, scale=None, level = None): #, where = None):
    return (a,)

@core.set_module('isopy')
@array_function_dispatch(_mad_dispatcher)
def mad(a, axis=None, scale= 'normal', level = 1): #,  where = core.NotGiven):
    """
    Compute the median absolute deviation of the data along the given axis.

    Shortcut for ``scipy.stats.median_abs_deviation(a, axis, scale=scale, nan_policy='propagate') * level``.

    If *level* is less than 1 it assumes *level* represents a t-distribution percentage point
    and recalculates *level* using ``scipy.stats.t.ppf``.

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
    None  , nan , 2.1232 , 3.3028

    See Also
    --------
    :func:`nanmad`, `scipy.stats.median_abs_deviation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_abs_deviation.html#scipy.stats.median_abs_deviation>`_
    """
    if level < 1:
        df = np.size(a, axis=axis) - 1
        level = stats.t.ppf(0.5 + level / 2, df)

    return stats.median_abs_deviation(a, scale=scale, nan_policy='propagate', axis=axis) * level

@core.set_module('isopy')
@array_function_dispatch(_mad_dispatcher)
def nanmad(a, axis=None, scale = 'normal', level = 1): #, where = core.NotGiven):
    """
    Compute the median absolute deviation along the specified axis, while ignoring NaNs.

    Shortcut for ``scipy.stats.median_abs_deviation(a, axis, scale=scale, nan_policy='omit')``.

    If *level* is less than 1 it assumes *level* represents a t-distribution percentage point
    and recalculates *level* using ``scipy.stats.t.ppf``.

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
    None  , 2.5516 , 2.1232 , 3.3028

    See Also
    --------
    :func:`mad`, `scipy.stats.median_abs_deviation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_abs_deviation.html#scipy.stats.median_abs_deviation>`_
    """
    if level < 1:
        df = np.count_nonzero(np.invert(np.isnan(a)), axis=axis) - 1
        level = stats.t.ppf(0.5 + level / 2, df)

    return stats.median_abs_deviation(a, scale=scale, nan_policy = 'omit', axis=axis) * level

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

def _count_dispatcher(a, axis=None): #, where = None):
    return (a,)


@core.set_module('isopy')
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


#####################################
### Functions without dispatchers ###
#####################################
# These should  only be used with isopy arrays
@core.set_module('isopy')
def add(x1, x2, default_value=np.nan, keys=None):
    """
    Add *x2* to *x1* substituting *default_value* for absent columns.

    *default_values* can either be a single value which will be used for
    all the inputs or a tuple containing a default value for each input.
    If *keys* are given then the operation if only performed for those
    keys.

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
    return core.call_array_function(np.add, x1, x2, default_value=default_value, keys=keys)

@core.set_module('isopy')
def subtract(x1, x2, default_value=np.nan, keys=None):
    """
    Subtract *x2* from *x1* substituting *default_value* for absent columns.

    *default_values* can either be a single value which will be used for
    all the inputs or a tuple containing a default value for each input.
    If *keys* are given then the operation if only performed for those
    keys.

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
    return core.call_array_function(np.subtract, x1, x2, default_value=default_value, keys=keys)

@core.set_module('isopy')
def multiply(x1, x2, default_value=np.nan, keys=None):
    """
    Multiply *x1* by *x2* substituting *default_value* for absent columns.

    *default_values* can either be a single value which will be used for
    all the inputs or a tuple containing a default value for each input.
    If *keys* are given then the operation if only performed for those
    keys.

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
    return core.call_array_function(np.multiply, x1, x2, default_value=default_value, keys=keys)

@core.set_module('isopy')
def divide(x1, x2, default_value=np.nan, keys=None):
    """
    Divide *x1* by *x2* substituting *default_value* for absent columns.

    *default_values* can either be a single value which will be used for
    all the inputs or a tuple containing a default value for each input.
    If *keys* are given then the operation if only performed for those
    keys.

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
    return core.call_array_function(np.divide, x1, x2, default_value=default_value, keys=keys)

@core.set_module('isopy')
def power(x1, x2, default_value=np.nan, keys=None):
    """
    Raise *x1* to the power of *x2* substituting *default_value* for absent columns.

    *default_values* can either be a single value which will be used for
    all the inputs or a tuple containing a value for each input.
    If *keys* are given then the operation if only performed for those
    keys.

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
    return core.call_array_function(np.power, x1, x2, default_value=default_value, keys=keys)

@core.set_module('isopy')
def arrayfunc(func, *inputs, default_value=np.nan, keys=None, **kwargs):
    """
    Call a numpy function *func* on the *inputs* values substituting *default_value* for absent columns.

    *default_values* can either be a single value which will be used for
    all the inputs or a tuple containing a value for each input.
    If *keys* are given then the operation if only performed for those
    keys.

    This function is useful for calling numpy functions with the isopy specific
    *default_value* and *keys* arguments or for array functions that dont employ
    numpys array function dispatch.

    Examples
    --------
    >>> array = isopy.random(100, keys=('ru', 'pd', 'cd')
    >>> isopy.arrayfunc(np.std, array, keys=('ru', 'cd'))
    (row) , Ru      , Cd
    None  , 0.90836 , 0.89753
    >>> isopy.arrayfunc(scipy.stats.sem, array) #Scipy functions do not support isopy arrays natively
    (row) , Ru      , Pd      , Cd
    None  , 0.09129 , 0.09430 , 0.09021
    """
    return core.call_array_function(func, *inputs, default_value=default_value, keys=keys, **kwargs)

@core.set_module('isopy')
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

@core.set_module('isopy')
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

