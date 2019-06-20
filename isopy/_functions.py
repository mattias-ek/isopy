from isopy._dtypes import _IsopyArray
import numpy as np

##################
### Statistics ###
##################

#https://docs.scipy.org/doc/numpy/reference/routines.statistics.html

#Order statistics
def _numpy_function(func, arg_keys, a, args, kwargs):
    arg_keys = [x.strip() for x  in arg_keys.split(',')]
    for i in range(len(args)): kwargs[arg_keys[i]] = args[i]

    if isinstance(a, _IsopyArray): return a._array_function_(func, **kwargs)
    else: return func(a, **kwargs)


def amin(a, *args, **kwargs):
    """
    amin(a[, axis, out, keepdims, initial])

    Return the minimum of an array or minimum along an axis.
    See
    `NumPy documentation <https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.amin.html#numpy.amin>`_.
    """
    return _numpy_function(np.amin, 'axis,out,keepdims,initial', a, args, kwargs)

def amax(a, *args, **kwargs):
    """
    amax(a[, axis, out, keepdims, initial])

    Return the maximum of an array or maximum along an axis.
    See
    `NumPy documentation <https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.amax.html#numpy.amax>`_.
    """
    return _numpy_function(np.amax, 'axis,out,keepdims,initial', a, args, kwargs)

def min(a, *args, **kwargs):
    """
    min(a[, axis, out, keepdims, initial])

    Return the maximum of an array or maximum along an axis.
    See
    `NumPy documentation <https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.amin.html#numpy.amin>`_.
    """
    return _numpy_function(np.min, 'axis,out,keepdims,initial', a, args, kwargs)

def max(a, *args, **kwargs):
    """
    max(a[, axis, out, keepdims, initial])

    Return the maximum of an array or maximum along an axis.
    See
    `NumPy documentation <https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.amax.html#numpy.amax>`_.
    """
    return _numpy_function(np.amax, 'axis,out,keepdims,initial', a, args, kwargs)

def ptp(a, *args, **kwargs):
    """
    ptp(a[, axis, out, keepdims])

    Range of values (maximum - minimum) along an axis.
    See
    `NumPy documentation <https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.ptp.html#numpy.ptp>`_.
    """
    return _numpy_function(np.ptp, 'axis,out,keepdims', a, args, kwargs)

def percentile(a, *args, **kwargs):
    """
    percentile(a, q[, axis, out, overwrite_input, interpolation, keepdims])

    Compute the qth percentile of the data along the specified axis.
    See
    `NumPy documentation <https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.percentile.html#numpy.percentile>`_.
    """
    return _numpy_function(np.percentile, 'q, axis, out, overwrite_input, interpolation, keepdims', a, args, kwargs)

def quantile(a, *args, **kwargs):
    """
    quantile(a, q[, axis, out, overwrite_input, interpolation, keepdims])

    Compute the `q`th quantile of the data along the specified axis.
    See
    `NumPy documentation <https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.quantile.html#numpy.quantile>`_.
    """
    return _numpy_function(np.quantile, 'q, axis, out, overwrite_input, interpolation, keepdims', a, args, kwargs)

def nanmin(a, *args, **kwargs):
    """
    nanmin(a[, axis, out, keepdims])

    Return minimum of an array or minimum along an axis, ignoring any NaNs.
    See
    `NumPy documentation <https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.nanmin.html#numpy.nanmin>`_.
    """
    return _numpy_function(np.nanmin, 'axis,out,keepdims', a, args, kwargs)

def nanmax(a, *args, **kwargs):
    """
    nanmax(a[, axis, out, keepdims])

    Return the maximum of an array or maximum along an axis, ignoring any NaNs.
    See
    `NumPy documentation <https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.nanmax.html#numpy.nanmax>`_.
    """
    return _numpy_function(np.nanmax, 'axis,out,keepdims', a, args, kwargs)

def nanpercentile(a, *args, **kwargs):
    """
    nanpercentile(a, q[, axis, out, overwrite_input, interpolation, keepdims])

    Compute the qth percentile of the data along the specified axis, while ignoring nan values.
    See
    `NumPy documentation <https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.nanpercentile.html#numpy.nanpercentile>`_.
    """
    return _numpy_function(np.nanpercentile, 'q, axis, out, overwrite_input, interpolation, keepdims', a, args, kwargs)

def nanquantile(a, *args, **kwargs):
    """
    nanquantile(a, q[, axis, out, overwrite_input, interpolation, keepdims])

    Compute the qth quantile of the data along the specified axis, while ignoring nan values.
    See
    `NumPy documentation <https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.nanquantile.html#numpy.nanquantile>`_.
    """
    return _numpy_function(np.nanquantile, 'q, axis, out, overwrite_input, interpolation, keepdims', a, args, kwargs)

#Averages and variances
def median(a, *args, **kwargs):
    """
    median(a[, axis, out, overwrite_input, keepdims])

    Compute the median along the specified axis.
    See
    `NumPy documentation <https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.median.html#numpy.median>`_.
    """
    return _numpy_function(np.median, 'axis, out, overwrite_input, keepdims', a, args, kwargs)

def average(a, *args, **kwargs):
    """
    average(a[, axis, weights, returned])

    Compute the weighted average along the specified axis.
    See
    `NumPy documentation <https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.average.html#numpy.average>`_.
    """
    return _numpy_function(np.average, 'axis, weights, returned', a, args, kwargs)

def mean(a, *args, **kwargs):
    """
    mean(a[, axis, dtype, out, keepdims])

    Compute the arithmetic mean along the specified axis.
    See
    `NumPy documentation <https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.mean.html#numpy.mean>`_.
    """
    return _numpy_function(np.mean, 'axis, dtype, out, keepdims', a, args, kwargs)

def std(a, *args, **kwargs):
    """
    std(a[, axis, dtype, out, ddof, keepdims])

    Compute the standard deviation along the specified axis.
    See
    `NumPy documentation <https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.std.html#numpy.std>`_.
    """
    return _numpy_function(np.std, 'axis, dtype, out, ddof, keepdims', a, args, kwargs)

def var(a, *args, **kwargs):
    """
    var(a[, axis, dtype, out, ddof, keepdims])

    Compute the variance along the specified axis.
    See
    `NumPy documentation <https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.var.html#numpy.var>`_.
    """
    return _numpy_function(np.var, 'axis, dtype, out, ddof, keepdims', a, args, kwargs)

def nanmedian(a, *args, **kwargs):
    """
    nanmedian(a[, axis, out, overwrite_input, keepdims])

    Compute the median along the specified axis, while ignoring NaNs.
    See
    `NumPy documentation <https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.nanmedian.html#numpy.nanmedian>`_.
    """
    return _numpy_function(np.nanmedian, 'axis, out, overwrite_input, keepdims', a, args, kwargs)

def nanmean(a, *args, **kwargs):
    """
    nanmean(a[, axis, dtype, out, keepdims])

    Compute the arithmetic mean along the specified axis, ignoring NaNs.
    See
    `NumPy documentation <https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.nanmean.html#numpy.nanmean>`_.
    """
    return _numpy_function(np.nanmean, 'axis, dtype, out, keepdims', a, args, kwargs)

def nanstd(a, *args, **kwargs):
    """
    nanstd(a[, axis, dtype, out, ddof, keepdims])

    Compute the standard deviation along the specified axis, while ignoring NaNs.
    See
    `NumPy documentation <https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.nanstd.html#numpy.nanstd>`_.
    """
    return _numpy_function(np.nanstd, 'axis, dtype, out, ddof, keepdims', a, args, kwargs)

def nanvar(a, *args, **kwargs):
    """
    nanvar(a[, axis, dtype, out, ddof, keepdims])

    Compute the variance along the specified axis, while ignoring NaNs.
    See
    `NumPy documentation <https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.nanvar.html#numpy.nanvar>`_.
    """
    return _numpy_function(np.nanvar, 'axis, dtype, out, ddof, keepdims', a, args, kwargs)

##############################
### Mathematical functions ###
##############################

#https://docs.scipy.org/doc/numpy/reference/routines.math.html

#TODO