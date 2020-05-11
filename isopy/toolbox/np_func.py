import numpy as _np
from numpy.lib.function_base import array_function_dispatch


__all__ = ['sd', 'se', 'mad', 'nansd', 'nanse', 'nanmad', 'count_notnan']

#These functions will all call __array_function__

def _sd_dispatcher(a, axis=None, out=None):
    return (a,out)

@array_function_dispatch(_sd_dispatcher)
def sd(a, axis=None, out=None):
    """
    Compute the standard deviation along the specified axis for N-1 degrees of freedom.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    axis : int or None, optional
        Axis along which the range is computed. Default is ``None``. If ``None`` compute
        over the entire array.
    out : IsopyArray, optional
        If provided, the destination to place the result. Must have the same number of rows and columns as the
        result if not provided.

    Shortcut for ``np.std(a, ddof=1, axis=axis)``.

    See documentation for `np.std <https://numpy.org/devdocs/reference/generated/numpy.std.html#numpy.std>`_
    for additional information.

    See Also
    --------
    :func:`nansd`
    """
    return _np.std(a, ddof = 1, axis=axis, out=out)

@array_function_dispatch(_sd_dispatcher)
def nansd(a, axis=None, out = None):
    """
    Compute the standard deviation along the specified axis for N-1 degrees of freedom, while ignoring NaNs.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    axis : int or None, optional
        Axis along which the range is computed. Default is ``None``. If ``None`` compute
        over the entire array.
    out : IsopyArray, optional
        If provided, the destination to place the result. Must have the same number of rows and columns as the
        result if not provided.

    Shortcut for ``np.nanstd(a, ddof=1, axis=axis)``.

    See documentation for `numpy.nanstd <https://numpy.org/devdocs/reference/generated/numpy.nanstd.html#numpy.nanstd>`_
    for additional information.

    See Also
    --------
    :func:`sd`
    """
    return _np.nanstd(a, ddof = 1, axis=axis, out=out)


def _se_dispatcher(a, axis=None, out=None):
    return (a,out)


@array_function_dispatch(_se_dispatcher)
def se(a, axis=None, out=None):
    """
    Compute the standard error along the specified axis for N-1 degrees of freedom.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    axis : int or None, optional
        Axis along which the range is computed. Default is ``None``. If ``None`` compute
        over the entire array.
    out : IsopyArray, optional
        If provided, the destination to place the result. Must have the same number of rows and columns as the
        result if not provided.

    See Also
    --------
    :func:`nanse`
    """
    a = _np.asarray(a)
    if axis is None:
        n = a.size
    else:
        n = a.shape[axis]
    return _np.divide(_np.std(a, ddof=1, axis=axis), _np.sqrt(n), out=out)


@array_function_dispatch(_se_dispatcher)
def nanse(a, axis=None, out=None):
    """
    Compute the standard error along the specified axis for N-1 degrees of freedom, while ignoring NaNs.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    axis : int or None, optional
        Axis along which the range is computed. Default is ``None``. If ``None`` compute
        over the entire array.
    out : IsopyArray, optional
        If provided, the destination to place the result. Must have the same number of rows and columns as the
        result if not provided.

    See Also
    --------
    :func:`se`
    """
    n = count_notnan(a, axis)
    return _np.divide(_np.nanstd(a, ddof=1, axis=axis), _np.sqrt(n), out=out)


def _mad_dispatcher(a, axis=None, scale=None, out=None):
    return (a,out)


@array_function_dispatch(_mad_dispatcher)
def mad(a, axis=None, scale = 1.4826, out=None):
    """
    Compute the median absolute deviation of the data along the given axis.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    axis : int or None, optional
        Axis along which the range is computed. Default is ``None``. If ``None`` compute
        over the entire array.
    scale : int, optional
        The scaling factor applied to the MAD. The default scale (1.4826) ensures consistency with the standard
        deviation for normally distributed data.
    out : IsopyArray, optional
        If provided, the destination to place the result. Must have the same number of rows and columns as the
        result if not provided.

    See Also
    --------
    :func:`nanmad`
    """
    if axis is None:
        med = _np.nanmedian(a)
        mad = _np.median(_np.abs(a - med), out=out)
    else:
        med = _np.apply_over_axes(_np.nanmedian, a, axis)
        mad = _np.median(_np.abs(a - med), axis=axis, out=out)

    return _np.multiply(mad, scale, out=out)


@array_function_dispatch(_mad_dispatcher)
def nanmad(a, axis=None, scale = 1.4826, out=None):
    """
    Compute the median absolute deviation along the specified axis, while ignoring NaNs.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    axis : int or None, optional
        Axis along which the range is computed. Default is ``None``. If ``None`` compute
        over the entire array.
    scale : int, optional
        The scaling factor applied to the MAD. The default scale (1.4826) ensures consistency with the standard
        deviation for normally distributed data.
    out : IsopyArray, optional
        If provided, the destination to place the result. Must have the same number of rows and columns as the
        result if not provided.

    See Also
    --------
    :func:`mad`
    """

    if axis is None:
        med = _np.nanmedian(a)
        mad = _np.nanmedian(_np.abs(a - med), out=out)
    else:
        med = _np.apply_over_axes(_np.nanmedian, a, axis)
        mad = _np.nanmedian(_np.abs(a - med), axis=axis, out=out)

    return _np.multiply(mad, scale, out=out)


def _notnan_dispatcher(a, axis=None):
    return (a,)

@array_function_dispatch(_notnan_dispatcher)
def count_notnan(a, axis=None):
    """
    Count all values in array that are not NaN's along the specified axis.

    Parameters
    ----------
    a : IsopyArray, array_like
        The array over which the operation will be performed
    axis : int, optional
        The axis along which the count should be performed. If not given all values in the array will be counted.

    """
    return _np.count_nonzero(~_np.isnan(a), axis=axis)



