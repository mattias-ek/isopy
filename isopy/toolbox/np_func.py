import numpy as _np
from numpy.lib.function_base import array_function_dispatch


__all__ = ['sd', 'se', 'mad', 'nansd', 'nanse', 'nanmad']

#These functions will all call __array_function__

def _sd_dispatcher(a, axis=None):
    return (a,)

@array_function_dispatch(_sd_dispatcher)
def sd(a, axis=None):
    """
    Compute the standard deviation along the specified axis for N-1 degrees of freedom.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    axis : int or None, optional
        Axis along which the range is computed. Default is ``None``. If ``None`` compute
        over the entire array.


    Shortcut for ``np.std(a, ddof=1, axis=axis)``.

    See documentation for `np.std <https://numpy.org/devdocs/reference/generated/numpy.std.html#numpy.std>`_
    for additional information.

    See Also
    --------
    :func:`nansd`
    """
    return _np.std(a, ddof = 1, axis=axis)

@array_function_dispatch(_sd_dispatcher)
def nansd(a, axis=None):
    """
    Compute the standard deviation along the specified axis for N-1 degrees of freedom, while ignoring NaNs.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    axis : int or None, optional
        Axis along which the range is computed. Default is ``None``. If ``None`` compute
        over the entire array.


    Shortcut for ``np.nanstd(a, ddof=1, axis=axis)``.

    See documentation for `numpy.nanstd <https://numpy.org/devdocs/reference/generated/numpy.nanstd.html#numpy.nanstd>`_
    for additional information.

    See Also
    --------
    :func:`sd`
    """
    return _np.nanstd(a, ddof = 1, axis=axis)


def _se_dispatcher(a, axis=None):
    return (a,)


@array_function_dispatch(_se_dispatcher)
def se(a, axis=None):
    """
    Compute the standard error along the specified axis for N-1 degrees of freedom.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    axis : int or None, optional
        Axis along which the range is computed. Default is ``None``. If ``None`` compute
        over the entire array.

    See Also
    --------
    :func:`nanse`
    """
    if axis is None:
        n = a.size
        return _np.std(a, ddof=1) / _np.sqrt(n)
    else:
        n = a.shape[axis]
        return _np.std(a, ddof=1, axis=axis) / _np.sqrt(n)


@array_function_dispatch(_se_dispatcher)
def nanse(a, axis=None):
    """
    Compute the standard error along the specified axis for N-1 degrees of freedom, while ignoring NaNs.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    axis : int or None, optional
        Axis along which the range is computed. Default is ``None``. If ``None`` compute
        over the entire array.

    See Also
    --------
    :func:`se`
    """
    if axis is None:
        n = _np.count_nonzero(~_np.isnan(a))
        return  _np.nanstd(a, ddof=1) / _np.sqrt(n)
    else:
        n = _np.count_nonzero(~_np.isnan(a), axis=axis)
        return _np.nanstd(a, ddof=1, axis=axis) / _np.sqrt(n)


def _mad_dispatcher(a, axis=None, scale=None):
    return (a,)


@array_function_dispatch(_mad_dispatcher)
def mad(a, axis=None, scale = 1.4826):
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


    Shortcut for
    ``scipy.stats.median_absolute_deviation(a, axis=axis, center=np.median, scale=scale, nan_policy='propagate')``.

    See documentation for `scipy.stats.median_absolute_deviation
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_absolute_deviation.html>`_
    for additional information.

    See Also
    --------
    :func:`nanmad`
    """
    if axis is None:
        med = _np.nanmedian(a)
        mad = _np.median(_np.abs(a - med))
    else:
        med = _np.apply_over_axes(_np.nanmedian, a, axis)
        mad = _np.median(_np.abs(a - med), axis=axis)

    return mad * scale


@array_function_dispatch(_mad_dispatcher)
def nanmad(a, axis=None, scale = 1.4826):
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

    See Also
    --------
    :func:`nanmad`
    """

    if axis is None:
        med = _np.nanmedian(a)
        mad = _np.nanmedian(_np.abs(a - med))
    else:
        med = _np.apply_over_axes(_np.nanmedian, a, axis)
        mad = _np.nanmedian(_np.abs(a - med), axis=axis)

    return mad * scale



