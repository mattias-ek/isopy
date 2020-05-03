from isopy import exceptions as _e
from isopy import dtypes as _dt
from isopy.toolbox import np_func as _f
import numpy as _np

__all__ = ['find_outliers', 'find_outliers_mad', 'normalise_data', 'denomralise_data', ]

def normalise_data(data, reference_values, factor=1, subtract_one=True):
    """
    Normalise data to the given reference values.

    Parameters
    ----------
    data : IsopyArray
        Data to be normalised
    reference_values : IsopyArray, dict
        The reference values to be used.
    factor : float, {'delta', 'permil', 'ppt','epsilon','mu','ppm}
        The normalised data is multiplied by this value before it is returned.
    subtract_one : bool
        If ``True`` then ``1`` is subtracted from the data before it is multiplied by *factor*

    Returns
    -------
    IsopyArray
        The normalised data
    """
    data = _e.check_type('data', data, _dt.IsopyArray, coerce=True, coerce_into=_dt.asarray)
    reference_values = _e.check_type('reference_values', reference_values, _dt.IsopyArray, dict, coerce=True)
    factor = _e.check_type('factor', factor, _np.float, str, coerce=True)
    if isinstance(factor, str):
        if factor.lower() in ['delta', 'permil', 'ppt']:
            factor = 1000
        elif factor.lower() in ['epsilon']:
            factor = 10000
        elif factor.lower() in ['mu', 'ppm']:
            factor = 1000000
        else:
            raise ValueError('parameter "factor": "{}" not an avaliable option.'.format(factor))
    subtract_one = _e.check_type('subtract_one', subtract_one, bool)

    new = data / reference_values.get(data.keys())
    if subtract_one: new = new - 1
    new = new * factor

    return new


def denomralise_data(data, reference_values, factor=1, add_one=True):
    """
    Normalise data to the given reference values.

    Parameters
    ----------
    data : IsopyArray
        Normalised data to be denormalised
    reference_values : IsopyArray, dict
        The reference values used to normalise the data.
    factor : float, {'delta', 'permil', 'ppt','epsilon','mu','ppm}
        The value the normalised data was multiplied by.
    add_one : bool
        If ``True`` then ``1`` was subtracted from the normalised data before it is multiplied by *factor*

    Returns
    -------
    IsopyArray
        The denormalised data.
    """
    data = _e.check_type('data', data, _dt.IsopyArray, coerce=True, coerce_into=_dt.asarray)
    reference_values = _e.check_type('reference_values', reference_values, _dt.IsopyArray, dict, coerce=True)
    factor = _e.check_type('factor', factor, _np.float, str, coerce=True)
    if isinstance(factor, str):
        if factor.lower() in ['delta', 'permil', 'ppt']:
            factor = 1000
        elif factor.lower() in ['epsilon']:
            factor = 10000
        elif factor.lower() in ['mu', 'ppm']:
            factor = 1000000
        else:
            raise ValueError('parameter "factor": "{}" not an avaliable option.'.format(factor))
    add_one = _e.check_type('subtract_one', add_one, bool)

    new = data / factor
    if add_one: new = new + 1
    new = new * reference_values.get(data.keys())

    return new


def find_outliers(data, lower_limit, upper_limit, axis = None):
    """
    Find all outliers in data.

    Returns an array where outliers are marked with ``True`` and everything else ``False``.

    Parameters
    ----------
    data : IsopyArray
        Array containing the values to be compared against *lower_limit* and *upper_limit*
    lower_limit : IsopyArray, float
        Values below this value will be marked as outliers.
    upper_limit : IsotpyArray, float
        Values above this value will be marked as outliers.
    axis : int, optional
        If ``None`` then an array with each individual outlier marked is returned. Otherwise ``np.any(outliers, axis)``
        is returned. Default value is ``None``.

    Returns
    -------
    IsopyArray, np.ndarray
        Array of bools with outliers marked with ``True``.
    """
    data = _e.check_type('data', data, _dt.IsopyArray, coerce=True)
    lower_limit = _e.check_type('lower_limit', lower_limit, _dt.IsopyArray, _np.float, coerce=True)
    upper_limit = _e.check_type('upper_limit', upper_limit, _dt.IsopyArray, _np.float, coerce=True)
    axis = _e.check_type('axis', axis, int, allow_none=True)

    outliers = (data > upper_limit) + (data < lower_limit)

    if axis is None:
        return outliers
    else:
        return _np.any(outliers, axis=axis)


def find_outliers_mad(data, level=3, axis=None):
    """
    Find all outliers in data using the median absolute deviation.

    Returns an array where outliers are marked with ``True`` and everything else ``False``.

    Parameters
    ----------
    data : IsopyArray
        Array containing the values to be compared against the median absolute deviation of itself.
    level : float
        Values outside this many levels of the median absolute deviation are considered outliers.
    axis : int, optional
        If ``None`` then an array with each individual outlier marked is returned. Otherwise ``np.any(outliers, axis)``
        is returned. Default value is ``None``.

    Returns
    -------
    IsopyArray, np.ndarray
        Array of bools with outliers marked with ``True``.

    See Also
    --------
    :func:`find_outliers`, :func:`mad`, :func:`nanmad`
    """
    data = _e.check_type('data', data, _dt.IsopyArray, coerce=True)
    level = _e.check_type('level', level, _np.float, coerce=True)
    axis = _e.check_type('axis', axis, int, allow_none=True)

    data = _dt.asarray(data)
    median = _np.nanmedian(data)
    mad = _f.nanmad(data) * level
    return find_outliers(data, median - mad, median + mad, axis=axis)