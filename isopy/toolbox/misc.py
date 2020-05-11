from matplotlib.axes import Axes as _Axes
from matplotlib.patches import Polygon as _Polygon
import numpy as _np
import isopy as _isopy



__add__ = ['plot_york_regression', 'york_regression', 'johnson_nyquist_noise']


def plot_york_regression(axes, regression, color = 'black', slopeline = True, errorline = True, shaded = True,
                         zorder = 1, shade_kwargs = None, **line_kwargs):
    """
    Plots the regression within the limits of the supplies axes.

    Parameters
    ----------
    axes : matplotlib.axes.Axes, matplotlib.pytplot.plt
        An axes instance where the regression envelope will be plotted.
    regression : YorkRegression
        The regression to be plotted
    color : str
        The color of the lines/polygon drawn for this regression. See
        `matplotlib documentation <https://matplotlib.org/tutorials/colors/colors.html>`_.
    slopeline : bool
        If ``True`` the slope will be plotted as a solid line. Default value is ``True``.
    errorline : bool, optional
        If ``True`` the outline of the regression envelope will be plotted as a dashed line. Default value is ``True``.
    shaded : bool, optional
        If ``True`` the regression envelope will be shaded. Default value is ``True``.
    zorder : int, optional
        The higher the value the later the regression will be be drawn. Default value is ``1``.
    shade_kwargs : dict, optional
        kwargs that will be passed to polygon that makes up the shaded area.
    **line_kwargs
        Keyword arguments passed when drawing the slopeline and the errorline.

    """
    #TODO add
    slopeline_kwargs = {'linestyle': '-', 'marker': '', 'color': color, 'zorder': zorder}
    slopeline_kwargs.update(line_kwargs)
    errorline_kwargs = {'linestyle': '--', 'marker': '', 'color': color, 'zorder': zorder}
    errorline_kwargs.update(line_kwargs)
    errorline_kwargs.pop('label', None)
    poly_kwargs = {'color': color, 'alpha': 0.1, 'zorder': zorder-0.1}
    if shade_kwargs is not None: poly_kwargs.update(shade_kwargs)
    if not isinstance(axes, _Axes):
        try:
            axes = axes.gca()
        except:
            raise ValueError('axes must be a matplotlib Axes or pyplot obj')

    xlim = axes.get_xlim()
    ylim = axes.get_ylim()

    xval = _np.linspace(xlim[0], xlim[1], 10000)
    slope = xval * regression.slope + regression.intercept
    upper = slope + regression.y_error(xval)
    lower = slope - regression.y_error(xval)

    index_slope = _np.all([slope > ylim[0], slope < ylim[1]], 0)
    index_lower = _np.all([lower > ylim[0], lower < ylim[1]], 0)
    index_upper = _np.all([upper > ylim[0], upper < ylim[1]], 0)

    upper_coordinates = [*zip(xval[index_upper], upper[index_upper])]
    lower_coordinates = [*zip(xval[index_lower], lower[index_lower])]
    lower_coordinates.reverse()

    if shaded:
        poly = _Polygon(upper_coordinates + lower_coordinates, **poly_kwargs)
        axes.add_patch(poly)

    if slopeline: axes.plot(xval[index_slope], slope[index_slope], **slopeline_kwargs)
    if errorline:
        axes.plot(xval[index_upper], upper[index_upper], **errorline_kwargs)
        axes.plot(xval[index_lower], lower[index_lower], **errorline_kwargs)

    #axes.show(xlim, [regression.slope*xlim[0]+regression.intercept, regression.slope*xlim[1]+regression.intercept], **kwargs)


def york_regression(X, Y, Xerr, Yerr, r=0, tol=0.00001):
    """
    Calculate the slope and intercept taking into account the uncertainty on x and y.

    Uncertainties on the slope and intercept are given as 2σ. Based on the formulas from `York et al. (2004)
    American Journal of Physics 72, 367 <https://doi.org/10.1119/1.1632486>`_.

    Parameters
    ----------
    X : np.ndarray
        Values for x axis
    Y : np.ndarray
        Values for y axis
    Xerr : float, np.ndarray
        Uncertainties for x axis. If a single value then it will be used for all values of *X*.
    Yerr : float, np.ndarray
        Uncertainties for y axis. If a single value then it will be used for all values of *Y*.
    r : float, optional
        Correlation coefficient between errors in x and y. Default value is ``0``.
    tol : float, optional
        Tolerance for fit. Default value is ``0.00001``.


    Returns
    -------
    YorkRegression
        Contains the slope and intercept values and all information needed to calculate the y for any given x.

    See Also
    --------
    :class:`YorkRegression`, :func:`plot_york_regression`
    """
    X = _isopy.core.check_type('X', X, _np.ndarray, coerce=True, coerce_into=_np.array)
    Y = _isopy.core.check_type('Y', Y, _np.ndarray, coerce=True, coerce_into=_np.array)
    Xerr = _isopy.core.check_type('Xerr', Xerr, _np.ndarray, _np.float, coerce=True, coerce_into=[_np.float, _np.array])
    Yerr = _isopy.core.check_type('Yerr', Yerr, _np.ndarray, _np.float, coerce=True, coerce_into=[_np.float, _np.array])
    r = _isopy.core.check_type('r', r, _np.float, coerce=True)
    tol = _isopy.core.check_type('tol', tol, _np.float, coerce=True)

    if r > 1 or r < 0:
        raise ValueError('r must be between 0 and 1')
    if X.ndim != 1:
        raise ValueError('parameter "X": must have 1 dimension not {}'.format(X.ndim))
    if Y.ndim != 1:
        raise ValueError('parameter "Y": must have 1 dimension not {}'.format(X.ndim))

    if X.size != Y.size:
        raise ValueError('X and Y must have the same size')

    if not isinstance(Xerr, _np.ndarray):
        Xerr = _np.ones(X.size) * Xerr
    elif Xerr.size != X.size:
        raise ValueError('Xerr and X must have the same size')

    if not isinstance(Yerr, _np.ndarray):
        Yerr = _np.ones(Y.size) * Yerr
    elif Yerr.size != Y.size:
        raise ValueError('Yerr and Y must have the same size')

    # Do calculations
    WX = 1 / (Xerr ** 2)
    WY = 1 / (Yerr ** 2)

    k = _np.sqrt(WX * WY)
    slope = (_np.mean(X * Y) - _np.mean(X) * _np.mean(Y)) / (_np.mean(X ** 2) - _np.mean(X) ** 2)

    # Iterativley converge self.slope until tol is reached
    for i in range(1000):
        W = (WX * WY) / (WX + (slope ** 2) * WY - 2 * slope * r * k)
        Xbar = _np.sum(W * X) / _np.sum(W)
        Ybar = _np.sum(W * Y) / _np.sum(W)
        U = X - Xbar
        V = Y - Ybar
        beta = W * (U / WY + (slope * V) / WX - (slope * U + V) * (r / k))

        b2 = _np.sum(W * beta * V) / _np.sum(W * beta * U)
        dif = _np.sqrt((b2 / slope - 1) ** 2)
        slope = b2
        if dif < tol: break
    if i == 999:
        raise ValueError('Unable to calculate a fit after 1000 iterations.')

    # Calculate self.intercept
    intercept = Ybar - slope * Xbar

    # Calculate adjusted points
    x = Xbar + beta
    xbar = _np.sum(W * x) / _np.sum(W)
    u = x - xbar

    y = Ybar + slope * beta
    ybar = _np.sum(W * y) / _np.sum(W)
    v = y - ybar

    #Calculate the uncertianty on the slope
    slope_error = _np.sqrt(1 / _np.sum(W * (u ** 2)))
    intercept_error = _np.sqrt(1 / _np.sum(W) + ((0 - xbar) ** 2) * (slope_error ** 2))

    #Goodness of fit
    S = _np.sum(W * (Y - slope * X - intercept) ** 2)

    #implementation in isoplotr
    return YorkRegression(slope, slope_error, intercept, intercept_error, _np.sum(W), xbar)


class YorkRegression:
    """
    Returned by :func:`york_regression`.

    This object contains all the necessary attributes to calculate the value and uncertainty of y for any given
    value of x.

    Attributes
    ----------
    slope : float
        The best fit slope of the regression
    slope_error : float
        The uncertainty of the slope
    intercept : float
        The y axis intercept of the regression
    intercept_error : float
        The uncertainty of the y axis intercept
    sumW : float
        Sum of the weights
    xbar : float
        The weighted x mean.

    See Also
    --------
    :func:`york_regression`, :func:`plot_york_regression`
    """
    def __init__(self, slope, slope_error, intercept, intercept_error, sumW, xbar):
        self.slope = slope
        self.slope_error = slope_error
        self.intercept = intercept
        self.intercept_error = intercept_error
        self.sumW = sumW
        self.xbar = xbar

    def __repr__(self):
        return 'y=({} +- {})x + ({} +- {}); xbar = {}, sum(W) = {}'.format(self.slope, self.slope_error,
                                                self.intercept, self.intercept_error, self.xbar, self.sumW)

    def y_error(self, x):
        """
        Return the uncertainty of y at *x*.
        """
        return _np.sqrt(1 / self.sumW + ((x - self.xbar) ** 2) * (self.slope_error ** 2))

    def y(self, x):
        """
        Return y at *x*.
        """
        return self.slope * x + self.intercept


def johnson_nyquist_noise(voltage, resistor = 1E11, integration_time = 8.389, include_counting_statistics = True,
                          T = 309, R = 1E11, cpv = 6.25E7):
    """
    Calculate the Johnson-Nyquist noise and counting statistics for a given voltage.

    The Johnson-Nyquist noise (:math:`n_{jn}` is calculated as:

    .. math::
        n_{jn} = \\sqrt{ \\frac{4*k_{b}*T*r} {t} } * \\frac{R} {r}

    The counting statistics, or shot noise, (:math:`n_{cs}` is calculated as:

    .. math::
        n_{cs} = \\sqrt{ \\frac{1} {v * c_{V} * t}} * v

    The two are combined as:

    .. math::
        n_{all} = \\sqrt{ (n_{jn})^2 + (n_{cs})^2 }

    The noise for a specific isotope ratio can be calculated as:

    .. math::
        n_{n,d} = \\sqrt{ (n_{n})^2 + (n_{d})^2 }

    where :math:`n` is the numerator isotope and :math:`d` is the denominator isotope.

    Adapted from the equations in `Liu & Pearson (2014) Chemical Geology, 10, 301-311
    <https://doi.org/10.1016/j.chemgeo.2013.11.008>`_.

    Parameters
    ----------
    voltage : IsotopeArray, float, np.ndarray
        The measured voltages. :math:`v` in the equations above.
    resistor : IsotopeArray, float, np.ndarray
        The resistor for the measurement. Default value is ``1E11``. :math:`r` in the equations above.
    integration_time : float, optional
        The integration time in seconds for a single measurement. Default value is ``8.389``.
        :math:`t` in the equations above.
    include_counting_statistics: bool, optional
        If ``True`` then the counting statistics are included in the returned value. Default value is ``True``.
    T : float, optional
        Amplifier housing temperature in kelvin. Default value is ``309``. :math:`T` in the equations above.
    R : float, optional
        Voltage is reported as volts for this resistor value. Default value is ``1E11``. :math:`R` in the equations
        above.
    cpv : float, optional
        Counts per volt per second. Default value is ``6.25E7``. :math:`C_{V}` in the equations above.

    Returns
    -------
    np.float or np.ndarray or IsotopeArray
        The noise in V for the given voltage/set of voltages.
    """

    voltage = _isopy.core.check_type('voltage', voltage, _isopy.core.IsotopeArray, _np.ndarray, _np.float, coerce=True,
                            coerce_into=[_isopy.core.IsotopeArray, _np.float, _np.array])
    resistor = _isopy.core.check_type('resistor', resistor, _isopy.core.IsotopeArray, _np.ndarray, _np.float, coerce=True,
                             coerce_into=[_isopy.core.IsotopeArray, _np.float, _np.array])
    integration_time = _isopy.core.check_type('integration_time', integration_time, _np.float, coerce=True)
    include_counting_statistics = _isopy.core.check_type('include_counting_statistics', include_counting_statistics,
                                                bool)
    T = _isopy.core.check_type('T', T, _np.float, coerce=True)
    R = _isopy.core.check_type('R', R, _np.float, coerce=True)
    cpv = _isopy.core.check_type('cpv', cpv, _np.float, coerce=True)

    kB = _np.float(1.3806488E-023) # Boltsman constant

    t_noise = _np.sqrt((4 * kB * T * resistor) / integration_time) / (voltage * (resistor / R)) * voltage
    t_noise = _np.sqrt((4 * kB * T * resistor) / integration_time) * (R / resistor)

    if include_counting_statistics:
        c_stat = _np.sqrt(1 / (voltage * cpv * integration_time)) * voltage
        return _np.sqrt(c_stat ** 2 + t_noise ** 2)
    else:
        return t_noise


