from matplotlib.axes import Axes as _Axes
from matplotlib.patches import Polygon as _Polygon
from scipy import stats
import numpy as _np
import isopy as _isopy
from isopy import core
from isopy import toolbox
from collections import namedtuple as _namedtuple
from typing import NamedTuple



__add__ = ['regression_york1', 'regression_york2', 'regression_linear']

import isopy.checks


def _plot_york_regression(axes, regression, color = 'black', slopeline = True, errorline = True, shaded = True,
                         zorder = 1, shade_kwargs = None, **line_kwargs):
    """
    Plots the regression within the limits of the supplies axes.

    Parameters
    ----------
    axes : matplotlib.axes.Axes, matplotlib.pytplot.plt
        An axes instance where the regression envelope will be plotted.
    regression : YorkregressResult
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

def regression_linear(x, y):
    """
    Calculate a linear lest-squares regression for two sets of measurements.

    Shortcut for ``scipy.stats.linregress(x, y)``. See
    `here<https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html>`_
    for the scipy documentation.

    Parameters
    ----------
    x, y : numpy_array_like
        Data points through which the regression should be calculated.


    Returns
    -------
    linear_regression_result : LinregressResult
        The returned object contains the following attributes:

        * **slope** - Slope of the regression line.
        * **intercept** - Intercept of the regression line.
        * **rvalue** - Correlation coefficient.
        * **pvalue** - Two-sided p-value for a hypothesis test whose null hypothesis is that the slope is zero, using Wald Test with t-distribution of the test statistic.
        * **stderr** - Standard error of the estimated slope (gradient), under the assumption of residual normality.
        * **intercept_stderr** - Standard error of the estimated intercept, under the assumption of residual normality.

        The returned object also behaves as a tuple so it can be unpacked as
        ``slope, intercept, rvalue, pvalue, stderr = linear_regression_result``

    """
    return stats.linregress(x, y)

def regression_yorkn(x, y, xerr, yerr, r=0, tol=1e-15, n = 1):
    """
    Calculate the slope and intercept taking into account the uncertainty on x and y where the
    uncertainties represent {n} SE/SD.

    Uncertainties on the slope and intercept are given as 2 SE. Based on the formulas from `York et al. (2004)
    American Journal of Physics 72, 367 <https://doi.org/10.1119/1.1632486>`_.

    Parameters
    ----------
    x, y : numpy_array_like
        Data points through which the regression should be calculated.
    xerr, yerr : numpy_array_like
        {n} SE/SD uncertainties for *x* and *y* values. Can be an array with the same size as *x* and *y* or
        a scalar value which will be used for every datapoint.
        Uncertainties for x axis. If a single value then it will be used for all values of *X*.
    r : float, optional
        Correlation coefficient between errors in x and y. Default value is ``0``.
    tol : float, optional
        Tolerance for fit. Default value is ``0.00001``.

    Returns
    -------
    york_regression_result : YorkregressResult
        The returned object contains the following attributes:

        * **slope** - The slope of the regression
        * **intercept** - The y-axis intercept of the regression
        * **slope_err** - The 2se uncertainty of the slope
        * **slope_err** - The 2se uncertainty of the y-axis intercept
        * **msdw** - The mean square weighted deviate of the regression.
        * **p** - The right-tailed probability of the chi-squared distribution.
        * **sumW** - The sum of the weights.
        * **xbar** - The weighted mean of x.

        The returned object also contains the following methods:

        * **y(x)** - Return the value of y at *x*
        * **yerr(x)** -  Return the 2se uncertainty on y at *x*.
        * **label(sigfig)** - Return a string in the format "y=mx + c, msdw" with *sigfig* significant figures.

        The returned object also behaves like a tuple so it can be
        unpacked as ``slope, intercept, slope_err, intercept_err, msdw, p = york_regression_result``
    """
    x = isopy.checks.check_type('x', x, _np.ndarray, coerce=True, coerce_into=_np.array)
    y = isopy.checks.check_type('y', y, _np.ndarray, coerce=True, coerce_into=_np.array)
    xerr = isopy.checks.check_type('xerr', xerr, _np.ndarray, _np.float, coerce=True, coerce_into=[_np.float, _np.array])
    yerr = isopy.checks.check_type('yerr', yerr, _np.ndarray, _np.float, coerce=True, coerce_into=[_np.float, _np.array])
    r = isopy.checks.check_type('r', r, _np.ndarray, _np.float, coerce=True, coerce_into=[_np.float, _np.array])
    tol = isopy.checks.check_type('tol', tol, _np.float, coerce=True)

    X = x
    Y = y
    Xerr = xerr
    Yerr = yerr

    if r > 1 or r < 0:
        raise ValueError('r must be between 0 and 1')
    if X.ndim != 1:
        raise ValueError('parameter "x": must have 1 dimension not {}'.format(X.ndim))
    if Y.ndim != 1:
        raise ValueError('parameter "y": must have 1 dimension not {}'.format(X.ndim))

    if X.size != Y.size:
        raise ValueError('x and y must have the same size')

    if not isinstance(Xerr, _np.ndarray):
        Xerr = _np.ones(X.size) * Xerr
    elif Xerr.size != X.size:
        raise ValueError('xerr and x must have the same size')

    if not isinstance(Yerr, _np.ndarray):
        Yerr = _np.ones(Y.size) * Yerr
    elif Yerr.size != Y.size:
        raise ValueError('yerr and y must have the same size')

    Xerr = Xerr / n
    Yerr = Yerr / n

    # Do calculations
    wX = 1 / (Xerr ** 2)
    wY = 1 / (Yerr ** 2)

    k = _np.sqrt(wX * wY)
    slope = (_np.mean(X * Y) - _np.mean(X) * _np.mean(Y)) / (_np.mean(X ** 2) - _np.mean(X) ** 2)

    # Iterativley converge self.slope until tol is reached
    for i in range(1000):
        W = (wX * wY) / (wX + (slope ** 2) * wY - 2 * slope * r * k)
        Xbar = _np.sum(W * X) / _np.sum(W)
        Ybar = _np.sum(W * Y) / _np.sum(W)
        U = X - Xbar
        V = Y - Ybar
        beta = W * (U / wY + (slope * V) / wX - (slope * U + V) * (r / k))

        b2 = _np.sum(W * beta * V) / _np.sum(W * beta * U)
        dif = _np.abs(b2 / slope - 1)
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

    dof = x.size - 2
    if dof > 0:
        mXY = _np.array([X - x, Y - y]).T.reshape(-1, 1, 2)
        mXYerr = _np.array([Xerr * Xerr, r * Xerr * Yerr, r * Xerr * Yerr, Yerr * Yerr]).T.reshape(-1, 2, 2)
        miXYerr = _np.linalg.inv(mXYerr)
        mXYt = mXY.reshape(-1, 2, 1)
        rmXY = _np.sum(mXY @ miXYerr @ mXYt)

        msdw = rmXY / dof
        p = 1 - stats.chi2.cdf(rmXY, dof)
    else:
        msdw = 1
        p = 1

    return YorkregressResult(slope, intercept, slope_error,  intercept_error, msdw, p, _np.sum(W), xbar)

@core.updatedoc(regression_yorkn, n = 1)
def regression_york1(x, y, xerr, yerr, r=0, tol=1e-15):
    return regression_yorkn(x, y, xerr, yerr, r, tol, n=1)

@core.updatedoc(regression_yorkn, n = 2)
def regression_york2(x, y, xerr, yerr, r=0, tol=1e-15):
    return regression_yorkn(x, y, xerr, yerr, r, tol, n=2)

class YorkregressResult(tuple):
    """
    Contains the result of a york regression

    Attributes
    ----------
    slope : float
        The slope of the regression
    intercept : float
        The y axis intercept of the regression
    slope_err : float
        The 2se uncertainty of the slope
    intercept_err : float
        The 2 se uncertainty of the y axis intercept
    msdw : float
        The mean square weighted deviate of the regression.
    p : float
        The right-tailed probability of the chi-squared distribution.
    sumW : float
        Sum of the weights.
    xbar : float
        The weighted mean of x.

    Methods
    -------
    y(x)
        Return y at *x*. Same as ``slope * x + intercept``
    yerr(x)
        Return the 2se uncertainty on y at *x*. Same as
        ``sqrt( 1 / sumW + ((x - xbar) ** 2) * (slope_err ** 2)``

    Notes
    -----
    Behaves as a namedtuple so can be unpacked to ``slope, intercept, slope_err, intercept_err, msdw, p = york_regression_result``
    """
    def __new__(cls, slope, intercept, slope_err, intercept_err, msdw, p, sumW, xbar):
        obj = super(YorkregressResult, cls).__new__(cls, (slope, intercept, slope_err, intercept_err, msdw, p))
        obj.slope = slope
        obj.intercept = intercept
        obj.slope_err = slope_err
        obj.intercept_err = intercept_err
        obj.msdw = msdw
        obj.p = p
        obj.sumW = sumW
        obj.xbar = xbar
        return obj

    def __repr__(self):
        return self.label()

    def y(self, x):
        """
        Return y at *x*.
        """
        return self.slope * x + self.intercept

    def yerr(self, x):
        """
        Return the uncertainty of y at *x*.
        """
        return _np.sqrt(1 / self.sumW + ((x - self.xbar) ** 2) * (self.slope_err ** 2))

    def label(self, sigfig = 5):
        label = 'y='
        label = f'{label}({toolbox.plotting._format_sigfig(self.slope, sigfig, self.slope_err)}'
        label = f'{label}±{toolbox.plotting._format_sigfig(self.slope_err, sigfig, self.slope_err)})x'
        label = f'{label} + ({toolbox.plotting._format_sigfig(self.intercept, sigfig, self.intercept_err)}'
        label = f'{label}±{toolbox.plotting._format_sigfig(self.intercept_err, sigfig, self.intercept_err)})'
        label = f'{label}, msdw={self.msdw:.2f}'
        return label


