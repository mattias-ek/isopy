from matplotlib.axes import Axes as _Axes
from matplotlib.patches import Polygon as _Polygon
from scipy import stats
import numpy as np
from isopy import core
from isopy import toolbox
from isopy import array_functions


__add__ = ['linregress', 'yorkregress', 'yorkregress2', 'yorkregress3', 'yorkregress95', 'yorkregress99']

import isopy.checks


def linregress(x, y):
    """
    Calculate a linear least-squares regression for two sets of measurements.

    Uses ``scipy.stats.linregress(x, y)`` for the regression. See
    `here<https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html>`_
    for the scipy documentation.

    Parameters
    ----------
    x, y : numpy_array_like
        Data points through which the regression should be calculated.


    Returns
    -------
    result : LinregressResult
        The returned object contains the following attributes:

        * **slope** - Slope of the regression line.
        * **intercept** - Intercept of the regression line.
        * **slope_se** - Standard error of the slope.
        * **intercept_se** - Standard error of the intercept.
        * **df** - The degrees of freedom of the regression (N - 2).

        The returned object also contains the following methods:

        * **y(x)** - Returns the value of y at *x*
        * **label(sigfig)** - Returns a string in the format "y=mx + c" with *sigfig* significant figures.

    """
    #  * **pvalue** - Two-sided p-value for a hypothesis test whose null hypothesis is that the slope is zero, using Wald Test with t-distribution of the test statistic.
    # * **rvalue** - Correlation coefficient.
    result = stats.linregress(x, y)

    return LinregressResult(result.slope, result.intercept, result.stderr, result.intercept_stderr, len(x) - 2)

def yorkregress(x, y, xerr, yerr, r=0, err_ci = None, err_zscore=None, result_ci = None, result_zscore=None, tol=1e-10):
    """
    Calculate the slope and intercept taking into account the uncertainty on x and y.

    Uncertainties on the slope and intercept are given as standard errors. Based on the formulas from `York et al. (2004)
    American Journal of Physics 72, 367 <https://doi.org/10.1119/1.1632486>`_.

    If neither *err_ci* or *err_zscore* are given it assumes the uncertainties represent 1 SD/SE. If neither
    *result_ci* or *result_zscore* are giuen the *err_ci*/*err_zscore* is also used for the
    uncertainty on the slope and intercept.

    The functions *yorkregress2* and *yorkregress3* are also avaliable with preset values for *err_zscore*
    of 2 and 3 respectively. Similarly, *yorkregress95* and *yorkregress99* exists with preset values
    *err_ci* 0.95 and 0.99.

    Parameters
    ----------
    x, y : numpy_array_like
        Data points through which the regression should be calculated.
    xerr, yerr : numpy_array_like
        Uncertainties for *x* and *y* values. Can be an array with the same size as *x* and *y* or
        a scalar value which will be used for every datapoint.
    r : float, optional
        Correlation coefficient between errors in x and y. Default value is ``0``.
    err_ci : float, optional
        The confidence interval of *xerr* and *yerr*.
    err_zscore : float, optional
        The zscore of *xerr* and *yerr*.
    result_ci : float, optional
        The desired confidence interval of the uncertainty on the slope and intercept.
    result_zscore : float, optional
        The desired zscore of the uncertainty on the slope and intercept.
    tol : float, optional
        Tolerance for fit. Default value is ``1e-10``.

    Returns
    -------
    result : YorkregressResult
        The returned object contains the following attributes:

        * **slope** - The slope of the regression
        * **intercept** - The y-axis intercept of the regression
        * **slope_se** - The standard error of the slope
        * **intercept_se** - The standard error of the y-axis intercept
        * **msdw** - The mean square weighted deviate of the regression.
        * **df** - The degrees of freedom of the regression (N - 2).
        * **pvalue** - The two-tailed probability of the chi-squared value with *df* degrees of freedom.

        The returned object also contains the following methods:

        * **y(x)** - Returns the value of y at *x*
        * **yerr(x)** -  Returns the standard error on y at *x*.
        * **label(sigfig)** - Returns a string in the format "y=mx + c, msdw" with *sigfig* significant figures.
    """
    #   * **sumW** - The sum of the weights.
    #   * **xbar** - The weighted mean of x.

    x = isopy.checks.check_type('x', x, np.ndarray, coerce=True, coerce_into=np.array)
    y = isopy.checks.check_type('y', y, np.ndarray, coerce=True, coerce_into=np.array)
    xerr = isopy.checks.check_type('xerr', xerr, np.ndarray, np.float64, coerce=True,
                                   coerce_into=[np.float64, np.array])
    yerr = isopy.checks.check_type('yerr', yerr, np.ndarray, np.float64, coerce=True,
                                   coerce_into=[np.float64, np.array])
    r = isopy.checks.check_type('r', r, np.ndarray, np.float64, coerce=True,
                                coerce_into=[np.float64, np.array])
    tol = isopy.checks.check_type('tol', tol, np.float64, coerce=True)

    if err_ci is not None and err_zscore is not None:
        raise ValueError('Both error ci and zscore was given')
    if result_ci is not None and result_zscore is not None:
        raise ValueError('Both result ci and zscore was given')
    if err_ci is not None and (err_ci < 0 or err_ci > 1):
        raise ValueError('error confidence interval must be between 0 and 1')
    if result_ci is not None and (result_ci < 0 or result_ci > 1):
        raise ValueError('result confidence interval must be between 0 and 1')

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

    if not isinstance(Xerr, np.ndarray):
        Xerr = np.ones(X.size) * Xerr
    elif Xerr.size != X.size:
        raise ValueError('xerr and x must have the same size')

    if not isinstance(Yerr, np.ndarray):
        Yerr = np.ones(Y.size) * Yerr
    elif Yerr.size != Y.size:
        raise ValueError('yerr and y must have the same size')

    if err_ci is not None:
        ci = array_functions.calculate_ci(err_ci)
        Xerr = Xerr / ci
        Yerr = Yerr / ci
        if result_ci is None and result_zscore is None:
            result_ci = err_ci
    elif err_zscore is not None:
        Xerr = Xerr / err_zscore
        Yerr = Yerr / err_zscore
        if result_ci is None and result_zscore is None:
            result_zscore = err_zscore

    # Do calculations
    wX = 1 / (Xerr ** 2)
    wY = 1 / (Yerr ** 2)

    k = np.sqrt(wX * wY)
    slope = (np.mean(X * Y) - np.mean(X) * np.mean(Y)) / (np.mean(X ** 2) - np.mean(X) ** 2)

    # Iteratively converge self.slope until tol is reached
    for i in range(1000):
        W = (wX * wY) / (wX + (slope ** 2) * wY - 2 * slope * r * k)
        Xbar = np.sum(W * X) / np.sum(W)
        Ybar = np.sum(W * Y) / np.sum(W)
        U = X - Xbar
        V = Y - Ybar
        beta = W * (U / wY + (slope * V) / wX - (slope * U + V) * (r / k))

        b2 = np.sum(W * beta * V) / np.sum(W * beta * U)
        dif = np.abs(b2 / slope - 1)
        slope = b2
        if dif < tol: break
    else:
        raise ValueError('Unable to calculate a fit after 1000 iterations.')

    # Calculate self.intercept
    intercept = Ybar - slope * Xbar

    # Calculate adjusted points
    x = Xbar + beta
    xbar = np.sum(W * x) / np.sum(W)
    u = x - xbar

    y = Ybar + slope * beta
    ybar = np.sum(W * y) / np.sum(W)
    v = y - ybar

    # Calculate the uncertianty on the slope
    slope_error = np.sqrt(1 / np.sum(W * (u ** 2)))
    intercept_error = np.sqrt(1 / np.sum(W) + ((0 - xbar) ** 2) * (slope_error ** 2))

    # Goodness of fit
    S = np.sum(W * (Y - slope * X - intercept) ** 2)

    dof = x.size - 2
    if dof > 0:
        #mXY = np.array([X - x, Y - y]).T.reshape(-1, 1, 2)
        #mXYerr = np.array([Xerr * Xerr, r * Xerr * Yerr, r * Xerr * Yerr, Yerr * Yerr]).T.reshape(-1, 2, 2)
        #miXYerr = np.linalg.inv(mXYerr)
        #mXYt = mXY.reshape(-1, 2, 1)
        #rmXY = np.sum(mXY @ miXYerr @ mXYt) same as S

        msdw = S / dof
        p = 1 - stats.chi2.cdf(S, dof)
        p2 = (p-0.5)*2
    else:
        msdw = 1
        p2 = 1

    if result_ci is not None:
        slope_se = slope_error * stats.norm.ppf(0.5 + result_ci / 2)
    elif result_zscore is not None:
        slope_se = slope_error * result_zscore
    else:
        slope_se = slope_error

    intercept_se = np.sqrt(1 / np.sum(W) + ((0 - xbar) ** 2) * (slope_se ** 2))

    return YorkregressResult(slope, intercept, slope_se, intercept_se, dof, msdw, p2,
                             np.sum(W), xbar)


yorkregress1 = core.partial_func(yorkregress, 'yorkregress2', err_zscore=1, result_zscore=1)
yorkregress2 = core.partial_func(yorkregress, 'yorkregress2', err_zscore=2, result_zscore=2)
yorkregress3 = core.partial_func(yorkregress, 'yorkregress3', err_zscore=3, result_zscore=3)
yorkregress95 = core.partial_func(yorkregress, 'yorkregress95', err_ci=0.95, result_ci=0.95)
yorkregress99 = core.partial_func(yorkregress, 'yorkregress99', err_ci=0.99, result_ci=0.99)


class LinregressResult:
    def __init__(self, slope, intercept, slope_se, intercept_se, df):
        self.slope = slope
        self.intercept = intercept
        self.slope_se = slope_se
        self.intercept_se = intercept_se
        self.df = df

    def __repr__(self):
        return (f'{self.__class__.__name__}(slope={self.slope}, intercept={self.intercept}, '
                f'slope_se={self.slope_se}, intercept_se={self.intercept_se}, '
                f'df={self.df})')

    def y(self, x):
        """
        Return y at *x*.
        """
        return self.slope * x + self.intercept

    def y_se(self, x):
        """
        Currently not avaliable for linear regressions. Raises ``NotImplementedError``.
        """
        raise NotImplementedError('"y_se" is currently not avaliable for linear regressions')

    def label(self, sigfig=5):
        label = 'y='
        if np.isnan(self.slope):
            label = f'{label}(nan±nan) + (nan±nan)'
        else:
            label = f'{label}({toolbox.plot._format_sigfig(self.slope, sigfig, self.slope_se)}'
            label = f'{label}±{toolbox.plot._format_sigfig(self.slope_se, sigfig, self.slope_se)})x'
            label = f'{label} + ({toolbox.plot._format_sigfig(self.intercept, sigfig, self.intercept_se)}'
            label = f'{label}±{toolbox.plot._format_sigfig(self.intercept_se, sigfig, self.intercept_se)})'
        return label

class YorkregressResult(LinregressResult):
    def __init__(self, slope, intercept, slope_se, intercept_se, df, msdw, pvalue, sumW, xbar):
        self.slope = slope
        self.intercept = intercept
        self.slope_se = slope_se
        self.intercept_se = intercept_se
        self.df = df
        self.msdw = msdw
        self.pvalue = pvalue
        self._sumW = sumW
        self._xbar = xbar

    def __repr__(self):
        return (f'{self.__class__.__name__}(slope={self.slope}, intercept={self.intercept}, '
                f'slope_se={self.slope_se}, intercept_se={self.intercept_se}, df={self.df}, '
                f'msdw={self.msdw}, pvalue={self.pvalue})')

    def y_se(self, x):
        """
        Return the standard error of y at *x*.
        """
        return np.sqrt(1 / self._sumW + ((x - self._xbar) ** 2) * (self.slope_se ** 2))

    def label(self, sigfig=5):
        label = 'y='
        if np.isnan(self.slope):
            label = f'{label}(nan±nan) + (nan±nan), msdw=nan'
        else:
            label = f'{label}({toolbox.plot._format_sigfig(self.slope, sigfig, self.slope_se)}'
            label = f'{label}±{toolbox.plot._format_sigfig(self.slope_se, sigfig, self.slope_se)})x'
            label = f'{label} + ({toolbox.plot._format_sigfig(self.intercept, sigfig, self.intercept_se)}'
            label = f'{label}±{toolbox.plot._format_sigfig(self.intercept_se, sigfig, self.intercept_se)})'
            label = f'{label}, msdw={self.msdw:.2f}'
        return label
