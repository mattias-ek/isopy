import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Polygon
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import ticker, cm, colors

def plot_york_regression(axes, regression, slopeline = True, errorline = True, shaded = True, shade_kwargs = None, **line_kwargs):
    slopeline_kwargs = {'linestyle': '-', 'marker': '', 'color': 'black'}
    slopeline_kwargs.update(line_kwargs)
    errorline_kwargs = {'linestyle': '--', 'marker': '', 'color': 'black'}
    errorline_kwargs.update(line_kwargs)
    poly_kwargs = {'color': slopeline_kwargs['color'], 'alpha': 0.1}
    if shade_kwargs is not None: poly_kwargs.update(shade_kwargs)
    if axes is plt: axes = plt.gca()
    if not isinstance(axes, Axes): raise ValueError('axes must be a matplotlib Axes')

    xlim = axes.get_xlim()
    ylim = axes.get_ylim()
    print(xlim, ylim)

    print(regression)

    xval = np.linspace(xlim[0], xlim[1], 10000)
    slope = xval * regression.slope + regression.intercept
    upper = slope + regression.y_error(xval)
    lower = slope - regression.y_error(xval)

    index_slope = np.all([slope > ylim[0], slope < ylim[1]], 0)
    index_lower = np.all([lower > ylim[0], lower < ylim[1]], 0)
    index_upper = np.all([upper > ylim[0], upper < ylim[1]], 0)


    upper_coordinates = [*zip(xval[index_upper], upper[index_upper])]
    lower_coordinates = [*zip(xval[index_lower], lower[index_lower])]
    lower_coordinates.reverse()

    poly = Polygon(upper_coordinates+lower_coordinates, **poly_kwargs)
    axes.add_patch(poly)

    if slopeline: axes.plot(xval[index_slope], slope[index_slope], **slopeline_kwargs)
    if errorline:
        axes.plot(xval[index_upper], upper[index_upper], **errorline_kwargs)
        axes.plot(xval[index_lower], lower[index_lower], **errorline_kwargs)
    #axes.show(xlim, [regression.slope*xlim[0]+regression.intercept, regression.slope*xlim[1]+regression.intercept], **kwargs)


def york_regression(X, Y, Xerr, Yerr, r=0, tol=0.00001):
    # Based on the formulas given in York et al 2014
    # Check input
    if not isinstance(r, (float, int)): raise ValueError('r must be an integer or float')
    if r > 1 or r < 0: raise ValueError('r must be between 0 and 1')
    if not isinstance(tol, float): raise ValueError('tol must be a float')

    if not isinstance(X, np.ndarray): X = np.array(X)
    X = X.flatten()

    if not isinstance(Y, np.ndarray): Y = np.array(Y)
    Y = Y.flatten()

    if X.size != Y.size: raise ValueError('X and Y must have the same size')

    if isinstance(Xerr, (float, int)):
        Xerr = np.ones(X.size) * Xerr
    else:
        if not isinstance(Xerr, np.ndarray): Xerr = np.array(Xerr)
        Xerr = Xerr.flatten()
        if Xerr.size != X.size: raise ValueError('Xerr and X must have the same size')

    if isinstance(Yerr, (float, int)):
        Yerr = np.ones(Y.size) * Yerr
    else:
        if not isinstance(Yerr, np.ndarray): Yerr = np.array(Yerr)
        Yerr = Yerr.flatten()
        if Yerr.size != Y.size: raise ValueError('Yerr and Y must have the same size')

    # Do calculations
    WX = 1 / (Xerr ** 2)
    WY = 1 / (Yerr ** 2)

    k = np.sqrt(WX * WY)
    slope = (np.mean(X * Y) - np.mean(X) * np.mean(Y)) / (np.mean(X ** 2) - np.mean(X) ** 2)

    # Iterativley converge self.slope until tol is reached
    for i in range(1000):
        W = (WX * WY) / (WX + (slope ** 2) * WY - 2 * slope * r * k)
        Xbar = np.sum(W * X) / np.sum(W)
        Ybar = np.sum(W * Y) / np.sum(W)
        U = X - Xbar
        V = Y - Ybar
        beta = W * (U / WY + (slope * V) / WX - (slope * U + V) * (r / k))

        b2 = np.sum(W * beta * V) / np.sum(W * beta * U)
        dif = np.sqrt((b2 / slope - 1) ** 2)
        slope = b2
        if dif < tol: break

    # Calculate self.intercept
    intercept = Ybar - slope * Xbar

    # Calcualte adjusted points
    x = Xbar + beta
    xbar = np.sum(W * x) / np.sum(W)
    u = x - xbar

    y = Ybar + slope * beta
    ybar = np.sum(W * y) / np.sum(W)
    v = y - ybar

    slope_error = np.sqrt(1 / np.sum(W * (u ** 2)))
    intercept_error = np.sqrt(1 / np.sum(W) + ((0 - xbar) ** 2) * (slope_error ** 2))

    S = np.sum(W * (Y - slope * X - intercept) ** 2)

    # TODO add MSDW
    return YorkRegression(slope, slope_error, intercept, intercept_error, np.sum(W), xbar)


class YorkRegression:
    def __init__(self, slope, slope_error, intercept, intercept_error, sumW, xbar):
        self.slope = slope
        self.slope_error = slope_error
        self.intercept = intercept
        self.intercept_error = intercept_error
        self.sumW = sumW
        self.xbar = xbar

    def __repr__(self):
        return 'y=({} +- {})x + ({} +- {}); xbar = {}, sum(W) = {}'.format(self.slope, self.slope_error,
                                                                           self.intercept, self.intercept_error,
                                                                           self.xbar, self.sumW)

    def y_error(self, x):
        # TODO if x kist return list
        return np.sqrt(1 / self.sumW + ((x - self.xbar) ** 2) * (self.slope_error ** 2))

    def y(self, x):
        return self.slope * x + self.intercept