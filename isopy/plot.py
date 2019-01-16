import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Polygon
import numpy as np

def york_regression(axes, regression, slopeline = True, errorline = True, shaded = True, shade_kwargs = None, **line_kwargs):
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
    #axes.plot(xlim, [regression.slope*xlim[0]+regression.intercept, regression.slope*xlim[1]+regression.intercept], **kwargs)

