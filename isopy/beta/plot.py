import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Polygon
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import ticker, cm, colors

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
    #axes.show(xlim, [regression.slope*xlim[0]+regression.intercept, regression.slope*xlim[1]+regression.intercept], **kwargs)

def doublespike_uncertianty_grid(x,y,z,title = None, levels = None):
    if levels is None: levels= [0] + [x * 0.0001 for x in range(1, 10)] + [x * 0.001 for x in range(1, 10)] + [x * 0.01 for x in
                                                                                            range(1, 10)] + [0.1]
    if title is None:
        min_z = np.nanmin(z)
        argmin_z = np.nanargmin(z)
        min_x = argmin_z % z.shape[0]
        min_y = int(argmin_z / z.shape[0])
        title = 'Smallest uncertianty: {:.5f} ($2\\sigma/\\sqrt{}n{}$) at x: {:.2f}, y: {:.2f}'.format(min_z, '{', '}', x[min_x], y[min_y])
    min_z = np.nanmin(z)
    argmin_z = np.nanargmin(z)
    min_x = argmin_z % z.shape[0]
    min_y = int(argmin_z / z.shape[0])

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.subplots_adjust(hspace=0.3)
    CS = ax.contourf(x, y, z, levels=levels, cmap=cm.get_cmap(cm.jet), norm=colors.PowerNorm(0.23))
    fig.colorbar(CS, ax=ax, ticks=[0, 0.001, 0.01, 0.1, 1])
    plt.xlabel('Spike fraction in Spike/Sample mix')
    plt.ylabel('Spike1 fraction in Spike1/Spike2 mix')
    plt.title(title)
    plt.show()