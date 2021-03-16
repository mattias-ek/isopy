import isopy as isopy
import isopy.core as core
import isopy.toolbox as toolbox
import scipy.stats as stats
import matplotlib as mpl
from matplotlib.patches import Polygon as mplPolygon
from matplotlib.figure import Figure as mplFigure
import numpy as np
from typing import Optional, Union, Literal

__all__ = ['plot_scatter', 'plot_regression', 'plot_vstack', 'plot_hstack',
           'plot_spider', 'plot_hcompare', 'plot_vcompare',
           'create_subplots', 'update_figure',
           'Markers', 'Colors', 'ColorPairs']

scatter_style = {'markeredgecolor': 'black', 'ecolor': 'black', 'capsize': 3, 'elinewidth': 1, 'zorder': 2}
regression_style = {'linestyle': '-', 'color': 'black', 'edgeline_linestyle': '--', 'fill_alpha': 0.1, 'zorder': 1}
equation_line_style = {'linestyle': '-', 'color': 'black', 'zorder': 1}
stack_style = {'markeredgecolor': 'black', 'ecolor': 'black', 'capsize': 3, 'elinewidth': 1, 'cline_linestyle': '-',
               'pmline_linestyle': '--', 'box_alpha': 0.1, 'zorder': 2}
spider_style = {'markeredgecolor': 'black', 'ecolor': 'black', 'capsize': 3, 'elinewidth': 1, 'zorder': 2}
box_style = dict(color='black', zorder = 0.9, alpha=0.1)
polygon_style = dict(color='black', zorder = 0.9, alpha=0.1)

class RotatingList(list):
    """
    Rotates through a list of *items*

    Is a subclass of ``list``. Only new methods/attributes, or methods/attributes that differ from
    the original implementation, are shown below.
    """
    def __init__(self, items):
        super(RotatingList, self).__init__(items)
        self._original = items[:]
        self.reset()

    def reset(self):
        """Reset to the initial condition"""
        self.clear()
        self.extend(self._original)

    def next(self):
        """Rotate the list to the right and return the new first item."""
        self.append(self.pop(0))
        return self.current

    def previous(self):
        """Rotate the list to the left and return the new first item."""
        self.insert(0, self.pop(-1))
        return self.current

    @property
    def current(self):
        """The first item in the list"""
        return self[0]


class Markers(RotatingList):
    """
    Rotate through a set of markers.

    The markers, in the inital order, are circle, square, triangle, diamond, plus, cross, pentagon.

    Is a subclass of ``list``. Only new methods/attributes, or methods/attributes that differ from
    the original implementation, are shown below.

    Examples
    --------
    >>> markers = isopy.tb.Colors()
    >>> for i in range(len(markers)):
    >>>     isopy.tb.plot_scatter(plt, 1, i, marker=markers.current, markersize=20)
    >>>     markers.next()
    >>> plt.show()

    .. figure:: Markers.png
            :alt: output of the example above
            :align: center
    """
    def __init__(self,):
        super(Markers, self).__init__(['o', 's', '^', 'D', 'P', 'X', 'p'])


class ColorPairs(RotatingList):
    """
    Rotate through a set of colour pairs consisting of a matching dark and pale colours.

    The colours, in the inital order, are dark and pale shades of blue, red, green, orange, purple, brown

    Is a subclass of ``list``. Only new methods/attributes, or methods/attributes that differ from
    the original implementation, are shown below.

    Examples
    --------
    >>> colorpairs = isopy.tb.ColorPairs()
    >>> for i in range(len(colorpairs)):
    >>>     isopy.tb.plot_scatter(plt, [1, 1.5], [i, i], color=colorpairs.current[0], markersize=20, line=True)
    >>>     isopy.tb.plot_scatter(plt, [2, 2.5], [i, i], color=colorpairs.current[1], markersize=20, line=True)
    >>>     colorpairs.next()
    >>> plt.show()

    .. figure:: ColorPairs.png
            :alt: output of the example above
            :align: center
    """
    def __init__(self):
        super(ColorPairs, self).__init__([('#1f78b4', '#a6cee3'), ('#e31a1c', '#fb9a99'),('#33a02c', '#b2df8a'),
                            ('#ff7f00', '#fdbf6f'), ('#6a3d9a', '#cab2d6'), ('#b15928', '#ffff99')])


class Colors(RotatingList):
    """
    Rotate through a set of colours.

    The colours pairs, in the inital order, are shades of blue, red, green, orange, purple,
    brown, yellow, pink, grey.

    Is a subclass of ``list``. Only new methods/attributes, or methods/attributes that differ from
    the original implementation, are shown below.

    Examples
    --------
    >>> colors = isopy.tb.Colors()
    >>> for i in range(len(colors)):
    >>>     isopy.tb.plot_scatter(plt, 1, i, color=colors.current, markersize=20)
    >>>     colors.next()
    >>> plt.show()

    .. figure:: Colors.png
            :alt: output of the example above
            :align: center
    """
    def __init__(self):
        super(Colors, self).__init__(['#377eb8', '#e41a1c', '#4daf4a', '#ff7f00', '#984ea3',
                                      '#a65628', '#ffff33', '#f781bf', '#999999'])


def update_figure(figure, figwidth=None, figheight=None, dpi=None, facecolor=None, edgecolor=None,
                  frameon=None, tight_layout=None, constrained_layout=None):
    """
    Update the figure options.

    See the matplotlib documentation
    `here <https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.html>`_
    for more details on these options.

    Parameters
    ----------
    figure : figure, plt
        A matplotlib Figure object or any object
        with a ``.gcf()`` method that returns a matplotlib Figure object, Such as a matplotlib
        pyplot instance.
    figwidth : scalar
        Width of *figure* in inches
    figheight : scalar
        Width of *figure* in inches
    dpi : scalar
        Resolution of *figure* in dots per inch
    facecolor : str
        The face colour of *figure*
    edgecolor : str
        The edge colours of *figure*
    frameon : bool
        If ``False`` does not draw the figure background colour
    tight_layout : bool
        If ``True`` use ``tight_layout`` when drawing the figure.
    constrained_layout : bool
        If ``True`` use ``constrained_layout`` when drawing the figure.

    Returns
    -------
    figure : Figure
        Returns the figure
    """
    figure = _check_figure('figure', figure)
    if figwidth is not None: figure.set_figwidth(figwidth)
    if figheight is not None: figure.set_figheight(figheight)
    if dpi is not None: figure.set_dpi(dpi)
    if facecolor is not None: figure.set_facecolor(facecolor)
    if edgecolor is not None: figure.set_edgecolor(edgecolor)
    if frameon is not None: figure.set_frameon(frameon)
    if tight_layout is not None: figure.set_tight_layout(tight_layout)
    if constrained_layout is not None: figure.set_constrained_layout(constrained_layout)
    return figure

def create_subplots(figure, subplots, *, constrained_layout=True, **figure_options):
    """
    Create subplots for a figure.

    Parameters
    ----------
    figure : figure, plt
        A matplotlib Figure object or any object
        with a ``.gcf()`` method that returns a matplotlib Figure object, Such as a matplotlib
        pyplot instance.
    subplots : str, list
        A nested list of subplot names. A visual layout of how you want your subplots
        to be arranged labeled as strings. *subplots* must be a 2-dimensional but items in the
        list can contain futher 2-dimensional lists. Use ``None`` for empty spaces.
    constraned_layout : bool
        If ``True`` use ``constrained_layout`` when drawing the figure. Advised to avoid
        overlapping axis labels.
    figure_options
        Any valid option for :func:`update_figure`.

    Returns
    -------
    subplots : dict
        A dictionary containing the freshly created subplots.

    Examples
    --------
    >>> subplots = [['one', 'two', 'three', 'four']]
    >>> axes = isopy.tb.create_subplots(plt, subplots)
    >>> for name, ax in axes.items(): ax.set_title(name)
    >>> plt.show()

    .. figure:: create_subplots1.png
            :alt: output of the example above
            :align: center


    >>> subplots = [['one'], ['two'], ['three'], ['four']]
    >>> axes = isopy.tb.create_subplots(plt, subplots)
    >>> for name, ax in axes.items(): ax.set_title(name)
    >>> plt.show()

    .. figure:: create_subplots2.png
        :alt: output of the example above
        :align: center


    >>> subplots = [[ [['one', 'two']] ], ['three'], ['four']]
    >>> axes = isopy.tb.create_subplots(plt, subplots)
    >>> for name, ax in axes.items(): ax.set_title(name)
    >>> plt.show()

    .. figure:: create_subplots3.png
            :alt: output of the example above
            :align: center
    """
    figure = _check_figure('figure', figure)
    update_figure(figure, constrained_layout=constrained_layout, **figure_options)
    axes = figure.subplot_mosaic(subplots, empty_sentinel=None)

    #Make sure that no key strings are used.
    out = {}
    for key, ax in axes.items():
        ax.set_label(str(key))
        out[str(key)] = ax
    return out


def plot_scatter(axes, x, y, xerr = None, yerr = None,
                 color= None, marker = True, line = False, regression = None, **kwargs):
    """
    Plot data points with error bars on *axes*.


    Parameters
    ----------
    axes : axes, plt
        The axes on which the data points will we plotted. Must be a matplotlib axes object or any
        object with a gca() method that return a matplotlib axes object, such as a matplotlib
        pyplot instance.
    x, y : numpy_array_like
        Any object that can be converted to a numpy array. Multidimensional
        array will be flattened. *x* and *y* must be the same size.
    xerr, yerr : scalar, numpy_array_like
        Uncertainties associated with *x* and *y* values. Can be any object that can be converted to a
        numpy array. Multidimensional array will be flattened. Must either have the same size as *x*
        and *y* or be a single value which will be used for all datapoints. If ``None`` no
        errorbars are shown.
    color : str, Optional
        Color of the marker face and line between points, if *marker* and *line* are not ``False``.
        Accepted strings are named colour in matplotlib or a string of a hex triplet begining with "#".
        See `here <https://matplotlib.org/stable/gallery/color/named_colors.html>`_ for a list of
        named colours in matplotlib. If not given the next colour on the internal matplotlib colour
        cycle is used.
    marker : bool, str, Default = True
        If ``True`` a marker is shown for each data point. If ``False`` no marker is shown. Can also be a
        string describing any marker accepted by matplotlib. See `here
        <https://matplotlib.org/stable/api/markers_api.html>`_ for a list of avaliable markers.
    line : bool, str, Default = False
        If ``True`` a line is drawn between each datapoint. If ``False`` no line is shown. Can also
        be a string describing a linestyle defined by matplotlib. See `here
        <https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html>`_ for a list
        of avaliable linestyles. ``True`` defaults to ``"solid"``.
    regression : {'york1', 'york2', 'linear'}, Optional
        A string with the name of a regression to be applied to the data and then plotted. Valid
        options are *york1*, *york2*, and *linear*.
    kwargs
        Any keyword argument accepted by matplotlib axes method *errorbar*. See `here
        <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.errorbar.html>`_
        for a list of keyword arguments.


    Examples
    --------
    >>> x = np.random.default_rng().normal(1, 0.5, 20)
    >>> y = (x * 3) + np.random.default_rng().normal(2, 0.5, 20)
    >>> isopy.tb.plot_scatter(plt, x, y)
    >>> plt.show()

    .. figure:: plot_scatter1.png
            :alt: output of the example above
            :align: center

    >>> xerr = 0.2
    >>> yerr = np.random.default_rng().normal(0.3, 0.1, 20)
    >>> isopy.tb.plot_scatter(plt, x, y, xerr, yerr, regression='york1', color='red', marker='s')
    >>> plt.show()

    .. figure:: plot_scatter2.png
        :alt: output of the example above
        :align: center
    """
    if isinstance(x, tuple) and len(x) == 2: x, xerr = x
    if isinstance(y, tuple) and len(y) == 2: y, yerr = y
    axes = _check_axes(axes)

    x = _check_val('x', x)
    xerr = _check_err('xerr', x, xerr)
    y = _check_val('y', y)
    yerr = _check_err('yerr', y, yerr)
    if x.size != y.size:
        raise ValueError(f'size of x ({x.size}) does not match size of y ({y.size})')

    style = {}
    #Dont use prop_cyclet directly as the default style
    #It resulted in wierd results i could not explain
    style.update(scatter_style)
    style.update(kwargs)

    if isinstance(color, tuple): style['color'] = color[0]
    elif color: style['color'] = color
    else: style.setdefault('color', next(axes._get_lines.prop_cycler).get('color', 'blue'))

    if marker is True: style.setdefault('marker', 'o')
    elif marker is False: style['marker'] = ''
    elif marker: style['marker'] = marker

    if line is True: style.setdefault('linestyle', 'solid')
    elif line is False: style['linestyle'] = ''
    elif line: style['linestyle'] = line

    style.setdefault('markerfacecolor', style['color'])
    style.setdefault('zorder', 2)

    axes.errorbar(x, y, yerr, xerr, **style)
    _axes_add_data(axes, x, y, xerr, yerr)

    if regression is not None:
        if regression == 'york1':
            regression_result = isopy.tb.regression_york1(x, y, xerr, yerr)
        elif regression == 'york2':
            regression_result = isopy.tb.regression_york2(x, y, xerr, yerr)
        elif regression == 'linear':
            regression_result = isopy.tb.regression_linear(x, y)
        else:
            raise ValueError(f'"{regression}" not a valid regression')
        plot_regression(axes, regression_result, color=style.get('color', None), label=('label' in style))

def plot_regression(axes, regression_result, color=None, line=True, xlim = None, autoscale = False,
                    fill = True, edgeline = True,  **kwargs):
    """
    Plot the result of a regression on matplotlib *axes*.

    If *regression_result* has a ``.yerr(x) -> yerr`` method the error envelope of the regression
    will also be plotted.

    Parameters
    ----------
    axes : axes, plt
        If *y* is a numpy array *axes* must be a matplotlib axes object or
        object with a gca() method that return a matplotlib axes object, such as a matplotlib
        pyplot instance. If *y* is an isopy array *axes* must be a matplotlib Figure or any object
        with a ``.gcf()`` method that returns a matplotlib Figure object, Such as a matplotlib
        pyplot instance.
    regression_result
        Any object returned by one of isopy's regression functions, an object with ``.slope`` and
        ``.intercept`` attributes or a callable object which takes the x value and returns the y
        value.
    color : str, Optional
        Color of the regression line if *line* is not ``False``.
        Accepted strings are named colour in matplotlib or a string of a hex triplet begining with "#".
        See `here <https://matplotlib.org/stable/gallery/color/named_colors.html>`_ for a list of
        named colours in matplotlib. If not given the next colour on the internal matplotlib colour
        cycle is used.
    line : bool, str, Default = True
        If ``True`` a the regression line is drawn within *xlim*.
        If ``False`` no line is shown. Can also
        be a string describing a linestyle defined by matplotlib. See `here
        <https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html>`_ for a list
        of avaliable linestyles. ``True`` defaults to ``"solid"``.
    xlim : tuple[int, int]
        A tuple of the lower and upper x value for which to plot the regression line.
        If ``None`` the current xlim of the axes is used.
    autoscale : bool, Default = False
        If ``False`` the regression will not be taken into account when figure is rescaled.
        This if not officially supported by matplotlib and thus might not always work.
    fill : bool, Default = True
        If ``True`` the error enveloped is filled in with *color*. If ``False`` the error envelope
        is not filled in.
    edgeline : bool, Default = True
        If ``True`` the edges of the error envelope are drawn as a line.
        If ``False`` no line is shown. Can also
        be a string describing a linestyle defined by matplotlib. See `here
        <https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html>`_ for a list
        of avaliable linestyles. ``True`` defaults to ``"dashed"``.
    label : bool, str
        If ``True`` a legend label in the form of *y=mc+c* used when creating the regression *line*.
        If *label* is a string that string will be used as the legend label.
    kwargs
        Any keyword argument accepted by matplotlib axes methods. By default kwargs are
        attributed to the regression line. Prefix kwargs for the
        fill area with ``fill_`` and kwargs for the edge lines with ``edgeline_``.
        See `here
        <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html>`_
        for a list of keyword arguments for the regression line and edge lines.
        See `here
        <https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Polygon.html>`_
        for a list of keyword arguments for the fill area.

    Examples
    --------
    >>> x = np.random.default_rng().normal(1, 0.5, 20)
    >>> y = x* 3 + np.random.default_rng().normal(2, 0.5, 20)
    >>> regression = isopy.tb.regression_linear(x, y)
    >>> isopy.tb.plot_scatter(plt, x, y)
    >>> isopy.tb.plot_regression(plt, regression)
    >>> plt.show()

    .. figure:: plot_regression1.png
            :alt: output of the example above
            :align: center

    >>> regression = lambda x: x ** 2 + 4 #Any callable that takes x and return y is a valid
    >>> isopy.tb.plot_scatter(plt, x, y, color='red')
    >>> isopy.tb.plot_regression(plt, regression, color='red', xlim=(0.6, 1.8))
    >>> plt.show()

    .. figure:: plot_regression2.png
            :alt: output of the example above
            :align: center

    >>> xerr = 0.2
    >>> yerr = np.random.default_rng().normal(0.3, 0.1, 20)
    >>> regression = isopy.tb.regression_york1(x, y, xerr, yerr)
    >>> isopy.tb.plot_scatter(plt, x, y, xerr, yerr)
    >>> isopy.tb.plot_regression(plt, regression)
    >>> plt.show()

    .. figure:: plot_regression3.png
            :alt: output of the example above
            :align: center

    >>> isopy.tb.plot_scatter(plt, x, y, xerr, yerr, color='red')
    >>> isopy.tb.plot_regression(plt, regression, color='red', line='dashed', edgeline=False)
    >>> plt.show()

    .. figure:: plot_regression4.png
            :alt: output of the example above
            :align: center
    """
    axes = _check_axes(axes)

    if callable(regression_result):
        y_eq = regression_result
    elif hasattr(regression_result, 'slope') and hasattr(regression_result, 'intercept'):
        y_eq = lambda x: x * regression_result.slope + regression_result.intercept
    elif isinstance(regression_result, tuple) and len(regression_result) == 2:
        y_eq = lambda x: x * regression_result[0] + regression_result[1]

    yerr_eq = getattr(regression_result, 'yerr', None)
    if not callable(yerr_eq): yerr_eq = None

    style = {}
    style.update(regression_style)
    style.update(kwargs)

    if isinstance(color, tuple):
        style['color'] = color[0]
    elif color:
        style['color'] = color
    else:
        style.setdefault('color', 'black')

    if line is True:
        style.setdefault('linestyle', 'solid')
    elif line is False:
        style['linestyle'] = ''
    elif line:
        style['linestyle'] = line

    if edgeline is True: style.setdefault('edgeline_linestyle', '--')
    elif edgeline is False: style['edgeline_linestyle'] = ''
    elif edgeline: style['edgeline_linestyle'] = edgeline

    style.setdefault('edgeline_color', style['color'])
    style.setdefault('fill_color', style['color'])
    style.setdefault('zorder', 2)
    style.setdefault('edgeline_zorder', style['zorder']-0.001)
    style.setdefault('fill_zorder', style['zorder']-0.002)

    fill_style = _extract_style(style, 'fill')
    edge_style = _extract_style(style, 'edgeline')

    if not xlim:
        xlim = axes.get_xlim()

    if (label:=style.get('label', None)) is True:
        sigfig = 2
        if isinstance(regression_result, toolbox.regress.YorkregressResult):
            style['label'] = regression_result.label(sigfig)
        else:
            label_intercept = y_eq(0)
            label_slope = y_eq(1) - label_intercept
            var = y_eq(xlim[1]) - y_eq(xlim[0])

            style['label'] = f'y={_format_sigfig(label_slope, sigfig, var)}x + ' \
                             f'{_format_sigfig(label_intercept, sigfig, var)}'
    elif label is False:
        style.pop('label')

    x = np.linspace(xlim[0], xlim[1], 10000)
    y = np.array([y_eq(xval) for xval in x])

    if yerr_eq is not None:
        yerr = np.array([yerr_eq(xval) for xval in x])
        yerr_upper = y + yerr
        yerr_lower = y - yerr
        yerr_coordinates = [*zip(x, yerr_upper)] + list(reversed([*zip(x, yerr_lower)]))

    ull_function = None
    if not autoscale and hasattr(axes, '_update_line_limits'):
        ull_function = axes._update_line_limits
        axes._update_line_limits = lambda *args, **kwargs: None

    try:
        if line is not False:
            axes.plot(x, y, scalex=False, scaley=False, **style)

        if yerr_eq is not None and edgeline:
            axes.plot(x, yerr_upper, scalex=False, scaley=False, **edge_style)
            axes.plot(x, yerr_lower, scalex=False, scaley=False, **edge_style)
    finally:
        if ull_function:
            axes._update_line_limits = ull_function

    if yerr_eq is not None and fill:
        _plot_polygon(axes, yerr_coordinates, autoscale, **fill_style)


def plot_spider(axes, y, yerr = None, x = None, constants = None, xscatter  = None, color = None, marker = True, line = True, **kwargs):
    """
    Plot data as a spider diagram on matplotlib *axes* .


    Parameters
    ----------
    axes : axes, plt
        The axes on which the data points will we plotted. Must be a matplotlib axes object or any
        object with a gca() method that return a matplotlib axes object, such as a matplotlib
        pyplot instance.
    y : numpy_array_like, dict, isopy_array
        Any object that can be converted to an isopy array or an
        numpy array. Multidimensional array will be flattened. Will be sorted by
        increasing *x* values before plotted.
    yerr : numpy_array_like, dict, isopy_array
        Uncertianties associated with *y* values. Can be any object that can be converted to a
        numpy array. Multidimensional array will be flattened. Must either be an array with the same
        size as *y* or a single value which will be used for all values of *y*.
        If ``None`` no errorbars are shown.
    x : numpy_array_like, isopy_key_list, Optional
        Any object or sequence of object that can be converted to float values. If *x* is an isotope key list the nass number
        of each key is used, for ratio key lists the numerator key string is used. An exception is thrown if there is no
        common denominator. If ``None`` and *y* is an isopy array then *x* will be inferred from the key strings.
    constants : dict[scalar, scalar], Optional
        A dictionary mapping constant y values to their x value. Both the key and the value must be
        scalars.
    xscatter : scalar, Optional
        If given, *xscatter* is used to create variation in the x axis values used for each
        successive set of y values, within the range (*X* - *xscatter*, *x*+*xscatter*). This helps
        differentiate samples that all pass through the same point.
    color : str, Optional
        Color of the marker face and line between points, if *marker* and *line* are not ``False``.
        Accepted strings are named colour in matplotlib or a string of a hex triplet begining with "#".
        See `here <https://matplotlib.org/stable/gallery/color/named_colors.html>`_ for a list of
        named colours in matplotlib. If not given the next colour on the internal matplotlib colour
        cycle is used.
    marker : bool, str, Default = True
        If ``True`` a marker is shown for each data point. If ```False`` no marker is shown. Can also be a
        string describing any marker accepted by matplotlib. See `here
        <https://matplotlib.org/stable/api/markers_api.html>`_ for a list of avaliable markers.
    line : bool, str, Default = True
        If ``True`` a line is drawn between each datapoint. If ``False`` no line is shown. Can also
        be a string describing a linestyle defined by matplotlib. See `here
        <https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html>`_ for a list
        of avaliable linestyles. ``True`` defaults to ``"solid"``.
    kwargs
        Any keyword argument accepted by matplotlib axes method *errorbar*. See `here
        <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.errorbar.html>`_
        for a list of keyword arguments.

    Examples
    --------
    >>> array = isopy.tb.make_ms_array('pd')
    >>> isopy.tb.plot_spider(plt, array) #Will plot the fraction of each Pd isotope
    >>> plt.show()

    .. figure:: plot_spider1.png
        :alt: output of the example above
        :align: center


    >>> subplots = isopy.tb.create_subplots(plt, [['left', 'right']], figwidth=8)
    >>> array = isopy.tb.make_ms_array('pd', mf_factor = [0.001, 0.002]).ratio('105pd')
    >>> array = isopy.toolbox.isotope.normalise_data(array, isopy.refval.isotope.abundance, 1000)
    >>> isopy.tb.plot_spider(subplots['left'], array) #The numerator mass numbers are used as x
    >>> isopy.tb.plot_spider(subplots['right'], array, constants={105: 0}) #Adds a zero for the denominator mass number
    >>> plt.show()

    .. figure:: plot_spider2.png
        :alt: output of the example above
        :align: center

    >>> subplots = isopy.tb.create_subplots(plt, [['left', 'right']], figwidth=8)
    >>> values = {100: [1, 1, -1], 101.5: [0, 0, 0], 103: [-1, 1, -1]} #keys can be floats
    >>> isopy.array(values)
    >>> isopy.tb.plot_spider(subplots['left'], values) #Impossible to tell the rows apart
    >>> isopy.tb.plot_spider(subplots['right'], values, xscatter=0.15) #Much clearer
    >>> plt.show()

    .. figure:: plot_spider3.png
        :alt: output of the example above
        :align: center
    """
    axes = _check_axes(axes)
    if type(y) is tuple and len(y) == 2: y, yerr = y

    try:
        y = isopy.asanyarray(y)
    except Exception as err:
        raise TypeError(f'invalid y: {err}') from err

    if isinstance(y, isopy.core.IsopyArray):
        if x is None:
            x = y.keys()

        if isinstance(yerr, isopy.core.IsopyArray):
            if type(yerr) is not type(y):
                raise TypeError(f'yerr ({type(yerr).__name__}) is not the same type as y ({type(y).__name__})')
            else:
                yerr = np.array([yerr.get(key) for key in y.keys()]).T
        y = np.array(y.to_list())

    if x is None:
        raise ValueError('x values could not be inferred from the y values')

    if type(x) is isopy.RatioKeyList:
        if not x.has_common_denominator:
            raise ValueError(
                'x values can only be inferred from a ratio key list with a common denominator')
        else:
            x = x.numerators

    if type(x) is isopy.IsotopeKeyList:
        x = x.mass_numbers

    try:
        x = np.asarray(x, dtype=float)
    except Exception as err:
        raise TypeError(f'invalid x: {err}') from err

    if x.ndim == 0: x = x.reshape(1, -1)
    elif x.ndim == 1: x = x.reshape(1, -1)
    elif x.ndim > 2:
        raise TypeError(f'x has more than two dimensions ({x.ndim})')

    if y.ndim == 0: y = y.reshape(1, -1)
    elif y.ndim == 1: y = y.reshape(1, -1)
    elif y.ndim > 2:
        raise TypeError(f'y has more than two dimensions ({y.ndim})')

    if yerr is not None:
        yerr = np.asarray(yerr)
        if yerr.ndim < 2: yerr = yerr.reshape(1, -1)
        elif yerr.ndim > 2:
            raise TypeError(f'yerr has more than two dimensions ({yerr.ndim})')

    if x.shape[1] != y.shape[1]:
        raise ValueError(f'size of x ({x.shape[1]}) and y ({y.shape[1]}) do not match')

    if constants:
        if not isinstance(constants, dict):
            raise ValueError('constants must be a dictionary')
        xconstants = np.array(list(constants.keys()), dtype=np.float64)
        yconstants = np.array(list(constants.values()), dtype=np.float64)

        if xconstants.ndim != 1 or yconstants.ndim != 1:
            raise ValueError(f'constants must be scalar values')
        xconstants.reshape(1, -1)
        yconstants.reshape(1, -1)

        x = np.append(x, np.full((x.shape[0], xconstants.size), xconstants), axis = 1)
        y = np.append(y, np.full((y.shape[0], yconstants.size), yconstants), axis = 1)
        if yerr is not None:
            yerr = np.append(yerr, np.full((yerr.shape[0], constants.size), np.nan), axis = 1)

    if xscatter and not isinstance(xscatter, (float, int)):
        raise TypeError(f'xscatter must be a float or an integer')

    if x.shape[0] != y.shape[0]:
        if x.shape[0] == 1:
            if xscatter:
                x = x + np.linspace(-1 * xscatter, xscatter, y.shape[0]).reshape(-1, 1)
            else:
                x = x + np.zeros(y.shape)
        else:
            raise ValueError(f'shape of x ({x.shape[0]}) and y ({y.shape[0]}) do not match')

    if yerr is not None and y.shape[1] != yerr.shape[1]:
        raise ValueError(f'size of y ({y.shape[1]}) and yerr ({yerr.shape[1]}) do not match')
    if yerr is not None and y.shape[0] != yerr.shape[0]:
        raise ValueError(f'shape of y ({y.shape[0]}) and yerr ({yerr.shape[0]}) do not match')
    if yerr is None:
        yerr = [None for i in range(y.shape[0])]

    style = {}
    style.update(spider_style)
    style.update(kwargs)

    if isinstance(color, tuple): style['color'] = color[0]
    elif color: style['color'] = color
    else: style.setdefault('color', next(axes._get_lines.prop_cycler).get('color', 'blue'))

    if marker is True: style.setdefault('marker', 'o')
    elif marker is False: style['marker'] = ''
    elif marker: style['marker'] = marker

    if line is True: style.setdefault('linestyle', '-')
    elif line is False: style['linestyle'] = ''
    elif line: style['linestyle'] = line

    style.setdefault('markerfacecolor', style['color'])
    style.setdefault('zorder', 2)

    regress_style = _extract_style(style, 'regress')
    regress_style.setdefault('color', style['color'])

    for i in range(x.shape[0]):
        xi = x[i]
        # Sort the arrays so that they are in order of increasing x
        order = sorted([j for j in range(x.shape[1])], key=lambda j: xi[j])
        if yerr[i] is None: yerri = None
        else: yerri = yerr[i][order]

        axes.errorbar(xi[order], y[i][order], yerr=yerri, **style)
        style.pop('label', None)

    _axes_add_data(axes, x, y, None, yerr=yerr)


def plot_vstack(axes, x, xerr = None, ystart = None, *, outliers=None, cval = None, pmval=None, color=None, marker=True, line=False,
                cline = True, pmline= False, box = True, pad = 2, box_pad = None,
                spacing = -1, hide_yaxis=True, **kwargs):
    """
    Plot values along the x-axis at a regular y interval. Also known as Caltech or Carnegie plot.

    Can also take an array in which case it will create a grid of subplots, one for each column in
    the array.


    Parameters
    ----------
    axes : axes, figure, plt
        If *x* is a numpy array *axes* must be a matplotlib axes object or
        object with a gca() method that return a matplotlib axes object, such as a matplotlib
        pyplot instance. If *x* is an isopy array *axes* must be a matplotlib Figure or any object
        with a ``.gcf()`` method that returns a matplotlib Figure object, Such as a matplotlib
        pyplot instance.
    x : numpy_array_like, isopy_array
        Any object that can be converted to a numpy array. Multidimensional
        array will be flattened. 'x* and *y* must be the same size.
    xerr : numpy_array_like, isopy_array, Optional
        Uncertianties associated with *y* values. Must be an object that can be converted to a
        numpy array. Multidimensional array will be flattened. Must either be an array with the same
        size as *y* or a single value which will be used for all values of *y*.
        If ``None`` no errorbars are shown.
    ystart : scalar, Optional
        The starting valeu on the yaxis. If ``None`` it is inferred from values already plotted on
        the *axes*. If *x* is an isopy array the *ystart* will be the same for all subplots.
    outliers : bool, Optional
        A boolean array same size as *x* where ``True`` means a data point is
        considered an outlier.
    cval : scalar, Callable, Optional
        The center value of the data. Can either be a scalar or a function that returns a scalar
        when called with *x*. If *cval* is a function is it called on the values that are not
        outliers.
    pmval : scalar, Callable, Optional
        The uncertainty of the the data. Can either be a scalar or a function that returns a scalar
        when called with *x*. If *pmval* is a function is it called on the values that are not
        outliers.
    color : str, Optional
        Color of the marker face and line between points, if *marker* and *line* are not ``False``.
        Accepted strings are named colour in matplotlib or a string of a hex triplet begining with "#".
        See `here <https://matplotlib.org/stable/gallery/color/named_colors.html>`_ for a list of
        named colours in matplotlib. If not given the next colour on the internal matplotlib colour
        cycle is used.
    marker : bool, str, Default = True
        If ``True`` a marker is shown for each data point. If ```False`` no marker is shown. Can also be a
        string describing any marker accepted by matplotlib. See `here
        <https://matplotlib.org/stable/api/markers_api.html>`_ for a list of avaliable markers.
    line : bool, str, Default = False
        If ``True`` a line is drawn between each datapoint. If ``False`` no line is shown. Can also
        be a string describing a linestyle defined by matplotlib. See `here
        <https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html>`_ for a list
        of avaliable linestyles. ``True`` defaults to ``"solid"``.
    cline : bool, str, Default = True
        If ``True`` a center line is drawn at *cval* alongside the data points. If ``False`` no line is shown.
        Can also be a string describing a linestyle defined by matplotlib. See `here
        <https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html>`_ for a list
        of avaliable linestyles. ``True`` defaults to ``"solid"``. Only shown if *cval* is given.
    pmline : bool, str, Default = False
        If ``True`` two lines at the upper uncertainty (``cval+pmval``) and lower uncertainty
        (``cval-pmval``) alongside the data points, . If ``False`` no line is shown.
        Can also be a string describing a linestyle defined by matplotlib. See `here
        <https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html>`_ for a list
        of avaliable linestyles. ``True`` defaults to ``"solid"``. Only shown if *cval* and *pmval*
        are given.
    box : bool, Default = True
        If ``True`` a semitransparent box is drawn alongisde the data points basedon the *cval*
        and *pmval* values.
    pad : scalar, Default = 2
        The space between previous data points and the new data points on the y-axis.
    box_pad : scalar, Optional
        The space the cline, pmline and box extends past the data points on the y-axis. If ``None``
        a 1/4 of *pad* is used.
    spacing : scalar, Default = 1
        The space between data points on the y-axis.
    hide_yaxis : bool, Default = True
        Hides the y-axis ticks and label.
    kwargs
        Any keyword argument accepted by matplotlib axes methods. By default kwargs are
        attributed to the data points. Prefix kwargs for the center line
        with ``cline_``, kwargs for the  plus/minus lines with ``pmline_`` and kwargs for the
        box area with ``box_``.
        See `here
        <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.errorbar.htm>`_
        for a list of keyword arguments for the data points.
        See `here
        <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html>`_
        for a list of keyword arguments for the center and plus/minus lines.
        See `here
        <https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Polygon.html>`_
        for a list of keyword arguments for the box area.

    Examples
    --------
    >>> array1 = isopy.tb.make_ms_beams('pd', integrations = 20)
    >>> array2 = isopy.tb.make_ms_beams('pd', integrations = 20)
    >>> isopy.tb.plot_vstack(plt, array1['105pd'])
    >>> isopy.tb.plot_vstack(plt, array2['105pd'], cval=np.mean, pmval=isopy.sd2)
    >>> plt.show()

    .. figure:: plot_vstack1.png
        :alt: output of the example above
        :align: center

    >>> keys = isopy.keylist('pd105', 'ru101', 'cd111')
    >>> isopy.tb.create_subplots(plt, [[key for key in keys.sorted()]], figwidth=8)
    >>> array1 = isopy.tb.make_ms_beams('pd105', 'cd111', integrations = 20)
    >>> array2 = isopy.tb.make_ms_beams('pd105', 'ru101', integrations = 20)
    >>> isopy.tb.plot_vstack(plt, array1, color=isopy.tb.Colors()[0])
    >>> isopy.tb.plot_vstack(plt, array2, cval=np.mean, pmval=isopy.sd2, color=isopy.tb.Colors()[1])
    >>> plt.show()

    .. figure:: plot_vstack2.png
        :alt: output of the example above
        :align: center


    >>> isopy.tb.update_figure(plt, figwidth=8)
    >>> colorpairs = isopy.tb.ColorPairs()
    >>> array = isopy.tb.make_ms_beams('pd104', 'pd105', 'pd106')
    >>> mean = np.mean(array)
    >>> sd2 = isopy.sd2(array)
    >>> outliers = isopy.toolbox.isotope.find_outliers(array, mean, sd2)
    >>> isopy.tb.plot_vstack(plt, array, outliers=outliers, cval=mean, pmval=sd2, color=colorpairs.current)
    >>> plt.show()

    .. figure:: plot_vstack3.png
        :alt: output of the example above
        :align: center

    """
    if isinstance(x, tuple) and len(x) == 2:
        x, xerr = x

    if isinstance(x, core.IsopyArray):
        _plot_stack_array(plot_vstack, axes, 'x', x=x, xerr=xerr, ystart=ystart, cval=cval,
                          pmval=pmval, color=color, marker=marker, line=line, cline=cline,
                          pmline=pmline, box=box, pad=pad, box_pad=box_pad, spacing=spacing,
                          outliers=outliers, hide_yaxis=hide_yaxis, **kwargs)

    else:
        axes = _check_axes(axes)
        x = _check_val('x', x)
        xerr = _check_err('xerr', x, xerr)
        outliers = _check_mask('outliers', x, outliers, default_value=False)

        if ystart is None:
            if spacing > 0:
                ystart = _axes_data_func(axes, 'y', np.nanmax, default_value=np.nan) + pad
            else:
                ystart = _axes_data_func(axes, 'y', np.nanmin, default_value=np.nan) - pad
        if np.isnan(ystart): ystart = 0

        x, y, xerr, cx, cy, pmx, pmy, boxx, boxy, styles = _plot_stack_step1(axes, x, xerr, ystart, cval, pmval, color,
                                                                             marker, line, cline, pmline, box, pad, box_pad, spacing, outliers, **kwargs)

        axes.get_yaxis().set_visible(not hide_yaxis)
        _plot_stack_step2(axes, x, y, xerr, None, cx, cy, pmx, pmy, boxx, boxy, styles, outliers)

def plot_hstack(axes, y, yerr = None, xstart = None, *, outliers=None, cval = None, pmval=None, color=None, marker=True, line=False,
                cline = True, pmline= False, box = True, pad = 2, box_pad = None,
                spacing = 1, hide_xaxis=True, **kwargs):
    """
    Plot values along the x-axis at a regular y interval. Also known as Caltech or Carnegie plot.

    Can also take an array in which case it will create a grid of subplots, one for each column in
    the array.


    Parameters
    ----------
    axes : axes, figure, plt
        If *y* is a numpy array *axes* must be a matplotlib axes object or
        object with a gca() method that return a matplotlib axes object, such as a matplotlib
        pyplot instance. If *y* is an isopy array *axes* must be a matplotlib Figure or any object
        with a ``.gcf()`` method that returns a matplotlib Figure object, Such as a matplotlib
        pyplot instance.
    y : numpy_array_like, isopy_array
        Any object that can be converted to a numpy array. Multidimensional
        array will be flattened. *x* and *y* must be the same size.
    yerr : numpy_array_like, isopy_array, Optional
        Uncertianties associated with *y* values. Must be an object that can be converted to a
        numpy array. Multidimensional array will be flattened. Must either be an array with the same
        size as *y* or a single value which will be used for all values of *y*.
        If ``None`` no errorbars are shown.
    xstart : scalar, Optional
        The starting value on the yaxis. If ``None`` it is inferred from values already plotted on
        the *axes*. If *y* is an isopy array the *xstart* will be the same for all subplots.
    outliers : bool, Optional
        A boolean array same size as *y* where ``True`` means a data point is
        considered an outlier.
    cval : scalar, Callable, Optional
        The center value of the data. Can either be a scalar or a function that returns a scalar
        when called with *x*. If *cval* is a function is it called on the values that are not
        outliers.
    pmval : scalar, Callable, Optional
        The uncertainty of the the data. Can either be a scalar or a function that returns a scalar
        when called with *x*. If *pmval* is a function is it called on the values that are not
        outliers.
    color : str, Optional
        Color of the marker face and line between points, if *marker* and *line* are not ``False``.
        Accepted strings are named colour in matplotlib or a string of a hex triplet begining with "#".
        See `here <https://matplotlib.org/stable/gallery/color/named_colors.html>`_ for a list of
        named colours in matplotlib. If not given the next colour on the internal matplotlib colour
        cycle is used.
    marker : bool, str, Default = True
        If ``True`` a marker is shown for each data point. If ```False`` no marker is shown. Can also be a
        string describing any marker accepted by matplotlib. See `here
        <https://matplotlib.org/stable/api/markers_api.html>`_ for a list of avaliable markers.
    line : bool, str, Default = False
        If ``True`` a line is drawn between each datapoint. If ``False`` no line is shown. Can also
        be a string describing a linestyle defined by matplotlib. See `here
        <https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html>`_ for a list
        of avaliable linestyles. ``True`` defaults to ``"solid"``.
    cline : bool, str, Default = True
        If ``True`` a center line is drawn at *cval* alongside the data points. If ``False`` no line is shown.
        Can also be a string describing a linestyle defined by matplotlib. See `here
        <https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html>`_ for a list
        of avaliable linestyles. ``True`` defaults to ``"solid"``. Only shown if *cval* is given.
    pmline : bool, str, Default = False
        If ``True`` two lines at the upper uncertainty (``cval+pmval``) and lower uncertainty
        (``cval-pmval``) alongside the data points, . If ``False`` no line is shown.
        Can also be a string describing a linestyle defined by matplotlib. See `here
        <https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html>`_ for a list
        of avaliable linestyles. ``True`` defaults to ``"solid"``. Only shown if *cval* and *pmval*
        are given.
    box : bool, Default = True
        If ``True`` a semitransparent box is drawn alongisde the data points basedon the *cval*
        and *pmval* values.
    pad : scalar, Default = 2
        The space between previous data points and the new data points on the x-axis.
    box_pad : scalar, Optional
        The space the cline, pmline and box extends past the data points on the x-axis. If ``None``
        a 1/4 of *pad* is used.
    spacing : scalar, Default = 1
        The space between data points on the x-axis.
    hide_yaxis : bool, Default = True
        Hides the x-axis ticks and label.
    kwargs
        Any keyword argument accepted by matplotlib axes methods. By default kwargs are
        attributed to the data points. Prefix kwargs for the center line
        with ``cline_``, kwargs for the  plus/minus lines with ``pmline_`` and kwargs for the
        box area with ``box_``.
        See `here
        <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.errorbar.htm>`_
        for a list of keyword arguments for the data points.
        See `here
        <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html>`_
        for a list of keyword arguments for the center and plus/minus lines.
        See `here
        <https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Polygon.html>`_
        for a list of keyword arguments for the box area.


    Examples
    --------
    >>> array1 = isopy.tb.make_ms_beams('pd', integrations = 20)
    >>> array2 = isopy.tb.make_ms_beams('pd', integrations = 20)
    >>> isopy.tb.plot_hstack(plt, array1['105pd'])
    >>> isopy.tb.plot_hstack(plt, array2['105pd'], cval=np.mean, pmval=isopy.sd2)
    >>> plt.show()

    .. figure:: plot_hstack1.png
        :alt: output of the example above
        :align: center

    >>> keys = isopy.keylist('pd105', 'ru101', 'cd111')
    >>> isopy.tb.create_subplots(plt, [[key] for key in keys.sorted()])
    >>> array1 = isopy.tb.make_ms_beams('pd105', 'cd111', integrations = 20)
    >>> array2 = isopy.tb.make_ms_beams('pd105', 'ru101', integrations = 20)
    >>> isopy.tb.plot_hstack(plt, array1, color=isopy.tb.Colors()[0])
    >>> isopy.tb.plot_hstack(plt, array2, cval=np.mean, pmval=isopy.sd2, color=isopy.tb.Colors()[1])
    >>> plt.show()

    .. figure:: plot_hstack2.png
        :alt: output of the example above
        :align: center

    >>> colorpairs = isopy.tb.ColorPairs()
    >>> array = isopy.tb.make_ms_beams('pd104', 'pd105', 'pd106')
    >>> mean = np.mean(array)
    >>> sd2 = isopy.sd2(array)
    >>> outliers = isopy.toolbox.isotope.find_outliers(array, mean, sd2)
    >>> isopy.tb.plot_hstack(plt, array, outliers=outliers, cval=mean, pmval=sd2, color=colorpairs.current)
    >>> plt.show()

    .. figure:: plot_hstack3.png
        :alt: output of the example above
        :align: center

    """
    if isinstance(y, tuple) and len(y) == 2: y, yerr = y

    if isinstance(y, core.IsopyArray):
        _plot_stack_array(plot_hstack, axes, 'y', y=y, yerr=yerr, xstart=xstart, cval=cval,
                          pmval=pmval, color=color, marker=marker, line=line, cline=cline,
                          pmline=pmline, box=box, pad=pad, box_pad=box_pad, spacing=spacing,
                          outliers=outliers, hide_xaxis=hide_xaxis, **kwargs)

    else:
        axes = _check_axes(axes)
        y = _check_val('y', y)
        yerr = _check_err('yerr', y, yerr)
        outliers = _check_mask('outliers', y, outliers, default_value=False)

        if xstart is None:
            if spacing > 0:
                xstart = _axes_data_func(axes, 'x', np.nanmax, default_value=np.nan) + pad
            else:
                xstart = _axes_data_func(axes, 'x', np.nanmin, default_value=np.nan) - pad
        if np.isnan(xstart): xstart = 0

        y, x, yerr, cy, cx, pmy, pmx, boxy, boxx, styles = _plot_stack_step1(axes, y, yerr, xstart, cval, pmval, color,
                                                                             marker, line, cline, pmline, box, pad, box_pad, spacing, outliers, **kwargs)

        axes.get_xaxis().set_visible(not hide_xaxis)
        _plot_stack_step2(axes, x, y, None, yerr, cx, cy, pmx, pmy, boxx, boxy, styles, outliers)

def _plot_stack_step2(axes, x, y, xerr, yerr, cx, cy, pmx, pmy, boxx, boxy, styles, isoutlier):
    _plot_errorbar(axes, x, y, xerr, yerr, np.invert(isoutlier), styles['main'])
    _plot_errorbar(axes, x, y, xerr, yerr, isoutlier, styles['outlier'])
    _axes_add_data(axes, x, y, xerr, yerr, isoutlier)

    if cx is not None and cy is not None:
        axes.plot(cx, cy, **styles['cline'])

    if pmx is not None and pmy is not None:
        axes.plot(pmx[0], pmy[0], **styles['pmline'])
        axes.plot(pmx[1], pmy[1], **styles['pmline'])

    if boxx is not None and boxy is not None:
        plot_box(axes, boxx, boxy, **styles['box'])


def _plot_stack_step1(axes, v, verr, wstart, box_cval, box_pmval, color, marker, line, cline, pmline, box_shaded, pad, box_pad, spacing, isoutlier,
                      **kwargs):
    axes = _check_axes(axes)
    include = np.invert(isoutlier)
    if not isinstance(spacing, (float, int)): raise TypeError(f'spacing must be float or an integer')

    w = np.array([wstart + (i * spacing) for i in range(v.size)])
    wstop = w[-1]

    if box_cval:
        if callable(box_cval):
            box_cval = box_cval(v[include])
        try:
            box_cval = np.array(float(box_cval))
        except:
            raise ValueError(f'unable to convert cval {box_cval} to float')


    if box_pmval:
        if callable(box_pmval):
            box_pmval = box_pmval(v[include])

        try:
            box_pmval = np.asarray(float(box_pmval))
        except:
            raise ValueError(f'unable to convert pmval {box_pmval} to float')

    style = {}
    style.update(stack_style)
    style.update(kwargs)

    if isinstance(color, tuple):
        style['color'] = color[0]
        style['outlier_color'] = color[1]
    elif color: style['color'] = color
    else: style.setdefault('color', next(axes._get_lines.prop_cycler).get('color', 'blue'))

    style.setdefault('outlier_color', style['color'])
    style.setdefault('markerfacecolor', style['color'])
    style.setdefault('outlier_markerfacecolor', style['outlier_color'])

    if marker is True: style.setdefault('marker', 'o')
    elif marker is False: style['marker'] = ''
    elif marker: style['marker'] = marker

    if line is True: style.setdefault('linestyle', '-')
    elif line is False: style['linestyle'] = ''
    elif line: style['linestyle'] = line

    if cline is True or cline is None: style.setdefault('cline_linestyle', '-')
    elif cline is False: style['cline_linestyle'] = ''
    elif cline: style['cline_linestyle'] = cline

    if pmline is True or pmline is None: style.setdefault('pmline_linestyle', '--')
    elif pmline is False: style['pmline_linestyle'] = ''
    elif pmline: style['pmline_linestyle'] = pmline

    style.setdefault('cline_color', style['color'])
    style.setdefault('pmline_color', style['color'])
    style.setdefault('box_color', style['color'])
    style.setdefault('zorder', 2)
    style.setdefault('cline_zorder', style['zorder'] - 0.001)
    style.setdefault('box_zorder', style['zorder'] - 0.002)

    cline_style = _extract_style(style, 'cline')
    pmline_style = _extract_style(style, 'pmline')
    box_style = _extract_style(style, 'box')

    temp_style = _extract_style(style, 'outlier')
    outlier_style = style.copy()
    outlier_style.pop('label', None) #Otherwise the label gets shown twice
    outlier_style.update(temp_style)

    styles = dict(main=style, cline=cline_style, pmline = pmline_style, box=box_style, outlier=outlier_style)

    if not isinstance(pad, (float, int)): raise TypeError(f'pad must be float or an integer')
    if box_pad is None: box_pad = pad/4
    elif not isinstance(box_pad, (float, int)): raise TypeError(f'pad must be float or an integer')

    if box_cval and cline is not False:
        cv = np.array([box_cval, box_cval])
        cw = np.array([min((wstart, wstop)) - box_pad, max((wstart, wstop)) + box_pad])
    else:
        cv = None
        cw = None

    if box_pmval and box_cval and pmline is not False:
        pmv = (np.array([box_cval + box_pmval, box_cval + box_pmval]),
               np.array([box_cval - box_pmval, box_cval - box_pmval]))
        pmw = (np.array([max((wstart, wstop)) + box_pad, min((wstart, wstop)) - box_pad]),
               np.array([max((wstart, wstop)) + box_pad, min((wstart, wstop)) - box_pad]))
    else:
        pmv = None
        pmw = None

    if box_pmval and box_cval and box_shaded is not False:
        boxv = np.array([box_cval + box_pmval, box_cval - box_pmval])
        boxw = np.array([max((wstart, wstop)) + box_pad, min((wstart, wstop)) - box_pad])
    else:
        boxv = None
        boxw = None

    return v, w, verr, cv, cw, pmv, pmw, boxv, boxw, styles

def _plot_stack_array(func, axes, vaxis, /, **kwargs):
    if vaxis == 'x':
        waxis = 'y'
    elif vaxis == 'y':
        waxis = 'x'
    else:
        raise ValueError('vaxis incorrect')

    keys = kwargs[vaxis].keys()

    if isinstance(axes, dict):
        named_axes = axes
    else:
        figure =_check_figure('axes', axes)
        axes = figure.get_axes()
        if len(axes) == 0:
            if vaxis == 'y':
                subplots = [[str(key)] for key in keys]
            else:
                subplots = [[str(key) for key in keys]]
            named_axes = create_subplots(figure, subplots)
        else:
            named_axes = {str(ax.get_label()): ax for ax in axes}

    if kwargs[f'{waxis}start'] is None:
        find_wstart = True
    else:
        find_wstart = False

    if find_wstart:
        wstart = np.nan
        for name, ax in named_axes.items():
            if name == '_': continue
            if kwargs['spacing'] > 0:
                wstart = np.nanmax([wstart, _axes_data_func(ax, waxis, np.nanmax, default_value=np.nan)])
            else:
                wstart = np.nanmin([wstart, _axes_data_func(ax, waxis, np.nanmin, default_value=np.nan)])

        if kwargs['spacing'] > 0:
            kwargs[f'{waxis}start'] = wstart + kwargs['pad']
        else:
            kwargs[f'{waxis}start'] = wstart - kwargs['pad']

    for key in keys:
        try:
            ax = named_axes[str(key)]
        except KeyError:
            raise KeyError(f'axes with name "{key}" not found')
        func(ax, **{k: v.get(str(key)) if isinstance(v, core.IsopyArray) else v for k, v in
                    kwargs.items()})

        if waxis == 'x':
            ax.set_ylabel(str(key))
        elif waxis == 'y':
            ax.set_xlabel(str(key))

    maxw = np.nan
    minw = np.nan
    for name, ax in named_axes.items():
        if name == '_': continue
        maxw = np.nanmax([maxw, _axes_data_func(ax, waxis, np.nanmax, default_value=np.nan)])
        minw = np.nanmin([minw, _axes_data_func(ax, waxis, np.nanmin, default_value=np.nan)])

    if not np.isnan(maxw) and not np.isnan(minw):
        for name, ax in named_axes.items():
            if name == '_': continue
            if waxis == 'x':
                ax.set_xlim((minw-kwargs['pad']/2, maxw+kwargs['pad']/2))
            elif waxis == 'y':
                ax.set_ylim((minw - kwargs['pad']/2, maxw + kwargs['pad']/2))


def _plot_errorbar(axes, x, y, xerr = None, yerr = None, mask=None, style={}):
    if mask is not None:
        if yerr is not None: yerr = yerr[mask]
        if xerr is not None: xerr = xerr[mask]
        axes.errorbar(x[mask], y[mask], yerr, xerr, **style)
    else:
        axes.errorbar(x, y, yerr, xerr, **style)


def plot_hcompare(axes, cval = np.nanmean, pmval = isopy.nansd2, sigfig = 2, pmunit='diff', cline=True, pmline=True,
                  color='black', axis_color = None, ignore_outliers=True, combine_ticklabels=5):
    """
    A comparison plot for data sets on a hstack.

    Calculate the center and uncertainty values for all data plotted on a hstack and
    set the *y* axis tick marks at ``[cval-pmval, cval, cval+pmval]`` and format the
    tick labels according to *pmunit*.


    Parameters
    ----------
    axes : axes, figure, plt
        Either a matplotlib axes object or a matplotlib Figure object. Objects with a gca() method
        that return a matplotlib axes object, such as a matplotlib pyplot instance are also accepted.
        If *axes* is a matplotlib Figure object then :func:`plot_hcompare` will be called on every
        named subplot in the figure.
    cval : scalar, Callable, Optional
        Either a scalar representing the center value or a function that will be used to
        calculate the center value based on all the y-axis data points in *axes*.
    pmval : scalar, Callable, Optional
        Either a scalar representing the uncertainty on the center value or a function
        that will be used to calculate the uncertainty based on all the y-axis data points in *axes*.
    sigfig : int, Default = 2
        The number of significant figures to be shown on the y-axis tick labels.
    pmunit : str, Default = 'diff'
        Defines the unit of the tick labels for the *pmline*'s. Avaliable options are:

        * ``"diff"`` - The tick labels are shown as ``f'"Δ {+pmval}"`` and ``f'"Δ {-pmval}"``.
        * ``"abs"`` - The tick labels are shown as ``f'"{cval+pmval}"`` and ``f'"{cval-pmval}"``.
        * ``"percent"`` or ``"%"`` - The tick labels are shown as ``f'"{+pmval/cval*100} %"`` and ``f'"{-pmval/cval*100} %"``.
        * ``"permil"`` or ``"ppt"`` - The tick labels are shown as ``f'"{+pmval/cval*1000} ‰"`` and ``f'"{-pmval/cval*1000} ‰"``.
        * ``"ppm"`` - The tick labels are shown as ``f'"{+pmval/cval*1E6} ppm"`` and ``f'"{-pmval/cval*1E6} ppm"``.
        * ``"ppb"`` - The tick labels are shown as ``f'"{+pmval/cval*1E9} ppb"`` and ``f'"{-pmval/cval*1E9} ppb"``.

        A tuple can be used to set different units for the upper and lower tick labels.
    cline : bool, str, Default = True
        If ``True`` a horizontal line is drawn along *cval*. If ``False`` no line is shown. Can also
        be a string describing a linestyle defined by matplotlib. See `here
        <https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html>`_ for a list
        of avaliable linestyles. ``True`` defaults to ``"solid"``.
    pmline : bool, str, Default = True
        If ``True`` horizontal lines are drawn along ``cval-pmval` and ``cval+pmval``.
         If ``False`` no line is shown. Can also
        be a string describing a linestyle defined by matplotlib. See `here
        <https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html>`_ for a list
        of avaliable linestyles. ``True`` defaults to ``"dashed"``.
    color : str, Optional
        Color *cline* and *pmline* lines.  Accepted strings are named colour in matplotlib or a
        string of a hex triplet begining with "#".
        See `here <https://matplotlib.org/stable/gallery/color/named_colors.html>`_ for a list of
        named colours in matplotlib. If ``None`` the default value is ``black``.
    axis_color : bool, str, Optional
        If ``True`` the elements on the y-axis are given the same colour as *color*.
    ignore_outliers : bool, Default = True
        If ``True`` outliers are not used to calculate *cval* and *pmline*.
    combine_ticklabels : bool, scalar, Default = 5
        If ``True`` the all three tick labels are combined into the center tick label.
        This is useful if there isnt enough space to display them all next to each other.
        If *combinde_ticklabels* is a scalar then the tick labels are only combined if the
        difference between the limits and the plus/minus tick label positions exceed this value.

    Examples
    --------
    >>> array1 = isopy.tb.make_ms_beams('pd', integrations = 20)
    >>> array2 = isopy.tb.make_ms_beams('pd', integrations = 20)
    >>> isopy.tb.plot_hstack(plt, array1['105pd'])
    >>> isopy.tb.plot_hstack(plt, array2['105pd'], cval=np.mean, pmval=isopy.sd2)
    >>> isopy.tb.plot_hcompare(plt)
    >>> plt.show()

    .. figure:: plot_hcompare1.png
        :alt: output of the example above
        :align: center

    >>> array1 = isopy.tb.make_ms_beams('pd', integrations = 20)
    >>> array2 = isopy.tb.make_ms_beams('pd', integrations = 20)
    >>> isopy.tb.plot_hstack(plt, array1['105pd'])
    >>> isopy.tb.plot_hstack(plt, array2['105pd'], cval=np.mean, pmval=isopy.sd2)
    >>> isopy.tb.plot_hcompare(plt, pmval=isopy.se2, sigfig=1)
    >>> plt.show()

    .. figure:: plot_hcompare2.png
        :alt: output of the example above
        :align: center

    >>> pmunits = ['diff', 'abs', '%', 'ppt', 'ppm', 'ppb']
    >>> subplots = isopy.tb.create_subplots(plt, [[unit] for unit in pmunits], figheight=6)
    >>> array1 = isopy.tb.make_ms_beams('pd', integrations = 20)
    >>> array2 = isopy.tb.make_ms_beams('pd', integrations = 20)
    >>> for unit, axes in subplots.items():
    >>>     isopy.tb.plot_hstack(axes, array1['105pd'])
    >>>     isopy.tb.plot_hstack(axes, array2['105pd'], cval=np.mean, pmval=isopy.sd2)
    >>>     isopy.tb.plot_hcompare(axes, pmunit=unit, combine_ticklabels=True)
    >>>     axes.set_ylabel(f'pmunit="{unit}"')
    >>> plt.show()

    .. figure:: plot_hcompare3.png
        :alt: output of the example above
        :align: center

    >>> figure = isopy.tb.update_figure(plt, figheight=6)
    >>> array1 = isopy.tb.make_ms_beams('pd', integrations = 20)
    >>> array2 = isopy.tb.make_ms_beams('pd', integrations = 20)
    >>> isopy.tb.plot_hstack(plt, array1)
    >>> isopy.tb.plot_hstack(plt, array2, cval=np.mean, pmval=isopy.sd2)
    >>> isopy.tb.plot_hcompare(figure)
    >>> plt.show()

    .. figure:: plot_hcompare4.png
        :alt: output of the example above
        :align: center
    """
    if isinstance(axes, mplFigure) or isinstance(axes, dict):
        kwargs = dict(cval=cval, pmval=pmval, sigfig=sigfig, pmunit=pmunit, cline=cline,
                      pmline=pmline,
                      color=color, axis_color=axis_color, ignore_outliers=ignore_outliers,
                      combine_ticklabels=combine_ticklabels)

        if isinstance(axes, mplFigure):
            figure = axes
            named_axes = {ax.get_label(): ax for ax in figure.get_axes()}
        else:
            named_axes = axes

        for name, ax in named_axes.items():
            if name == '_': continue
            plot_hcompare(ax, **{k: v.get(name) if isinstance(v, core.IsopyArray) else v for k, v in
                    kwargs.items()})

    else:
        axes = _check_axes(axes)

        tickmarks, ticklabels, min_lim, max_lim = _compare_step1(axes, 'y', cval, pmval, sigfig, pmunit, ignore_outliers, combine_ticklabels)
        if isinstance(color, tuple): color = color[0]
        if isinstance(axis_color, tuple): axis_color = axis_color[0]
        axes.set_yticks(tickmarks)
        axes.set_yticklabels(ticklabels)
        axes.set_ylim(min_lim, max_lim)
        if cline is not False:
            if cline is True: cline = 'solid'
            axes.axhline(tickmarks[1], linestyle = cline, color=color)
        if pmline is not False:
            if pmline is True: pmline = 'dashed'
            axes.axhline(tickmarks[0], linestyle=pmline, color=color)
            axes.axhline(tickmarks[2], linestyle=pmline, color=color)

        if axis_color is True:
            axes.tick_params(axis='y', colors=color)
        elif axis_color is not None:
            axes.tick_params(axis='y', colors=axis_color)

def plot_vcompare(axes, cval = np.nanmean, pmval = isopy.nansd2, sigfig = 2, pmunit='diff', cline='-', pmline='--',
                  color='black', axis_color = None, ignore_outliers=True, combine_ticklabels=5):
    """
    A comparison plot for data sets on a vstack.

    Calculate the center and uncertainty values for all data plotted on a vstack and
    set the *x* axis tick marks at ``[cval-pmval, cval, cval+pmval]`` and format the
    tick labels according to *pmunit*.

    Parameters
    ----------
    axes : axes, figure, plt
        Either a matplotlib axes object or a matplotlib Figure object. Objects with a gca() method
        that return a matplotlib axes object, such as a matplotlib pyplot instance are also accepted.
        If *axes* is a matplotlib Figure object then :func:`plot_vcompare` will be called on every
        named subplot in the figure.
    cval : scalar, Callable, Optional
        Either a scalar representing the center value or a function that will be used to
        calculate the center value based on all the x-axis data points in *axes*.
    pmval : scalar, Callable, Optional
        Either a scalar representing the uncertainty on the center value or a function
        that will be used to calculate the uncertainty based on all the x-axis data points in *axes*.
    sigfig : int, Default = 2
        The number of significant figures to be shown on the x-axis tick labels.
    pmunit : str, Default = 'diff'
        Defines the unit of the tick labels for the *pmline*'s. Avaliable options are:

        * ``"diff"`` - The tick labels are shown as ``f'"Δ {+pmval}"`` and ``f'"Δ {-pmval}"``.
        * ``"abs"`` - The tick labels are shown as ``f'"{cval+pmval}"`` and ``f'"{cval-pmval}"``.
        * ``"percent"`` or ``"%"`` - The tick labels are shown as ``f'"{+pmval/cval*100} %"`` and ``f'"{-pmval/cval*100} %"``.
        * ``"permil"`` or ``"ppt"`` - The tick labels are shown as ``f'"{+pmval/cval*1000} ‰"`` and ``f'"{-pmval/cval*1000} ‰"``.
        * ``"ppm"`` - The tick labels are shown as ``f'"{+pmval/cval*1E6} ppm"`` and ``f'"{-pmval/cval*1E6} ppm"``.
        * ``"ppb"`` - The tick labels are shown as ``f'"{+pmval/cval*1E9} ppb"`` and ``f'"{-pmval/cval*1E9} ppb"``.

        A tuple can be used to set different units for the upper and lower tick labels.
    cline : bool, str, Default = True
        If ``True`` a vertical line is drawn along *cval*. If ``False`` no line is shown. Can also
        be a string describing a linestyle defined by matplotlib. See `here
        <https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html>`_ for a list
        of avaliable linestyles. ``True`` defaults to ``"solid"``.
    pmline : bool, str, Default = True
        If ``True`` vertical lines are drawn along ``cval-pmval` and ``cval+pmval``.
         If ``False`` no line is shown. Can also
        be a string describing a linestyle defined by matplotlib. See `here
        <https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html>`_ for a list
        of avaliable linestyles. ``True`` defaults to ``"dashed"``.
    color : str, Optional
        Color *cline* and *pmline* lines.  Accepted strings are named colour in matplotlib or a
        string of a hex triplet begining with "#".
        See `here <https://matplotlib.org/stable/gallery/color/named_colors.html>`_ for a list of
        named colours in matplotlib. If ``None`` the default value is ``black``.
    axis_color : bool, str, Optional
        If ``True`` the elements on the x-axis are given the same colour as *color*.
    ignore_outliers : bool, Default = True
        If ``True`` outliers are not used to calculate *cval* and *pmline*.
    combine_ticklabels : bool, scalar, Default = 5
        If ``True`` the all three tick labels are combined into the center tick label.
        This is useful if there isnt enough space to display them all next to each other.
        If *combinde_ticklabels* is a scalar then the tick labels are only combined if the
        difference between the limits and the plus/minus tick label positions exceed this value.

    Examples
    --------
    >>> array1 = isopy.tb.make_ms_beams('pd', integrations = 20)
    >>> array2 = isopy.tb.make_ms_beams('pd', integrations = 20)
    >>> isopy.tb.plot_vstack(plt, array1['105pd'])
    >>> isopy.tb.plot_vstack(plt, array2['105pd'], cval=np.mean, pmval=isopy.sd2)
    >>> isopy.tb.plot_vcompare(plt)
    >>> plt.show()

    .. figure:: plot_vcompare1.png
        :alt: output of the example above
        :align: center


    >>> array1 = isopy.tb.make_ms_beams('pd', integrations = 20)
    >>> array2 = isopy.tb.make_ms_beams('pd', integrations = 20)
    >>> isopy.tb.plot_vstack(plt, array1['105pd'])
    >>> isopy.tb.plot_vstack(plt, array2['105pd'], cval=np.mean, pmval=isopy.sd2)
    >>> isopy.tb.plot_vcompare(plt, pmval=isopy.se2, sigfig=1)
    >>> plt.show()

    .. figure:: plot_vcompare2.png
            :alt: output of the example above
            :align: center

    >>> pmunits = ['diff', 'abs', '%', 'ppt', 'ppm', 'ppb']
    >>> subplots = isopy.tb.create_subplots(plt, [[unit for unit in pmunits]], figwidth=8)
    >>> array1 = isopy.tb.make_ms_beams('pd', integrations = 20)
    >>> array2 = isopy.tb.make_ms_beams('pd', integrations = 20)
    >>> for unit, axes in subplots.items():
    >>>     isopy.tb.plot_vstack(axes, array1['105pd'])
    >>>     isopy.tb.plot_vstack(axes, array2['105pd'], cval=np.mean, pmval=isopy.sd2)
    >>>     isopy.tb.plot_vcompare(axes, pmunit=unit, combine_ticklabels=True)
    >>>     axes.set_xlabel(f'pmunit="{unit}"')
    >>> plt.show()

    .. figure:: plot_vcompare3.png
            :alt: output of the example above
            :align: center

    >>> figure = isopy.tb.update_figure(plt, figwidth=8)
    >>> array1 = isopy.tb.make_ms_beams('pd', integrations = 20)
    >>> array2 = isopy.tb.make_ms_beams('pd', integrations = 20)
    >>> isopy.tb.plot_vstack(plt, array1)
    >>> isopy.tb.plot_vstack(plt, array2, cval=np.mean, pmval=isopy.sd2)
    >>> isopy.tb.plot_vcompare(figure, combine_ticklabels=True)
    >>> plt.show()

    .. figure:: plot_vcompare4.png
            :alt: output of the example above
            :align: center
    """
    if isinstance(axes, mplFigure) or isinstance(axes, dict):
        kwargs = dict(cval=cval, pmval=pmval, sigfig=sigfig, pmunit=pmunit, cline=cline, pmline=pmline,
                      color=color, axis_color=axis_color, ignore_outliers=ignore_outliers,
                      combine_ticklabels=combine_ticklabels)

        if isinstance(axes, mplFigure):
            figure = axes
            named_axes = {ax.get_label(): ax for ax in figure.get_axes()}
        else:
            named_axes = axes

        for name, ax in named_axes.items():
            if name == '': continue
            plot_vcompare(ax, **{k: v.get(name) if isinstance(v, core.IsopyArray) else v for k, v in
                    kwargs.items()})

    else:
        axes = _check_axes(axes)

        tickmarks, ticklabels, min_lim, max_lim = _compare_step1(axes, 'x', cval, pmval, sigfig, pmunit, ignore_outliers, combine_ticklabels)
        if isinstance(color, tuple): color = color[0]
        if isinstance(axis_color, tuple): axis_color = axis_color[0]
        axes.set_xticks(tickmarks)
        axes.set_xticklabels(ticklabels)
        axes.set_xlim(min_lim, max_lim)
        if cline:
            axes.axvline(tickmarks[1], linestyle = cline, color=color)
        if pmline:
            axes.axvline(tickmarks[0], linestyle=pmline, color=color)
            axes.axvline(tickmarks[2], linestyle=pmline, color=color)

        if axis_color is True:
            axes.tick_params(axis='x', colors=color)
        elif axis_color is not None:
            axes.tick_params(axis='x', colors=axis_color)

def _compare_step1(axes, axis, cval, pmval, sigfig, pmunit, ignore_outliers, overlap_treshold):
    if callable(cval):
        cval = _axes_data_func(axes, axis, cval, ignore_outliers=ignore_outliers)
    cval = np.asarray(cval, dtype=float)
    if callable(pmval):
        pmval = _axes_data_func(axes, axis, pmval, ignore_outliers=ignore_outliers)
    pmval = np.asarray(pmval, dtype=float)
    tickmarks = [cval - pmval, cval, cval + pmval]

    if isinstance(pmunit, tuple) and len(pmunit)==2:
        punit, munit = pmunit
    elif '_' in pmunit:
        punit, munit = pmunit.split('_', 1)
    else:
        punit = pmunit
        munit = pmunit

    min1, max1 = _axes_data_lim(axes, axis)
    min2, max2 = _axes_data_lim(axes, axis, ignore_outliers=True)

    min3 = np.nanmin([cval - pmval, min2])
    max3 = np.nanmax([cval + pmval, max2])
    diff = (max3 - min3) * 5

    max_lim = np.nanmax([max3, np.nanmin([cval + diff/2, max1])])
    min_lim = np.nanmin([min3, np.nanmax([max_lim-diff, min1])])

    diff = (max_lim-min_lim)*0.05
    max_lim += diff
    min_lim -= diff

    if overlap_treshold is True:
        overlap = True
    elif overlap_treshold is False:
        overlap = False
    else:
        text_rat = (max_lim-min_lim) / (pmval*2)
        if text_rat > overlap_treshold:
            overlap = True
        else:
            overlap = False
    if overlap:
        ticklabels = ['',f'{_label_unit(cval, -pmval, munit, sigfig)}\n{_format_sigfig(cval, sigfig, pmval)}\n{_label_unit(cval, pmval, punit, sigfig)}', '']
    else:
        ticklabels = [_label_unit(cval, -pmval, munit, sigfig),
                      _format_sigfig(cval, sigfig, pmval),
                      _label_unit(cval, pmval, punit, sigfig)]

    return tickmarks, ticklabels, min_lim, max_lim

def _format_sigfig(value, sigfig, variation = None, pm=False):
    if variation is None: variation = value
    if sigfig is not None:
        precision = sigfig - int(np.log10(np.abs(variation)))
        if precision < 0: precision = 0
    if pm:
        precision = f'+.{precision}'
    else:
        precision = f'.{precision}'

    return f'{{:{precision}f}}'.format(value)

def _label_unit(refval, val, unit, sigfig):
    if unit[-1].isdigit():
        sigfig = int(unit[-1])
        unit = unit[:-1]

    if unit == 'diff':
        return f'\u0394 {_format_sigfig(val, sigfig, pm=True)}'
    if unit == 'abs':
        return _format_sigfig(refval+val, sigfig, val)
    if unit == 'percent' or unit == '%':
        return f'{_format_sigfig(val/refval * 100, sigfig, pm=True)} %'
    if unit == 'permil' or unit == 'ppt':
        return f'{_format_sigfig(val/refval * 1000, sigfig, pm=True)} ‰'
    if unit == 'ppm':
        return f'{_format_sigfig(val/refval * 1E6, sigfig, pm=True)} ppm'
    if unit == 'ppb':
        return f'{_format_sigfig(val/refval * 1E9, sigfig, pm=True)} ppb'

def plot_box(axes, x = None, y = None, color = None, autoscale = True, **style_kwargs):
    """
    Plot a semi-transparent box on *axes*.

    Parameters
    ----------
    axes
        The axes on which the regression will we plotted. Must be a matplotlib axes object or any
        object with a gca() method that return a matplotlib axes object, such as a matplotlib
        pyplot instance.
    x, y
        Any object that can be converted to a numpy array. If given, both *x* and *y* must contain
        exactly two items, the initial and the final position. If ``None`` the value if taken as
        the axis limit of *axes*.
    color
        Color of the box polygon.
        Accepted strings are named colour in matplotlib or a string of a hex triplet begining with "#".
        See `here <https://matplotlib.org/stable/gallery/color/named_colors.html>`_ for a list of
        named colours in matplotlib. If not given the next colour on the internal matplotlib colour
        cycle is used.
    autoscale
        If ``False`` the regression will not be taken into account when figure is rescaled.
        This if not officially supported by matplotlib and thus might not always work.
    style_kwargs
        Any keyword argument accepted a matplotlib polygon object. See `here
        <https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Polygon.html>`_ for a
        list of keyword arguments.
    """
    axes = _check_axes(axes)
    if x is not None:
        x = _check_val('x', x)
        if x.size != 2:
            raise ValueError(f'size of x ({x.size}) must be 2')
    else:
        x = np.array(axes.get_xlim()).flatten()

    if y is not None:
        y = _check_val('y', y)
        if y.size != 2:
            raise ValueError(f'size of y ({y.size}) must be 2')
    else:
        y = np.array(axes.get_ylim()).flatten()

    coordinates = [(x[0], y[0]), (x[1], y[0]), (x[1], y[1]), (x[0], y[1])]

    style = {}
    style.update(box_style)
    style.update(**style_kwargs)

    if isinstance(color, tuple): style['color'] = color[0]
    elif color: style['color'] = color
    else: style.setdefault('color', 'black')

    _plot_polygon(axes, coordinates, autoscale, **style)

def plot_polygon(axes, x, y=None, color = None, autoscale = True, **style_kwargs):
    """
    Plot a semi-transparent polygon on *axes*.

    Parameters
    ----------
    axes
        The axes on which the regression will we plotted. Must be a matplotlib axes object or any
        object with a gca() method that return a matplotlib axes object, such as a matplotlib
        pyplot instance.
    x
        Any object that can be converted to a numpy array. Must either be a 1-dimensional array
        the same size as *y* or a 2-dimensional array with the shape (-1, 2) if *y* is ``None``. If
        *x* is a two dimensional array
    y
        Any object that can be converted to a numpy array. Must be the same size as *x*.
        Can be omitted if *x* is a 2-dimensional array.
    color
        Color of the polygon.
        Accepted strings are named colour in matplotlib or a string of a hex triplet begining with "#".
        See `here <https://matplotlib.org/stable/gallery/color/named_colors.html>`_ for a list of
        named colours in matplotlib. If not given the next colour on the internal matplotlib colour
        cycle is used.
    autoscale
        If ``False`` the regression will not be taken into account when figure is rescaled.
        This if not officially supported by matplotlib and thus might not always work.
    style_kwargs
        Any keyword argument accepted a matplotlib polygon object. See `here
        <https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Polygon.html>`_ for a
        list of keyword arguments.

    """
    axes = _check_axes(axes)

    if y is not None:
        x = _check_val('x', x)
        y = _check_val('y', y)
        if x.size != y.size:
            raise ValueError(f'size of x ({x.size}) and y ({y.size}) are not the same')
        coordinates = np.append(x, y).reshape(2, -1).transpose()
    else:
        x = _check_val('x', x, flatten=False)
        if x.ndim != 2 or x.shape[-1] != 2:
            raise ValueError(f'shape of x ({x.shape}) invalid. Must be (n,2)')
        coordinates = x

    style = {}
    style.update(polygon_style)
    style.update(**style_kwargs)

    if isinstance(color, tuple): style['color'] = color[0]
    elif color: style['color'] = color
    else: style.setdefault('color', 'black')

    _plot_polygon(axes, coordinates, autoscale, **style)

def _plot_polygon(axes, coordinates, autoscale = True, **style):
    upl_func = None
    if not autoscale and hasattr(axes, '_update_patch_limits'):
        upl_func = axes._update_patch_limits
        axes._update_patch_limits = lambda *args, **kwargs: None

    try:
        axes.add_patch(mplPolygon(coordinates, **style))
    finally:
        if upl_func:
            axes._update_patch_limits = upl_func

def _extract_style(style, prefix):
    new_style = {}
    keys = list(style.keys())
    for thing in keys:
        things = thing.split('_', 1)
        if things[0] == prefix and len(things) == 2:
            new_style[things[1]] = style.pop(thing)
    return new_style

def _check_axes(axes):
    if not isinstance(axes, mpl.axes.Axes):
        try:
            axes = axes.gca()
        except:
            pass
        if not isinstance(axes, mpl.axes.Axes):
            raise TypeError(f'"axes" must be a matplotlib axes object or matplotlib.pyplot instance')
    return axes

def _check_figure(var_name, figure):
    if not isinstance(figure, mplFigure):
        try:
            figure = figure.gcf()
        except:
            pass
        if not isinstance(figure, mplFigure):
            raise TypeError(f'"{var_name}" must be a matplotlib figure object or matplotlib.pyplot instance')
    return figure

def _check_val(name, val, flatten = True):
    try:
        val = np.asarray(val)
    except Exception as error:
        raise TypeError(f'invalid {name}: {error}') from error

    if flatten: val = val.flatten()

    return val

def _check_err(name, val, err, flatten=True):
    if err is not None:
        try:
            err = np.asarray(err)
        except Exception as error:
            raise TypeError(f'invalid {name}: {error}') from error

        if err.ndim == 0:
            err = np.zeros(val.shape) + err
        else:
            if flatten:
                err = err.flatten()
                if err.size != val.size:
                    raise ValueError(f'size of {name} ({err.size}) does not match size of value ({val.size})')
            else:
                if err.shape != val.shape:
                    raise ValueError(f'shape of {name} ({err.shape}) does not match shape of value ({val.shape})')
    return err

def _check_mask(name, val, mask, flatten=True, default_value=None):
    if mask is None and default_value is not None:
        mask = default_value

    if mask is not None:
        try:
            mask = np.asarray(mask, dtype=bool)
        except Exception as error:
            raise TypeError(f'invalid {name}: {error}') from error

        if mask.ndim == 0:
            mask = np.zeros(val.shape, dtype=bool) + mask
        else:
            if flatten:
                mask = mask.flatten()
                if mask.size != val.size:
                    raise ValueError(f'size of {name} ({mask.size}) does not match size of value ({val.size})')
            else:
                if mask.shape != val.shape:
                    raise ValueError(f'shape of {name} ({mask.shape}) does not match shape of value ({val.shape})')
    return mask

def _axes_add_data(axes, x, y, xerr=None, yerr=None, isoutlier=None):
    if not hasattr(axes, '_ip__data'):
        axes._ip__data = dict(x=[], y=[], xerr=[], yerr=[], isoutlier=[], isnotoutlier=[])

    axes._ip__data['x'].append(x)
    axes._ip__data['xerr'].append(xerr)
    axes._ip__data['y'].append(y)
    axes._ip__data['yerr'].append(yerr)
    if isoutlier is None:
        isoutlier = _check_mask('isoutlier', x, None, default_value=False)
    axes._ip__data['isoutlier'].append(isoutlier)
    axes._ip__data['isnotoutlier'].append(np.invert(isoutlier))

def _axes_data_func(axes, var, func, ignore_outliers=False, default_value = None, func_dict={}):
    multiplier = 1
    if isinstance(func, str):
        if func[-1].isdigit():
            multiplier=int(func[-1])
            func = func[:-1]
        try:
            func = func_dict[func]
        except KeyError:
            raise ValueError(f'"{func}" is not a recognised function')

    if not hasattr(axes, '_ip__data'):
        if default_value is not None:
            return default_value
        else:
            raise ValueError('axes has no data stored')

    if ignore_outliers:
        data = np.concatenate(
            [v[axes._ip__data['isnotoutlier'][i]] for i, v in enumerate(axes._ip__data[var]) if v is not None])
    else:
        data = np.concatenate([v for v in axes._ip__data[var] if v is not None])

    return func(data) * multiplier

def _axes_data_lim(axes, axis, ignore_outliers=False):
    if not hasattr(axes, '_ip__data'):
        raise ValueError('axes has no data stored')
    pvals = []
    mvals = []
    if ignore_outliers:
        values = [v[axes._ip__data['isnotoutlier'][i]] if v is not None else 0 for i, v in enumerate(axes._ip__data[axis])]
        errors = [v[axes._ip__data['isnotoutlier'][i]] if v is not None else 0 for i, v in enumerate(axes._ip__data[f'{axis}err'])]
        for i, err in enumerate(errors):
            if err is not None:
                pvals.append(values[i] + err)
                mvals.append(values[i] - err)
            else:
                pvals.append(values[i])
                mvals.append(values[i])
    else:
        values = [v if v is not None else 0 for v in axes._ip__data[axis]]
        errors = [v if v is not None else 0 for v in axes._ip__data[f'{axis}err']]
        for i, err in enumerate(errors):
            if err is not None:
                pvals.append(values[i] + err)
                mvals.append(values[i] - err)
            else:
                pvals.append(values[i])
                mvals.append(values[i])
    try:
        min = np.nanmin(np.concatenate(mvals))
    except IOError:
        min = np.nan
    try:
        max = np.nanmax(np.concatenate(pvals))
    except ValueError:
        max = np.nan
    return (min, max)
