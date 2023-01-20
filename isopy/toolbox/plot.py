import isopy as isopy
import isopy.core as core
import isopy.toolbox as toolbox
import scipy.stats as stats
import matplotlib as mpl
from matplotlib.patches import Polygon as mplPolygon
from matplotlib.figure import Figure as mplFigure
from matplotlib.container import ErrorbarContainer
from matplotlib.axes import Axes as mplAxes
import numpy as np
import functools
import warnings
from typing import Optional, Union, Literal
import cycler
import colorsys

__all__ = ['plot_scatter', 'plot_regression', 'plot_vstack', 'plot_hstack',
           'plot_spider', 'plot_hcompare', 'plot_vcompare', 'plot_contours',
           'create_subplots', 'update_figure', 'update_axes', 'create_legend',
           'plot_text', 'plot_box', 'plot_polygon',
           'Markers', 'Colors']

scatter_style = {'markeredgecolor': 'black', 'ecolor': 'black', 'capsize': 3, 'elinewidth': 1, 'zorder': 2}
regression_style = {'linestyle': '-', 'color': 'black', 'edgeline_linestyle': '--', 'fill_alpha': 0.3, 'zorder': 1, 'marker': '', 'edgeline_marker': ''}
equation_line_style = {'linestyle': '-', 'color': 'black', 'zorder': 1}
stack_style = {'markeredgecolor': 'black', 'ecolor': 'black', 'capsize': 3, 'elinewidth': 1, 'cline_linestyle': '-',
               'pmline_linestyle': '--', 'box_alpha': 0.3, 'zorder': 2, 'cline_marker': ''}
spider_style = {'markeredgecolor': 'black', 'ecolor': 'black', 'capsize': 3, 'elinewidth': 1, 'zorder': 2}
box_style = dict(color='black', zorder = 0.9, alpha=0.3)
polygon_style = dict(color='black', zorder = 0.9, alpha=0.3)
grid_style = dict(extend='both')


class RotatingList(list):
    """
    Rotates through a list of *items*

    Is a subclass of ``list``. Only new methods/attributes, or methods/attributes that differ from
    the original implementation, are shown below.
    """
    _original = [None]
    def __init__(self, style = None, key = None):
        if style is None:
            prop_cycler = mpl.rcParams['axes.prop_cycle'].by_key()
            items = prop_cycler.get(key, self._original)
        else:
            style_l = mpl.style.library.get(style, {})
            prop_cycler = style_l.get('axes.prop_cycle', None)
            if prop_cycler is None:
                raise ValueError(f'Style "{style}" does not have a prop_cycle defined')

            items = prop_cycler.by_key().get(key, self._original)

        self._original = items
        super(RotatingList, self).__init__()
        self.reset()
        
    def __getitem__(self, item):
        if type(item) is int:
            item = item % len(self)

        return super(RotatingList, self).__getitem__(item)

    def __class_getitem__(cls, item):
        if type(item) is int:
            item = item % len(cls._original)
        return cls._original[item]

    def reset(self):
        """Reset to the initial condition"""
        self.clear()
        self.extend(self._original)

    def next(self):
        """Rotate the list to the right and return the new first item."""
        self.append(self.pop(0))
        return self.current

    def current_next(self):
        """Return the first item in the list then rotate the list to the right."""
        self.next()
        return self[-1]

    def previous(self):
        """Rotate the list to the left and return the new first item."""
        self.insert(0, self.pop(-1))
        return self.current

    def current_previous(self):
        """Return the first item in the list then rotate the list to the left."""
        self.previous()
        return self[1]

    @property
    def current(self):
        """The first item in the list"""
        return self[0]


class Markers(RotatingList):
    """
    Rotate through the markers of the current matplotlib style.

    Optionally the name of a different style can be passed when creating the object to return the colours of that style.

    If a style does not have any markers defined a default list of markers is used. These are circle, square,
    triangle, diamond, plus, cross, pentagon.

    Is a subclass of ``list``. Only new methods/attributes, or methods/attributes that differ from
    the original implementation, are shown below.

    Examples
    --------
    >>> markers = isopy.tb.Colors()
    >>> for i in range(len(markers)):
    >>>     isopy.tb.plot_scatter(plt, 1, i, marker=markers.current, markersize=20)
    >>>     markers.next()
    >>> plt.show()

    .. figure:: plots/Markers.png
            :alt: output of the example above
            :align: center
    """
    _original = ['o', 's', '^', 'D', 'P', 'X', 'p']

    def __init__(self, style=None):
        super(Markers, self).__init__(style, 'marker')


class Colors(RotatingList):
    """
    Rotate through the colors of the current matplotlib style.

    Optionally the name of a different style can be passed when creating the object to return the colours of that style.

    Is a subclass of ``list``. Only new methods/attributes, or methods/attributes that differ from
    the original implementation, are shown below.

    Examples
    --------
    >>> colors = isopy.tb.Colors() # Colours can vary depending on the default style
    >>> for i in range(len(colors)):
    >>>     isopy.tb.plot_scatter(plt, 1, i, color=colors.current, markersize=20)
    >>>     colors.next()
    >>> plt.show()

    .. figure:: plots/Colors.png
            :alt: output of the example above
            :align: center
    """
    _original = ['#377eb8', '#e41a1c', '#4daf4a', '#ff7f00', '#984ea3', '#a65628', '#ffff33', '#f781bf', '#999999']

    def __init__(self, style=None):
        super(Colors, self).__init__(style, 'color')

    def lighten(self, color=None, lightness = 0.85):
        """
        Change the lightness of *color* to *lightness*.

        If the inital colour is very light then the default *lightness* might actually make the colour darker.

        """
        if color is None:
            color = self.current
        return modify_color(color, lightness=lightness)


def modify_color(color, *, scale=None, lightness=None):
    color = mpl.colors.ColorConverter.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(*color)

    if lightness is not None:
        l = lightness
    if scale is not None:
        l = min(1, l * scale)

    return colorsys.hls_to_rgb(h, l, s)


def update_figure(figure, *, size = None, width=None, height=None, dpi=None, facecolor=None, edgecolor=None,
                  frameon=None, tight_layout=None, constrained_layout=None):
    """
    Update the figure options.

    See the matplotlib documentation
    `here <https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.html>`_
    for more details on these options.

    All isopy plotting functions will automatically send any *kwargs* prefixed with ``figure_`` to
    this function.

    Parameters
    ----------
    figure : figure, plt
        A matplotlib Figure object or any object
        with a ``.gcf()`` method that returns a matplotlib Figure object, Such as a matplotlib
        pyplot instance.
    size : scalar, (scalar, scalar)
        Either a single value representing both deimensions of a tuple consisting of width and height.
    width : scalar
        Width of *figure* in inches
    height : scalar
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

    Examples
    --------
    >>> isopy.tb.update_figure(plt, size=(10,2), facecolor='orange')
    >>> isopy.tb.plot_scatter(plt, np.arange(10), np.arange(10))
    >>> plt.show()

    .. figure:: plots/update_figure1.png
            :alt: output of the example above
            :align: center

    >>> isopy.tb.plot_scatter(plt, np.arange(10), np.arange(10),
                     figure_size=(10, 2), figure_facecolor='orange')
    >>> plt.savefig('update_figure2.png')

    .. figure:: plots/update_figure2.png
            :alt: output of the example above
            :align: center


    """
    figure = _check_figure('figure', figure)
    if size is not None:
        if type(size) is tuple:
            width, height = size
        else:
            width, height = size, size
    if width is not None: figure.set_figwidth(width)
    if height is not None: figure.set_figheight(height)
    if dpi is not None: figure.set_dpi(dpi)
    if facecolor is not None: figure.set_facecolor(facecolor)
    if edgecolor is not None: figure.set_edgecolor(edgecolor)
    if frameon is not None: figure.set_frameon(frameon)
    if tight_layout is not None: figure.set_tight_layout(tight_layout)
    if constrained_layout is not None: figure.set_constrained_layout(constrained_layout)
    return figure

def update_axes(axes, *, legend=None, **kwargs):
    """
    Update axes attributes.

    All isopy plotting functions will automatically send any *kwargs* prefixed with ``axes_`` to
    this function.

    Parameters
    ----------
    axes : axes, dict[str, axes]
        A matplotlib axes or a a dictionary of matplotlib axes. In the latter case the options are
        applied to each axes in the dictionary.
    legend : bool
        If ``True`` the legend will be shown on the axes. If ``False`` no legend is shown.
    kwargs
        Any attribute that can be set using ``axes.set_<key>(value)``.
        If the value is a tuple ``axes.set_<key>(*value)`` is used. If
        the value is a dictionary ``axes.set_<key>(**value)``.

    Examples
    --------
    >>> isopy.tb.update_axes(plt, xlabel='This it the x-axis', ylabel='This is the y-axis')
    >>> isopy.tb.plot_scatter(plt, np.arange(10), np.arange(10))
    >>> plt.show()

    .. figure:: plots/update_axes1.png
            :alt: output of the example above
            :align: center

    >>> isopy.tb.plot_scatter(plt, np.arange(10), np.arange(10),
                            axes_xlabel='This it the x-axis',
                            axes_ylabel='This is the y-axis')
    >>> plt.show()

    .. figure:: plots/update_axes2.png
            :alt: output of the example above
            :align: center
    """
    if type(axes) is dict:
        axes = (_check_axes(ax) for ax in axes.values())
    else:
        axes = (_check_axes(axes),)

    for ax in axes:
        for name, value in kwargs.items():
            set_func = getattr(ax, f'set_{name}')
            if type(value) is tuple:
                set_func(*value)
            elif type(value) is dict:
                set_func(**value)
            else:
                set_func(value)

        if legend is True:
            ax.legend()
        elif legend is False:
            try:
                ax._remove_legend(None)
            except:
                pass


def _update_figure_and_axes(func):
    """
    Allows figure kwargs to be passed to functions
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        fig_kwargs = core.extract_kwargs(kwargs, 'figure')
        axes_kwargs = core.extract_kwargs(kwargs, 'axes')

        if fig_kwargs and len(args) > 0:
            update_figure(args[0], **fig_kwargs)

        ret = func(*args, **kwargs)

        if axes_kwargs and len(args) > 0:
            update_axes(args[0], **axes_kwargs)

        return ret
    return wrapper

@core.append_preset_docstring
@core.add_preset('v', grid = (-1, 1))
@core.add_preset('h', grid = (1, -1))
@_update_figure_and_axes
def create_subplots(figure, subplots, grid = None, *,
                    row_height = None, column_width = None,
                    legend = None, legend_ratio = None,
                    constrained_layout=True, height_ratios = None, width_ratios=None,  **kwargs):
    """
    Create subplots for a figure.

    Use None instead of a name to create an empty space.

    Parameters
    ----------
    figure : figure, plt
        A matplotlib Figure object or any object
        with a ``.gcf()`` method that returns a matplotlib Figure object, Such as a matplotlib
        pyplot instance.
    subplots : list, int
        A nested list of subplot names. A visual layout of how you want your subplots
        to be arranged labeled as strings. *subplots* can be a 1-dimensional list if grid is specified. ``None`` will
        create an empty space. An integer of the number of subplots can be given if grid is specified. Each axes will be
        names ``ax<i>`` e.g. ``[["ax0", "ax1"], ["ax2", "ax3"]]`` . A tuple can be passed for twinned axes as
        (name, [name_twinx], [name_twiny]).
    grid : tuple[int, int]
        If supplied then subplots will be arranged into grid with this shape. One dimension can
        be -1 and it wil be calculated based on the size of *subplots* and the other given
        dimension. Required that *subplots* is a one dimensional sequence.
    row_height
        The height of each row in final figure. If *height_ratios* is given then the height of each row is
        *row_height* multiplied by the height ratio.
    row_height
        The width of each column in final figure. If *width_ratios* is given then the width of each column is
        *column_width* multiplied by the width ratio.
    legend : {"n", "e", "s", "w"}, None
        Creates an additional row/column for the legend at the position indicated. This
        subplot is given the name "legend" in the output.
    legend_ratio
        The relative size of the legend. If *height_ratios* or *width_ratios* has not been given then the
        value for other each row/column is assumed to be 1.
    constrained_layout : bool
        If ``True`` use ``constrained_layout`` when drawing the figure. Advised to avoid
        overlapping axis labels.
    height_ratios
        A list of the relative height for each row.
    width_ratios
        A list of the relative width of each column.
    kwargs
        Prefix kwargs to be passed along witht he creation of each subplot with ``subplot_``.
        Kwargs to be passed to the GridSpec should be prefixed ``gridspec_``. See matplotlib docs
        for
        `add_subplot <https://matplotlib.org/stable/api/figure_api.html?highlight=subplot_mosaic#matplotlib.figure.Figure.add_subplot>`_
         and `Gridspec <https://matplotlib.org/stable/api/_as_gen/matplotlib.gridspec.GridSpec.html>`_
        for a list of avaliable kwargs.

    Returns
    -------
    subplots : dict
        A dictionary containing the newly created subplots.

    Examples
    --------
    >>> subplots = [['one', 'two', 'three', 'four']]
    >>> axes = isopy.tb.create_subplots(plt, subplots)
    >>> for name, ax in axes.items(): ax.set_title(name)
    >>> plt.show()

    .. figure:: plots/create_subplots1.png
            :alt: output of the example above
            :align: center


    >>> subplots = ['one', 'two', 'three', 'four', 'five', 'six']
    >>> axes = isopy.tb.create_subplots(plt, subplots, (-1, 2))
    >>> for name, ax in axes.items(): ax.set_title(name)
    >>> plt.show()

    .. figure:: plots/create_subplots2.png
        :alt: output of the example above
        :align: center


    >>> subplots = [['one', 'two'], ['three', 'two'], ['four', 'four']]
    >>> axes = isopy.tb.create_subplots(plt, subplots)
    >>> for name, ax in axes.items(): ax.set_title(name)
    >>> plt.show()

    .. figure:: plots/create_subplots3.png
            :alt: output of the example above
            :align: center
    """

    empty_sentinel = None
    figure = _check_figure('figure', figure)
    update_figure(figure, constrained_layout=constrained_layout)
    subplot_kw = core.extract_kwargs(kwargs, 'subplot')
    gridspec_kw = core.extract_kwargs(kwargs, 'gridspec')
    if height_ratios:
        gridspec_kw['height_ratios'] = height_ratios
    if width_ratios:
        gridspec_kw['width_ratios'] = width_ratios
    if type(subplots) is int:
        subplots = [f'ax{i}' for i in range(subplots)]

    if grid is not None:
        nrows = grid[0]
        ncols = grid[1]
        if nrows < 1 and ncols < 1:
            raise ValueError('Invalid grid values')
        if nrows == -1:
            nrows = len(subplots) // ncols
            if len(subplots) % ncols > 0: nrows += 1
        elif ncols == -1:
            ncols = len(subplots) // nrows
            if len(subplots) % nrows > 0: ncols += 1

        grid = []
        for r in range(nrows):
            grid_row = []
            for c in range(ncols):
                try:
                    axname = subplots[r*ncols + c]
                except:
                    axname = empty_sentinel


                grid_row.append(axname)

            grid.append(grid_row)
        subplots = grid

    nrows = len(subplots)
    ncols = len(subplots[0])

    axtwinx = {}
    axtwiny = {}
    for r in range(nrows):
        for c in range(ncols):
            axname = subplots[r][c]
            if axname != empty_sentinel:
                twinx, twiny = None, None
                if type(axname) is tuple:
                    if len(axname) == 1:
                        axname = axname[0]
                    elif len(axname) == 2:
                        axname, twinx = axname
                    elif len(axname) == 3:
                        axname, twinx, twiny = axname
                    else:
                        raise ValueError('Invalid tuple')

                axname = str(axname)
                if twinx is not empty_sentinel:
                    axtwinx[axname] = str(twinx)

                if twiny is not empty_sentinel:
                    axtwiny[axname] = str(twiny)

                subplots[r][c] = axname

    if legend is not None:
        if legend == 'n':
            subplots.insert(0, ['legend' for i in range(ncols)])
            if legend_ratio is not None:
                ratios = gridspec_kw.pop('height_ratios', [1 for i in range(nrows)])
                gridspec_kw['height_ratios'] = [legend_ratio] + ratios
            nrows += 1
        elif legend == 's':
            subplots.append(['legend' for i in range(ncols)])
            if legend_ratio is not None:
                ratios = gridspec_kw.pop('height_ratios', [1 for i in range(nrows)])
                gridspec_kw['height_ratios'] = ratios + [legend_ratio]
            nrows += 1
        elif legend == 'e':
            for row in subplots:
                row.append('legend')
            if legend_ratio is not None:
                ratios = gridspec_kw.pop('width_ratios', [1 for i in range(ncols)])
                gridspec_kw['width_ratios'] = ratios + [legend_ratio]
            ncols += 1
        elif legend == 'w':
            for row in subplots:
                row.insert(0, 'legend')
            if legend_ratio is not None:
                ratios = gridspec_kw.pop('width_ratios', [1 for i in range(ncols)])
                gridspec_kw['width_ratios'] = [legend_ratio] + ratios
            ncols += 1

    if row_height is not None:
        ratios = gridspec_kw.get('height_ratios', [1 for i in range(nrows)])
        height = sum([row_height * r for r in ratios])
        update_figure(figure, height=height)

    if column_width is not None:
        ratios = gridspec_kw.get('column_widths', [1 for i in range(ncols)])
        width = sum([column_width * r for r in ratios])
        update_figure(figure, width=width)

    axes = figure.subplot_mosaic(subplots, empty_sentinel=empty_sentinel,
                                 gridspec_kw=gridspec_kw, subplot_kw=subplot_kw)

    out = SubplotDict()
    for label, ax in axes.items():
        ax.set_label(label) # Is this already done?
        out[label] = ax
        if label in axtwinx:
            out[axtwinx[label]] = ax.twinx()
        if label in axtwiny:
            out[axtwiny[label]] = ax.twiny()
    return out

class SubplotDict(dict):
    def __getattr__(self, item):
        return self.__getitem__(item)

@_update_figure_and_axes
def create_legend(axes, *include_axes, labels = None, hide_axis=None, errorbars = True, newlines=None, **kwargs):
    """
    Create a legend encompassing multiple axes.

    Items in *axes*, if there are any, will be included in the legend.

    If mutilple axes contains items with the same label then only the item from the last axes is included
    in the legend. Empty labels or labels beggining with ``'_'`` are not included in the legend.

    Parameters
    ----------
    axes
        The axes on which the legend will be drawn. any items in this axes will be included
        in the legend.
    include_axes
        Items from axes given here will be included in the legend. Accepts a dict containing axes.
    labels
        If given then only items with this label will be included in the legend.
    hide_axis
        If ``True`` then the x and y axis of axes will be hidden. If ``None`` then the axis is hidden
        only if *include_axes* are given and there are not items *axes* with a label.
    errorbars
        If ``False`` then the errobars wont appear on the markers in the legend.
    newlines
        A dictionary of new lines to be created and included in the legend. The key corresponds to the label
        and the value is another dict of the kwargs for a
        `line <https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html>`_.
    kwargs
        Any additional kwargs to be passed when creating the legend. See
        `legend <https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure.legend>`_
        for a list of avaliable kwargs.

    Examples
    --------
    >>> data = isopy.random(100, keys='ru pd cd'.split())
    >>> axes = isopy.tb.create_subplots(plt, [['left', 'right']], figure_width=8)
    >>> isopy.tb.plot_scatter(axes['left'], data['pd'], data['ru'], label='ru/pd', color='red')
    >>> isopy.tb.plot_scatter(axes['right'], data['pd'], data['cd'], label='cd/pd', color='blue')
    >>> isopy.tb.create_legend(axes['right'], axes['left'])
    >>> plt.show()

    .. figure:: plots/create_legend1.png
            :alt: output of the example above
            :align: center

    >>> data = isopy.random(100, keys='ru pd cd'.split())
    >>> axes = isopy.tb.create_subplots(plt, [['left', 'right', 'legend']],
                                        figure_width=9, gridspec_width_ratios=[4, 4, 1])
    >>> isopy.tb.plot_scatter(axes['left'], data['pd'], data['ru'], label='ru/pd', color='red')
    >>> isopy.tb.plot_scatter(axes['right'], data['pd'], data['cd'], label='cd/pd', color='blue')
    >>> isopy.tb.create_legend(axes['legend'], axes, hide_axis=True)
    >>> plt.show()

    .. figure:: plots/create_legend2.png
            :alt: output of the example above
            :align: center
    """
    
    axes = _check_axes(axes)
    ha, la = axes.get_legend_handles_labels()
    legend = dict(zip(la, ha))

    if hide_axis is None:
        if len(include_axes) > 0 and len(legend) == 0:
            hide_axis = True
        else:
            hide_axis = False

    if isinstance(labels, str): labels = [labels]
    incaxes = [axes]

    for item in include_axes:
        if isinstance(item, dict): incaxes += list(item.values())
        else: incaxes.append(item)

    for ax in incaxes:
        ha, la = ax.get_legend_handles_labels()
        legend.update(dict(zip(la, ha)))

    if newlines is not None:
        for line_name, line_value in newlines.items():
            legend[line_name] = mpl.lines.Line2D([np.nan], [np.nan], **line_value)

    if labels:
        legend = {la: ha for la, ha in legend.items() if la in labels}

    if errorbars is False:
        legend = {la: ha[0] if type(ha) is ErrorbarContainer else ha for la, ha in legend.items()}

    axes.legend(list(legend.values()), list(legend.keys()), **kwargs)
    if hide_axis:
        axes.axis(False)

@_update_figure_and_axes
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
    >>> x = isopy.random(20, seed=46)
    >>> y = x * 3 + isopy.random(20, seed=47)
    >>> isopy.tb.plot_scatter(plt, x, y)
    >>> plt.show()

    .. figure:: plots/plot_scatter1.png
            :alt: output of the example above
            :align: center

    >>> x = isopy.random(20, seed=46)
    >>> y = x * 3 + isopy.random(20, seed=47)
    >>> xerr = 0.2
    >>> yerr = isopy.random(20, seed=48)
    >>> isopy.tb.plot_scatter(plt, x, y, xerr, yerr, regression='york1', color='red', marker='s')
    >>> plt.show()

    .. figure:: plots/plot_scatter2.png
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

    prop_cycler = None

    if color: style['color'] = color
    else:
        if prop_cycler is None: prop_cycler = next(axes._get_lines.prop_cycler)
        style.setdefault('color', prop_cycler.get('color', 'blue'))

    if marker is True:
        if prop_cycler is None: prop_cycler = next(axes._get_lines.prop_cycler)
        style.setdefault('marker', prop_cycler.get('marker', 'o'))
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
        if regression == 'york' or regression == 'york1':
            regression_result = isopy.tb.yorkregress(x, y, xerr, yerr)
        elif regression == 'york2':
            regression_result = isopy.tb.yorkregress2(x, y, xerr, yerr)
        elif regression == 'york95':
            regression_result = isopy.tb.yorkregress95(x, y, xerr, yerr)
        elif regression == 'york99':
            regression_result = isopy.tb.yorkregress99(x, y, xerr, yerr)
        elif regression == 'linear':
            regression_result = isopy.tb.linregress(x, y)
        else:
            raise ValueError(f'"{regression}" not a valid regression')
        plot_regression(axes, regression_result, color=style.get('color', None), label_equation=('label' in style))

@_update_figure_and_axes
def plot_regression(axes, regression_result, color=None, line=True, xlim = None, autoscale = False,
                    fill = True, edgeline = True, label_equation = True,  **kwargs):
    """
    Plot the result of a regression on matplotlib *axes*.

    If *regression_result* has a ``y_se(x)`` method the error envelope of the regression
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
        value. Can also be either a single scalar representing the slope or a tuple of two scalars,
        representing the slope and the intercept.
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
    label_equation : bool
        If ``True`` the equation of the regression line will be added to the label of the
        regression line.
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
    >>> x = isopy.random(20, seed=46)
    >>> y = x * 3 + isopy.random(20, seed=47)
    >>> regression = isopy.tb.regression_linear(x, y)
    >>> isopy.tb.plot_scatter(plt, x, y)
    >>> isopy.tb.plot_regression(plt, regression)
    >>> plt.show()

    .. figure:: plots/plot_regression1.png
            :alt: output of the example above
            :align: center


    >>> x = isopy.random(20, seed=46)
    >>> y = x * 3 + isopy.random(20, seed=47)
    >>> regression = lambda x: 2 * x + x ** 2 #Any callable that takes x and return y is a valid
    >>> isopy.tb.plot_scatter(plt, x, y, color='red')
    >>> isopy.tb.plot_regression(plt, regression, color='red', xlim=(-1, 1))
    >>> plt.show()

    .. figure:: plots/plot_regression2.png
            :alt: output of the example above
            :align: center

    >>> x = isopy.random(20, seed=46)
    >>> y = x * 3 + isopy.random(20, seed=47)
    >>> xerr = 0.2
    >>> yerr = isopy.random(20, seed=48)
    >>> regression = isopy.tb.regression_york1(x, y, xerr, yerr)
    >>> isopy.tb.plot_scatter(plt, x, y, xerr, yerr)
    >>> isopy.tb.plot_regression(plt, regression)
    >>> plt.show()

    .. figure:: plots/plot_regression3.png
            :alt: output of the example above
            :align: center

    >>> x = isopy.random(20, seed=46)
    >>> y = x * 3 + isopy.random(20, seed=47)
    >>> xerr = 0.2
    >>> yerr = isopy.random(20, seed=48)
    >>> isopy.tb.plot_scatter(plt, x, y, xerr, yerr, color='red')
    >>> isopy.tb.plot_regression(plt, regression, color='red', line='dashed', edgeline=False)
    >>> plt.show()

    .. figure:: plots/plot_regression4.png
            :alt: output of the example above
            :align: center
    """
    axes = _check_axes(axes)

    if isinstance(regression_result, toolbox.regress.LinregressResult):
        y_eq = regression_result.y
        y_se_eq = regression_result.y_se
    elif callable(regression_result):
        y_eq = regression_result
        y_se_eq = None
    elif hasattr(regression_result, 'slope') and hasattr(regression_result, 'intercept'):
        y_eq = lambda x: x * regression_result.slope + regression_result.intercept
        y_se_eq = None
    elif isinstance(regression_result, tuple) and len(regression_result) == 2:
        try:
            slope = np.float64(regression_result[0])
            intercept = np.float64(regression_result[1])
        except:
            raise ValueError('regression_value cannot be converted to a floats')
        else:
            y_eq = lambda x: x * slope + intercept
            y_se_eq = None
    else:
        try:
            slope = np.float64(regression_result)
        except:
            raise ValueError('regression_value cannot be converted to a float')
        else:
            y_eq = lambda x: x * slope
            y_se_eq = None

    style = {}
    style.update(regression_style)
    style.update(kwargs)

    prop_cycler = None

    if color:
        style['color'] = color
    else:
        if prop_cycler is None: prop_cycler = next(axes._get_lines.prop_cycler)
        style.setdefault('color', prop_cycler.get('color', 'blue'))

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

    fill_style = core.extract_kwargs(style, 'fill')
    edge_style = core.extract_kwargs(style, 'edgeline')

    if not xlim:
        xlim = axes.get_xlim()

    if label_equation is True:
        sigfig = 2
        if isinstance(regression_result, toolbox.regress.LinregressResult):
            style['label'] = f'{style.get("label", "")} {regression_result.label(sigfig)}'.strip()
        else:
            var = y_eq(xlim[1]) - y_eq(xlim[0])
            label_intercept = y_eq(0)
            label_slope = y_eq(1) - label_intercept

            if not np.isnan(label_intercept):
                label_intercept = _format_sigfig(label_intercept, sigfig, var)
            if not np.isnan(label_slope):
                label_slope = _format_sigfig(label_slope, sigfig, var)

            style['label'] = f'{style.get("label", "")} y={label_slope}x + ' \
                             f'{label_intercept}'.strip()


    x = np.linspace(xlim[0], xlim[1], 10000)
    y = np.array([y_eq(xval) for xval in x])


    if y_se_eq is not None:
        try:
            yerr = np.array([y_se_eq(xval) for xval in x])
            yerr_upper = y + yerr
            yerr_lower = y - yerr
            yerr_coordinates = [*zip(x, yerr_upper)] + list(reversed([*zip(x, yerr_lower)]))
        except NotImplementedError:
            y_se_eq = None

    ull_function = None
    if not autoscale and hasattr(axes, '_update_line_limits'):
        # This turns of auto updating the limits
        ull_function = axes._update_line_limits
        axes._update_line_limits = lambda *args, **kwargs: None

    try:
        if line is not False:
            axes.plot(x, y, scalex=False, scaley=False, **style)

        if y_se_eq is not None and edgeline:
            axes.plot(x, yerr_upper, scalex=False, scaley=False, **edge_style)
            axes.plot(x, yerr_lower, scalex=False, scaley=False, **edge_style)
    finally:
        if ull_function:
            axes._update_line_limits = ull_function

    if y_se_eq is not None and fill:
        _plot_polygon(axes, yerr_coordinates, autoscale, **fill_style)

@_update_figure_and_axes
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
    >>> array = isopy.to_refval.isotope.fraction.to_array(element_symbol='pd')
    >>> isopy.tb.plot_spider(plt, array) #Will plot the fraction of each Pd isotope
    >>> plt.show()

    .. figure:: plots/plot_spider1.png
        :alt: output of the example above
        :align: center


    >>> subplots = isopy.tb.create_subplots(plt, [['left', 'right']])
    >>> array = isopy.to_refval.isotope.fraction.to_array(element_symbol='pd').ratio('105pd')
    >>> isopy.tb.plot_spider(subplots['left'], array) #The numerator mass numbers are used as x
    >>> isopy.tb.plot_spider(subplots['right'], array, constants={105: 1}) #Adds a one for the denominator
    >>> plt.show()

    .. figure:: plots/plot_spider2.png
        :alt: output of the example above
        :align: center

    >>> subplots = isopy.tb.create_subplots(plt, [['left', 'right']], figwidth=8)
    >>> values = {100: [1, 1, -1], 101.5: [0, 0, 0], 103: [-1, 1, -1]} #keys can be floats
    >>> isopy.array(values)
    >>> isopy.tb.plot_spider(subplots['left'], values) #Impossible to tell the rows apart
    >>> isopy.tb.plot_spider(subplots['right'], values, xscatter=0.15) #Much clearer
    >>> plt.show()

    .. figure:: plots/plot_spider3.png
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

    if isopy.iskeylist(x) and x.flavour in isopy.asflavour('ratio'):
        if x.common_denominator is None:
            raise ValueError(
                'x values can only be inferred from a ratio key list with a common denominator')
        else:
            x = x.numerators

    if isopy.iskeylist(x) and x.flavour in isopy.asflavour('isotope|molecule[isotope]'):
        x = x.mz

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
        if isinstance(constants, str):
            constants = isopy.askeystring(constants)
            if constants.flavour == 'ratio':
                xconstants = np.array([constants.numerator.mz, constants.denominator.mz])
                yconstants = np.array([0, 0])
            else:
                xconstants = np.array([constants.mz])
                yconstants = np.array([0])
        elif isinstance(constants, dict):
            xconstants = np.array(list(constants.keys()), dtype=np.float64)
            yconstants = np.array(list(constants.values()), dtype=np.float64)
        else:
            raise ValueError(f'constants must be a dictionary not "{type(constants).__name__}"')

        if xconstants.ndim != 1 or yconstants.ndim != 1:
            raise ValueError(f'constants must be scalar values')
        xconstants.reshape(1, -1)
        yconstants.reshape(1, -1)

        x = np.append(x, np.full((x.shape[0], xconstants.size), xconstants), axis = 1)
        y = np.append(y, np.full((y.shape[0], yconstants.size), yconstants), axis = 1)
        if yerr is not None:
            yerr = np.append(yerr, np.full((yerr.shape[0], yconstants.size), np.nan), axis = 1)

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

    prop_cycler = next(axes._get_lines.prop_cycler)

    if color:
        style['color'] = color
    else:
        style.setdefault('color', prop_cycler.get('color', 'blue'))

    if marker is True:
        style.setdefault('marker', prop_cycler.get('marker', 'o'))
    elif marker is False:
        style['marker'] = ''
    elif marker:
        style['marker'] = marker

    if line is True: style.setdefault('linestyle', '-')
    elif line is False: style['linestyle'] = ''
    elif line: style['linestyle'] = line

    style.setdefault('markerfacecolor', style['color'])
    style.setdefault('zorder', 2)

    regress_style = core.extract_kwargs(style, 'regress')
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

@_update_figure_and_axes
def plot_vstack(axes, x, xerr = None, ystart = None, *, outliers=None, cval = None, pmval=None, color=None, marker=True, line=False,
                cline = True, pmline= False, box = True, pad = 2, box_pad = None,
                spacing = -1, hide_yaxis=True, compare=False, subplots_grid = (1, -1), **kwargs):
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
        outliers. If *cval* is ``True`` ``np.mean`` is used to calculate *cval*.
    pmval : scalar, Callable, Optional
        The uncertainty of the the data. Can either be a scalar or a function that returns a scalar
        when called with *x*. If *pmval* is a function is it called on the values that are not
        outliers. If *pmval* is ``True`` ``isopt.sd2`` is used to calculate *pmval*.
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
    compare : bool
        If ``True`` *axes* is passed to ``plot_vcompare`` at the end of the function.
    subplots_grid : tuple[int, int]
        The shape of the grid to be created if nessecary.
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
    >>> array1 = isopy.random(100, -0.5, seed=46)
    >>> array2 = isopy.random(100, 0.5, seed=47)
    >>> isopy.tb.plot_vstack(plt, array1)
    >>> isopy.tb.plot_vstack(plt, array2, cval=np.mean, pmval=isopy.sd2)
    >>> plt.show()

    .. figure:: plots/plot_vstack1.png
        :alt: output of the example above
        :align: center

    >>> keys = isopy.keylist('pd105', 'ru101', 'cd111')
    >>> array1 = isopy.random(100, -0.5, keys, seed=46)
    >>> array2 = isopy.random(100, 0.5, keys, seed=47)
    >>> isopy.tb.create_subplots(plt, keys.sorted(), (1, -1))
    >>> isopy.tb.plot_vstack(plt, array1)
    >>> isopy.tb.plot_vstack(plt, array2, cval=np.mean, pmval=isopy.sd2)
    >>> plt.show()

    .. figure:: plots/plot_vstack2.png
        :alt: output of the example above
        :align: center


    >>> keys = isopy.keylist('pd105', 'ru101', 'cd111')
    >>> array = isopy.random(100, -0.5, keys, seed=46)
    >>> mean = np.mean(array);
    >>> sd = isopy.sd(array)
    >>> outliers = isopy.tb.find_outliers(array, mean, sd)
    >>> isopy.tb.create_subplots(plt, keys.sorted(), (1, -1))
    >>> isopy.tb.plot_vstack(plt, array, cval=mean, pmval=sd, outliers=outliers, color=('red', 'pink'))
    >>> plt.show()

    .. figure:: plots/plot_vstack3.png
        :alt: output of the example above
        :align: center

    """
    if isinstance(x, tuple) and len(x) == 2:
        x, xerr = x

    if hasattr(x, 'items'):
        _plot_stack_array(plot_vstack, axes, 'x', type(x), subplots_grid, x=x, xerr=xerr, ystart=ystart, cval=cval,
                          pmval=pmval, color=color, marker=marker, line=line, cline=cline,
                          pmline=pmline, box=box, pad=pad, box_pad=box_pad, spacing=spacing,
                          outliers=outliers, hide_yaxis=hide_yaxis, compare=compare, **kwargs)

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

        x, y, xerr, cx, cy, pmx, pmy, boxx, boxy, styles, compare_kwargs = _plot_stack_step1(axes, x, xerr, ystart, cval, pmval, color,
                                                                             marker, line, cline, pmline, box, pad, box_pad, spacing, outliers, **kwargs)

        axes.get_yaxis().set_visible(not hide_yaxis)
        _plot_stack_step2(axes, x, y, xerr, None, cx, cy, pmx, pmy, boxx, boxy, styles, outliers)

        if compare:
            plot_vcompare(axes, **compare_kwargs)

@_update_figure_and_axes
def plot_hstack(axes, y, yerr = None, xstart = None, *, outliers=None, cval = None, pmval=None, color=None, marker=True, line=False,
                cline = True, pmline= False, box = True, pad = 2, box_pad = None,
                spacing = 1, hide_xaxis=True, compare=False, subplots_grid = (-1, 1), **kwargs):
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
        outliers. If *cval* is ``True`` ``np.mean`` is used to calculate *cval*.
    pmval : scalar, Callable, Optional
        The uncertainty of the the data. Can either be a scalar or a function that returns a scalar
        when called with *x*. If *pmval* is a function is it called on the values that are not
        outliers. If *pmval* is ``True`` ``isopt.sd2`` is used to calculate *pmval*.
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
    compare : bool
        If ``True`` *axes* is passed to ``plot_vcompare`` at the end of the function.
    subplots_grid : tuple[int, int]
        The shape of the grid to be created if nessecary.
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
    >>> array1 = isopy.random(100, -0.5, seed=46)
    >>> array2 = isopy.random(100, 0.5, seed=47)
    >>> isopy.tb.plot_hstack(plt, array1)
    >>> isopy.tb.plot_hstack(plt, array2, cval=np.mean, pmval=isopy.sd2)
    >>> plt.show()

    .. figure:: plots/plot_hstack1.png
        :alt: output of the example above
        :align: center

    >>> keys = isopy.keylist('pd105', 'ru101', 'cd111')
    >>> array1 = isopy.random(100, -0.5, keys, seed=46)
    >>> array2 = isopy.random(100, 0.5, keys, seed=47)
    >>> isopy.tb.create_subplots(plt, keys.sorted(), (-1, 1))
    >>> isopy.tb.plot_hstack(plt, array1)
    >>> isopy.tb.plot_hstack(plt, array2, cval=np.mean, pmval=isopy.sd2)
    >>> plt.show()

    .. figure:: plots/plot_hstack2.png
        :alt: output of the example above
        :align: center

    >>> keys = isopy.keylist('pd105', 'ru101', 'cd111')
    >>> array = isopy.random(100, -0.5, keys, seed=46)
    >>> mean = np.mean(array);
    >>> sd = isopy.sd(array)
    >>> outliers = isopy.tb.find_outliers(array, mean, sd)
    >>> isopy.tb.create_subplots(plt, keys.sorted(), (-1, 1))
    >>> isopy.tb.plot_hstack(plt, array, cval=mean, pmval=sd, outliers=outliers, color=('red', 'pink'))
    >>> plt.show()

    .. figure:: plots/plot_hstack3.png
        :alt: output of the example above
        :align: center

    """
    if isinstance(y, tuple) and len(y) == 2: y, yerr = y

    if hasattr(y, 'items'):
        _plot_stack_array(plot_hstack, axes, 'y', type(y), subplots_grid, y=y, yerr=yerr, xstart=xstart, cval=cval,
                          pmval=pmval, color=color, marker=marker, line=line, cline=cline,
                          pmline=pmline, box=box, pad=pad, box_pad=box_pad, spacing=spacing,
                          outliers=outliers, hide_xaxis=hide_xaxis, compare=compare, **kwargs)

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

        y, x, yerr, cy, cx, pmy, pmx, boxy, boxx, styles, compare_kwargs = _plot_stack_step1(axes, y, yerr, xstart, cval, pmval, color,
                                                                             marker, line, cline, pmline, box, pad, box_pad, spacing, outliers, **kwargs)

        axes.get_xaxis().set_visible(not hide_xaxis)
        _plot_stack_step2(axes, x, y, None, yerr, cx, cy, pmx, pmy, boxx, boxy, styles, outliers)

        if compare:
            plot_hcompare(axes, **compare_kwargs)

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

    if box_cval is not None:
        if box_cval is True:
            box_cval = np.mean
        if callable(box_cval):
            box_cval = box_cval(v[include])
        try:
            box_cval = np.array(float(box_cval))
        except:
            raise ValueError(f'unable to convert cval {box_cval} to float')


    if box_pmval is not None:
        if box_pmval is True:
            box_pmval = isopy.sd2
        if callable(box_pmval):
            box_pmval = box_pmval(v[include])

        try:
            box_pmval = np.asarray(float(box_pmval))
        except:
            raise ValueError(f'unable to convert pmval {box_pmval} to float')

    core.extract_kwargs(kwargs, 'subplots') #get rid of these
    compare_kwargs = core.extract_kwargs(kwargs, 'compare')
    style = {}
    style.update(stack_style)
    style.update(kwargs)

    prop_cycler = next(axes._get_lines.prop_cycler)

    if isinstance(color, tuple):
        style['color'] = color[0]
        style['outlier_color'] = color[1]
    else:
        if color:
            style['color'] = color
        else:
            style.setdefault('color', prop_cycler.get('color', 'blue'))
        style['outlier_color'] = modify_color(style['color'], lightness=0.85)

    if marker is True:
        style.setdefault('marker', prop_cycler.get('marker', 'o'))
    elif marker is False:
        style['marker'] = ''
    elif marker:
        style['marker'] = marker

    style.setdefault('markerfacecolor', style['color'])
    style.setdefault('outlier_markerfacecolor', style['outlier_color'])

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

    cline_style = core.extract_kwargs(style, 'cline')
    pmline_style = core.extract_kwargs(style, 'pmline')
    box_style = core.extract_kwargs(style, 'box')

    temp_style = core.extract_kwargs(style, 'outlier')
    outlier_style = style.copy()
    outlier_style.pop('label', None) #Otherwise the label gets shown twice
    outlier_style.update(temp_style)

    styles = dict(main=style, cline=cline_style, pmline = pmline_style, box=box_style, outlier=outlier_style)

    if not isinstance(pad, (float, int)): raise TypeError(f'pad must be float or an integer')
    if box_pad is None: box_pad = pad/4
    elif not isinstance(box_pad, (float, int)): raise TypeError(f'pad must be float or an integer')

    if box_cval is not None and cline is not False:
        cv = np.array([box_cval, box_cval])
        cw = np.array([min((wstart, wstop)) - box_pad, max((wstart, wstop)) + box_pad])
    else:
        cv = None
        cw = None

    if box_pmval is not None and box_cval is not None and pmline is not False:
        pmv = (np.array([box_cval + box_pmval, box_cval + box_pmval]),
               np.array([box_cval - box_pmval, box_cval - box_pmval]))
        pmw = (np.array([max((wstart, wstop)) + box_pad, min((wstart, wstop)) - box_pad]),
               np.array([max((wstart, wstop)) + box_pad, min((wstart, wstop)) - box_pad]))
    else:
        pmv = None
        pmw = None

    if box_pmval is not None and box_cval is not None and box_shaded is not False:
        boxv = np.array([box_cval + box_pmval, box_cval - box_pmval])
        boxw = np.array([max((wstart, wstop)) + box_pad, min((wstart, wstop)) - box_pad])
    else:
        boxv = None
        boxw = None

    return v, w, verr, cv, cw, pmv, pmw, boxv, boxw, styles, compare_kwargs

def _plot_stack_array(func, axes, vaxis, vtype, grid, /, **kwargs):
    if vaxis == 'x':
        waxis = 'y'
    elif vaxis == 'y':
        waxis = 'x'
    else:
        raise ValueError('vaxis incorrect')

    keys = kwargs[vaxis].keys()
    subplot_kwargs = core.extract_kwargs(kwargs, 'subplots')

    if isinstance(axes, dict):
        named_axes = axes
    else:
        figure =_check_figure('axes', axes)
        axes = figure.get_axes()
        if len(axes) == 0:
            named_axes = create_subplots(figure, [str(key) for key in keys], grid, **subplot_kwargs)
        else:
            named_axes = {str(ax.get_label()): ax for ax in axes}

    if kwargs[f'{waxis}start'] is None:
        find_wstart = True
    else:
        find_wstart = False

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')

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
        func(ax, **{k: v.get(str(key)) if type(v) is vtype else v for k, v in
                    kwargs.items()})

        if waxis == 'x':
            ax.set_ylabel(str(key))
        elif waxis == 'y':
            ax.set_xlabel(str(key))

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')

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

@_update_figure_and_axes
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
    >>> array1 = isopy.random(100, 0.9, seed=46)
    >>> array2 = isopy.random(100, 1.1, seed=47)
    >>> isopy.tb.plot_hstack(plt, array1, cval=np.mean, pmval=isopy.sd2)
    >>> isopy.tb.plot_hstack(plt, array2, cval=np.mean, pmval=isopy.sd2)
    >>> isopy.tb.plot_hcompare(plt)
    >>> plt.show()

    .. figure:: plots/plot_hcompare1.png
        :alt: output of the example above
        :align: center

    >>> array1 = isopy.random(100, 0.9, seed=46)
    >>> array2 = isopy.random(100, 1.1, seed=47)
    >>> isopy.tb.plot_hstack(plt, array1, cval=np.mean, pmval=isopy.sd2)
    >>> isopy.tb.plot_hstack(plt, array2, cval=np.mean, pmval=isopy.sd2)
    >>> isopy.tb.plot_hcompare(plt, pmval=isopy.sd, sigfig=3)
    >>> plt.show()

    .. figure:: plots/plot_hcompare2.png
        :alt: output of the example above
        :align: center

    >>> pmunits = ['diff', 'abs', '%', 'ppt', 'ppm', 'ppb']
    >>> subplots = isopy.tb.create_subplots(plt, pmunits, (-1, 1), figure_height=8)
    >>> array1 = isopy.random(100, 0.9, seed=46)
    >>> array2 = isopy.random(100, 1.1, seed=47)
    >>> for unit, axes in subplots.items():
            isopy.tb.plot_hstack(axes, array1, cval=np.mean, pmval=isopy.sd2)
            isopy.tb.plot_hstack(axes, array2, cval=np.mean, pmval=isopy.sd2)
            isopy.tb.plot_hcompare(axes, pmval=isopy.sd, pmunit=unit, combine_ticklabels=True)
            axes.set_ylabel(f'pmunit="{unit}"')
    >>> plt.show()

    .. figure:: plots/plot_hcompare3.png
        :alt: output of the example above
        :align: center

    >>> keys = isopy.keylist('pd105', 'ru101', 'cd111')
    >>> array1 = isopy.random(100, 0.9, keys, seed=46)
    >>> array2 = isopy.random(100, 1.1, keys, seed=47)
    >>> isopy.tb.plot_hstack(plt, array1, cval=np.mean, pmval=isopy.sd2)
    >>> isopy.tb.plot_hstack(plt, array2, cval=np.mean, pmval=isopy.sd2)
    >>> isopy.tb.plot_hcompare(plt.gcf(), combine_ticklabels=True)
    >>> plt.show()

    .. figure:: plots/plot_hcompare4.png
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

@_update_figure_and_axes
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
    >>> array1 = isopy.random(100, -0.5, seed=46)
    >>> array2 = isopy.random(100, 0.5, seed=47)
    >>> isopy.tb.plot_vstack(plt, array1, cval=np.mean, pmval=isopy.sd2)
    >>> isopy.tb.plot_vstack(plt, array2, cval=np.mean, pmval=isopy.sd2)
    >>> isopy.tb.plot_vcompare(plt)
    >>> plt.show()

    .. figure:: plots/plot_vcompare1.png
        :alt: output of the example above
        :align: center


    >>> array1 = isopy.random(100, -0.5, seed=46)
    >>> array2 = isopy.random(100, 0.5, seed=47)
    >>> isopy.tb.plot_vstack(plt, array1, cval=np.mean, pmval=isopy.sd2)
    >>> isopy.tb.plot_vstack(plt, array2, cval=np.mean, pmval=isopy.sd2)
    >>> isopy.tb.plot_vcompare(plt, pmval=isopy.sd, sigfig=3)
    >>> plt.show()

    .. figure:: plots/plot_vcompare2.png
            :alt: output of the example above
            :align: center

    >>> pmunits = ['diff', 'abs', '%', 'ppt', 'ppm', 'ppb']
    >>> subplots = isopy.tb.create_subplots(plt, pmunits, (1, -1), figure_width=8)
    >>> array1 = isopy.random(100, 0.9, seed=46)
    >>> array2 = isopy.random(100, 1.1, seed=47)
    >>> for unit, axes in subplots.items():
            isopy.tb.plot_vstack(axes, array1, cval=np.mean, pmval=isopy.sd2)
            isopy.tb.plot_vstack(axes, array2, cval=np.mean, pmval=isopy.sd2)
            isopy.tb.plot_vcompare(axes, pmval=isopy.sd, pmunit=unit, combine_ticklabels=True)
            axes.set_xlabel(f'pmunit="{unit}"')
    >>> plt.show()

    .. figure:: plots/plot_vcompare3.png
            :alt: output of the example above
            :align: center

    >>> keys = isopy.keylist('pd105', 'ru101', 'cd111')
    >>> array1 = isopy.random(100, 0.9, keys, seed=46)
    >>> array2 = isopy.random(100, 1.1, keys, seed=47)
    >>> isopy.tb.plot_vstack(plt, array1)
    >>> isopy.tb.plot_vstack(plt, array2, cval=np.mean, pmval=isopy.sd2)
    >>> isopy.tb.plot_vcompare(plt.gcf(), combine_ticklabels=True)
    >>> plt.show()

    .. figure:: plots/plot_vcompare4.png
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
        if variation == 0:
            precision = sigfig
        else:
            precision = sigfig - int(np.log10(np.abs(variation)))
        if precision < 0:
            precision = 0
    else:
        precision = 0

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

@_update_figure_and_axes
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

@_update_figure_and_axes
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

@_update_figure_and_axes
def plot_text(axes, xpos, ypos, text, rotation = None, fontsize = None, posabs = True, **kwargs):
    """
    Add text to plot.

    Parameters
    ----------
    axes
        The axes on which the regression will we plotted. Must be a matplotlib axes object or any
        object with a gca() method that return a matplotlib axes object, such as a matplotlib
        pyplot instance.
    xpos, ypos
        Position of the text in the plot. If *posabs* is True then the text is placed at these data coordinates.
        If *posabs* is False then the text is placed at this relative position.
    text
        The text to be included in the plot.
    rotation
        If given the text is rotated anti-clockwise by this many degrees.
    fontsize
        The fontsize of the text.
    posabs
        If True *xpos* and *ypos* represents data coordinates. If False *xpos* and *ypos* represents
        the relative position on the axes in the interval 0 to 1.
    kwargs
        Any other valid keyword argument for the :meth:`matplotlib.axes.Axes.text` method.
    """
    axes = _check_axes(axes)

    if rotation:
        kwargs['rotation'] = rotation

    if fontsize:
        kwargs['fontsize'] = fontsize

    if posabs is False:
        kwargs['transform'] = axes.transAxes

    axes.text(xpos, ypos, text, **kwargs)

@_update_figure_and_axes
def plot_contours(axes, x, y, z, zmin=None, zmax=None, levels=100, colors='jet',
                  colorbar=None, label = False, label_levels = None, filled = True, **kwargs):
    """
    Create a contour plot on *axes* from a data grid.

    Parameters
    ----------
    axes : axes, figure, plt
        Either a matplotlib axes object or a matplotlib Figure object. Objects with a gca() method
        that return a matplotlib axes object, such as a matplotlib pyplot instance are also accepted.
        If *axes* is a matplotlib Figure object then :func:`plot_vcompare` will be called on every
        named subplot in the figure.
    x
        A 1-dimensional array containing the x-axis values.
    y
        A 1-dimensional array containing the y-axis values.
    z
        A 2-dimensional array containing the z-axis values.
    zmin
        The smallest value in the colour map. Any z values smaller than this value will
        have the same colour as the *zmin* value.
    zmax
        The largest value in the colour map. Any z values larger than this value will
        have the same colour as the *zmax* value.
    levels
        The number of intervals in the colour map. Can also be a sequence of levels to be used.
    colors
        This can either be a list of colours or a single string of the colour map to be used.
        Names of avaliable colour maps can be found `here <https://matplotlib.org/stable/tutorials/colors/colormaps.html#>`.
    colorbar : bool, axes
        If ``False`` no colour bar is drawn. If ``True`` a colour bar is drawn on the same axes as
        contours. If *colorbar* is an axes the colourbar will be drawn on that axes. If ``None`` it
        default to ``True`` if *filled* is ``True`` otherwise it is ``False``.
    label
        If ``True`` then a label will be added to each *label_level*.
    label_levels
        The number of intervals to be labeled. Can also be a sequence of levels to be labeled.
    filled : bool
        If ``False`` only contour lines are shown. If ```True`` filled contours are drawn.
    kwargs
        Kwargs prefixed with ``'label_`` will be sent to to the ``clabel`` function.
        See avaliable options `here <https://matplotlib.org/stable/api/contour_api.html#matplotlib.contour.ContourLabeler.clabel>`.
        All other kwargs will be send to the matplotlib functions ``contour`` or ``contourf``.
        See avaliable options `here <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contour.html>`.

    Examples
    --------
    >>> x = np.arange(10)
    >>> y = np.arange(10, 101, 10)
    >>> z = np.arange(100).reshape((10, 10))
    >>> isopy.tb.plot_contours(plt, x, y, z)
    >>> plt.show()


    .. figure:: plots/plot_contours1.png
        :alt: output of the example above
        :align: center

    >>> element = 'fe'
    >>> spike1 = isopy.array(fe54=0, fe56=0, fe57=1, fe58=0)
    >>> spike2 = isopy.array(fe54=0, fe56=0, fe57=0, fe58=1)
    >>> axes_labels = dict(axes_xlabel = 'Proportion of double spike in double spike-sample mix',
                   axes_ylabel='Proportion of spike1 in double spike',
                   axes_title = 'Uncertainty in α (1SD)')
    >>> dsgrid = isopy.tb.ds_grid(element, spike1, spike2)
    >>> isopy.tb.plot_contours(plt, *dsgrid.xyz(), zmin= 0, zmax=0.01, **axes_labels)
    >>> plt.show()

    .. figure:: plots/plot_contours2.png
        :alt: output of the example above
        :align: center

    """
    axes = _check_axes(axes)

    x = _check_val('x', x, flatten=False)
    y = _check_val('y', y, flatten=False)
    z = _check_val('z', z, flatten=False)

    if x.ndim != 1:
        raise ValueError(f'x must have a dimension of 1')
    if y.ndim != 1:
        raise ValueError(f'y must have a dimension of 1')
    if z.ndim != 2:
        if z.ndim == 1 and z.size == x.size * y.size:
            z = z.reshape((y.size, x.size))
        else:
            raise ValueError(f'z must have a dimension of 2')
    elif z.shape[0] != y.size or z.shape[1] != x.size:
        raise ValueError(f'shape of z {z.shape} does not match shape of x and z ({y.size}, {x.size})')

    label_kwargs = core.extract_kwargs(kwargs, 'label')

    if type(colors) is str:
        style = dict(cmap=colors)
    else:
        style = dict(colors=colors)
    style.update(grid_style)
    style.update(kwargs)

    if zmin is None:
        zmin = np.nanmin(z)
    elif callable(zmin):
        zmin = zmin(z)
    style.setdefault('vmin', zmin)

    if zmax is None:
        zmax = np.nanmax(z)
    elif callable(zmax):
        zmax = zmax(z)
    style.setdefault('vmax', zmax)

    if hasattr(levels, '__iter__'):
        style['levels'] = levels
    elif levels is not None:
        style['levels'] = np.linspace(zmin, zmax, levels)

    if filled:
        contours = axes.contourf(x, y, z, **style)
    else:
        contours = axes.contour(x, y, z, **style)

    if colorbar is None:
        colorbar = filled

    if isinstance(colorbar, mplAxes):
        figure = axes.get_figure()
        figure.colorbar(contours, cax=colorbar)
    elif colorbar is True:
        figure = axes.get_figure()
        figure.colorbar(contours, ax=axes)

    if label is True:
        if type(label_levels) is int:
            label_levels = np.linspace(zmin, zmax, levels)

        if label_levels is None:
            axes.clabel(contours, **label_kwargs)
        else:
            axes.clabel(contours, label_levels, **label_kwargs)

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

def _check_axes(axes):
    if not isinstance(axes, mpl.axes.Axes):
        try:
            axes = axes.gca()
        except:
            pass
        if not isinstance(axes, mpl.axes.Axes):
            raise TypeError(f'"axes" must be a matplotlib axes object or matplotlib.pyplot instance')
    return axes



    return wrap_cla(axes)

def _check_figure(var_name, figure):
    if isinstance(figure, mplAxes):
        return figure.get_figure()
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

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')

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

def update_style():
    prop_cycler = mpl.rcParams['axes.prop_cycle'].by_key()
    colors = prop_cycler.get('color', Colors._original)
    markers = prop_cycler.get('marker', Markers._original)

    Colors._original = colors
    Markers._original = markers
    mpl.rcParams['axes.prop_cycle'] = cycler.cycler(color = colors * len(markers), marker = markers * len(colors))

update_style()