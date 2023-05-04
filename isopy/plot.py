import isopy as isopy
import isopy.core as core
import matplotlib as mpl
import matplotlib.pyplot as mplplt
from matplotlib.patches import Polygon as mplPolygon
from matplotlib.figure import Figure as mplFigure
from matplotlib.container import ErrorbarContainer
from matplotlib.axes import Axes as mplAxes
import numpy as np
import functools
import warnings
import cycler
import colorsys

scatter_style = {'markeredgecolor': 'black', 'ecolor': 'grey', 'capsize': 3, 'elinewidth': 1, 'zorder': 2}
regression_style = {'linestyle': '-', 'color': 'black', 'edgeline_linestyle': '--', 'fill_alpha': 0.3, 'zorder': 1, 'marker': '', 'edgeline_marker': ''}
equation_line_style = {'linestyle': '-', 'color': 'black', 'zorder': 1}
stack_style = {'markeredgecolor': 'black', 'ecolor': 'black', 'capsize': 3, 'elinewidth': 1, 'cline_linestyle': '-',
               'pmline_linestyle': '--', 'box_alpha': 0.3, 'zorder': 2, 'cline_marker': ''}
spider_style = {'markeredgecolor': 'black', 'ecolor': 'black', 'capsize': 3, 'elinewidth': 1, 'zorder': 2}
box_style = dict(color='black', zorder = 0.9, alpha=0.3)
polygon_style = dict(color='black', zorder = 0.9, alpha=0.3)
grid_style = dict(extend='both')
cross_style = dict(color='black', linestyle='-', linewidth=1, zorder=0.8)

nan = np.nan

class AxesPropCycler:
    def __init__(self, axes):
        self._propcycler = axes._get_lines.prop_cycler
        self._current = None
    
    def next(self):
        self._current = next(self._propcycler)
        
    def get(self, attr, default):
        if self._current is None:
            self.next()
        
        return self._current.get(attr, default)

    
def modify_color(color, *, scale=None, lightness=None):
    color = mpl.colors.ColorConverter.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(*color)

    if lightness is not None:
        l = lightness
    if scale is not None:
        l = min(1, l * scale)

    return colorsys.hls_to_rgb(h, l, s)

# New
def update_figure_and_axes(func):
    """
    Allows figure kwargs to be passed to functions
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        fig_kwargs = core.extract_kwargs(kwargs, 'figure')
        axes_kwargs = core.extract_kwargs(kwargs, 'axes')

        if fig_kwargs:
            self.update_figure(**fig_kwargs)

        ret = func(self, *args, **kwargs)

        if axes_kwargs:
            self.update_axes(**axes_kwargs)

        return ret
    return wrapper

class IsopyPlot:
    def __init__(self, figure = None) -> None:
        if figure is None or figure is mplplt:
            self._fig = mplplt
        elif isinstance(figure, mplFigure):
            self._fig = figure
        elif isinstance(figure, mplAxes):
            self._fig = figure.gcf()
            self._fig.sca(figure)
        else:
            raise TypeError('figure must be an instance of matplotlib.Figure')
    
    def __getattr__(self, name):
        ax = self.subplots.get(name, None)
        if ax:
            self._fig.sca(ax)
            return self
        else:
            raise AttributeError(f'No method or subplot named "{name}"')
    
    def __getitem__(self, name):
        return self.__getattr__(name)
        
    @property
    def figure(self):
        if isinstance(self._fig, mplFigure):
            return self._fig
        else:
            return self._fig.gcf()
        
    def get_axes(self):
        return self.figure.get_axes()
    
    @property
    def ax(self):
        return self.gca()
    
    def gcf(self):
        return self.figure
    
    def gca(self):
        return self.figure.gca()
    
    def sca(self, ax):
        self._fig.sca(ax)
        return self
    
    def clear(self):
        return self.cla()
        
    def cla(self):
        return self.gca().clear()
    
    def clf(self):
        return self.figure.clear()
    
    @property
    def subplots(self):
        return SubplotDict({label: ax for ax in self.figure.get_axes() if (label:=ax.get_label())})
    
    ###########################
    ### Plotting Functions ####
    ###########################
    def update_figure(self, *, size = None, width=None, height=None, dpi=None, facecolor=None, edgecolor=None,
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
        """
        figure = self.gcf()
        
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

    def update_axes(self, *, legend=None, cross = None, **kwargs):
        """
        Update axes attributes.

        All isopy plotting functions will automatically send any *kwargs* prefixed with ``axes_`` to
        this function.

        Parameters
        ----------
        legend : bool
            If ``True`` the legend will be shown on the axes. If ``False`` no legend is shown.
        kwargs
            Any attribute that can be set using ``axes.set_<key>(value)``.
            If the value is a tuple ``axes.set_<key>(*value)`` is used. If
            the value is a dictionary ``axes.set_<key>(**value)``.
        """
        figure_kwargs = core.extract_kwargs(kwargs, 'figure')
        if figure_kwargs:
            self.update_figure(**figure_kwargs)
            
        ax = self.gca()

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
        
        if type(cross) in (list, tuple) and len(cross) == 2:
            if cross[0] is not None:
                ax.axvline(cross[0], **cross_style)
            if cross[1] is not None:
                ax.axhline(cross[1], **cross_style)
        elif cross is not None:
            ax.axvline(cross, **cross_style)
            ax.axhline(cross, **cross_style)
    
    @update_figure_and_axes
    def update_subplots(self, *subplots, legend=None, exclude='legend', **kwargs):
        if exclude is None or exclude is False:
            exclude = []
        elif type(exclude) == str:
            exclude = [exclude]
        elif type(exclude) not in (list, tuple):
            raise TypeError(f'exclude must be a None, str, list or tuple not {type(exclude)}')
        
        included_subplots = {name: subplot for name, subplot in self.subplots if name not in exclude}
        if len(subplots) == 0:
            included_subplots = included_subplots.values()
        else:
            included_subplots = [subplot for name, subplot in included_subplots.items() if name in subplots]
        
        for subplot in included_subplots:
            self.sca(subplot)
            self.update_axes(legend, **kwargs)
        
        
    @core.add_preset_method('v', grid = (-1, 1))
    @core.add_preset_method('h', grid = (1, -1))
    def create_subplots(self, subplots, grid = None, *,
                        row_height = None, column_width = None,
                        legend = None, legend_ratio = None,
                        constrained_layout=True, height_ratios = None, width_ratios=None,  
                        twinx = None, twiny=None, **kwargs):
        """
        create_subplots(subplots, grid = None, *,
                        row_height = None, column_width = None,
                        legend = None, legend_ratio = None,
                        constrained_layout=True, height_ratios = None, width_ratios=None,  **kwargs)
                        
        Create subplots for a figure.

        Use None instead of a name to create an empty space.

        Parameters
        ----------
        subplots : list, int
            A nested list of subplot names. A visual layout of how you want your subplots
            to be arranged labeled as strings. *subplots* can be a 1-dimensional list if grid is specified. ``None`` will
            create an empty space. An integer of the number of subplots can be given if grid is specified. Each axes will be
            names ``ax<i>`` e.g. ``[["ax0", "ax1"], ["ax2", "ax3"]]`` .
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
        twinx
            The name, or list of names, for the subplot(s) where a twin x-axis should be created. The twin axis will be given
            the name ``<name>_twinx``.
        twiny
            The name, or list of names, for the subplot(s) where a twin y-axis should be created. The twin axis will be given
            the name ``<name>_twiny``.
        kwargs
            Prefix kwargs to be passed along with he creation of each subplot with ``subplot_``.
            Kwargs to be passed to the GridSpec should be prefixed ``gridspec_``. See matplotlib docs
            for
            `add_subplot <https://matplotlib.org/stable/api/figure_api.html?highlight=subplot_mosaic#matplotlib.figure.Figure.add_subplot>`_
            and `Gridspec <https://matplotlib.org/stable/api/_as_gen/matplotlib.gridspec.GridSpec.html>`_
            for a list of avaliable kwargs.

        Returns
        -------
        subplots : dict
            A dictionary containing the newly created subplots.
        """
        figure_kwargs = core.extract_kwargs(kwargs, 'figure')
        figure_kwargs['constrained_layout'] = constrained_layout
        self.update_figure(**figure_kwargs)
        axes_kwargs = core.extract_kwargs(kwargs, 'axes')
        
        figure = self.gcf()
        empty_sentinel = None
        subplot_kw = core.extract_kwargs(kwargs, 'subplot')
        gridspec_kw = core.extract_kwargs(kwargs, 'gridspec')
        
        if height_ratios:
            gridspec_kw['height_ratios'] = height_ratios
        if width_ratios:
            gridspec_kw['width_ratios'] = width_ratios
        if type(subplots) is int:
            subplots = [f'ax{i}' for i in range(subplots)]
        if hasattr(subplots, 'keys'):
            subplots = list(subplots.keys)

        if legend is True:
            if grid is not None and grid[0] == -1:
                legend = 's'
            else:
                legend = 'e'
                
        if grid is None and type(subplots[0]) not in (list, tuple):
            raise ValueError('*subplots* must be a nested list if *grid* is not given')
                    
        if grid is not None and type(subplots[0]) not in (list, tuple):
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
                    twinxname, twinyname = None, None
                    if type(axname) is tuple:
                        if len(axname) == 1:
                            axname = axname[0]
                        elif len(axname) == 2:
                            axname, twinxname = axname
                        elif len(axname) == 3:
                            axname, twinxname, twinyname = axname
                        else:
                            raise ValueError('Invalid tuple')

                    axname = str(axname)
                    if twinxname is not empty_sentinel:
                        axtwinx[axname] = str(twinxname)

                    if twinyname is not empty_sentinel:
                        axtwiny[axname] = str(twinyname)

                    subplots[r][c] = axname

        if legend:          
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

        if isinstance(twinx, str):
            twinx = [twinx]
        if isinstance(twiny, str):
            twiny = [twiny]
        
        for label, ax in axes.items():
            ax.set_label(label) # Is this already done?
            if label in axtwinx:
                twin_label = axtwinx[label]
                twin_ax = ax.twinx()
                twin_ax.set_label(twin_label)
            if label in axtwiny:
                twin_label = axtwiny[label]
                twin_ax = ax.twiny()
                twin_ax.set_label(twin_label)
            if twinx and label in twinx:
                twin_label = f"{label}_twinx"
                twin_ax = ax.twinx()
                twin_ax.set_label(twin_label)
            if twiny and label in twiny:
                twin_label = f"{label}_twiny"
                twin_ax = ax.twiny()
                twin_ax.set_label(twin_label)         
                
        if legend:
            ax = axes['legend']
            ax.axis(False)
            
        if axes_kwargs:
            axes_kwargs = core.KeyKwargs(axes_kwargs, core.IsopyArray, dict)
            for name, ax in self.subplots.items():
                if name != 'legend' or not legend:
                    kwargs = axes_kwargs.get(name)
                    if kwargs:
                        self.sca(ax)
                        self.update_axes(**kwargs)
        
        return self.subplots

    @update_figure_and_axes
    def legend(self, *subplots, labels = None, hide_axis=None, errorbars = True, newlines=None, **kwargs):
        """
        create_legend(*include_axes, labels = None, hide_axis=None, errorbars = True, newlines=None, **kwargs):
        
        Create a legend on the current subplot.
        
        The legend will also include items present in other subplots when they are passed as
        arguments to this function.

        If mutilple subplots contains items with the same label then only the item from the last subplot is included
        in the legend. Empty labels or labels beggining with ``'_'`` are not included in the legend.

        Parameters
        ----------
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
        """
        axes = self.gca()
        if 'legend' in self.subplots:
            subplots = subplots if axes in subplots else (axes,) + subplots 
            axes = self.subplots['legend']
        
        ha, la = axes.get_legend_handles_labels()
        legend = dict(zip(la, ha))

        if hide_axis is None:
            if len(subplots) > 0 and len(legend) == 0:
                hide_axis = True
            else:
                hide_axis = False

        if isinstance(labels, str): labels = [labels]
        incaxes = [axes]

        for item in subplots:
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

        # TODO ncols based on size
        # get size https://stackoverflow.com/questions/19306510/determine-matplotlib-axis-size-in-pixels
        # mpl.rcParams['legend.fontize] is by default medium, bedefault 10
        
        axes.legend(list(legend.values()), list(legend.keys()), **kwargs)
        if hide_axis:
            axes.axis(False)

    @update_figure_and_axes
    def scatter(self, x, y, xerr = None, yerr = None,
                    color= None, marker = True, line = False, regression = None, **kwargs):
        """
        scatter(x, y, xerr = None, yerr = None,
                    color= None, marker = True, line = False, regression = None, **kwargs):
                    
        Plot data points with error bars.


        Parameters
        ----------
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
        if isinstance(x, tuple) and len(x) == 2: 
            x, xerr = x
        if isinstance(y, tuple) and len(y) == 2: 
            y, yerr = y
        axes = self.gca()

        x, xerr = check_value1d('x', x, xerr)
        y, yerr = check_value1d('y', y, yerr)
        if x.size != y.size:
            raise ValueError(f'size of x ({x.size}) does not match size of y ({y.size})')

        style = {}
        #Dont use prop_cyclet directly as the default style
        #It resulted in wierd results i could not explain
        style.update(scatter_style)
        style.update(kwargs)

        prop_cycler = AxesPropCycler(axes)

        if color: style['color'] = color
        else:
            style.setdefault('color', prop_cycler.get('color', 'blue'))

        if marker is True:
            style.setdefault('marker', prop_cycler.get('marker', 'o'))
        elif marker is False: style['marker'] = ''
        elif marker: style['marker'] = marker

        if line is True: style.setdefault('linestyle', 'solid')
        elif line is False: style['linestyle'] = ''
        elif line: style['linestyle'] = line

        style.setdefault('markerfacecolor', style['color'])
        style.setdefault('zorder', 2)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'Warning: converting a masked element to nan')
            axes.errorbar(x, y, yerr, xerr, **style)
            
        add_axes_xydata(axes, x, y, xerr, yerr, label=style.get('label', None))

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
            self.regression(regression_result, color=style.get('color', None), label_equation=('label' in style))

    def xstack(self, x, xerr = None, ystart = None, *, 
               outliers=None, cval = None, pmval=None, color=None, marker=True,  line=False,
               cline = True, pmline= False, box = True, pad = 2, box_pad = None,
               spacing = -1, hide_axis=True, summary = None, subplot_grid = (1,-1), **kwargs):
        """
        xstack(x, xerr = None, ystart = None, *, outliers=None, cval = None, pmval=None, color=None, marker=True, line=False,
                    cline = True, pmline= False, box = True, pad = 2, box_pad = None,
                    spacing = -1, hide_yaxis=True, summary=None, subplots_grid = (1, -1), **kwargs)
                    
        Plot values along the x-axis at a regular y interval. Also known as Caltech or Carnegie plot.

        Can also take an array in which case it will create a grid of subplots, one for each column in
        the array.

        Parameters
        ----------
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
        summary : bool
            If ``True`` then ``summary_xstack`` is called at the end of the function. Arguments can be passed
            to this function by prefixing kwargs with ``summary_``. The function is also called if *summary* is 
            ``False`` and summary kwargs are passed.
        subplot_grid : tuple[int, int]
            The shape of the grid to be created if nessecary.
        subplot_legend : bool
            If True a legend subplot will be created.
        kwargs
            Any keyword argument accepted by matplotlib axes methods. By default kwargs are
            attributed to the data points. Prefix kwargs for the center line
            with ``cline_``, kwargs for the  plus/minus lines with ``pmline_``, kwargs for the
            box area with ``box_``, kwargs for the summary function with ``summary_``,
             kwargs for the create_subplots function with ``subplot_``
            See `here
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.errorbar.htm>`_
            for a list of keyword arguments for the data points.
            See `here
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html>`_
            for a list of keyword arguments for the center and plus/minus lines.
            See `here
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Polygon.html>`_
            for a list of keyword arguments for the box area.

        """
        kwargs = dict(outliers=outliers, cval = cval, pmval=pmval, color=color, marker=marker, line=line,
               cline = cline, pmline= pmline, box = box, pad = pad, box_pad = box_pad,
               spacing = spacing, hide_axis=hide_axis, summary = summary, subplot_grid = subplot_grid, **kwargs)
        if hasattr(x, 'items'):
            self._stack_array_('x', val=x, err=xerr, start=ystart, **kwargs)
        else:
            self._stack_('x', x, xerr, ystart, **kwargs)
    
    def ystack(self, y, yerr = None, xstart = None, *, 
               outliers=None, cval = None, pmval=None, color=None, marker=True,  line=False,
               cline = True, pmline= False, box = True, pad = 2, box_pad = None,
               spacing = 1, hide_axis=True, summary = None, subplot_grid = (-1,1), **kwargs):
        """
        ystack(y, yerr = None, ystart = None, *, outliers=None, cval = None, pmval=None, color=None, marker=True, line=False,
                    cline = True, pmline= False, box = True, pad = 2, box_pad = None,
                    spacing = -1, hide_yaxis=True, summary=None, subplots_grid = (1, -1), **kwargs)
                    
        Plot values along the x-axis at a regular y interval. Also known as Caltech or Carnegie plot.

        Can also take an array in which case it will create a grid of subplots, one for each column in
        the array.


        Parameters
        ----------
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
        summary : bool
            If ``True`` then ``summary_ystack`` is called at the end of the function. Arguments can be passed
            to this function by prefixing kwargs with ``summary_``. The function is also called if *summary* is 
            ``False`` and summary kwargs are passed.
        subplot_grid : tuple[int, int]
            The shape of the grid to be created if nessecary.
        subplot_legend : bool
            If True a legend subplot will be created.
        kwargs
            Any keyword argument accepted by matplotlib axes methods. By default kwargs are
            attributed to the data points. Prefix kwargs for the center line
            with ``cline_``, kwargs for the  plus/minus lines with ``pmline_``, kwargs for the
            box area with ``box_``, kwargs for the summary function with ``summary_``,
             kwargs for the create_subplots function with ``subplot_``
            See `here
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.errorbar.htm>`_
            for a list of keyword arguments for the data points.
            See `here
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html>`_
            for a list of keyword arguments for the center and plus/minus lines.
            See `here
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Polygon.html>`_
            for a list of keyword arguments for the box area.

        """
        kwargs = dict(outliers=outliers, cval = cval, pmval=pmval, color=color, marker=marker, line=line,
               cline = cline, pmline= pmline, box = box, pad = pad, box_pad = box_pad,
               spacing = spacing, hide_axis=hide_axis, summary = summary, subplot_grid = subplot_grid, **kwargs)
        if hasattr(y, 'items'):
            self._stack_array_('y', val=y, err=yerr, start=xstart, **kwargs)
        else:
            self._stack_('y', y, yerr, xstart, **kwargs)
 
    @update_figure_and_axes
    def _stack_(self, axis, val, err, start, *, outliers=None, cval = None, pmval=None, color=None,                
                marker=True,  line=False,
                cline = True, pmline= False, box = True, pad = 2, box_pad = None,
                spacing = 1, hide_axis=True, summary = None, 
                **kwargs):
        if isinstance(val, tuple) and len(val) == 2:
            val, err = val

        axes = self.gca()
        val, err = check_value1d(axis, val, err if err is not None else np.nan)
        outliers = check_mask1d('outliers', val, outliers, default_value=False)
        summary_kwargs = core.extract_kwargs(kwargs, 'summary')
        core.extract_kwargs(kwargs, 'subplot')

        if axis == 'x' and start is None:
            ymin, ymax = get_axes_xylim(axes, 'y')
            if spacing > 0:
                start = np.nanmax([ymax + pad, 0]) 
            else:
                start = np.nanmin([ymin - pad, 0])
        elif axis == 'y' and start is None:
            xmin, xmax = get_axes_xylim(axes, 'x')
            if spacing > 0:
                start = np.nanmax([xmax + pad, 0]) 
            else:
                start = np.nanmin([xmin - pad, 0])

        result = self._stack_calc_(val, err, start, cval, pmval, color,
                                    marker, line, cline, pmline, box, pad, box_pad, spacing, outliers, **kwargs)
        
        if axis == 'x':
            axes.get_yaxis().set_visible(not hide_axis)
            x, y, xerr, cx, cy, pmx, pmy, boxx, boxy, styles = result
            yerr=np.full(y.shape, np.nan)
        else:
            axes.get_xaxis().set_visible(not hide_axis)
            y, x, yerr, cy, cx, pmy, pmx, boxy, boxx, styles = result
            xerr = np.full(y.shape, np.nan)
            
        notoutliers = np.invert(outliers)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'Warning: converting a masked element to nan')
            axes.errorbar(x[notoutliers], y[notoutliers], yerr[notoutliers], xerr[notoutliers], **styles['main'])
            axes.errorbar(x[outliers], y[outliers], yerr[outliers], xerr[outliers], **styles['outlier'])
        add_axes_xydata(axes, x, y, xerr, yerr, isoutlier=outliers, label=styles['main'].get('label', None))
        
        if cx is not None and cy is not None:
            axes.plot(cx, cy, **styles['cline'])

        if pmx is not None and pmy is not None:
            axes.plot(pmx[0], pmy[0], **styles['pmline'])
            axes.plot(pmx[1], pmy[1], **styles['pmline'])

        if boxx is not None and boxy is not None:
            self.box(boxx, boxy, **styles['box'])
        
        if summary is True or (summary_kwargs and summary is not False):
            self.summary_xstack(**summary_kwargs)
    
    def _stack_array_(self, vaxis, /, **kwargs):
        if vaxis == 'x':
            waxis = 'y'
        elif vaxis == 'y':
            waxis = 'x'
        else:
            raise ValueError('vaxis incorrect')

        keys = kwargs['val'].keys()
        subplot_kwargs = core.extract_kwargs(kwargs, 'subplot')
        if len(self.get_axes()) == 0:
            self.create_subplots(keys, **subplot_kwargs)
        
        named_axes = self.subplots

        if kwargs['start'] is None:
            wlim = [(0,0)]
            for name, ax in named_axes.items():
                if name == '_': continue
                wlim.append(get_axes_xylim(ax, waxis))
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
                wmin = np.nanmin([wl[0] for wl in wlim])
                wmax = np.nanmax([wl[1] for wl in wlim])
            
                if kwargs['spacing'] > 0:
                    kwargs[f'start'] = np.nanmax([wmax + kwargs['pad'], 0])
                else:
                    kwargs['start'] = np.nanmin([wmin - kwargs['pad'], 0])

        kwargs = core.KeyKwargs(kwargs, core.IsopyArray, dict)
        for key in keys:
            if str(key) in named_axes:
                ax = named_axes[str(key)]
                self.sca(ax)
                self._stack_(vaxis, **kwargs.get(key))

                if waxis == 'x':
                    ax.set_ylabel(str(key))
                elif waxis == 'y':
                    ax.set_xlabel(str(key))
                
        wlim = []
        for name, ax in named_axes.items():
            if name == '_': 
                continue
            else:
                wlim.append(get_axes_xylim(ax, waxis))
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
            wmin = np.nanmin([wl[0] for wl in wlim])
            wmax = np.nanmax([wl[1] for wl in wlim])

        if not np.isnan(wmax) and not np.isnan(wmax):
            for name, ax in named_axes.items():
                if name == '_': continue
                if waxis == 'x':
                    ax.set_xlim((wmin-kwargs['pad']/2, wmax+kwargs['pad']/2))
                elif waxis == 'y':
                    ax.set_ylim((wmin - kwargs['pad']/2, wmax + kwargs['pad']/2))
 
    def _stack_calc_(self, v, verr, wstart, box_cval, box_pmval, color, marker, line, 
                     cline, pmline, box_shaded, pad, box_pad, spacing, isoutlier,
                      **kwargs):
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
        style = {}
        style.update(stack_style)
        style.update(kwargs)

        prop_cycler = AxesPropCycler(self.gca())

        if isinstance(color, tuple):
            style['color'] = color[0]
            style['outlier_color'] = color[1]
        else:
            if color:
                style['color'] = color
            elif color not in style:
                style['color'] = prop_cycler.get('color', 'blue')
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

        return v, w, verr, cv, cw, pmv, pmw, boxv, boxw, styles

    @update_figure_and_axes
    def summary_xstack(self, cval = np.nanmean, pmval = isopy.nansd2, sigfig = 2, pmunit='diff', cline='-', pmline='--',
                    color='black', axis_color = None, ignore_outliers=True, combine_ticklabels=5, remove_previous=True):
        """
        summary_xstack(cval = np.nanmean, pmval = isopy.nansd2, sigfig = 2, pmunit='diff', cline='-', pmline='--',
                    color='black', axis_color = None, ignore_outliers=True, combine_ticklabels=5, remove_previous=True)
        
        A comparison plot for data sets on a xstack.

        Calculate the center and uncertainty values for all data plotted on a xstack and
        set the *x* axis tick marks at ``[cval-pmval, cval, cval+pmval]`` and format the
        tick labels according to *pmunit*.

        Parameters
        ----------
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
            * ``"ppt"`` - The tick labels are shown as ``f'"{+pmval/cval*1000} ppt"`` and ``f'"{-pmval/cval*1000} ppt"``.
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

        """
        self._summary_stack_('x', cval, pmval, sigfig, pmunit, cline, pmline, color, axis_color, ignore_outliers, combine_ticklabels, remove_previous)
        
    @update_figure_and_axes
    def summary_ystack(self, cval = np.nanmean, pmval = isopy.nansd2, sigfig = 2, pmunit='diff', cline='-', pmline='--',
                    color='black', axis_color = None, ignore_outliers=True, combine_ticklabels=5, remove_previous=True):
        """
        summary_ystack(cval = np.nanmean, pmval = isopy.nansd2, sigfig = 2, pmunit='diff', cline='-', pmline='--',
                    color='black', axis_color = None, ignore_outliers=True, combine_ticklabels=5, remove_previous=True)
        
        A comparison plot for data sets on a ystack.

        Calculate the center and uncertainty values for all data plotted on a ystack and
        set the *y* axis tick marks at ``[cval-pmval, cval, cval+pmval]`` and format the
        tick labels according to *pmunit*.

        Parameters
        ----------
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
            * ``"ppt"`` - The tick labels are shown as ``f'"{+pmval/cval*1000} ppt"`` and ``f'"{-pmval/cval*1000} ppt"``.
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
            If ``True`` the elements on the y-axis are given the same colour as *color*.
        ignore_outliers : bool, Default = True
            If ``True`` outliers are not used to calculate *cval* and *pmline*.
        combine_ticklabels : bool, scalar, Default = 5
            If ``True`` the all three tick labels are combined into the center tick label.
            This is useful if there isnt enough space to display them all next to each other.
            If *combinde_ticklabels* is a scalar then the tick labels are only combined if the
            difference between the limits and the plus/minus tick label positions exceed this value.

        """
        self._summary_stack_('y', cval, pmval, sigfig, pmunit, cline, pmline, color, axis_color, ignore_outliers, combine_ticklabels, remove_previous)
        
    def _summary_stack_(self, axis, cval, pmval, sigfig, pmunit, cline, pmline, color, axis_color, ignore_outliers, combine_ticklabels, remove_previous):
        axes = self.gca()
        val, err = get_axes_xydata(axes, axis, isoutlier=False if ignore_outliers else None)
        if callable(cval):
            cval = cval(val)
        cval = np.asarray(cval, dtype=float)
        if callable(pmval):
            pmval = pmval(val)
        pmval = np.asarray(pmval, dtype=float)
        tickmarks = [cval - pmval, cval, cval + pmval]

        if isinstance(pmunit, tuple) and len(pmunit)==2:
            punit, munit = pmunit
        elif '_' in pmunit:
            punit, munit = pmunit.split('_', 1)
        else:
            punit = pmunit
            munit = pmunit

        min1, max1 = get_axes_xylim(axes, axis, isoutlier=None)
        min2, max2 = get_axes_xylim(axes, axis, isoutlier=False)
        
        min3 = np.nanmin([cval - pmval, min2])
        max3 = np.nanmax([cval + pmval, max2])
        diff = (max3 - min3) * 5

        max_lim = np.nanmax([max3, np.nanmin([cval + diff/2, max1])])
        min_lim = np.nanmin([min3, np.nanmax([max_lim-diff, min1])])

        diff = (max_lim-min_lim)*0.05
        max_lim += diff
        min_lim -= diff

        if combine_ticklabels is True:
            overlap = True
        elif combine_ticklabels is False:
            overlap = False
        else:
            text_rat = (max_lim-min_lim) / (pmval*2)
            if text_rat > combine_ticklabels:
                overlap = True
            else:
                overlap = False
        
        if overlap:
            ticklabels = ['',f'{label_unit(cval, -pmval, munit, sigfig)}\n{format_sigfig(cval, sigfig, pmval)}\n{label_unit(cval, pmval, punit, sigfig)}', '']
        else:
            ticklabels = [label_unit(cval, -pmval, munit, sigfig),
                        format_sigfig(cval, sigfig, pmval),
                        label_unit(cval, pmval, punit, sigfig)]
            
        if isinstance(color, tuple): 
            color = color[0]
        if isinstance(axis_color, tuple): 
            axis_color = axis_color[0]
        
        if axis == 'x':
            set_ticks = axes.set_xticks
            set_ticklabels = axes.set_xticklabels
            set_lim = axes.set_xlim
            axline = axes.axvline
        else:
            set_ticks = axes.set_yticks
            set_ticklabels = axes.set_yticklabels
            set_lim = axes.set_ylim
            axline = axes.axhline    
        
        set_ticks(tickmarks)
        set_ticklabels(ticklabels)
        set_lim(min_lim, max_lim)
        lines = []
        if cline:
            lines.append(axline(tickmarks[1], linestyle = cline, color=color))
        if pmline:
            lines.append(axline(tickmarks[0], linestyle=pmline, color=color))
            lines.append(axline(tickmarks[2], linestyle=pmline, color=color))

        if axis_color is True:
            axes.tick_params(axis=axis, colors=color)
        elif axis_color is not None:
            axes.tick_params(axis=axis, colors=axis_color)

        old_lines = get_axes_attr(axes, 'summary_stack_lines', [])
        if remove_previous:
            for line in old_lines:
                line.remove()
        else:
            lines = old_lines + lines
        
        add_axes_attr(axes, 'summary_stack_lines', lines)

    #TODO regression_result can be either a Regress result,
    @update_figure_and_axes
    def regression(self, *args, color=None, line=True, xlim = None, autoscale = False,
                        fill = True, edgeline = True, label_equation = True,  regression = None, include_outliers = False, **kwargs):
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
        """
        axes = self.gca()

        if len(args) == 1 and isinstance(args[0], toolbox.regress.LinregressResult):
            regression = args[0]
            y_eq = regression.y
            y_se_eq = regression.y_se
        elif len(args) == 1 and callable(args[0]):
            regression = None
            y_eq = args[0]
            y_se_eq = None
        elif len(args) == 1 and hasattr(args[0], 'slope') and hasattr(args[0], 'intercept'):
            regression = None
            y_eq = lambda x: x * args[0].slope + args[0].intercept
            y_se_eq = None
        elif (len(args) >= 1 and type(args[0]) is str) or len(args) == 0:
            if len(args) == 0:
                regression_type, labels = 'linear', None
            elif len(args) == 1:
                regression_type, labels = args[0], None
            else:
                regression_type, labels = args[0], args[1:]
            
            x, xerr = get_axes_xydata(axes, 'x', labels=labels, isoutlier = None if include_outliers else False)
            y, yerr = get_axes_xydata(axes, 'y', labels=labels, isoutlier = None if include_outliers else False)
            
            if regression_type == 'york' or regression_type == 'york1':
                regression = isopy.tb.yorkregress(x, y, xerr, yerr)
            elif regression_type == 'york2':
                regression = isopy.tb.yorkregress2(x, y, xerr, yerr)
            elif regression_type == 'york95':
                regression = isopy.tb.yorkregress95(x, y, xerr, yerr)
            elif regression_type == 'york99':
                regression = isopy.tb.yorkregress99(x, y, xerr, yerr)
            elif regression_type == 'linear':
                regression = isopy.tb.linregress(x, y)
            else:
                raise ValueError(f'"{regression_type}" not a valid regression')
            y_eq = regression.y
            y_se_eq = regression.y_se
            
        elif len(args) == 2:
            try:
                slope = np.float64(args[0])
                intercept = np.float64(args[1])
            except:
                raise ValueError('regression_value cannot be converted to a floats')
            else:
                regression = None
                y_eq = lambda x: x * slope + intercept
                y_se_eq = None
        elif len(args) == 1:
            try:
                slope = np.float64(args[0])
            except:
                raise ValueError('regression_value cannot be converted to a float')
            else:
                regression = None
                y_eq = lambda x: x * slope
                y_se_eq = None
        else:
            raise TypeError(f'Unable to parse input: {", ".join([str(arg) for arg in args])}')

        style = {}
        style.update(regression_style)
        style.update(kwargs)

        prop_cycler = AxesPropCycler(axes)

        if color:
            style['color'] = color
        else:
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
            if regression is not None:
                style['label'] = f'{style.get("label", "")} {regression.label(sigfig)}'.strip()
            else:
                var = y_eq(xlim[1]) - y_eq(xlim[0])
                label_intercept = y_eq(0)
                label_slope = y_eq(1) - label_intercept

                if not np.isnan(label_intercept):
                    label_intercept = format_sigfig(label_intercept, sigfig, var)
                if not np.isnan(label_slope):
                    label_slope = format_sigfig(label_slope, sigfig, var)

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
            self.polygon(yerr_coordinates, autoscale=autoscale, **fill_style)

    @update_figure_and_axes
    def plot_contours(self, x, y, z, zmin=None, zmax=None, levels=100, colors='jet',
                    colorbar=None, label = False, label_levels = None, filled = True, **kwargs):
        """
        Create a contour plot on *axes* from a data grid.

        Parameters
        ----------
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
        """
        axes = self.gca()

        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)

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
            raise ValueError(f'shape of z {z.shape} does not match shape of y and x ({y.size}, {x.size})')

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
    
    @update_figure_and_axes
    def plot_spider(self, y, yerr = None, x = None, constants = None, xscatter  = None, color = None, marker = True, 
                    line = True, **kwargs):
        """
        Plot data as a spider diagram.

        Parameters
        ----------
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
        """
        axes = self.gca()
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
        label = style.get('label', None)

        prop_cycler = AxesPropCycler(axes)

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

        add_axes_xydata(axes, x, y, None, yerr=yerr, label=label)
    
    @update_figure_and_axes
    def box(self, x = None, y = None, color = None, autoscale = True, **style_kwargs):
        """
        box(x = None, y = None, color = None, autoscale = True, **style_kwargs)
        
        Plot a semi-transparent box on the current subplot.

        Parameters
        ----------
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
        axes = self.gca()
        if x is not None:
            x, _ = check_value1d('x', x)
            if x.size != 2:
                raise ValueError(f'size of x ({x.size}) must be 2')
        else:
            x = np.array(axes.get_xlim()).flatten()

        if y is not None:
            y, _ = check_value1d('y', y)
            if y.size != 2:
                raise ValueError(f'size of y ({y.size}) must be 2')
        else:
            y = np.array(axes.get_ylim()).flatten()

        xcoord = [x[0], x[1], x[1], x[0]]
        ycoord = [y[0], y[0], y[1], y[1]]

        style = {}
        style.update(box_style)
        style.update(**style_kwargs)

        if isinstance(color, tuple): style['color'] = color[0]
        elif color: style['color'] = color
        else: style.setdefault('color', 'black')

        self.polygon(xcoord, ycoord, autoscale=autoscale, **style)

    @update_figure_and_axes
    def polygon(self, x, y=None, color = None, autoscale = True, **style_kwargs):
        """
        polygon(x, y=None, color = None, autoscale = True, **style_kwargs)
        
        Plot a semi-transparent polygon on the current axes.

        Parameters
        ----------
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
        axes = self.gca()

        if y is not None:
            x, _ = check_value1d('x', x)
            y, _ = check_value1d('y', y)
            if x.size != y.size:
                raise ValueError(f'size of x ({x.size}) and y ({y.size}) are not the same')
            coordinates = np.append(x, y).reshape(2, -1).transpose()
        else:
            x = np.array(x)
            if x.ndim != 2 or x.shape[-1] != 2:
                raise ValueError(f'shape of x ({x.shape}) invalid. Must be (n,2)')
            coordinates = x

        style = {}
        style.update(polygon_style)
        style.update(**style_kwargs)

        if isinstance(color, tuple): style['color'] = color[0]
        elif color: style['color'] = color
        else: style.setdefault('color', 'black')

        add_patch(axes, coordinates, autoscale, **style)

    @update_figure_and_axes
    def text(self, xpos, ypos, text, rotation = None, fontsize = None, posabs = True, **kwargs):
        """
        text(xpos, ypos, text, rotation = None, fontsize = None, posabs = True, **kwargs)
        
        Add text to the current axes.

        Parameters
        ----------
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
        axes = self.gca()

        if rotation:
            kwargs['rotation'] = rotation

        if fontsize:
            kwargs['fontsize'] = fontsize

        if posabs is False:
            kwargs['transform'] = axes.transAxes

        axes.text(xpos, ypos, text, **kwargs)

def check_value1d(name, val, err = None):
    try:
        val = np.asarray(val).flatten()
    except Exception as error:
        raise TypeError(f'invalid {name}: {error}') from error
    
    if err is None:
        err = np.full(val.shape, np.nan)
    else:
        try:
            err = np.asarray(err)
        except Exception as error:
            raise TypeError(f'invalid {name}err: {error}') from error

        if err.ndim == 0:
            err = np.full(val.shape, err)
        else:
            err = err.flatten()
            if err.size != val.size:
                raise ValueError(f'size of {name}err ({err.size}) does not match size of {name} ({val.size})')

    return val, err

def check_mask1d(name, val, mask, default_value=None):
    if mask is None and default_value is not None:
        mask = default_value

    if mask is not None:
        try:
            mask = np.asarray(mask, dtype=bool)
        except Exception as error:
            raise TypeError(f'invalid {name}: {error}') from error

        if mask.ndim == 0:
            mask = np.full(val.shape, mask, dtype=bool)
        else:
            mask = mask.flatten()
            if mask.size != val.size:
                raise ValueError(f'size of {name} ({mask.size}) does not match size of value ({val.size})')
    return mask

def get_axes_attrs(axes):
    if not hasattr(axes, '_isopy_attrs'):
        axes._isopy_attrs = dict()
        
        clear_func = axes._AxesBase__clear
        def clear(*args, **kwargs):
            axes._isopy_attrs.clear()
            return clear_func(*args, **kwargs)
        axes._AxesBase__clear = clear

    return axes._isopy_attrs

def get_axes_attr(axes, name, default_value=None):
    if not hasattr(axes, '_isopy_attrs'):
        return default_value
    else:
        return axes._isopy_attrs.get(name, default_value)

def add_axes_attr(axes, name, value):
    get_axes_attrs(axes)[name] = value

def add_axes_xydata(axes, x, y, xerr=None, yerr=None, isoutlier=None, label = None):
    attrs = get_axes_attrs(axes)
    
    if '_xydata' not in attrs:
        attrs['_xydata'] = dict(x=[], y=[], xerr=[], yerr=[], isoutlier=[], isnotoutlier=[], label=[])
    
    data = attrs['_xydata']
    
    if xerr is None:
        xerr = np.full(np.nan, x.shape)
    if yerr is None:
        yerr = np.full(np.nan, y.shape)
    
    data['x'].append(x)
    data['xerr'].append(xerr)
    data['y'].append(y)
    data['yerr'].append(yerr)
    if isoutlier is None:
        isoutlier = check_mask1d('isoutlier', x, None, default_value=False)
    
    data['isoutlier'].append(isoutlier)
    data['isnotoutlier'].append(np.invert(isoutlier))
    data['label'].append(label)

def get_axes_xydata(axes, axis, labels = None, isoutlier = None):
    data = get_axes_attr(axes, '_xydata', None)
    
    if data is None:
        return np.array([np.nan]), np.array([np.nan])
    else:
        if type(labels) is str:
            labels = [labels]
        
        if labels is None:
            val, err, mask = data[axis], data[f'{axis}err'], data['isoutlier']
        else:
            val, err, mask = [], [], []
            for label in labels:
                if label in data['label']:
                    i = data['label'].index(label)
                    val.append(data[axis][i])
                    err.append(data[f'{axis}err'][i])
                    mask.append(data['isoutlier'][i])
                else:
                    raise ValueError(f'label {label} not found in plot')
                    
        if len(val) == 0:
            raise ValueError('')
        val = np.concatenate(val)
        err = np.concatenate(err)
        mask = np.concatenate(mask)
            
        if isoutlier is None:
            return val, err
        elif isoutlier is True:
            val, err = val[mask], err[mask]
        else:
            mask = np.invert(mask)
            val, err = val[mask], err[mask]
        
        if val.size == 0:
            return np.array([np.nan]), np.array([np.nan])
        else:
            return val, err

def get_axes_xylim(axes, axis, isoutlier = None):
    val, err = get_axes_xydata(axes, axis, isoutlier=isoutlier)
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
        
        vmin = np.nanmin([np.nanmin(val), np.nanmin(val-err)])
        vmax = np.nanmax([np.nanmax(val), np.nanmax(val+err)])
    
    return vmin, vmax
 
def add_patch(axes, coordinates, autoscale = True, **style):
    upl_func = None
    if not autoscale and hasattr(axes, '_update_patch_limits'):
        upl_func = axes._update_patch_limits
        axes._update_patch_limits = lambda *args, **kwargs: None

    try:
        axes.add_patch(mplPolygon(coordinates, **style))
    finally:
        if upl_func:
            axes._update_patch_limits = upl_func

def format_sigfig(value, sigfig, variation = None, pm=False):
    if variation is None: 
        variation = value
    if sigfig is not None:
        if variation == 0 or np.isnan(variation):
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

def label_unit(refval, val, unit, sigfig):
    if unit[-1].isdigit():
        sigfig = int(unit[-1])
        unit = unit[:-1]

    if unit == 'diff':
        return f'\u0394 {format_sigfig(val, sigfig, pm=True)}'
    if unit == 'abs':
        return format_sigfig(refval+val, sigfig, val)
    if unit == 'percent' or unit == '%':
        return f'{format_sigfig(val/refval * 100, sigfig, pm=True)} %'
    if unit == 'ppt':
        return f'{format_sigfig(val/refval * 1000, sigfig, pm=True)} ppt'
    if unit == 'ppm':
        return f'{format_sigfig(val/refval * 1E6, sigfig, pm=True)} ppm'
    if unit == 'ppb':
        return f'{format_sigfig(val/refval * 1E9, sigfig, pm=True)} ppb'

def update_style():
    prop_cycler = mpl.rcParams['axes.prop_cycle'].by_key()
    colors = prop_cycler.get('color', ['#377eb8', '#e41a1c', '#4daf4a', '#ff7f00', '#984ea3', '#a65628', '#ffff33', '#f781bf', '#999999'])
    markers = prop_cycler.get('marker', ['o', 's', '^', 'D', 'P', 'X', 'p'])

    mpl.rcParams['axes.prop_cycle'] = cycler.cycler(color = colors * len(markers), marker = markers * len(colors))

update_style()