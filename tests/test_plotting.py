import pytest
import isopy
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.figsize'] = (6, 4)
plt.rcParams['figure.dpi'] = 72

#These are the examples included in the docstrings

@pytest.mark.mpl_image_compare(tolerance=5, style='default')
def test_colors():
    plt.close()
    colors = isopy.tb.Colors()
    for i in range(len(colors)):
        isopy.tb.plot_scatter(plt, [1, 2], [i, i], color=colors.current, markersize=20, line=True)
        colors.next()
    return plt

@pytest.mark.mpl_image_compare(tolerance=5, style='default')
def test_markers():
    plt.close()
    markers = isopy.tb.Markers()
    for i in range(len(markers)):
        isopy.tb.plot_scatter(plt, [1, 2], [i, i], marker=markers.current, markersize=20, line=True)
        markers.next()
    return plt

@pytest.mark.mpl_image_compare(tolerance=5, style='default')
def test_colorpairs():
    plt.close()
    colorpairs = isopy.tb.ColorPairs()
    for i in range(len(colorpairs)):
        isopy.tb.plot_scatter(plt, [1, 1.5], [i, i], color=colorpairs.current[0], markersize=20,
                              line=True)
        isopy.tb.plot_scatter(plt, [2, 2.5], [i, i], color=colorpairs.current[1], markersize=20,
                              line=True)
        colorpairs.next()
    return plt

@pytest.mark.mpl_image_compare(tolerance=5, style='default')
def test_update_figure1():
    plt.close()
    isopy.tb.update_figure(plt, size=(10, 2), facecolor='orange')
    isopy.tb.plot_scatter(plt, np.arange(10), np.arange(10))
    return plt

@pytest.mark.mpl_image_compare(tolerance=5, style='default')
def test_update_figure2():
    plt.close()
    isopy.tb.plot_scatter(plt, np.arange(10), np.arange(10),
                          figure_size=(10, 2), figure_facecolor='orange')
    return plt

@pytest.mark.mpl_image_compare(tolerance=5, style='default')
def test_update_axes1():
    plt.close()
    isopy.tb.update_axes(plt, xlabel='This it the x-axis', ylabel='This is the y-axis')
    isopy.tb.plot_scatter(plt, np.arange(10), np.arange(10))
    return plt

@pytest.mark.mpl_image_compare(tolerance=5, style='default')
def test_update_axes2():
    plt.close()
    isopy.tb.plot_scatter(plt, np.arange(10), np.arange(10), axes_xlabel='This it the x-axis',
                          axes_ylabel='This is the y-axis')

    return plt

@pytest.mark.mpl_image_compare(tolerance=5, style='default')
def test_create_subplots1():
    plt.close()
    subplots = [['one', 'two', 'three', 'four']]
    axes = isopy.tb.create_subplots(plt, subplots)
    for name, ax in axes.items(): ax.set_title(name)
    return plt

@pytest.mark.mpl_image_compare(tolerance=5, style='default')
def test_create_subplots2():
    plt.close()
    subplots = ['one', 'two', 'three', 'four', 'five', 'six']
    axes = isopy.tb.create_subplots(plt, subplots, (-1, 2))
    for name, ax in axes.items(): ax.set_title(name)
    return plt

@pytest.mark.mpl_image_compare(tolerance=5, style='default')
def test_create_subplots3():
    plt.close()
    subplots = [['one', 'two'], ['three', 'two'], ['four', 'four']]
    axes = isopy.tb.create_subplots(plt, subplots)
    for name, ax in axes.items(): ax.set_title(name)
    return plt

@pytest.mark.mpl_image_compare(tolerance=5, style='default')
def test_create_legend1():
    plt.close()
    data = isopy.random(100, keys='ru pd cd'.split(), seed=46)
    axes = isopy.tb.create_subplots(plt, [['left', 'right']], figure_width=8)
    isopy.tb.plot_scatter(axes['left'], data['pd'], data['ru'], label='ru/pd', color='red')
    isopy.tb.plot_scatter(axes['right'], data['pd'], data['cd'], label='cd/pd', color='blue')
    isopy.tb.create_legend(axes['right'], axes['left'])
    return plt

@pytest.mark.mpl_image_compare(tolerance=5, style='default')
def test_create_legend2():
    plt.close()
    data = isopy.random(100, keys='ru pd cd'.split(), seed=46)
    axes = isopy.tb.create_subplots(plt, [['left', 'right', 'legend']],
                                    figure_width=9, gridspec_width_ratios=[4, 4, 1])
    isopy.tb.plot_scatter(axes['left'], data['pd'], data['ru'], label='ru/pd', color='red')
    isopy.tb.plot_scatter(axes['right'], data['pd'], data['cd'], label='cd/pd', color='blue')
    isopy.tb.create_legend(axes['legend'], axes, hide_axis=True)
    return plt

@pytest.mark.mpl_image_compare(tolerance=5, style='default')
def test_plot_scatter1():
    plt.close()
    x = isopy.random(20, seed=46)
    y = x * 3 + isopy.random(20, seed=47)
    isopy.tb.plot_scatter(plt, x, y)
    return plt

@pytest.mark.mpl_image_compare(tolerance=5, style='default')
def test_plot_scatter2():
    plt.close()
    x = isopy.random(20, seed=46)
    y = x * 3 + isopy.random(20, seed=47)
    xerr = 0.2
    yerr = isopy.random(20, seed=48)
    isopy.tb.plot_scatter(plt, x, y, xerr, yerr, regression='york1', color='red', marker='s')
    return plt

@pytest.mark.mpl_image_compare(tolerance=5, style='default')
def test_plot_regression1():
    plt.close()
    x = isopy.random(20, seed=46)
    y = x * 3 + isopy.random(20, seed=47)
    regression = isopy.tb.regression_linear(x, y)
    isopy.tb.plot_scatter(plt, x, y)
    isopy.tb.plot_regression(plt, regression)
    return plt

@pytest.mark.mpl_image_compare(tolerance=5, style='default')
def test_plot_regression2():
    plt.close()
    x = isopy.random(20, seed=46)
    y = x * 3 + isopy.random(20, seed=47)
    regression = lambda x: 2 * x + x ** 2  # Any callable that takes x and return y is a valid
    isopy.tb.plot_scatter(plt, x, y, color='red')
    isopy.tb.plot_regression(plt, regression, color='red', xlim=(-1, 1))
    return plt

@pytest.mark.mpl_image_compare(tolerance=5, style='default')
def test_plot_regression3():
    plt.close()
    x = isopy.random(20, seed=46)
    y = x * 3 + isopy.random(20, seed=47)
    xerr = 0.2
    yerr = isopy.random(20, seed=48)
    regression = isopy.tb.regression_york1(x, y, xerr, yerr)
    isopy.tb.plot_scatter(plt, x, y, xerr, yerr)
    isopy.tb.plot_regression(plt, regression)
    return plt

@pytest.mark.mpl_image_compare(tolerance=5, style='default')
def test_plot_regression4():
    plt.close()
    x = isopy.random(20, seed=46)
    y = x * 3 + isopy.random(20, seed=47)
    xerr = 0.2
    yerr = isopy.random(20, seed=48)
    regression = isopy.tb.regression_york1(x, y, xerr, yerr)
    isopy.tb.plot_scatter(plt, x, y, xerr, yerr, color='red')
    isopy.tb.plot_regression(plt, regression, color='red', line='dashed', edgeline=False)
    return plt

@pytest.mark.mpl_image_compare(tolerance=5, style='default')
def test_plot_spider1():
    plt.close()
    array = isopy.refval.isotope.fraction.asarray(element_symbol='pd')
    isopy.tb.plot_spider(plt, array)  # Will plot the fraction of each Pd isotope
    return plt

@pytest.mark.mpl_image_compare(tolerance=5, style='default')
def test_plot_spider2():
    plt.close()
    subplots = isopy.tb.create_subplots(plt, [['left', 'right']])
    array = isopy.refval.isotope.fraction.asarray(element_symbol='pd').ratio('105pd')
    isopy.tb.plot_spider(subplots['left'], array)  # The numerator mass numbers are used as x
    isopy.tb.plot_spider(subplots['right'], array,
                         constants={105: 1})  # Adds a 1 for the denominator
    return plt

@pytest.mark.mpl_image_compare(tolerance=5, style='default')
def test_plot_spider3():
    plt.close()
    subplots = isopy.tb.create_subplots(plt, [['left', 'right']], figwidth=8)
    values = {100: [1, 1, -1], 101.5: [0, 0, 0], 103: [-1, 1, -1]}  # keys can be floats
    isopy.array(values)
    isopy.tb.plot_spider(subplots['left'], values)  # Impossible to tell the rows apart
    isopy.tb.plot_spider(subplots['right'], values, xscatter=0.15)  # Much clearer
    return plt

@pytest.mark.mpl_image_compare(tolerance=5, style='default')
def test_plot_vstack1():
    plt.close()
    array1 = isopy.random(100, -0.5, seed=46)
    array2 = isopy.random(100, 0.5, seed=47)
    isopy.tb.plot_vstack(plt, array1)
    isopy.tb.plot_vstack(plt, array2, cval=np.mean, pmval=isopy.sd2)
    return plt

@pytest.mark.mpl_image_compare(tolerance=5, style='default')
def test_plot_vstack2():
    plt.close()
    keys = isopy.keylist('pd105', 'ru101', 'cd111')
    array1 = isopy.random(100, -0.5, keys, seed=46)
    array2 = isopy.random(100, 0.5, keys, seed=47)
    isopy.tb.create_subplots(plt, keys.sorted(), (1, -1))
    isopy.tb.plot_vstack(plt, array1)
    isopy.tb.plot_vstack(plt, array2, cval=np.mean, pmval=isopy.sd2)
    return plt

@pytest.mark.mpl_image_compare(tolerance=5, style='default')
def test_plot_vstack3():
    plt.close()
    keys = isopy.keylist('pd105', 'ru101', 'cd111')
    array = isopy.random(100, -0.5, keys, seed=46)
    mean = np.mean(array);
    sd = isopy.sd(array)
    outliers = isopy.tb.find_outliers(array, mean, sd)
    isopy.tb.create_subplots(plt, keys.sorted(), (1, -1))
    isopy.tb.plot_vstack(plt, array, cval=mean, pmval=sd, outliers=outliers, color=('red', 'pink'))
    return plt

@pytest.mark.mpl_image_compare(tolerance=5, style='default')
def test_plot_hstack1():
    plt.close()
    array1 = isopy.random(100, -0.5, seed=46)
    array2 = isopy.random(100, 0.5, seed=47)
    isopy.tb.plot_hstack(plt, array1)
    isopy.tb.plot_hstack(plt, array2, cval=np.mean, pmval=isopy.sd2)
    return plt

@pytest.mark.mpl_image_compare(tolerance=5, style='default')
def test_plot_hstack2():
    plt.close()
    keys = isopy.keylist('pd105', 'ru101', 'cd111')
    array1 = isopy.random(100, -0.5, keys, seed=46)
    array2 = isopy.random(100, 0.5, keys, seed=47)
    isopy.tb.create_subplots(plt, keys.sorted(), (-1, 1))
    isopy.tb.plot_hstack(plt, array1)
    isopy.tb.plot_hstack(plt, array2, cval=np.mean, pmval=isopy.sd2)
    return plt

@pytest.mark.mpl_image_compare(tolerance=5, style='default')
def test_plot_hstack3():
    plt.close()
    keys = isopy.keylist('pd105', 'ru101', 'cd111')
    array = isopy.random(100, -0.5, keys, seed=46)
    mean = np.mean(array);
    sd = isopy.sd(array)
    outliers = isopy.tb.find_outliers(array, mean, sd)
    isopy.tb.create_subplots(plt, keys.sorted(), (-1, 1))
    isopy.tb.plot_hstack(plt, array, cval=mean, pmval=sd, outliers=outliers, color=('red', 'pink'))
    return plt

@pytest.mark.mpl_image_compare(tolerance=5, style='default')
def test_plot_hcompare1():
    plt.close()
    array1 = isopy.random(100, 0.9, seed=46)
    array2 = isopy.random(100, 1.1, seed=47)
    isopy.tb.plot_hstack(plt, array1, cval=np.mean, pmval=isopy.sd2)
    isopy.tb.plot_hstack(plt, array2, cval=np.mean, pmval=isopy.sd2)
    isopy.tb.plot_hcompare(plt)
    return plt

@pytest.mark.mpl_image_compare(tolerance=5, style='default')
def test_plot_hcompare2():
    plt.close()
    array1 = isopy.random(100, 0.9, seed=46)
    array2 = isopy.random(100, 1.1, seed=47)
    isopy.tb.plot_hstack(plt, array1, cval=np.mean, pmval=isopy.sd2)
    isopy.tb.plot_hstack(plt, array2, cval=np.mean, pmval=isopy.sd2)
    isopy.tb.plot_hcompare(plt, pmval=isopy.sd, sigfig=3)
    return plt

@pytest.mark.mpl_image_compare(tolerance=5, style='default')
def test_plot_hcompare3():
    plt.close()
    pmunits = ['diff', 'abs', '%', 'ppt', 'ppm', 'ppb']
    subplots = isopy.tb.create_subplots(plt, pmunits, (-1, 1), figure_height=8)
    array1 = isopy.random(100, 0.9, seed=46)
    array2 = isopy.random(100, 1.1, seed=47)
    for unit, axes in subplots.items():
        isopy.tb.plot_hstack(axes, array1, cval=np.mean, pmval=isopy.sd2)
        isopy.tb.plot_hstack(axes, array2, cval=np.mean, pmval=isopy.sd2)
        isopy.tb.plot_hcompare(axes, pmval=isopy.sd, pmunit=unit, combine_ticklabels=True)
        axes.set_ylabel(f'pmunit="{unit}"')
    return plt

@pytest.mark.mpl_image_compare(tolerance=5, style='default')
def test_plot_hcompare4():
    plt.close()
    keys = isopy.keylist('pd105', 'ru101', 'cd111')
    array1 = isopy.random(100, 0.9, keys, seed=46)
    array2 = isopy.random(100, 1.1, keys, seed=47)
    isopy.tb.plot_hstack(plt, array1, cval=np.mean, pmval=isopy.sd2)
    isopy.tb.plot_hstack(plt, array2, cval=np.mean, pmval=isopy.sd2)
    isopy.tb.plot_hcompare(plt.gcf(), combine_ticklabels=True)
    return plt

@pytest.mark.mpl_image_compare(tolerance=5, style='default')
def test_plot_vcompare1():
    plt.close()
    array1 = isopy.random(100, -0.5, seed=46)
    array2 = isopy.random(100, 0.5, seed=47)
    isopy.tb.plot_vstack(plt, array1, cval=np.mean, pmval=isopy.sd2)
    isopy.tb.plot_vstack(plt, array2, cval=np.mean, pmval=isopy.sd2)
    isopy.tb.plot_vcompare(plt)
    return plt

@pytest.mark.mpl_image_compare(tolerance=5, style='default')
def test_plot_vcompare2():
    plt.close()
    array1 = isopy.random(100, -0.5, seed=46)
    array2 = isopy.random(100, 0.5, seed=47)
    isopy.tb.plot_vstack(plt, array1, cval=np.mean, pmval=isopy.sd2)
    isopy.tb.plot_vstack(plt, array2, cval=np.mean, pmval=isopy.sd2)
    isopy.tb.plot_vcompare(plt, pmval=isopy.sd, sigfig=3)
    return plt

@pytest.mark.mpl_image_compare(tolerance=5, style='default')
def test_plot_vcompare3():
    plt.close()
    pmunits = ['diff', 'abs', '%', 'ppt', 'ppm', 'ppb']
    subplots = isopy.tb.create_subplots(plt, pmunits, (1, -1), figure_width=8)
    array1 = isopy.random(100, 0.9, seed=46)
    array2 = isopy.random(100, 1.1, seed=47)
    for unit, axes in subplots.items():
        isopy.tb.plot_vstack(axes, array1, cval=np.mean, pmval=isopy.sd2)
        isopy.tb.plot_vstack(axes, array2, cval=np.mean, pmval=isopy.sd2)
        isopy.tb.plot_vcompare(axes, pmval=isopy.sd, pmunit=unit, combine_ticklabels=True)
        axes.set_xlabel(f'pmunit="{unit}"')
    return plt

@pytest.mark.mpl_image_compare(tolerance=5, style='default')
def test_plot_vcompare4():
    plt.close()
    keys = isopy.keylist('pd105', 'ru101', 'cd111')
    array1 = isopy.random(100, 0.9, keys, seed=46)
    array2 = isopy.random(100, 1.1, keys, seed=47)
    isopy.tb.plot_vstack(plt, array1)
    isopy.tb.plot_vstack(plt, array2, cval=np.mean, pmval=isopy.sd2)
    isopy.tb.plot_vcompare(plt.gcf(), combine_ticklabels=True)
    return plt

@pytest.mark.mpl_image_compare(tolerance=5, style='default')
def test_plot_contours1():
    plt.close()
    x = np.arange(10)
    y = np.arange(10, 101, 10)
    z = np.arange(100).reshape((10, 10))
    isopy.tb.plot_contours(plt, x, y, z)
    return plt

@pytest.mark.mpl_image_compare(tolerance=5, style='default')
def test_plot_contours2():
    plt.close()
    element = 'fe'
    spike1 = isopy.array(fe54=0, fe56=0, fe57=1, fe58=0)
    spike2 = isopy.array(fe54=0, fe56=0, fe57=0, fe58=1)
    axes_labels = dict(axes_xlabel='Proportion of double spike in double spike-sample mix',
                       axes_ylabel='Proportion of spike1 in double spike',
                       axes_title='Uncertainty in α (1SD)')
    dsgrid = isopy.tb.ds_grid(element, spike1, spike2)
    isopy.tb.plot_contours(plt, *dsgrid.xyz(), zmin=0, zmax=0.01, **axes_labels)
    return plt