Toolbox
*******

The isopy toolbox contains an number of useful functions for processing and evaluating
geochemical data. You can access the toolbox through the ``isopy.tb`` namespace.

Isotope Data Processing
=======================
The following functions are useful for processing isotopic data.

.. currentmodule:: isopy.tb
.. autosummary::
    :toctree: tb_reduce_iso

    normalise_data
    denormalise_data
    find_outliers
    johnson_nyquist_noise
    make_ms_array
    make_ms_beams
    make_ms_sample
    mass_independent_correction
    calculate_mass_fractionation_factor
    remove_mass_fractionation
    add_mass_fractionation
    remove_isobaric_interferences
    ds_inversion
    ds_correction
    ds_grid

Regressions
=============
Below are functions for calculating regressions

.. currentmodule:: isopy.tb
.. autosummary::
    :toctree: tb_regress

    regression_york1
    regression_york2
    regression_linear


Plotting
=========
Below are functions for plotting geochemical data using matplotlib

.. currentmodule:: isopy.tb
.. autosummary::
    :toctree: tb_plotting

    update_figure
    create_subplots
    plot_scatter
    plot_regression
    plot_spider
    plot_vstack
    plot_hstack
    plot_vcompare
    plot_hcompare

    :template: cycler_template.rst

    Colors
    ColorPairs
    Markers






