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

    rDelta
    inverse_rDelta
    find_outliers
    upper_limit
    lower_limit
    johnson_nyquist_noise
    make_ms_array
    make_ms_beams
    make_ms_sample
    internal_normalisation
    calculate_mass_fractionation_factor
    remove_mass_fractionation
    add_mass_fractionation
    find_isobaric_interferences
    remove_isobaric_interferences
    ds_correction
    ds_grid
    ds_Delta
    ds_Delta_prime

Regressions
=============
Below are functions for calculating regressions

.. currentmodule:: isopy.tb
.. autosummary::
    :toctree: tb_regress

    linregress
    yorkregress


Plotting
=========
Below are functions for plotting geochemical data using matplotlib

.. currentmodule:: isopy.tb
.. autosummary::
    :toctree: tb_plotting

    update_figure
    update_axes
    create_subplots
    create_legend
    plot_scatter
    plot_regression
    plot_spider
    plot_vstack
    plot_hstack
    plot_vcompare
    plot_hcompare
    plot_contours

    :template: cycler_template.rst

    Colors
    Markers






