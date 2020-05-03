Toolbox
*******

Here is a list of all the toolboxes that are currently avaliable in isopy. The most common functions are avaliable in
the ``isopypy.tb`` namespace. More specific functions/classes can be accessed via ``isopy.toolbox.<toolbox>``.

Note that using ``from isopy.toolbox.<toolbox> import *`` will only import the functions
present in ``isopy.tb``. Use ``from isopy.toolbox import <toolbox>`` instead to import a specific toolbox.

General Data Processing
=======================
This toolbox contains functions for general data processing.

Avaliable in ``isopy.tb`` and ``isopy.toolbox.general``

.. currentmodule:: isopy.tb
.. autosummary::
    :toctree: tb_general

    normalise_data
    denomralise_data
    find_outliers
    find_outliers_mad

Isotope Data Processing
=======================
This toolbox contains functions for processing of isotope data.

Avaliable in ``isopy.tb`` and ``isopy.toolbox.isotope``

.. currentmodule:: isopy.tb
.. autosummary::
    :toctree: tb_isotope

    make_sample
    mass_independent_correction
    calculate_mass_fractionation_factor
    remove_mass_fractionation
    add_mass_fractionation
    remove_isobaric_interferences
    add_isobaric_interferences

Miscellaneous
=============
This toolbox contains miscellaneous functions and classes.

Avaliable in ``isopy.tb`` and ``isopy.toolbox.misc``

.. currentmodule:: isopy.tb
.. autosummary::
    :toctree: tb_misc

    york_regression
    plot_york_regression
    johnson_nyquist_noise

Only avaliable in ``isopy.toolbox.misc``

.. currentmodule:: isopy.toolbox.misc
.. autosummary::
    :toctree: tb_misc

    YorkRegression


Numpy like functions
====================
The functions in this toolbox behave like numpy functions.

Avaliable in ``isopy.tb`` and ``isopy.toolbox.np_like``

.. currentmodule:: isopy.toolbox.np_func
.. autosummary::
    :toctree: tb_np_func

    sd
    se
    mad
    nansd
    nanse
    nanmad





