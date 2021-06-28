Array functions
===============
.. rubric:: Isopy functions

Isopy has a number of its own array functions that behave like numpy array functions.

The following array functions will accept both numpy like arrays and isopy arrays as input.

.. currentmodule:: isopy
.. autosummary::
    :toctree: array_functions

    sd
    nansd
    se
    nanse
    mad
    nanmad
    nancount


The following isopy array functions require that at least one isopy array as input.

.. currentmodule:: isopy
.. autosummary::
    :toctree: array_functions

    keymax
    keymin
    add
    subtract
    multiply
    divide
    power
    arrayfunc

.. rubric:: Numpy functions

Isopy arrays are compatible with many numpy functions. Numpy functions that have not been tested with
isopy arrays will raise a warning the first time they are used.

The following function will return a list of the name of each numpy function that has been tested with
isopy arrays.

.. autosummary::
    :toctree: array_functions

    allowed_numpy_functions