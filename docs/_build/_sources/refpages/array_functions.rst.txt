Array functions
===============
.. rubric:: Numpy functions

Isopy arrays are compatible with most numpy functions:

>>> array = isopy.tb.make_ms_array('pd')
>>> np.log(array)
(row) , 102Pd    , 104Pd    , 105Pd    , 106Pd    , 108Pd    , 110Pd
None  , -4.58537 , -2.19463 , -1.49924 , -1.29719 , -1.32954 , -2.14387

For numpy functions that take an ``axis`` argument the function is by default applied to
each column:

>>> array = isopy.tb.make_ms_beams('pd', integrations=10)
>>> np.mean(array)
(row) , 102Pd   , 104Pd   , 105Pd   , 106Pd   , 108Pd   , 110Pd
None  , 0.37322 , 4.07611 , 8.17043 , 9.99998 , 9.68167 , 4.28836

However, you can specify the axis to apply the function over a different axis:

>>> np.mean(array, axis=None) #Performes the function over the entire array
6.098293273315375

>>> np.mean(array, axis=0) #Performes the function on each column. Same as omitting the axis argument
(row) , 102Pd   , 104Pd   , 105Pd   , 106Pd   , 108Pd   , 110Pd
None  , 0.37322 , 4.07611 , 8.17043 , 9.99998 , 9.68167 , 4.28836

>>> np.mean(array, axis=1) #Performes the function on each row in the array
array([6.09832176, 6.098316  , 6.09826115, 6.09826404, 6.09823369,
       6.09835421, 6.09827056, 6.09831986, 6.09832989, 6.09826158])

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
    count_finite


The following isopy array functions require that at least one isopy array as input.

.. currentmodule:: isopy
.. autosummary::
    :toctree: array_functions

    argmaxkey
    argminkey
    add
    subtract
    multiply
    divide
    power
