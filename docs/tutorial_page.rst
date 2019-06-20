Tutorial
*********************




Creating arrays
---------------
There are several possible ways to create arrays.

Empty arrays are created by giving the size and a set of keys

>>> array = isopy.IsotopeArray(size = 5, keys = ['104pd', '105pd', '106pd'])
>>> print(array)
104Pd, 105Pd, 106Pd
0.0, 0.0, 0.0
0.0, 0.0, 0.0
0.0, 0.0, 0.0
0.0, 0.0, 0.0
0.0, 0.0, 0.0

There are a number of ways in which arrays can be created from existing data.

>>> datadict = {'104pd': [0.11004982, 0.11172253, 0.1138327 , 0.11066279, 0.11123413],
                '105pd': [0.22524588, 0.2254826 , 0.22228121, 0.22352313, 0.2216239 ],
                '106Pd': [0.27218295, 0.2687675 , 0.27360541, 0.27352446, 0.27843918]}
>>> array = IsotopeArray(datadict)
>>> print(array)




There are several ways to create an array. To create an empty array you just need to supply an integer of the number of
records desired and a list of keys.
>>> a = IsotopeArray(10,['105pd','106pd','108pd'])
>>> a
whatever this looks like
>>> a.keys()
['105Pd', '106Pd', '108Pd']
>>> len(a)
10

The example above creates an IsotopeArray with 10 empty records for each key given. The keys are automatically
formatted into the correct format. See IsotopeSting and IsotopeList for more information on this.

Finally, if we already have a structured numpy array with keys that can be formatted to an IsotopeSting we can simply
change the view of the array to a IsotopeArray


>>> a = numpy.array([[1,2,3],[4,5,6]], dtype = [('105pd', 'f8'), ('106Pd', 'f8'), ('108pd', 'f8'])
>>> IsotopeArray(a)
ValueError:

Th