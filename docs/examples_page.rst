Examples
********

Getting started with IsotopeArray and RatioArray

These arrays are the backbone of isopy and allow us to easily maniplulate data. These classes are a custom view for
structured numpy array and therefore behave like a normal numpy array with a few exceptions.

The only difference between an IsotopeArray and RatioArray is that data is stored with an IsotopeString and RatioSting, respecvivley.

For the following examples we will use an IsotopeArray but will work the same for an RatioArray

Creating an IsotopeArray/RatioArray
-----------------------------------
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



