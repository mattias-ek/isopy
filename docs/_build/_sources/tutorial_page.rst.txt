Introduction to isopy
*********************

The isopy package revolves around a set of custom data types used to manipulate three different kinds of data commonly
used in geo/cosmochemistry: Element data, Isotope data and Ratio data.

isopy strings are used to store a sting representation of the type of

 The input string does not have to be correctly formatted as by default the input string will be reformatted, if possible, to the correct format.


ElementString can only contain alphabetical characters. The first character is always in upper case whereas the remaining characters are always in lower case.
>>> isopy.ElementString(‘Pd’)
‘Pd’
>>> isopy.ElementString(‘pd’)
‘Pd’
>>> isopy.ElementString(‘pD’)
‘Pd’

IsotopeString begins with a integer representing the nucleon number (A) followed by an ElementString.
>>> isopy.IsotopeString(‘105pd’)
‘105Pd’
>>> isopy.IsotopeString(‘pd105’)
‘105Pd’

The nucleon number (A) and element of an IsotopeString can be accessed individually
>>> isotope = isopy.IsotopeString(‘105Pd’)
>>> isotope.A
105
>>> isotope.element
‘Pd’

RatioString consists of a single string with numerator (ElementString or IsotopeString) and a denominator (ElementString or IsotopeString) seperated by a ‘/’. Any combination of ElementString and IsotopeString can be used for the numerator and denominator.

>>> isopy.RatioString(‘ru/pd’)
‘Ru/Pd’
>>> isopy.RatioString(‘ru101/pd’)
‘101Ru/Pd’
>>> isopy.RatioString(‘ru101/105pd’)
‘101Ru/105Pd’

The numerator and denominator of an RatioString can be accessed individually
>>> ratio = isopy.RatioString(‘ru/105pd’)
>>> ratio.numerator
‘Ru’
>>> ratio.denominator
‘105Pd’