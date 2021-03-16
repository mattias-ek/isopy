import numpy as np
from typing import Union
from . import core

__all__ = ['log', 'exp', 'add', 'subtract', 'multiply', 'divide', 'power',
           'ValueError']

#TODO add exp,anation to how these were calcualted
#I cant remember...
###################################
### Error propagation functions ###
###################################

def log(x,
        xerr=0
        ) -> ValueError:
    if isinstance(x, tuple) and len(x) == 2: x, xerr = x #So that nesting functions work
    x = core.asanyarray(x)
    xerr = core.asanyarray(xerr)

    result = np.log(x)
    xdiff = np.divide(1, x)

    error = np.sqrt( (np.power(xerr,2) * np.power(xdiff, 2)) )

    return result, error


def exp(x,
        xerr=0
        ) -> ValueError:
    if isinstance(x, tuple) and len(x) == 2: x, xerr = x #So that nesting functions work
    x = core.asanyarray(x)
    xerr = core.asanyarray(xerr)

    result = np.exp(x)
    xdiff = result
    error = np.sqrt((np.power(xerr, 2) * np.power(xdiff, 2)))

    return result, error


def add(x1,
        x2,
        x1err=0,
        x2err=0
        ) -> ValueError:
    if isinstance(x1, tuple) and len(x1) == 2: x1, x1err = x1 #So that nesting functions work
    if isinstance(x2, tuple) and len(x2) == 2: x2, x2err = x2
    x1 = core.asanyarray(x1)
    x1err = core.asanyarray(x1err)
    x2 = core.asanyarray(x2)
    x2err = core.asanyarray(x2err)

    result = np.add(x1, x2)
    x1diff = np.ones_like(x1)
    x2diff = np.ones_like(x2) #Should be negative but since its squared it doesnt matter
    error = np.sqrt( (np.power(x1err, 2) * x1diff) + (np.power(x2err, 2) * x2diff) )

    return ValueError(result, error)


def subtract(x1,
        x2,
        x1err=0,
        x2err=0
        ) -> ValueError:
    if isinstance(x1, tuple) and len(x1) == 2: x1, x1err = x1  # So that nesting functions work
    if isinstance(x2, tuple) and len(x2) == 2: x2, x2err = x2
    x1 = core.asanyarray(x1)
    x1err = core.asanyarray(x1err)
    x2 = core.asanyarray(x2)
    x2err = core.asanyarray(x2err)

    result = np.subtract(x1, x2)
    x1diff = np.ones_like(x1)
    x2diff = np.ones_like(x2) #Should be negative but since its squared it doesnt matter
    error = np.sqrt( (np.power(x1err, 2) * x1diff) + (np.power(x2err, 2) * x2diff) )

    return ValueError(result, error)


def multiply(x1,
        x2,
        x1err=0,
        x2err=0
        ) -> ValueError:
    if isinstance(x1, tuple) and len(x1) == 2: x1, x1err = x1  # So that nesting functions work
    if isinstance(x2, tuple) and len(x2) == 2: x2, x2err = x2
    x1 = core.asanyarray(x1)
    x1err = core.asanyarray(x1err)
    x2 = core.asanyarray(x2)
    x2err = core.asanyarray(x2err)

    result = np.multiply(x1, x2)
    x1diff = x2
    x2diff = x1
    error = np.sqrt( np.power(x1err, 2)*np.power(x1diff, 2) + np.power(x2err, 2)*np.power(x2diff,2) )

    return ValueError(result, error)


def divide(x1,
        x2,
        x1err=0,
        x2err=0
        ) -> ValueError:
    if isinstance(x1, tuple) and len(x1) == 2: x1, x1err = x1  # So that nesting functions work
    if isinstance(x2, tuple) and len(x2) == 2: x2, x2err = x2
    x1 = core.asanyarray(x1)
    x1err = core.asanyarray(x1err)
    x2 = core.asanyarray(x2)
    x2err = core.asanyarray(x2err)

    result = np.divide(x1, x2)
    x1diff = np.divide(1, x2)
    x2diff = np.divide(x1, np.power(x2, 2)) #Should be negative but we ignore that as we are squaring it
    error = np.sqrt( np.power(x1err, 2)*np.power(x1diff, 2) + np.power(x2err, 2)*np.power(x2diff,2) )

    return ValueError(result, error)


def power(x1,
        x2,
        x1err=0,
        x2err=0
        ) -> ValueError:

    if isinstance(x1, tuple) and len(x1) == 2: x1, x1err = x1  # So that nesting functions work
    if isinstance(x2, tuple) and len(x2) == 2: x2, x2err = x2
    x1 = core.asanyarray(x1)
    x1err = core.asanyarray(x1err)
    x2 = core.asanyarray(x2)
    x2err = core.asanyarray(x2err)

    result = np.power(x1, x2)
    x1diff = x2 * np.power(x1, (x2-1))
    x2diff = np.log(x1) * np.power(x1, x2)
    error = np.sqrt( np.power(x1err, 2)*np.power(x1diff, 2) + np.power(x2err, 2)*np.power(x2diff,2) )

    return ValueError(result, error)


class ValueError(tuple):
    def __new__(cls, value, error=0):
        if isinstance(value, tuple) and len(value) == 2: value, error = value
        value = core.asanyarray(value)
        error = core.asanyarray(error)
        obj = super(ValueError, cls).__new__(cls, (value, error))
        obj.value = value
        obj.error = error
        return obj

    def exp(self):
        return exp(self)

    def log(self):
        return log(self)

    def __pow__(self, power, modulo=None):
        return power(self, power)

    def __rpow__(self, other):
        return power(other, self)

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __sub__(self, other):
        return subtract(self, other)

    def __rsub__(self, other):
        return subtract(other, self)

    def __mul__(self, other):
        return multiply(self, other)

    def __rmul__(self, other):
        return multiply(other, self)

    def __rtruediv__(self, other):
        return divide(self, other)

    def __rtruediv__(self, other):
        return divide(other, self)

    def _o__gt__(self, other):
        return self.value > other

    def _o__lt__(self, other):
        return self.value < other