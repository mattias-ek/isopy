from .core import *
from .io import *
from .array_functions import *
from .plot import IsopyPlot

from . import tb
from .reference_values import ReferenceValues as _RefVal

refval = _RefVal()
plt = IsopyPlot()
