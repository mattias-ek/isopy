import os as _os
import isopy._dtypes as _dtypes

def get_reference_values(name):
    'initial isotope fraction'
    'mass'
    if name in ['inital isotope fraction']: name = 'inital isotope fraction L09'
    elif name in ['mass']: name = 'isotope mass H17'

    filepath = _os.path.join(_os.path.dirname(__file__), 'referencedata', '{}.csv'.format(name))
    return _dtypes.IsopyDict(filepath=filepath)