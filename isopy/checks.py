import isopy

# This will be deprecated in favour of Ruthenium
# But can stay until that package is ready
def check_type(name, value, *accepted_types, coerce=False, coerce_into=None, allow_none=False):
    if allow_none and value is None:
        return value

    for accepted_type in accepted_types:
        if isinstance(value, accepted_type):
            return value
    if coerce:
        if coerce_into is None:
            coerce_into = accepted_types
        elif not isinstance(coerce_into, list):
            coerce_into = [coerce_into]
        for coerce_type in coerce_into:
            try:
                return coerce_type(value)
            except:
                continue
        raise TypeError('parameter "{}": unable to coerce \'{}\' into \'{}\''.format(name,
                                                                                     value.__class__.__name__, '\' or \''.join([ct.__name__ for ct in coerce_into])))
    else:
        raise TypeError('parameter "{}": must be \'{}\' not \'{}\''.format(name,
                            '\' or \''.join([ct.__name__ for ct in coerce_into], value.__class__.__name__)))