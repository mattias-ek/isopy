import isopy

def check_type(name, value, *accepted_types, coerce=False, coerce_into=None, allow_list = False, allow_none=False):
    if allow_none and value is None:
        return value
    if allow_list and isinstance(value, list):
        out = []
        try:
            for i in range(len(value)):
                out.append(check_type(name, value[i], *accepted_types, coerce=coerce, coerce_into=coerce_into))
        except TypeError as err:
            raise TypeError('index {} of {}'.format(i, str(err)))
        else:
            return out

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


def check_type_list(name, values, *accepted_types, coerce=False, coerce_into=None, make_list=False, allow_none =False):
    if allow_none and values is None:
        return values
    if not isinstance(values, list):
        if make_list:
            values = [values]
        else:
            raise TypeError('parameter "{}": must be \'{}\' not \'{}\''.format(name, list.__name__,
                                                                               values.__class__.__name__))
    return check_type(name, values, *accepted_types, coerce=coerce, coerce_into=coerce_into, allow_list=True)


def check_reference_value(name, value, reference_values):
    if value is None: return reference_values
    if not hasattr(value, 'get'):
        raise TypeError('parameter "{}": \'{}\' does not have a ".get()" method'.format(name, value.__class__.__name__))
    return value