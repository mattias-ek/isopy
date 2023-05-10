import isopy.core as core
import hashlib
import numpy as np
import pyperclip

# TODO create functions for creating arrays with random data keys etc

class TextRepr:
    def __init__(self, type, text, hash=None):
        self.text = text
        self.hash = hash

        self._repr = 'None'
        self._markdown = None
        self._latex = None
        self._html = None

        setattr(self, f'_{type}', text)

    def __repr__(self):
        return self._repr

    def _repr_markdown_(self):
        return self._markdown

    def _repr_latex_(self):
        return self._latex

    def _repr_html_(self):
        return self._html

    @classmethod
    def repr(cls, text, hash=None):
        return cls('repr', text, hash)

    @classmethod
    def markdown(cls, text, hash=None):
        return cls('markdown', text, hash)

    @classmethod
    def latex(cls, text, hash=None):
        return cls('latex', text, hash)

    @classmethod
    def html(cls, text, hash=None):
        return cls('html', text, hash)


def hash_str(obj):
    string = obj.__str__()
    if string is None:
        raise TypeError('repr returned None')
    
    strhash  = core.hashstr(string)
    pyperclip.copy(strhash)
    
    return TextRepr.repr(string, strhash)
    
def hash_repr(obj):
    string = obj.__repr__()
    if string is None:
        raise TypeError('repr returned None')
    
    strhash  = core.hashstr(string)
    pyperclip.copy(strhash)
    
    return TextRepr.repr(string, strhash)

def hash_markdown(obj):
    string = obj._repr_markdown_()
    if string is None:
        raise TypeError('repr returned None')
    
    strhash  = core.hashstr(string)
    pyperclip.copy(strhash)
    
    return TextRepr.markdown(string, strhash)

def hash_html(obj):
    string = obj._repr_html_()
    if string is None:
        raise TypeError('repr returned None')
    
    strhash  = core.hashstr(string)
    pyperclip.copy(strhash)
    
    return TextRepr.html(string, strhash)

def hash_latex(obj):
    string = obj._repr_latex_()
    if string is None:
        raise TypeError('repr returned None')
    
    strhash  = core.hashstr(string)
    pyperclip.copy(strhash)
    
    return TextRepr.latex(string, strhash)

def generate_hash_test(repr = True, str = True, **arrays):
    return _ght_(arrays, _ght_str_repr_, 'repr', repr, str)

def generate_hash_test_html(*, repr = False, str = False, **arrays):
    return _ght_(arrays, _ght_special_, 'html', repr, str, '<br>', html = hash_html)

def generate_hash_test_markdown(*, repr = False, str = False, **arrays):
    return _ght_(arrays, _ght_special_, 'markdown', repr, str, '\n', markdown = hash_markdown)

def generate_hash_test_latex(*, repr = False, str = False, **arrays):
    return _ght_(arrays, _ght_special_, 'latex', repr, str, '\n', latex = hash_latex)

def _ght_(arrays, func, out_type,  *args, **kwargs):
    test_strings = []
    texts = []
    for var, item in arrays.items():
        if type(item) is tuple and len(item) == 2:
            item, descr = item
        else:
            descr = ''
        test_string, text = func(var, item, descr, *args, **kwargs)
        test_strings.append(test_string)
        texts.append(text)
    return _ght_finalize2_(out_type, test_strings, texts)

def _ght_str_repr_(var, item, descr, repr, str):
    text = ''
    test_hash = {}
    if descr:
        descr = f'\n{var} = ' + descr

    if repr:
        reprobj = hash_repr(item)
        test_hash["repr"] = f'"{reprobj.hash}"'
        text += f'*** {var}: repr = {reprobj.hash} ***' + descr + '\n\n' + reprobj.text + '\n\n'

    if str:
        reprobj = hash_str(item)
        test_hash["str"] = f'"{reprobj.hash}"'
        text += f'*** {var}: str = {reprobj.hash} ***' + descr + '\n\n' + reprobj.text + '\n\n'

    return _ght_finalize1_(var, text, test_hash)

def _ght_special_(var, item, descr, repr, str, linebreak, **hash_func):
    text = ''
    test_hash = {}
    if descr:
        descr = f'{linebreak}{var} = ' + descr

    hash_type, hash_func = tuple(hash_func.items())[0]

    reprobj = hash_func(item)
    test_hash[hash_type] = f'"{reprobj.hash}"'
    text += f'*** {var}: {hash_type} = {reprobj.hash} ***' + descr + linebreak*2 + reprobj.text + linebreak*2

    if repr:
        reprobj = hash_func(item.__repr__())
        test_hash[f"repr_{hash_type}"] = f'"{reprobj.hash}"'
        text += f'*** {var}: repr-{hash_type} = {reprobj.hash} ***' + descr + linebreak*2 + reprobj.text + linebreak*2

    if str:
        reprobj = hash_func(item.__str__())
        test_hash[f"str_{hash_type}"] = f'"{reprobj.hash}"'
        text += f'*** {var}: str-{hash_type} = {reprobj.hash} ***' + descr + linebreak*2 + reprobj.text + linebreak*2

    return _ght_finalize1_(var, text, test_hash)

def _ght_finalize1_(var, text, test_hash):
    test_string = ',\n'.join([f'{" " * 26}{k}={v}' for k, v in test_hash.items()])
    test_string = f"isopy.testing.assert_hash({var},\n{test_string})"

    return test_string, text

def _ght_finalize2_(hash_type, test_strings, texts):
    test_string = '\n'.join(test_strings)
    text = ''.join(texts)

    pyperclip.copy(test_string)
    return TextRepr(hash_type, text, None)

            
def assert_hash(obj, str = None, repr = None,
                markdown = None, repr_markdown = None, str_markdown = None,
                latex = None, repr_latex = None, str_latex = None,
                html=None, repr_html = None, str_html=None):
    if str is not None:
        string = obj.__str__()
        correct = str
        
        objhash = core.hashstr(string)
        if objhash != correct:
            raise AssertionError(f'str hashes dont match: {objhash} != {correct}')
        
    if repr is not None:
        string = obj.__repr__()
        correct = repr
        
        objhash = core.hashstr(string)
        if objhash != correct:
            raise AssertionError(f'repr hashes dont match: {objhash} != {correct}')
        
    if markdown is not None:
        string = obj._repr_markdown_()
        correct = markdown
        
        objhash = core.hashstr(string)
        if objhash != correct:
            raise AssertionError(f'markdown hashes dont match: {objhash} != {correct}')

    if repr_markdown is not None:
        string = obj.__repr__()._repr_markdown_()
        correct = repr_markdown

        objhash = core.hashstr(string)
        if objhash != correct:
            raise AssertionError(f'repr-markdown hashes dont match: {objhash} != {correct}')

    if str_markdown is not None:
        string = obj.__str__()._repr_markdown_()
        correct = str_markdown

        objhash = core.hashstr(string)
        if objhash != correct:
            raise AssertionError(f'str-markdown hashes dont match: {objhash} != {correct}')

    if latex is not None:
        string = obj._repr_latex_()
        correct = latex
        
        objhash = core.hashstr(string)
        if objhash != correct:
            raise AssertionError(f'latex hashes dont match: {objhash} != {correct}')

    if repr_latex is not None:
        string = obj.__repr__()._repr_latex_()
        correct = repr_latex

        objhash = core.hashstr(string)
        if objhash != correct:
            raise AssertionError(f'repr-latex hashes dont match: {objhash} != {correct}')

    if str_latex is not None:
        string = obj.__str__()._repr_latex_()
        correct = str_latex

        objhash = core.hashstr(string)
        if objhash != correct:
            raise AssertionError(f'str-latex hashes dont match: {objhash} != {correct}')
        
    if html is not None:
        string = obj._repr_html_()
        correct = html
        
        objhash = core.hashstr(string)
        if objhash != correct:
            raise AssertionError(f'html hashes dont match: {objhash} != {correct}')

    if repr_html is not None:
        string = obj.__repr__()._repr_html_()
        correct = repr_html

        objhash = core.hashstr(string)
        if objhash != correct:
            raise AssertionError(f'repr -tml hashes dont match: {objhash} != {correct}')

    if str_html is not None:
        string = obj.__str__()._repr_html_()
        correct = str_html

        objhash = core.hashstr(string)
        if objhash != correct:
            raise AssertionError(f'str-html hashes dont match: {objhash} != {correct}')
        
def assert_compare_arrays(copy, original):
    if isinstance(original, core.IsopyArray):
        if not isinstance(copy, core.IsopyArray):
            raise AssertionError(f'copy is not an IsopyArray: {type(copy)}')
        if copy.keys != original.keys:
            raise AssertionError(f'keys dont match: {copy.keys} != {original.keys}')
        if copy.ndim != original.ndim:
            raise AssertionError(f'ndim doesnt match: {copy.ndim} != {original.ndim}')
        if copy.datatypes != original.datatypes:
            raise AssertionError(f'datatypes dont match: {copy.datatypes} != {original.datatypes}')
        for k in original.keys:
            np.testing.assert_allclose(copy[k], original[k])
    else:
        if isinstance(copy, core.IsopyArray):
            raise AssertionError(f'copy is an IsopyArray: {type(copy)}')
        if not isinstance(copy, np.ndarray):
            raise AssertionError(f'copy is not an numpy ndarray: {type(copy)}')
        if copy.ndim != original.ndim:
            raise AssertionError(f'ndim doesnt match: {copy.ndim} != {original.ndim}')
        if copy.dtype != original.dtype:
            raise AssertionError(f'dtype doesnt match: {copy.dtype} != {original.dtype}')
        np.testing.assert_allclose(copy, original)
    