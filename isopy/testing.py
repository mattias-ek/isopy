import isopy.core as core
import hashlib
import numpy as np
import pyperclip

# TODO create functions for creating arrays with random data keys etc

class ReprObj:
    def __init__(self, thing, string, hash):
        self._string = {}
        self._string[thing] = string
        self.hash = hash
        
    def __repr__(self):
        return self._string.get('repr', 'No __repr__')
    
    def __str__(self):
        return self._string.get('str', 'No __str__')
    
    def _repr_markdown_(self):
        return self._string.get('markdown', None)
    
    def _repr_html_(self):
        return self._string.get('html', None)
    
    def _repr_latex_(self):
        return self._string.get('latex', None)

def hash_str(obj):
    string = obj.__str__()
    if string is None:
        raise TypeError('repr returned None')
    
    strhash  = core.hashstr(string)
    pyperclip.copy(strhash)
    print(f'__str__ hash = {strhash}')
    
    return ReprObj("str", string, strhash)
    
def hash_repr(obj):
    string = obj.__repr__()
    if string is None:
        raise TypeError('repr returned None')
    
    strhash  = core.hashstr(string)
    pyperclip.copy(strhash)
    print(f'__repr__ hash = {strhash}')
    
    return ReprObj("repr", string, strhash)

def hash_markdown(obj):
    string = obj._repr_markdown_()
    if string is None:
        raise TypeError('repr returned None')
    
    strhash  = core.hashstr(string)
    pyperclip.copy(strhash)
    print(f'markdown hash = {strhash}')
    
    return ReprObj("markdown", string, strhash)

def hash_html(obj):
    string = obj._repr_html_()
    if string is None:
        raise TypeError('repr returned None')
    
    strhash  = core.hashstr(string)
    pyperclip.copy(strhash)
    print(f'html hash = {strhash}')
    
    return ReprObj("html", string, strhash)

def hash_latex(obj):
    string = obj._repr_latex_()
    if string is None:
        raise TypeError('repr returned None')
    
    strhash  = core.hashstr(string)
    pyperclip.copy(strhash)
    print(f'latex hash = {strhash}')
    
    return ReprObj("latex", string, strhash)

def hash_test(obj,  name, *hashes):
    test = {}
    for thing in hashes:
        if thing == "repr":
            print('***')
            reprobj = hash_repr(obj)
            print(reprobj.__repr__())
            test["repr"] = f'"{reprobj.hash}"'
        elif thing == "str":
            print('***')
            reprobj = hash_str(obj)
            print(reprobj.__str__())
            test["str"] = f'"{reprobj.hash}"'
        elif thing == "markdown":
            print('***')
            reprobj = hash_markdown(obj)
            print(reprobj._repr_markdown_())
            test["markdown"] = f'"{reprobj.hash}"'
        elif thing == "latex":
            print('***')
            reprobj = hash_latex(obj)
            print(reprobj._repr_latex_())
            test["latex"] = f'"{reprobj.hash}"'
        elif thing == "html":
            print('***')
            reprobj = hash_html(obj)
            print(reprobj._repr_html_())
            test["html"] = f'"{reprobj.hash}"'
        else:
            raise TypeError(f'{thing} is invalid')
    
    things = ',\n'.join([f'{" " * 26}{k}={v}' for k,v in test.items()])
    output = f"isopy.testing.assert_hash({name},\n{things})"
    print(f'***\n{output}\n***')
    pyperclip.copy(output)
    return reprobj
            
        

def assert_hash(obj, str = None, repr = None, markdown = None, latex = None, html=None):
    if str is not None:
        string = obj.__str__()
        correct = str
        
        objhash = core.hashstr(string)
        if objhash != correct:
            raise AssertionError(f'hashes dont match: {objhash} != {correct}')
        
    if repr is not None:
        string = obj.__repr__()
        correct = repr
        
        objhash = core.hashstr(string)
        if objhash != correct:
            raise AssertionError(f'hashes dont match: {objhash} != {correct}')
        
    if markdown is not None:
        string = obj._repr_markdown_()
        correct = markdown
        
        objhash = core.hashstr(string)
        if objhash != correct:
            raise AssertionError(f'hashes dont match: {objhash} != {correct}')
        
    if latex is not None:
        string = obj._repr_latex_()
        correct = latex
        
        objhash = core.hashstr(string)
        if objhash != correct:
            raise AssertionError(f'hashes dont match: {objhash} != {correct}')
        
    if html is not None:
        string = obj._repr_html_()
        correct = html
        
        objhash = core.hashstr(string)
        if objhash != correct:
            raise AssertionError(f'hashes dont match: {objhash} != {correct}')
        

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
    