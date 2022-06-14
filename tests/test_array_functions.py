import isopy
from isopy import core, array_functions
import numpy as np
import pytest
import warnings

def assert_array_equal_array(array1, array2, match_dtype=True):
    assert isinstance(array1, core.IsopyArray)
    assert array1.flavour == array1.keys.flavour

    assert array1.dtype.names is not None
    assert array2.dtype.names is not None

    assert len(array1.dtype.names) == len(array2.dtype.names)
    for i in range(len(array1.dtype.names)):
        assert str(array1.dtype.names[i]) == str(array2.dtype.names[i])
    assert array1.ndim == array2.ndim
    assert array1.size == array2.size
    assert array1.shape == array2.shape
    if match_dtype:
        for i in range(len(array1.dtype)):
            assert array1.dtype[i] == array2.dtype[i]
    for name in array1.dtype.names:
        np.testing.assert_allclose(array1[name], array2[name])

class Test_ArrayFunctions:
    def test_singleinput(self):
        keys = 'ru pd cd'.split()
        array1 = isopy.random(None, [(0, 1), (1, 0.1), (0.5, 0.5)], keys, seed=46)
        array2 = isopy.random(100, [(0, 1), (1, 0.1), (0.5, 0.5)], keys, seed=46)
        array3 = array2.copy()
        array3['pd'][5] = np.nan

        dict1 = array1.to_scalardict()
        dict2 = array2.to_scalardict()
        dict3 = array3.to_scalardict()

        # Elementwise functions
        for func in array_functions.np_elementwise:
            self.singleinput_a(func, array1)
            self.singleinput_a(func, array2)
            self.singleinput_a(func, array3)

            self.singleinput_ds(func, dict1)
            self.singleinput_ds(func, dict2)
            self.singleinput_ds(func, dict3)

            # There isnt enough arguments to an error should be thrown
            with pytest.raises((ValueError, TypeError)):
                func()

            # In case its not caught by the array dispatcher.
            with pytest.raises((ValueError, TypeError)):
                isopy.arrayfunc(func)

        # Cumulative functions
        for func in array_functions.np_cumulative:
            self.singleinput_a(func, array1)
            self.singleinput_a(func, array2)
            self.singleinput_a(func, array3)

            self.singleinput_a(func, array1, axis=None)
            self.singleinput_a(func, array2, axis=None)
            self.singleinput_a(func, array3, axis=None)

            self.singleinput_a(func, array1, axis=0)
            self.singleinput_a(func, array2, axis=0)
            self.singleinput_a(func, array3, axis=0)

            self.singleinput_a(func, array2, axis=1)
            self.singleinput_a(func, array3, axis=1)

            self.singleinput_ds(func, dict1)
            self.singleinput_ds(func, dict2)
            self.singleinput_ds(func, dict3)

            self.singleinput_ds(func, dict1, axis=None)
            self.singleinput_ds(func, dict2, axis=None)
            self.singleinput_ds(func, dict3, axis=None)

            self.singleinput_ds(func, dict1, axis=0)
            self.singleinput_ds(func, dict2, axis=0)
            self.singleinput_ds(func, dict3, axis=0)

            self.singleinput_ds(func, dict2, axis=1)
            self.singleinput_ds(func, dict3, axis=1)

            # There isnt enough arguments to an error should be thrown
            with pytest.raises((ValueError, TypeError)):
                func()

            # In case its not caught by the array dispatcher.
            with pytest.raises((ValueError, TypeError)):
                isopy.arrayfunc(func)

        # reducing numpy functions
        for func in array_functions.np_reducing:
            self.singleinput_a(func, array1)
            self.singleinput_a(func, array2)
            self.singleinput_a(func, array3)

            self.singleinput_a(func, array1, axis=None)
            self.singleinput_a(func, array2, axis=None)
            self.singleinput_a(func, array3, axis=None)

            self.singleinput_a(func, array1, axis=0)
            self.singleinput_a(func, array2, axis=0)
            self.singleinput_a(func, array3, axis=0)

            self.singleinput_a(func, array2, axis=1)
            self.singleinput_a(func, array3, axis=1)

            self.singleinput_ds(func, dict1)
            self.singleinput_ds(func, dict2)
            self.singleinput_ds(func, dict3)

            self.singleinput_ds(func, dict1, axis=None)
            self.singleinput_ds(func, dict2, axis=None)
            self.singleinput_ds(func, dict3, axis=None)

            self.singleinput_ds(func, dict1, axis=0)
            self.singleinput_ds(func, dict2, axis=0)
            self.singleinput_ds(func, dict3, axis=0)

            self.singleinput_ds(func, dict2, axis=1)
            self.singleinput_ds(func, dict3, axis=1)

            # There isnt enough arguments to an error should be thrown
            with pytest.raises((ValueError, TypeError)):
                func()

            # In case its not caught by the array dispatcher.
            with pytest.raises((ValueError, TypeError)):
                isopy.arrayfunc(func)

        # reducing isopy functions
        for func in [isopy.mad, isopy.nanmad, isopy.se, isopy.nanse, isopy.sd, isopy.nansd,
                     isopy.sd2, isopy.sd95, isopy.se2, isopy.se95, isopy.mad2, isopy.mad95,
                     isopy.nansd2, isopy.nansd95, isopy.nanse2, isopy.nanse95, isopy.nanmad2, isopy.nanmad95]:
            self.singleinput_a(func, array1)
            self.singleinput_a(func, array2)
            self.singleinput_a(func, array3)

            self.singleinput_a(func, array1, axis=None)
            self.singleinput_a(func, array2, axis=None)
            self.singleinput_a(func, array3, axis=None)

            self.singleinput_a(func, array1, axis=0)
            self.singleinput_a(func, array2, axis=0)
            self.singleinput_a(func, array3, axis=0)

            self.singleinput_a(func, array2, axis=1)
            self.singleinput_a(func, array3, axis=1)

            self.singleinput_ds(func, dict1)
            self.singleinput_ds(func, dict2)
            self.singleinput_ds(func, dict3)

            self.singleinput_ds(func, dict1, axis=None)
            self.singleinput_ds(func, dict2, axis=None)
            self.singleinput_ds(func, dict3, axis=None)

            self.singleinput_ds(func, dict1, axis=0)
            self.singleinput_ds(func, dict2, axis=0)
            self.singleinput_ds(func, dict3, axis=0)

            self.singleinput_ds(func, dict2, axis=1)
            self.singleinput_ds(func, dict3, axis=1)
            self.singleinput_a(func, array3, axis=1)

            # There isnt enough arguments to an error should be thrown
            with pytest.raises((ValueError, TypeError)):
                func()

            # In case its not caught by the array dispatcher.
            with pytest.raises((ValueError, TypeError)):
                isopy.arrayfunc(func)

        # Test the reduce method
        self.singleinput_a(np.add.reduce, array1)
        self.singleinput_a(np.add.reduce, array2)
        self.singleinput_a(np.add.reduce, array3)

        self.singleinput_a(np.add.reduce, array1, axis=None)
        self.singleinput_a(np.add.reduce, array2, axis=None)
        self.singleinput_a(np.add.reduce, array3, axis=None)

        self.singleinput_a(np.add.reduce, array1, axis=0)
        self.singleinput_a(np.add.reduce, array2, axis=0)
        self.singleinput_a(np.add.reduce, array3, axis=0)

        self.singleinput_a(np.add.reduce, array2, axis=1)
        self.singleinput_a(np.add.reduce, array3, axis=1)

        self.singleinput_ds(np.add.reduce, dict1)
        self.singleinput_ds(np.add.reduce, dict2)
        self.singleinput_ds(np.add.reduce, dict3)

        self.singleinput_ds(np.add.reduce, dict1, axis=None)
        self.singleinput_ds(np.add.reduce, dict2, axis=None)
        self.singleinput_ds(np.add.reduce, dict3, axis=None)

        self.singleinput_ds(np.add.reduce, dict1, axis=0)
        self.singleinput_ds(np.add.reduce, dict2, axis=0)
        self.singleinput_ds(np.add.reduce, dict3, axis=0)

        self.singleinput_ds(np.add.reduce, dict2, axis=1)
        self.singleinput_ds(np.add.reduce, dict3, axis=1)

        # Test the accumulate method
        # self.singleinput(np.add.accumulate, array1) # Dosnt work for 0-dim
        self.singleinput_a(np.add.accumulate, array2)
        self.singleinput_a(np.add.accumulate, array3)

        self.singleinput_a(np.add.accumulate, array1, axis=None)
        # self.singleinput(np.add.accumulate, array2, axis=None) # Doesnt work for > 1-dim
        # self.singleinput(np.add.accumulate, array3, axis=None) # Doesnt work for > 1-dim

        # self.singleinput(np.add.accumulate, array1, axis=0) # Dosnt work for 0-dim
        self.singleinput_a(np.add.accumulate, array2, axis=0)
        self.singleinput_a(np.add.accumulate, array3, axis=0)

        self.singleinput_a(np.add.accumulate, array2, axis=1)
        self.singleinput_a(np.add.accumulate, array3, axis=1)

        # self.singleinpud(np.add.accumulate, dict1) # Dosnt work for 0-dim
        self.singleinput_ds(np.add.accumulate, dict2)
        self.singleinput_ds(np.add.accumulate, dict3)

        self.singleinput_ds(np.add.accumulate, dict1, axis=None)
        # self.singleinpud(np.add.accumulate, dict2, axis=None) # Doesnt work for > 1-dim
        # self.singleinpud(np.add.accumulate, dict3, axis=None) # Doesnt work for > 1-dim

        # self.singleinpud(np.add.accumulate, dict1, axis=0) # Dosnt work for 0-dim
        self.singleinput_ds(np.add.accumulate, dict2, axis=0)
        self.singleinput_ds(np.add.accumulate, dict3, axis=0)

        self.singleinput_ds(np.add.accumulate, dict2, axis=1)
        self.singleinput_ds(np.add.accumulate, dict3, axis=1)

    def singleinput_a(self, func, a, axis=core.NotGiven, out=core.NotGiven, where=core.NotGiven):
        kwargs = {}
        if axis is not core.NotGiven: kwargs['axis'] = axis
        if out is not core.NotGiven: kwargs['out'] = out
        if where is not core.NotGiven: kwargs['where'] = where

        if axis is core.NotGiven or axis == 0:

            result = func(a, **kwargs)
            assert isinstance(result, core.IsopyArray)
            assert result.keys == a.keys
            for key in result.keys:
                true = func(a[key])
                assert result[key].size == true.size
                assert result[key].ndim == true.ndim
                np.testing.assert_allclose(result[key], true)

            result = isopy.arrayfunc(func, a, **kwargs)
            assert isinstance(result, core.IsopyArray)
            assert result.keys == a.keys
            for key in result.keys:
                true = func(a[key])
                assert result[key].size == true.size
                assert result[key].ndim == true.ndim
                np.testing.assert_allclose(result[key], true)

            # For builtin functions
            funcname = {'amin': 'min', 'amax': 'max'}.get(func.__name__, func.__name__)
            if hasattr(a, funcname):
                result = getattr(a, funcname)(**kwargs)
                assert isinstance(result, core.IsopyArray)
                assert result.keys == a.keys
                for key in result.keys:
                    true = func(a[key])
                    assert result[key].size == true.size
                    assert result[key].ndim == true.ndim
                    np.testing.assert_allclose(result[key], true)

        else:
            true = func(a.to_list(), **kwargs)

            result = func(a, **kwargs)
            assert not isinstance(result, core.IsopyArray)
            assert result.size == true.size
            assert result.ndim == true.ndim
            np.testing.assert_allclose(result, true)

            result = isopy.arrayfunc(func, a, **kwargs)
            assert not isinstance(result, core.IsopyArray)
            assert result.size == true.size
            assert result.ndim == true.ndim
            np.testing.assert_allclose(result, true)

            # For builtin functions
            funcname = {'amin': 'min', 'amax': 'max'}.get(func.__name__, func.__name__)
            if hasattr(a, funcname):
                result = getattr(a, funcname)(**kwargs)
                assert not isinstance(result, core.IsopyArray)
                assert result.size == true.size
                assert result.ndim == true.ndim
                np.testing.assert_allclose(result, true)

    def singleinput_ds(self, func, ds, axis=core.NotGiven, out=core.NotGiven, where=core.NotGiven):
        kwargs = {}

        if axis is not core.NotGiven: kwargs['axis'] = axis
        if out is not core.NotGiven: kwargs['out'] = out
        if where is not core.NotGiven: kwargs['where'] = where

        if axis is core.NotGiven or axis == 0:
            result = func(ds, **kwargs)
            assert isinstance(result, core.ScalarDict)
            assert result.keys == ds.keys
            for key in result.keys:
                true = func(ds[key])
                assert result[key].size == true.size
                if true.size == 1:
                    assert result[key].ndim == 0
                else:
                    assert result[key].ndim == 1
                np.testing.assert_allclose(result[key], true)

            result = isopy.arrayfunc(func, ds, **kwargs)
            assert isinstance(result, core.ScalarDict)
            assert result.keys == ds.keys
            for key in result.keys:
                true = func(ds[key])
                assert result[key].size == true.size
                if true.size == 1:
                    assert result[key].ndim == 0
                else:
                    assert result[key].ndim == 1
                np.testing.assert_allclose(result[key], true)

            result = isopy.arrayfunc(func, ds.to_dict(), **kwargs)
            assert isinstance(result, core.ScalarDict)
            assert result.keys == ds.keys
            for key in result.keys:
                true = func(ds[key])
                assert result[key].size == true.size
                if true.size == 1:
                    assert result[key].ndim == 0
                else:
                    assert result[key].ndim == 1
                np.testing.assert_allclose(result[key], true)

        else:
            true = func(ds.to_list(), **kwargs)

            result = func(ds, **kwargs)
            assert not isinstance(result, core.ScalarDict)
            assert result.size == true.size
            assert result.ndim == true.ndim
            np.testing.assert_allclose(result, true)

            result = isopy.arrayfunc(func, ds, **kwargs)
            assert not isinstance(result, core.ScalarDict)
            assert result.size == true.size
            assert result.ndim == true.ndim
            np.testing.assert_allclose(result, true)

    def test_dualinput(self):
        keys1 = 'ru pd cd'.split()
        keys2 = 'pd ag cd'.split()

        array1 = isopy.random(None, [(0, 1), (1, 0.1), (0.5, 0.5)], keys1, seed=1)
        array2 = isopy.random(None, [(0, 1), (1, 0.1), (0.5, 0.5)], keys2, seed=2)
        array3 = isopy.random(1, [(0, 1), (1, 0.1), (0.5, 0.5)], keys1, seed=3)
        array4 = isopy.random(1, [(0, 1), (1, 0.1), (0.5, 0.5)], keys2, seed=4)
        array5 = isopy.random(10, [(0, 1), (1, 0.1), (0.5, 0.5)], keys1, seed=5)
        array6 = isopy.random(10, [(0, 1), (1, 0.1), (0.5, 0.5)], keys2, seed=6)
        array7 = isopy.random(3, [(0, 1), (1, 0.1), (0.5, 0.5)], keys1, seed=7)
        dict1 = array1.to_dict()
        dict2 = array2.to_dict()
        dict3 = array3.to_dict()
        dict4 = array4.to_dict()
        dict5 = array5.to_dict()
        dict6 = array6.to_dict()
        dict7 = array7.to_dict()
        value1 = 1.5
        value2 = [3.2]
        value3 = [i for i in range(10)]
        value4 = [i for i in range(3)]

        for func in (array_functions.np_dual):
            # all different array combinations
            self.dualinput_aa(func, array1, array2, 0)
            self.dualinput_aa(func, array1, array4, 1)
            self.dualinput_aa(func, array1, array6, 1)

            self.dualinput_aa(func, array3, array2, 1)
            self.dualinput_aa(func, array3, array4, 1)
            self.dualinput_aa(func, array3, array6, 1)

            self.dualinput_aa(func, array5, array2, 1)
            self.dualinput_aa(func, array5, array4, 1)
            self.dualinput_aa(func, array5, array6, 1)

            with pytest.raises(ValueError):
                self.dualinput_aa(func, array5, array7, 1)
            with pytest.raises(ValueError):
                self.dualinput_aa(func, array7, array6, 1)

            # array dict combinations
            self.dualinput_ad(func, array1, dict2, 0)
            self.dualinput_ad(func, array1, dict4, 0)
            self.dualinput_ad(func, array1, dict6, 1)

            self.dualinput_ad(func, array3, dict2, 1)
            self.dualinput_ad(func, array3, dict4, 1)
            self.dualinput_ad(func, array3, dict6, 1)

            self.dualinput_ad(func, array5, dict2, 1)
            self.dualinput_ad(func, array5, dict4, 1)
            self.dualinput_ad(func, array5, dict6, 1)

            with pytest.raises(ValueError):
                self.dualinput_ad(func, array5, dict7, 1)
            with pytest.raises(ValueError):
                self.dualinput_ad(func, array7, dict6, 1)

            # all array value combinations
            self.dualinput_av(func, array1, value1)
            self.dualinput_av(func, array1, value2)
            self.dualinput_av(func, array1, value3)

            self.dualinput_av(func, array3, value1)
            self.dualinput_av(func, array3, value2)
            self.dualinput_av(func, array3, value3)

            self.dualinput_av(func, array5, value1)
            self.dualinput_av(func, array5, value2)
            self.dualinput_av(func, array5, value3)

            with pytest.raises(ValueError):
                self.dualinput_av(func, array5, value4)
            with pytest.raises(ValueError):
                self.dualinput_av(func, array7, value3)

            #dict dict combinations
            self.dualinput_dd(func, dict1, dict2)
            self.dualinput_dd(func, dict1, dict4)
            self.dualinput_dd(func, dict1, dict6)

            self.dualinput_dd(func, dict3, dict2)
            self.dualinput_dd(func, dict3, dict4)
            self.dualinput_dd(func, dict3, dict6)

            self.dualinput_dd(func, dict5, dict2)
            self.dualinput_dd(func, dict5, dict4)
            self.dualinput_dd(func, dict5, dict6)

            with pytest.raises(ValueError):
                self.dualinput_dd(func, dict5, dict7)
            with pytest.raises(ValueError):
                self.dualinput_dd(func, dict7, dict6)

            #dict value combiunations
            self.dualinput_dv(func, dict1, value1)
            self.dualinput_dv(func, dict1, value2)
            self.dualinput_dv(func, dict1, value3)

            self.dualinput_dv(func, dict3, value1)
            self.dualinput_dv(func, dict3, value2)
            self.dualinput_dv(func, dict3, value3)

            self.dualinput_dv(func, dict5, value1)
            self.dualinput_dv(func, dict5, value2)
            self.dualinput_dv(func, dict5, value3)

            with pytest.raises(ValueError):
                self.dualinput_dv(func, dict5, value4)
            with pytest.raises(ValueError):
                self.dualinput_dv(func, dict7, value3)

    def dualinput_aa(self, func, a1, a2, ndim):
        keys = a1.keys | a2.keys

        result = func(a1, a2)
        assert isinstance(result, core.IsopyArray)
        assert result.keys == keys
        assert result.ndim == ndim
        for key in keys:
            true = func(a1.get(key, np.nan), a2.get(key, np.nan))
            assert result.size == true.size
            np.testing.assert_allclose(result[key], true)

        result2 = isopy.arrayfunc(func, a1, a2)
        assert result2.ndim == ndim
        assert_array_equal_array(result, result2)


        result2 = func(a1.default(1), a2)
        for key in keys:
            true = func(a1.get(key, 1), a2.get(key))
            assert result2.size == true.size
            np.testing.assert_allclose(result2[key], true)
        try:
            assert_array_equal_array(result2, result)
        except AssertionError:
            pass
        else:
            raise AssertionError()

        result2 = func(a1, a2.default(2))
        for key in keys:
            true = func(a1.get(key), a2.get(key, 2))
            assert result2.size == true.size
            np.testing.assert_allclose(result2[key], true)
        try:
            assert_array_equal_array(result2, result)
        except AssertionError:
            pass
        else:
            raise AssertionError()

        result2 = func(a1.default(1), a2.default(2))
        assert result2.ndim == ndim
        for key in keys:
            true = func(a1.get(key, 1), a2.get(key, 2))
            assert result2.size == true.size
            np.testing.assert_allclose(result2[key], true)
        try:
            assert_array_equal_array(result, result2)
        except AssertionError:
            pass
        else:
            raise AssertionError()

    def dualinput_ad(self, func, a, d, ndim):
        keys = a.keys
        ds = isopy.ScalarDict(d)

        result = func(a, d)
        assert isinstance(result, core.IsopyArray)
        assert result.keys == keys
        assert result.ndim == ndim
        for key in keys:
            true = func(a.get(key, np.nan), ds.get(key, np.nan))
            assert result.size == true.size
            np.testing.assert_allclose(result[key], true)

        result2 = func(a, ds)
        assert result2.keys == keys
        assert result2.ndim == ndim
        assert_array_equal_array(result, result2)

        result2 = isopy.arrayfunc(func,a, d)
        assert result2.keys == keys
        assert result2.ndim == ndim
        assert_array_equal_array(result, result2)

        # Flip input arguments
        result = func(d, a)
        assert isinstance(result, core.IsopyArray)
        assert result.keys == keys
        assert result.ndim == ndim
        for key in keys:
            true = func(ds.get(key, np.nan), a.get(key, np.nan))
            assert result.size == true.size
            np.testing.assert_allclose(result[key], true)

        result2 = func(ds, a)
        assert result2.keys == keys
        assert result2.ndim == ndim
        assert_array_equal_array(result, result2)

        result2 = isopy.arrayfunc(func, d, a)
        assert result2.keys == keys
        assert result2.ndim == ndim
        assert_array_equal_array(result, result2)

    def dualinput_dd(self, func, d1, d2):
        ds1 = isopy.ScalarDict(d1)
        ds2 = isopy.ScalarDict(d2)
        keys = ds1.keys | ds2.keys

        with pytest.raises(TypeError):
            func(d1, d2) # At least one dictionaries must be a scalar dict

        result = func(ds1, d2)
        assert isinstance(result, core.ScalarDict)
        assert result.keys == keys
        for key in keys:
            true = func(ds1.get(key, np.nan), ds2.get(key, np.nan))
            assert result.size == true.size
            np.testing.assert_allclose(result[key], true)

        result = func(d1, ds2)
        assert isinstance(result, core.ScalarDict)
        assert result.keys == keys
        for key in keys:
            true = func(ds1.get(key, np.nan), ds2.get(key, np.nan))
            assert result.size == true.size
            np.testing.assert_allclose(result[key], true)

        result = func(ds1, ds2)
        assert isinstance(result, core.ScalarDict)
        assert result.keys == keys
        for key in keys:
            true = func(ds1.get(key, np.nan), ds2.get(key, np.nan))
            assert result.size == true.size
            np.testing.assert_allclose(result[key], true)

        #Array func
        result = isopy.arrayfunc(func, d1, d2)
        assert isinstance(result, core.ScalarDict)
        assert result.keys == keys
        for key in keys:
            true = func(ds1.get(key, np.nan), ds2.get(key, np.nan))
            assert result.size == true.size
            np.testing.assert_allclose(result[key], true)

        result = isopy.arrayfunc(func, ds1, d2)
        assert isinstance(result, core.ScalarDict)
        assert result.keys == keys
        for key in keys:
            true = func(ds1.get(key, np.nan), ds2.get(key, np.nan))
            assert result.size == true.size
            np.testing.assert_allclose(result[key], true)

        result = isopy.arrayfunc(func, d1, ds2)
        assert isinstance(result, core.ScalarDict)
        assert result.keys == keys
        for key in keys:
            true = func(ds1.get(key, np.nan), ds2.get(key, np.nan))
            assert result.size == true.size
            np.testing.assert_allclose(result[key], true)

        result = isopy.arrayfunc(func, ds1, ds2)
        assert isinstance(result, core.ScalarDict)
        assert result.keys == keys
        for key in keys:
            true = func(ds1.get(key, np.nan), ds2.get(key, np.nan))
            assert result.size == true.size
            np.testing.assert_allclose(result[key], true)

        ds1 = isopy.ScalarDict(d1, default_value=1)
        ds2 = isopy.ScalarDict(d2, default_value=2)
        keys = ds1.keys | ds2.keys

        with pytest.raises(TypeError):
            func(d1, d2)  # At least one dictionaries must be a scalar dict

        result = func(ds1, d2)
        assert isinstance(result, core.ScalarDict)
        assert result.keys == keys
        for key in keys:
            true = func(ds1.get(key, 1), ds2.get(key, np.nan))
            assert result.size == true.size
            np.testing.assert_allclose(result[key], true)

        result = func(d1, ds2)
        assert isinstance(result, core.ScalarDict)
        assert result.keys == keys
        for key in keys:
            true = func(ds1.get(key, np.nan), ds2.get(key, 2))
            assert result.size == true.size
            np.testing.assert_allclose(result[key], true)

        result = func(ds1, ds2)
        assert isinstance(result, core.ScalarDict)
        assert result.keys == keys
        for key in keys:
            true = func(ds1.get(key, 1), ds2.get(key, 2))
            assert result.size == true.size
            np.testing.assert_allclose(result[key], true)

    def dualinput_av(self, func, a, v):
        keys = a.keys

        result = func(a, v)
        assert isinstance(result, core.IsopyArray)
        assert result.keys == keys
        for key in keys:
            true = func(a.get(key), v)
            assert result.size == true.size
            assert result.ndim == true.ndim
            np.testing.assert_allclose(result[key], true)

        result2 = isopy.arrayfunc(func, a, v)
        assert result2.keys == keys
        assert_array_equal_array(result, result2)

        # Flip order of values
        result = func(v, a)
        assert isinstance(result, core.IsopyArray)
        assert result.keys == keys
        for key in keys:
            true = func(v, a.get(key))
            assert result.size == true.size
            assert result.ndim == true.ndim
            np.testing.assert_allclose(result[key], true)

        result2 = isopy.arrayfunc(func, v, a)
        assert result2.keys == keys
        assert_array_equal_array(result, result2)

    def dualinput_dv(self, func, d, v):
        ds = isopy.ScalarDict(d)
        keys = ds.keys

        with pytest.raises(TypeError):
            func(d, v)  # At least one dictionaries must be a scalar dict

        result = isopy.arrayfunc(func, d, v)
        assert isinstance(result, core.ScalarDict)
        assert result.keys == keys
        for key in keys:
            true = func(ds.get(key, np.nan), v)
            assert result.size == true.size
            np.testing.assert_allclose(result[key], true)

        result = func(ds, v)
        assert isinstance(result, core.ScalarDict)
        assert result.keys == keys
        for key in keys:
            true = func(ds.get(key, np.nan), v)
            assert result.size == true.size
            np.testing.assert_allclose(result[key], true)

        result = isopy.arrayfunc(func, ds, v)
        assert isinstance(result, core.ScalarDict)
        assert result.keys == keys
        for key in keys:
            true = func(ds.get(key, np.nan), v)
            assert result.size == true.size
            np.testing.assert_allclose(result[key], true)

        # Flip the order or inputs
        with pytest.raises(TypeError):
            func(v, d)  # At least one dictionaries must be a scalar dict

        result = isopy.arrayfunc(func, v, d)
        assert isinstance(result, core.ScalarDict)
        assert result.keys == keys
        for key in keys:
            true = func(v, ds.get(key, np.nan))
            assert result.size == true.size
            np.testing.assert_allclose(result[key], true)

        result = func(v, ds)
        assert isinstance(result, core.ScalarDict)
        assert result.keys == keys
        for key in keys:
            true = func(v, ds.get(key, np.nan))
            assert result.size == true.size
            np.testing.assert_allclose(result[key], true)

        result = isopy.arrayfunc(func, v, ds)
        assert isinstance(result, core.ScalarDict)
        assert result.keys == keys
        for key in keys:
            true = func(v, ds.get(key, np.nan))
            assert result.size == true.size
            np.testing.assert_allclose(result[key], true)

    def test_deprecated_dualinput(self):
        keys1 = 'ru pd cd'.split()
        keys2 = 'pd ag cd'.split()

        a1 = isopy.random(1, [(0, 1), (1, 0.1), (0.5, 0.5)], keys1, seed=1)
        a2 = isopy.random(1, [(0, 1), (1, 0.1), (0.5, 0.5)], keys2, seed=2)
        d1 = a1.to_dict()
        d2 = a2.to_dict()
        ds1 = a1.to_scalardict()
        ds2 = a2.to_scalardict()

        # a, a
        result = isopy.add(a1, a2)
        true = a1 + a2
        assert_array_equal_array(result, true)

        result = isopy.add(a1, a2, 0)
        true = a1.default(0) + a2.default(0)
        assert_array_equal_array(result, true)

        result = isopy.add(a1, a2, (0,1))
        true = a1.default(0) + a2.default(1)
        assert_array_equal_array(result, true)

        # a, d
        result = isopy.add(a1, d2)
        true = a1 + a2.to_scalardict()
        assert_array_equal_array(result, true)

        result = isopy.add(a1, d2, 0)
        true = a1.default(0) + a2.to_scalardict(0)
        assert_array_equal_array(result, true)

        result = isopy.add(a1, d2, (0, 1))
        true = a1.default(0) + a2.to_scalardict(1)
        assert_array_equal_array(result, true)

        # d a
        result = isopy.add(d1, a2)
        true = a1.to_scalardict() + a2
        assert_array_equal_array(result, true)

        result = isopy.add(d1, a2, 0)
        true = a1.to_scalardict(0) + a2.default(0)
        assert_array_equal_array(result, true)

        result = isopy.add(d1, a2, (0, 1))
        true = a1.to_scalardict(0) + a2.default(1)
        assert_array_equal_array(result, true)

        # a ds
        result = isopy.add(a1, ds2)
        true = a1 + ds2
        assert_array_equal_array(result, true)

        result = isopy.add(a1, ds2, 0)
        true = a1.default(0) + ds2
        assert_array_equal_array(result, true)

        result = isopy.add(a1, ds2, (0, 1))
        true = a1.default(0) + ds2
        assert_array_equal_array(result, true)

        # ds a
        result = isopy.add(ds1, a2)
        true = ds1 + a2
        assert_array_equal_array(result, true)

        result = isopy.add(ds1, a2, 0)
        true = ds1 + a2.default(0)
        assert_array_equal_array(result, true)

        result = isopy.add(ds1, a2, (0, 1))
        true = ds1 + a2.default(1)
        assert_array_equal_array(result, true)

        # keys
        result = isopy.add(a1, ds2, keys=('ru', 'cd'))
        assert result.keys == ('ru', 'cd')

    def test_count_finite(self):
        array = isopy.array([[1, 2, 3], [4, np.nan, 6]], 'ru pd cd'.split())
        answer = isopy.array((2, 1, 2), 'ru pd cd'.split())

        assert_array_equal_array(isopy.nancount(array), answer)
        np.testing.assert_allclose(isopy.nancount(array, axis=1), [3, 2])
        assert isopy.nancount(array, axis=None) == 5

        data = [[1, 2, 3], [4, np.nan, 6]]
        answer = (2, 1, 2)

        assert isopy.nancount(data) == 5
        np.testing.assert_allclose(isopy.nancount(data, axis=0), answer)
        np.testing.assert_allclose(isopy.nancount(data, axis=1), [3, 2])

    def test_keyminmax(self):
        array = isopy.random(10, [3, 1, 5], 'ru pd cd'.split(), seed=46)
        array['ru'][1] = 100
        array['ru'][2] = -2

        assert isopy.keymax(array) == 'cd'
        assert isopy.keymax(array, np.mean) == 'ru'

        assert isopy.keymin(array) == 'pd'
        assert isopy.keymin(array, np.min) == 'ru'

    def test_where(self):
        data = [[1, 2, 3], [3, np.nan, 5], [6, 7, 8], [9, 10, np.nan]]
        array = isopy.array(data, 'ru pd cd'.split())

        where1 = [True, False, True, True]
        where1t = [[True, True, True], [False, False, False], [True, True, True], [True, True, True]]

        where2 = [[False, True, True], [False, False, True], [True, True, True], [True, True, False]]
        arrayw2 = isopy.array(where2, 'ru pd cd'.split(), dtype=bool)

        for func in [np.mean, np.std]:
            try:
                result = func(array, where=where1, axis=0)
                true = func(data, where=where1t, axis=0)
                np.testing.assert_allclose(result.tolist(), true)

                result = func(array, where=arrayw2, axis=0)
                true = func(data, where=where2, axis=0)
                np.testing.assert_allclose(result.tolist(), true)
            except Exception as err:
                raise AssertionError(f'func {func} test failed') from err

    def test_out(self):
        data = [[1, 2, 3], [3, np.nan, 5], [6, 7, 8], [9, 10, np.nan]]
        array = isopy.array(data, 'ru pd cd'.split())

        for func in [np.std, np.nanstd]:
            out = isopy.ones(None, 'ru pd cd'.split())
            result1 = func(array)
            result2 = func(array, out=out)
            assert_array_equal_array(result1, result2)
            assert result2 is out

            out = np.ones(4)
            result1 = func(array, axis=1)
            result2 = func(array, axis=1, out=out)
            np.testing.assert_allclose(result1, result2)
            assert result2 is out

            out = isopy.ones(None, 'ag ru cd'.split())
            result1 = func(array)
            assert result1.keys == 'ru pd cd'.split()
            result2 = func(array, out=out)
            assert result2 is out
            assert result2.keys == 'ag ru cd'.split()

            np.testing.assert_allclose(result1['ru'], result2['ru'])
            np.testing.assert_allclose(result1['cd'], result2['cd'])
            np.testing.assert_allclose(np.ones(4), result2['ag'])

    def test_special(self):
        # np.average, np.copyto
        tested = []

        # Tests for copyto
        tested.append(np.copyto)

        array1 = isopy.array([[1, 2, 3], [11, 12, 13]], 'ru pd cd'.split())
        array2 = isopy.empty(2, 'ru pd cd'.split())
        np.copyto(array2, array1)
        assert array2 is not array1
        assert_array_equal_array(array2, array1)

        dictionary = dict(ru=1, rh=1.5, pd=2, ag=2.5, cd=3)
        array1 = isopy.array([1, 2, 3], 'ru pd cd'.split())
        array2 = isopy.empty(-1, 'ru pd cd'.split())
        np.copyto(array2, dictionary)
        assert_array_equal_array(array2, array1)

        # Tests for average
        tested.append(np.average)

        # axis = 0
        array = isopy.array([[1, 2, 3], [11, 12, 13]], 'ru pd cd'.split())
        weights = [2, 1]
        correct = isopy.array((13 / 3, 16 / 3, 19 / 3), 'ru pd cd'.split())
        result = np.average(array, weights=weights)
        assert_array_equal_array(result, correct)

        result = np.average(array, 0, weights=weights)
        assert_array_equal_array(result, correct)

        weights = isopy.array((2, 0.5, 1), 'ru pd cd'.split())
        with pytest.raises(TypeError):
            np.average(array, weights=weights)

        # axis = 1
        weights = [2, 0.5, 1]
        correct = np.array([(1 * 2 + 2 * 0.5 + 3 * 1) / 3.5, (11 * 2 + 12 * 0.5 + 13 * 1) / 3.5])
        result = np.average(array, axis=1, weights=weights)
        np.testing.assert_allclose(result, correct)

        result = np.average(array, 1, weights)
        np.testing.assert_allclose(result, correct)

        # axis = None
        # shape mismatch
        weights = [2, 1]
        with pytest.raises(TypeError):
            np.average(array, axis=None, weights=weights)

        weights = isopy.array((2, 0.5, 1), 'ru pd cd'.split())
        with pytest.raises(TypeError):
            np.average(array, axis=None, weights=weights)

        # axis = 0
        array = isopy.array([[1, 2, 3], [11, 12, 13]], 'ru pd cd'.split())
        weights = [[2, 0.5, 1], [0.5, 0.5, 0.5]]
        with pytest.raises(TypeError):
            # Shape mismatch
            np.average(array, weights=weights)

        weights = isopy.array([[2, 0.5, 1], [0.5, 0.5, 0.5]], 'ru pd cd'.split())
        correct = isopy.array(((1 * 2 + 11 * 0.5) / 2.5, (2 * 0.5 + 12 * 0.5) / 1, (3 * 1 + 13 * 0.5) / 1.5),
                              'ru pd cd'.split())
        result = np.average(array, weights=weights)
        assert_array_equal_array(result, correct)

        result = np.average(array, 0, weights)
        assert_array_equal_array(result, correct)

        weights2 = isopy.IsopyDict(weights)
        result = np.average(array, weights=weights2)
        assert_array_equal_array(result, correct)

        weights2 = isopy.ScalarDict(weights)
        result = np.average(array, weights=weights2)
        assert_array_equal_array(result, correct)

        weights2 = weights.to_dict()
        with pytest.raises(Exception):
            # Not sure what will throw an error but soemthign will
            np.average(array, weights=weights2)

        # axis = 1
        weights = isopy.array([[2, 0.5, 1], [0.5, 0.5, 0.5]], 'ru pd cd'.split())
        correct = np.array([(1 * 2 + 2 * 0.5 + 3 * 1) / 3.5, (11 * 0.5 + 12 * 0.5 + 13 * 0.5) / 1.5])
        result = np.average(array, axis=1, weights=weights)
        np.testing.assert_allclose(result, correct)

        result = np.average(array, 1, weights)
        np.testing.assert_allclose(result, correct)

        weights2 = isopy.IsopyDict(weights)
        result = np.average(array, axis=1, weights=weights2)
        np.testing.assert_allclose(result, correct)

        weights2 = isopy.ScalarDict(weights)
        result = np.average(array, axis=1, weights=weights2)
        np.testing.assert_allclose(result, correct)

        weights2 = weights.to_dict()
        with pytest.raises(Exception):
            # Not sure what will throw an error but soemthign will
            np.average(array, axis=1, weights=weights2)

        weights = [[2, 0.5, 1], [0.5, 1, 2]]
        correct = np.array([(1 * 2 + 2 * 0.5 + 3 * 1) / 3.5, (11 * 0.5 + 12 * 1 + 13 * 2) / 3.5])
        result = np.average(array, axis=1, weights=weights)
        np.testing.assert_allclose(result, correct)

        result = np.average(array, 1, weights)
        np.testing.assert_allclose(result, correct)

        # axis = None
        weights = isopy.array([[2, 0.5, 1], [0.5, 0.5, 0.5]], 'ru pd cd'.split())
        correct = (1 * 2 + 2 * 0.5 + 3 * 1 + 11 * 0.5 + 12 * 0.5 + 13 * 0.5) / 5
        result = np.average(array, axis=None, weights=weights)
        np.testing.assert_allclose(result, correct)

        result = np.average(array, None, weights)
        np.testing.assert_allclose(result, correct)

        weights2 = isopy.IsopyDict(weights)
        result = np.average(array, axis=None, weights=weights2)
        np.testing.assert_allclose(result, correct)

        weights2 = isopy.ScalarDict(weights)
        result = np.average(array, axis=None, weights=weights2)
        np.testing.assert_allclose(result, correct)

        weights2 = weights.to_dict()
        with pytest.raises(Exception):
            # Not sure what will throw an error but soemthign will
            np.average(array, axis=None, weights=weights2)

        weights = [[2, 0.5, 1], [0.5, 0.5, 0.5]]
        correct = (1 * 2 + 2 * 0.5 + 3 * 1 + 11 * 0.5 + 12 * 0.5 + 13 * 0.5) / 5
        result = np.average(array, axis=None, weights=weights)
        np.testing.assert_allclose(result, correct)

        result = np.average(array, None, weights)
        np.testing.assert_allclose(result, correct)

        # Make sure functions are tested.
        for func in array_functions.np_special:
            if func not in tested:
                raise ValueError(f'special function {func.__name__} not tested')

    def test_concatenate_1(self):
        # Axis = 0

        array1 = isopy.ones(2, 'ru pd cd'.split())
        array2 = isopy.ones(-1, 'pd ag107 cd'.split()) * 2
        array3 = isopy.ones(2, '101ru ag107'.split()) * 3

        # Concatenate
        result = isopy.concatenate(array1)
        assert result is not array1
        assert_array_equal_array(result, array1)

        result = isopy.concatenate(array1, array2, array3)

        keys = array1.keys | array2.keys | array3.keys
        assert result.keys == keys
        for key in keys:
            true = np.concatenate((array1.get(key), [array2.get(key)], array3.get(key)))
            np.testing.assert_allclose(result[key], true)

        result = isopy.concatenate(array1, np.nan, array3, axis=0)
        keys = array1.keys + array3.keys
        assert result.keys == keys
        for key in keys:
            true = np.concatenate((array1.get(key), [np.nan], array3.get(key)))
            np.testing.assert_allclose(result[key], true)

        # rstack
        result = isopy.rstack(array1)
        assert result is not array1
        assert_array_equal_array(result, array1)

        result = isopy.rstack(array1, array2, array3)

        keys = array1.keys | array2.keys | array3.keys
        assert result.keys == keys
        for key in keys:
            true = np.concatenate((array1.get(key), [array2.get(key)], array3.get(key)))
            np.testing.assert_allclose(result[key], true)

        result = isopy.rstack(array1, np.nan, array3)
        keys = array1.keys + array3.keys
        assert result.keys == keys
        for key in keys:
            true = np.concatenate((array1.get(key), [np.nan], array3.get(key)))
            np.testing.assert_allclose(result[key], true)

        # Axis = 1
        array1 = isopy.ones(2, 'ru pd cd'.split())
        array2 = isopy.ones(-1, 'rh ag'.split()) * 2
        array3 = isopy.ones(2, '103rh 107ag'.split()) * 3
        array4 = isopy.ones(3, '103rh 107ag'.split()) * 4

        result = isopy.concatenate(array1, axis=1)
        assert result is not array1
        assert_array_equal_array(result, array1)

        result = isopy.concatenate(array1, array2, array3, axis=1)
        assert result.keys == array1.keys + array2.keys + array3.keys
        for key, v in array1.items():
            np.testing.assert_allclose(result[key], v)
        for key, v in array2.items():
            np.testing.assert_allclose(result[key], v.repeat(2))
        for key, v in array3.items():
            np.testing.assert_allclose(result[key], v)

        with pytest.raises(ValueError):
            isopy.concatenate(array1, np.nan, array3, axis=1)

        # cstack
        result = isopy.cstack(array1)
        assert result is not array1
        assert_array_equal_array(result, array1)

        result = isopy.cstack(array1, array2, array3)
        assert result.keys == array1.keys + array2.keys + array3.keys
        for key, v in array1.items():
            np.testing.assert_allclose(result[key], v)
        for key, v in array2.items():
            np.testing.assert_allclose(result[key], v.repeat(2))
        for key, v in array3.items():
            np.testing.assert_allclose(result[key], v)

        with pytest.raises(ValueError):
            isopy.cstack(array1, np.nan, array3)

        with pytest.raises(ValueError):
            isopy.cstack(array3, array4)

        # invalid axis

        with pytest.raises(np.AxisError):
            result = isopy.concatenate(array1, array2, array3, axis=2)

    def test_cocatenate_2(self):
        array1 = isopy.ones(2) * 4
        array2 = isopy.ones(2) * 5
        array1 = array1.reshape((1, -1))
        array2 = array2.reshape((1, -1))

        true = np.concatenate([array1, array2], axis=0)
        result = isopy.concatenate(array1, array2)
        assert not isinstance(result, core.IsopyArray)
        np.testing.assert_allclose(result, true)

        true = np.concatenate([array1, array2], axis=1)
        result = isopy.concatenate(array1, array2, axis=1)
        assert not isinstance(result, core.IsopyArray)
        np.testing.assert_allclose(result, true)


def test_allowed_numpy_functions():
        # These are non-vital so just make sure they return a string
        result = array_functions.approved_numpy_functions()
        assert isinstance(result, str)

        result = array_functions.approved_numpy_functions('name')
        assert isinstance(result, str)

        result = array_functions.approved_numpy_functions('link')
        assert isinstance(result, str)

        result = array_functions.approved_numpy_functions('rst')
        assert isinstance(result, str)

        result = array_functions.approved_numpy_functions('markdown')
        assert isinstance(result, str)

        # this returns a list
        result = array_functions.approved_numpy_functions(delimiter=None)
        assert isinstance(result, list)
        assert False not in [isinstance(string, str) for string in result]