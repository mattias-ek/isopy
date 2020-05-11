import isopy
import numpy as np
import pytest

def test_sd():
    keys = ['Ru', 'Pd', 'Cd']
    data = [1,2,3]
    array = isopy.array(data, keys)

    for axis in [None, 1]:
        correct = np.std(data, ddof=1)
        test = isopy.tb.sd(array, axis=axis)
        assert not isinstance(test, isopy.IsopyArray)
        np.testing.assert_allclose(test, correct, 1E-10)
    test = isopy.tb.sd(data)
    assert not isinstance(test, isopy.IsopyArray)
    np.testing.assert_allclose(test, correct, 1E-10)
    with pytest.raises(Exception):
        test = isopy.tb.sd(data, axis=1)

    correct = np.std([data], ddof=1, axis=0)
    test = isopy.tb.sd(array, axis=0)
    for i, key in enumerate(keys):
        np.testing.assert_allclose(test[key], correct[i], 1E-10)
    correct = np.std(data, ddof=1, axis=0)
    test = isopy.tb.sd(data, axis=0)
    assert not isinstance(test, isopy.IsopyArray)
    np.testing.assert_allclose(test, correct, 1E-10)

    data = [1, 2, np.nan]
    array = isopy.array(data, keys)

    for axis in [None, 1]:
        correct = np.std(data, ddof=1)
        test = isopy.tb.sd(array, axis=axis)
        assert not isinstance(test, isopy.IsopyArray)
        np.testing.assert_allclose(test, correct, 1E-10)
    test = isopy.tb.sd(data)
    assert not isinstance(test, isopy.IsopyArray)
    np.testing.assert_allclose(test, correct, 1E-10)
    with pytest.raises(Exception):
        test = isopy.tb.sd(data, axis=1)

    correct = np.std([data], ddof=1, axis=0)
    test = isopy.tb.sd(array, axis=0)
    for i, key in enumerate(keys):
        np.testing.assert_allclose(test[key], correct[i], 1E-10)
    correct = np.std(data, ddof=1, axis=0)
    test = isopy.tb.sd(data, axis=0)
    assert not isinstance(test, isopy.IsopyArray)
    np.testing.assert_allclose(test, correct, 1E-10)

    data = [[1, 2, np.nan], [2,3,4], [8,3,1], [11,15,13]]
    array = isopy.array(data, keys)
    for axis in [None, 1]:
        correct = np.std(data, ddof=1, axis=axis)
        test = isopy.tb.sd(array, axis=axis)
        assert not isinstance(test, isopy.IsopyArray)
        np.testing.assert_allclose(test, correct, 1E-10)
        test = isopy.tb.sd(data, axis=axis)
        assert not isinstance(test, isopy.IsopyArray)
        np.testing.assert_allclose(test, correct, 1E-10)


    correct = np.std(data, ddof=1, axis=0)
    test = isopy.tb.sd(array, axis=0)
    for i, key in enumerate(keys):
        np.testing.assert_allclose(test[key], correct[i], 1E-10)
    test = isopy.tb.sd(data, axis=0)
    assert not isinstance(test, isopy.IsopyArray)
    np.testing.assert_allclose(test, correct, 1E-10)

def test_nansd():
    keys = ['Ru', 'Pd', 'Cd']
    data = [1,2,3]
    array = isopy.array(data, keys)

    for axis in [None, 1]:
        correct = np.nanstd(data, ddof=1)
        test = isopy.tb.sd(array, axis=axis)
        assert not isinstance(test, isopy.IsopyArray)
        np.testing.assert_allclose(test, correct, 1E-10)
    test = isopy.tb.sd(data)
    assert not isinstance(test, isopy.IsopyArray)
    np.testing.assert_allclose(test, correct, 1E-10)
    with pytest.raises(Exception):
        test = isopy.tb.sd(data, axis=1)

    correct = np.std([data], ddof=1, axis=0)
    test = isopy.tb.sd(array, axis=0)
    for i, key in enumerate(keys):
        np.testing.assert_allclose(test[key], correct[i], 1E-10)
    correct = np.std(data, ddof=1, axis=0)
    test = isopy.tb.sd(data, axis=0)
    assert not isinstance(test, isopy.IsopyArray)
    np.testing.assert_allclose(test, correct, 1E-10)

    data = [1, 2, np.nan]
    array = isopy.array(data, keys)

    for axis in [None, 1]:
        correct = np.std(data, ddof=1)
        test = isopy.tb.sd(array, axis=axis)
        assert not isinstance(test, isopy.IsopyArray)
        np.testing.assert_allclose(test, correct, 1E-10)
    test = isopy.tb.sd(data)
    assert not isinstance(test, isopy.IsopyArray)
    np.testing.assert_allclose(test, correct, 1E-10)
    with pytest.raises(Exception):
        test = isopy.tb.sd(data, axis=1)

    correct = np.std([data], ddof=1, axis=0)
    test = isopy.tb.sd(array, axis=0)
    for i, key in enumerate(keys):
        np.testing.assert_allclose(test[key], correct[i], 1E-10)
    correct = np.std(data, ddof=1, axis=0)
    test = isopy.tb.sd(data, axis=0)
    assert not isinstance(test, isopy.IsopyArray)
    np.testing.assert_allclose(test, correct, 1E-10)

    data = [[1, 2, np.nan], [2,3,4], [8,3,1], [11,15,13]]
    array = isopy.array(data, keys)
    for axis in [None, 1]:
        correct = np.std(data, ddof=1, axis=axis)
        test = isopy.tb.sd(array, axis=axis)
        assert not isinstance(test, isopy.IsopyArray)
        np.testing.assert_allclose(test, correct, 1E-10)
        test = isopy.tb.sd(data, axis=axis)
        assert not isinstance(test, isopy.IsopyArray)
        np.testing.assert_allclose(test, correct, 1E-10)


    correct = np.std(data, ddof=1, axis=0)
    test = isopy.tb.sd(array, axis=0)
    for i, key in enumerate(keys):
        np.testing.assert_allclose(test[key], correct[i], 1E-10)
    test = isopy.tb.sd(data, axis=0)
    assert not isinstance(test, isopy.IsopyArray)
    np.testing.assert_allclose(test, correct, 1E-10)

def test_se():
    keys = ['Ru', 'Pd', 'Cd']
    data = [1, 2, 3]
    array = isopy.array(data, keys)

    for axis in [None, 1]:
        correct = np.std(data, ddof=1) / np.sqrt(3)
        test = isopy.tb.se(array, axis=axis)
        assert not isinstance(test, isopy.IsopyArray)
        np.testing.assert_allclose(test, correct, 1E-10)
    test = isopy.tb.se(data)
    assert not isinstance(test, isopy.IsopyArray)
    np.testing.assert_allclose(test, correct, 1E-10)
    with pytest.raises(Exception):
        test = isopy.tb.sd(data, axis=1)

    correct = np.std([data], ddof=1, axis=0) / np.sqrt(1)
    test = isopy.tb.se(array, axis=0)
    for i, key in enumerate(keys):
        np.testing.assert_allclose(test[key], correct[i], 1E-10)
    correct = np.std(data, ddof=1, axis=0) / np.sqrt(3)
    test = isopy.tb.se(data, axis=0)
    assert not isinstance(test, isopy.IsopyArray)
    np.testing.assert_allclose(test, correct, 1E-10)

    data = [1, 2, np.nan]
    array = isopy.array(data, keys)

    for axis in [None, 1]:
        correct = np.std(data, ddof=1) / np.sqrt(3)
        test = isopy.tb.se(array, axis=axis)
        assert not isinstance(test, isopy.IsopyArray)
        np.testing.assert_allclose(test, correct, 1E-10)
    test = isopy.tb.se(data) / np.sqrt(3)
    assert not isinstance(test, isopy.IsopyArray)
    np.testing.assert_allclose(test, correct, 1E-10)
    with pytest.raises(Exception):
        test = isopy.tb.sd(data, axis=1)

    correct = np.std([data], ddof=1, axis=0) / np.sqrt(1)
    test = isopy.tb.se(array, axis=0)
    for i, key in enumerate(keys):
        np.testing.assert_allclose(test[key], correct[i], 1E-10)
    correct = np.std(data, ddof=1, axis=0) / np.sqrt(3)
    test = isopy.tb.se(data, axis=0)
    assert not isinstance(test, isopy.IsopyArray)
    np.testing.assert_allclose(test, correct, 1E-10)

    data = [[1, 2, np.nan], [2, 3, 4], [8, 3, 1], [11, 15, 13]]
    array = isopy.array(data, keys)

    correct = np.std(data, ddof=1, axis=None) / np.sqrt(12)
    test = isopy.tb.se(array, axis=None)
    assert not isinstance(test, isopy.IsopyArray)
    np.testing.assert_allclose(test, correct, 1E-10)

    correct = np.std(data, ddof=1, axis=1) / np.sqrt(3)
    test = isopy.tb.se(data, axis=1)
    assert not isinstance(test, isopy.IsopyArray)
    np.testing.assert_allclose(test, correct, 1E-10)

    correct = np.std(data, ddof=1, axis=0) / np.sqrt(4)
    test = isopy.tb.se(array, axis=0)
    for i, key in enumerate(keys):
        np.testing.assert_allclose(test[key], correct[i], 1E-10)
    test = isopy.tb.se(data, axis=0)
    assert not isinstance(test, isopy.IsopyArray)
    np.testing.assert_allclose(test, correct, 1E-10)
    
def test_nanse():
    keys = ['Ru', 'Pd', 'Cd']
    data = [1, 2, 3]
    array = isopy.array(data, keys)

    for axis in [None, 1]:
        correct = np.nanstd(data, ddof=1) / np.sqrt(3)
        test = isopy.tb.nanse(array, axis=axis)
        assert not isinstance(test, isopy.IsopyArray)
        np.testing.assert_allclose(test, correct, 1E-10)
    test = isopy.tb.nanse(data)
    assert not isinstance(test, isopy.IsopyArray)
    np.testing.assert_allclose(test, correct, 1E-10)
    with pytest.raises(Exception):
        test = isopy.tb.sd(data, axis=1)

    correct = np.nanstd([data], ddof=1, axis=0) / np.sqrt(1)
    test = isopy.tb.nanse(array, axis=0)
    for i, key in enumerate(keys):
        np.testing.assert_allclose(test[key], correct[i], 1E-10)
    correct = np.nanstd(data, ddof=1, axis=0) / np.sqrt(3)
    test = isopy.tb.nanse(data, axis=0)
    assert not isinstance(test, isopy.IsopyArray)
    np.testing.assert_allclose(test, correct, 1E-10)

    data = [1, 2, np.nan]
    array = isopy.array(data, keys)

    for axis in [None, 1]:
        correct = np.nanstd(data, ddof=1) / np.sqrt(2)
        test = isopy.tb.nanse(array, axis=axis)
        assert not isinstance(test, isopy.IsopyArray)
        np.testing.assert_allclose(test, correct, 1E-10)
    test = isopy.tb.nanse(data)
    assert not isinstance(test, isopy.IsopyArray)
    np.testing.assert_allclose(test, correct, 1E-10)
    with pytest.raises(Exception):
        test = isopy.tb.sd(data, axis=1)

    correct = np.nanstd([data], ddof=1, axis=0) / np.sqrt(1)
    test = isopy.tb.nanse(array, axis=0)
    for i, key in enumerate(keys):
        np.testing.assert_allclose(test[key], correct[i], 1E-10)
    correct = np.nanstd(data, ddof=1, axis=0) / np.sqrt(2)
    test = isopy.tb.nanse(data, axis=0)
    assert not isinstance(test, isopy.IsopyArray)
    np.testing.assert_allclose(test, correct, 1E-10)

    data = [[1, 2, np.nan], [2, 3, 4], [8, 3, 1], [11, 15, 13]]
    array = isopy.array(data, keys)

    correct = np.nanstd(data, ddof=1, axis=None) / np.sqrt(11)
    test = isopy.tb.nanse(array, axis=None)
    assert not isinstance(test, isopy.IsopyArray)
    np.testing.assert_allclose(test, correct, 1E-10)

    correct = np.nanstd(data, ddof=1, axis=1) / np.sqrt([2,3,3,3])
    test = isopy.tb.nanse(data, axis=1)
    assert not isinstance(test, isopy.IsopyArray)
    np.testing.assert_allclose(test, correct, 1E-10)

    correct = np.nanstd(data, ddof=1, axis=0) / np.sqrt([4,4,3])
    test = isopy.tb.nanse(array, axis=0)
    for i, key in enumerate(keys):
        np.testing.assert_allclose(test[key], correct[i], 1E-10)
    test = isopy.tb.nanse(data, axis=0)
    assert not isinstance(test, isopy.IsopyArray)
    np.testing.assert_allclose(test, correct, 1E-10)

def test_count_notnan():
    keys = ['Ru', 'Pd', 'Cd']
    data = [1, 2, 3]
    array = isopy.array(data, keys)

    assert isopy.tb.count_notnan(array, None) == 3
    assert isopy.tb.count_notnan(array, axis=1) == 3
    test = isopy.tb.count_notnan(array, axis=0)
    for i, key in enumerate(keys):
        assert test[key] == 1

    assert isopy.tb.count_notnan(data) == 3
    assert isopy.tb.count_notnan(data, axis=0) == 3
    with pytest.raises(Exception):
        isopy.tb.count_notnan(data, axis=1)

    data = [1, 2, np.nan]
    array = isopy.array(data, keys)

    assert isopy.tb.count_notnan(array, None) == 2
    assert isopy.tb.count_notnan(array, axis=1) == 2
    test = isopy.tb.count_notnan(array, axis=0)
    correct = [1,1,0]
    for i, key in enumerate(keys):
        assert test[key] == correct[i]

    assert isopy.tb.count_notnan(data) == 2
    assert isopy.tb.count_notnan(data, axis=0) == 2
    with pytest.raises(Exception):
        isopy.tb.count_notnan(data, axis=1)

    data = [[1, 2, np.nan], [2, 3, 4], [8, 3, 1], [11, 15, 13]]
    array = isopy.array(data, keys)

    assert isopy.tb.count_notnan(array, None) == 11
    np.testing.assert_allclose(isopy.tb.count_notnan(array, axis=1), [2,3,3,3])
    test = isopy.tb.count_notnan(array, axis=0)
    correct = [4, 4, 3]
    for i, key in enumerate(keys):
        assert test[key] == correct[i]

    assert isopy.tb.count_notnan(data) == 11
    np.testing.assert_allclose(isopy.tb.count_notnan(data, axis=0), [4, 4, 3])
    np.testing.assert_allclose(isopy.tb.count_notnan(data, axis=1), [2,3,3,3])



