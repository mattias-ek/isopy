import isopy
import numpy as np
import os

import isopy.core

def filename(filename):
    return os.path.join(os.path.dirname(__file__), 'files', filename)

class Test_CSV:
    def test_dict_1d(self):
        data = dict(Pd=[1.0, 2.0, 3.0], Cd=[11.0, 12.0, 13.0], Ru=[21.0, 22.0, 23.0])

        self.run(data, data)
        self.run(data, data, keys_in='r')

        save_data = isopy.array(data)

        self.run(save_data, data)
        self.run(save_data, data, keys_in='r')

    def test_dict_0d(self):
        save_data = dict(Pd=1.0, Cd=2.0, Ru=3.0)
        data = {key: [value] for key, value in save_data.items()}

        self.run(save_data, data)
        self.run(data, data, keys_in='r')

        save_data = isopy.array(data)

        self.run(save_data, data)
        self.run(save_data, data, keys_in='r')

    def test_list_2d(self):
        data = [[1.0, 2.0, 3.0], [11.0, 12.0, 13.0], [21.0, 22.0, 23.0]]

        self.run(data, data, keys_in=None)

        save_data = np.array(data)

        self.run(save_data, data, keys_in=None)

    def test_list_1d(self):
        save_data = [1.0,2.0,3.0]
        data = [save_data]

        self.run(save_data, data, keys_in=None)

        save_data = np.array(data)

        self.run(save_data, data, keys_in=None)

    def run(self, save_data, data, **kwargs):
        isopy.write_csv(filename('test_io1.csv'), save_data, **kwargs)
        read_s = isopy.read_csv(filename('test_io1.csv'), **kwargs)
        read_f = isopy.read_csv(filename('test_io1.csv'), float_preferred=True, **kwargs)
        self.compare(data, read_s, str)
        self.compare(data, read_f, float)

        # With a comment
        isopy.write_csv(filename('test_io2.csv'), save_data, comments='This is a comment', **kwargs)
        read_s = isopy.read_csv(filename('test_io2.csv'), **kwargs)
        read_f = isopy.read_csv(filename('test_io2.csv'), float_preferred=True, **kwargs)
        self.compare(data, read_s, str)
        self.compare(data, read_f, float)

        # Several comments and different comment symbol
        isopy.write_csv(filename('test_io3.csv'), save_data, comments=['This is a comment', 'so is this'],
                        comment_symbol='%', **kwargs)
        read_s = isopy.read_csv(filename('test_io3.csv'), comment_symbol='%', **kwargs)
        read_f = isopy.read_csv(filename('test_io3.csv'), float_preferred=True, comment_symbol='%', **kwargs)
        self.compare(data, read_s, str)
        self.compare(data, read_f, float)

        if isinstance(save_data, isopy.core.IsopyArray) and kwargs.get('keys_in', 'c') == 'c':
            # to functions
            save_data.to_csv(filename('test_io3.csv'), **kwargs)
            read_s = isopy.read_csv(filename('test_io3.csv'), **kwargs)
            read_f = isopy.read_csv(filename('test_io3.csv'), float_preferred=True, **kwargs)
            self.compare(data, read_s, str)
            self.compare(data, read_f, float)
            new_array1 = isopy.array_from_csv(filename('test_io3.csv'))
            assert isinstance(new_array1, isopy.core.IsopyArray)
            assert isopy.isflavour(new_array1, save_data)
            assert new_array1.keys == save_data.keys
            for key in new_array1.keys:
                np.testing.assert_allclose(new_array1[key], save_data[key])

            # With a comment
            save_data.to_csv(filename('test_io4.csv'), comments='This is a comment', **kwargs)
            read_s = isopy.read_csv(filename('test_io4.csv'), **kwargs)
            read_f = isopy.read_csv(filename('test_io4.csv'), float_preferred=True, **kwargs)
            self.compare(data, read_s, str)
            self.compare(data, read_f, float)
            new_array1 = isopy.array_from_csv(filename('test_io4.csv'))
            assert isinstance(new_array1, isopy.core.IsopyArray)
            assert isopy.isflavour(new_array1, save_data)
            assert new_array1.keys == save_data.keys
            for key in new_array1.keys:
                np.testing.assert_allclose(new_array1[key], save_data[key])

            # Several comments. Cannot change comment symbol using to method
            save_data.to_csv(filename('test_io5.csv'),
                            comments=['This is a comment', 'so is this'], **kwargs)
            read_s = isopy.read_csv(filename('test_io5.csv'), **kwargs)
            read_f = isopy.read_csv(filename('test_io5.csv'), float_preferred=True,
                                    **kwargs)
            self.compare(data, read_s, str)
            self.compare(data, read_f, float)
            new_array1 = isopy.array_from_csv(filename('test_io5.csv'))
            assert isinstance(new_array1, isopy.core.IsopyArray)
            assert isopy.isflavour(new_array1, save_data)
            assert new_array1.keys == save_data.keys
            for key in new_array1.keys:
                np.testing.assert_allclose(new_array1[key], save_data[key])

    def compare(self, written, read, t):
        if isinstance(written, (dict, isopy.core.IsopyArray)):
            assert type(read) is dict
            for key in written.keys():
                assert str(key) in read.keys()
                assert [t(v) for v in read[key]] == [t(v) for v in written[key]]
        else:
            assert type(read) is list
            for i, row in enumerate(read):
                assert [t(v) for v in row] == [t(v) for v in written[i]]


class Test_xlsx:
    def test_dict_1d(self):
        data = dict(Pd=[1.1, 2.2, 3.3], Cd=[11.1, 12.2, 13.3], Ru=[21.1, 22.2, 23.3])
        save_data = data
        data2 = [[1.1, 11.1, 21.1], [2.2, 12.2, 22.2], [3.3, 13.3, 23.3]]

        self.run_sheet(save_data, data)
        self.run_sheetname(save_data, data)
        self.run_sheet(save_data, data, keys_in='r')
        self.run_sheetname(save_data, data, keys_in='r')
        self.run_sheet(save_data, data2, keys_in=None)
        self.run_sheetname(save_data, data2, keys_in=None)

        save_data = isopy.array(data)

        self.run_sheet(save_data, data)
        self.run_sheetname(save_data, data)
        self.run_sheet(save_data, data, keys_in='r')
        self.run_sheetname(save_data, data, keys_in='r')
        self.run_sheet(save_data, data2, keys_in=None)
        self.run_sheetname(save_data, data2, keys_in=None)

    def test_dict_0d(self):
        save_data = dict(Pd=1.1, Cd=2.2, Ru=3.3)
        data = {key: [value] for key, value in save_data.items()}

        self.run_sheet(save_data, data)
        self.run_sheetname(save_data, data)
        self.run_sheet(save_data, data, keys_in='r')
        self.run_sheetname(save_data, data, keys_in='r')

        save_data = isopy.array(data)

        self.run_sheet(save_data, data)
        self.run_sheetname(save_data, data)
        self.run_sheet(save_data, data, keys_in='r')
        self.run_sheetname(save_data, data, keys_in='r')

    def test_list_2d(self):
        data = [[1.1, 2.1, 3.1], [11.1, 12.1, 13.1], [21.1, 22.1, 23.1]]
        save_data = data

        self.run_sheet(save_data, data, keys_in=None)
        self.run_sheetname(save_data, data, keys_in=None)

        save_data = np.array(data)

        self.run_sheet(save_data, data, keys_in=None)
        self.run_sheetname(save_data, data, keys_in=None)

    def test_list_1d(self):
        save_data = [1.1, 2.1, 3.1]
        data = [save_data]

        self.run_sheet(save_data, data, keys_in=None)
        self.run_sheetname(save_data, data, keys_in=None)

        save_data = np.array(data)

        self.run_sheet(save_data, data, keys_in=None)
        self.run_sheetname(save_data, data, keys_in=None)

    def test_append(self):
        data1 = dict(Pd=[1.1, 2.2, 3.3], Cd=[11.1, 12.2, 13.3], Ru=[21.1, 22.2, 23.3])
        data2 = dict(Pd=[100.1, 200.2, 300.3], Cd=[110.1, 120.2, 130.3], Ru=[210.1, 220.2, 230.3])

        self.run_append(data1, data1, data2, data2)

        self.run_append(isopy.array(data1), data1, isopy.array(data2), data2)

    def run_sheet(self, save_data, data, **kwargs):
        isopy.write_xlsx(filename('test_io1.xlsx'), save_data, **kwargs)
        read_s = isopy.read_xlsx(filename('test_io1.xlsx'), **kwargs)
        assert type(read_s) is dict
        assert len(read_s) == 1
        assert 'sheet1' in read_s
        self.compare(data, read_s['sheet1'], float)

        read_s = isopy.read_xlsx(filename('test_io1.xlsx'), 'sheet1', **kwargs)
        self.compare(data, read_s, float)

        # With a comment
        isopy.write_xlsx(filename('test_io2.xlsx'), save_data, comments='This is a comment', **kwargs)
        read_s = isopy.read_xlsx(filename('test_io2.xlsx'), **kwargs)
        assert type(read_s) is dict
        assert len(read_s) == 1
        assert 'sheet1' in read_s
        self.compare(data, read_s['sheet1'], float)

        read_s = isopy.read_xlsx(filename('test_io2.xlsx'), 'sheet1', **kwargs)
        self.compare(data, read_s, float)

        # Several comments and different comment symbol
        isopy.write_xlsx(filename('test_io3.xlsx'), save_data, comments=['This is a comment', 'so is this'],
                         comment_symbol='%', **kwargs)
        read_s = isopy.read_xlsx(filename('test_io3.xlsx'), comment_symbol='%', **kwargs)
        assert type(read_s) is dict
        assert len(read_s) == 1
        assert 'sheet1' in read_s
        self.compare(data, read_s['sheet1'], float)

        read_s = isopy.read_xlsx(filename('test_io3.xlsx'), 'sheet1', comment_symbol='%', **kwargs)
        self.compare(data, read_s, float)

    def run_sheetname(self, save_data, data, **kwargs):
        isopy.write_xlsx(filename('test_io1.xlsx'), test_sheet = save_data, **kwargs)
        read_s = isopy.read_xlsx(filename('test_io1.xlsx'), **kwargs)
        assert type(read_s) is dict
        assert len(read_s) == 1
        assert 'test_sheet' in read_s
        self.compare(data, read_s['test_sheet'], float)

        read_s = isopy.read_xlsx(filename('test_io1.xlsx'), 'test_sheet', **kwargs)
        self.compare(data, read_s, float)

        # With a comment
        isopy.write_xlsx(filename('test_io2.xlsx'), test_sheet = save_data, comments='This is a comment', **kwargs)
        read_s = isopy.read_xlsx(filename('test_io2.xlsx'), **kwargs)
        assert type(read_s) is dict
        assert len(read_s) == 1
        assert 'test_sheet' in read_s
        self.compare(data, read_s['test_sheet'], float)

        read_s = isopy.read_xlsx(filename('test_io2.xlsx'), 'test_sheet', **kwargs)
        self.compare(data, read_s, float)

        # Several comments and different comment symbol
        isopy.write_xlsx(filename('test_io3.xlsx'), test_sheet = save_data, comments=['This is a comment', 'so is this'],
                         comment_symbol='%', **kwargs)
        read_s = isopy.read_xlsx(filename('test_io3.xlsx'), comment_symbol='%', **kwargs)
        assert type(read_s) is dict
        assert len(read_s) == 1
        assert 'test_sheet' in read_s
        self.compare(data, read_s['test_sheet'], float)

        read_s = isopy.read_xlsx(filename('test_io3.xlsx'), 'test_sheet', comment_symbol='%', **kwargs)
        self.compare(data, read_s, float)

        if isinstance(save_data, isopy.core.IsopyArray) and kwargs.get('keys_in', 'c') == 'c':
            save_data.to_xlsx(filename('test_io4.xlsx'), sheetname='test_sheet', **kwargs)
            read_s = isopy.read_xlsx(filename('test_io4.xlsx'), **kwargs)
            assert type(read_s) is dict
            assert len(read_s) == 1
            assert 'test_sheet' in read_s
            self.compare(data, read_s['test_sheet'], float)

            read_s = isopy.read_xlsx(filename('test_io4.xlsx'), 'test_sheet', **kwargs)
            self.compare(data, read_s, float)

            new_array1 = isopy.array_from_xlsx(filename('test_io4.xlsx'), sheetname='test_sheet')
            assert isinstance(new_array1, isopy.core.IsopyArray)
            assert isopy.isflavour(new_array1, save_data)
            assert new_array1.keys == save_data.keys
            for key in new_array1.keys:
                np.testing.assert_allclose(new_array1[key], save_data[key])

            # With a comment
            save_data.to_xlsx(filename('test_io5.xlsx'), sheetname='test_sheet',
                             comments='This is a comment', **kwargs)
            read_s = isopy.read_xlsx(filename('test_io5.xlsx'), **kwargs)
            assert type(read_s) is dict
            assert len(read_s) == 1
            assert 'test_sheet' in read_s
            self.compare(data, read_s['test_sheet'], float)

            read_s = isopy.read_xlsx(filename('test_io5.xlsx'), 'test_sheet', **kwargs)
            self.compare(data, read_s, float)

            new_array1 = isopy.array_from_xlsx(filename('test_io5.xlsx'), 'test_sheet')
            assert isinstance(new_array1, isopy.core.IsopyArray)
            assert isopy.isflavour(new_array1, save_data)
            assert new_array1.keys == save_data.keys
            for key in new_array1.keys:
                np.testing.assert_allclose(new_array1[key], save_data[key])

            # Several comments
            save_data.to_xlsx(filename('test_io6.xlsx'), sheetname='test_sheet',
                             comments=['This is a comment', 'so is this'],  **kwargs)
            read_s = isopy.read_xlsx(filename('test_io6.xlsx'), **kwargs)
            assert type(read_s) is dict
            assert len(read_s) == 1
            assert 'test_sheet' in read_s
            self.compare(data, read_s['test_sheet'], float)

            read_s = isopy.read_xlsx(filename('test_io6.xlsx'), 'test_sheet', **kwargs)
            self.compare(data, read_s, float)

            new_array1 = isopy.array_from_xlsx(filename('test_io6.xlsx'), sheetname='test_sheet')
            assert isinstance(new_array1, isopy.core.IsopyArray)
            assert isopy.isflavour(new_array1, save_data)
            assert new_array1.keys == save_data.keys
            for key in new_array1.keys:
                np.testing.assert_allclose(new_array1[key], save_data[key])

    def run_append(self, save_data1, data1, save_data2, data2, **kwargs):
        isopy.write_xlsx(filename('test_io1.xlsx'), save_data1, **kwargs)
        read_s = isopy.read_xlsx(filename('test_io1.xlsx'), **kwargs)
        assert type(read_s) is dict
        assert len(read_s) == 1
        assert 'sheet1' in read_s
        self.compare(data1, read_s['sheet1'], float)

        #Should overwrite sheet
        isopy.write_xlsx(filename('test_io1.xlsx'), test_sheet = save_data2, append = True, **kwargs)
        read_s = isopy.read_xlsx(filename('test_io1.xlsx'), **kwargs)
        assert type(read_s) is dict
        assert len(read_s) == 2
        assert 'sheet1' in read_s
        assert 'test_sheet' in read_s
        self.compare(data1, read_s['sheet1'], float)
        self.compare(data2, read_s['test_sheet'], float)

        #appends sheetname
        isopy.write_xlsx(filename('test_io1.xlsx'), test_sheet = save_data1, append=True, **kwargs)
        read_s = isopy.read_xlsx(filename('test_io1.xlsx'), **kwargs)
        assert type(read_s) is dict
        assert len(read_s) == 2
        assert 'sheet1' in read_s
        assert 'test_sheet' in read_s
        self.compare(data1, read_s['sheet1'], float)
        self.compare(data1, read_s['test_sheet'], float)

        if isinstance(save_data1, isopy.core.IsopyArray) and kwargs.get('keys_in', 'c') == 'c':
            save_data1.to_xlsx(filename('test_io1.xlsx'), **kwargs)
            read_s = isopy.read_xlsx(filename('test_io1.xlsx'), **kwargs)
            assert type(read_s) is dict
            assert len(read_s) == 1
            assert 'sheet1' in read_s
            self.compare(data1, read_s['sheet1'], float)

            new_array = isopy.array_from_xlsx(filename('test_io1.xlsx'), 'sheet1')
            assert new_array.keys == save_data1.keys
            for key in new_array.keys:
                np.testing.assert_allclose(new_array[key], save_data1[key])

            # Should overwrite sheet
            save_data2.to_xlsx(filename('test_io1.xlsx'), sheetname='test_sheet', append=True, **kwargs)
            read_s = isopy.read_xlsx(filename('test_io1.xlsx'), **kwargs)
            assert type(read_s) is dict
            assert len(read_s) == 2
            assert 'sheet1' in read_s
            assert 'test_sheet' in read_s
            self.compare(data1, read_s['sheet1'], float)
            self.compare(data2, read_s['test_sheet'], float)

            new_array = isopy.array_from_xlsx(filename('test_io1.xlsx'), 'sheet1')
            assert new_array.keys == save_data1.keys
            for key in new_array.keys:
                np.testing.assert_allclose(new_array[key], save_data1[key])

            new_array = isopy.array_from_xlsx(filename('test_io1.xlsx'), sheetname='test_sheet')
            assert new_array.keys == save_data2.keys
            for key in new_array.keys:
                np.testing.assert_allclose(new_array[key], save_data2[key])

            # appends sheetname
            save_data1.to_xlsx(filename('test_io1.xlsx'), sheetname='test_sheet', append=True, **kwargs)
            read_s = isopy.read_xlsx(filename('test_io1.xlsx'), **kwargs)
            assert type(read_s) is dict
            assert len(read_s) == 2
            assert 'sheet1' in read_s
            assert 'test_sheet' in read_s
            self.compare(data1, read_s['sheet1'], float)
            self.compare(data1, read_s['test_sheet'], float)

            new_array = isopy.array_from_xlsx(filename('test_io1.xlsx'), 'sheet1')
            assert new_array.keys == save_data1.keys
            for key in new_array.keys:
                np.testing.assert_allclose(new_array[key], save_data1[key])

            new_array = isopy.array_from_xlsx(filename('test_io1.xlsx'), sheetname='test_sheet')
            assert new_array.keys == save_data1.keys
            for key in new_array.keys:
                np.testing.assert_allclose(new_array[key], save_data1[key])

    def compare(self, written, read, t):
        if isinstance(written, (dict, isopy.core.IsopyArray)):
            assert type(read) is dict
            for key in written.keys():
                assert str(key) in read.keys()
                for v in read[key]: assert type(v) is t
                assert [t(v) for v in read[key]] == [t(v) for v in written[key]]
        else:
            assert type(read) is list
            for i, row in enumerate(read):
                for v in row: assert type(v) is t
                assert [t(v) for v in row] == [t(v) for v in written[i]]


class Test_exp:
    def test_read(self):
        data = isopy.read_exp(filename('palladium.exp'))
