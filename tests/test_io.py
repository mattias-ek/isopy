import isopy
import numpy as np
import os

import isopy.core


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
        isopy.write_csv('test_io1.csv', save_data, **kwargs)
        read_s = isopy.read_csv('test_io1.csv', **kwargs)
        read_f = isopy.read_csv('test_io1.csv', float_preferred=True, **kwargs)
        self.compare(data, read_s, str)
        self.compare(data, read_f, float)

        # With a comment
        isopy.write_csv('test_io2.csv', save_data, comments='This is a comment', **kwargs)
        read_s = isopy.read_csv('test_io2.csv', **kwargs)
        read_f = isopy.read_csv('test_io2.csv', float_preferred=True, **kwargs)
        self.compare(data, read_s, str)
        self.compare(data, read_f, float)

        # Several comments and different comment symbol
        isopy.write_csv('test_io3.csv', save_data, comments=['This is a comment', 'so is this'],
                        comment_symbol='%', **kwargs)
        read_s = isopy.read_csv('test_io3.csv', comment_symbol='%', **kwargs)
        read_f = isopy.read_csv('test_io3.csv', float_preferred=True, comment_symbol='%', **kwargs)
        self.compare(data, read_s, str)
        self.compare(data, read_f, float)

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

    def _test_list_2d(self):
        data = [[1.0, 2.0, 3.0], [11.0, 12.0, 13.0], [21.0, 22.0, 23.0]]
        save_data = data

        self.run_sheet(save_data, data, keys_in=None)
        self.run_sheetname(save_data, data, keys_in=None)

        save_data = np.array(data)

        self.run_sheet(save_data, data, keys_in=None)
        self.run_sheetname(save_data, data, keys_in=None)

    def _test_list_1d(self):
        save_data = [1.0, 2.0, 3.0]
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

    def run_sheet(self, save_data, data, **kwargs):
        isopy.write_xlsx('test_io1.xlsx', save_data, **kwargs)
        read_s = isopy.read_xlsx('test_io1.xlsx', **kwargs)
        assert type(read_s) is dict
        assert len(read_s) == 1
        assert 'sheet1' in read_s
        self.compare(data, read_s['sheet1'], float)

        read_s = isopy.read_xlsx('test_io1.xlsx', 'sheet1', **kwargs)
        self.compare(data, read_s, float)

        # With a comment
        isopy.write_xlsx('test_io2.xlsx', save_data, comments='This is a comment', **kwargs)
        read_s = isopy.read_xlsx('test_io2.xlsx', **kwargs)
        assert type(read_s) is dict
        assert len(read_s) == 1
        assert 'sheet1' in read_s
        self.compare(data, read_s['sheet1'], float)

        read_s = isopy.read_xlsx('test_io2.xlsx', 'sheet1', **kwargs)
        self.compare(data, read_s, float)

        # Several comments and different comment symbol
        isopy.write_xlsx('test_io3.xlsx', save_data, comments=['This is a comment', 'so is this'],
                        comment_symbol='%', **kwargs)
        read_s = isopy.read_xlsx('test_io3.xlsx', comment_symbol='%', **kwargs)
        assert type(read_s) is dict
        assert len(read_s) == 1
        assert 'sheet1' in read_s
        self.compare(data, read_s['sheet1'], float)

        read_s = isopy.read_xlsx('test_io3.xlsx', 'sheet1', comment_symbol='%', **kwargs)
        self.compare(data, read_s, float)

    def run_sheetname(self, save_data, data, **kwargs):
        isopy.write_xlsx('test_io1.xlsx', test_sheet = save_data, **kwargs)
        read_s = isopy.read_xlsx('test_io1.xlsx', **kwargs)
        assert type(read_s) is dict
        assert len(read_s) == 1
        assert 'test_sheet' in read_s
        self.compare(data, read_s['test_sheet'], float)

        read_s = isopy.read_xlsx('test_io1.xlsx', 'test_sheet', **kwargs)
        self.compare(data, read_s, float)

        # With a comment
        isopy.write_xlsx('test_io2.xlsx', test_sheet = save_data, comments='This is a comment', **kwargs)
        read_s = isopy.read_xlsx('test_io2.xlsx', **kwargs)
        assert type(read_s) is dict
        assert len(read_s) == 1
        assert 'test_sheet' in read_s
        self.compare(data, read_s['test_sheet'], float)

        read_s = isopy.read_xlsx('test_io2.xlsx', 'test_sheet', **kwargs)
        self.compare(data, read_s, float)

        # Several comments and different comment symbol
        isopy.write_xlsx('test_io3.xlsx', test_sheet = save_data, comments=['This is a comment', 'so is this'],
                         comment_symbol='%', **kwargs)
        read_s = isopy.read_xlsx('test_io3.xlsx', comment_symbol='%', **kwargs)
        assert type(read_s) is dict
        assert len(read_s) == 1
        assert 'test_sheet' in read_s
        self.compare(data, read_s['test_sheet'], float)

        read_s = isopy.read_xlsx('test_io3.xlsx', 'test_sheet', comment_symbol='%', **kwargs)
        self.compare(data, read_s, float)

    def run_append(self, save_data1, data1, save_data2, data2, **kwargs):
        isopy.write_xlsx('test_io1.xlsx', save_data1, **kwargs)
        read_s = isopy.read_xlsx('test_io1.xlsx', **kwargs)
        assert type(read_s) is dict
        assert len(read_s) == 1
        assert 'sheet1' in read_s
        self.compare(data1, read_s['sheet1'], float)

        #Should overwrite sheet
        isopy.write_xlsx('test_io1.xlsx', save_data2, append = True, **kwargs)
        read_s = isopy.read_xlsx('test_io1.xlsx', **kwargs)
        assert type(read_s) is dict
        assert len(read_s) == 1
        assert 'sheet1' in read_s
        self.compare(data2, read_s['sheet1'], float)

        #appends sheetname
        isopy.write_xlsx('test_io1.xlsx', test_sheet = save_data1, append=True, **kwargs)
        read_s = isopy.read_xlsx('test_io1.xlsx', **kwargs)
        assert type(read_s) is dict
        assert len(read_s) == 2
        assert 'sheet1' in read_s
        assert 'test_sheet' in read_s
        self.compare(data2, read_s['sheet1'], float)
        self.compare(data1, read_s['test_sheet'], float)

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