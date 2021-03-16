import os as _os
from isopy import core

import csv as csv
import datetime as dt
import numpy as np
import chardet
import openpyxl
from openpyxl import load_workbook
import functools
import itertools
import io
import pkg_resources as _pkgr
from typing import TypeVar, Union, Optional, Any, NoReturn, NamedTuple, Literal, Callable
from collections.abc import Sequence, Iterable

__all__ = ['read_exp', 'read_csv', 'write_csv', 'read_xlsx', 'write_xlsx']

import isopy.checks

################
### read exp ###
################
class NeptuneData:
    """
    Container for the data returned by ``read_exp``.
    """
    def __init__(self, info, cycle, time, isotope_data, ratio_data, other_data):
        self.info = info
        self.cycle = cycle
        self.time = time
        self.isotope_data = isotope_data
        self.ratio_data = ratio_data
        self.other_data = other_data

def read_exp(filename, rename = None) -> NeptuneData:
    """
    Load data from a Neptune/Triton export file.

    Parameters
    ----------
    filename : str, bytes
        Path for file to be opened. Alternatively a file like byte string can be supplied.
    rename : dict, Callable, Optional
        For renaming keys in the analysed data. Useful for cases when the key is the mass rather
        than the isotope measured. If a dictionary is passed then every key present in the
        dictionary will be replaced by the associated value. A callable can also be passed that
        takes one key in the file and returns the new key.


    Returns
    -------
    neptune_data : NeptuneData
        An object containing the following attributes:

        * info - Dictionary containing the metadata included at the beginning of the file.
        * cycle - A list containing the cycle number for each measurement.
        * time - A list containing datetime objects for each measurement.
        * isotope_data - A dictionary containing an isotope array, with all the data with an isotope keystring, for each line measured. Static measurements are always given as line ``1``.
        * ratio_data - A dictionary containing an ratio array, with all the data with an ratio keystring, for each line measured. Static measurements are always given as line ``1``.
        * other_data - A dictionary containing an general array, with all leftover data, for each line measured. Static measurements are always given as line ``1``.
    """
    information = {}

    if type(filename) is not bytes:
        fileio = open(filename, 'rb')
        file = fileio.read()
        fileio.close()
    else:
        file = filename

    encoding = chardet.detect(file).get('encoding')
    file = file.decode(encoding)

    dialect = csv.Sniffer().sniff(file)

    csv_reader = csv.reader(io.StringIO(file), dialect=dialect)
    for row in csv_reader:
        if ':' in row[0]:  # mete
            name, value = row[0].split(':', 1)
            information[name] = value

        if row[0] == 'Cycle':
            # We can reuse the function from before to read the integration data
            data = _read_csv_ckeys(row, csv_reader, float_prefered=False, termination_symbol='***')

    if rename is None:
        renamer = lambda name: name
    elif isinstance(rename, dict):
        renamer = lambda name: rename.get(name, name)
    elif callable(rename):
        renamer = rename
    else:
        raise TypeError('rename must be a dict or a callable function')

    cycle = np.array(data.pop('Cycle'), dtype ='int')
    time = data.pop('Time')
    time =[dt.datetime.strptime(time[i], '%H:%M:%S:%f') for i in range(len(time))]
    isotope_data = {}
    ratio_data = {}
    other_data = {}

    for key in data:
        if ":" in key:
            line, text = key.split(":", 1)
            line = int(line)
        else:
            line = 1
            text = key
        text = renamer(text)

        try:
            text = isopy.IsotopeKeyString(text)
        except:
            pass
        else:
            isotope_data.setdefault(line, {})[text] == data[key]
            continue

        try:
            text = isopy.RatioKeyString(text)
        except:
            pass
        else:
            ratio_data.setdefault(line, {})[text] == data[key]
            continue

        other_data.setdefault(line, {})[text] = data[key]

    isotope_data = {line: isopy.core.IsotopeArray(isotope_data[line]) for line in isotope_data}
    ratio_data = {line: isopy.core.RatioArray(ratio_data[line]) for line in ratio_data}
    other_data = {line: isopy.core.GeneralArray(other_data[line]) for line in other_data}
    return NeptuneData(information, cycle, time, isotope_data, ratio_data, other_data)

######################
### read/write CSV ###
######################
def read_csv(filename, comment_symbol ='#', keys_in = 'c',
             float_preferred = False, encoding = None,
             dialect = None ):
    """
    Load data from a csv file.

    Parameters
    ----------
    filename : str, bytes
        Path for file to be opened. Alternatively a file like byte string can be supplied.
    comment_symbol : str, Default = '#'
        Rows starting with this string will be ignored.
    keys_in : {'c', 'r', None}, Default = 'c'
        If keys are given as the first value in each column pass ``c``. If keys are given as the
        first value in each row pass ``r``. If there are no keys pass ``None``.
    float_preferred : bool, Default = False
        If `True` all values will be converted to float if possible. If conversion fails the
        value will be left as a string.
    encoding
        Encoding of the file. If None the encoding will be guessed from the file.
    dialect
        Dialect of the csv file. If None the dialect will be guessed from the file.

    Returns
    -------
    data : dict or list
        Returns a dictionary for data with keys otherwise a list.
    """
    filename = isopy.checks.check_type('filename', filename, str, bytes)
    comment_symbol = isopy.checks.check_type('comment_symbol', comment_symbol, str)
    keys_in = isopy.checks.check_type('keys_in', keys_in, str,allow_none=True)
    encoding = isopy.checks.check_type('encoding', encoding, str, allow_none=True)
    dialect = isopy.checks.check_type('dialect', dialect, str, allow_none=True)

    #If filename is a string load the files.
    if type(filename) is not bytes:
        fileio = open(filename, 'rb')
        file = fileio.read()
        fileio.close()
    else:
        file = filename

    #find the files encoding
    if encoding is None:
        encoding = chardet.detect(file).get('encoding')

    #Decode the bytes into string.
    file = file.decode(encoding)

    #find the csv dialect
    if dialect is None:
        dialect = csv.Sniffer().sniff(file)

    # Create a reader object by converting the file string to a file-like object
    csv_reader = csv.reader(io.StringIO(file), dialect=dialect)
    for row in csv_reader:
        row_data = [r.strip() for r in
                    row]  # Remove any training whitespaces from the data in this row

        if row_data.count('') == len(row):
            # All the columns in this row are empty. Ignore it
            continue

        if comment_symbol is not None and row[0].startswith(comment_symbol):
            # This is a comment so ignore it.
            continue

        if keys_in == 'c':
            return _read_csv_ckeys(row_data, csv_reader, comment_symbol, float_prefered=float_preferred)
        elif keys_in == 'r':
            return _read_csv_rkeys(row_data, csv_reader, comment_symbol, float_prefered=float_preferred)
        elif keys_in is None:
            return _read_csv_nokeys(row_data, csv_reader, comment_symbol, float_prefered=float_preferred)
        else:
            raise ValueError(f'Unknown value for "keys_in" {keys_in}')


def _read_csv_ckeys(first_row, reader, comment_symbol=None, termination_symbol=None, float_prefered = False):
    keys = first_row  # First row contains the the name of each column
    values = [[] for i in range(len(keys))]  # Create an empty list for each key

    # Loop over the remaining rows
    for i, row in enumerate(reader):
        row = [r.strip() for r in row]

        if termination_symbol is not None and row[0].startswith(termination_symbol):
            # Stop reading data if we find this string at the beginning of a row
            break

        if comment_symbol is not None and row[0].startswith(comment_symbol):
            # Row is a comment, ignore
            continue

        if row.count('') == len(row):
            # Empty row, ignore
            continue

        if len(row) < len(keys):
            # There is not enough values in this row to give one to each key
            raise ValueError(f'Row {i} does not have enough columns')

        for j in range(len(keys)):
            value = row[j]

            if float_prefered:
                try:
                    value = float(value)
                except:
                    value = str(value)

            values[j].append(value)

    return dict(zip(keys, values))  # creates a dictionary from two lists


def _read_csv_rkeys(first_row, reader, comment_symbol=None, termination_symbol=None,
                    float_prefered=False):
    data = {}
    dlen = None

    # Loop over the remaining rows
    for i, row in enumerate(itertools.chain([first_row], reader)):
        row = [r.strip() for r in row]

        if termination_symbol is not None and row[0].startswith(termination_symbol):
            # Stop reading data if we find this string at the beginning of a row
            break

        if comment_symbol is not None and row[0].startswith(comment_symbol):
            # Row is a comment, ignore
            continue

        if row.count('') == len(row):
            # Empty row, ignore
            continue

        data[row[0]] = []
        if dlen is None: dlen = len(row)
        elif dlen != len(row):
            raise ValueError('Rows have different number of values')

        for j in range(1, len(row)):
            value = row[j]

            if float_prefered:
                try:
                    value = float(value)
                except:
                    value = str(value)

            data[row[0]].append(value)

    return data


def _read_csv_nokeys(first_row, reader, comment_symbol=None, termination_symbol=None,
                    float_prefered=False):
    data = []
    dlen = None

    # Loop over the remaining rows
    for i, row in enumerate(itertools.chain([first_row], reader)):
        row = [r.strip() for r in row]

        if termination_symbol is not None and row[0].startswith(termination_symbol):
            # Stop reading data if we find this string at the beginning of a row
            break

        if comment_symbol is not None and row[0].startswith(comment_symbol):
            # Row is a comment, ignore
            continue

        if row.count('') == len(row):
            # Empty row, ignore
            continue

        data.append([])
        if dlen is None:
            dlen = len(row)
        elif dlen != len(row):
            raise ValueError('Rows have different number of values')

        for j in range(len(row)):
            value = row[j]

            if float_prefered:
                try:
                    value = float(value)
                except:
                    value = str(value)

            data[-1].append(value)

    return data


def write_csv(filename, data, comments=None, keys_in='c',
              delimiter = ',', comment_symbol = '#' ) -> None:
    """
    Save data to a csv file.

    Parameters
    ----------
    filename : str
        Path/name of the csv file to be created. Any existing file with the same path/name will be overwritten.
    data : isopy_array_like, numpy_array_like
        Data to be saved in the array.
    comments : str, Sequence[str], Optional
        Comments to be included at the top of the file
    keys_in : {'c', 'r'}, Default = 'c'
        Only applicable for isopy arrays like *data*. For keys as the first value in each column
        pass ``"c"``. For keys as the first value in each row pass ``"r"``.
    delimiter : str, Default = ','
        This string will be used to separate values in the file
    comment_symbol : str, Default = '#'
        This string will precede any comments at the beginning of the file.
    """
    filename = isopy.checks.check_type('filename', filename, str)
    comments = isopy.checks.check_type_list('comments', comments, str, allow_none=True, make_list=True)
    keys_in = isopy.checks.check_type('keys_in', keys_in, str, allow_none=True)
    delimiter = isopy.checks.check_type('delimiter', delimiter, str)
    comment_symbol = isopy.checks.check_type('comment_symbol', comment_symbol, str)

    data = isopy.asanyarray(data)
    if isinstance(data, core.IsopyArray):
        if data.ndim == 0:
            data = data.reshape(-1)

        dsize = data.nrows
        csize = data.ncols
        ndim = data.ndim
        data = data.to_dict()

        if keys_in == 'c':
            func = _write_csv_ckeys
            csize = csize - 1
        elif keys_in == 'r':
            func = _write_csv_rkeys
            csize = dsize
        elif keys_in is None:
            func = _write_csv_nokeys
            data = np.array(data.to_list())
            csize = csize - 1
    else:
        ndim = data.ndim
        if ndim == 0:
            data = data.reshape((1,1))
        elif ndim == 1:
            data = data.reshape((1, -1))
        elif ndim > 2:
            raise ValueError('Cannot save data with more than two dimensions')
        dsize = data.shape[0]
        csize = data.shape[1] - 1
        func = _write_csv_nokeys

    with open(filename, mode='w', newline='') as file:
        csv_writer = csv.writer(file, delimiter=delimiter)

        if comments is not None:
            for comment in comments:
                csv_writer.writerow([f'{comment_symbol}{comment}'] + ['' for i in range(csize)])

        func(csv_writer, data, dsize)

def _write_csv_ckeys(csv_writer, data, dsize):
    keys = data.keys()

    csv_writer.writerow([f'{key}' for key in keys])

    for i in range(dsize):
        csv_writer.writerow([f'{data[key][i]}' for key in keys])

def _write_csv_rkeys(csv_writer, data, dsize):
    keys = data.keys()
    for key in keys:
        csv_writer.writerow([f'{key}'] + [f'{data[key][i]}' for i in range(dsize)])

def _write_csv_nokeys(csv_writer, data, dsize):
    for i in range(data.shape[0]):
        csv_writer.writerow([f'{data[i][j]}' for j in range(data.shape[-1])])

#######################
### Read/write xlsx ###
#######################
def read_xlsx(filename, sheetname=None, keys_in='c',
              default_value = np.nan, comment_symbol='#'):
    """
    Load data from an excel file.

    Parameters
    ----------
    filename : str
        Path for file to be opened.
    sheetname : str, int, Optional
        To load data from a single sheet in the workbook pass either the name of the sheet or
        the position of the sheet. If nothing is specified all the data for all sheets in the
        workbook is returned.
    keys_in : {'c', 'r', None}, Default = 'c'
        If keys are given as the first value in each column pass ``c``. If keys are given as the
        first value in each row pass ``r``. If there are no keys pass ``None``.
    default_value : scalar, Default = np.nan
        Value substituted for empty cells or cells containing errors
    comment_symbol : str, Default = '#'
        Rows starting with this string will be ignored.

    Returns
    -------
    data : dict
        A dictionary containing the data for a single sheet or a dictionary containing the data
        for multiple sheets in the workbook indexed by sheet name.
    """
    workbook = openpyxl.load_workbook(filename, data_only=True, read_only=True)

    sheetnames = workbook.sheetnames

    if sheetname is None:
        sheet_data = {}
        for sheetname in sheetnames:
            sheet_data[sheetname] = _read_xlsx_sheet(workbook[sheetname], keys_in, comment_symbol, default_value)
        return sheet_data

    elif type(sheetname) is int:
        return _read_xlsx_sheet(workbook[sheetnames[sheetname]], keys_in, comment_symbol, default_value)

    elif sheetname in sheetnames:
        return _read_xlsx_sheet(workbook[sheetname], keys_in, comment_symbol, default_value)

    else:
        raise ValueError(f'sheetname "{sheetname}" not found in workbook"')


def _read_xlsx_sheet(worksheet, keys_in, comment_symbol, default_value):
    for ri in range(1, worksheet.max_row + 1):
        cell = worksheet.cell(ri, 1)
        if cell.data_type == 's' and cell.value[:len(comment_symbol)] == comment_symbol:
            continue

        if keys_in == 'c':
            return _read_xlsx_sheet_ckeys(worksheet, ri, default_value)
        elif keys_in == 'r':
            return _read_xlsx_sheet_rkeys(worksheet, ri, default_value)
        elif keys_in is None:
            return _read_xlsx_sheet_nokeys(worksheet, ri, default_value)
        else:
            raise ValueError(f'Unknown value for "keys_in" {keys_in}')

def _read_xlsx_sheet_ckeys(worksheet, row_index, default_value):
    data = {}
    ERROR = object()

    # Iterate over each column
    for ci in range(1, worksheet.max_column + 1):
        # First value is the key
        key = f'{worksheet.cell(row_index, ci).value}'.strip()

        if key == '':
            continue

        data[key] = []

        for ri in range(row_index + 1, worksheet.max_row + 1):
            cell = worksheet.cell(ri, ci)
            if cell.data_type == 'e':
                data[key].append(ERROR)
            elif cell.data_type == 's':
                value = cell.value.strip()
                if value == '':
                    data[key].append(None)
                else:
                    data[key].append(value)
            else:
                data[key].append(cell.value)

    keys = list(data.keys())
    for i in range(len(data[keys[0]])):
        is_none = [(data[key][i] is None) for key in keys]
        if is_none.count(True) == len(keys):
            for key in keys:
                data[key].pop(i)
        else:
            for j, key in enumerate(keys):
                if is_none[j] or data[key][i] is ERROR:
                    data[key][i] = default_value
    return data

def _read_xlsx_sheet_rkeys(worksheet, row_index, default_value):
    data = {}
    ERROR = object()

    # Iterate over each column
    for ri in range(row_index, worksheet.max_row + 1):
        # First value is the key
        key = f'{worksheet.cell(ri, 1).value}'.strip()

        if key == '':
            continue

        data[key] = []

        for ci in range(2, worksheet.max_column + 1):
            cell = worksheet.cell(ri, ci)
            if cell.data_type == 'e':
                data[key].append(ERROR)
            elif cell.data_type == 's':
                value = cell.value.strip()
                if value == '':
                    data[key].append(None)
                else:
                    data[key].append(value)
            else:
                data[key].append(cell.value)

    keys = list(data.keys())
    for i in range(len(data[keys[0]])):
        is_none = [(data[key][i] is None) for key in keys]
        if is_none.count(True) == len(keys):
            for key in keys:
                data[key].pop(i)
        else:
            for j, key in enumerate(keys):
                if is_none[j] or data[key][i] is ERROR:
                    data[key][i] = default_value
    return data

def _read_xlsx_sheet_nokeys(worksheet, row_index, default_value):
    data = []
    ERROR = object()

    # Iterate over each column
    for ri in range(row_index, worksheet.max_row + 1):
        # First value is the key
        data.append([])

        for ci in range(1, worksheet.max_column + 1):
            cell = worksheet.cell(ri, ci)
            if cell.data_type == 'e':
                data[-1].append(ERROR)
            elif cell.data_type == 's':
                value = cell.value.strip()
                if value == '':
                    data[-1].append(None)
                else:
                    data[-1].append(value)
            else:
                data[-1].append(cell.value)

        if data[-1].count(None) == len(data[-1]):
            data.pop(-1)
        else:
            for i, v in enumerate(data[-1]):
                if v is None or v is ERROR:
                    data[-1][i] = default_value
    return data

#TODO change np.nan values to '=NA()'
def write_xlsx(filename, *sheets, comments = None, comment_symbol= '#',
               keys_in= 'c', append = False, **sheetnames):
    """
    Save data to an excel file.

    Parameters
    ----------
    filename : str
        Path/name of the excel file to be created. Any existing file with the same path/name
        will be overwritten.
    sheets : isopy_array_like, numpy_array_like
        Data arrays given here will be saved as sheet1, sheet2 etc.
    comments : str, Sequence[str], Optional
        Comments to be included at the top of the file
    comment_symbol : str, Default = '#'
        This string will precede any comments at the beginning of the file
    keys_in : {'c', 'r'}, Default = 'c'
        Only applicable for isopy arrays like *data*. For keys as the first value in each column
        pass ``"c"``. For keys as the first value in each row pass ``"r"``.
    append : bool, Default = False
        If ``True`` and *filename* exists it will append the data to this workbook. An exception
        is raised if *filename* is not a valid excel workbook.
    sheets : isopy_array, numpy_array
        Data array given here will be saved in the workbook with the keyword as the sheet name
    """
    if append and _os.path.exists(filename):
        workbook = openpyxl.load_workbook(filename=filename)
    else:
        workbook = openpyxl.Workbook()
        workbook.remove(workbook.active)

    sheetname_data = {f'sheet{i+1}': data for i, data in enumerate(sheets)}
    sheetname_data.update(sheetnames)
    try:
        for sheetname, data in sheetname_data.items():
            if sheetname in workbook.sheetnames:
                workbook.remove(workbook[sheetname])

            worksheet = workbook.create_sheet(sheetname)
            _write_xlsx(worksheet, data, comments, comment_symbol, keys_in)

        if len(workbook.sheetnames) == 0:
            workbook.create_sheet('sheet1')

        workbook.save(filename)
    finally:
        workbook.close()

def _write_xlsx(worksheet, data, comments, comment_symbol, keys_in):
    data = isopy.asanyarray(data)

    ri = 1
    if comments is not None:
        if type(comments) is not list:
            comments = [comments]
        for comment in comments:
            worksheet.cell(ri, 1).value = f'{comment_symbol}{comment}'
            ri += 1

    if isinstance(data, isopy.core.IsopyArray):
        if data.ndim == 0:
            data = data.reshape(-1)
        if keys_in == 'c':
            data = data.to_dict()
            _write_xlsx_ckeys(worksheet, data, ri)
        elif keys_in == 'r':
            data = data.to_dict()
            _write_xlsx_rkeys(worksheet, data, ri)
        elif keys_in is None:
            data = np.array(data.to_list())
            _write_xlsx_nokeys(worksheet, data, ri)
    else:
        ndim = data.ndim
        if ndim == 0:
            data = data.reshape((1, 1))
        elif ndim == 1:
            data = data.reshape((1, -1))
        elif ndim > 2:
            raise ValueError('Cannot save data with more than two dimensions')
        _write_xlsx_nokeys(worksheet, data, ri)


def _write_xlsx_ckeys(worksheet, data, ri):
    for ci, key in enumerate(data.keys()):
        worksheet.cell(ri, ci + 1).value = f'{key}'

        for rj, value in enumerate(data[key]):
            worksheet.cell(ri + rj + 1, ci + 1).value = value

def _write_xlsx_rkeys(worksheet, data, ri):
    for ci, key in enumerate(data.keys()):
        worksheet.cell(ri, 1).value = f'{key}'

        for rj, value in enumerate(data[key]):
            worksheet.cell(ri, 2 + rj).value = value
        ri += 1

def _write_xlsx_nokeys(worksheet, data, ri):
    for i, row in enumerate(data):
        for rj, value in enumerate(row):
            worksheet.cell(ri + i, 1 + rj).value = value

#################################
### Old versions of functions ###
#################################
def _read_excel(filename, sheetname = None, comment_symbol='#', default_value='nan'):
    """
    Read data from sheets in an excel file.

    The first row of the sheet will be used as the key and the subsequent rows will be taken as the value.

    Parameters
    ----------
    filename : str
        Location of the file to be opened.
    sheetname : int, str, None, optional
        If ``int`` then the sheet at this index will be returned. If ``str`` then the sheet with this name will be
        returned. If ``None`` a dict containing the data for all the sheets in the array will be returned.
    comment_symbol : str, optional
        Rows at the top of each sheet starting with *comment_symbol* will be ignored. Default value is ``"#"``.

    Returns
    -------
    dict
        If *sheetname* is ``None`` a dict of each sheet in the file will be returned. Otherwise a dict of the data in
        the given sheet is returned.

    Examples
    --------
    Need to make examples
    """
    wb = load_workbook(filename, data_only=True, read_only=True)
    try:
        if sheetname is None:
            return {sheetname: _read_sheet(wb[sheetname], comment_symbol, default_value) for sheetname in wb.sheetnames}
        elif isinstance(sheetname, int):
            return _read_sheet(wb[wb.sheetnames[sheetname]], comment_symbol, default_value)
        else:
            return _read_sheet(wb.sheet_by_name(sheetname), comment_symbol, default_value)
    finally:
        wb.close()


def _read_sheet(sheet, comment_symbol, default_value = 'nan'):
    for ri in range(1,sheet.max_row):
        if not ((c:=sheet.cell(ri, 1)).data_type == 's' and c.value.startswith(comment_symbol)):
            data = {}
            for ci in range(1, sheet.max_column+1):
                cell = sheet.cell(ri, ci)
                if cell.data_type == 's' and cell.value.strip() != '':
                    column = f'{cell.value}'
                elif cell.value is not None:
                    column = f'column{ci}'
                else:
                    continue
                data[column] = []
                for rj in range(ri+1, sheet.max_row+1):
                    vcell = sheet.cell(rj, ci)
                    if vcell.data_type == 'e':
                        data[column].append(default_value)
                    else:
                        data[column].append(vcell.value)

            for i in range(sheet.max_row + 1 - (ri + 1)):
                if True in (rowisnone:=[data[c][i] is None for c in data.keys()]):
                    if set(rowisnone) == 1:
                        #All values in row is None. In this case just remove the row
                        for c in data.keys():
                            data[c].pop(i)
                    else:
                        #Replace all None values with the default value
                        for c in data.keys():
                            if data[c][i] is None: data[c][i] = default_value

            return data