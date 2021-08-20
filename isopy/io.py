import os as _os
from isopy import core
import csv as csv
import datetime as dt
import numpy as np
import chardet
import openpyxl
import pyperclip
from openpyxl import load_workbook
import itertools
import io


__all__ = ['read_exp', 'read_csv', 'write_csv', 'read_xlsx', 'write_xlsx', 'read_clipboard']

import isopy.checks

################
### read exp ###
################
class NeptuneData:
    """
    Container for the data returned by ``read_exp``.
    """
    def __init__(self, info, cycle, time, measurements):
        self.info = info
        self.cycle = cycle
        self.time = time
        self.measurements = measurements

def read_exp(filename, rename = None) -> NeptuneData:
    """
    Load data from a Neptune/Triton export file.

    Parameters
    ----------
    filename : str, bytes, StringIO, BytesIO
        Path for file to be opened. Alternatively a file like byte string can be supplied.
        Also accepts file like objects.
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
        * measurements - An dictionary containing an an isopy array with the values in for each line measured. Static measurements are always given as line ``1``.
         to extract e.g only the isotope data from the measurement use ``neptune_data.measurement[line].copy(flavour_eq='isotope')``.
    """
    information = {}

    # If filename is a string load the files.
    if type(filename) is str:
        with open(filename, 'rb') as fileio:
            file = fileio.read()
    elif type(filename) is bytes:
        file = filename
    elif type(filename) is io.BytesIO:
        filename.seek(0)
        file = filename.read()
    elif type(filename) is str:
        file = filename
    elif type(filename) is io.StringIO:
        filename.seek(0)
        file = filename.read()
    else:
        raise TypeError('filename is of unknown type')

    if type(file is bytes):
        # find the files encoding
        encoding = chardet.detect(file).get('encoding')

        # Decode the bytes into string.
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

    data.pop("", None)
    cycle = np.array(data.pop('Cycle'), dtype ='int')
    time = data.pop('Time')
    try:
        time =[dt.datetime.strptime(time[i], '%H:%M:%S:%f') for i in range(len(time))]
    except ValueError:
        try:
            time = [dt.datetime.fromtimestamp(time[i]).strftime('%H:%M:%S:%f') for i in range(len(time))]
        except:
            time = [dt.datetime.fromtimestamp(0).strftime('%H:%M:%S:%f') for i in range(len(time))]
    measurements = {}

    for key in data:
        if ":" in key:
            line, keystring = key.split(":", 1)
            line = int(line)
        else:
            line = 1
            keystring = key
        keystring = renamer(keystring)
        measurements.setdefault(line, {})[isopy.keystring(keystring)] = data[key]


    measurements = {line: isopy.asarray(measurements[line]) for line in measurements}
    return NeptuneData(information, cycle, time, measurements)

######################
### read/write CSV ###
######################
def read_csv(filename, comment_symbol ='#', keys_in = 'c',
             float_preferred = False, encoding = None,
             dialect = None):
    """
    Load data from a csv file.

    Parameters
    ----------
    filename : str, bytes, StringIO, BytesIO
        Path for file to be opened. Alternatively a file like byte string can be supplied.
        Also accepts file like objects.
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
    if type(filename) is str:
        with open(filename, 'rb') as fileio:
            file = fileio.read()
    elif type(filename) is bytes:
        file = filename
    elif type(filename) is io.BytesIO:
        filename.seek(0)
        file = filename.read()
    elif type(filename) is str:
        file = filename
    elif type(filename) is io.StringIO:
        filename.seek(0)
        file = filename.read()
    else:
        raise TypeError('filename is of unknown type')

    if type(file is bytes):
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

        if comment_symbol is not None and row[0][:len(comment_symbol)] == comment_symbol: #startswith(comment_symbol):
            # This is a comment so ignore it.
            continue

        if keys_in == 'c':
            data =  _read_csv_ckeys(row_data, csv_reader, comment_symbol, float_prefered=float_preferred)
            data.pop("", None)
            return data
        elif keys_in == 'r':
            data =  _read_csv_rkeys(row_data, csv_reader, comment_symbol, float_prefered=float_preferred)
            data.pop("", None)
            return data
        elif keys_in is None:
            return _read_csv_nokeys(row_data, csv_reader, comment_symbol, float_prefered=float_preferred)
        else:
            raise ValueError(f'Unknown value for "keys_in" {keys_in}')

def read_clipboard(comment_symbol ='#', keys_in = 'c',
             float_preferred = False, dialect = None):
    """
    Load data from values in the clipboard.

    Parameters
    ----------
    comment_symbol : str, Default = '#'
        Rows starting with this string will be ignored.
    keys_in : {'c', 'r', None}, Default = 'c'
        If keys are given as the first value in each column pass ``c``. If keys are given as the
        first value in each row pass ``r``. If there are no keys pass ``None``.
    float_preferred : bool, Default = False
        If `True` all values will be converted to float if possible. If conversion fails the
        value will be left as a string.
    dialect
        Dialect of the values in the clipboard. If None the dialect will be guessed from the values.

    Returns
    -------
    data : dict or list
        Returns a dictionary for data with keys otherwise a list.
    """
    data = pyperclip.paste()
    data = data.encode('UTF-8')
    return read_csv(data, encoding='UTF-8', comment_symbol=comment_symbol, keys_in=keys_in,
                    float_preferred=float_preferred, dialect=dialect)


def _read_csv_ckeys(first_row, reader, comment_symbol=None, termination_symbol=None, float_prefered = False):
    keys = first_row  # First row contains the the name of each column
    values = [[] for i in range(len(keys))]  # Create an empty list for each key

    # Loop over the remaining rows
    for i, row in enumerate(reader):
        row = [r.strip() for r in row]

        if termination_symbol is not None and row[0][:len(termination_symbol)] == termination_symbol: #startswith(termination_symbol):
            # Stop reading data if we find this string at the beginning of a row
            break

        if comment_symbol is not None and row[0][:len(comment_symbol)] == comment_symbol: #.startswith(comment_symbol):
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

        if termination_symbol is not None and row[0][:len(termination_symbol)] == termination_symbol: #.startswith(termination_symbol):
            # Stop reading data if we find this string at the beginning of a row
            break

        if comment_symbol is not None and row[0][:len(comment_symbol)] == comment_symbol: #.startswith(comment_symbol):
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

        if termination_symbol is not None and row[0][:len(termination_symbol)] == termination_symbol: #.startswith(termination_symbol):
            # Stop reading data if we find this string at the beginning of a row
            break

        if comment_symbol is not None and row[0][:len(comment_symbol)] == comment_symbol: #.startswith(comment_symbol):
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
    filename : str, StringIO, BytesIO
        Path/name of the csv file to be created. Any existing file with the same path/name will be
        over written. Also accepts file like objects.
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
    #filename = isopy.checks.check_type('filename', filename, str)
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
            data = np.array(data.to_array())
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

    rows = []

    if comments is not None:
        for comment in comments:
            rows.append([f'{comment_symbol}{comment}'] + ['' for i in range(csize)])

    rows += func(data, dsize)
    if type(filename) is str:
        with open(filename, mode='w', newline='') as file:
            csv_writer = csv.writer(file, delimiter=delimiter)
            for row in rows:
                csv_writer.writerow(row)

    if type(filename) is io.StringIO:
        csv_writer = csv.writer(filename, delimiter=delimiter)
        for row in rows:
            csv_writer.writerow(row)

    if type(filename) is io.BytesIO:
        file = io.StringIO()
        csv_writer = csv.writer(file, delimiter=delimiter)
        for row in rows:
            csv_writer.writerow(row)
        file.seek(0)
        string = file.read()
        filename.write(string.encode('UTF-8'))

def _write_csv_ckeys(data, dsize):
    keys = data.keys()
    rows = []
    rows.append([f'{key}' for key in keys])

    for i in range(dsize):
        rows.append([f'{data[key][i]}' for key in keys])

    return rows

def _write_csv_rkeys(data, dsize):
    keys = data.keys()
    rows = []

    for key in keys:
        rows.append([f'{key}'] + [f'{data[key][i]}' for i in range(dsize)])
    return rows

def _write_csv_nokeys(data, dsize):
    rows = []
    for i in range(data.shape[0]):
        rows.append([f'{data[i][j]}' for j in range(data.shape[-1])])
    return rows

#######################
### Read/write xlsx ###
#######################
def read_xlsx(filename, sheetname=None, keys_in='c',
              default_value = np.nan, comment_symbol='#'):
    """
    Load data from an excel file.

    Parameters
    ----------
    filename : str, bytes, BytesIO
        Path for file to be opened. Also accepts file like objects.
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
    if filename is bytes:
        filename = io.BytesIO(filename)

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
    filename : str, BytesIO
        Path/name of the excel file to be created. Any existing file with the same path/name
        will be overwritten. Also accepts file like objects.
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

    save = True
    if type(filename) is openpyxl.Workbook:
        workbook = filename
        save = False
    elif type(filename) is io.BytesIO:
        if append:
            #TODO some sort of check to make sure the file isnt empty
            try:
                workbook = openpyxl.load_workbook(filename=filename)
            except:
                workbook = openpyxl.Workbook()
                workbook.remove(workbook.active)
        else:
            workbook = openpyxl.Workbook()
            workbook.remove(workbook.active)
    elif type(filename) is str:
        if append and _os.path.exists(filename):
            workbook = openpyxl.load_workbook(filename=filename)
        else:
            workbook = openpyxl.Workbook()
            workbook.remove(workbook.active)

    sheetname_data = {f'sheet{i+1}': data for i, data in enumerate(sheets)}
    sheetname_data.update(sheetnames)
    try:
        for sheetname, data in sheetname_data.items():
            #If appending then delete any preexisting workbooks
            if sheetname in workbook.sheetnames:
                workbook.remove(workbook[sheetname])

            worksheet = workbook.create_sheet(sheetname)
            _write_xlsx(worksheet, data, comments, comment_symbol, keys_in)

        #Workbooks must have at least one sheet
        if len(workbook.sheetnames) == 0:
            workbook.create_sheet('sheet1')

        if save: workbook.save(filename)
    finally:
        if save: workbook.close()

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