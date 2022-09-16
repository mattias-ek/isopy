import functools
import os as _os
import warnings

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
import numpy as np


__all__ = ['read_exp',
           'read_csv', 'write_csv',
           'read_xlsx', 'write_xlsx',
           'read_clipboard', 'write_clipboard']

import isopy.checks

NAN_STRINGS = 'nan #NA #N/A N/A NA =NA() =na()'.split()

def rows_to_data(data, has_keys, keys_in_first):
    if len(data) == 0 or len(data[0])==0:
        if has_keys is not True and keys_in_first is None:
            return [[]]
        else:
            return dict()

    # Everything needs to be converted to nan before it gets here
    if has_keys is not False and keys_in_first is None:
        r = [type(v) is str for v in data[0]]
        c = [type(d[0]) is str for d in data]

        if False not in c and False not in r:
            # Both are strings
            try:
                float(data[0][0])
            except ValueError:
                f = True
            else:
                f = False

            try:
                [float(v) for v in data[0][1:]]
            except ValueError:
                r = True
            else:
                r = False

            try:
                [float(v[0]) for v in data[1:]]
            except ValueError:
                c = True
            else:
                c = False

            if has_keys is None and not c and not r:
                if f and (len(data)) == 1:
                    keys_in_first = 'c'
                elif f and len(data[0]) == 1:
                    keys_in_first = 'r'
                elif f:
                    # Both the first row and the first
                    warnings.warn('Both the first row and the first column contain strings. Using first row as keys')
                    keys_in_first = 'r'
                    #raise ValueError('Unable to determine whether the first rows or columns contains the keys')
                else:
                    # All values in the first row and the first column can be converted to float
                    has_keys = False
            elif c and not r:
                keys_in_first = 'c'
            elif r and not c:
                keys_in_first = 'r'
            else:
                # Both the first row and the first column contains at least one value that cant be converted to float
                warnings.warn('Both the first row and the first column contain strings. Using first row as keys')
                keys_in_first = 'r'
                #raise ValueError('Unable to determine whether the first rows or columns contains the keys')

        elif False not in c:
            # Only c contains string
            keys_in_first = 'c'
        elif False not in r:
            # Only r contains strings
            keys_in_first = 'r'
        else:
            # Neither contain only strings
            has_keys=False

    if has_keys is False:
        return data
    elif keys_in_first == 'c':
        return {v[0]: v[1:] for v in data}
    elif keys_in_first == 'r':
        return {v[0]: v[1:] for v in zip(*data)}
    else:
        raise ValueError(f'Unknown value for "keys_in_first" {keys_in_first}')

def data_to_rows(data, keys_in_first):
    data = isopy.asanyarray(data)

    if isinstance(data, core.IsopyArray):
        if data.ndim == 0:
            data = data.reshape(-1)
        if keys_in_first == 'r':
            rows = [[str(k) for k in data.keys()]]
            rows += data.to_list()
        elif keys_in_first == 'c':
            rows = [[str(k)] + v.tolist() for k, v in data.items()]
        else:
            raise ValueError(f'Unknown value for "keys_in_first" {keys_in_first}')
    else:
        if data.ndim == 0:
            data = data.reshape((1, 1))
        elif data.ndim == 1:
            data = data.reshape((1, -1))
        elif data.ndim > 2:
            raise ValueError('data cannot have more than 2 dimensions')
        rows = data.tolist()

    return rows

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
        Path for file to be opened. Alternatively a file like byte string or a file like object can be supplied.
    rename : dict, Callable, Optional
        For renaming keys in the analysed data. Useful for cases when the key is the mass rather
        than the isotope measured. If a dictionary is passed then every key present in the
        dictionary will be replaced by the associated value. A callable can also be passed that
        takes the key in the file and returns the new key.

    Returns
    -------
    neptune_data : NeptuneData
        An object containing the following attributes:

        * info - Dictionary containing the metadata included at the beginning of the file.
        * cycle - A list containing the cycle number for each measurement.
        * time - A list containing datetime objects for each measurement.
        * measurements - An dictionary containing an an isopy array with the values in for each line measured. Static measurements are always given as line ``1``.

         To extract e.g only the isotope data from a measurement use ``neptune_data.measurement[line].copy(flavour_eq='isotope')``.
    """


    # If filename is a string load the files.
    if type(filename) is str:
        with open(filename, 'rb') as fileio:
            file = fileio.read()
    elif type(filename) is bytes:
        file = filename
    elif type(filename) is io.BytesIO:
        filename.seek(0)
        file = filename.read()
    elif type(filename) is io.StringIO:
        filename.seek(0)
        file = filename.read()
    else:
        raise TypeError('filename is of unknown type')

    if type(file) is bytes:
        # find the files encoding
        encoding = chardet.detect(file).get('encoding')

        # Decode the bytes into string.
        file = file.decode(encoding)

    csv_reader = csv.reader(io.StringIO(file), dialect='excel-tab')

    information = {}
    for row in csv_reader:
        if ':' in row[0]:  # meta
            name, value = row[0].split(':', 1)
            information[name.strip()] = value.strip()

        if row[0] == 'Cycle':
            # We can reuse the function from before to read the integration data
            data = _read_csv_data(row, csv_reader, termination_symbol='***')
            data = rows_to_data(data, True, 'r')

    if rename is None:
        renamer = lambda name: name
    elif isinstance(rename, dict):
        renamer = lambda name: rename.get(name, name)
    elif callable(rename):
        renamer = rename
    else:
        raise TypeError('rename must be a dict or a callable function')

    data.pop("", None)
    cycle = [int(i) for i in data.pop('Cycle')]

    time = data.pop('Time')
    try:
        time = [dt.datetime.strptime(time[i], '%H:%M:%S:%f') for i in range(len(time))]
    except:
        # Dont think the time will ever be given as a float
        # time = [dt.datetime.fromtimestamp(time[i]).strftime('%H:%M:%S:%f') for i in range(len(time))]
        warnings.warn('Unable to parse the time column')
        time = [dt.datetime.fromtimestamp(0).strftime('%H:%M:%S:%f') for i in range(len(time))]

    measurements = {}

    for key in data:
        # Check if the data has more than one line
        if ":" in key:
            line, keystring = key.split(":", 1)
            line = int(line)
        else:
            line = 1
            keystring = key

        # Rename columns if needed
        keystring = renamer(keystring)
        measurements.setdefault(line, {})[isopy.keystring(keystring)] = data[key]

    # Convert to isopy arrays
    measurements = {line: isopy.asarray(measurements[line]) for line in measurements}

    return NeptuneData(information, cycle, time, measurements)

######################
### read/write CSV ###
######################
def read_csv(filename, has_keys= None, keys_in_first = None, comment_symbol ='#', encoding = None, dialect = 'excel'):
    """
    Load data from a csv file.

    Parameters
    ----------
    filename : str, bytes, StringIO, BytesIO
        Path for file to be opened. Alternatively a file like byte string can be supplied.
        Also accepts file like objects.
    has_keys : bool, None
        If True or *keys_in_first* is not None a dictionary will always be returned. If False an nexted list is
        always returned. If None it will return a nested list of all values in the first row and column
        can be converted to/is a float.
    keys_in_first : {'c', 'r', None}
        Where the keys are found. Give 'r' if the keys are found in the first row and 'c' if the
        keys are in first column. If None it will analyse the data to determine where the keys are. If *has_keys*
        is not False an exception will be raised if it cant determine where the keys are.
    comment_symbol : str, Default = '#'
        Rows starting with this string will be ignored.
    encoding : str
        Encoding of the file. If None the encoding will be guessed from the file.
    dialect
        Dialect of the csv file. If None the dialect will be guessed from the file.

    Returns
    -------
    data : dict or list
        Returns a dictionary for data with keys otherwise a list.
    """

    #If filename is a string load the files.
    if type(filename) is str:
        with open(filename, 'rb') as fileio:
            text = fileio.read()
    elif type(filename) is bytes:
        text = filename
    elif type(filename) is io.BytesIO:
        filename.seek(0)
        text = filename.read()
    elif type(filename) is io.StringIO:
        filename.seek(0)
        text = filename.read()
    else:
        raise TypeError(f'filename is of unknown type {type(filename)}')

    if type(text) is bytes:
        #find the files encoding
        if encoding is None:
            encoding = chardet.detect(text).get('encoding')

        #Decode the bytes into string.
        text = text.decode(encoding)

    #find the csv dialect
    if dialect is None:
        dialect = csv.Sniffer().sniff(text)

    # Create a reader object by converting the file string to a file-like object
    csv_reader = csv.reader(io.StringIO(text), dialect=dialect)
    for row in csv_reader:
        row_data = [r.strip() for r in row]  # Remove any training whitespaces from the data in this row
        if len(row_data) == 0:
            return rows_to_data([[]], has_keys, keys_in_first)

        if comment_symbol is not None and row[0][:len(comment_symbol)] == comment_symbol:
            pass # This is a comment so ignore it.
        else:
            data = _read_csv_data(row_data, csv_reader, comment_symbol)
            return rows_to_data(data, has_keys, keys_in_first)

    return rows_to_data([[]], has_keys, keys_in_first)

def _read_csv_data(first_row, reader, comment_symbol=None, termination_symbol=None):
    data = []
    dlen = None

    # Loop over the remaining rows
    for i, row in enumerate(itertools.chain([first_row], reader)):
        row = [r.strip() for r in row]
        if len(row) > 0:
            if termination_symbol is not None and row[0][:len(termination_symbol)] == termination_symbol:
                # Stop reading data if we find this string at the beginning of a row
                break

            if comment_symbol is not None and row[0][:len(comment_symbol)] == comment_symbol:
                # Row is a comment, ignore
                continue

        data.append([])
        if dlen is None:
            dlen = len(row)

        elif dlen != len(row):
            raise ValueError('Rows have different number of values')

        for j in range(len(row)):
            value = row[j]

            if value in NAN_STRINGS:
                value = float('nan')

            data[-1].append(value)

    return data


def write_csv(filename, data, comments=None, keys_in_first='r', comment_symbol = '#',
              dialect = 'excel') -> None:
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
    keys_in_first : {'c', 'r'}
        Only used if the input has keys. Give 'r' if the keys should be in the first row and 'c' if the
        keys should be in the first column.
    comment_symbol : str, Default = '#'
        This string will precede any comments at the beginning of the file.
    dialect
        The CSV dialect used to save the file. Default to 'excel' which is a ', ' seperated file.
    """

    rows = data_to_rows(data, keys_in_first)

    if comments is not None:
        if type(comments) is not list:
            comments = [comments]

        crows = []
        for comment in comments:
            crows.append([f'{comment_symbol}{comment}'] + ['' for i in range(len(rows[0][1:]))])

        if len(rows[0]) == 0:
            rows = crows
        else:
            rows = crows + rows

    if type(filename) is str:
        with open(filename, mode='w', newline='') as file:
            csv_writer = csv.writer(file, dialect=dialect)
            for row in rows:
                csv_writer.writerow(row)

    elif type(filename) is io.StringIO:
        filename.truncate(0)
        filename.seek(0)
        csv_writer = csv.writer(filename, dialect=dialect)
        for row in rows:
            csv_writer.writerow(row)

    elif type(filename) is io.BytesIO:
        filename.truncate(0)
        filename.seek(0)

        file = io.StringIO()
        csv_writer = csv.writer(file, dialect=dialect)
        for row in rows:
            csv_writer.writerow(row)
        file.seek(0)
        text = file.read()
        filename.write(text.encode('UTF-8'))
    else:
        raise TypeError(f'filename is of unknown type {type(filename)}')

def read_clipboard(has_keys= None, keys_in_first = None, comment_symbol ='#', dialect = None):
    """
    Load data from values in the clipboard.

    Parameters
    ----------
    comment_symbol : str, Default = '#'
        Rows starting with this string will be ignored.
    has_keys : bool, None
        If True or *keys_in_first* is not None a dictionary will always be returned. If False an nexted list is
        always returned. If None it will return a nested list of all values in the first row and column
        can be converted to/is a float.
    keys_in_first : {'c', 'r', None}
        Where the keys are found. Give 'r' if the keys are found in the first row and 'c' if the
        keys are in first column. If None it will analyse the data to determine where the keys are. If *has_keys*
        is not False an exception will be raised if it cant determine where the keys are.
    dialect
        Dialect of the values in the clipboard. If None the dialect will be guessed from the values.
    """
    data = pyperclip.paste()
    return read_csv(io.StringIO(data), comment_symbol=comment_symbol, has_keys=has_keys,
                    keys_in_first=keys_in_first, dialect=dialect)

def write_clipboard(data, comments=None, keys_in_first='r',
              comment_symbol = '#', dialect = 'excel'):
    """
    Copies data to the clipboard

    Parameters
    ----------
    data : isopy_array_like, numpy_array_like
        Data to be copied.
    comments : str, Sequence[str], Optional
        Comments to be included
    keys_in_first : {'c', 'r'}
        Only used if the input has keys. Give 'r' if the keys should be in the first row and 'c' if the
        keys should be in the first column.
    dialect
        The CSV dialect used to copy the data to the clipboard. Default to 'excel' which is a ', ' seperated file.
    comment_symbol : str, Default = '#'
        This string will precede any comments.
    """
    text = io.StringIO()
    write_csv(text, data, comments=comments, keys_in_first=keys_in_first, dialect=dialect, comment_symbol=comment_symbol)

    text.seek(0)
    pyperclip.copy(text.read())

#######################
### Read/write xlsx ###
#######################
def read_xlsx(filename, sheetname=None, has_keys = None, keys_in_first=None, comment_symbol='#', start_at='A1'):
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
    has_keys : bool, None
        If True or *keys_in_first* is not None a dictionary will always be returned. If False an nexted list is
        always returned. If None it will return a nested list of all values in the first row and column
        can be converted to/is a float.
    keys_in_first : {'c', 'r', None}
        Where the keys are found. Give 'r' if the keys are found in the first row and 'c' if the
        keys are in first column. If None it will analyse the data to determine where the keys are. If *has_keys*
        is not False an exception will be raised if it cant determine where the keys are.
    comment_symbol : str, Default = '#'
        Rows starting with this string will be ignored.
    start_at : str, (int, int)
        Start scanning at this cell. Can either be a excel style cell reference or a (row, column) tuple of integers.

    Returns
    -------
    data : dict
        A dictionary containing the data for a single sheet or a dictionary containing the data
        for multiple sheets in the workbook indexed by sheet name.
    """
    if type(filename) is bytes:
        filename = io.BytesIO(filename)

    if type(filename) is openpyxl.Workbook:
        workbook = filename
    else:
        workbook = openpyxl.load_workbook(filename, data_only=True, read_only=True)

    sheetnames = workbook.sheetnames

    if sheetname is None:
        sheet_data = {}
        for sheetname in sheetnames:
            sheet_data[sheetname] = _read_xlsx_sheet(workbook[sheetname], has_keys, keys_in_first, comment_symbol, start_at)
        return sheet_data

    elif type(sheetname) is int and sheetname < len(sheetnames):
        return _read_xlsx_sheet(workbook[sheetnames[sheetname]], has_keys, keys_in_first, comment_symbol, start_at)

    elif sheetname in sheetnames:
        return _read_xlsx_sheet(workbook[sheetname], has_keys, keys_in_first, comment_symbol, start_at)

    else:
        raise ValueError(f'sheetname "{sheetname}" not found in workbook"')


def _read_xlsx_sheet(worksheet, has_keys, keys_in_first, comment_symbol, start_at):
    if type(start_at) is tuple:
        start_r, start_c = start_at
    else:
        start_r, start_c = openpyxl.utils.cell.coordinate_to_tuple(start_at)

    #Remove inital comments
    for ri in range(start_r, worksheet.max_row + 1):
        value = worksheet.cell(ri, start_c).value
        if type(value) is str and value[:len(comment_symbol)] == comment_symbol and value not in NAN_STRINGS:
            continue
        else:
            start_r = ri
            break

    # Select area containing data
    stop_c = worksheet.max_column + 1
    stop_r = worksheet.max_row + 1

    for ri in range(start_r + 1, stop_r+1):
        values_r = []
        for ci in range(start_c, stop_c):
            if ri != stop_r:
                # All rows should be included
                cell = worksheet.cell(ri, start_c)
                value = cell.value

                if value == '':
                    values_r.append(None)
                else:
                    values_r.append(value)

            if values_r.count(None) == len(values_r):
                values_c = []
                if ci != (stop_c-1):
                    for rj in range(start_r, ri + 1):
                        cell = worksheet.cell(rj, ci + 1)
                        value = cell.value

                        if value == '':
                            values_c.append(None)
                        else:
                            values_c.append(value)

                if values_c.count(None) == len(values_c):
                    stop_r = ri
                    stop_c = ci + 1

            if ci == stop_c: break
        if ri == stop_r: break

    if (stop_r-start_r) == 1 and (stop_c-start_c) == 1 and ((cell:=worksheet.cell(start_r,start_c)).value is None or cell.value==''):
        return rows_to_data([[]], has_keys, keys_in_first)

    data = []

    # Iterate over each column
    for ri in range(start_r, stop_r):
        value = worksheet.cell(ri, start_c).value
        if type(value) is str and value[:len(comment_symbol)] == comment_symbol and value not in NAN_STRINGS:
            continue

        data.append([])
        for ci in range(start_c, stop_c):
            cell = worksheet.cell(ri, ci)
            value = cell.value
            if type(value) is str:
                value = value.strip()

            if cell.data_type == 'e' or (type(value) is str and value.upper() == '=NA()'):
                data[-1].append(float('nan'))
            elif value == '' or value in NAN_STRINGS or value is None:
                data[-1].append(float('nan'))
            else:
                data[-1].append(value)

    return rows_to_data(data, has_keys, keys_in_first)


def write_xlsx(filename, *sheets, comments = None,
               keys_in_first= 'r', comment_symbol= '#', start_at ="A1", append = False, clear = True, **sheetnames):
    """
    Save data to an excel file.

    Parameters
    ----------
    filename : str, BytesIO
        Path/name of the excel file to be created. Any existing file with the same path/name
        will be overwritten. Also accepts file like objects.
    sheets : isopy_array_like, numpy_array_like
        Data given here will be saved as sheet1, sheet2 etc.
    comments : str, Sequence[str], Optional
        Comments to be included at the top of the file
    comment_symbol : str, Default = '#'
        This string will precede any comments at the beginning of the file
    keys_in_first : {'c', 'r'}
        Only used if the input has keys. Give 'r' if the keys should be in the first row and 'c' if the
        keys should be in the first column.
    start_at: str, (int, int)
        The first cell where the data is written. Can either be a excel style cell reference or a (row, column)
        tuple of integers.
    append : bool, Default = False
        If ``True`` and *filename* exists it will append the data to this workbook. An exception
        is raised if *filename* is not a valid excel workbook.
    clear : bool, Default = True
        If ``True`` any preexisting sheets are cleared before any new data is written to it.
    sheetnames : isopy_array, numpy_array
        Data given here will be saved in a sheet with *sheetname* name.
    """
    save = True
    if type(filename) is openpyxl.Workbook:
        workbook = filename
        save = False
    elif type(filename) is io.BytesIO:
        if append and filename.seek(0,2) > 0:
            workbook = openpyxl.load_workbook(filename=filename)
        else:
            # openpyxl truncates BytesIO objects automatically via the use of ZipFile('w')

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
            if sheetname not in workbook.sheetnames:
                worksheet = workbook.create_sheet(sheetname)
            elif clear:
                workbook.remove(workbook[sheetname])
                worksheet = workbook.create_sheet(sheetname)
            else:
               worksheet = workbook[sheetname]

            _write_xlsx(worksheet, data, comments, comment_symbol, keys_in_first, start_at)

        #Workbooks must have at least one sheet
        if len(workbook.sheetnames) == 0:
            workbook.create_sheet('sheet1')

        if save: workbook.save(filename)
    finally:
        if save: workbook.close()

def _write_xlsx(worksheet, data, comments, comment_symbol, keys_in_first, start_at):
    rows = data_to_rows(data, keys_in_first)
    if type(start_at) is tuple:
        start_r, start_c = start_at
    else:
        start_r, start_c = openpyxl.utils.cell.coordinate_to_tuple(start_at)

    if comments is not None:
        if not isinstance(comments, (list, tuple)):
            comments = [comments]
        for comment in comments:
            worksheet.cell(start_r, start_c).value = f'{comment_symbol}{comment}'
            start_r += 1

    for ri, row in enumerate(rows):
        for ci, value in enumerate(row):
            if str(value) == 'nan':
                value = '#N/A'
            if value is not None and value != '':
                worksheet.cell(start_r + ri, start_c + ci).value = value
