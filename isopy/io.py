import os as _os
import isopy as _isopy

import csv as _csv
import datetime as _dt
import numpy as _np
import xlrd as _xlrd
import pkg_resources as _pkgr

__all__ = ['get_reference_values', 'import_exp', 'read_csv', 'write_csv', 'read_excel']

#This stores already loaded data sets to they dont have to be read from file each time they are requested.
reference_values = {}

def get_reference_values(name):
    """
    Get predefined reference values.

    The following reference values are currently avaliable:

    * "best isotope fraction M16" - Best measurement of isotopic abundances from a single terrestrial source. Taken
      from Table 1 in `Meija et al. (2016) Pure and Applied Chemistry, 88, 293-306
      <https://doi.org/10.1515/pac-2015-0503>`_.

    * "initial isotope fraction L09" - Isotope abundances 4.56 Ga ago. Taken from Table 10
      in `Lodders et al. (2009) Abundances of the Elements in Astronomy and Astrophysics, Landolt-Börnstein, Berlin
      <https://materials.springer.com/lb/docs/sm_lbs_978-3-540-88055-4_34>`_.

    * "initial isotope abundance L09" - Isotope abundances 4.56 Ga ago. Normalised to Si=10E6. Taken from Table 10
      in `Lodders et al. (2009) Abundances of the Elements in Astronomy and Astrophysics, Landolt-Börnstein, Berlin
      <https://materials.springer.com/lb/docs/sm_lbs_978-3-540-88055-4_34>`_.

    * "isotope mass W17" - The atomic mass of each isotope relative to :math:`^{12}C`. Taken from Table 1 in
      `Wang et al. (2017) Chinese Physics C, 41, 3 <https://doi.org/10.1088/1674-1137/41/3/030003>`_.

    * "sprocess isotope fraction B11" - The s-process contribution to each isotope as a fraction of the abundance.
      Taken from Table 5 in `Bisterzo et al. (2011) Monthly Notices of the
      Royal Astronomical Society, 418, 284-319 <https://doi.org/10.1111/j.1365-2966.2011.19484.x>`_.

    Functions that request default values will automatically receive the most recent data set if more than one exists.

    Parameters
    ----------
    name : str
        Name if the reference data set to be returned. If the last three reference characters (e.g. M16) are omitted
        then the most recent set of reference values with *name* will be returned. The data returned might therefore
        change in future versions. **It is therefore recommended that you specify the data set to be used.**

    Returns
    -------
    ReferenceDict
        A dict containing the specified reference values.
    """
    name = _isopy.core.check_type('name', name, str)
    name = name.strip()
    default_datasets = {'best isotope fraction': 'best isotope fraction M16',
                        'initial isotope fraction': 'initial isotope fraction L09',
                        'initial isotope abundance': 'initial isotope abundance L09',
                        'isotope mass': 'isotope mass W17',
                        'sprocess isotope fraction': 'sprocess isotope fraction B11'}

    #check if the reference characters were omitted and select the newst dataset.
    if name in default_datasets:
        name = default_datasets[name]

    #Check if the dataset has already been loaded
    if name in reference_values:
        return reference_values[name].copy()
    else:
        #filepath = _pkgr.resource_filename('isopy', 'referencedata/{}.csv'.format(name))
        filepath = _os.path.join(_os.path.dirname(__file__), 'referencedata', '{}.csv'.format(name))
        if not _os.path.exists(filepath):
            raise ValueError('A data set called "{}" does not exist'.format(name))

        #Load the reference values and add them to the
        reference_values[name] = _isopy.core.ReferenceDict(read_csv(filepath))
        return reference_values[name].copy()


def import_exp(filename):
    """
    Return the data stored in the exported files from Neptune and Triton instruments.

    Parameters
    ----------
    filename : str
        Location of the **.exp** file to be read.

    Returns
    -------
    NeptuneData
        Stores the data from the exp file

    Examples
    --------
    Need to make examples
    """

    info = {}

    with open(filename, 'r') as exp:
        dialect = _csv.Sniffer().sniff(exp.read())
        exp.seek(0)
        reader = _csv.reader(exp, dialect, quoting=_csv.QUOTE_NONE)

        #Check that we have right data
        title = reader.__next__()[0]
        if title != 'Neptune Analysis Data Report' and title != 'Triton Analysis Data Report' and False:
            raise TypeError('"{}" is not a valid neptune file'.format(filename))

        for row in reader:
            # Collect run information
            if ":" in row[0]:
                i = row[0].split(":", 1)
                info[i[0]] = i[1].strip()

            # Cycle is the first field in the data table
            if row[0] == 'Cycle':
                cycle, time, isotope_data, ratio_data, other_data = _import_exp_data(row, reader)
                break

    return NeptuneData(info, cycle, time, isotope_data, ratio_data, other_data)


def _import_exp_data(legend_row, reader):
    idata = len(legend_row)
    data = [[] for i in range(idata)]

    for cycle in reader:
        # Signals the end of the cycle data
        if cycle[0] == "***" or cycle[1].strip() == '': break

        for i in range(idata):
            data[i].append(cycle[i])

    time = None
    cycles = None
    isotope_data = {}
    ratio_data = {}
    other_data = {}

    for i in range(idata):
        if legend_row[i] == 'Cycle':
            cycles = data[i]
        elif legend_row[i] == 'Time':
            time = data[i]
        else:
            if ":" in legend_row[i]:
                line, text = legend_row[i].split(":", 1)
                line = int(line)
            else:
                line = 1
                text = legend_row[i]

            if "/" in text:
                try: text = _isopy.core.RatioString(text)
                except:
                    other_data.setdefault(line, {})[text] = data[i]
                else:
                    ratio_data.setdefault(line, {})[text] = data[i]
            else:
                try: text = _isopy.core.IsotopeString(text)
                except:
                    other_data.setdefault(line, {})[text] = data[i]
                else:
                    isotope_data.setdefault(line, {})[text] = data[i]

    return cycles, time, isotope_data, ratio_data, other_data


class NeptuneData():
    """
    Returned by :func:`read_exp`.

    If no line numbers are given in the exp file then the line number is set to 1.

    Attributes
    ----------
    info : dict
        The information stored in the exp file before the isotope data.
    cycle : np.ndarray
        An array containing the cycle number of each row in the data.
    isotope_data : dict
        A dict of :class:`IsotopeArray` each containing the isotope data for the line given by the dict key.
    ratio_data : dict
        A dict of :class:`RatioArray` each containing the ratio data for the line given by the dict key.
    other_data : dict
        A dict of dict each containing the remaining data for the line given by the dict key.
    """
    def __init__(self, information, cycle, time, isotope_data, ratio_data, other_data):
        self.info = information.copy()
        self.cycle = _np.array(cycle, dtype = 'int')
        self.time = [_dt.datetime.strptime(time[i], '%H:%M:%S:%f') for i in range(len(time))]

        self.isotope_data = {key: _isopy.core.IsotopeArray(isotope_data[key]) for key in isotope_data}
        self.ratio_data = {key: _isopy.core.RatioArray(ratio_data[key]) for key in ratio_data}
        self.other_data = other_data


def read_csv(filepath, comment_symbol='#', skip_columns=None):
    """
    Read data from an csv file.

    The first row of the file will be used as the key and the subsequent rows will be taken as the value.

    Parameters
    ----------
    filepath : str
        Location of file to be read
    comment_symbol : str, optional
        Rows at the beginning of the file that start with this *comment_symbol* are ignored. Default value is ``"#"``.
    skip_columns : int, list of int, optional
        If given then the columns with these indexes are not read.

    Returns
    -------
    dict
        A dict containing the contents of the file.

    Examples
    --------
    Need to make examples
    """
    filepath = _isopy.core.check_type('filepath', filepath, str)
    comment_symbol = _isopy.core.check_type('comment_symbol', comment_symbol, str)
    skip_columns = _isopy.core.check_type_list('skip_columns', skip_columns, int, allow_none=True, make_list=True
                                      )
    keys = []
    values = []
    key_len = None

    if skip_columns is None: skip_columns = []
    elif not isinstance(skip_columns, (list, tuple)): skip_columns = [skip_columns]

    with open(filepath, 'r', newline='') as file:
        firstline = file.readline()
        dialect = _csv.Sniffer().sniff(file.read())
        file.seek(0)
        if firstline[:3] == 'ï»¿':
            #Ugly way of detecting if file is encoded with a Byte order mark
            #Excel csv files are encoded like this
            file.seek(3)
        reader = _csv.reader(file, dialect=dialect)

        for row in reader:
            if [len(r.strip()) for r in row].count(0) == len(row): break
            if comment_symbol is None or not row[0].strip().startswith(comment_symbol):
                if key_len is None:
                    keys = [x.strip() for x in row]
                    while keys[-1] == '': keys = keys[:-1]
                    key_len = len(keys)
                    values = [list([]) for x in keys]
                else:
                    if len(row) < key_len: raise ValueError('Row does not have a value for every key')
                    for i in range(key_len):
                        values[i].append(row[i].strip())
    return {keys[i]: values[i] for i in range(key_len) if i not in skip_columns}


def write_csv(data, filepath, comments = None, delimiter =',', comment_symbol='#'):
    """
    Write data to a csv file.

    The first row in each column will contain the key of each item in *data*. Subsequent rows will contains the
    values of each item in the dict.

    Parameters
    ----------
    data : dict, IsopyArray
        The data to be writen to the file.
    filepath : str
        Location where to save the file. It the file already exists it will overwrite the old file.
    comments : str, list of str, optional
        Comments to be included before the file.
    delimiter :str, optional
        The delimiter for columns in the file. Default value is ``","``
    comment_symbol : str, optional
        Used to denote that the line contains a comment.  Default value is ``"#"``.
    """
    filepath = _isopy.core.check_type('filepath', filepath, str)
    comments = _isopy.core.check_type_list('comments', comments, str, allow_none=True, make_list=True)
    delimiter = _isopy.core.check_type('delimiter', delimiter, str)
    comment_symbol = _isopy.core.check_type('comment_symbol', comment_symbol, str)

    with open(filepath, mode='w', newline='') as file:
        writer = _csv.writer(file, delimiter=delimiter)

        if comments is not None:
            for c in comments:
                writer.writerow(['{}{}'.format(comment_symbol, c)])

        writer.writerow([k for k in data.keys()])

        if data.ndim == 0:
            writer.writerow(['{}'.format(data[k]) for k in data.keys()])
        else:
            keys = data.keys()
            for i in range(data.size):
                writer.writerow(['{}'.format(data[k][i]) for k in keys])


def read_excel(filename, sheetname = None, comment_symbol='#'):
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
    wb = _xlrd.open_workbook(filename)
    try:
        if sheetname is None:
            return {sheetname: _read_sheet(wb.sheet_by_name(sheetname), comment_symbol) for sheetname in wb.sheet_names()}
        elif isinstance(sheetname, int):
            return _read_sheet(wb.sheet_by_index(sheetname), comment_symbol)
        else:
            return _read_sheet(wb.sheet_by_name(sheetname), comment_symbol)
    finally:
        wb.release_resources()


def _read_sheet(sheet, comment_symbol):
    for ri in range(sheet.nrows):
        if sheet.cell_value(ri, 0)[:len(comment_symbol)] != comment_symbol:
            return {sheet.cell_value(ri, c): [sheet.cell_value(r, c) for r in (range(ri+1, sheet.nrows))] for c in
                    range(sheet.ncols)}


def write_excel(filename, *sheets):
    pass