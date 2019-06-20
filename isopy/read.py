import numpy as _np
import isopy._dtypes as _dtypes
import csv as _csv
import datetime as _dt
import os


def file(filepath, first_column_is_key = True, comments ='#', fixed_row_size = False):
    keys = []
    values = []
    row_size = None

    #TODO propblems opening file etc

    with open(filepath, 'r') as file:
        dialect = _csv.Sniffer().sniff(file.read(2048))
        file.seek(0)
        reader = _csv.reader(file, dialect, quoting=_csv.QUOTE_NONE)

        for row in reader:
            if comments is not None:
                try:
                    if row[0].strip()[0] == comments: continue
                except:
                    pass
            if len(row) == 0: continue
            if fixed_row_size:
                if row_size is None: row_size = len(row)
                elif len(row) != row_size: raise ValueError('All rows must be the same size')

            if first_column_is_key:
                keys.append(row[0].strip())
                values.append(row[1:])
            else:
                values.append(row[:])

        if first_column_is_key: return keys, values
        else: return values

def reference_data(name):
    #TODO file not found etc
    filepath = os.path.join(os.path.dirname(__file__), 'referencedata', '{}.csv'.format(name))
    keys, values = file(filepath)
    return _dtypes.IsopyDict(values, keys = keys)

#Needs to be updated
class neptune(object):
    """Used to read data files produced by a Neptune ICP-MS.

    Parameters
    ----------
    info : dict
        A dict of the information provided in the begining of the file (if given)
    lines : dict
        A dict with the lines specified in the file.

    """

    def __init__(self, file):
        # TODO block
        """

        Parameters
        ----------
        file
        """
        self.info = {}

        self.data = {}
        self.isotopes = {}
        self.cycle = {}
        self.time = {}
        self.block = {}
        self.line = {}

        data = self._read_file(file)
        for l in self.isotopes:
            lc = len(self.cycle[l])

            self.data[l] = _np.zeros(lc, dtype=[('{}'.format(iso), 'f8') for iso in self.isotopes[l]])
            for iso in self.isotopes[l]: self.data[l][str(iso)] = data[l][str(iso)]

    def _read_file(self, file):
        field = {}
        data = {}

        with open(file, 'r') as exp:
            dialect = _csv.Sniffer().sniff(exp.read(1024))
            exp.seek(0)
            reader = _csv.reader(exp, dialect, quoting=_csv.QUOTE_NONE)

            title = reader.__next__()[0]
            if title != 'Neptune Analysis Data Report' and title != 'Triton Analysis Data Report' and False:
                raise TypeError('"{}" is not a valid neptune file'.format(file))

            for row in reader:
                # Collect run information
                if ":" in row[0]:
                    i = row[0].split(":", 1)
                    self.info[i[0]] = i[1]

                # Cycle is the first field in the data table
                if row[0] == 'Cycle':

                    if 'Analysis date' in self.info:
                        date = self.info['Analysis date']
                    elif 'Date' in self.info:
                        date = self.info
                    else:
                        date = None

                    field['cycle'] = 0

                    # Find fields with isotope data
                    for ci in range(1, len(row)):
                        if row[ci] == 'Time':
                            field['time'] = ci
                            continue

                        if "/" in row[ci]:
                            # Not interested in ratios
                            continue

                        # Check if multiple lines
                        if ":" in row[ci]:
                            line, isotope = row[ci].split(":", 1)
                            line = int(line)
                        else:
                            line = 1
                            isotope = row[ci]

                        try:
                            self.isotopes.setdefault(line, []).append(IsotopeString(isotope, False))
                        except:
                            # Column text does not fit the isotope format
                            pass
                        else:
                            field.setdefault(line, []).append(ci)

                    for cycle in reader:
                        # Signals the end of the cycle data
                        if cycle[0] == "***" or cycle[1].strip() == '': return data

                        for l in self.isotopes:
                            self.cycle.setdefault(l, []).append(int(cycle[field['cycle']]))
                            if date is None:
                                self.time.setdefault(l, []).append(
                                    _dt.time.strftime(cycle[field['time']], '%H:%M:%S:%f'))
                            else:
                                self.time.setdefault(l, []).append(
                                    _dt.datetime.strptime('{} {}'.format(date.strip(), cycle[field['time']]),
                                                          '%m/%d/%Y %H:%M:%S:%f'))
                            for i in range(len(self.isotopes[l])):
                                data.setdefault(l, {}).setdefault(str(self.isotopes[l][i]), []).append(
                                    cycle[field[l][i]])
        return data

#Obsolete
def _key_csvfile(filepath, column_key=True, skip_first_n_rows=0, skip_first_n_columns=0, empty_string_default=None):
    if empty_string_default is None: empty_string_default = ''
    key = []
    out = {}
    column_title = {}
    column_key_set = False

    with open(filepath, 'r') as file:
        dialect = _csv.Sniffer().sniff(file.read(2048))
        file.seek(0)
        reader = _csv.reader(file, dialect, quoting=_csv.QUOTE_NONE)

        for row in reader:
            try:
                if row[0].strip()[0] == '#': continue
            except:
                pass

            if skip_first_n_rows > 0:
                skip_first_n_rows -= 1
                continue

            if not column_key_set:
                # TODO make sure header doesnt already exist
                column_key_set = True
                if column_key:
                    for i in range(skip_first_n_columns + 1, len(row)):
                        column_title[i] = row[i].strip()
                    continue
                else:
                    for i in range(skip_first_n_columns + 1, len(row)): column_title[i] = i

            # get key
            try:
                key_val = row[skip_first_n_columns].strip()
            except:
                continue
            if key_val == '': continue
            if key_val in key: continue
            key.append(key_val)

            # read data
            for i in range(skip_first_n_columns + 1, len(column_title) + 1):
                try:
                    row_val = row[i].strip()
                except:
                    row_val = ''
                if row_val == '': row_val = empty_string_default
                out.setdefault(column_title[i], []).append(row_val)

    return key, out


def _reference_isotope_data():
    """'

    Reference data keys:
    - 'initial isotope abundance L09'

    - 'initial isotope fraction L09'

    - 'isotope fraction M16'

    - 'isotope fraction uncertainty M16'

    - 'isotope mass H17'

    - 'sprocess isotope abundance B11'

    - 'sprocess isotope fraction B11'

    - 'sprocess isotope abundance A99'

    - 'sprocess isotope fraction A99'

    """
    filepath = os.path.join(os.path.dirname(__file__), 'IsotopeData.csv')
    return data_dict(filepath)


def data_dict(filepath=None, column_key=True, skip_first_n_rows=0, skip_first_n_columns=0, empty_string_default=None):
    keys, data = _key_csvfile(filepath, column_key, skip_first_n_rows, skip_first_n_columns, empty_string_default)
    out = {}
    for k in data:
        out[k] = _dtypes.IsopyDict(keys=keys, values=data[k])

    return out


