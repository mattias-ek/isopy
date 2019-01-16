import numpy as _np
import isopy.dtypes as _dtypes
import csv as _csv
import datetime as _dt

def csvfile(filepath, header = False, skip_first_n_rows = 0, skip_first_n_columns = 0, empty_string_default = None):
    if empty_string_default is None: empty_string_default = ''
    out = {}
    column_title = {}
    header_set = False
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

            if not header_set:
                #TODO make sure header doesnt already exist
                header_set = True
                if header:
                    for i in range(skip_first_n_columns, len(row)): column_title[i] = row[i].strip()
                    continue
                else:
                    for i in range(skip_first_n_columns, len(row)): column_title[i] = i


            for i in range(skip_first_n_columns, len(column_title)):
                try: row_val = row[i].strip()
                except: row_val = ''
                if row_val == '': row_val = empty_string_default
                out.setdefault(column_title[i], []).append(row_val)

    return out

def reference_isotope_data(filepath = None, header = True, **kwargs):
    if filepath is None: filepath = 'isopy/IsotopeData.csv'
    kwargs['header'] = header
    raw = csvfile(filepath=filepath, **kwargs)
    #TODO check that raw is big enough
    #TODO dict of dicts with float64 or np nan

    isotopes = None
    iso_header = None
    if header:
        for iso in ['isotope', 'isotopes', 'ISOTOPE', 'ISOTOPES', 'Isotope', 'Isotopes']:
            if iso in raw:
                isotopes = raw[iso]
                iso_header = iso
                break
        if isotopes is None: raise ValueError('"isotope" header not found in "{}"'.format(filepath))
    else:
        isotopes = raw[0]
        iso_header = 0

    out = {}
    for key in raw:
        if key == iso_header: continue
        out[key] = _dtypes.IsoRatDict(keys = isotopes, values = raw[key])

    return out

def load_standard_ratio_data(filepath = None, header = True):
    pass

def _isorat_array(filepath, **kwargs):
    kwargs['header'] = True
    raw = csvfile(filepath, empty_string_default='nan' ,**kwargs)
    out = {'sample': None, 'value': {}, 'uncertainty': {}}
    for smp in ['sample','samples', 'Sample', 'Samples', 'SAMPLE', 'SAMPLES']:
        if smp in raw:
            out['sample'] = raw.pop(smp)
            break

    for key in raw:
        if '+-' in key:
            out['uncertainty'][key.replace('+-', '').strip()] = raw[key]
        else:
            out['value'][key] = raw[key]

    return out

def isotope_array(filepath, **kwargs):
    out = _isorat_array(filepath, **kwargs)
    out['value'] = _dtypes.IsotopeArray(out['value'])
    if len(out['uncertainty']) == 0: out['uncertainty'] = None
    else: out['uncertinaty'] = _dtypes.IsotopeArray(out['uncertinaty'])
    return out

def ratio_array(filepath, **kwargs):
    out = _isorat_array(filepath,**kwargs)
    out['value'] = _dtypes.RatioArray(out['value'])
    if len(out['uncertainty']) == 0: out['uncertainty'] = None
    else: out['uncertainty'] = _dtypes.RatioArray(out['uncertainty'])
    return out

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
        #TODO block
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

            self.data[l] = _np.zeros(lc, dtype = [('{}'.format(iso), 'f8') for iso in self.isotopes[l]])
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
                #Collect run information
                if ":" in row[0]:
                    i = row[0].split(":",1)
                    self.info[i[0]] = i[1]

                #Cycle is the first field in the data table
                if row[0] == 'Cycle':

                    if 'Analysis date' in self.info: date = self.info['Analysis date']
                    elif 'Date' in self.info: date = self.info
                    else: date = None

                    field['cycle'] = 0


                    #Find fields with isotope data
                    for ci in range(1,len(row)):
                        if row[ci] == 'Time':
                            field['time'] = ci
                            continue

                        if "/" in row[ci]:
                            #Not interested in ratios
                            continue

                        #Check if multiple lines
                        if ":" in row[ci]:
                            line, isotope = row[ci].split(":", 1)
                            line = int(line)
                        else:
                            line = 1
                            isotope = row[ci]

                        try:
                            self.isotopes.setdefault(line, []).append(IsotopeString(isotope, False))
                        except:
                            #Column text does not fit the isotope format
                            pass
                        else:
                            field.setdefault(line, []).append(ci)

                    for cycle in reader:
                        #Signals the end of the cycle data
                        if cycle[0] == "***" or cycle[1].strip() == '': return data

                        for l in self.isotopes:
                            self.cycle.setdefault(l,[]).append(int(cycle[field['cycle']]))
                            if date is None: self.time.setdefault(l, []).append(_dt.time.strftime(cycle[field['time']], '%H:%M:%S:%f'))
                            else: self.time.setdefault(l, []).append(_dt.datetime.strptime('{} {}'.format(date.strip(), cycle[field['time']]), '%m/%d/%Y %H:%M:%S:%f'))
                            for i in range(len(self.isotopes[l])):
                                data.setdefault(l,{}).setdefault(str(self.isotopes[l][i]),[]).append(cycle[field[l][i]])
        return data