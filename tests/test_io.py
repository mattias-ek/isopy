import isopy


def test_read_csv():
    data_dict1 = isopy.read_csv('array_test_file.csv')
    data_dict2 = isopy.read_csv('array_test_file2.csv')

    assert list(data_dict1.keys()) == ['104Pd', '105Pd', '106Pd']
    assert list(data_dict2.keys()) == ['104Pd', '105Pd', '106Pd']

    assert data_dict1['104Pd'] == ['1.0', '2.0', '3.0', '4.0', '5.0']
    assert data_dict1['105Pd'] == ['10.0', '20.0', '30.0', '40.0', '50.0']
    assert data_dict1['106Pd'] == ['100.0', '200.0', '300.0', '400.0', '500.0']
    assert data_dict2['104Pd'] == ['1.0']
    assert data_dict2['105Pd'] == ['10.0']
    assert data_dict2['106Pd'] == ['100.0']

    isopy.array(data_dict1)
    isopy.array(data_dict2)


def test_read_excel():
    data_dict1 = isopy.read_excel('excel_test_file.xlsx', 0)
    data_dict2 = isopy.read_excel('excel_test_file.xlsx', 'Sheet2')

    assert list(data_dict1.keys()) == ['104Pd', '105Pd', '106Pd']
    assert list(data_dict2.keys()) == ['104Pd', '105Pd', '106Pd']

    for key in data_dict1: assert len(data_dict1[key]) == 5
    for key in data_dict2: assert len(data_dict2[key]) == 1

    assert data_dict1['104Pd'] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert data_dict1['105Pd'] == [10.0, 20.0, 30.0, 40.0, 50.0]
    assert data_dict1['106Pd'] == [100.0, 200.0, 300.0, 400.0, 500.0]
    assert data_dict2['104Pd'] == [1.0]
    assert data_dict2['105Pd'] == [10.0]
    assert data_dict2['106Pd'] == [100.0]

    sheet_dict = isopy.read_excel('excel_test_file.xlsx')
    assert len(sheet_dict) == 2
    assert list(sheet_dict) == ['Sheet1', 'Sheet2']

    isopy.array(data_dict1)
    isopy.array(data_dict2)


def test_import_exp():
    data = isopy.import_exp('001_blk.exp')

    assert data.info['Sample ID'] == 'blk'
    assert data.info['Instrument'] == 'PROTEUS'

    assert len(data.cycle) == 50

    assert list(data.isotope_data.keys()) == [1]

    iso_data = data.isotope_data[1]

    assert list(iso_data.keys()) == ['68Zn', '70Ge', '72Ge', '73Ge', '74Ge', '75As', '76Ge', '78Se']

    isopy.array(iso_data)