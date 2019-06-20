from isopy._dtypes import *
import isopy
import pytest

class Test_IsopyString(object):
    def test_creation(self):
        #ElementString
        self.creation(ElementString, str, 'Pd', ['pd', 'pD', 'PD'],['', '1', '1Pd', '/Pd', '_Pd'])

        #IsotopeString
        self.creation(IsotopeString, str, '105Pd', ['105PD', '105pd', '105pD', 'Pd105', 'Pd105', 'pd105', 'pD105'], ['','Pd', '105', '_Pd105', 'P105D', '105Pd/108Pd'])

        #MassInteger
        self.creation(MassInteger, int, 105, ['105'], ['pd105', 'onehundred and five'])

        #RatioString
        self.creation(RatioString, str, 'Pd/Pd',
                      ['PD/Pd', 'pd/Pd', 'pD/Pd', 'Pd/pd', 'Pd/PD',
                            'pd/pD'], ['','Pd', '105Pd', 'Pd//Pd', '/Pd', 'Pd/'])
        self.creation(RatioString, str, '105Pd/Pd',
                      ['105PD/Pd', '105pd/Pd', '105pD/Pd', 'Pd105/Pd',
                            'PD105/Pd', 'pd105/Pd', '105Pd/pd', '105Pd/PD',
                            'pd105/pD'])
        self.creation(RatioString, str, 'Pd/105Pd',
                      ['PD/105Pd', 'pd/105Pd', 'pD/105Pd', 'Pd/Pd105',
                            'Pd/PD105', 'Pd/pd105', 'Pd/105pd', 'Pd/105PD',
                            'pd/105pd', 'PD/pd105'])
        self.creation(RatioString, str, '105Pd/108Pd',
                      ['105PD/108Pd', '105pd/108Pd', '105pD/108Pd', 'Pd105/108Pd',
                            'PD105/108Pd', 'pd105/108Pd', '105Pd/108pd', '105Pd/pd108',
                            'pd105/pd108'])

    def creation(self, string_class, base_type, correct_format, strings_incorrect_format, invalid_strings = []):
        #Make sure the correcttly formatted string can be created and is not reformatted
        formatted_correct_format = string_class(correct_format)
        assert formatted_correct_format == correct_format
        assert formatted_correct_format == string_class(correct_format)
        assert isinstance(formatted_correct_format, base_type)
        assert isinstance(formatted_correct_format, string_class)
        assert type(formatted_correct_format) == string_class

        for string in strings_incorrect_format:
            formatted_string = string_class(string)
            assert isinstance(formatted_string, string_class)
            assert isinstance(formatted_string, base_type)
            assert formatted_string != string
            assert formatted_string == correct_format
            assert formatted_string == formatted_correct_format
            assert type(formatted_string) == string_class

            for string in invalid_strings:
                with pytest.raises(ValueError):
                    formatted_string = string_class(string)
                    print('"{}" was formatted to "{}" when it should have raised a ValueError'.format(string,
                                                                                                          formatted_string))

    def test_isotope_attributes(self):
        isotope = IsotopeString('105Pd')
        assert hasattr(isotope, 'mass_number')
        assert hasattr(isotope, 'element_symbol')

        assert isotope.mass_number == 105
        assert type(isotope.mass_number) == MassInteger

        assert isotope.element_symbol == 'Pd'
        assert type(isotope.element_symbol) == ElementString

    def test_isotope_contains(self):
        isotope = IsotopeString('105Pd')

        assert 105 in isotope
        assert 'Pd' in isotope
        assert 'PD' in isotope

        assert 106 not in isotope
        assert 'Ru' not in isotope

    def test_ratio_attributes(self):
        ratio = RatioString('105Pd/108Pd')
        assert hasattr(ratio, 'numerator')
        assert hasattr(ratio, 'denominator')

        assert ratio.numerator == '105Pd'
        assert type(ratio.numerator) == IsotopeString

        assert ratio.denominator == '108Pd'
        assert type(ratio.denominator) == IsotopeString

        ratio = RatioString('Ru/Pd')
        assert hasattr(ratio, 'numerator')
        assert hasattr(ratio, 'denominator')

        assert ratio.numerator == 'Ru'
        assert type(ratio.numerator) == ElementString

        assert ratio.denominator == 'Pd'
        assert type(ratio.denominator) == ElementString

    def test_ratio_contains(self):
        ratio  = RatioString('105Pd/108Pd')

        assert '105Pd' in ratio
        assert 'pd105' in ratio
        assert '108Pd' in ratio
        assert 'pd108' in ratio

        assert '105Ru' not in ratio
        assert '104Pd' not in ratio
        assert 105 not in ratio
        assert 'Pd' not in ratio

    def test_string_manipulation(self):
        element = ElementString('Pd')
        isotope = IsotopeString('105Pd')
        ratio = RatioString('105Pd/108Pd')

        assert type(element.upper()) == str
        assert type(isotope.lower()) == str
        assert type(ratio.capitalize()) == str
        assert type(isotope[:3]) == str

        assert type(element[0]) == str
        assert type(element[3:]) == str

        assert type(ratio.split('/')[0]) == str

    def test_integer_manipulation(self):
        mass = MassInteger(105)

        assert type(mass) == MassInteger
        assert mass == 105
        assert type(mass+1) != MassInteger
        assert type(mass+1) == int

        mass+=1
        assert type(mass) == int
        assert type(mass) != MassInteger


class Test_IsopyList(object):
    def test_creation(self):
        correct_list = ['Ru', 'Pd', 'Cd']
        unformatted_list = ['ru', 'PD', 'cD']
        invalid_lists = (['Ru', '105Pd', 'Cd'],)

        isopy_correct_list = ElementList(correct_list)
        isopy_unformatted_list = ElementList(unformatted_list)

        assert isinstance(isopy_correct_list, list)
        assert isinstance(isopy_unformatted_list, list)

        assert isinstance(isopy_correct_list, ElementList)
        assert isinstance(isopy_unformatted_list, ElementList)

        assert type(isopy_correct_list) == ElementList
        assert type(isopy_unformatted_list) == ElementList

        assert isopy_correct_list == correct_list
        assert isopy_unformatted_list == correct_list
        assert isopy_unformatted_list != unformatted_list

        for invalid_list in invalid_lists:
            with pytest.raises(ValueError):
                formatted_list = ElementList(invalid_list)
                print('"{}" was formatted to "{}" when it should have raised a ValueError'.format(invalid_list,
                                                                                                         formatted_list))

        with pytest.raises(ValueError):
            formatted_list = RatioList(['101Ru/Pd', 'Pd/Pd', '111Cd/Pd'])
            print('"{}" was formatted to "{}" when it should have raised a ValueError'.format(invalid_list,
                                                                                              formatted_list))

    def test_compare(self):
        element_list = ['Ru', 'Pd', 'Cd']
        rearranged_list = ['Pd', 'Ru','Cd']
        different_lists = (['Pd', 'Cd'], ['Mo', 'Ru', 'Pd', 'Cd'])

        isopy_correct_list = ElementList(element_list)
        isopy_rearranged_list = ElementList(rearranged_list)

        assert isopy_correct_list == isopy_rearranged_list

        for different_list in different_lists:
            isopy_different_list = ElementList(different_list)
            assert isopy_correct_list != isopy_different_list

    def test_contain(self):
        element_list = ElementList(['Ru', 'Pd', 'Cd'])
        assert 'pd' in element_list
        assert '105Pd' not in element_list
        assert ' Mo' not in element_list

        assert ['Pd', 'cd'] in element_list
        assert ['Pd', 'Mo'] not in element_list

    def test_getitem(self):
        element_list = ElementList(['Ru', 'Pd', 'Cd'])

        assert element_list[1] == 'Pd'
        assert element_list[:2] == ['Ru', 'Pd']

        assert type(element_list[0]) == ElementString
        assert type(element_list[:2]) == ElementList

        assert element_list is not element_list[:]

    def test_insert(self):
        element_list = ElementList(['Ru', 'Pd', 'Cd'])

        element_list.insert(0, 'mo')
        assert element_list == ['Mo', 'Ru', 'Pd', 'Cd']

        ratio_list = RatioList(['101Ru/Pd', '105Pd/Pd', '111Cd/Pd'])
        ratio_list.insert(0,'96Mo/Pd')

        with pytest.raises(ValueError):
            element_list.insert(0,'Zr/Pd')

    def test_index(self):
        element_list = ElementList(['Ru', 'Pd', 'Cd'])
        assert element_list.index('pd') == 1
        with pytest.raises(ValueError):
            element_list.index('Zr')

    def test_remove(self):
        element_list = ElementList(['Ru', 'Pd', 'Cd'])
        element_list.remove('ru')
        assert element_list == ['Pd', 'Cd']

    def test_copy(self):
        element_list = ElementList(['Ru', 'Pd', 'Cd'])
        copy_list = element_list.copy()

        assert type(copy_list) == ElementList
        assert copy_list == element_list
        assert copy_list is not element_list

        copy_list = element_list[:]

        assert type(copy_list) == ElementList
        assert copy_list == element_list
        assert copy_list is not element_list

    def test_divide(self):
        element_list = ElementList(['Ru', 'Pd', 'Cd'])
        isotope_list = IsotopeList(['101Ru', '105Pd', '111Cd'])


        ratio_list = element_list / 'Pd'
        assert ratio_list == ['Ru/Pd', 'Pd/Pd', 'Cd/Pd']

        ratio_list = element_list / isotope_list
        assert ratio_list == ['Ru/101Ru', 'Pd/105Pd', 'Cd/111Cd']

        ratio_list = isotope_list / 'Pd'
        assert ratio_list == ['101Ru/Pd', '105Pd/Pd', '111Cd/Pd']

        ratio_list = isotope_list / element_list
        assert ratio_list == ['101Ru/Ru', '105Pd/Pd', '111Cd/Cd']

        with pytest.raises(TypeError):
            ratio_list / 'Pd'

        with pytest.raises(ValueError):
            isotope_list / ['Pd', 'Cd']
            element_list / ['Pd', 'Cd']

    def test_element_filter(self):
        element_list = ElementList(['Ru', 'Pd', 'Cd', 'Pd'])

        assert element_list.filter('ru') == ['Ru']
        assert element_list.filter(['RU', 'Cd']) == ['Ru', 'Cd']
        assert element_list.filter(['cd', 'Ru']) == ['Ru', 'Cd']

        assert element_list.filter(element_symbol_not='Pd') == ['Ru', 'Cd']
        assert element_list.filter(element_symbol_not=['cd', 'ru']) == ['Pd', 'Pd']

    def test_mass_filter(self):
        mass_list = MassList([102, 104, 105, 106, 108, 110])

        assert mass_list.filter('104') == [104]
        assert mass_list.filter(mass_number_not = [106,'108',110, 111]) == [102,104,105]

        assert mass_list.filter(mass_number_gt=106) == [108, 110]
        assert mass_list.filter(mass_number_ge=106) == [106, 108, 110]

        assert mass_list.filter(mass_number_lt=106) == [102, 104, 105]
        assert mass_list.filter(mass_number_le=106) == [102, 104, 105, 106]

    def test_isotope_filter(self):
        isotope_list = IsotopeList(['101Ru', '105Pd', '111Cd', '105Pd'])

        assert isotope_list.filter('pd105') == ['105Pd', '105Pd']
        assert isotope_list.filter(['111cd', 'ru101']) == ['101Ru', '111Cd']

        assert isotope_list.filter(isotope_not='111cd') == ['101Ru', '105Pd', '105Pd']
        assert isotope_list.filter(isotope_not=['105pd','111cd']) == ['101Ru']

        assert isotope_list.filter(mass_number=105) ==['105Pd', '105Pd']
        assert isotope_list.filter(mass_number=[105, 111]) == ['105Pd', '111Cd', '105Pd']

        assert isotope_list.filter(mass_number_gt = 105) == ['111Cd']
        assert isotope_list.filter(mass_number_lt = 111) == ['101Ru', '105Pd', '105Pd']

    def test_ratio_filter(self):
        ratio_list = RatioList(['101Ru/Ru', '105Pd/Pd', '111Cd/Cd', '105Pd/Pd'])

        assert ratio_list.filter('105Pd/Pd') == ['105Pd/Pd', '105Pd/Pd']
        assert ratio_list.filter(['101Ru/Ru', '111Cd/Cd']) == ['101Ru/Ru', '111Cd/Cd']

        assert ratio_list.filter(ratio_not = '105Pd/Pd') == ['101Ru/Ru', '111Cd/Cd']
        assert ratio_list.filter(ratio_not = ['101Ru/Ru', '111Cd/Cd']) == ['105Pd/Pd', '105Pd/Pd']

        assert ratio_list.filter(numerator_isotope = '105Pd') == ['105Pd/Pd', '105Pd/Pd']
        assert ratio_list.filter(numerator_element_symbol = ['Ru', 'Cd']) == ['101Ru/Ru', '111Cd/Cd']

        assert ratio_list.filter(denominator_element_symbol='Pd') == ['105Pd/Pd', '105Pd/Pd']
        assert ratio_list.filter(denominator_element_symbol_not = 'Pd') == ['101Ru/Ru', '111Cd/Cd']

        assert ratio_list.filter(n_mass_number_gt = 101, d_element_symbol_not = 'Cd') == ['105Pd/Pd', '105Pd/Pd']

    def test_isotope_get_has(self):
        isotope_list = IsotopeList(['101Ru', '105Pd', '111Cd', '105Pd'])

        assert isotope_list.mass_numbers == [101, 105, 111, 105]
        assert isotope_list.get_mass_numbers() == [101, 105, 111, 105]
        assert type(isotope_list.get_mass_numbers()) == MassList

        assert isotope_list.element_symbols == ['Ru', 'Pd', 'Cd', 'Pd']
        assert isotope_list.get_element_symbols() == ['Ru', 'Pd', 'Cd', 'Pd']
        assert type(isotope_list.get_element_symbols()) == ElementList

    def test_ratio_get_has(self):
        ratio_list = RatioList(['101Ru/Ru', '105Pd/Pd', '111Cd/Cd', '105Pd/Pd'])

        assert ratio_list.numerators == ['101Ru', '105Pd', '111Cd', '105Pd']
        assert ratio_list.get_numerators() == ['101Ru', '105Pd', '111Cd', '105Pd']
        assert type(ratio_list.get_numerators()) == IsotopeList

        assert ratio_list.denominators == ['Ru', 'Pd', 'Cd', 'Pd']
        assert ratio_list.get_denominators() == ['Ru', 'Pd', 'Cd', 'Pd']
        assert type(ratio_list.get_denominators()) == ElementList

        assert ratio_list.has_common_denominator() == False

        ratio_list = RatioList(['101Ru/108Pd', '105Pd/108Pd', '111Cd/108Pd', '105Pd/108Pd'])
        assert ratio_list.has_common_denominator() == True
        assert ratio_list.get_common_denominator() == '108Pd'
        assert type(ratio_list.get_common_denominator()) == IsotopeString

    def test_modification(self):
        element_list = ElementList(['Ru', 'Pd', 'Cd', 'Pd'])

        assert type(element_list+element_list) == list
        assert type(element_list + ['a']) == list


class Test_IsopyArray(object):
    def test_creation_dict(self):
        #creating array from dict
        dict_data = {'104Pd': [1,2,3,4,5], '105Pd': [10, 20, 30, 40, 50], '106Pd': [100, 200, 300, 400, 500]}

        array = IsotopeArray(dict_data)
        assert type(array) == IsotopeArray
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == dict_data['104Pd'])
        assert np.all(array['105Pd'] == dict_data['105Pd'])
        assert np.all(array['106Pd'] == dict_data['106Pd'])
        assert array.ndim == 1
        assert array.size == 5

        array = IsotopeArray(dict_data, ndim = -1)
        assert type(array) == IsotopeArray
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == dict_data['104Pd'])
        assert np.all(array['105Pd'] == dict_data['105Pd'])
        assert np.all(array['106Pd'] == dict_data['106Pd'])
        assert array.ndim == 1
        assert array.size == 5

        with pytest.raises(ValueError):
            array = IsotopeArray(dict_data, ndim = 0)

        with pytest.raises(ValueError):
            array = RatioArray(dict_data)

        dict_data = {'104Pd': 1, '105Pd': [10], '106Pd': [100]}

        array = IsotopeArray(dict_data)
        assert type(array) == IsotopeArray
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == dict_data['104Pd'])
        assert np.all(array['105Pd'] == dict_data['105Pd'])
        assert np.all(array['106Pd'] == dict_data['106Pd'])
        assert array.ndim == 1
        assert array.size == 1

        array = IsotopeArray(dict_data, ndim = 1)
        assert type(array) == IsotopeArray
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == dict_data['104Pd'])
        assert np.all(array['105Pd'] == dict_data['105Pd'])
        assert np.all(array['106Pd'] == dict_data['106Pd'])
        assert array.ndim == 1
        assert array.size == 1

        array = IsotopeArray(dict_data, ndim = 0)
        assert type(array) == IsotopeArray
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == dict_data['104Pd'])
        assert np.all(array['105Pd'] == dict_data['105Pd'])
        assert np.all(array['106Pd'] == dict_data['106Pd'])
        assert array.ndim == 0
        assert array.size == 1

        array = IsotopeArray(dict_data, ndim=-1)
        assert type(array) == IsotopeArray
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == dict_data['104Pd'])
        assert np.all(array['105Pd'] == dict_data['105Pd'])
        assert np.all(array['106Pd'] == dict_data['106Pd'])
        assert array.ndim == 0
        assert array.size == 1

        dict_data = {'104Pd': 1, '105Pd': 10, '106Pd': 100}

        array = IsotopeArray(dict_data)
        assert type(array) == IsotopeArray
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == dict_data['104Pd'])
        assert np.all(array['105Pd'] == dict_data['105Pd'])
        assert np.all(array['106Pd'] == dict_data['106Pd'])
        assert array.ndim == 0
        assert array.size == 1

        array = IsotopeArray(dict_data, ndim=0)
        assert type(array) == IsotopeArray
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == dict_data['104Pd'])
        assert np.all(array['105Pd'] == dict_data['105Pd'])
        assert np.all(array['106Pd'] == dict_data['106Pd'])
        assert array.ndim == 0
        assert array.size == 1

        array = IsotopeArray(dict_data, ndim=-1)
        assert type(array) == IsotopeArray
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == dict_data['104Pd'])
        assert np.all(array['105Pd'] == dict_data['105Pd'])
        assert np.all(array['106Pd'] == dict_data['106Pd'])
        assert array.ndim == 0
        assert array.size == 1

        array = IsotopeArray(dict_data, ndim = 1)
        assert type(array) == IsotopeArray
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == dict_data['104Pd'])
        assert np.all(array['105Pd'] == dict_data['105Pd'])
        assert np.all(array['106Pd'] == dict_data['106Pd'])
        assert array.ndim == 1
        assert array.size == 1

        dict_data = {'104Pd': [1, 2, 3], '105Pd': [10, 20, 30, 40, 50], '106Pd': [100, 200, 300, 400, 500]}

        with pytest.raises(ValueError):
            array = IsotopeArray(dict_data)

    def text_creation_list(self):
        list_data = [[1,2,3,4,5], [10, 20, 30, 40, 50], [100, 200, 300, 400, 500]]
        keys = ['104Pd', '105Pd', '106Pd']

        array = IsotopeArray(list_data, keys = keys)
        assert type(array) == IsotopeArray
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == list_data[0])
        assert np.all(array['105Pd'] == list_data[1])
        assert np.all(array['106Pd'] == list_data[2])
        assert array.ndim == 1
        assert array.size == 5

        array = IsotopeArray(list_data, keys=keys, ndim = 1)
        assert type(array) == IsotopeArray
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == list_data[0])
        assert np.all(array['105Pd'] == list_data[1])
        assert np.all(array['106Pd'] == list_data[2])
        assert array.ndim == 1
        assert array.size == 5

        with pytest.raises(ValueError):
            array = IsotopeArray(list_data, keys=keys, ndim=0)

        with pytest.raises(ValueError):
            array = RatioArray(list_data, keys=keys)

        list_data = [1, [10], [100]]

        array = IsotopeArray(list_data, keys=keys)
        assert type(array) == IsotopeArray
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == list_data[0])
        assert np.all(array['105Pd'] == list_data[1])
        assert np.all(array['106Pd'] == list_data[2])
        assert array.ndim == 1
        assert array.size == 1

        array = IsotopeArray(list_data, keys=keys, ndim=1)
        assert type(array) == IsotopeArray
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == list_data[0])
        assert np.all(array['105Pd'] == list_data[1])
        assert np.all(array['106Pd'] == list_data[2])
        assert array.ndim == 1
        assert array.size == 1

        array = IsotopeArray(list_data, keys=keys, ndim=0)
        assert type(array) == IsotopeArray
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == list_data[0])
        assert np.all(array['105Pd'] == list_data[1])
        assert np.all(array['106Pd'] == list_data[2])
        assert array.ndim == 0
        assert array.size == 1

        array = IsotopeArray(list_data, keys=keys, ndim=-1)
        assert type(array) == IsotopeArray
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == list_data[0])
        assert np.all(array['105Pd'] == list_data[1])
        assert np.all(array['106Pd'] == list_data[2])
        assert array.ndim == 0
        assert array.size == 1

        list_data = [1, 10, 100]

        array = IsotopeArray(list_data, keys=keys)
        assert type(array) == IsotopeArray
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == list_data[0])
        assert np.all(array['105Pd'] == list_data[1])
        assert np.all(array['106Pd'] == list_data[2])
        assert array.ndim == 0
        assert array.size == 1

        array = IsotopeArray(list_data, keys=keys, ndim=1)
        assert type(array) == IsotopeArray
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == list_data[0])
        assert np.all(array['105Pd'] == list_data[1])
        assert np.all(array['106Pd'] == list_data[2])
        assert array.ndim == 1
        assert array.size == 1

        array = IsotopeArray(list_data, keys=keys, ndim=0)
        assert type(array) == IsotopeArray
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == list_data[0])
        assert np.all(array['105Pd'] == list_data[1])
        assert np.all(array['106Pd'] == list_data[2])
        assert array.ndim == 0
        assert array.size == 1

        array = IsotopeArray(list_data, keys=keys, ndim=-1)
        assert type(array) == IsotopeArray
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == list_data[0])
        assert np.all(array['105Pd'] == list_data[1])
        assert np.all(array['106Pd'] == list_data[2])
        assert array.ndim == 0
        assert array.size == 1

        list_data = [[1, 2, 3], [10, 20, 30, 40, 50], [100, 200, 300, 400, 500]]
        keys = ['104Pd', '105Pd', '106Pd']
        with pytest.raises(ValueError):
            array = RatioArray(list_data, keys=keys)

        list_data = [[1, 2, 3,4,5], [10, 20, 30, 40, 50], [100, 200, 300, 400, 500]]
        keys = ['104Pd', '105Pd']
        with pytest.raises(ValueError):
            array = RatioArray(list_data, keys=keys)

    def test_creation_numpy(self):
        array_data = np.array([(1,10,100), (2,20,200), (3,30,300), (4,40,400), (5,50,500)],
                        dtype = [('104Pd', 'f8'), ('105Pd', 'f8'), ('106Pd', 'f8')])

        array = IsotopeArray(array_data)
        assert type(array) == IsotopeArray
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == array_data['104Pd'])
        assert np.all(array['105Pd'] == array_data['105Pd'])
        assert np.all(array['106Pd'] == array_data['106Pd'])
        assert array.ndim == 1
        assert array.size == 5

        array = IsotopeArray(array_data, ndim=1)
        assert type(array) == IsotopeArray
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == array_data['104Pd'])
        assert np.all(array['105Pd'] == array_data['105Pd'])
        assert np.all(array['106Pd'] == array_data['106Pd'])
        assert array.ndim == 1
        assert array.size == 5

        with pytest.raises(ValueError):
            array = IsotopeArray(array_data, ndim=0)

        with pytest.raises(ValueError):
            array = RatioArray(array_data)

        array = IsotopeArray(array_data, keys = {'104Pd': '104Ru', '106Pd': '106Cd'})
        assert type(array) == IsotopeArray
        assert array.keys() == ['104Ru', '105Pd', '106Cd']
        assert np.all(array['104Ru'] == array_data['104Pd'])
        assert np.all(array['105Pd'] == array_data['105Pd'])
        assert np.all(array['106Cd'] == array_data['106Pd'])
        assert array.ndim == 1
        assert array.size == 5

        array_data = np.array([(1, 10, 100)],
                              dtype=[('104Pd', 'f8'), ('105Pd', 'f8'), ('106Pd', 'f8')])

        array = IsotopeArray(array_data)
        assert type(array) == IsotopeArray
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == array_data['104Pd'])
        assert np.all(array['105Pd'] == array_data['105Pd'])
        assert np.all(array['106Pd'] == array_data['106Pd'])
        assert array.ndim == 1
        assert array.size == 1

        array = IsotopeArray(array_data, ndim = 1)
        assert type(array) == IsotopeArray
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == array_data['104Pd'])
        assert np.all(array['105Pd'] == array_data['105Pd'])
        assert np.all(array['106Pd'] == array_data['106Pd'])
        assert array.ndim == 1
        assert array.size == 1

        array = IsotopeArray(array_data, ndim = 0)
        assert type(array) == IsotopeArray
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == array_data['104Pd'])
        assert np.all(array['105Pd'] == array_data['105Pd'])
        assert np.all(array['106Pd'] == array_data['106Pd'])
        assert array.ndim == 0
        assert array.size == 1

        array = IsotopeArray(array_data, ndim=-1)
        assert type(array) == IsotopeArray
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == array_data['104Pd'])
        assert np.all(array['105Pd'] == array_data['105Pd'])
        assert np.all(array['106Pd'] == array_data['106Pd'])
        assert array.ndim == 0
        assert array.size == 1

        array_data = np.array((1, 10, 100),
                              dtype=[('104Pd', 'f8'), ('105Pd', 'f8'), ('106Pd', 'f8')])

        array = IsotopeArray(array_data)
        assert type(array) == IsotopeArray
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == array_data['104Pd'])
        assert np.all(array['105Pd'] == array_data['105Pd'])
        assert np.all(array['106Pd'] == array_data['106Pd'])
        assert array.ndim == 0
        assert array.size == 1

        array = IsotopeArray(array_data, ndim=1)
        assert type(array) == IsotopeArray
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == array_data['104Pd'])
        assert np.all(array['105Pd'] == array_data['105Pd'])
        assert np.all(array['106Pd'] == array_data['106Pd'])
        assert array.ndim == 1
        assert array.size == 1

        array = IsotopeArray(array_data, ndim=0)
        assert type(array) == IsotopeArray
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == array_data['104Pd'])
        assert np.all(array['105Pd'] == array_data['105Pd'])
        assert np.all(array['106Pd'] == array_data['106Pd'])
        assert array.ndim == 0
        assert array.size == 1

        array = IsotopeArray(array_data, ndim=-1)
        assert type(array) == IsotopeArray
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == array_data['104Pd'])
        assert np.all(array['105Pd'] == array_data['105Pd'])
        assert np.all(array['106Pd'] == array_data['106Pd'])
        assert array.ndim == 0
        assert array.size == 1

    def test_creation_isopy_array(self):
        dict_data = {'104Pd': [1, 2, 3, 4, 5], '105Pd': [10, 20, 30, 40, 50], '106Pd': [100, 200, 300, 400, 500]}

        array = IsotopeArray(dict_data)

        array2 = IsotopeArray(array)
        assert array is not array2
        assert array.keys() == dict_data.keys()
        assert array.keys() == array2.keys()
        assert np.all(array['104Pd'] == array2['104Pd'])
        assert np.all(array['105Pd'] == array2['105Pd'])
        assert np.all(array['106Pd'] == array2['106Pd'])

        array['104Pd'][0] = 1000
        assert array['104Pd'][0] == 1000
        assert array2['104Pd'][0] == 1

        dict_data = {'104Pd': [1], '105Pd': [10], '106Pd': [100]}
        array = IsotopeArray(dict_data)
        array2 = IsotopeArray(array, ndim=0)
        assert array.keys() == dict_data.keys()
        assert array.keys() == array2.keys()
        assert array.ndim == 1
        assert array2.ndim == 0
        assert array is not array2
        array['104Pd'][0] = 1000
        assert array2['104Pd'] == 1

        array2 = IsotopeArray(array, keys = {'104Pd': '104Ru', '106Pd': '106Cd'})
        assert array is not array2
        assert array.keys() == dict_data.keys()
        assert array2.keys() == ['104Ru', '105Pd', '106Cd']
        assert array.keys() != array2.keys()
        assert np.all(array['104Pd'] == array2['104Ru'])
        assert np.all(array['105Pd'] == array2['105Pd'])
        assert np.all(array['106Pd'] == array2['106Cd'])

    def test_creation_file(self):
        dict_data = {'104Pd': [1, 2, 3, 4, 5], '105Pd': [10, 20, 30, 40, 50], '106Pd': [100, 200, 300, 400, 500]}
        array = IsotopeArray(filepath='array_test_file.csv')
        assert type(array) == IsotopeArray
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == dict_data['104Pd'])
        assert np.all(array['105Pd'] == dict_data['105Pd'])
        assert np.all(array['106Pd'] == dict_data['106Pd'])
        assert array.ndim == 1
        assert array.size == 5

        dict_data = {'104Pd': [1], '105Pd': [10], '106Pd': [100]}
        array = IsotopeArray(filepath='array_test_file2.csv')
        assert type(array) == IsotopeArray
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == dict_data['104Pd'])
        assert np.all(array['105Pd'] == dict_data['105Pd'])
        assert np.all(array['106Pd'] == dict_data['106Pd'])
        assert array.ndim == 1
        assert array.size == 1

        array = IsotopeArray(filepath='array_test_file2.csv', ndim = 0)
        assert type(array) == IsotopeArray
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == dict_data['104Pd'])
        assert np.all(array['105Pd'] == dict_data['105Pd'])
        assert np.all(array['106Pd'] == dict_data['106Pd'])
        assert array.ndim == 0
        assert array.size == 1

        array = IsotopeArray(filepath='array_test_file2.csv', ndim=-1)
        assert type(array) == IsotopeArray
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == dict_data['104Pd'])
        assert np.all(array['105Pd'] == dict_data['105Pd'])
        assert np.all(array['106Pd'] == dict_data['106Pd'])
        assert array.ndim == 0
        assert array.size == 1

    def test_creation_empty(self):
        array = IsotopeArray(size = 5, keys = ['104Pd', '105Pd', '106Pd'])
        assert array.size == 5
        assert array.ndim == 1
        assert np.all(array['104Pd'] == 0)
        assert np.all(array['105Pd'] == 0)
        assert np.all(array['106Pd'] == 0)

        array = IsotopeArray(size=1, keys=['104Pd', '105Pd', '106Pd'])
        assert array.size == 1
        assert array.ndim == 1
        assert np.all(array['104Pd'] == 0)
        assert np.all(array['105Pd'] == 0)
        assert np.all(array['106Pd'] == 0)

        array = IsotopeArray(size=1, keys=['104Pd', '105Pd', '106Pd'], ndim = 0)
        assert array.size == 1
        assert array.ndim == 0
        assert np.all(array['104Pd'] == 0)
        assert np.all(array['105Pd'] == 0)
        assert np.all(array['106Pd'] == 0)

        array = IsotopeArray(size=1, keys=['104Pd', '105Pd', '106Pd'], ndim=-1)
        assert array.size == 1
        assert array.ndim == 0
        assert np.all(array['104Pd'] == 0)
        assert np.all(array['105Pd'] == 0)
        assert np.all(array['106Pd'] == 0)


    def test_filter(self):
        pass

    def test_numpy_statistics(self):
        return None
        dict_data = {'104Pd': [1, 2, 3, 4, np.nan], '105Pd': [10, 20, 30, 40, 50], '106Pd': [100, 200, 300, 400, 500]}
        array = IsotopeArray(dict_data)
        np.nanmean(array)

        for func in [np.mean, np.var, np.std]:
            array = IsotopeArray(dict_data)
            result = func(array)
            assert type(result) == IsotopeArray
            assert result.ndim == 0
            assert result.size == 1
            np.testing.assert_array_equal(result['104Pd'], func(dict_data['104Pd']))
            np.testing.assert_array_equal(result['105Pd'], func(dict_data['105Pd']))
            np.testing.assert_array_equal(result['106Pd'], func(dict_data['106Pd']))

        for func in [np.max, np.min, np.amax, np.amin, np.ptp]:
            print(func)
            array = IsotopeArray(dict_data)
            result = func(array)
            assert type(result) == IsotopeArray
            assert result.ndim == 0
            assert result.size == 1
            np.testing.assert_array_equal(result['104Pd'], func(dict_data['104Pd']))
            np.testing.assert_array_equal(result['105Pd'], func(dict_data['105Pd']))
            np.testing.assert_array_equal(result['106Pd'], func(dict_data['106Pd']))

    def test_isopy_statistics(self):
        dict_data = {'104Pd': [1, 2, 3, 4, np.nan], '105Pd': [10, 20, 30, 40, 50], '106Pd': [100, 200, 300, 400, 500]}

        for func in [isopy.mean, isopy.var, isopy.std, isopy.median, isopy.nanmean, isopy.nanvar, isopy.nanstd, isopy.nanmedian]:
            array = IsotopeArray(dict_data)
            result = func(array)
            assert type(result) == IsotopeArray
            assert result.ndim == 0
            assert result.size == 1
            np.testing.assert_array_equal(result['104Pd'], func(dict_data['104Pd']))
            np.testing.assert_array_equal(result['105Pd'], func(dict_data['105Pd']))
            np.testing.assert_array_equal(result['106Pd'], func(dict_data['106Pd']))

        array = IsotopeArray(dict_data)
        result = isopy.average(array, weights=[1,2,3,4,5])
        assert type(result) == IsotopeArray
        assert result.ndim == 0
        assert result.size == 1
        np.testing.assert_array_equal(result['104Pd'], isopy.average(dict_data['104Pd'], weights=[1,2,3,4,5]))
        np.testing.assert_array_equal(result['105Pd'], isopy.average(dict_data['105Pd'], weights=[1,2,3,4,5]))
        np.testing.assert_array_equal(result['106Pd'], isopy.average(dict_data['106Pd'], weights=[1,2,3,4,5]))

        for func in [isopy.max, isopy.min, isopy.amax, isopy.amin, isopy.ptp, isopy.nanmax, isopy.nanmin]:
            array = IsotopeArray(dict_data)
            result = func(array)
            assert type(result) == IsotopeArray
            assert result.ndim == 0
            assert result.size == 1
            np.testing.assert_array_equal(result['104Pd'], func(dict_data['104Pd']))
            np.testing.assert_array_equal(result['105Pd'], func(dict_data['105Pd']))
            np.testing.assert_array_equal(result['106Pd'], func(dict_data['106Pd']))

        for func in [isopy.percentile, isopy.percentile]:
            array = IsotopeArray(dict_data)
            result = func(array, 65)
            assert type(result) == IsotopeArray
            assert result.ndim == 0
            assert result.size == 1
            np.testing.assert_array_equal(result['104Pd'], func(dict_data['104Pd'], 65))
            np.testing.assert_array_equal(result['105Pd'], func(dict_data['105Pd'], 65))
            np.testing.assert_array_equal(result['106Pd'], func(dict_data['106Pd'], 65))

        for func in [isopy.quantile, isopy.nanquantile]:
            array = IsotopeArray(dict_data)
            result = func(array, 0.65)
            assert type(result) == IsotopeArray
            assert result.ndim == 0
            assert result.size == 1
            np.testing.assert_array_equal(result['104Pd'], func(dict_data['104Pd'], 0.65))
            np.testing.assert_array_equal(result['105Pd'], func(dict_data['105Pd'], 0.65))
            np.testing.assert_array_equal(result['106Pd'], func(dict_data['106Pd'], 0.65))


    def test_add_etc(self):
        pass

    def test_save(self):
        pass



