from isopy.dtypes import *
import pytest

class Test_IsopyString(object):
    def test_creation(self):
        #ElementString
        self.creation(ElementString, 'Pd', ['pd', 'pD', 'PD'],*['', '1', '1Pd', '/Pd', '_Pd'])

        #IsotopeString
        self.creation(IsotopeString, '105Pd', ['105PD', '105pd', '105pD', 'Pd105', 'Pd105', 'pd105', 'pD105'], *['','Pd', '105', '_Pd105', 'P105D', '105Pd/108Pd'])


        #RatioString
        self.creation(RatioString, 'Pd/Pd',
                      ['PD/Pd', 'pd/Pd', 'pD/Pd', 'Pd/pd', 'Pd/PD',
                            'pd/pD'], *['','Pd', '105Pd', '105/105Pd', '105/108', 'Pd//Pd', '/Pd', 'Pd/'])
        self.creation(RatioString, '105Pd/Pd',
                      ['105PD/Pd', '105pd/Pd', '105pD/Pd', 'Pd105/Pd',
                            'PD105/Pd', 'pd105/Pd', '105Pd/pd', '105Pd/PD',
                            'pd105/pD'])
        self.creation(RatioString, 'Pd/105Pd',
                      ['PD/105Pd', 'pd/105Pd', 'pD/105Pd', 'Pd/Pd105',
                            'Pd/PD105', 'Pd/pd105', 'Pd/105pd', 'Pd/105PD',
                            'pd/105pd', 'PD/pd105'])
        self.creation(RatioString, '105Pd/108Pd',
                      ['105PD/108Pd', '105pd/108Pd', '105pD/108Pd', 'Pd105/108Pd',
                            'PD105/108Pd', 'pd105/108Pd', '105Pd/108pd', '105Pd/pd108',
                            'pd105/pd108'])

    def creation(self, string_class, correct_format, strings_incorrect_format, *invalid_strings):
        #Make sure the correcttly formatted string can be created and is not reformatted
        formatted_correct_format = string_class(correct_format, False)
        assert formatted_correct_format == correct_format
        assert formatted_correct_format == string_class(correct_format)
        assert isinstance(formatted_correct_format, str)
        assert isinstance(formatted_correct_format, string_class)
        assert type(formatted_correct_format) == string_class

        for string in strings_incorrect_format:
            formatted_string = string_class(string)
            assert isinstance(formatted_string, string_class)
            assert isinstance(formatted_string, str)
            assert formatted_string != string
            assert formatted_string == correct_format
            assert formatted_string == formatted_correct_format
            assert type(formatted_string) == string_class

            #Should fail since string is not in the correct format
            with pytest.raises(ValueError):
                formatted_string = string_class(string, False)
                print('"{}" was formatted to "{}" when it should have raised a ValueError'.format(string, formatted_string))


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
        assert type(isotope.mass_number) == int

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

        with pytest.raises(ValueError):
            formatted_list = ElementList(unformatted_list, False)
            print('"{}" was formatted to "{}" when it should have raised a ValueError'.format(unformatted_list,
                                                                                              formatted_list))
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

        assert ratio_list.filter(numerator = '105Pd') == ['105Pd/Pd', '105Pd/Pd']
        assert ratio_list.filter(numerator_element_symbol = ['Ru', 'Cd']) == ['101Ru/Ru', '111Cd/Cd']

        assert ratio_list.filter(denominator='Pd') == ['105Pd/Pd', '105Pd/Pd']
        assert ratio_list.filter(denominator_element_symbol_not = 'Pd') == ['101Ru/Ru', '111Cd/Cd']

        assert ratio_list.filter(n_mass_number_gt = 101, d_element_symbol_not = 'Cd') == ['105Pd/Pd', '105Pd/Pd']

    def test_isotope_get_has(self):
        isotope_list = IsotopeList(['101Ru', '105Pd', '111Cd', '105Pd'])

        assert isotope_list.mass_numbers == [101, 105, 111, 105]
        assert isotope_list.get_mass_numbers() == [101, 105, 111, 105]
        assert type(isotope_list.get_mass_numbers()) == list

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







