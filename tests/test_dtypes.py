from isopy import core as dt
from isopy.core import *
import isopy
import isopy.toolbox.np_func as ipf
import numpy as np
import pytest


def assert_isopyarray_equal_ndarray(arr1, arr2, **kwargs):
    assert isinstance(arr1, IsopyArray)
    assert not isinstance(arr2, IsopyArray)
    arr2 = np.asarray(arr2)
    assert (arr1.ndim + 1) == arr2.ndim
    assert arr1.ncols == arr2.shape[0]
    keys = arr1.keys()
    for i in range(arr1.ncols):
        try:
            np.testing.assert_array_equal(arr1[keys[i]], arr2[i])
        except:
            print(kwargs)
            raise


def assert_isoparray_equal_isopyarray(arr1, arr2, **kwargs):
    assert isinstance(arr1, IsopyArray)
    assert isinstance(arr2, IsopyArray)
    assert arr1.keys == arr2.keys()
    for key in arr1.keys():
        np.testing.assert_allclose(arr1[key], arr2[key], 1E-10)


def assert_ndarray_equal_ndarray(arr1, arr2, **kwargs):
    assert not isinstance(arr1, IsopyArray)
    assert not isinstance(arr2, IsopyArray)
    np.testing.assert_allclose(arr1, arr2, 1E-10)


class Test_IsopyString(object):
    def test_creation(self):
        #ElementString
        self.creation(ElementString, str, 'Pd', ['pd', 'pD', 'PD'],['', '1', '1Pd', '/Pd', '_Pd', 'Palladium', 'Pdd'])

        #IsotopeString
        self.creation(IsotopeString, str, '105Pd', ['105PD', '105pd', '105pD', 'Pd105', 'Pd105', 'pd105', 'pD105', '_105Pd'], ['','Pd', '105', 'Pd105a', 'P105D', '105Pd/108Pd'])

        #MassInteger
        self.creation(MassString, str, '105', [105], ['pd105', 'onehundred and five', -105, '-105'])

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
        formatted_correct_format2 = isopy.string(correct_format)
        assert formatted_correct_format == correct_format
        assert formatted_correct_format == string_class(correct_format)
        assert isinstance(formatted_correct_format, base_type)
        assert isinstance(formatted_correct_format, string_class)
        assert isinstance(formatted_correct_format2, string_class)
        assert type(formatted_correct_format) == string_class

        for string in strings_incorrect_format:
            formatted_string = string_class(string)
            assert isinstance(formatted_string, string_class)
            assert isinstance(formatted_string, base_type)
            assert formatted_string == string
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
        assert isotope.mass_number == '105'
        assert type(isotope.mass_number) == MassString

        assert isotope.element_symbol == 'Pd'
        assert type(isotope.element_symbol) == ElementString

    def test_isotope_contains(self):
        isotope = IsotopeString('105Pd')

        assert 105 in isotope
        assert '105' in isotope
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

        assert ratio.denominator == 'Pd108'
        assert type(ratio.denominator) == IsotopeString

        ratio = RatioString('Ru/Pd')
        assert hasattr(ratio, 'numerator')
        assert hasattr(ratio, 'denominator')

        assert ratio.numerator == 'Ru'
        assert type(ratio.numerator) == ElementString

        assert ratio.denominator == 'Pd'
        assert type(ratio.denominator) == ElementString

    def test_ratio_contains(self):
        ratio = RatioString('105Pd/108Pd')

        assert '105Pd' in ratio
        assert 'pd105' in ratio
        assert '108Pd' in ratio
        assert 'pd108' in ratio

        assert '105Ru' not in ratio
        assert '104Pd' not in ratio
        assert 105 not in ratio
        assert 'Pd' not in ratio

    def test_string_manipulation(self):
        ratio = RatioString('108Pd/Pd105')

        assert type(ratio.upper()) == str
        assert type(ratio.lower()) == str
        assert type(ratio.capitalize()) == str
        assert type(ratio[:1]) == str

        assert type(ratio[0]) == str
        assert type(ratio[1:]) == str

        assert type(ratio.split('/')[0]) == str

    def test_integer_manipulation(self):
        integer = 105
        mass = MassString(integer)

        assert type(mass) == MassString
        assert mass == 105

        assert mass > 100
        assert not mass > 105

        assert mass < 110
        assert not mass < 105

        assert mass >= 100
        assert mass >= 105
        assert not mass >= 110

        assert mass <= 110
        assert mass <= 105
        assert not mass <= 100

    def test_divide_to_ratio(self):
        denom = '74Ge'

        for i in [MassString('105'), ElementString('Pd'), IsotopeString('105Pd')]:
            ratio = i / denom

            assert type(ratio) == RatioString
            assert ratio == '{}/{}'.format(i, denom)
            assert ratio.denominator == denom
            assert ratio.numerator == i

            ratio = denom / i

            assert type(ratio) == RatioString
            assert ratio == '{}/{}'.format(denom, i)
            assert ratio.denominator == i
            assert ratio.numerator == denom

    def test_add_to_isotope(self):
        isotope = MassString(101) + 'Ru'
        assert type(isotope) == IsotopeString
        assert isotope == '101Ru'
        assert isotope.mass_number == 101
        assert isotope.element_symbol == 'Ru'

        isotope = 'Ru' + MassString(101)
        assert type(isotope) == IsotopeString
        assert isotope == '101Ru'
        assert isotope.mass_number == 101
        assert isotope.element_symbol == 'Ru'

        isotope = 101 + ElementString('Ru')
        assert type(isotope) == IsotopeString
        assert isotope == '101Ru'
        assert isotope.mass_number == 101
        assert isotope.element_symbol == 'Ru'

        isotope = ElementString('Ru') + 101
        assert type(isotope) == IsotopeString
        assert isotope == '101Ru'
        assert isotope.mass_number == 101
        assert isotope.element_symbol == 'Ru'

    def test_safe_string(self):
        mass = MassString('105')
        element = ElementString('Pd')
        isotope = IsotopeString('105Pd')
        ratio = RatioString('108Pd/105Pd')

        assert mass.identifier() == 'Mass_105'
        assert element.identifier() == 'Element_Pd'
        assert isotope.identifier() == 'Isotope_105Pd'
        assert ratio.identifier() == 'Ratio_108Pd_105Pd'


class Test_IsopyList(object):
    def test_creation(self):
        correct_list = ['Ru', 'Pd', 'Cd']
        correct_list2 = ['Ru', 'Pd', 'Cd', 'Cd']
        unformatted_list = ['ru', 'PD', 'cD']
        invalid_lists = (['Ru', '105Pd', 'Cd'],)

        isopy_correct_list = ElementList(correct_list2)
        isopy_correct_list2 = ElementList(correct_list2, skip_duplicates=True)
        isopy_correct_list3 = isopy.list_(correct_list2, skip_duplicates=True)
        isopy_correct_list4 = isopy.aslist(isopy_correct_list3)
        isopy_unformatted_list = ElementList(unformatted_list)

        assert isinstance(isopy_correct_list, list)
        assert isinstance(isopy_correct_list2, list)
        assert isinstance(isopy_correct_list3, list)
        assert isinstance(isopy_correct_list4, list)
        assert isinstance(isopy_unformatted_list, list)

        assert isinstance(isopy_correct_list, ElementList)
        assert isinstance(isopy_correct_list2, ElementList)
        assert isinstance(isopy_correct_list3, ElementList)
        assert isinstance(isopy_correct_list4, ElementList)
        assert isinstance(isopy_unformatted_list, ElementList)

        assert type(isopy_correct_list) == ElementList
        assert type(isopy_correct_list2) == ElementList
        assert type(isopy_correct_list3) == ElementList
        assert type(isopy_correct_list4) == ElementList
        assert type(isopy_unformatted_list) == ElementList

        assert isopy_correct_list == correct_list2
        assert isopy_correct_list2 == correct_list
        assert isopy_correct_list3 == correct_list
        assert isopy_correct_list4 == correct_list
        assert isopy_unformatted_list == correct_list
        assert isopy_unformatted_list == unformatted_list
        assert isopy_correct_list is not isopy_correct_list2
        assert isopy_correct_list3 is isopy_correct_list4

        with pytest.raises(ValueError):
            ElementList(correct_list2, allow_duplicates=False)
        with pytest.raises(dt.StringDuplicateError):
            ElementList(correct_list2, allow_duplicates = False)

        for invalid_list in invalid_lists:
            with pytest.raises(ValueError):
                formatted_list = ElementList(invalid_list)
                print('"{}" was formatted to "{}" when it should have raised a ValueError'.format(invalid_list,
                                                                                                         formatted_list))

        with pytest.raises(TypeError):
            formatted_list = RatioList(['101Ru/Pd', 'Pd/Pd', '111Cd/Pd'])
            print('"{}" was formatted to "{}" when it should have raised a TypeError'.format(invalid_list,
                                                                                              formatted_list))

    def test_compare(self):
        element_list = ['Ru', 'Pd', 'Cd']
        rearranged_list = ['Pd', 'Ru','Cd']
        different_lists = (['Pd', 'Cd'], ['Mo', 'Ru', 'Pd', 'Cd'], ['Ru', 'Ru', 'Cd', 'Pd'])

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

    def test_append(self):
        list1 = ['Ru', 'Pd', 'Cd']
        element = 'Ru'

        isopy_list = ElementList(list1)
        isopy_list.append(element)
        combinedlist = list1 + [element]
        assert len(isopy_list) == len(combinedlist)
        assert isopy_list == combinedlist
        assert combinedlist == isopy_list

        isopy_list = ElementList(list1)
        isopy_list.append(element, skip_duplicates=True)
        combinedlist = list1
        assert len(isopy_list) == len(combinedlist)
        assert isopy_list == combinedlist
        assert combinedlist == isopy_list

        with pytest.raises(ValueError):
            isopy_list = ElementList(list1)
            isopy_list.append(element, allow_duplicates=False)
        with pytest.raises(dt.StringDuplicateError):
            isopy_list = ElementList(list1)
            isopy_list.append(element, allow_duplicates=False)

    def text_extend(self):
        list1 = ['Ru', 'Pd', 'Cd']
        list2 = ['Mo', 'Ru', 'Pd', 'Cd']

        isopy_list = ElementList(list1)
        isopy_list.extend(list2)
        combinedlist = list1 + list2
        assert len(isopy_list) == len(combinedlist)
        assert isopy_list == combinedlist
        assert combinedlist == isopy_list

        isopy_list = ElementList(list1)
        isopy_list.extend(list2, skip_duplicates=True)
        combinedlist = list1 + [l for l in list2 if l not in list1]
        assert len(isopy_list) == len(combinedlist)
        assert isopy_list == combinedlist
        assert combinedlist == isopy_list

        with pytest.raises(ValueError):
            isopy_list = ElementList(list1)
            isopy_list.append(list2, allow_duplicates=False)
        with pytest.raises(dt.StringDuplicateError):
            isopy_list = ElementList(list1)
            isopy_list.append(list2, allow_duplicates=False)

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

    def test_element_copy(self):
        element_list = ElementList(['Ru', 'Pd', 'Cd', 'Pd'])

        assert element_list.copy() == ['Ru', 'Pd', 'Cd', 'Pd']
        assert element_list.copy() is not element_list
        assert element_list.copy(element_symbol = 'ru') == ['Ru']
        assert element_list.copy(element_symbol = ['RU', 'Cd']) == ['Ru', 'Cd']
        assert element_list.copy(element_symbol = ['cd', 'Ru']) == ['Ru', 'Cd']

        assert element_list.copy(element_symbol_not='Pd') == ['Ru', 'Cd']
        assert element_list.copy(element_symbol_not=['cd', 'ru']) == ['Pd', 'Pd']

    def test_mass_copy(self):
        mass_list = MassList([102, 104, 105, 106, 108, 110])

        assert mass_list.copy() == mass_list
        assert mass_list.copy() is not mass_list
        assert mass_list.copy(mass_number='104') == [104]
        assert mass_list.copy(mass_number_not = [106,'108',110, 111]) == [102,104,105]

        assert mass_list.copy(mass_number_gt=106) == [108, 110]
        assert mass_list.copy(mass_number_ge=106) == [106, 108, 110]

        assert mass_list.copy(mass_number_lt=106) == [102, 104, 105]
        assert mass_list.copy(mass_number_le=106) == [102, 104, 105, 106]

    def test_isotope_copy(self):
        isotope_list = IsotopeList(['101Ru', '105Pd', '111Cd', '105Pd'])

        assert isotope_list.copy() == isotope_list
        assert isotope_list.copy() is not isotope_list
        assert isotope_list.copy(isotope='pd105') == ['105Pd', '105Pd']
        assert isotope_list.copy(isotope=['111cd', 'ru101']) == ['101Ru', '111Cd']

        assert isotope_list.copy(isotope_not='111cd') == ['101Ru', '105Pd', '105Pd']
        assert isotope_list.copy(isotope_not=['105pd','111cd']) == ['101Ru']

        assert isotope_list.copy(mass_number=105) ==['105Pd', '105Pd']
        assert isotope_list.copy(mass_number=[105, 111]) == ['105Pd', '111Cd', '105Pd']

        assert isotope_list.copy(mass_number_gt = 105) == ['111Cd']
        assert isotope_list.copy(mass_number_lt = 111) == ['101Ru', '105Pd', '105Pd']

    def test_ratio_copy(self):
        ratio_list = RatioList(['101Ru/Ru', '105Pd/Pd', '111Cd/Cd', '105Pd/Pd'])

        assert ratio_list.copy() == ratio_list
        assert ratio_list.copy() is not ratio_list
        assert ratio_list.copy(ratio='105Pd/Pd') == ['105Pd/Pd', '105Pd/Pd']
        assert ratio_list.copy(ratio=['101Ru/Ru', '111Cd/Cd']) == ['101Ru/Ru', '111Cd/Cd']

        assert ratio_list.copy(ratio_not = '105Pd/Pd') == ['101Ru/Ru', '111Cd/Cd']
        assert ratio_list.copy(ratio_not = ['101Ru/Ru', '111Cd/Cd']) == ['105Pd/Pd', '105Pd/Pd']

        assert ratio_list.copy(numerator_isotope = '105Pd') == ['105Pd/Pd', '105Pd/Pd']
        assert ratio_list.copy(numerator_element_symbol = ['Ru', 'Cd']) == ['101Ru/Ru', '111Cd/Cd']

        assert ratio_list.copy(denominator_element_symbol='Pd') == ['105Pd/Pd', '105Pd/Pd']
        assert ratio_list.copy(denominator_element_symbol_not = 'Pd') == ['101Ru/Ru', '111Cd/Cd']

        assert ratio_list.copy(n_mass_number_gt = 101, d_element_symbol_not = 'Cd') == ['105Pd/Pd', '105Pd/Pd']

    def test_isotope_get_has(self):
        isotope_list = IsotopeList(['101Ru', '105Pd', '111Cd', '105Pd'])

        assert isotope_list.mass_numbers() == [101, 105, 111, 105]
        assert isotope_list.mass_numbers() == [101, 105, 111, 105]
        assert type(isotope_list.mass_numbers()) == MassList

        assert isotope_list.element_symbols() == ['Ru', 'Pd', 'Cd', 'Pd']
        assert isotope_list.element_symbols() == ['Ru', 'Pd', 'Cd', 'Pd']
        assert type(isotope_list.element_symbols()) == ElementList

    def test_ratio_get_has(self):
        ratio_list = RatioList(['101Ru/Ru', '105Pd/Pd', '111Cd/Cd', '105Pd/Pd'])

        assert ratio_list.numerators() == ['101Ru', '105Pd', '111Cd', '105Pd']
        assert ratio_list.numerators() == ['101Ru', '105Pd', '111Cd', '105Pd']
        assert type(ratio_list.numerators()) == IsotopeList

        assert ratio_list.denominators() == ['Ru', 'Pd', 'Cd', 'Pd']
        assert ratio_list.denominators() == ['Ru', 'Pd', 'Cd', 'Pd']
        assert type(ratio_list.denominators()) == ElementList

        assert ratio_list.has_common_denominator() == False
        with pytest.raises(ValueError):
            ratio_list.get_common_denominator()
        with pytest.raises(dt.NoCommomDenominator):
            ratio_list.get_common_denominator()

        ratio_list = RatioList(['101Ru/108Pd', '105Pd/108Pd', '111Cd/108Pd', '105Pd/108Pd'])
        assert ratio_list.has_common_denominator() == True
        assert ratio_list.get_common_denominator() == '108Pd'
        assert type(ratio_list.get_common_denominator()) == IsotopeString

    def test_add(self):
        isotope_list = IsotopeList(['101Ru', '101Pd', '101Cd'])
        isotope_list2 = IsotopeList(['99Ru', '105Pd', '111Cd'])
        isotope_list3 = IsotopeList(['99Pd', '105Pd', '111Pd'])
        element_list = ElementList(['Ru', 'Pd', 'Cd'])
        mass_list = MassList([99,105,111])

        added = element_list + 101
        assert type(added) == IsotopeList
        assert added == isotope_list

        added = [99,105,111] + element_list
        assert type(added) == IsotopeList
        assert added == isotope_list2

        added = mass_list + 'Pd'
        assert type(added) == IsotopeList
        assert added == isotope_list3

        added = ['Ru', 'Pd', 'Cd'] + mass_list
        assert type(added) == IsotopeList
        assert added == isotope_list2

    def test_bitwise(self):
        one = ElementList(['ru', 'pd', 'cd'])
        two = ElementList(['pd', 'ag', 'cd'])

        oneandtwo = one & two
        oneortwo = one | two
        onexortwo = one ^ two
        assert type(oneandtwo) == ElementList
        assert type(oneortwo) == ElementList
        assert type(onexortwo) == ElementList

        #The order should be consistent
        assert list(oneandtwo) == ['Pd', 'Cd']
        assert list(oneortwo) == ['Ru', 'Pd', 'Cd', 'Ag']
        assert list(onexortwo) == ['Ru', 'Ag']


#TODO dtype
class Test_IsopyArray(object):
    def test_creation_dict(self):
        #creating array from dict
        dict_data = {'104Pd': [1,2,3,4,5], '105Pd': [10, 20, 30, 40, 50], '106Pd': [100, 200, 300, 400, 500]}

        array = IsotopeArray(dict_data)
        assert isinstance(array, IsotopeArray)
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == dict_data['104Pd'])
        assert np.all(array['105Pd'] == dict_data['105Pd'])
        assert np.all(array['106Pd'] == dict_data['106Pd'])
        assert array.ndim == 1
        assert array.size == 5
        assert array.ncols == len(array.keys())
        assert array.nrows == 5

        array = IsotopeArray(dict_data, ndim = -1)
        assert isinstance(array, IsotopeArray)
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == dict_data['104Pd'])
        assert np.all(array['105Pd'] == dict_data['105Pd'])
        assert np.all(array['106Pd'] == dict_data['106Pd'])
        assert array.ndim == 1
        assert array.size == 5
        assert array.ncols == len(array.keys())
        assert array.nrows == 5

        with pytest.raises(ValueError):
            array = IsotopeArray(dict_data, ndim = 0)

        with pytest.raises(ValueError):
            array = RatioArray(dict_data)

        dict_data = {'104Pd': 1, '105Pd': 10, '106Pd': 100}

        array = IsotopeArray(dict_data)
        assert isinstance(array, IsotopeArray)
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == dict_data['104Pd'])
        assert np.all(array['105Pd'] == dict_data['105Pd'])
        assert np.all(array['106Pd'] == dict_data['106Pd'])
        assert array.ndim == 0
        assert array.size == 1
        assert array.ncols == len(array.keys())
        assert array.nrows == -1

        array = IsotopeArray(dict_data, ndim=0)
        assert isinstance(array, IsotopeArray)
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == dict_data['104Pd'])
        assert np.all(array['105Pd'] == dict_data['105Pd'])
        assert np.all(array['106Pd'] == dict_data['106Pd'])
        assert array.ndim == 0
        assert array.size == 1
        assert array.ncols == len(array.keys())
        assert array.nrows == -1

        array = IsotopeArray(dict_data, ndim=-1)
        assert isinstance(array, IsotopeArray)
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == dict_data['104Pd'])
        assert np.all(array['105Pd'] == dict_data['105Pd'])
        assert np.all(array['106Pd'] == dict_data['106Pd'])
        assert array.ndim == 0
        assert array.size == 1
        assert array.ncols == len(array.keys())
        assert array.nrows == -1

        array = IsotopeArray(dict_data, ndim = 1)
        assert isinstance(array, IsotopeArray)
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == dict_data['104Pd'])
        assert np.all(array['105Pd'] == dict_data['105Pd'])
        assert np.all(array['106Pd'] == dict_data['106Pd'])
        assert array.ndim == 1
        assert array.size == 1
        assert array.ncols == len(array.keys())
        assert array.nrows == 1

        dict_data = {'104Pd': [1, 2, 3], '105Pd': [10, 20, 30, 40, 50], '106Pd': [100, 200, 300, 400, 500]}

        with pytest.raises(ValueError):
            array = IsotopeArray(dict_data)

    def test_creation_list(self):
        col_data = [[1,2,3,4,5], [10, 20, 30, 40, 50], [100, 200, 300, 400, 500]]
        list_data = [[1, 10, 100], [2, 20, 200], [3, 30, 300], [4, 40, 400], [5, 50, 500]]
        keys = ['104Pd', '105Pd', '106Pd']

        array = IsotopeArray(list_data, keys = keys)
        assert isinstance(array, IsotopeArray)
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == col_data[0])
        assert np.all(array['105Pd'] == col_data[1])
        assert np.all(array['106Pd'] == col_data[2])
        assert array.ndim == 1
        assert array.size == 5

        array = IsotopeArray(list_data, keys=keys, ndim = 1)
        assert isinstance(array, IsotopeArray)
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == col_data[0])
        assert np.all(array['105Pd'] == col_data[1])
        assert np.all(array['106Pd'] == col_data[2])
        assert array.ndim == 1
        assert array.size == 5

        with pytest.raises(ValueError):
            array = IsotopeArray(list_data, keys=keys, ndim=0)

        with pytest.raises(ValueError):
            array = RatioArray(list_data, keys=keys)

        col_data = [1, 10, 100]
        list_data = [[1, 10, 100]]

        array = IsotopeArray(list_data, keys=keys)
        assert isinstance(array, IsotopeArray)
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == col_data[0])
        assert np.all(array['105Pd'] == col_data[1])
        assert np.all(array['106Pd'] == col_data[2])
        assert array.ndim == 1
        assert array.size == 1

        array = IsotopeArray(list_data, keys=keys, ndim=1)
        assert isinstance(array, IsotopeArray)
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == col_data[0])
        assert np.all(array['105Pd'] == col_data[1])
        assert np.all(array['106Pd'] == col_data[2])
        assert array.ndim == 1
        assert array.size == 1

        array = IsotopeArray(list_data, keys=keys, ndim=0)
        assert isinstance(array, IsotopeArray)
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == col_data[0])
        assert np.all(array['105Pd'] == col_data[1])
        assert np.all(array['106Pd'] == col_data[2])
        assert array.ndim == 0
        assert array.size == 1

        array = IsotopeArray(list_data, keys=keys, ndim=-1)
        assert isinstance(array, IsotopeArray)
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == col_data[0])
        assert np.all(array['105Pd'] == col_data[1])
        assert np.all(array['106Pd'] == col_data[2])
        assert array.ndim == 0
        assert array.size == 1

        list_data = [1, 10, 100]

        array = IsotopeArray(list_data, keys=keys)
        assert isinstance(array, IsotopeArray)
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == list_data[0])
        assert np.all(array['105Pd'] == list_data[1])
        assert np.all(array['106Pd'] == list_data[2])
        assert array.ndim == 0
        assert array.size == 1

        array = IsotopeArray(list_data, keys=keys, ndim=1)
        assert isinstance(array, IsotopeArray)
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == list_data[0])
        assert np.all(array['105Pd'] == list_data[1])
        assert np.all(array['106Pd'] == list_data[2])
        assert array.ndim == 1
        assert array.size == 1

        array = IsotopeArray(list_data, keys=keys, ndim=0)
        assert isinstance(array, IsotopeArray)
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == list_data[0])
        assert np.all(array['105Pd'] == list_data[1])
        assert np.all(array['106Pd'] == list_data[2])
        assert array.ndim == 0
        assert array.size == 1

        array = IsotopeArray(list_data, keys=keys, ndim=-1)
        assert isinstance(array, IsotopeArray)
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == list_data[0])
        assert np.all(array['105Pd'] == list_data[1])
        assert np.all(array['106Pd'] == list_data[2])
        assert array.ndim == 0
        assert array.size == 1

        list_data = [[1, 10, 100, 1000], [2, 20, 200], [3, 30, 300], [4, 40, 400], [5, 50, 500]]
        keys = ['104Pd', '105Pd', '106Pd']
        with pytest.raises(ValueError):
            array = RatioArray(list_data, keys=keys)

        list_data = [[1, 100], [2, 20, 200], [3, 30, 300], [4, 40, 400], [5, 50, 500]]
        keys = ['104Pd', '105Pd']
        with pytest.raises(ValueError):
            array = RatioArray(list_data, keys=keys)

    def test_creation_numpy(self):
        array_data = np.array([(1,10,100), (2,20,200), (3,30,300), (4,40,400), (5,50,500)],
                        dtype = [('104Pd', 'f8'), ('105Pd', 'f8'), ('106Pd', 'f8')])

        array = IsotopeArray(array_data)
        assert isinstance(array, IsotopeArray)
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == array_data['104Pd'])
        assert np.all(array['105Pd'] == array_data['105Pd'])
        assert np.all(array['106Pd'] == array_data['106Pd'])
        assert array.ndim == 1
        assert array.size == 5

        array = IsotopeArray(array_data, ndim=1)
        assert isinstance(array, IsotopeArray)
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

        #No longer supported
        #array = IsotopeArray(array_data, keys = {'104Pd': '104Ru', '106Pd': '106Cd'})
        #assert isinstance(array, IsotopeArray)
        #assert array.keys() == ['104Ru', '105Pd', '106Cd']
        #assert np.all(array['104Ru'] == array_data['104Pd'])
        #assert np.all(array['105Pd'] == array_data['105Pd'])
        #assert np.all(array['106Cd'] == array_data['106Pd'])
        #assert array.ndim == 1
        #assert array.size == 5

        array_data = np.array([(1, 10, 100)],
                              dtype=[('104Pd', 'f8'), ('105Pd', 'f8'), ('106Pd', 'f8')])

        array = IsotopeArray(array_data)
        assert isinstance(array, IsotopeArray)
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == array_data['104Pd'])
        assert np.all(array['105Pd'] == array_data['105Pd'])
        assert np.all(array['106Pd'] == array_data['106Pd'])
        assert array.ndim == 1
        assert array.size == 1

        array = IsotopeArray(array_data, ndim = 1)
        assert isinstance(array, IsotopeArray)
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == array_data['104Pd'])
        assert np.all(array['105Pd'] == array_data['105Pd'])
        assert np.all(array['106Pd'] == array_data['106Pd'])
        assert array.ndim == 1
        assert array.size == 1

        array = IsotopeArray(array_data, ndim = 0)
        assert isinstance(array, IsotopeArray)
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == array_data['104Pd'])
        assert np.all(array['105Pd'] == array_data['105Pd'])
        assert np.all(array['106Pd'] == array_data['106Pd'])
        assert array.ndim == 0
        assert array.size == 1

        array = IsotopeArray(array_data, ndim=-1)
        assert isinstance(array, IsotopeArray)
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == array_data['104Pd'])
        assert np.all(array['105Pd'] == array_data['105Pd'])
        assert np.all(array['106Pd'] == array_data['106Pd'])
        assert array.ndim == 0
        assert array.size == 1

        array_data = np.array((1, 10, 100),
                              dtype=[('104Pd', 'f8'), ('105Pd', 'f8'), ('106Pd', 'f8')])

        array = IsotopeArray(array_data)
        assert isinstance(array, IsotopeArray)
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == array_data['104Pd'])
        assert np.all(array['105Pd'] == array_data['105Pd'])
        assert np.all(array['106Pd'] == array_data['106Pd'])
        assert array.ndim == 0
        assert array.size == 1

        array = IsotopeArray(array_data, ndim=1)
        assert isinstance(array, IsotopeArray)
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == array_data['104Pd'])
        assert np.all(array['105Pd'] == array_data['105Pd'])
        assert np.all(array['106Pd'] == array_data['106Pd'])
        assert array.ndim == 1
        assert array.size == 1

        array = IsotopeArray(array_data, ndim=0)
        assert isinstance(array, IsotopeArray)
        assert array.keys() == ['104Pd', '105Pd', '106Pd']
        assert np.all(array['104Pd'] == array_data['104Pd'])
        assert np.all(array['105Pd'] == array_data['105Pd'])
        assert np.all(array['106Pd'] == array_data['106Pd'])
        assert array.ndim == 0
        assert array.size == 1

        array = IsotopeArray(array_data, ndim=-1)
        assert isinstance(array, IsotopeArray)
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
        assert array.keys() == list(dict_data.keys())
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
        assert array.keys() == list(dict_data.keys())
        assert array.keys() == array2.keys()
        assert array.ndim == 1
        assert array2.ndim == 0
        assert array is not array2
        array['104Pd'][0] = 1000
        assert array2['104Pd'] == 1

        #array2 = IsotopeArray(array, keys = {'104Pd': '104Ru', '106Pd': '106Cd'})
        #assert array is not array2
        #assert array.keys() == dict_data.keys()
        #assert array2.keys() == ['104Ru', '105Pd', '106Cd']
        #assert array.keys() != array2.keys()
        #assert np.all(array['104Pd'] == array2['104Ru'])
        #assert np.all(array['105Pd'] == array2['105Pd'])
        #assert np.all(array['106Pd'] == array2['106Cd'])

    def test_creation_empty(self):
        array = IsotopeArray(5, keys = ['104Pd', '105Pd', '106Pd'])
        assert array.size == 5
        assert array.ndim == 1
        assert np.all(array['104Pd'] == 0)
        assert np.all(array['105Pd'] == 0)
        assert np.all(array['106Pd'] == 0)

        array = IsotopeArray(1, keys=['104Pd', '105Pd', '106Pd'])
        assert array.size == 1
        assert array.ndim == 1
        assert np.all(array['104Pd'] == 0)
        assert np.all(array['105Pd'] == 0)
        assert np.all(array['106Pd'] == 0)

        array = IsotopeArray(1, keys=['104Pd', '105Pd', '106Pd'], ndim = 0)
        assert array.size == 1
        assert array.ndim == 0
        assert np.all(array['104Pd'] == 0)
        assert np.all(array['105Pd'] == 0)
        assert np.all(array['106Pd'] == 0)

        array = IsotopeArray(1, keys=['104Pd', '105Pd', '106Pd'], ndim=-1)
        assert array.size == 1
        assert array.ndim == 0
        assert np.all(array['104Pd'] == 0)
        assert np.all(array['105Pd'] == 0)
        assert np.all(array['106Pd'] == 0)
        
    def test_1input_ufunc(self):
        dict_data = {'104Pd': [1.0, 2.0, 3.0, 4.0, 5.0], '105Pd': [10.0, 20.0, 30.0, 40.0, 50.0],
                     '106Pd': [100.0, 200.0, 300.0, 400.0, np.nan]}
        isopyarray = IsotopeArray(dict_data)

        for ufunc in [np.sin, np.cos, np.tan, np.arcsin, np.arccos, np.arctan, np.degrees, np.isnan,
                    np.radians, np.deg2rad, np.rad2deg, np.sinh, np.cosh, np.tanh, np.arcsinh, np.arccosh, np.arctanh,
                    np.rint, np.floor, np.ceil, np.trunc, np.exp, np.expm1, np.exp2, np.log, np.log10, np.log2,
                    np.log1p, np.reciprocal, np.positive, np.negative, np.sqrt, np.cbrt, np.square, np.fabs, np.sign,
                    np.absolute, np.abs]:

            isopyarray_result = ufunc(isopyarray)
            ndarray_result = [ufunc(dict_data[key]) for key in dict_data.keys()]
            assert_isopyarray_equal_ndarray(isopyarray_result, ndarray_result, ufunc=ufunc)

    def test_2input_ufunc1(self):
        #test
        dict_data = {'104Pd': [1.0, 2.0, 3.0, 4.0, 5.0], '105Pd': [10.0, 20.0, 30.0, 40.0, 50.0],
                     '106Pd': [100.0, 200.0, 300.0, 400.0, np.nan]}
        isopyarray = IsotopeArray(dict_data)
        
        ndarray = np.array([2.0,4.0,6.0,8.0,10.0])

        for ufunc in [np.add, np.multiply, np.divide, np.power, np.subtract,
                    np.true_divide, np.floor_divide, np.float_power, np.fmod, np.mod, np.remainder]:

            isopyarray_result = ufunc(isopyarray, ndarray)
            ndarray_result = [ufunc(dict_data[key], ndarray) for key in dict_data.keys()]
            assert_isopyarray_equal_ndarray(isopyarray_result, ndarray_result, ufunc=ufunc)

            isopyarray_result = ufunc(ndarray, isopyarray)
            ndarray_result = [ufunc(ndarray, dict_data[key]) for key in dict_data.keys()]
            assert_isopyarray_equal_ndarray(isopyarray_result, ndarray_result, ufunc=ufunc)

    def test_2input_ufunc2(self):
        dict_data1 = {'104Pd': [1.0, 2.0, 3.0, 4.0, 5.0], '105Pd': [10.0, 20.0, 30.0, 40.0, np.nan], '106Pd': [0.1, 0.2, 0.3, 0.4, 0.5]}
        isopyarray1 = IsotopeArray(dict_data1)

        dict_data2 = {'106Pd': [2, 4, 6, 8, 10], '105Pd': [1.0, 2.0, 3.0, 4.0, 5.0], '104Pd': [1.5, 2.5, 3.5, 4.5, 5.5],
                     '102Pd': [0.1, 0.2, 0.3, 0.4, 0.5]}
        isopyarray2 = IsotopeArray(dict_data2)

        nanarray = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
        zeroarray = np.zeros(5)

        for ufunc in [np.add, np.multiply, np.divide, np.power, np.subtract,
                      np.true_divide, np.floor_divide, np.float_power, np.fmod, np.mod, np.remainder]:

            key_list = IsotopeList(dict_data1.keys())
            key_list.extend(IsotopeList(dict_data2.keys()), skip_duplicates=True)
            assert len(key_list) == 4
            isopyarray_result = ufunc(isopyarray1, isopyarray2)
            ndarray_result = [ufunc(dict_data1.get(key, nanarray), dict_data2.get(key, nanarray)) for key in key_list]
            assert_isopyarray_equal_ndarray(isopyarray_result, ndarray_result, ufunc=ufunc, key_list=key_list)

            key_list = IsotopeList(dict_data2.keys())
            key_list.extend(IsotopeList(dict_data1.keys()), skip_duplicates=True)
            assert len(key_list) == 4
            isopyarray_result = ufunc(isopyarray2, isopyarray1)
            ndarray_result = [ufunc(dict_data2.get(key, nanarray), dict_data1.get(key, nanarray)) for key in key_list]
            assert_isopyarray_equal_ndarray(isopyarray_result, ndarray_result, ufunc=ufunc, key_list=key_list)

            key_list = IsotopeList(dict_data1.keys())
            key_list.extend(IsotopeList(dict_data2.keys()), skip_duplicates=True)
            assert len(key_list) == 4
            isopyarray_result = ufunc(isopyarray1, isopyarray2, default_value=0)
            ndarray_result = [ufunc(dict_data1.get(key, zeroarray), dict_data2.get(key, zeroarray)) for key in key_list]
            assert_isopyarray_equal_ndarray(isopyarray_result, ndarray_result, ufunc=ufunc, key_list=key_list)
            
    def test_1input_function1(self):
        dict_data = {'104Pd': [1.0, 2.0, 3.0, 4.0, 5.0], '105Pd': [10.0, 20.0, 30.0, 40.0, 50.0],
                     '106Pd': [100.0, 200.0, 300.0, 400.0, np.nan]}
        isopyarray = IsotopeArray(dict_data)

        for axis in [None, 1, 0]:
            for func in [np.prod, np.sum, np.nanprod, np.nansum, np.cumprod, np.cumsum, np.nancumprod, np.nancumsum,
                        np.amin, np.amax, np.nanmin, np.nanmax, np.ptp, np.median, np.average, np.mean, np.std,
                        np.var, np.nanmedian, np.nanmean, np.nanstd, np.nanvar, ipf.mad, ipf.nanmad, ipf.se, ipf.nanse,
                        ipf.sd, ipf.nansd]:
                isopyarray_result = func(isopyarray, axis=axis)
                ndarray_result = func(np.transpose([dict_data[key]for key in dict_data.keys()]), axis=axis)
                if axis == 0:
                    assert_isopyarray_equal_ndarray(isopyarray_result, ndarray_result.transpose(),
                                                    func=func, axis=axis)
                else:
                    assert_ndarray_equal_ndarray(isopyarray_result, ndarray_result,
                                                 func=func, axis=axis)

            for func in [np.percentile, np.nanpercentile, np.quantile,np.nanquantile]:
                isopyarray_result = func(isopyarray, q=0.5, axis=axis)
                ndarray_result = func(np.transpose([dict_data[key] for key in dict_data.keys()]), q=0.5, axis=axis)
                if axis == 0:
                    assert_isopyarray_equal_ndarray(isopyarray_result, ndarray_result.transpose(), func=func)
                else:
                    assert_ndarray_equal_ndarray(isopyarray_result, ndarray_result, func=func)

        for func in [np.around, np.round]:
            isopyarray_result = func(isopyarray, 3)
            ndarray_result = [func(dict_data[key], 3) for key in dict_data.keys()]
            assert_isopyarray_equal_ndarray(isopyarray_result, ndarray_result, func=func)
            isopyarray_result = func(isopyarray, decimals = 3)
            ndarray_result = func(np.transpose([dict_data[key] for key in dict_data.keys()]), decimals=3)
            assert_isopyarray_equal_ndarray(isopyarray_result, ndarray_result.transpose(), func=func)

        for func in [np.fix, np.nan_to_num]:
            isopyarray_result = func(isopyarray)
            ndarray_result = func(np.transpose([dict_data[key] for key in dict_data.keys()]))
            assert_isopyarray_equal_ndarray(isopyarray_result, ndarray_result.transpose(), func=func)

        dict_data2 = {'104Pd': [3.0, 2.3], '105Pd': [30.0, 20.3],
                     '106Pd': [300.0, 200.3]}
        isopyarray2 = IsotopeArray(dict_data2)
        isopyarray_result = np.append(isopyarray, isopyarray2)
        ndarray_result = [dict_data[key] + dict_data2[key] for key in dict_data.keys()]
        assert_isopyarray_equal_ndarray(isopyarray_result, ndarray_result, func=np.append)

    def test_copy(self):
        dict_data = {'104Pd': [1.0, 2.0, 3.0, 4.0, 5.0], '105Pd': [10.0, 20.0, 30.0, 40.0, 50.0],
                     '106Pd': [100.0, 200.0, 300.0, 400.0, np.nan]}
        isopyarray = IsotopeArray(dict_data)

        isopyarray2 = isopyarray.copy()
        assert isopyarray2 is not isopyarray
        assert isopyarray2.keys() == isopyarray.keys()
        isopyarray2['104pd'][0] = 666
        assert isopyarray2['104pd'][0] == 666
        assert isopyarray['104pd'][0] == 1.0

        isopyarray2 = isopyarray.copy(mass_number_ge=105)
        assert isopyarray2 is not isopyarray
        assert isopyarray2.keys() != isopyarray.keys()
        assert isopyarray2.keys() == ['105pd', '106pd']
        assert isopyarray2['105pd'][0] == 10.0
        assert isopyarray2['106pd'][0] == 100.0
        isopyarray2['105pd'][0] = 666
        assert isopyarray2['105pd'][0] == 666
        assert isopyarray['105pd'][0] == 10.0



