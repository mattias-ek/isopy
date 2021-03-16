import isopy
from isopy import core
import numpy as np
import pytest
import itertools
import pyperclip
import warnings


def assert_array_equal_array(array1, array2, match_dtype=True):
    assert isinstance(array1, core.IsopyArray)
    assert core.flavour(array1) is core.flavour(array1.keys())

    assert array1.dtype.names is not None
    assert array2.dtype.names is not None

    assert len(array1.dtype.names) == len(array2.dtype.names)
    for i in range(len(array1.dtype.names)):
        assert str(array1.dtype.names[i]) == str(array2.dtype.names[i])
    assert array1.ndim == array2.ndim
    assert array1.size == array2.size
    assert array1.shape == array2.shape
    if match_dtype:
        for i in range(len(array1.dtype)):
            assert array1.dtype[i] == array2.dtype[i]
    for name in array1.dtype.names:
        np.testing.assert_allclose(array1[name], array2[name])


class Test_IsopyKeyString:
    def test_direct_creation(self):
        # MassKeyString
        self.direct_creation(core.MassKeyString, correct='105',
                             same=[105, '105', 'MAS_105'],
                             fails=['pd105', 'onehundred and five', -105, '-105', 'GEN_105',
                                   'MAS__105', '_105'],
                             different=['96', 96, '5', 5, '300', 300, '1000', 1000])

        #ElementKeyString
        self.direct_creation(core.ElementKeyString, correct ='Pd',
                             same= ['pd', 'pD', 'PD', 'ELE_Pd', 'ELE_pd'],
                             fails= ['', 1, '1', 'P1', '1P', '1Pd', '/Pd', '_Pd', 'Palladium',
                                     'Pdd', 'GEN_Pd', 'ELE_p1', 'ELE__Pd'],
                             different= ['p', 'Cd', 'ru'])

        #IsotopeKeyString
        self.direct_creation(core.IsotopeKeyString, correct ='105Pd',
                             same= ['105PD', '105pd', '105pD', 'Pd105', 'Pd105', 'pd105',
                                     'pD105', 'ISO_105Pd', 'ISO_pd105'],
                             fails= ['', 'Pd', '105', 'Pd105a', 'P105D', '105Pd/108Pd', 'ISO__pd104',
                                     'GEN_105Pd'],
                             different=['104pd', 'Pd106', '104Ru', 'cd106', '1a', 'b2'])

        self.direct_creation(core.RatioKeyString, correct ='105Pd/Pd',
                             same = ['105PD/Pd', '105pd/Pd', '105pD/Pd', 'Pd105/Pd',
                            'PD105/Pd', 'pd105/Pd', '105Pd/pd', '105Pd/PD',
                            'pd105/pD', ('105Pd', 'Pd'),'105Pd//Pd',],
                             fails=['', 'Pd', '105Pd', '/Pd', 'Pd/', '105Pd_SLASH_Pd',
                                    'RAT1_105Pd/Pd', 'Pd/Pd/Pd'],
                             different = ['RAT2_RAT1_ELE_Pd_OVER1_Cd_OVER2_ELE_Ru'
                                            ,'Pd/Pd', 'Pd/105Pd', '105Pd/108Pd',
                                    '105Pd/Pd//Cd', 'a/b///c/d//e', ('pd105', 'Cd/Ru')])

        #GeneralKeyString
        self.direct_creation(core.GeneralKeyString, correct='Pd/pd',
                             same=['GEN_Pd/pd', 'GEN_Pd_SLASH_pd', 'Pd_SLASH_pd'],
                             fails = [''],
                             different=['Pd/Pd', 'GEN_Pd/Pd', 'Pd', '108Pd', 'Pd//Pd', 'Pd/Pd/Pd'])

        #Check behaviour of general string is for keystring and askeystring.
        for key in ['105', 'Pd', '108Pd']:
            gkey = isopy.GeneralKeyString(key)
            ikey = isopy.keystring(gkey)
            assert type(gkey) != type(ikey)
            assert gkey != ikey
            assert gkey == key
            assert ikey == key

            gkey2 = isopy.askeystring(gkey)
            assert type(gkey) == type(gkey2)
            assert gkey == gkey2

    def direct_creation(self, keytype, correct, same, fails = [], different = []):
        #Make sure the correcttly formatted string can be created and is not reformatted
        correct_key = keytype(correct)

        assert correct_key == correct
        assert correct_key == keytype(correct)

        assert isinstance(correct_key, str)
        assert isinstance(correct_key, keytype)
        assert type(correct_key) is keytype


        for string in same:
            key = keytype(string)
            assert type(key) == keytype
            assert isinstance(key, str)
            assert key == string
            assert key == correct
            assert key == correct_key

            assert hash(key) == hash(correct_key)


        for string in fails:
            with pytest.raises(core.KeyParseError):
                key = keytype(string)
                raise ValueError(f'{string!r} should have raised an'
                                     f'KeyParseError but did not. Returned value was '
                                     f'"{key}" of type {type(key)}')

        for string in different:
            key = keytype(string)
            assert type(key) == keytype
            assert isinstance(key, str)
            assert key == string
            assert key != correct
            assert key != correct_key

    def test_general_creation(self):
        self.general_creation(core.MassKeyString, [105, '105', 'MAS_105'])
        self.general_creation(core.ElementKeyString, ['Pd', 'pd', 'pD', 'PD'])
        self.general_creation(core.IsotopeKeyString, ['105Pd', '105PD', '105pd', '105pD', 'Pd105',
                                                  'Pd105', 'pd105', 'pD105'])

        self.general_creation(core.RatioKeyString, ['105PD/Pd', '105pd/Pd', '105pD/Pd', 'Pd105/Pd',
                            'PD105/Pd', 'pd105/Pd', '105Pd/pd', '105Pd/PD',
                            'pd105/pD', 'Pd//Pd', 'Pd/Cd//Ru', 'Ru///Pd//Cd/Ag', 'Cd/Ru//Ru/Cd'])


        self.general_creation(core.GeneralKeyString, ['test', '-1', '/Pd', '105Pdd', 'Pd/Pd/Pd'])

    def general_creation(self, keytype, correct = []):
        #Test creation with isopy.keystring() and isopy.askeystring
        for string in correct:
            key1 = isopy.keystring(string)
            assert type(key1) == keytype
            assert isinstance(key1, str)
            assert key1 == string

            key2 = isopy.askeystring(key1)

    def test_isotope_attributes(self):
        isotope = isopy.IsotopeKeyString('105Pd')
        assert hasattr(isotope, 'mass_number')
        assert hasattr(isotope, 'element_symbol')

        assert type(isotope.mass_number) is isopy.MassKeyString
        assert type(isotope.element_symbol) is isopy.ElementKeyString

        assert isotope.mass_number == '105'
        assert isotope.element_symbol == 'Pd'

        assert 'Pd' in isotope
        assert 'pd' in isotope
        assert 'ru' not in isotope

        assert 105 in isotope
        assert '105' in isotope
        assert 106 not in isotope
        assert '106' not in isotope

    def test_ratio_attributes(self):
        keys = ['105', 'Pd', '105Pd', 'test', '104Ru/106Cd']
        for numerator, denominator in itertools.permutations(keys, 2):
            ratio = isopy.RatioKeyString((numerator, denominator))
            assert hasattr(ratio, 'numerator')
            assert hasattr(ratio, 'denominator')

            assert type(ratio.numerator) is type(isopy.keystring(numerator))
            assert type(ratio.denominator) is type(isopy.keystring(denominator))

            assert ratio.numerator == numerator
            assert ratio.denominator == denominator

            assert numerator in ratio
            assert denominator in ratio

            assert 104 not in ratio
            assert 'cd' not in ratio
            assert '108Pd' not in ratio
            assert 'notin' not in ratio
            assert 'Pd/Cd' not in ratio

    def test_string_manipulation(self):
        ratio = isopy.GeneralKeyString('108Pd/Pd105')

        assert type(ratio.upper()) is str
        assert type(ratio.lower()) is str
        assert type(ratio.capitalize()) is str
        assert type(ratio[:1]) is str

        assert type(ratio[0]) is str
        assert type(ratio[1:]) is str

        assert type(ratio.split('/')[0]) == str

    def test_integer_manipulation(self):
        integer = 105
        mass = isopy.MassKeyString(integer)

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

        assert mass > '100'
        assert not mass > '105'

        assert mass < '110'
        assert not mass < '105'

        assert mass >= '100'
        assert mass >= '105'
        assert not mass >='110'

        assert mass <= '110'
        assert mass <= '105'
        assert not mass <= '100'

    def test_divide_to_ratio(self):
        for other in [isopy.IsotopeKeyString('74Ge'), isopy.RatioKeyString('88Sr/87Sr')]:
            for string in [isopy.MassKeyString('105'), isopy.ElementKeyString('Pd'),
                      isopy.IsotopeKeyString('105Pd'), isopy.GeneralKeyString('test'),
                    isopy.RatioKeyString('Cd/Ru')]:
                ratio = string / other

                assert type(ratio) is isopy.RatioKeyString
                assert ratio.denominator == other
                assert type(ratio.numerator) is type(string)
                assert ratio.numerator == string

                ratio = other / string

                assert type(ratio) is isopy.RatioKeyString
                assert type(ratio.denominator) is type(string)
                assert ratio.denominator == string
                assert ratio.numerator == other


class Test_IsopyList:
    def test_creation(self):
        mass = self.creation(isopy.MassKeyList,
                      correct = ['104', '105', '106'],
                      same = [('106', 105, 104)],
                      fail = ['105', '106', 'Pd'])

        element = self.creation(isopy.ElementKeyList,
                      correct=['Pd', 'Ag', 'Cd'],
                      same=[('pd', 'cD', 'AG')],
                      fail=['Cd', 'Pdd', 'Ag'])

        isotope = self.creation(isopy.IsotopeKeyList,
                      correct=['104Ru', '105Pd', '106Cd'],
                      same=[('pd105', '106CD', '104rU')],
                      fail=['p105d', '104ru', '106cd'])

        assert type(isotope.mass_numbers) == isopy.MassKeyList
        for masskey in ('104', '105', '106'):
            assert masskey in isotope.mass_numbers

        assert type(isotope.element_symbols) == isopy.ElementKeyList
        for elementkey in ('pd', 'ru', 'cd'):
            assert elementkey in isotope.element_symbols

        general = self.creation(isopy.GeneralKeyList,
                      correct=['ron', 'harry', 'hermione'],
                      same=[('harry', 'ron', 'hermione')],
                      fail=None)

        for numerator, denominator in itertools.permutations((mass, element, isotope, general), 2):
            correct = [f'{n}/{d}' for n, d in zip(numerator, denominator)]
            same = [(correct[0], correct[2], correct[1])]
            fail = [correct[0], correct[2], f'{numerator[0]}']
            ratio = self.creation(isopy.RatioKeyList,
                          correct=correct,
                          same=same,
                          fail=fail)

            assert type(ratio.numerators) is type(numerator)
            assert ratio.numerators == numerator
            assert type(ratio.denominators) is type(denominator)
            assert ratio.denominators == denominator

            assert numerator / denominator == correct

            keys = [f'{n}/{denominator[0]}' for n in numerator]
            assert numerator / denominator[0] == keys
            assert [f'{n}' for n in numerator] / denominator[0] == keys
            assert numerator / f'{denominator[0]}' == keys
            assert isopy.RatioKeyList(keys).common_denominator == denominator[0]

            keys = [f'{numerator[0]}/{d}' for d in denominator]
            assert numerator[0] / denominator == keys
            assert f'{numerator[0]}' / denominator == keys
            assert numerator[0] / [f'{d}' for d in denominator] == keys
            with pytest.raises(core.NoCommomDenominator):
                isopy.RatioKeyList(keys).common_denominator

        for numerator in (mass, element, isotope, general, ratio):
            correct = [(n,d) for n, d in zip(numerator, ratio)]
            same = [(correct[0], correct[2], correct[1])]
            fail = None
            ratio2 = self.creation(isopy.RatioKeyList,
                                  correct=correct,
                                  same=same,
                                  fail=fail)
            assert type(ratio2.numerators) is type(numerator)
            assert ratio2.numerators == numerator
            assert type(ratio2.denominators) is type(ratio)
            assert ratio2.denominators == ratio

    def creation(self, listtype, correct, same, fail = None):
        dtype = np.dtype(dict(names=[isopy.keystring(k) for k in correct],  formats=[float for k in correct]))
        array = np.ones(1, dtype)
        d = {k: None for k in correct}
        same.extend([dtype, array, d])

        correct_list = listtype(correct)
        assert type(correct_list) == listtype
        assert correct_list == correct
        assert len(correct_list) == len(correct)
        for k in correct:
            if isinstance(k, str):
                assert k in correct_list

        if not type(correct[0]) is tuple: #otherwise an error is given when a ratio is given as a tuple
            same_list = listtype(*correct)
            assert same_list == correct
            assert type(same_list) is listtype

        same_list = isopy.keylist(correct)
        assert len(same_list) == len(correct)
        assert same_list == correct
        assert same_list == correct_list
        assert type(same_list) is listtype

        same_list = isopy.askeylist(correct)
        assert len(same_list) == len(correct)
        assert same_list == correct
        assert type(same_list) is listtype

        same_list = isopy.keylist(correct_list)
        assert same_list == correct
        assert same_list is not correct_list

        same_list = isopy.askeylist(correct_list)
        assert same_list == correct_list
        assert same_list is correct_list

        if False:
            for s in same:
                same_list = listtype(s)
                assert type(same_list) == listtype
                assert same_list == correct
                assert same_list == correct_list

                same_list = isopy.keylist(s)
                assert type(same_list) is listtype
                assert same_list == correct

                same_list = isopy.askeylist(s)
                assert type(same_list) is listtype
                assert same_list == correct

        if fail:
            with pytest.raises(core.KeyParseError):
                listtype(fail)

            general_list = isopy.keylist(fail)
            assert type(general_list) is not listtype
            assert type(general_list) is isopy.GeneralKeyList
            assert general_list != correct

            general_list = isopy.askeylist(fail)
            assert type(general_list) is not listtype
            assert type(general_list) is isopy.GeneralKeyList
            assert general_list != correct

        return correct_list

    def test_compare(self):
        mass = self.compare(isopy.MassKeyList,
                             keys=['104', '105', '106'],
                             extra_keys=['99', '108', '111'],
                             notin=['70', '76', '80'])

        element = self.compare(isopy.ElementKeyList,
                                keys=['Ru', 'Pd', 'Cd'],
                                extra_keys=['Mo', 'Ag', 'Te'],
                                notin=['Ni', 'Ge', 'Se'])

        isotope = self.compare(isopy.IsotopeKeyList,
                                keys=['104Ru', '105Pd', '106Cd'],
                                extra_keys=['99Ru', '106Pd', '111Cd'],
                                notin=['70Ge', '76Ge', '80Se'])

        general = self.compare(isopy.GeneralKeyList,
                                keys=['ron', 'harry', 'hermione'],
                                extra_keys=['ginny', 'neville', 'luna'],
                                notin=['malfoy', 'crabbe', 'goyle'])

        for numerator, denominator in itertools.permutations((mass, element, isotope, general), 2):
            keys = [f'{n}/{d}' for n, d in zip(numerator, denominator)]
            ratio = self.compare(isopy.RatioKeyList,
                                  keys=keys)


        for numerator in (mass, element, isotope, general, ratio):
            keys = [(n, d) for n, d in zip(numerator, ratio)]
            ratio2 = self.compare(isopy.RatioKeyList,
                                   keys=keys)

            assert numerator / ratio == keys

    def compare(self, listtype, keys, extra_keys=[], notin=[]):
        keys2 = keys + keys + extra_keys
        keys3 = keys + extra_keys

        keylist = listtype(keys)
        assert keylist == keys
        assert keylist != keys[:-1]
        assert (keylist == keys[:-1]) is False
        assert len(keylist) == len(keys)
        assert keylist.has_duplicates() is False
        for key in keys:
            if isinstance(key, str):
                assert key in keylist
                assert keylist.count(key) == 1
        for i in range(len(keys)):
            assert keylist[i] == keys[i]
            assert keylist.index(keys[i]) == keys.index(keys[i])

        for k in notin:
            assert k not in keylist

        keylist_ = listtype(keys, skip_duplicates = True)
        assert keylist == keys

        keylist_ = listtype(keys, allow_duplicates=False)
        assert keylist == keys

        keylist2 = listtype(keys2)
        assert keylist2 != keylist
        assert keylist2 == keys2
        assert len(keylist2) == len(keys2)
        assert keylist2.has_duplicates() is True
        for key in keys:
            if isinstance(key, str):
                assert key in keylist2
                assert keylist2.count(key) == 2
        for i in range(len(keys2)):
            assert keylist2[i] == keys2[i]
            assert keylist2.index(keys2[i]) == keys2.index(keys2[i])

        keylist3 = listtype(keys2, skip_duplicates = True)
        assert keylist3 == keys3
        assert len(keylist3) == len(keys3)
        assert keylist3.has_duplicates() is False
        for key in keys3:
            if isinstance(key, str):
                assert key in keylist3
                assert keylist3.count(key) == 1
        for i in range(len(keys3)):
            assert keylist3[i] == keys3[i]
            assert keylist3.index(keys3[i]) == keys3.index(keys3[i])


        with pytest.raises(core.ListDuplicateError):
            listtype(keys2, allow_duplicates=False)

        return keylist

    def test_bitwise(self):
        #Last two of key1 should be the first two of key2
        #Otherwise the ratio test will fail
        mass = self.bitwise(isopy.MassKeyList, (102, 104, 105, 106),
                    ('105', '106', '108', '110'),
                     ('105', '106'),
                     (102, 104, 105, 106, 108, 110),
                     ('102', '104', 108, 110))

        element = self.bitwise(isopy.ElementKeyList, ('Mo', 'Ru', 'Pd', 'Rh'),
                     ('Pd', 'Rh', 'Ag', 'Cd'),
                     ('Pd', 'Rh'),
                     ('Mo', 'Ru', 'Pd', 'Rh', 'Ag', 'Cd'),
                     ('Mo', 'Ru', 'Ag', 'Cd'))

        isotope= self.bitwise(isopy.IsotopeKeyList, ('102Pd', '104Pd', '105Pd', '106Pd'),
                     ('105Pd', '106Pd', '108Pd', '110Pd'),
                     ('105Pd', '106Pd'),
                     ('102Pd', '104Pd', '105Pd', '106Pd', '108Pd', '110Pd'),
                     ('102Pd', '104Pd', '108Pd', '110Pd'))

        general = self.bitwise(isopy.GeneralKeyList, ('Hermione', 'Neville', 'Harry', 'Ron'),
                     ('Harry', 'Ron', 'George', 'Fred'),
                     ('Harry', 'Ron'),
                     ('Hermione', 'Neville', 'Harry', 'Ron', 'George', 'Fred'),
                     ('Hermione', 'Neville', 'George', 'Fred'))

        for numerator, denominator in itertools.permutations((mass, element, isotope, general), 2):
            key1 = (numerator[2][:2] + numerator[0]) / (denominator[2][:2] + denominator[0])
            key2 = (numerator[0] + numerator[2][2:])/ (denominator[0] + denominator[2][2:])

            rand = numerator[0] / denominator[0]
            ror = numerator[1] / denominator[1]
            rxor = numerator[2] / denominator[2]

            general = self.bitwise(isopy.RatioKeyList, key1,
                                   key2,
                                   rand,
                                   ror,
                                   rxor)

    def bitwise(self, listtype, keys1, keys2, band, bor, bxor):
        keylist1 = listtype(*keys1)
        keylist2 = listtype(*keys2, *keys2[-2:])

        keyband = keylist1 & keylist2
        assert type(keyband) is listtype
        assert keyband == band
        assert keylist1 & keys2 == band
        assert keys1 & keylist2 == band

        keyband = keyband.__and__(keys2, keys2)
        assert type(keyband) is listtype
        assert keyband == band

        keybor = keylist1 | keylist2
        assert type(keybor) is listtype
        assert keybor == bor
        assert keylist1 | keys2 == bor
        assert keys1 | keylist2 == bor

        keybor = keylist1.__or__(keys2[:2], keys2[2:])
        assert type(keybor) is listtype
        assert keybor == bor

        keybxor = keylist1 ^ keylist2
        assert type(keybxor) is listtype
        assert keybxor == bxor
        assert keylist1 ^ keys2 == bxor
        assert keys1 ^ keylist2 == bxor

        keybxor = keylist1.__xor__(keys2[:2], keys2[2:])
        assert type(keybxor) is listtype
        assert keybxor == bxor

        return keyband, keybor, keybxor

    def test_filter(self):
        mass = isopy.MassKeyList(['104', '105', '106', '108', '104', '108'])
        self.filter_key(isopy.MassKeyList, 'key', mass, mass, ['103', '107'])
        self.filter_mass(isopy.MassKeyList, 'key', mass, mass)

        element = isopy.ElementKeyList(['Ru', 'Pd', 'Pd', 'Cd', 'Ru', 'Cd'])
        self.filter_key(isopy.ElementKeyList, 'key', element, element, ['Rh', 'Ag'])

        isotope = isopy.IsotopeKeyList(['104Ru', '105Pd', '106Pd', '108Cd', '104Ru', '108Cd'])
        self.filter_key(isopy.IsotopeKeyList, 'key', isotope, isotope, ['103Rh', '107Ag'])
        self.filter_mass(isopy.IsotopeKeyList, 'mass_number', isotope, mass)
        self.filter_key(isopy.IsotopeKeyList, 'element_symbol', isotope, element, ['Rh', 'Ag'])

        assert isotope.filter(mass_number_gt = 105, element_symbol_eq = 'pd') == \
               [key for key in isotope if (key.element_symbol in ['pd'] and key.mass_number > 105)]

        general = isopy.GeneralKeyList(['harry', 'ron', 'hermione', 'neville', 'harry', 'neville'])
        self.filter_key(isopy.GeneralKeyList, 'key', general, general, ['ginny', 'luna'])

        ratio = general / isotope
        self.filter_key(isopy.RatioKeyList, 'key', ratio, ratio, ['ginny/103Rh', 'luna/107Ag'])
        self.filter_key(isopy.RatioKeyList, 'numerator', ratio, general, ['ginny', 'luna'])
        self.filter_key(isopy.RatioKeyList, 'denominator', ratio, isotope, ['103Rh', '107Ag'])
        self.filter_key(isopy.RatioKeyList, 'denominator_element_symbol', ratio, element, ['Rh', 'Ag'])
        self.filter_mass(isopy.RatioKeyList, 'denominator_mass_number', ratio, mass)

        result = ratio.filter(numerator_eq=['ron', 'harry', 'hermione'],
                              denominator_mass_number_ge=105, denominator_element_symbol_eq='pd')
        assert result == ratio[1:3]

        ratio2 = ratio / element
        self.filter_key(isopy.RatioKeyList, 'key', ratio2, ratio2, ['ginny//103Rh/Rh', 'luna//107Ag/ag'])
        self.filter_key(isopy.RatioKeyList, 'numerator', ratio2, ratio, ['ginny/103Rh', 'luna/107Ag'])
        self.filter_key(isopy.RatioKeyList, 'numerator_numerator', ratio2, general, ['ginny', 'luna'])
        self.filter_key(isopy.RatioKeyList, 'numerator_denominator', ratio2, isotope, ['103Rh', '107Ag'])
        self.filter_key(isopy.RatioKeyList, 'numerator_denominator_element_symbol', ratio2, element, ['Rh', 'Ag'])
        self.filter_mass(isopy.RatioKeyList, 'numerator_denominator_mass_number', ratio2, mass)
        self.filter_key(isopy.RatioKeyList, 'denominator', ratio2, element, ['Rh', 'Ag'])

        result = ratio2.filter(numerator_numerator_eq=['ron', 'harry', 'hermione'],
                              numerator_denominator_mass_number_ge=105,
                              numerator_denominator_element_symbol_eq='pd')
        assert result == ratio2[1:3]

        result = ratio2.filter(numerator_numerator_eq=['ron', 'harry', 'hermione'],
                              numerator_denominator_mass_number_ge=105,
                              denominator_eq='pd')
        assert result == ratio2[1:3]

    def filter_mass(self, listtype, prefix, keys, mass_list):
        keylist = listtype(keys)
        mass_list = isopy.MassKeyList(mass_list)

        result = keylist.filter(**{f'{prefix}_gt': mass_list[1]})
        assert result == [keylist[i] for i, key in enumerate(mass_list) if key > mass_list[1]]
        assert len(result) != 0

        result = keylist.filter(**{f'{prefix}_ge': mass_list[1]})
        assert result == [keylist[i] for i, key in enumerate(mass_list) if key >= mass_list[1]]
        assert type(result) is listtype
        assert len(result) != 0

        result = keylist.filter(**{f'{prefix}_lt': mass_list[1]})
        assert result == [keylist[i] for i, key in enumerate(mass_list) if key < mass_list[1]]
        assert type(result) is listtype
        assert len(result) != 0

        result = keylist.filter(**{f'{prefix}_le': mass_list[1]})
        assert result == [keylist[i] for i, key in enumerate(mass_list) if key <= mass_list[1]]
        assert type(result) is listtype
        assert len(result) != 0

        result = keylist.filter(**{f'{prefix}_le': mass_list[2], f'{prefix}_gt': mass_list[0]})
        assert result == [keylist[i] for i, key in enumerate(mass_list) if
                          (key <= mass_list[2] and key > mass_list[0])]
        assert type(result) is listtype
        assert len(result) != 0

        result = keylist.filter(**{f'{prefix}_lt': mass_list[2], f'{prefix}_ge': mass_list[0]})
        assert result == [keylist[i] for i, key in enumerate(mass_list) if
                          (key < mass_list[2] and key >= mass_list[0])]
        assert type(result) is listtype
        assert len(result) != 0

        result = keylist.filter(**{f'{prefix}_eq': mass_list[1:3], f'{prefix}_lt': mass_list[2]})
        assert result == [keylist[i] for i, key in enumerate(mass_list) if
                          (key in mass_list[1:3] and key < mass_list[2])]
        assert type(result) is listtype
        assert len(result) != 0

    def filter_key(self, listtype, prefix, keys, eq, neq):
        keylist = listtype(keys)
        eq = isopy.askeylist(eq)
        neq = isopy.askeylist(neq)

        result = keylist.filter()
        assert type(result) == listtype
        assert result == keylist
        assert result is not keylist

        result = keylist.filter(**{f'{prefix}_eq': eq[0]})
        assert type(result) is listtype
        assert result == [keylist[i] for i, key in enumerate(eq) if key == eq[0]]
        assert len(result) != 0

        result = keylist.filter(**{f'{prefix}_eq': eq[2:]})
        assert type(result) == listtype
        assert result == [keylist[i] for i, key in enumerate(eq) if key in eq[2:]]
        assert len(result) != 0

        result = keylist.filter(**{f'{prefix}_eq': (neq + eq[2:])})
        assert type(result) == listtype
        assert result == [keylist[i] for i, key in enumerate(eq) if key in (neq + eq[2:])]
        assert len(result) != 0

        result = keylist.filter(**{f'{prefix}_eq': neq})
        assert type(result) == listtype
        assert result == [keylist[i] for i, key in enumerate(eq) if key in neq]
        assert len(result) == 0

        result = keylist.filter(**{f'{prefix}_neq': neq})
        assert type(result) == listtype
        assert result == [keylist[i] for i, key in enumerate(eq) if key not in neq]
        assert len(result) != 0

        result = keylist.filter(**{f'{prefix}_neq': eq[-1]})
        assert type(result) == listtype
        assert result == [keylist[i] for i, key in enumerate(eq) if key != eq[-1]]
        assert len(result) != 0

        result = keylist.filter(**{f'{prefix}_neq': (eq[-1:] + neq)})
        assert type(result) == listtype
        assert result == [keylist[i] for i, key in enumerate(eq) if key not in (eq[-1:] + neq)]
        assert len(result) != 0

        result = keylist.filter(**{f'{prefix}_eq': eq[:-1], f'{prefix}_neq': (neq + eq[:2])})
        assert type(result) == listtype
        assert result == [keylist[i] for i, key in enumerate(eq) if
                          (key in eq[:-1] and key not in (neq + eq[:2]))]
        assert len(result) != 0

        return keylist


#TODO test table
#TODO test changing dtype
class Test_Array:
    def test_0dim(self):
        mass_keys = ('104', '105', '106', '107')
        element_keys = ('mo', 'ru', 'pd', 'rh')
        isotope_keys = ('104ru', '105Pd', '106Pd', 'cd111')
        general_keys = ('harry', 'ron', 'hermione', 'neville')
        ratio_keys = ('harry/104ru', 'ron/105pd', 'hermione/106Pd', 'neville/cd111')

        all_keys = (mass_keys, element_keys, isotope_keys, general_keys, ratio_keys)

        # 0-dim input
        data_list = [1, 2, 3, 4]
        data_tuple = (1, 2, 3, 4)
        data_array = np.array(data_list, dtype=np.float_)
        keylist2 = isopy.keylist(isotope_keys)
        data_correct2 = np.array(data_tuple, dtype=[(str(keystring), np.float_) for keystring in keylist2])

        for keys in all_keys:
            keylist = isopy.keylist(keys)

            data_dict = dict(zip(keys, data_list))
            data_structured = np.array(data_tuple, dtype=[(str(key), np.float_) for key in keys])
            data_correct = np.array(data_tuple, dtype=[(str(keystring), np.float_) for keystring in keylist])

            self.create_array(data_correct, keylist, 0, data_list, keys)
            self.create_array(data_correct, keylist, 0, data_tuple, keys)
            self.create_array(data_correct, keylist, 0, data_array, keys)
            self.create_array(data_correct, keylist, 0, data_dict)
            self.create_array(data_correct, keylist, 0, data_structured)

            self.create_array(data_correct, keylist, 0, data_list, keys, ndim=-1)
            self.create_array(data_correct, keylist, 0, data_tuple, keys, ndim=-1)
            self.create_array(data_correct, keylist, 0, data_array, keys, ndim=-1)
            self.create_array(data_correct, keylist, 0, data_dict, ndim=-1)
            self.create_array(data_correct, keylist, 0, data_structured, ndim=-1)

            self.create_array(data_correct, keylist, 0, data_list, keys, ndim=0)
            self.create_array(data_correct, keylist, 0, data_tuple, keys, ndim=0)
            self.create_array(data_correct, keylist, 0, data_array, keys, ndim=0)
            self.create_array(data_correct, keylist, 0, data_dict, ndim=0)
            self.create_array(data_correct, keylist, 0, data_structured, ndim=0)

            #Overwrite keys in data
            self.create_array(data_correct2, keylist2, 0, data_dict, isotope_keys)
            self.create_array(data_correct2, keylist2, 0, data_structured, isotope_keys)
            self.create_array(data_correct2, keylist2, 0, data_dict, isotope_keys, ndim=-1)
            self.create_array(data_correct2, keylist2, 0, data_structured, isotope_keys, ndim=-1)
            self.create_array(data_correct2, keylist2, 0, data_dict, isotope_keys, ndim=0)
            self.create_array(data_correct2, keylist2, 0, data_structured, isotope_keys, ndim=0)

        #1-dim input
        data_list = [[1, 2, 3, 4]]
        data_tuple = [(1, 2, 3, 4)]
        data_array = np.array(data_list, dtype=np.float_)
        for keys in all_keys:
            keylist = isopy.keylist(keys)

            data_dict = dict(zip(keys, [[1], [2], [3], [4]]))
            data_structured = np.array(data_tuple, dtype=[(str(key), np.float_) for key in keys])
            data_correct = np.array(data_tuple[0], dtype=[(str(keystring), np.float_) for keystring in keylist])

            self.create_array(data_correct, keylist, 0, data_list, keys, ndim=-1)
            self.create_array(data_correct, keylist, 0, data_tuple, keys, ndim=-1)
            self.create_array(data_correct, keylist, 0, data_array, keys, ndim=-1)
            self.create_array(data_correct, keylist, 0, data_dict, ndim=-1)
            self.create_array(data_correct, keylist, 0, data_structured, ndim=-1)

            self.create_array(data_correct, keylist, 0, data_list, keys, ndim=0)
            self.create_array(data_correct, keylist, 0, data_tuple, keys, ndim=0)
            self.create_array(data_correct, keylist, 0, data_array, keys, ndim=0)
            self.create_array(data_correct, keylist, 0, data_dict, ndim=0)
            self.create_array(data_correct, keylist, 0, data_structured, ndim=0)

            # Overwrite keys in data
            self.create_array(data_correct2, keylist2, 0, data_dict, isotope_keys, ndim=-1)
            self.create_array(data_correct2, keylist2, 0, data_structured, isotope_keys, ndim=-1)
            self.create_array(data_correct2, keylist2, 0, data_dict, isotope_keys, ndim=0)
            self.create_array(data_correct2, keylist2, 0, data_structured, isotope_keys, ndim=0)

    def test_1dim(self):
        mass_keys = ('104', '105', '106', '107')
        element_keys = ('mo', 'ru', 'pd', 'rh')
        isotope_keys = ('104ru', '105Pd', '106Pd', 'cd111')
        general_keys = ('harry', 'ron', 'hermione', 'neville')
        ratio_keys = ('harry/104ru', 'ron/105pd', 'hermione/106Pd', 'neville/cd111')

        all_keys = (mass_keys, element_keys, isotope_keys, general_keys, ratio_keys)

        # 0-dim input
        data_list = [1, 2, 3, 4]
        data_tuple = (1, 2, 3, 4)
        data_array = np.array(data_list, dtype=np.float_)
        keylist2 = isopy.keylist(isotope_keys)
        data_correct2 = np.array([data_tuple], dtype=[(str(keystring), np.float_) for keystring in keylist2])

        for keys in all_keys:
            keylist = isopy.keylist(keys)

            data_dict = dict(zip(keys, data_list))
            data_structured = np.array(data_tuple, dtype=[(str(key), np.float_) for key in keys])
            data_correct = np.array([data_tuple], dtype=[(str(keystring), np.float_) for keystring in keylist])

            self.create_array(data_correct, keylist, 1, data_list, keys, ndim=1)
            self.create_array(data_correct, keylist, 1, data_tuple, keys, ndim=1)
            self.create_array(data_correct, keylist, 1, data_array, keys, ndim=1)
            self.create_array(data_correct, keylist, 1, data_dict, ndim=1)
            self.create_array(data_correct, keylist, 1, data_structured, ndim=1)

            # Overwrite keys in data
            self.create_array(data_correct2, keylist2, 1, data_dict, isotope_keys, ndim=1)
            self.create_array(data_correct2, keylist2, 1, data_structured, isotope_keys, ndim=1)

        # 1-dim input, n == 1
        data_list = [[1, 2, 3, 4]]
        data_tuple = [(1, 2, 3, 4)]
        data_array = np.array(data_list, dtype=np.float_)
        for keys in all_keys:
            keylist = isopy.keylist(keys)

            data_dict = dict(zip(keys, [[1], [2], [3], [4]]))
            data_structured = np.array(data_tuple, dtype=[(str(key), np.float_) for key in keys])
            data_correct = np.array(data_tuple, dtype=[(str(keystring), np.float_) for keystring in keylist])

            self.create_array(data_correct, keylist, 1, data_list, keys)
            self.create_array(data_correct, keylist, 1, data_tuple, keys)
            self.create_array(data_correct, keylist, 1, data_array, keys)
            self.create_array(data_correct, keylist, 1, data_dict)
            self.create_array(data_correct, keylist, 1, data_structured)

            self.create_array(data_correct, keylist, 1, data_list, keys, ndim=1)
            self.create_array(data_correct, keylist, 1, data_tuple, keys, ndim=1)
            self.create_array(data_correct, keylist, 1, data_array, keys, ndim=1)
            self.create_array(data_correct, keylist, 1, data_dict, ndim=1)
            self.create_array(data_correct, keylist, 1, data_structured, ndim=1)

            # Overwrite keys in data
            self.create_array(data_correct2, keylist2, 1, data_dict, isotope_keys)
            self.create_array(data_correct2, keylist2, 1, data_structured, isotope_keys)
            self.create_array(data_correct2, keylist2, 1, data_dict, isotope_keys, ndim=1)
            self.create_array(data_correct2, keylist2, 1, data_structured, isotope_keys, ndim=1)

        # 1-dim input, n > 1
        data_list = [[1, 2, 3, 4], [11, 12, 13, 14], [21, 22, 23, 24]]
        data_tuple = [(1, 2, 3, 4), (11, 12, 13, 14), (21, 22, 23, 24)]
        data_array = np.array(data_list, dtype=np.float_)
        data_correct2 = np.array(data_tuple, dtype=[(str(keystring), np.float_) for keystring in keylist2])
        for keys in all_keys:
            keylist = isopy.keylist(keys)

            data_dict = dict(zip(keys, [[1, 11, 21], [2, 12, 22], [3, 13, 23], [4, 14, 24]]))
            data_structured = np.array(data_tuple,
                                       dtype=[(str(key), np.float_) for key in keys])
            data_correct = np.array(data_tuple,
                                    dtype=[(str(keystring), np.float_) for keystring in
                                           keylist])

            self.create_array(data_correct, keylist, 1, data_list, keys)
            self.create_array(data_correct, keylist, 1, data_tuple, keys)
            self.create_array(data_correct, keylist, 1, data_array, keys)
            self.create_array(data_correct, keylist, 1, data_dict)
            self.create_array(data_correct, keylist, 1, data_structured)

            self.create_array(data_correct, keylist, 1, data_list, keys, ndim=1)
            self.create_array(data_correct, keylist, 1, data_tuple, keys, ndim=1)
            self.create_array(data_correct, keylist, 1, data_array, keys, ndim=1)
            self.create_array(data_correct, keylist, 1, data_dict, ndim=1)
            self.create_array(data_correct, keylist, 1, data_structured, ndim=1)

            self.create_array(data_correct, keylist, 1, data_list, keys, ndim=-1)
            self.create_array(data_correct, keylist, 1, data_tuple, keys, ndim=-1)
            self.create_array(data_correct, keylist, 1, data_array, keys, ndim=-1)
            self.create_array(data_correct, keylist, 1, data_dict, ndim=-1)
            self.create_array(data_correct, keylist, 1, data_structured, ndim=-1)

            # Overwrite keys in data
            self.create_array(data_correct2, keylist2, 1, data_dict, isotope_keys)
            self.create_array(data_correct2, keylist2, 1, data_structured, isotope_keys)
            self.create_array(data_correct2, keylist2, 1, data_dict, isotope_keys, ndim=1)
            self.create_array(data_correct2, keylist2, 1, data_structured, isotope_keys, ndim=1)
            self.create_array(data_correct2, keylist2, 1, data_dict, isotope_keys, ndim=-1)
            self.create_array(data_correct2, keylist2, 1, data_structured, isotope_keys, ndim=-1)

    def create_array(self, correct, keylist, ndim, /, *args, **kwargs):
        result = isopy.array(*args, **kwargs)
        assert result.keys == keylist
        assert result.ndim == ndim
        assert_array_equal_array(result, correct)

    #TODO fails - mismatched axis
    #TODO axis = 1
    #TODO out, where
    #TODO specual cases. argmin, append concancate
    def test_functions(self):
        keys = ('104ru', '105Pd', '106Pd', 'cd111')

        values_one1 = [[j + i / 10 for i in range(10)] for j in range(4)]
        values_one2 = [[j] for j in range(4)]
        values_one3 = [j for j in range(4)]

        values_two1 = [[j*2 + i / 20 for i in range(10)] for j in range(4)]
        values_two1[0][0] = np.nan
        values_two2 = [[j*2] for j in range(4)]
        values_two2[0][0] = np.nan
        values_two3 = [j*2 for j in range(4)]
        values_two3[0] = np.nan

        all_values_one = (values_one1, values_one2, values_one3)
        all_values_two = (values_two1, values_two2, values_two3)

        for values1 in all_values_one:
            array1 = isopy.array({key: values1[i] for i, key in enumerate(keys)})

            #ufuncs
            for func in [np.sin, np.cos, np.tan, np.arcsin, np.arccos, np.arctan, np.degrees, np.isnan,
                          np.radians, np.deg2rad, np.rad2deg, np.sinh, np.cosh, np.tanh, np.arcsinh, np.arccosh, np.arctanh,
                          np.rint, np.floor, np.ceil, np.trunc, np.exp, np.expm1, np.exp2, np.log, np.log10, np.log2,
                          np.log1p, np.reciprocal, np.positive, np.negative, np.sqrt, np.cbrt, np.square, np.fabs, np.sign,
                          np.absolute, np.abs]:

                self.function1(func, array1)

            #array functions
            for func in [np.prod, np.sum, np.nanprod, np.nansum, np.cumprod, np.cumsum, np.nancumprod, np.nancumsum,
                          np.amin, np.amax, np.nanmin, np.nanmax, np.ptp, np.median, np.average, np.mean, np.std,
                          np.var, np.nanmedian, np.nanmean, np.nanstd, np.nanvar, isopy.mad, isopy.nanmad, isopy.se, isopy.nanse,
                          isopy.sd, isopy.nansd]:

                self.function1(func, array1)

            for values2 in all_values_two:
                input1 = {key: values2[i] for i, key in enumerate(keys)}
                input2 = {key: input1[key] for key in keys[:3]}
                array2 = isopy.array(input1)
                array3 = isopy.array(input2)

                for func in [np.add, np.multiply, np.divide, np.power, np.subtract,
                             np.true_divide, np.floor_divide, np.float_power, np.fmod, np.mod, np.remainder]:

                    self.function2(func, array1, array2)
                    self.function2(func, array1, array3)

    def function1(self, func, array):
        try: result = func(array)
        except Exception as err:
            raise AssertionError(f'function "{func}"') from err

        keys = array.keys()
        try:
            vres = func(array[keys[-1]])
        except Exception as err:
            warnings.warn(f'Could not run function {func} on 0-dim column values. Raised {type(err).__name__}("{str(err)}")')
        else:
            assert result.keys() == array.keys()
            assert result.size == vres.size
            assert result.ndim == vres.ndim
            for key in keys:
                np.testing.assert_allclose(result[key], func(array[key]))


    def function2(self, func, array1, array2):
        result = func(array1, array2)
        keys = array1.keys() | array2.keys()
        kand = array1.keys() & array2.keys()

        vres = func(array1[kand[-1]], array2[kand[-1]])
        assert result.keys() == keys
        assert result.size == vres.size
        assert result.ndim == vres.ndim
        for key in keys:
            np.testing.assert_allclose(result[key], func(array1.get(key), array2.get(key)))





