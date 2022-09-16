import IPython.core.display
import pandas

import isopy
import isopy.core
from isopy import core, checks
import numpy as np
import pytest
import itertools
import pyperclip
import warnings
import IPython
import pandas as pd


def assert_array_equal_array(array1, array2, match_dtype=True):
    assert isinstance(array1, core.IsopyArray)
    assert array1.flavour == array1.keys.flavour

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

class Test_Flavour:
    def test_creation(self):
        for flavour in 'mass element isotope molecule ratio general any'.split():
            assert isopy.asflavour(flavour) == flavour
            assert flavour in isopy.asflavour(flavour)
            assert flavour in isopy.asflavour('any')

            if flavour != 'any':
                assert flavour == isopy.core.FLAVOURS[flavour]()
                assert flavour in isopy.core.FLAVOURS[flavour]()

                assert isopy.core.FLAVOURS[flavour]() == flavour
                assert isopy.core.FLAVOURS[flavour]() == isopy.asflavour(flavour)
                assert isopy.core.FLAVOURS[flavour]() in isopy.asflavour(flavour)

                assert 'invalid' not in isopy.core.FLAVOURS[flavour]()
                assert isopy.core.FLAVOURS[flavour]() != 'invalid'

        assert isopy.asflavour('any') == 'any'
        assert isopy.asflavour('any') == 'mass|element|isotope|molecule|ratio|general'
        assert isopy.asflavour('mass|element|isotope|molecule|ratio|general') == 'any'
        assert isopy.asflavour('mass|element|isotope|molecule|ratio|general') == 'mass|element|isotope|molecule|ratio|general'

        assert isopy.asflavour('molecule') == 'molecule'
        assert isopy.asflavour('molecule') == 'molecule[any]'
        assert str(isopy.asflavour('molecule')) == 'molecule[any]'
        assert isopy.asflavour('molecule[any]') == 'molecule[any]'
        assert isopy.asflavour('molecule[any]') == 'molecule'
        assert isopy.asflavour('molecule[element]') == 'molecule'
        assert isopy.asflavour('molecule[element]') != 'molecule[any]'
        assert 'molecule[element]' in isopy.asflavour('molecule')
        assert 'molecule[element]' in isopy.asflavour('molecule[any]')
        assert 'molecule[element]' in isopy.asflavour('molecule[element|isotope]')
        assert 'molecule[element]' not in isopy.asflavour('molecule[isotope]')
        assert 'molecule[element, element]' not in isopy.asflavour('molecule[any]')

        with pytest.raises(ValueError):
            isopy.asflavour('element, isotope')

        with pytest.raises(ValueError):
            isopy.asflavour('invalid')

        with pytest.raises(ValueError):
            isopy.asflavour('molecule[isotope')

        with pytest.raises(ValueError):
            isopy.asflavour('isotope[mass, element]')

        with pytest.raises(ValueError):
            isopy.asflavour('molecule[element, isotope]')

        with pytest.raises(ValueError):
            isopy.asflavour('ratio[element, isotope, element]')

    def test_creation2(self):
        mass = isopy.keylist('101 105 111'.split())
        element = isopy.keylist('ru pd cd'.split())
        isotope = isopy.keylist('101ru 105pd 111cd'.split())
        ratio = isopy.keylist('ru/pd pd/pd cd/pd'.split())
        general = isopy.keylist('harry ron hermione'.split())
        mixed = isopy.keylist('ru 105pd cd/pd'.split())

        keys = (mass, element, isotope, ratio, general, mixed)
        flavours = (core.MassFlavour(), core.ElementFlavour(), core.IsotopeFlavour(),
                    core.RatioFlavour(numerator_flavour='element', denominator_flavour='element'),
                    core.GeneralFlavour(), isopy.asflavour('element|isotope|ratio[element, element]'))
        names = 'mass Element ISOTOPE RatiO[element] GENeral element|ISOTOPE|ratio[element,ELEMENT]'.split()

        for i in range(6):
            assert not flavours[i] == [i]
            assert not keys[i].flavour == i

            for j in range(6):
                if i == j:
                    assert keys[i].flavour == flavours[j]

                    assert keys[i].flavour == flavours[j]
                    assert keys[i].flavour == keys[j].flavour
                    assert keys[i].flavour == names[j]

                    assert names[i] == keys[j].flavour
                else:
                    assert keys[i].flavour != flavours[j]

                    assert keys[i].flavour != flavours[j]
                    assert keys[i].flavour != keys[j].flavour
                    assert keys[i].flavour != names[j]

                    assert names[i] != keys[j].flavour

    def test_is_flavour(self):
        key = isopy.keystring('ru')
        keys = isopy.keylist('ru', 'pd')
        array = isopy.ones(3, keys)

        assert isopy.iskeystring(key, flavour='element')
        assert not isopy.iskeylist(key, flavour='element')
        assert not isopy.isarray(key, flavour='element')
        assert not isopy.iskeystring(key, flavour='element|isotope')
        assert isopy.iskeystring(key, flavour_in='element|isotope')
        assert not isopy.iskeystring(key, flavour_in='mass|isotope')

        assert not isopy.iskeystring(keys, flavour='element')
        assert isopy.iskeylist(keys, flavour='element')
        assert not isopy.isarray(keys, flavour='element')
        assert not isopy.iskeylist(keys, flavour='element|isotope')
        assert isopy.iskeylist(keys, flavour_in='element|isotope')
        assert not isopy.iskeylist(keys, flavour_in='mass|isotope')

        assert not isopy.iskeystring(array, flavour='element')
        assert not isopy.iskeylist(array, flavour='element')
        assert isopy.isarray(array, flavour='element')
        assert not isopy.isarray(array, flavour='element|isotope')
        assert isopy.isarray(array, flavour_in='element|isotope')
        assert not isopy.isarray(array, flavour_in='mass|isotope')

    def test_keystring(self):
        general = isopy.asflavour('general')
        assert isopy.iskeystring(general._keystring_('pd'), flavour='general')

    def test_add(self):
        isotope = isopy.asflavour('isotope')
        element = core.FLAVOURS['element']()

        new = isotope + 'element'
        assert type(new) == core.ListFlavour
        assert new == 'element|isotope'

        new = 'element' + isotope
        assert type(new) == core.ListFlavour
        assert new == 'element|isotope'

        new = element + 'isotope'
        assert type(new) == core.ListFlavour
        assert new == 'element|isotope'

        new = 'isotope' + element
        assert type(new) == core.ListFlavour
        assert new == 'element|isotope'

        new = element + isotope
        assert type(new) == core.ListFlavour
        assert new == 'element|isotope'

        new =  isotope + element
        assert type(new) == core.ListFlavour
        assert new == 'element|isotope'


class Test_Exceptions:
    def test_KeyValueError(self):
        err = core.KeyValueError(core.ElementKeyString, 'pdd')
        assert err.cls is core.ElementKeyString
        assert err.string == 'pdd'
        assert err.additional_information == None
        assert str(err) == f'ElementKeyString: unable to parse "pdd"'

        err = core.KeyValueError(core.ElementKeyString, 'pdd', 'More info')
        assert err.cls is core.ElementKeyString
        assert err.string == 'pdd'
        assert err.additional_information == 'More info'
        assert str(err) == f'ElementKeyString: unable to parse "pdd". More info'

        try:
            core.ElementKeyString('pdd')
        except core.KeyValueError as err:
            assert err.string == 'pdd'
            assert err.cls is core.ElementKeyString
            assert type(err.additional_information) is str
            str(err)

    def test_KeyTypeError(self):
        err = core.KeyTypeError(core.ElementKeyString, 1)
        assert err.cls is core.ElementKeyString
        assert err.obj == 1
        assert str(err) == "ElementKeyString: cannot convert <class 'int'> into 'str'"

        try:
            core.ElementKeyString({'pdd'})
        except core.KeyTypeError as err:
            assert err.obj == {'pdd'}
            assert err.cls is core.ElementKeyString
            str(err)


class Test_IsopyKeyString:
    def test_direct_creation(self):
        # Test creation using the type directly
        # MassKeyString
        self.direct_creation(core.MassKeyString, correct='105',
                             same=[105, '105', 'MAS_105'],
                             fails=['pd105', 'onehundred and five', -105, '-105', 'GEN_105',
                                   'MAS__105', '_105'],
                             different=['96', 96, '5', 5, '300', 300, '1000', 1000])

        #ElementKeyString
        self.direct_creation(core.ElementKeyString, correct ='Pd',
                             same= ['pd', 'pD', 'PD', 'ELE_Pd', 'ELE_pd', 'Palladium', 'palladium'],
                             fails= ['', 1, '1', 'P1', '1P', '1Pd', '/Pd', '_Pd',
                                     'Pdd', 'GEN_Pd', 'ELE_p1', 'ELE__Pd'],
                             different= ['p', 'Cd', 'ru'])

        #IsotopeKeyString
        self.direct_creation(core.IsotopeKeyString, correct ='105Pd',
                             same= ['105PD', '105pd', '105pD', 'Pd105', 'Pd105', 'pd105',
                                     'pD105', 'ISO_105Pd', 'ISO_pd105', '105-pd', 'palladium-105'],
                             fails= ['', 'Pd', '105', 'Pd105a', 'P105D', '105Pd/108Pd', 'ISO__pd104',
                                     'GEN_105Pd'],
                             different=['104pd', 'Pd106', '104Ru', 'cd106', '1a', 'b2'])

        # MoleculeKeyString
        self.direct_creation(core.MoleculeKeyString, correct='H2O',
                             same=['(H2O)', '(H)2(O)', 'h2o', '[H2O]', '[H]2[O]', '[[H2O]]', '(((H2O)))', ('H', 2, 'O'), ('H2O'), (('H', 2, 'O'))],
                             fails='1(H2O) [H2O ++H2O 1H)O H(16O +(OH) 2 + OH!'.split() + [12, (), (2, 'H'), ('H', -2)],
                             different='HHO (2H)2O H2(16O) HNO3 HCl (OH)- 137Ba++ H+1HO H((OH)) ((OH)2)- H((OH)-)- O[H]'.split() +
                             [('H', 'H', 'O')])

        # MoleculeKeyString
        self.direct_creation(core.MoleculeKeyString, correct='HNO3',
                             same=['(HNO3)', ('H', 'N', 'O', 3), ('HNO', 3)],
                             different=[('(HNO)', 3), (('HNO',), 3)])

        self.direct_creation(core.RatioKeyString, correct ='105Pd/Pd',
                             same = ['105PD/Pd', '105pd/Pd', '105pD/Pd', 'Pd105/Pd',
                            'PD105/Pd', 'pd105/Pd', '105Pd/pd', '105Pd/PD',
                            'pd105/pD', 'Ratio_105Pd_Pd'],
                             fails=['', 'Pd', '105Pd', '/Pd', 'Pd/', '105Pd_SLASH_Pd',
                                    'RAT1_105Pd/Pd', 'Pd/Pd/Pd', 'Ratio_105Pd/Pd'],
                             different = ['RAT2_RAT1_ELE_Pd_OVER1_Cd_OVER2_ELE_Ru'
                                            ,'Pd/Pd', 'Pd/105Pd', '105Pd/108Pd', ('Pd', '105Pd'),
                                    '105Pd/Pd//Cd', 'a/b///c/d//e', ('pd105', 'Cd/Ru')])

        #GeneralKeyString
        self.direct_creation(core.GeneralKeyString, correct='Pd/pd',
                             same=['GEN_Pd/pd', 'GEN_Pd_SLASH_pd', 'Pd_SLASH_pd'],
                             fails = ['', None],
                             different=['Pd/Pd', 'GEN_Pd/Pd', 'Pd', '108Pd', 'Pd//Pd', 'Pd/Pd/Pd'])

    def direct_creation(self, keytype, correct, same, fails = [], different = []):
        #Make sure the correcttly formatted string can be created and is not reformatted
        correct_key = keytype(correct)

        assert correct_key == correct
        assert correct_key == keytype(correct)

        assert isinstance(correct_key, str)
        assert isinstance(correct_key, keytype)
        assert type(correct_key) is keytype

        # Make sure each of these string become the same key string
        for string in same:
            key = keytype(string)
            assert type(key) == keytype
            assert isinstance(key, str)
            assert key == string
            assert key == correct
            assert key == correct_key

            assert hash(key) == hash(correct_key)

            if str(key) == string:
                key = keytype(string, allow_reformatting=False)
                assert type(key) == keytype
                assert isinstance(key, str)
                assert key == string
                assert key == correct
                assert key == correct_key

                assert hash(key) == hash(correct_key)
            elif type(string) is str:
                try:
                    invalid_key = keytype(string, allow_reformatting=False)
                except core.KeyParseError:
                    pass
                else:
                    #This is for Ratios where keys will be general keys if reformatting is not allowed
                    #if invalid_key == key: raise ValueError(string)
                    assert invalid_key != key

        # Make sure these cannot be converted to a key string
        for string in fails:
            with pytest.raises(core.KeyParseError):
                key = keytype(string)
                raise ValueError(f'{string!r} should have raised an'
                                     f'KeyParseError but did not. Returned value was '
                                     f'"{key}" of type {type(key)}')

        # Make sure these string become a different key string
        for string in different:
            key = keytype(string)
            assert type(key) == keytype
            assert isinstance(key, str)
            assert key == string
            assert key != correct
            assert key != correct_key

            key2 = keytype(key)
            assert key2 == key

    def test_general_creation(self):
        # Test creation using *keystring* and *askeystring*
        self.general_creation(core.MassKeyString, [105, '105', 'MAS_105'])
        self.general_creation(core.ElementKeyString, ['Pd', 'pd', 'pD', 'PD', 'palladium', 'PALLADIUM'])
        self.general_creation(core.IsotopeKeyString, ['105Pd', '105PD', '105pd', '105pD', 'Pd105',
                                                  'Pd105', 'pd105', 'pD105', 'palladium-105', '105-PALLADIUM'])
        self.general_creation(core.MoleculeKeyString,
                              ['H2O', '(h2o)', '(H2O)', '(H)2(O)'])

        self.general_creation(core.RatioKeyString, ['105PD/Pd', '105pd/Pd', '105pD/Pd', 'Pd105/Pd',
                            'PD105/Pd', 'pd105/Pd', '105Pd/pd', '105Pd/PD', '105-Palladium/PALLADIUM',
                            'pd105/pD', 'Pd/Pd', 'Pd/Cd//Ru', 'Ru///Pd//Cd/Ag', 'Cd/Ru//Ru/Cd'])


        self.general_creation(core.GeneralKeyString, ['test', '-1', '/Pd', '105Pdd', 'Pd/Pd/Pd'])

        # Check behaviour of general string is as expected for keystring and askeystring.
        for key in ['105', 'Pd', '108Pd', 'H2O']:
            gkey = core.GeneralKeyString(key)
            ikey = isopy.keystring(gkey)
            assert type(gkey) != type(ikey)
            assert gkey != ikey
            assert gkey == key
            assert ikey == key

            gkey2 = isopy.askeystring(gkey)
            assert type(gkey) == type(gkey2)
            assert gkey == gkey2

    def general_creation(self, keytype, correct = []):
        #Test creation with isopy.keystring() and isopy.askeystring
        for string in correct:
            key1 = isopy.keystring(string)
            assert type(key1) == keytype
            assert isinstance(key1, str)
            assert key1 == string

            key2 = isopy.askeystring(key1)
            assert key2 == key1
            assert type(key2) == type(key1)

            key1 = isopy.askeystring(string)
            assert type(key1) == keytype
            assert isinstance(key1, str)
            assert key1 == string

            if str(key1) == string:
                key1 = isopy.keystring(string, allow_reformatting=False)
                assert type(key1) == keytype
                assert isinstance(key1, str)
                assert key1 == string

                key1 = isopy.askeystring(string, allow_reformatting=False)
                assert type(key1) == keytype
                assert isinstance(key1, str)
                assert key1 == string

            elif type(string) is str:
                #They will become general keys if reformatting is not allowed
                invalid_key = isopy.keystring(string, allow_reformatting=False)
                #if invalid_key == key1: raise ValueError(string)
                assert invalid_key != key1
                invalid_key = isopy.askeystring(string, allow_reformatting=False)
                #if invalid_key == key1: raise ValueError(string)
                assert invalid_key != key1

    def test_attributes(self):
        for key in '105 pd 105pd 108pd/105pd hermione'.split():
            key = isopy.keystring(key)
            with pytest.raises(AttributeError):
                key.random_attribute = None
                raise AssertionError('You should not be able to add attributes to key strings')

    def test_isotope_attributes(self):
        isotope = core.IsotopeKeyString('105Pd')
        assert hasattr(isotope, 'mass_number')
        assert hasattr(isotope, 'element_symbol')

        assert type(isotope.mass_number) is core.MassKeyString
        assert type(isotope.element_symbol) is core.ElementKeyString

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
        keys = ['105', 'Pd', '105Pd', 'test', '104Ru/106Cd', 'H2O']
        for numerator, denominator in itertools.permutations(keys, 2):
            ratio = core.RatioKeyString((numerator, denominator))
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
        ratio = core.GeneralKeyString('108Pd/Pd105')

        assert type(ratio.upper()) is str
        assert type(ratio.lower()) is str
        assert type(ratio.capitalize()) is str
        assert type(ratio[:1]) is str

        assert type(ratio[0]) is str
        assert type(ratio[1:]) is str

        assert type(ratio.split('/')[0]) == str

    def test_integer_manipulation(self):
        integer = 105
        mass = core.MassKeyString(integer)

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

        with pytest.raises(TypeError):
            assert mass > 'a'

        with pytest.raises(TypeError):
            assert mass >= 'a'

        with pytest.raises(TypeError):
            assert mass < 'a'

        with pytest.raises(TypeError):
            assert mass <= 'a'

    def test_divide_to_ratio(self):
        # Make sure you can divide key string to form ratio key strings
        for other in [core.IsotopeKeyString('74Ge'), core.RatioKeyString('88Sr/87Sr')]:
            for string in [core.MassKeyString('105'), core.ElementKeyString('Pd'),
                      core.IsotopeKeyString('105Pd'), core.GeneralKeyString('test'),
                    core.RatioKeyString('Cd/Ru'), core.MoleculeKeyString('H2O')]:
                ratio = string / other

                assert type(ratio) is core.RatioKeyString
                assert ratio.denominator == other
                assert type(ratio.numerator) is type(string)
                assert ratio.numerator == string

                ratio = other / string

                assert type(ratio) is core.RatioKeyString
                assert type(ratio.denominator) is type(string)
                assert ratio.denominator == string
                assert ratio.numerator == other

        prev = isopy.askeystring('la')
        for i, key in enumerate('ce pr nd pm sm eu gd tb dy'.split()):
            new = prev / key
            assert '/' * (i+1) in str(new)
            assert new.numerator == prev
            assert new.denominator == key

            prev = core.RatioKeyString(str(new))
            assert prev == new

        with pytest.raises(core.KeyValueError):
            prev / 'ho'


    def test_str(self):
        # Test the *str* method that turns key string into python strings

        key = core.MassKeyString('101')
        assert repr(key) == "MassKeyString('101')"
        assert key.str() == '101'
        str_options = dict(key='101', m='101')
        for k, v in str_options.items():
            assert key.str(k) == v
            assert key.str(f'key {{{k}}}') == f'key {v}'

        assert key.str('math') == r'101'
        assert key.str('latex') == '$101$'

        key = core.ElementKeyString('pd')
        assert repr(key) == "ElementKeyString('Pd')"
        assert key.str() == 'Pd'
        str_options = dict(key='Pd', es = 'pd', Es = 'Pd', ES = 'PD',
                           name = 'palladium', Name = 'Palladium', NAME = 'PALLADIUM')
        for k, v in str_options.items():
            assert key.str(k) == v
            assert key.str(f'key {{{k}}}') == f'key {v}'

        assert key.str('math') == r'\mathrm{Pd}'
        assert key.str('latex') == r'$\mathrm{Pd}$'

        key = core.IsotopeKeyString('101pd')
        assert repr(key) == "IsotopeKeyString('101Pd')"
        assert key.str() == '101Pd'
        str_options = dict(key = '101Pd', m = '101', es = 'pd', Es = 'Pd', ES = 'PD',
                           name = 'palladium', Name = 'Palladium', NAME = 'PALLADIUM',
                           mEs = '101Pd', ESm = 'PD101', namem = 'palladium101', mNAME = '101PALLADIUM')
        str_options.update({'NAME-m': 'PALLADIUM-101', 'm-es': '101-pd'})
        for k, v in str_options.items():
            assert key.str(k) == v
            assert key.str(f'key {{{k}}}') == f'key {v}'

        assert key.str('math') == r'{}^{101}\mathrm{Pd}'
        assert key.str('latex') == r'${}^{101}\mathrm{Pd}$'

        key = core.MoleculeKeyString('H2O')
        assert repr(key) == "MoleculeKeyString('[H2O]')"
        assert key.str() == '[H2O]'
        str_options = dict(key='[H2O]')
        for k, v in str_options.items():
            assert key.str(k) == v
            assert key.str(f'key {{{k}}}') == f'key {v}'

        assert key.str('math') == r'\mathrm{H}_{2}\mathrm{O}'
        assert key.str('latex') == r'$\mathrm{H}_{2}\mathrm{O}$'

        key = core.MoleculeKeyString('((16O)(2H))2-')
        assert repr(key) == "MoleculeKeyString('[([16O][2H])2-]')"
        assert key.str() == '[([16O][2H])2-]'
        str_options = dict(key='[([16O][2H])2-]')
        for k, v in str_options.items():
            assert key.str(k) == v
            assert key.str(f'key {{{k}}}') == f'key {v}'

        assert key.str('math') == r'\left({}^{16}\mathrm{O}{}^{2}\mathrm{H}\right)_{2}^{-}'
        assert key.str('latex') == r'$\left({}^{16}\mathrm{O}{}^{2}\mathrm{H}\right)_{2}^{-}$'

        key = core.RatioKeyString('pd/101pd')
        assert repr(key) == "RatioKeyString('Pd/101Pd')"
        assert key.str() == 'Pd/101Pd'
        str_options = {'key': 'Pd/101Pd', 'n/d': 'Pd/101Pd', 'n': 'Pd', 'd': '101Pd'}
        for k, v in str_options.items():
            assert key.str(k) == v
            assert key.str(f'key {{{k}}}') == f'key {v}'
        assert key.str('n/d', 'es', 'key {Name-m}') == 'pd/key Palladium-101'
        assert key.str(('n/d', 'es', 'Name-m')) == 'pd/Palladium-101'
        assert key.str({'format': 'n/d', 'nformat': 'es', 'dformat': 'Name-m'}) == 'pd/Palladium-101'

        assert key.str('math') == r'\cfrac{\mathrm{Pd}}{{}^{101}\mathrm{Pd}}'
        assert key.str('latex') == r'$\cfrac{\mathrm{Pd}}{{}^{101}\mathrm{Pd}}$'

        key = core.RatioKeyString('pd/ru//ag/cd')
        assert repr(key) == "RatioKeyString('Pd/Ru//Ag/Cd')"
        assert key.str() == 'Pd/Ru//Ag/Cd'
        str_options = {'key': 'Pd/Ru//Ag/Cd', 'n/d': 'Pd/Ru/Ag/Cd', 'n': 'Pd/Ru', 'd': 'Ag/Cd'}
        for k, v in str_options.items():
            assert key.str(k) == v
            assert key.str(f'key {{{k}}}') == f'key {v}'

        assert key.str('math') == r'\cfrac{\left(\cfrac{\mathrm{Pd}}{\mathrm{Ru}}\right)}{\left(\cfrac{\mathrm{Ag}}{\mathrm{Cd}}\right)}'
        assert key.str('latex') == r'$\cfrac{\left(\cfrac{\mathrm{Pd}}{\mathrm{Ru}}\right)}{\left(\cfrac{\mathrm{Ag}}{\mathrm{Cd}}\right)}$'

        key = core.GeneralKeyString('hermione')
        assert repr(key) == "GeneralKeyString('hermione')"
        assert key.str() == 'hermione'
        str_options = dict(key='hermione')
        for k, v in str_options.items():
            assert key.str(k) == v
            assert key.str(f'key {{{k}}}') == f'key {v}'

        assert key.str('math') == r'\mathrm{hermione}'
        assert key.str('latex') == r'$\mathrm{hermione}$'

        key = core.ElementKeyString('Pd')
        assert key._repr_latex_() == f'${key.str("latex")}$'

        core.IPYTHON_REPR = False
        assert key._repr_latex_() is None

        core.IPYTHON_REPR = True

    def test_charge(self):
        assert core.GeneralKeyString('Hermione++') == 'Hermione++'

        key = core.RatioKeyString('105Pd++/108Pd-')
        assert str(key) == '105Pd++/108Pd-'
        assert key.numerator.charge == 2
        assert key.denominator.charge == -1

        # Test charge attributes/methods for element and isotope key strings
        #Element
        for charge, strings in {-2: ['ba--'], -1: ['ba-'],
                                1: ['ba+'], 2: ['ba++']}.items():
            for string in strings:
                key = core.MoleculeKeyString(string)
                assert key.charge == charge
                assert str(key) == 'Ba' + ('+' * charge or '-' * abs(charge)) + ''

        key = core.ElementKeyString('Ba')
        for charge in [-2, -1, 1, 2]:
            c = '+' * charge or '-' * abs(charge)
            key2 = core.MoleculeKeyString((key, c))
            assert key2.charge is charge

        #isotope
        for charge, strings in {-2: [ '138ba--'], -1: ['138ba-'],
                                1: ['138ba+'], 2: ['138ba++']}.items():
            for string in strings:
                key = core.MoleculeKeyString(string)
                assert key.charge == charge
                assert str(key) == '138Ba' + ('+' * charge or '-' * abs(charge))
                assert key.mz == 138 / abs(charge)

        key = core.IsotopeKeyString('138Ba')
        for charge in [-2, -1, 1, 2]:
            c = '+' * charge or '-' * abs(charge)
            key2 = core.MoleculeKeyString((key, c))
            assert key2.charge is charge

        #molecule
        assert core.MoleculeKeyString('((1H)2(16O))').charge is None
        for charge, strings in {-2: ['((1H)2(16O))--'], -1: ['((1H)2(16O))-'],
                                1: ['((1H)2(16O))+'], 2: ['((1H)2(16O))++']}.items():
            for string in strings:
                key = core.MoleculeKeyString(string)
                assert key.charge == charge
                assert str(key) == '[([1H]2[16O])' + ('+' * charge or '-' * abs(charge)) + ']'
                assert key.mz == 18 / abs(charge)

        key = core.MoleculeKeyString('H2O')
        for charge in [-2, -1, 1, 2]:
            c = '+' * charge or '-' * abs(charge)
            key2 = core.MoleculeKeyString((key, c))
            assert key is not key2
            assert key.charge is None
            assert key2.charge is charge

    def test_contains_key(self):
        for string in '(2H)2(16O) (((16O)(1H))2)(18F)'.split():
            key = core.MoleculeKeyString(string)
            assert 'molecule[element]' not in key.flavour
            assert 'molecule[isotope]' in key.flavour

        for string in '(H)2(16O) ((OH)2)(18F)'.split():
            key = core.MoleculeKeyString(string)
            assert 'molecule[element]' in key.flavour
            assert 'molecule[isotope]' in key.flavour

        for string in '(H)2(O) ((OH)2)F'.split():
            key = core.MoleculeKeyString(string)
            assert 'molecule[element]' in key.flavour
            assert 'molecule[isotope]' not in key.flavour

    def test_element(self):
        molecule = core.MoleculeKeyString('H2O')
        assert molecule.element_symbol == 'H2O'

        molecule = core.MoleculeKeyString('(1H)2O')
        assert molecule.element_symbol == 'H2O'

        molecule = core.MoleculeKeyString('(1H)2(16O)')
        assert molecule.element_symbol == 'H2O'

        molecule = core.MoleculeKeyString('(1H)(2H)(16O)')
        assert molecule.element_symbol == 'HHO'

        molecule = core.MoleculeKeyString('(OH)2')
        assert molecule.element_symbol == '(OH)2'

        molecule = core.MoleculeKeyString('((16O)H)2')
        assert molecule.element_symbol == '(OH)2'

        molecule = core.MoleculeKeyString('((16O)(2H))2')
        assert molecule.element_symbol == '(OH)2'

        molecule = core.MoleculeKeyString('((16O)(2H))((18O)(1H))')
        assert molecule.element_symbol == '(OH)(OH)'

    def test_isotopes(self):
        molecule = core.MoleculeKeyString('O')
        n = len(isopy.refval.element.isotopes['o'])
        isotopes = molecule.isotopes
        assert len(isotopes) == n
        assert isotopes == [core.MoleculeKeyString(key) for key in isopy.refval.element.isotopes['o']]
        np.testing.assert_almost_equal(np.sum(isopy.refval.isotope.fraction.to_array(isotopes), axis=1), 1, decimal=5)

        molecule = core.MoleculeKeyString('O2')
        n = len(isopy.refval.element.isotopes['o']) ** 2
        isotopes = molecule.isotopes()
        assert len(isotopes) == n
        assert isotopes == [core.MoleculeKeyString(key) for key in
                            '(16O)(16O) (17O)(16O) (18O)(16O) ' \
                            '(16O)(17O) (17O)(17O) (18O)(17O) ' \
                            '(16O)(18O) (17O)(18O) (18O)(18O)'.split()]
        np.testing.assert_almost_equal(np.sum(isopy.refval.isotope.fraction.to_array(isotopes), axis=1), 1, decimal=5)

        molecule = core.MoleculeKeyString('OH')
        n = len(isopy.refval.element.isotopes['o']) * len(isopy.refval.element.isotopes['H'])
        isotopes = molecule.isotopes()
        assert len(isotopes) == n
        assert isotopes == [core.MoleculeKeyString(key) for key in
                            '(16O)(1H) (17O)(1H) (18O)(1H) ' \
                            '(16O)(2H) (17O)(2H) (18O)(2H)'.split()]
        np.testing.assert_almost_equal(np.sum(isopy.refval.isotope.fraction.to_array(isotopes), axis=1), 1, decimal=5)

        molecule = core.MoleculeKeyString('H2(16O)')
        n =  len(isopy.refval.element.isotopes['H']) ** 2
        isotopes = molecule.isotopes()
        assert len(isotopes) == n
        assert isotopes == [core.MoleculeKeyString(key) for key in
                            '((1H)(1H))(16O) ((2H)(1H))(16O) ' \
                            '((1H)(2H))(16O) ((2H)(2H))(16O)'.split()]
        np.testing.assert_almost_equal(np.sum(isopy.refval.isotope.fraction.to_array(isotopes), axis=1), isopy.refval.isotope.fraction['16o'], decimal=5)

        molecule = core.MoleculeKeyString('H2O')
        n = len(isopy.refval.element.isotopes['H']) ** 2 * len(isopy.refval.element.isotopes['O'])
        isotopes = molecule.isotopes()
        assert len(isotopes) == n
        np.testing.assert_almost_equal(np.sum(isopy.refval.isotope.fraction.to_array(isotopes), axis=1), 1, decimal=5)

        element = core.ElementKeyString('Pd')
        assert element.isotopes() == '102pd 104pd 105pd 106pd 108pd 110pd'.split()

        isotope = core.IsotopeKeyString('105pd')
        assert isotope.isotopes() == ['105pd']

    def test_molecule(self):
        key = core.MoleculeKeyString('pd')
        assert type(key) is core.MoleculeKeyString

        key = core.MoleculeKeyString('105pd')
        assert type(key) is core.MoleculeKeyString

    def test_mz(self):
        assert isopy.keystring('105Pd').mz == 105
        assert isopy.keystring('105Pd++').mz == 52.5


        assert isopy.keystring('(1H)2(16O)').mz == 18
        assert isopy.keystring('(1H)2(16O)++').mz == 10


# TODO creation add subflavours
class Test_IsopyList:
    def test_creation(self):
        mass = self.creation('mass',
                      correct = ['104', '105', '106'],
                      same = [('106', 105, 104)],
                      fail = ['105', '106', 'Pd'])

        element = self.creation('element',
                      correct=['Pd', 'Ag', 'Cd'],
                      same=[('pd', 'cD', 'AG')],
                      fail=['Cd', 'Pdd', 'Ag'])

        isotope = self.creation('isotope',
                      correct=['104Ru', '105Pd', '106Cd'],
                      same=[('pd105', '106CD', '104rU')],
                      fail=['p105d', '104ru', '106cd'])

        molecule = self.creation('molecule',
                                correct=['H2O', 'HNO3', 'HCl'],
                                same=[('h20', '(hNO3))', '(h)(cl)')],
                                fail=['2(H2O)', 'hno3', 'HCl'])

        molecule = self.creation('molecule[element]',
                                 correct=['H2O', 'HNO3', 'HCl'],
                                 same=[('h20', '(hNO3))', '(h)(cl)')],
                                 fail=['2(H2O)', 'hno3', 'HCl'])

        general = self.creation('general',
                      correct=['ron', 'harry', 'hermione'],
                      same=[('harry', 'ron', 'hermione')],
                      fail=None)

        for numerator, denominator in itertools.permutations((mass, element, isotope, molecule, general), 2):
            correct = [f'{n}/{d}' for n, d in zip(numerator, denominator)]
            same = [(correct[0], correct[2], correct[1])]
            fail = [correct[0], correct[2], f'{numerator[0]}']
            ratio = self.creation('ratio',
                          correct=correct,
                          same=same,
                          fail=fail)

            assert ratio.numerators.flavour == numerator.flavour
            assert ratio.numerators == numerator
            assert ratio.denominators.flavour == denominator.flavour
            assert ratio.denominators == denominator

            assert numerator / denominator == correct

            keys = [f'{n}/{denominator[0]}' for n in numerator]
            assert numerator / denominator[0] == keys
            assert [f'{n}' for n in numerator] / denominator[0] == keys
            assert numerator / f'{denominator[0]}' == keys
            assert isopy.askeylist(keys).common_denominator == denominator[0]

            keys = [f'{numerator[0]}/{d}' for d in denominator]
            assert numerator[0] / denominator == keys
            assert f'{numerator[0]}' / denominator == keys
            assert numerator[0] / [f'{d}' for d in denominator] == keys
            assert isopy.askeylist(keys).common_denominator is None

            ratio = self.creation(f'ratio[{numerator.flavour}, {denominator.flavour}]',
                                  correct=correct,
                                  same=same,
                                  fail=fail)

        for numerator in (mass, element, isotope, general, ratio):
            correct = [(n,d) for n, d in zip(numerator, ratio)]
            same = [(correct[0], correct[2], correct[1])]
            fail = None
            ratio2 = self.creation('ratio',
                                  correct=correct,
                                  same=same,
                                  fail=fail)
            assert ratio2.numerators.flavour == numerator.flavour
            assert ratio2.numerators == numerator
            assert ratio2.denominators.flavour == ratio.flavour
            assert ratio2.denominators == ratio

        assert isopy.keylist(['ru', 'pd', '105pd', '106pd', '110cd/105pd', '111cd/105pd']).flavour == 'element|isotope|ratio[isotope]'
        assert isopy.keylist(['ru', 'pd', '105pd', '106pd', '110cd/105pd', '111cd/105pd']).flavour == 'element|isotope|ratio[isotope, isotope]'

    def creation(self, flavour, correct, same, fail = None):
        dtype = np.dtype(dict(names=[isopy.keystring(k) for k in correct],  formats=[float for k in correct]))
        array = np.ones(1, dtype)
        d = {k: None for k in correct}
        same.extend([dtype, array, d])

        correct_list = isopy.keylist(correct)
        assert correct_list.flavour == flavour
        assert correct_list == correct
        assert len(correct_list) == len(correct)
        for k in correct:
            if isinstance(k, str):
                assert k in correct_list

        array = isopy.array([i for i in range(len(correct))], correct)
        same_list = isopy.keylist(array.dtype)
        assert len(same_list) == len(correct)
        assert same_list == correct
        assert same_list == correct_list
        assert same_list.flavour == flavour

        if not type(correct[0]) is tuple: #otherwise an error is given when a ratio is given as a tuple
            same_list = isopy.keylist(*correct)
            assert same_list == correct
            assert same_list.flavour == flavour

        same_list = isopy.keylist(correct)
        assert len(same_list) == len(correct)
        assert same_list == correct
        assert same_list == correct_list
        assert same_list.flavour == flavour

        same_list = isopy.askeylist(correct)
        assert len(same_list) == len(correct)
        assert same_list == correct
        assert same_list.flavour == flavour

        same_list = isopy.keylist(correct_list)
        assert same_list == correct
        assert same_list is not correct_list

        same_list = isopy.askeylist(correct_list)
        assert same_list == correct_list
        assert same_list.flavour == flavour

        dictionary = {k: i for i, k in enumerate(correct)}
        assert isopy.keylist(dictionary) == correct_list
        assert isopy.askeylist(dictionary) == correct_list

        array = isopy.array(dictionary)
        assert isopy.keylist(array) == correct_list
        assert isopy.askeylist(array) == correct_list

        if fail:
            with pytest.raises(core.KeyParseError):
                isopy.keylist(fail, flavour=flavour)

            mixed_list = isopy.keylist(fail)
            assert mixed_list.flavour != flavour
            assert flavour in mixed_list.flavour
            assert mixed_list != correct

            mixed_list = isopy.askeylist(fail)
            assert mixed_list.flavour != flavour
            assert flavour in mixed_list.flavour
            assert mixed_list != correct

        return correct_list

    def test_creation2(self):
        keys1 = 'ru cd pd'
        keys2 = keys1.split()

        assert isopy.keylist(keys1) == keys1
        assert isopy.keylist(keys1) == keys2
        assert isopy.askeylist(keys1) == keys1
        assert isopy.askeylist(keys1) == keys2

        key = isopy.keystring(keys1)
        assert isopy.iskeystring(key, flavour='general')

        keylist = isopy.keylist(key)
        assert len(keylist) == 1
        assert keylist[0] == key
        assert keylist != keys1

        keylist = isopy.askeylist(key)
        assert len(keylist) == 1
        assert keylist[0] == key
        assert keylist != keys1

        keylist1 = isopy.keylist(keys1, sort=False)
        keylist2 = isopy.keylist(keys1, sort=True)
        assert keylist1 != keylist2
        assert keylist1.sorted() == keylist2

    def test_compare(self):
        mass = self.compare('mass',
                             keys=['104', '105', '106'],
                             extra_keys=['99', '108', '111'],
                             notin=['70', '76', '80', 'Pd'],
                            other=['Ni', 'Ge', 'Se'])

        element = self.compare('element',
                                keys=['Ru', 'Pd', 'Cd'],
                                extra_keys=['Mo', 'Ag', 'Te'],
                                notin=['Ni', 'Ge', 'Se', '105Pd'],
                               other=['70Ge', '76Ge', '80Se'])

        isotope = self.compare('isotope',
                                keys=['104Ru', '105Pd', '106Cd'],
                                extra_keys=['99Ru', '106Pd', '111Cd'],
                                notin=['70Ge', '76Ge', '80Se', 'Pd'],
                               other=['Ni', 'Ge', 'Se'])

        general = self.compare('general',
                                keys=['ron', 'harry', 'hermione'],
                                extra_keys=['ginny', 'neville', 'luna'],
                                notin=['malfoy', 'crabbe', 'goyle', None])

        for numerator, denominator in itertools.permutations((mass, element, isotope, general), 2):
            keys = [f'{n}/{d}' for n, d in zip(numerator, denominator)]
            ratio = self.compare('ratio',
                                  keys=keys)


        for numerator in (mass, element, isotope, general, ratio):
            keys = [(n, d) for n, d in zip(numerator, ratio)]
            ratio2 = self.compare('ratio',
                                   keys=keys)

            assert numerator / ratio == keys

    def compare(self, flavour, keys, extra_keys=[], notin=[], other=None):
        keys2 = keys + keys + extra_keys
        keys3 = keys + extra_keys

        keylist = isopy.askeylist(keys)
        assert keylist.flavour == flavour
        assert keylist == keys
        assert keylist != keys[:-1]
        assert (keylist == keys[:-1]) is False
        assert len(keylist) == len(keys)
        assert len(set(keylist)) == len(keylist)
        for key in keys:
            if isinstance(key, str):
                assert key in keylist
                assert keylist.count(key) == 1
        for i in range(len(keys)):
            assert keylist[i] == keys[i]
            assert keylist.index(keys[i]) == keys.index(keys[i])

        for k in notin:
            assert k not in keylist

        if other is not None:
            assert keylist != other

        keylist = isopy.askeylist(keys, ignore_duplicates = True)
        assert keylist == keys

        keylist = isopy.askeylist(keys, allow_duplicates=False)
        assert keylist == keys

        keylist2 = isopy.askeylist(keys2)
        assert keylist2 != keylist
        assert keylist2 == keys2
        assert len(keylist2) == len(keys2)
        assert len(set(keylist2)) != len(keylist2)
        for key in keys:
            if isinstance(key, str):
                assert key in keylist2
                assert keylist2.count(key) == 2
        for i in range(len(keys2)):
            assert keylist2[i] == keys2[i]
            assert keylist2.index(keys2[i]) == keys2.index(keys2[i])

        keylist3 = isopy.askeylist(keys2, ignore_duplicates = True)
        assert keylist3 == keys3
        assert len(keylist3) == len(keys3)
        assert len(set(keylist3)) == len(keylist3)
        for key in keys3:
            if isinstance(key, str):
                assert key in keylist3
                assert keylist3.count(key) == 1
        for i in range(len(keys3)):
            assert keylist3[i] == keys3[i]
            assert keylist3.index(keys3[i]) == keys3.index(keys3[i])

        with pytest.raises(ValueError):
            isopy.askeylist(keys2, allow_duplicates=False)

        return keylist

    def test_bitwise(self):
        #Last two of key1 should be the first two of key2
        #Otherwise the ratio test will fail
        mass = self.bitwise('mass', (102, 104, 105, 106),
                    ('105', '106', '108', '110'),
                     ('105', '106'),
                     (102, 104, 105, 106, 108, 110),
                     ('102', '104', 108, 110))

        element = self.bitwise('element', ('Mo', 'Ru', 'Pd', 'Rh'),
                     ('Pd', 'Rh', 'Ag', 'Cd'),
                     ('Pd', 'Rh'),
                     ('Mo', 'Ru', 'Pd', 'Rh', 'Ag', 'Cd'),
                     ('Mo', 'Ru', 'Ag', 'Cd'))

        isotope= self.bitwise('isotope', ('102Pd', '104Pd', '105Pd', '106Pd'),
                     ('105Pd', '106Pd', '108Pd', '110Pd'),
                     ('105Pd', '106Pd'),
                     ('102Pd', '104Pd', '105Pd', '106Pd', '108Pd', '110Pd'),
                     ('102Pd', '104Pd', '108Pd', '110Pd'))

        general = self.bitwise('general', ('Hermione', 'Neville', 'Harry', 'Ron'),
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

            self.bitwise('ratio', key1, key2, rand, ror, rxor)

    def bitwise(self, flavour, keys1, keys2, band, bor, bxor):
        keylist1 = isopy.keylist(*keys1)
        keylist2 = isopy.keylist(*keys2, *keys2[-2:])

        keyband = keylist1 & keylist2
        assert keyband.flavour == flavour
        assert keyband == band
        assert keylist1 & keys2 == band
        assert keys1 & keylist2 == band


        keybor = keylist1 | keylist2
        assert keybor.flavour == flavour
        assert keybor == bor
        assert keylist1 | keys2 == bor
        assert keys1 | keylist2 == bor

        keybxor = keylist1 ^ keylist2
        assert keybxor.flavour == flavour
        assert keybxor == bxor
        assert keylist1 ^ keys2 == bxor
        assert keys1 ^ keylist2 == bxor

        return keyband, keybor, keybxor

    def test_filter(self):
        mass = isopy.keylist(['104', '105', '106', '108', '104', '108'])
        self.filter_key('mass', 'key', mass, mass, ['103', '107'])
        self.filter_mass('mass', 'key', mass, mass)

        element = isopy.keylist(['Ru', 'Pd', 'Pd', 'Cd', 'Ru', 'Cd'])
        self.filter_key('element', 'key', element, element, ['Rh', 'Ag'])

        isotope = isopy.keylist(['104Ru', '105Pd', '106Pd', '108Cd', '104Ru', '108Cd'])
        self.filter_key('isotope', 'key', isotope, isotope, ['103Rh', '107Ag'])
        self.filter_mass('isotope', 'mass_number', isotope, mass)
        self.filter_key('isotope', 'element_symbol', isotope, element, ['Rh', 'Ag'])

        assert isotope.filter(mass_number_gt = 105, element_symbol_eq = 'pd') == \
               [key for key in isotope if (key.element_symbol in ['pd'] and key.mass_number > 105)]

        molecule = isopy.keylist(['(H2O)', '(HNO3)', '(HBr)', '(HCl)', '(HF)', '(HI)'])
        self.filter_key('molecule', 'key', molecule, molecule, ['(OH)', '(H2O2)'])

        general = isopy.keylist(['harry', 'ron', 'hermione', 'neville', 'harry', 'neville'])
        self.filter_key('general', 'key', general, general, ['ginny', 'luna'])

        ratio = general / isotope
        self.filter_key('ratio', 'key', ratio, ratio, ['ginny/103Rh', 'luna/107Ag'])
        self.filter_key('ratio', 'numerator', ratio, general, ['ginny', 'luna'])
        self.filter_key('ratio', 'denominator', ratio, isotope, ['103Rh', '107Ag'])
        self.filter_key('ratio', 'denominator_element_symbol', ratio, element, ['Rh', 'Ag'])
        self.filter_mass('ratio', 'denominator_mass_number', ratio, mass)

        result = ratio.filter(numerator_eq=['ron', 'harry', 'hermione'],
                            denominator_mass_number_ge=105, denominator_element_symbol_eq='pd')
        assert result == ratio[1:3]

        ratio2 = ratio / element
        self.filter_key('ratio', 'key', ratio2, ratio2, ['ginny//103Rh/Rh', 'luna//107Ag/ag'])
        self.filter_key('ratio', 'numerator', ratio2, ratio, ['ginny/103Rh', 'luna/107Ag'])
        self.filter_key('ratio', 'numerator_numerator', ratio2, general, ['ginny', 'luna'])
        self.filter_key('ratio', 'numerator_denominator', ratio2, isotope, ['103Rh', '107Ag'])
        self.filter_key('ratio', 'numerator_denominator_element_symbol', ratio2, element, ['Rh', 'Ag'])
        self.filter_mass('ratio', 'numerator_denominator_mass_number', ratio2, mass)
        self.filter_key('ratio', 'denominator', ratio2, element, ['Rh', 'Ag'])

        result = ratio2.filter(numerator_numerator_eq=['ron', 'harry', 'hermione'],
                             numerator_denominator_mass_number_ge=105,
                             numerator_denominator_element_symbol_eq='pd')
        assert result == ratio2[1:3]

        result = ratio2.filter(numerator_numerator_eq=['ron', 'harry', 'hermione'],
                             numerator_denominator_mass_number_ge=105,
                             denominator_eq='pd')
        assert result == ratio2[1:3]

    def filter_mass(self, flavour, prefix, keys, mass_list):
        keylist = isopy.keylist(keys)
        mass_list = isopy.keylist(mass_list)

        result = keylist.filter(**{f'{prefix}_gt': mass_list[1]})
        assert result == [keylist[i] for i, key in enumerate(mass_list) if key > mass_list[1]]
        assert len(result) != 0

        result = keylist.filter(**{f'{prefix}_ge': mass_list[1]})
        assert result == [keylist[i] for i, key in enumerate(mass_list) if key >= mass_list[1]]
        assert result.flavour == flavour
        assert len(result) != 0

        result = keylist.filter(**{f'{prefix}_lt': mass_list[1]})
        assert result == [keylist[i] for i, key in enumerate(mass_list) if key < mass_list[1]]
        assert result.flavour == flavour
        assert len(result) != 0

        result = keylist.filter(**{f'{prefix}_le': mass_list[1]})
        assert result == [keylist[i] for i, key in enumerate(mass_list) if key <= mass_list[1]]
        assert result.flavour == flavour
        assert len(result) != 0

        result = keylist.filter(**{f'{prefix}_le': mass_list[2], f'{prefix}_gt': mass_list[0]})
        assert result == [keylist[i] for i, key in enumerate(mass_list) if
                          (key <= mass_list[2] and key > mass_list[0])]
        assert result.flavour == flavour
        assert len(result) != 0

        result = keylist.filter(**{f'{prefix}_lt': mass_list[2], f'{prefix}_ge': mass_list[0]})
        assert result == [keylist[i] for i, key in enumerate(mass_list) if
                          (key < mass_list[2] and key >= mass_list[0])]
        assert result.flavour == flavour
        assert len(result) != 0

        result = keylist.filter(**{f'{prefix}_eq': mass_list[1:3], f'{prefix}_lt': mass_list[2]})
        assert result == [keylist[i] for i, key in enumerate(mass_list) if
                          (key in mass_list[1:3] and key < mass_list[2])]
        assert result.flavour == flavour
        assert len(result) != 0

    def filter_key(self, flavour, prefix, keys, eq, neq):
        keylist = isopy.askeylist(keys)
        eq = isopy.askeylist(eq)
        neq = isopy.askeylist(neq)

        result = keylist.filter()
        assert result.flavour == flavour
        assert result == keylist

        result = keylist.filter(**{f'{prefix}': eq[0]})
        assert result.flavour == flavour
        assert result == [keylist[i] for i, key in enumerate(eq) if key == eq[0]]
        assert len(result) != 0

        result = keylist.filter(**{f'{prefix}_eq': eq[0]})
        assert result.flavour == flavour
        assert result == [keylist[i] for i, key in enumerate(eq) if key == eq[0]]
        assert len(result) != 0

        result = keylist.filter(**{f'{prefix}_eq': eq[2:]})
        assert result.flavour == flavour
        assert result == [keylist[i] for i, key in enumerate(eq) if key in eq[2:]]
        assert len(result) != 0

        result = keylist.filter(**{f'{prefix}_eq': (neq + eq[2:])})
        assert result.flavour == flavour
        assert result == [keylist[i] for i, key in enumerate(eq) if key in (neq + eq[2:])]
        assert len(result) != 0

        result = keylist.filter(**{f'{prefix}_eq': neq})
        assert result == [keylist[i] for i, key in enumerate(eq) if key in neq]
        assert len(result) == 0

        result = keylist.filter(**{f'{prefix}_neq': neq})
        assert result.flavour == flavour
        assert result == [keylist[i] for i, key in enumerate(eq) if key not in neq]
        assert len(result) != 0

        result = keylist.filter(**{f'{prefix}_neq': eq[-1]})
        assert result.flavour == flavour
        assert result == [keylist[i] for i, key in enumerate(eq) if key != eq[-1]]
        assert len(result) != 0

        result = keylist.filter(**{f'{prefix}_neq': (eq[-1:] + neq)})
        assert result.flavour == flavour
        assert result == [keylist[i] for i, key in enumerate(eq) if key not in (eq[-1:] + neq)]
        assert len(result) != 0

        result = keylist.filter(**{f'{prefix}_eq': eq[:-1], f'{prefix}_neq': (neq + eq[:2])})
        assert result.flavour == flavour
        assert result == [keylist[i] for i, key in enumerate(eq) if
                          (key in eq[:-1] and key not in (neq + eq[:2]))]
        assert len(result) != 0

        if prefix != 'key':
            result = keylist.filter(**{f'{prefix}': eq[0]})
            assert result.flavour == flavour
            assert result == [keylist[i] for i, key in enumerate(eq) if key == eq[0]]
            assert len(result) != 0

            result = keylist.filter(**{f'{prefix}': eq[2:]})
            assert result.flavour == flavour
            assert result == [keylist[i] for i, key in enumerate(eq) if key in eq[2:]]
            assert len(result) != 0

        return keylist

    def test_filter_mz(self):
        keylist = isopy.keylist('64zn  132ba-- 66zn 67zn 68zn 137ba++ 70zn 136ba'.split())

        assert keylist.filter(mz_eq = 66) == '132ba-- 66zn'.split()
        assert keylist.filter(mz_eq = [66, 68]) == '132ba-- 66zn 68zn'.split()

        assert keylist.filter(mz_neq=66) == '64zn  67zn 68zn 137ba++ 70zn 136ba'.split()
        assert keylist.filter(mz_neq=[66, 68]) == '64zn 67zn 137ba++ 70zn 136ba'.split()

        assert keylist.filter(mz_lt = 66) == '64zn'.split()

        assert keylist.filter(mz_le=66) == '64zn  132ba-- 66zn'.split()

        assert keylist.filter(mz_gt=66) == '67zn 68zn 137ba++ 70zn 136ba'.split()

        assert keylist.filter(mz_ge=66) == '132ba-- 66zn 67zn 68zn 137ba++ 70zn 136ba'.split()

        keylist = isopy.askeylist('64zn  132ba-- 66zn 67zn 68zn 137ba++ 70zn 136ba'.split())

        assert keylist.filter(mz_eq=66) == '132ba-- 66zn'.split()
        assert keylist.filter(mz_eq=[66, 68]) == '132ba-- 66zn 68zn'.split()

        assert keylist.filter(mz_neq=66) == '64zn  67zn 68zn 137ba++ 70zn 136ba'.split()
        assert keylist.filter(mz_neq=[66, 68]) == '64zn 67zn 137ba++ 70zn 136ba'.split()

        assert keylist.filter(mz_lt=66) == '64zn'.split()

        assert keylist.filter(mz_le=66) == '64zn  132ba-- 66zn'.split()

        assert keylist.filter(mz_gt=66) == '67zn 68zn 137ba++ 70zn 136ba'.split()

        assert keylist.filter(mz_ge=66) == '132ba-- 66zn 67zn 68zn 137ba++ 70zn 136ba'.split()

    def test_filter_flavour(self):
        keylist = isopy.keylist(['ru', 'pd', '105pd', '106pd', '110cd/105pd', '111cd/105pd'])

        sublist1 = keylist.filter(flavour_eq ='element')
        sublist2 = keylist.filter(flavour_neq = ['isotope', 'ratio'])
        assert sublist1.flavour == 'element'
        assert sublist2.flavour == 'element'
        assert sublist1 == ['ru', 'pd']
        assert sublist2 == ['ru', 'pd']

        sublist1 = keylist.filter(flavour_eq=['isotope','ratio'])
        sublist2 = keylist.filter(flavour_neq='element')
        assert sublist1.flavour == 'isotope|ratio[isotope]'
        assert sublist2.flavour in isopy.asflavour('isotope|ratio')
        assert sublist1 == ['105pd', '106pd', '110cd/105pd', '111cd/105pd']
        assert sublist2 == ['105pd', '106pd', '110cd/105pd', '111cd/105pd']

        keylist = isopy.keylist(['105', 'pd', '105pd', '110cd/105pd', 'H2O', 'harry',
                                 '96', 'ru', '96ru', '96ru/101ru', 'HNO3', 'hermione'])
        type_names = ['mass', 'element', 'isotope', 'ratio', 'molecule', 'general']
        for i in range(5):
            assert type_names[i] in keylist.flavour

            sublist = keylist.filter(flavour_eq=type_names[i])
            assert sublist.flavour == type_names[i]

            sublist = keylist.filter(flavour_neq=type_names[i])
            assert type_names[i] not in sublist.flavour

    def test_invalid_filter(self):
        keylist = isopy.keylist(['105', 'pd', '105pd', '110cd/105pd', 'H2O', 'Hermione'])

        for key in keylist:
            assert key._filter_(invalid = None) is False

        for key in keylist:
            assert key._filter_(invalid_eq = None) is False

        for key in keylist:
            assert key._filter_(mass_number_gt = 'a') is False

        for key in keylist:
            assert key._filter_(key_eq = [None]) is False

        assert len(keylist.filter(invalid = 'a')) == 0

        assert len(keylist.filter(invalid_eq = 'a')) == 0

        assert len(keylist.filter(mass_number_gt = 'a')) == 0

        assert len(keylist.filter(key_eq = [None])) == 0

    def test_add_subtract(self):
        # Add
        keylist = isopy.keylist('pd', 'cd')

        new = keylist + 'ru rh pd'.split()
        assert new == 'pd cd ru rh pd'.split()
        assert new.flavour == 'element'

        new = 'ru rh pd'.split() + keylist
        assert new == 'ru rh pd pd cd '.split()
        assert new.flavour == 'element'

        new = keylist + ['105pd', '111cd']
        assert new == 'pd cd 105pd 111cd'.split()
        assert new.flavour == 'element|isotope'

        new = ['105pd', '111cd'] + keylist
        assert new == '105pd 111cd pd cd '.split()
        assert new.flavour == 'element|isotope'

        new = ['105pd', '111cd+'] + keylist
        assert new == '105pd 111cd+ pd cd '.split()
        assert new.flavour == 'element|isotope|molecule[isotope]'

        #Sub
        keylist = isopy.keylist('pd', 'cd', '105pd', '111cd')

        new = keylist - 'cd 105pd'.split()
        assert new == 'pd 111cd'.split()
        assert new.flavour == 'element|isotope'

        new = keylist - '111cd 105pd'.split()
        assert new == 'pd cd'.split()
        assert new.flavour == 'element'

        new = 'rh pd 107ag'.split() - keylist
        assert new == 'rh 107ag'.split()
        assert new.flavour == 'element|isotope'

        new = 'rh pd ag'.split() - keylist
        assert new == 'rh ag'.split()
        assert new.flavour == 'element'

        keylist = isopy.keylist('pd', 'cd', '105pd', '111cd+')

        new = keylist - 'cd 105pd'.split()
        assert new == 'pd 111cd+'.split()
        assert new.flavour == 'element|molecule[isotope]'

        new = keylist - '111cd+ 105pd'.split()
        assert new == 'pd cd'.split()
        assert new.flavour == 'element'

        new = keylist - '111cd+'.split()
        assert new == 'pd cd 105pd'.split()
        assert new.flavour == 'element|isotope'

    def test_sorted(self):
        mass = isopy.askeylist('102 104 105 106 108 110'.split())
        mass2 = isopy.askeylist('104 108 102 110 106 105'.split())
        assert mass != mass2
        assert mass == mass2.sorted()

        element = isopy.askeylist('ru rh pd ag cd te'.split())
        element2 = isopy.askeylist('pd cd te ag rh ru'.split())
        assert element != element2
        assert element == element2.sorted()

        isotope = isopy.askeylist('102ru 102pd 104ru 104pd 106pd 106cd'.split())
        isotope2 = isopy.askeylist('106cd 104ru 102ru 104pd 102pd 106pd'.split())
        assert isotope != isotope2
        assert isotope == isotope2.sorted()

        molecule =  isopy.askeylist('H2O', 'HCl', '(OH)2', 'HNO3')
        molecule2 = isopy.askeylist('H2O HNO3 HCl (OH)2'.split())
        assert molecule != molecule2
        assert molecule == molecule2.sorted()

        molecule = isopy.askeylist('H2O (2H)2(16O)++ (1H)2(16O) (2H)2(16O)'.split())
        molecule2 = isopy.askeylist('(2H)2(16O)++ (2H)2(16O) (1H)2(16O) H2O'.split())
        assert molecule != molecule2
        assert molecule == molecule2.sorted()

        general = isopy.askeylist('ginny harry hermione luna neville ron'.split())
        general2 = isopy.askeylist('hermione ginny luna ron neville harry'.split())
        assert general != general2
        assert general == general2.sorted()

        ratio = element / 'pd'
        ratio2 = element2 / 'pd'
        assert ratio != ratio2
        assert ratio == ratio2.sorted()

        ratio = 'pd' / element
        ratio2 = 'pd' / element2
        assert ratio != ratio2
        assert ratio == ratio2.sorted()

        mixed = isopy.askeylist('105 pd 99ru H2O 108pd/105pd hermione'.split())
        mixed2 = isopy.askeylist('H2O 108pd/105pd 99ru 105 hermione pd'.split())
        assert mixed != mixed2
        assert mixed == mixed2.sorted()

    def test_reversed(self):
        mass = isopy.askeylist('104 108 102 110 106 105'.split()).reversed()
        assert mass != '104 108 102 110 106 105'.split()
        assert mass == list(reversed('104 108 102 110 106 105'.split()))

        element = isopy.askeylist('pd cd te ag rh ru'.split()).reversed()
        assert element != 'pd cd te ag rh ru'.split()
        assert element == list(reversed('pd cd te ag rh ru'.split()))

        isotope = isopy.askeylist('106cd 104ru 102ru 104pd 102pd 106pd'.split()).reversed()
        assert isotope != '106cd 104ru 102ru 104pd 102pd 106pd'.split()
        assert isotope == list(reversed('106cd 104ru 102ru 104pd 102pd 106pd'.split()))

        molecule = isopy.askeylist('H2O HNO3 HCl (OH)2'.split()).reversed()
        assert molecule != 'H2O HNO3 HCl (OH)2'.split()
        assert molecule == list(reversed('H2O HNO3 HCl (OH)2'.split()))

        general = isopy.askeylist('hermione ginny luna ron neville harry'.split()).reversed()
        assert general != 'hermione ginny luna ron neville harry'.split()
        assert general == list(reversed('hermione ginny luna ron neville harry'.split()))

        ratio = (isopy.askeylist('pd cd te ag rh ru'.split()) / 'pd').reversed()
        assert ratio.numerators != 'pd cd te ag rh ru'.split()
        assert ratio.numerators == list(reversed('pd cd te ag rh ru'.split()))

        ratio = ('pd' / isopy.askeylist('pd cd te ag rh ru'.split())).reversed()
        assert ratio.denominators != 'pd cd te ag rh ru'.split()
        assert ratio.denominators == list(reversed('pd cd te ag rh ru'.split()))

        mixed = isopy.askeylist('H2O 108pd/105pd 99ru 105 hermione pd'.split()).reversed()
        assert mixed != 'H2O 108pd/105pd 99ru 105 hermione pd'.split()
        assert mixed == list(reversed('H2O 108pd/105pd 99ru 105 hermione pd'.split()))

    def test_str(self):
        # Test the *strlist* method that turns key string into python strings

        key = isopy.askeylist('101')
        assert repr(key) == """IsopyKeyList('101', flavour='mass')"""
        assert key.strlist() == ['101']
        str_options = dict(key='101', m='101')
        for k, v in str_options.items():
            assert key.strlist(k) == [v]
            assert key.strlist(f'key {{{k}}}') == [f'key {v}']

        key = isopy.askeylist('pd')
        assert repr(key) == """IsopyKeyList('Pd', flavour='element')"""
        assert key.strlist() == ['Pd']
        str_options = dict(key='Pd', es = 'pd', Es = 'Pd', ES = 'PD',
                           name = 'palladium', Name = 'Palladium', NAME = 'PALLADIUM')
        for k, v in str_options.items():
            assert key.strlist(k) == [v]
            assert key.strlist(f'key {{{k}}}') == [f'key {v}']

        key = isopy.askeylist('101pd')
        assert repr(key) == """IsopyKeyList('101Pd', flavour='isotope')"""
        assert key.strlist() == ['101Pd']
        str_options = dict(key = '101Pd', m = '101', es = 'pd', Es = 'Pd', ES = 'PD',
                           name = 'palladium', Name = 'Palladium', NAME = 'PALLADIUM',
                           mEs = '101Pd', ESm = 'PD101', namem = 'palladium101', mNAME = '101PALLADIUM')
        str_options.update({'NAME-m': 'PALLADIUM-101', 'm-es': '101-pd'})
        for k, v in str_options.items():
            assert key.strlist(k) == [v]
            assert key.strlist(f'key {{{k}}}') == [f'key {v}']

        key = isopy.askeylist('H2O')
        assert repr(key) == """IsopyKeyList('[H2O]', flavour='molecule[element]')"""
        assert key.strlist() == ['[H2O]']
        str_options = dict(key='[H2O]')
        for k, v in str_options.items():
            assert key.strlist(k) == [v]
            assert key.strlist(f'key {{{k}}}') == [f'key {v}']

        key = isopy.askeylist('pd/101pd')
        assert repr(key) == "IsopyKeyList('Pd/101Pd', flavour='ratio[element, isotope]')"
        assert key.strlist() == ['Pd/101Pd']
        str_options = {'key': 'Pd/101Pd', 'n/d': 'Pd/101Pd', 'n': 'Pd', 'd': '101Pd'}
        for k, v in str_options.items():
            assert key.strlist(k) == [v]
            assert key.strlist(f'key {{{k}}}') == [f'key {v}']
        assert key.strlist(('n/d', 'es', 'Name-m')) == ['pd/Palladium-101']
        assert key.strlist({'format': 'n/d', 'nformat': 'es', 'dformat': 'Name-m'}) == ['pd/Palladium-101']

        key = isopy.askeylist('hermione')
        assert repr(key) == "IsopyKeyList('hermione', flavour='general')"
        assert key.strlist() == ['hermione']
        str_options = dict(key='hermione')
        for k, v in str_options.items():
            assert key.strlist(k) == [v]
            assert key.strlist(f'key {{{k}}}') == [f'key {v}']

        # The key.str() method is tested elsewhere so we can use it here
        keys = isopy.keylist('101 pd 105pd H2O pd/ru hermione'.split())
        keystr = ', '.join([k.str() for k in keys])
        assert keys.str() == f'[{keystr}]'

        keys = isopy.keylist('101 pd 105pd H2O pd/ru hermione'.split())
        keystr = ', '.join([k.str('key = {key}') for k in keys])
        assert keys.str('key = {key}') == f'[{keystr}]'

        keystr = ', '.join([k.str('math') for k in keys])
        assert keys.str('math') == fr'\left[{keystr}\right]'
        assert keys.str('latex') == fr'$\left[{keystr}\right]$'

        assert keys._repr_latex_() == fr'$$\left[{keystr}\right]$$'
        core.IPYTHON_REPR = False
        assert keys._repr_latex_() is None
        core.IPYTHON_REPR = True

    def test_flatten(self):
        element = isopy.keylist('ru pd pd cd'.split())
        assert element.flatten() == 'ru pd pd cd'.split()
        assert element.flatten(ignore_duplicates=True) == 'ru pd cd'.split()

        ratio = isopy.keylist('ru/pd cd/pd'.split())
        assert ratio.flatten() == 'ru pd cd pd'.split()
        assert ratio.flatten(ignore_duplicates=True) == 'ru pd cd'.split()

        mixed = isopy.keylist('ru/pd pd cd'.split())
        assert mixed.flatten() == 'ru pd pd cd'.split()
        assert mixed.flatten(ignore_duplicates=True) == 'ru pd cd'.split()

        ratio = isopy.keylist('ru//rh/pd cd/te//ag'.split())
        assert ratio.flatten() == 'ru rh pd cd te ag'.split()

    def test_getitem(self):
        keys = 'ru pd ag cd'.split()
        keylist = isopy.askeylist(keys)

        for i in range(len(keys)):
            assert keylist[i] == keys[i]

        assert keylist[:] == keys
        assert keylist[:2] == keys[:2]
        assert keylist[1:3] == keys[1:3]

        assert keylist[(1,3)] == 'pd cd'.split()

        assert keylist[(2,)] == 'ag'
        assert isinstance(keylist[(2,)], core.IsopyKeyList)

    def test_divide(self):
        keylist = isopy.keylist('ru pd ag cd'.split())
        keys = 'la ce sm nd'.split()
        rkeys = 'ru/la pd/ce ag/sm cd/nd'.split()
        r_rkeys = 'la/ru ce/pd sm/ag nd/cd'.split()

        assert keylist / keys == rkeys
        assert keys / keylist == r_rkeys

        keys = 'ge'
        rkeys = 'ru/ge pd/ge ag/ge cd/ge'.split()
        r_rkeys = 'ge/ru ge/pd ge/ag ge/cd'.split()

        assert keylist / keys == rkeys
        assert keys / keylist == r_rkeys

        with pytest.raises(ValueError):
            keylist / ['ge']

        with pytest.raises(ValueError):
            ['ge'] / keylist

        with pytest.raises(ValueError):
            keylist / ['ge', 'as']

        with pytest.raises(ValueError):
            ['ge', 'as'] / keylist

    def test_count(self):
        keylist = isopy.keylist('ru pd ag cd pd'.split())

        assert keylist.count('pd') == 2
        assert keylist.count('ag') == 1

        assert keylist.count('ge') == 0
        assert keylist.count('102pd') == 0

    def test_index(self):
        keylist = isopy.keylist('ru pd ag cd pd'.split())

        assert keylist.index('pd') == 1
        assert keylist.index('ag') == 2

        with pytest.raises(ValueError):
            keylist.index('ge')

        with pytest.raises(ValueError):
            keylist.index('102pd')

    def test_attributes(self):
        keys = isopy.keylist('ru 105pd'.split())
        
        assert isopy.iskeylist(keys, flavour='element|isotope')
        assert keys.flavours == ('element', 'isotope')

        assert keys.mass_numbers is None

        element_symbols = keys.element_symbols
        assert type(element_symbols) is core.IsopyKeyList
        assert element_symbols == ('ru', 'pd')

        isotopes = keys.isotopes
        assert isopy.iskeylist(isotopes, flavour='isotope')
        assert isotopes == ('96Ru', '98Ru', '99Ru', '100Ru', '101Ru', '102Ru', '104Ru', '105Pd')

        assert keys.mz is None

        assert keys.numerators is None
        assert keys.denominators is None
        assert keys.common_denominator is None
        
        assert isotopes.mass_numbers == (96, 98, 99, 100, 101, 102, 104, 105)
        assert isotopes.mz == (96.0, 98.0, 99.0, 100.0, 101.0, 102.0, 104.0, 105.0)

        keys = isopy.keylist('ru/105pd 108pd/105pd'.split())

        assert isopy.iskeylist(keys, flavour='ratio[element, isotope]|ratio[isotope, isotope]')
        assert keys.flavours == ('ratio[element, isotope]', 'ratio[isotope, isotope]')

        assert keys.mass_numbers is None

        assert keys.element_symbols is None
        assert keys.isotopes is None
        assert keys.mz is None

        numerators = keys.numerators
        assert isopy.iskeylist(numerators, flavour='element|isotope')
        assert numerators == ('ru', '108pd')

        denominators = keys.denominators
        assert isopy.iskeylist(denominators, flavour='isotope')
        assert denominators == ('105pd', '105pd')

        assert keys.common_denominator == '105pd'


class Test_Dict:
    def test_creation(self):
        # IsopyDict
        for v in [1, 1.4, {1,2,3}, [1,2,3], (1,2,3)]:
            with pytest.raises(TypeError):
                isopydict = core.IsopyDict(v)

        for v in ['str', [[1,2,3], [4,5,6]]]:
            with pytest.raises(ValueError):
                isopydict = core.IsopyDict(v)

        keys = '101 pd ru cd 105pd hermione'.split()
        values = [1, '2', 'a', [11, 12, 13], 5, 6]

        dictionary1 = dict(zip(keys, values))
        dictionary2 = dict(zip(keys[:3], values[:3]))
        dictionary3 = dict(zip(keys[3:], values[3:]))

        isopydict = core.IsopyDict(dictionary1)
        assert type(isopydict) is core.IsopyDict
        self.check_creation(isopydict, keys, values)
        assert isopydict.keys== keys

        isopydict = core.IsopyDict(**dictionary1)
        assert type(isopydict) is core.IsopyDict
        self.check_creation(isopydict, keys, values)
        assert isopydict.keys == keys

        isopydict = core.IsopyDict(dictionary2, dictionary3)
        assert type(isopydict) is core.IsopyDict
        self.check_creation(isopydict, keys, values)
        assert isopydict.keys == keys

        isopydict = core.IsopyDict(dictionary2)
        isopydict.update(dictionary3)
        assert type(isopydict) is core.IsopyDict
        self.check_creation(isopydict, keys, values)
        assert isopydict.keys == keys

        isopydict = core.IsopyDict(dictionary2)
        with pytest.raises(TypeError):
            isopydict.update({1,2,3})

        isopydict = core.IsopyDict(dictionary2)
        assert type(isopydict) is core.IsopyDict
        isopydict.update(dictionary3)
        self.check_creation(isopydict, keys, values)

        isopydict = core.IsopyDict(dictionary2, **dictionary3)
        assert type(isopydict) is core.IsopyDict
        self.check_creation(isopydict, keys, values)
        assert isopydict.keys == keys

        isopydict = core.IsopyDict(**dictionary2, **dictionary3)
        assert type(isopydict) is core.IsopyDict
        self.check_creation(isopydict, keys, values)
        assert isopydict.keys== keys

        isopydict = core.IsopyDict(zip(keys, values))
        assert type(isopydict) is core.IsopyDict
        self.check_creation(isopydict, keys, values)
        assert isopydict.keys== keys

        with pytest.raises(TypeError):
            # This doesnt work for isopy dicts at present
            isopydict = core.IsopyDict(values, keys)

        keys = '101 pd ru cd 105pd hermione'.split()
        values = [1, 2, 3, 4, 5, 6]
        array = isopy.array(values, keys)
        refval = isopy.RefValDict(dict(zip(keys, values)))
        df = pandas.DataFrame(dict(zip(keys, values)), index=[0])

        isopydict = core.IsopyDict(array)
        assert type(isopydict) is core.IsopyDict
        self.check_creation(isopydict, keys, values)

        isopydict = core.IsopyDict(refval)
        assert type(isopydict) is core.IsopyDict
        self.check_creation(isopydict, keys, values)

        isopydict = core.IsopyDict(df)
        assert type(isopydict) is core.IsopyDict
        self.check_creation(isopydict, keys, [[v] for v in values])

    def test_creation_refval(self):
        # Scalar dict

        keys = '101 pd ru cd 105pd hermione'.split()
        values = [1, 2, 3, 4, 5, 6]

        dictionary1 = dict(zip(keys, values))
        dictionary2 = dict(zip(keys[:3], values[:3]))
        dictionary3 = dict(zip(keys[3:], values[3:]))
        dictionary4 = isopy.asdict(dict(zip(keys, values)))
        df = pandas.DataFrame(dict(zip(keys, values)), index=[0])

        isopydict = core.RefValDict(dictionary1)
        assert type(isopydict) is core.RefValDict
        self.check_creation_refval(isopydict, keys, values, 0, 1)

        isopydict = core.RefValDict(dictionary1, default_value=1)
        assert type(isopydict) is core.RefValDict
        self.check_creation_refval(isopydict, keys, values, 0, 1, 1)

        isopydict = core.RefValDict(dictionary1, default_value=[1, 2])
        assert type(isopydict) is core.RefValDict
        self.check_creation_refval(isopydict, keys, values, 1, 2, [1, 2])

        isopydict = core.RefValDict(**dictionary1)
        assert type(isopydict) is core.RefValDict
        self.check_creation_refval(isopydict, keys, values, 0, 1)

        isopydict = core.RefValDict(dictionary2, dictionary3)
        assert type(isopydict) is core.RefValDict
        self.check_creation_refval(isopydict, keys, values, 0, 1)

        isopydict = core.RefValDict(dictionary2)
        assert type(isopydict) is core.RefValDict
        isopydict.update(dictionary3)
        self.check_creation_refval(isopydict, keys, values, 0, 1)

        isopydict = core.RefValDict(dictionary2, **dictionary3)
        assert type(isopydict) is core.RefValDict
        self.check_creation_refval(isopydict, keys, values, 0, 1)

        isopydict = core.RefValDict(**dictionary2, **dictionary3)
        assert type(isopydict) is core.RefValDict
        self.check_creation_refval(isopydict, keys, values, 0, 1)

        isopydict = core.RefValDict(zip(keys, values))
        assert type(isopydict) is core.RefValDict
        self.check_creation_refval(isopydict, keys, values, 0, 1)

        isopydict = core.RefValDict(dictionary4)
        assert type(isopydict) is core.RefValDict
        self.check_creation_refval(isopydict, keys, values, 0, 1)

        isopydict = core.RefValDict(df)
        assert type(isopydict) is core.RefValDict
        self.check_creation_refval(isopydict, keys, values, 0, 1)

        with pytest.raises(TypeError):
            # This doesnt work for isopy dicts at present
            isopydict = core.RefValDict(values, keys)

        keys = '101 pd ru cd 105pd hermione'.split()
        values = [[1], [2], [3], [4], [5], [6]]

        dictionary1 = dict(zip(keys, values))
        dictionary2 = dict(zip(keys[:3], values[:3]))
        dictionary3 = dict(zip(keys[3:], values[3:]))

        isopydict = core.RefValDict(dictionary1)
        assert type(isopydict) is core.RefValDict
        self.check_creation_refval(isopydict, keys, values, 0, 1)

        isopydict = core.RefValDict(dictionary1, default_value=1)
        assert type(isopydict) is core.RefValDict
        self.check_creation_refval(isopydict, keys, values, 0, 1, 1)

        isopydict = core.RefValDict(dictionary1, default_value=[1, 2])
        assert type(isopydict) is core.RefValDict
        self.check_creation_refval(isopydict, keys, values, 1, 2, [1, 2])

        isopydict = core.RefValDict(**dictionary1)
        assert type(isopydict) is core.RefValDict
        self.check_creation_refval(isopydict, keys, values, 0, 1)

        isopydict = core.RefValDict(dictionary2, dictionary3)
        assert type(isopydict) is core.RefValDict
        self.check_creation_refval(isopydict, keys, values, 0, 1)

        isopydict = core.RefValDict(dictionary2)
        assert type(isopydict) is core.RefValDict
        isopydict.update(dictionary3)
        self.check_creation_refval(isopydict, keys, values, 0, 1)

        isopydict = core.RefValDict(dictionary2, **dictionary3)
        assert type(isopydict) is core.RefValDict
        self.check_creation_refval(isopydict, keys, values, 0, 1)

        isopydict = core.RefValDict(**dictionary2, **dictionary3)
        assert type(isopydict) is core.RefValDict
        self.check_creation_refval(isopydict, keys, values, 0, 1)

        isopydict = core.RefValDict(zip(keys, values))
        assert type(isopydict) is core.RefValDict
        self.check_creation_refval(isopydict, keys, values, 0, 1)

        keys = '101 pd ru cd 105pd hermione'.split()
        values = [[1], 2, [3], [4,5], [5,7], 6]

        dictionary1 = dict(zip(keys, values))
        dictionary2 = dict(zip(keys[:3], values[:3]))
        dictionary3 = dict(zip(keys[3:], values[3:]))

        isopydict = core.RefValDict(dictionary1)
        assert type(isopydict) is core.RefValDict
        self.check_creation_refval(isopydict, keys, values, 1, 2)

        isopydict = core.RefValDict(dictionary1, default_value=1)
        assert type(isopydict) is core.RefValDict
        self.check_creation_refval(isopydict, keys, values, 1, 2, 1)

        isopydict = core.RefValDict(dictionary1, default_value=[1, 2])
        assert type(isopydict) is core.RefValDict
        self.check_creation_refval(isopydict, keys, values, 1, 2, [1, 2])

        isopydict = core.RefValDict(**dictionary1)
        assert type(isopydict) is core.RefValDict
        self.check_creation_refval(isopydict, keys, values, 1, 2)

        isopydict = core.RefValDict(dictionary2, dictionary3)
        assert type(isopydict) is core.RefValDict
        self.check_creation_refval(isopydict, keys, values, 1, 2)

        isopydict = core.RefValDict(dictionary2)
        assert type(isopydict) is core.RefValDict
        isopydict.update(dictionary3)
        self.check_creation_refval(isopydict, keys, values, 1, 2)

        isopydict = core.RefValDict(dictionary2, **dictionary3)
        assert type(isopydict) is core.RefValDict
        self.check_creation_refval(isopydict, keys, values, 1, 2)

        isopydict = core.RefValDict(**dictionary2, **dictionary3)
        assert type(isopydict) is core.RefValDict
        self.check_creation_refval(isopydict, keys, values, 1, 2)

        isopydict = core.RefValDict(zip(keys, values))
        assert type(isopydict) is core.RefValDict
        self.check_creation_refval(isopydict, keys, values, 1, 2)

        keys = '101 pd ru cd 105pd hermione'.split()
        values = [[1, 2], 2, [3], [4, 5, 7], [5], 6]

        dictionary1 = dict(zip(keys, values))
        dictionary2 = dict(zip(keys[:3], values[:3]))
        dictionary3 = dict(zip(keys[3:], values[3:]))

        with pytest.raises(ValueError):
            core.RefValDict(dictionary1)

        with pytest.raises(ValueError):
            core.RefValDict(**dictionary1)

        with pytest.raises(ValueError):
            core.RefValDict(dictionary2, dictionary3)

        isopydict = core.RefValDict(dictionary2)
        with pytest.raises(ValueError):
            isopydict.update(dictionary3)

        with pytest.raises(ValueError):
            core.RefValDict(dictionary2, **dictionary3)

        with pytest.raises(ValueError):
            core.RefValDict(**dictionary2, **dictionary3)

        with pytest.raises(ValueError):
            core.RefValDict(zip(keys, values))

        keys = '101 pd ru cd 105pd hermione'.split()
        values = [[1, 2], 2, [3], [], [5], 6]

        dictionary1 = dict(zip(keys, values))
        dictionary2 = dict(zip(keys[:3], values[:3]))
        dictionary3 = dict(zip(keys[3:], values[3:]))

        with pytest.raises(ValueError):
            core.RefValDict(dictionary1)

        with pytest.raises(ValueError):
            core.RefValDict(**dictionary1)

        with pytest.raises(ValueError):
            core.RefValDict(dictionary2, dictionary3)

        isopydict = core.RefValDict(dictionary2)
        with pytest.raises(ValueError):
            isopydict.update(dictionary3)

        with pytest.raises(ValueError):
            core.RefValDict(dictionary2, **dictionary3)

        with pytest.raises(ValueError):
            core.RefValDict(**dictionary2, **dictionary3)

        with pytest.raises(ValueError):
            core.RefValDict(zip(keys, values))

        keys = '101 pd ru cd 105pd hermione'.split()
        values = [[1, 2], 2, [3], [[1], [2]], [5], 6]

        dictionary1 = dict(zip(keys, values))
        dictionary2 = dict(zip(keys[:3], values[:3]))
        dictionary3 = dict(zip(keys[3:], values[3:]))

        with pytest.raises(ValueError):
            core.RefValDict(dictionary1)

        with pytest.raises(ValueError):
            core.RefValDict(**dictionary1)

        with pytest.raises(ValueError):
            core.RefValDict(dictionary2, dictionary3)

        isopydict = core.RefValDict(dictionary2)
        with pytest.raises(ValueError):
            isopydict.update(dictionary3)

        with pytest.raises(ValueError):
            core.RefValDict(dictionary2, **dictionary3)

        with pytest.raises(ValueError):
            core.RefValDict(**dictionary2, **dictionary3)

        with pytest.raises(ValueError):
            core.RefValDict(zip(keys, values))

    def check_creation(self, isopydict, keys, values):
        keylist = isopy.keylist(keys)
        assert len(isopydict) == len(keys)

        for i, (key, value) in enumerate(isopydict.items()):
            assert type(key) == type(keylist[i])
            assert key == keylist[i]
            assert value == values[i]

        for i, key in enumerate(isopydict.keys()):
            assert type(key) == type(keylist[i])
            assert key == keylist[i]

        for i, value in enumerate(isopydict.values()):
            assert value == values[i]

        for i, key in enumerate(keys):
            assert key in isopydict
            assert isopydict[key] == values[i]

    def check_creation_refval(self, isopydict, keys, values, ndim, size, default_value=np.nan):
        keylist = isopy.keylist(keys)
        assert len(isopydict) == len(keys)
        assert isopydict.default_value.ndim == ndim
        assert isopydict.default_value.size == size
        assert isopydict.ndim == ndim
        assert isopydict.size == size
        np.testing.assert_allclose(isopydict.default_value, np.full(size, default_value))

        for i, (key, value) in enumerate(isopydict.items()):
            assert key in keylist
            assert value.dtype == np.float64
            assert value.ndim == ndim
            assert value.size == size

            np.testing.assert_allclose(value, np.full(size, values[keylist.index(key)]))

        for i, value in enumerate(isopydict.values()):
            key = isopydict.keys[i]
            assert value.dtype == np.float64
            assert value.ndim == ndim
            assert value.size == size

            np.testing.assert_allclose(value, np.full(size, values[keylist.index(key)]))

        for i, key in enumerate(isopydict.keys()):
            assert key in keylist

        for i, key in enumerate(keys):
            assert key in isopydict
            assert key in isopydict.keys

    def test_repr(self):
        keys = '101 pd ru cd 105pd hermione'.split()
        values = [1, '2', 'a', [11, 12, 13], 5, 6]

        isopydict = core.IsopyDict(dict(zip(keys, values)))

        part1, part2 = repr(isopydict).split('\n', 1)
        assert part1 == "IsopyDict(readonly=False, key_flavour='any')"
        assert part2 == repr({key:value for key, value in isopydict.items()})

        part1, part2 = str(isopydict).split('\n', 1)
        assert part1 == "IsopyDict(readonly=False, key_flavour='any')"
        assert part2 == repr({key:value for key, value in isopydict.items()})

        keys = '101 pd ru cd 105pd hermione'.split()
        values = [1, 2, 3, 4, 5, 6]

        isopydict = core.IsopyDict(dict(zip(keys, values)), default_value=1)

        part1, part2 = repr(isopydict).split('\n', 1)
        assert part1 == "IsopyDict(readonly=False, key_flavour='any', default_value=1)"

        part1, part2 = str(isopydict).split('\n', 1)
        assert part1 == "IsopyDict(readonly=False, key_flavour='any', default_value=1)"

        # RefValDict repr is tested in ToMixing test

    # Tests most methods
    def test_methods(self):
        keys = '101 pd ru cd 105pd hermione'.split()
        values = [1, '2', 'a', [11, 12, 13], 5, 6]

        dictionary1 = dict(zip(keys, values))

        # readonly = False

        isopydict = core.IsopyDict(dictionary1)
        self.check_creation(isopydict, keys, values)

        assert isopydict.readonly is False

        # set
        isopydict['Palladium'] = 666
        assert isopydict['pd'] == 666
        isopydict['107ag'] = 107
        assert isopydict['107ag'] == 107

        # del
        del(isopydict['palladium'])
        assert 'pd' not in isopydict

        # pop
        value = isopydict.pop('107ag')
        assert value == 107
        assert '107ag' not in isopydict
        value = isopydict.pop('107ag', None)
        assert value is None
        with pytest.raises(ValueError):
            value = isopydict.pop('107ag')

        assert isopydict.pop(None, 0) == 0

        # set default
        value = isopydict.setdefault('ru')
        assert value == 'a'
        value = isopydict.setdefault('107ag', None)
        assert value is None
        assert '107ag' in isopydict
        assert isopydict['107ag'] == None
        with pytest.raises(ValueError):
            value = isopydict.setdefault('103rh')

        #update
        isopydict.update({'pd': 222, '107ag': 107})
        assert 'pd' in isopydict
        assert '107ag' in isopydict
        assert isopydict['pd'] == 222
        assert isopydict['107ag'] == 107

        # copy
        copy = isopydict.copy()
        self.check_creation(copy, list(isopydict.keys()), list(isopydict.values()))
        assert copy is not isopydict
        assert copy.readonly is False

        # clear
        isopydict.clear()
        assert len(isopydict) == 0

        with pytest.raises(AttributeError):
            isopydict.readonly = True

        # with default Value
        isopydict = core.IsopyDict(dictionary1, default_value='default')
        assert isopydict.default_value == 'default'

        value = isopydict.pop('107ag')
        assert value == 'default'
        value = isopydict.pop('107ag', None)
        assert value == None

        value = isopydict.setdefault('103rh')
        assert value == 'default'
        assert '103rh' in isopydict
        assert isopydict['103rh'] == 'default'

        value = isopydict.setdefault('109ag', None)
        assert value is None
        assert '109ag' in isopydict
        assert isopydict['109ag'] is None

        copy = isopydict.copy()
        assert copy is not isopydict
        assert copy.default_value == 'default'

        isopydict.default_value = 'fail'
        assert isopydict.default_value == 'fail'
        assert copy.default_value == 'default'

        # readonly = True

        isopydict = core.IsopyDict(dictionary1, readonly=True)
        self.check_creation(isopydict, keys, values)

        assert isopydict.readonly is True

        # set
        with pytest.raises(TypeError):
            isopydict['Palladium'] = 666
        with pytest.raises(TypeError):
            isopydict['107ag'] = 107

        # del
        with pytest.raises(TypeError):
            del (isopydict['palladium'])

        # pop
        with pytest.raises(TypeError):
            value = isopydict.pop('107ag')
        with pytest.raises(TypeError):
            value = isopydict.pop('107ag', None)
        with pytest.raises(TypeError):
            value = isopydict.pop('107ag')

        # set default
        with pytest.raises(TypeError):
            value = isopydict.setdefault('ru')
        with pytest.raises(TypeError):
            value = isopydict.setdefault('107ag', None)
        with pytest.raises(TypeError):
            value = isopydict.setdefault('107ag')

        # update
        with pytest.raises(TypeError):
            isopydict.update({'pd': 222, '107ag': 107})

        # copy
        copy = isopydict.copy()
        self.check_creation(copy, list(isopydict.keys()), list(isopydict.values()))
        assert copy is not isopydict
        assert copy.readonly is False

        # clear
        with pytest.raises(TypeError):
            isopydict.clear()

        with pytest.raises(AttributeError):
            isopydict.readonly = False

        # ScalarDict with default Value
        keys = '101 pd ru cd 105pd hermione'.split()
        values = [1, 2, 3, 4, 5, 6]
        dictionary1 = dict(zip(keys, values))

        isopydict = core.RefValDict(dictionary1, default_value=666)
        assert isopydict.default_value == 666
        assert isopydict.default_value.dtype == np.float64

        with pytest.raises(ValueError):
            isopydict['107ag'] = 'a'

        copy = isopydict.copy()
        assert copy is not isopydict
        assert copy.default_value == 666

        with pytest.raises(ValueError):
            isopydict.default_value = 'fail'

        isopydict.default_value = 1
        assert isopydict.default_value == 1

        with pytest.raises(ValueError):
            isopydict = core.RefValDict(dictionary1, default_value='a')

    def test_get_isopydict(self):
        keys = '101 pd ru cd 105pd hermione'.split()
        values = [1, '2', 'a', [11, 12, 13], 5, 6]

        dictionary = dict(zip(keys, values))
        isopydict1 = core.IsopyDict(dictionary)
        isopydict2 = core.IsopyDict(dictionary, default_value = 'default')

        for key in keys:
            assert isopydict1.get(key) == dictionary.get(key)
            assert isopydict2.get(key) == dictionary.get(key)
        assert isopydict1.get(101) == dictionary.get('101')
        assert isopydict2.get(101) == dictionary.get('101')
        for key in '103rh 107ag'.split():
            with pytest.raises(ValueError):
                isopydict1.get(key)
            assert isopydict2.get(key) == 'default'

            assert isopydict1.get(key, None) is None
            assert isopydict2.get(key, None) is None

        with pytest.raises(TypeError):
            isopydict1.get(3.14)

        # Test ratio
        with pytest.raises(ValueError):
            isopydict1.get('105pd/ru')
        assert isopydict2.get('105pd/ru') == 'default'
        assert isopydict1.get('105pd/ru', None) is None
        assert isopydict2.get('105pd/ru', None) is None

        # Test with charge
        keys = 'ru++ cd- 101ru+ 105pd++ 108pd- 111cd--'.split()
        basekeys = 'ru cd 101ru 105pd 108pd 111cd'.split()
        values = [1, 2, 3, 4, 5, 6]
        dictionary1 = dict(zip(keys, values))
        dictionary2 = dict(zip(basekeys, values))

        isopydict1 = core.IsopyDict(dictionary1)
        isopydict2 = core.IsopyDict(dictionary2)
        for key in keys:
            assert isopy.keystring(key).charge is not None

            assert key in isopydict1
            assert key not in isopydict2

            assert isopydict1.get(key, 666) == dictionary1[key]
        assert isopydict1.get('137Ba++', 666) == 666
        assert isopydict2.get('137Ba++', 666) == 666

        for key in basekeys:
            assert key not in isopydict1
            assert key in isopydict2

            assert isopydict1.get(key, 666) == 666
        assert isopydict1.get('137Ba', 666) == 666
        assert isopydict2.get('137Ba', 666) == 666

        a = core.IsopyDict(dict(ru=1, pd=2, cd=3), key_flavour='element')

        np.testing.assert_equal(a.get('101ru', 10), 10)

        b = isopy.array(dict(rh=10, pd=20, cd=30))
        np.testing.assert_array_equal(a.get('rh', b), b['rh'])
        np.testing.assert_array_equal(a.get('pd', b), a['pd'])

        b = b.to_refval(ratio_function='divide')

        np.testing.assert_array_equal(a.get('rh', b), b['rh'])
        np.testing.assert_array_equal(a.get('pd', b), a['pd'])
        np.testing.assert_array_equal(a.get('rh/pd', b), b['rh'] / b['pd'])

    def test_get_refval(self):
        keys = '101 pd ru cd 105pd hermione'.split()
        values = [1, 2, 3, 4, 5, 6]

        dictionary = dict(zip(keys, values))
        isopydict1 = core.RefValDict(dictionary)
        isopydict2 = core.RefValDict(dictionary, default_value=10)

        for key in keys:
            assert isopydict1[key] == dictionary.get(key)
            assert isopydict2[key] == dictionary.get(key)
            assert isopydict1[key] is isopydict1[key]

            assert isopydict1.get(key) == dictionary.get(key)
            assert isopydict2.get(key) == dictionary.get(key)
            assert isopydict1.get(key) is isopydict1.get(key)
            assert isopydict2.get(key) is isopydict2.get(key)

            assert isopydict1.get(key, [1, 2]) == dictionary.get(key)

        with pytest.raises(ValueError):
            isopydict1.get('ge76', [1, 2])

        with pytest.raises(ValueError):
            isopydict1.get('ge76', 'a')

        assert isopydict1.get(101) == dictionary.get('101')
        assert isopydict2.get(101) == dictionary.get('101')
        for key in '103rh 107ag'.split():
            assert np.isnan(isopydict1.get(key))
            assert isopydict2.get(key) == 10

            assert isopydict1.get(key) is isopydict1.default_value
            assert isopydict2.get(key) is isopydict2.default_value

            value1 = isopydict1.get(key, 666)
            value2 = isopydict2.get(key, 666)

            assert value1 == 666
            assert value2 == 666
            assert value1.dtype == np.float64
            assert value2.dtype == np.float64

            with pytest.raises(ValueError):
                isopydict1.get(key, [1, 2])

        a = core.RefValDict(dict(ru=1, pd=2, cd=3), key_flavour='element')
        assert type(a) is core.RefValDict

        np.testing.assert_equal(a.get('101ru', 10), 10)

        get = a.get('rh', 10)
        assert get.dtype == np.float64

        get = a.get('rh', np.array(10, dtype=np.int8))
        assert get.dtype == np.float64

        b = isopy.array(dict(rh=10, pd=20, cd=30))
        np.testing.assert_array_equal(a.get('rh', b), b['rh'])
        np.testing.assert_array_equal(a.get('pd', b), a['pd'])

        b = b.to_refval(ratio_function='divide')

        np.testing.assert_array_equal(a.get('rh', b), b['rh'])
        np.testing.assert_array_equal(a.get('pd', b), a['pd'])
        np.testing.assert_array_equal(a.get('rh/pd', b), b['rh'] / b['pd'])

    def test_getitem_refval(self):
        d = isopy.asrefval(dict(ru=[1,2,3,4], pd=[11, 12,13,14], cd=[21,22,23,24]))

        d2 = d[:]
        assert d is not d2
        assert d.keys == d2.keys
        for key in d2.keys:
            np.array_equal(d[key], d2[key])

        d2 = d[:2]
        assert d is not d2
        assert d.keys == d2.keys
        for key in d2.keys:
            np.array_equal(d[key][:2], d2[key])

        d2 = d[1]
        assert d is not d2
        assert d.keys == d2.keys
        for key in d2.keys:
            np.array_equal(d[key][1], d2[key])

        # List of indexes
        d2 = d[[1,3]]
        assert d is not d2
        assert d.keys == d2.keys
        for key in d2.keys:
            np.array_equal(d[key][[1,3]], d2[key])

        with pytest.raises(IndexError):
            d[(1, 3)]

        with pytest.raises(IndexError):
            d[1, 3]

        #list of key strings
        d2 = d[['ru', 'ag', 'cd']]
        assert d is not d2
        assert d2.keys == ['ru', 'cd']
        for key in d2.keys:
            np.array_equal(d[key], d2[key])

        d2 = d[('ru', 'ag', 'cd')]
        assert d is not d2
        assert d2.keys == ['ru', 'cd']
        for key in d2.keys:
            np.array_equal(d[key], d2[key])

        d2 = d['ru', 'ag', 'cd']
        assert d is not d2
        assert d2.keys == ['ru', 'cd']
        for key in d2.keys:
            np.array_equal(d[key], d2[key])

        d2 = d[['ag', 'rh', 'te']]
        assert d is not d2
        assert len(d2.keys) == 0
        assert d2.size == 4

        d2 = d[[]]
        assert d is not d2
        assert len(d2.keys) == 0
        assert d2.size == 4

        # Size becomes 0 which gives an error
        with pytest.raises(ValueError):
            d[5:]
        
        #Value not in dict
        with pytest.raises(KeyError):
            d[3.14]

    def test_get_refval_ratio(self):
        keys = '101 pd ru cd 105pd hermione'.split()
        values = [1, 2, 3, 4, 5, 6]

        dictionary = dict(zip(keys, values))
        isopydict1 = core.RefValDict(dictionary)
        isopydict2 = core.RefValDict(dictionary, default_value=10)

        assert np.isnan(isopydict1.get('105pd/ru'))
        assert isopydict2.get('105pd/ru') == 10

        isopydict1 = core.RefValDict(dictionary, ratio_function=np.divide)
        isopydict2 = core.RefValDict(dictionary, default_value=10, ratio_function=np.divide)

        assert isopydict1.get('105pd/ru') == 5 / 3
        assert isopydict2.get('105pd/ru') == 5/3
        assert isopydict1.get('105pd/ru', 666) == 5/3
        assert isopydict2.get('105pd/ru', 666) == 5/3
        assert isopydict1.get('105pd/ru') is not isopydict1.get('105pd/ru')
        assert isopydict2.get('105pd/ru') is not isopydict2.get('105pd/ru')
        assert isopydict1.get('105pd/ru', 666) is not isopydict1.get('105pd/ru', 666)
        assert isopydict2.get('105pd/ru', 666) is not isopydict2.get('105pd/ru', 666)

        # Nested list do not work
        assert isopydict1.get('105pd/ru//hermione') == (5 / 3) / 6
        assert isopydict2.get('105pd/ru//hermione') == (5 / 3) / 6
        assert isopydict1.get('105pd/ru//hermione', 666) == (5 / 3) / 6
        assert isopydict2.get('105pd/ru//hermione', 666) == (5 / 3) / 6

        assert np.isnan(isopydict1.get('105pd/ru//harry'))
        assert isopydict2.get('105pd/ru//harry') == (5 / 3) / 10
        assert isopydict1.get('105pd/ru//harry', 666) == (5 / 3) / 666
        assert isopydict2.get('105pd/ru//harry', 666) == (5 / 3) / 666

        assert np.isnan(isopydict1.get('105pd/ag//hermione'))
        assert isopydict2.get('105pd/ag//hermione') == (5 / 10) / 6
        assert isopydict1.get('105pd/ag//hermione', 666) == (5 / 666) / 6
        assert isopydict2.get('105pd/ag//hermione', 666) == (5 / 666) / 6

        assert np.isnan(isopydict1.get('105pd/ag'))
        assert isopydict2.get('105pd/ag') == 5 / 10
        assert isopydict1.get('105pd/ag', 666) == 5 / 666
        assert isopydict2.get('105pd/ag', 666) == 5/ 666

        assert np.isnan(isopydict1.get('107ag/ru'))
        assert isopydict2.get('107ag/ru') == 10 / 3
        assert isopydict1.get('107ag/ru', 666) == 666 / 3
        assert isopydict2.get('107ag/ru', 666) == 666 / 3

    def test_get_refval_molecule(self):
        keys = 'ru rh pd ag cd'.split()
        values = (1,2 ,3, 4, 5)

        scalardict1 = core.RefValDict(zip(keys, values))
        scalardict2 = core.RefValDict(zip(keys, values), default_value=10)

        assert np.isnan(scalardict1.get('RuPd'))
        assert scalardict2.get('RuPd') == 10
        assert np.isnan(scalardict1.get('GePd'))
        assert scalardict2.get('GePd') == 10

        assert np.isnan(scalardict1.get('(Pd)2'))
        assert scalardict2.get('(Pd)2') == 10
        assert np.isnan(scalardict1.get('(Ge)2'))
        assert scalardict2.get('(Ge)2') == 10

        assert np.isnan(scalardict1.get('Pd++'))
        assert scalardict2.get('Pd++') == 10
        assert np.isnan(scalardict1.get('Ge++'))
        assert scalardict2.get('Ge++') == 10

        scalardict1 = core.RefValDict(zip(keys, values), molecule_functions=(np.add, np.multiply, None))
        scalardict2 = core.RefValDict(zip(keys, values), default_value=10, molecule_functions=(np.add, np.multiply, None))

        assert scalardict1.get('RuPd') == 4
        assert scalardict2.get('RuPd') == 4
        assert np.isnan(scalardict1.get('GePd'))
        assert scalardict2.get('GePd') == 13

        assert scalardict1.get('(Pd)2') == 6
        assert scalardict2.get('(Pd)2') == 6
        assert np.isnan(scalardict1.get('(Ge)2'))
        assert scalardict2.get('(Ge)2') == 20

        assert scalardict1.get('Pd++') == 3
        assert scalardict2.get('Pd++') == 3
        assert np.isnan(scalardict1.get('Ge++'))
        assert scalardict2.get('Ge++') == 10

        scalardict1 = core.RefValDict(zip(keys, values), molecule_functions=(np.add, np.multiply, np.divide))
        scalardict2 = core.RefValDict(zip(keys, values), default_value=10, molecule_functions=(np.add, np.multiply, np.divide))

        assert scalardict1.get('Pd++') == 1.5
        assert scalardict2.get('Pd++') == 1.5
        assert np.isnan(scalardict1.get('Ge++'))
        assert scalardict2.get('Ge++') == 5

        assert scalardict1.get('Pd--') == -1.5
        assert scalardict2.get('Pd--') == -1.5
        assert np.isnan(scalardict1.get('Ge--'))
        assert scalardict2.get('Ge--') == -5

        assert scalardict1.get('Pd+') == 3
        assert scalardict2.get('Pd+') == 3
        assert np.isnan(scalardict1.get('Ge+'))
        assert scalardict2.get('Ge+') == 10

        assert scalardict1.get('Pd-') == -3
        assert scalardict2.get('Pd-') == -3
        assert np.isnan(scalardict1.get('Ge-'))
        assert scalardict2.get('Ge-') == -10

    def test_refval_default_value(self):
        a = isopy.random(None, (1, 0.1), 'ru pd cd'.split(), seed=46)

        d = a.to_refval()
        assert type(d) is core.RefValDict
        np.testing.assert_allclose(d.default_value, np.nan)

        d = a.to_refval(default_value = 1)
        assert type(d) is core.RefValDict
        np.testing.assert_allclose(d.default_value, 1)

        d = a.default(2).to_refval()
        assert type(d) is core.RefValDict
        np.testing.assert_allclose(d.default_value, 2)

    def test_asdict(self):
        keys = '101 pd ru cd 105pd hermione'.split()
        values = [1, '2', 'a', [11, 12, 13], 5, 6]

        input = zip(keys, values)
        result = isopy.asdict(input)
        assert type(result) is core.IsopyDict
        self.check_creation(result, keys, values)

        input = dict(zip(keys, values))
        result = isopy.asdict(input)
        assert type(result) is core.IsopyDict
        self.check_creation(result, keys, values)

        input = core.IsopyDict(zip(keys, values))
        result = isopy.asdict(input)
        assert result is input

        result = isopy.asdict(input, 1)
        assert result.default_value == 1
        assert result is not input
        assert type(result) is core.IsopyDict
        self.check_creation(result, keys, values)

        input = result
        result = isopy.asdict(input, 1)
        assert result is input

        input = 'element.symbol_name'
        result = isopy.asdict(input)
        assert result is isopy.refval.element.symbol_name

        keys = list(isopy.refval.element.symbol_name.keys())
        values = list(isopy.refval.element.symbol_name.values())
        result = isopy.asdict(input, 1)
        assert result.default_value == 1
        assert result is not input
        assert type(result) is core.IsopyDict
        self.check_creation(result, keys, values)

        keys = '101 pd ru cd 105pd hermione'.split()
        values = [1, 2, 3, 4, 5, 6]

        input = core.RefValDict(zip(keys, values))
        result = isopy.asdict(input)
        assert result is input

    def test_asrefval(self):
        keys = '101 pd ru cd 105pd hermione'.split()
        values = [1, 2, 3, 4, 5, 6]

        input = zip(keys, values)
        result = isopy.asrefval(input)
        assert type(result) is core.RefValDict
        self.check_creation_refval(result, keys, values, 0, 1)

        input = dict(zip(keys, values))
        result = isopy.asrefval(input)
        assert type(result) is core.RefValDict
        self.check_creation_refval(result, keys, values, 0, 1)

        input = core.IsopyDict(zip(keys, values))
        result = isopy.asrefval(input, 1)
        assert result is not input
        assert type(result) is core.RefValDict
        self.check_creation_refval(result, keys, values, 0, 1, 1)

        input = core.RefValDict(zip(keys, values))
        result = isopy.asrefval(input)
        assert result is input

        result = isopy.asrefval(input, 1)
        assert result is not input
        assert type(result) is core.RefValDict
        self.check_creation_refval(result, keys, values, 0, 1, 1)

        input = result
        result = isopy.asrefval(input, 1)
        assert result is input

        input = 'isotope.fraction'
        result = isopy.asrefval(input)
        assert result is isopy.refval.isotope.fraction

        keys = list(isopy.refval.isotope.fraction.keys())
        values = list(isopy.refval.isotope.fraction.values())
        result = isopy.asrefval(input, 1)
        assert result is not input
        assert type(result) is core.RefValDict
        self.check_creation_refval(result, keys, values, 0, 1, 1)

    def test_ratio_function(self):
        d = isopy.asrefval( dict(ru=1, pd=2, cd=3) )

        assert d.ratio_function is None
        assert np.isnan(d.get('ru/pd'))

        d.ratio_function = 'divide'
        assert d.ratio_function == np.divide
        assert d.get('ru/pd') == 0.5

        with pytest.raises(ValueError):
            d.ratio_function = 'add'

        d.ratio_function = np.add
        assert d.ratio_function == np.add
        assert d.get('ru/pd') == 3

        d.ratio_function = None
        assert d.ratio_function is None
        assert np.isnan(d.get('ru/pd'))

    def test_molecule_functions(self):
        d = isopy.asrefval(dict(h=3, o=4))

        assert d.molecule_functions is None
        assert np.isnan(d.get('(H2O)++'))

        d.molecule_functions = 'mass'
        #assert d.molecule_functions == (np.add, np.multiply, lambda v, c: np.multiply(v, np.abs(c)))
        # lambdas dont evaluate with ==
        assert d.get('(H2O)++') == (((3 * 2) + 4) * 1) / 2
        assert d.get('(H2O)--') == (((3 * 2) + 4) * 1) / 2

        d.molecule_functions = 'abundance'
        assert d.molecule_functions == (np.add, np.multiply, None)
        assert d.get('(H2O)++') == (((3 * 2) + 4) * 1)

        d.molecule_functions = 'fraction'
        assert d.molecule_functions == (np.multiply, np.multiply, None)
        assert d.get('(H2O)++') == (((3 * 2) * 4) * 1)

        with pytest.raises(ValueError):
            d.molecule_functions = 'unknown'

        with pytest.raises(ValueError):
            d.molecule_functions = (np.add)

        with pytest.raises(ValueError):
            d.molecule_functions = (np.add, np.add)

        d.molecule_functions = (np.add, np.add, np.add)
        assert d.molecule_functions == (np.add, np.add, np.add)
        assert d.get('(H2O)++') == (((3 + 2) + 4) + 1) + 2

        d.molecule_functions = (np.add, np.add, None)
        assert d.molecule_functions == (np.add, np.add, None)
        assert d.get('(H2O)++') == (((3 + 2) + 4) + 1)
        
        with pytest.raises(ValueError):
            d.molecule_functions = (np.add, np.add, 1)
            
        with pytest.raises(ValueError):
            d.molecule_functions = (np.add, None, np.add)

        with pytest.raises(ValueError):
            d.molecule_functions = (None, np.add, np.add)

    def test_isdict(self):
        a = isopy.asdict({})
        b = isopy.asdict({}, default_value=1)
        c = isopy.asdict({}, key_flavour='element', default_value = None)
        d = isopy.asdict({}, default_value = np.nan)

        assert not  isopy.isdict('dict')
        assert isopy.isdict(a)

        assert not isopy.isdict(a, default_value=1)
        assert isopy.isdict(b, default_value=1)
        assert isopy.isdict(c, default_value=None)
        assert isopy.isdict(d, default_value=np.nan)

        assert isopy.isdict(a, key_flavour='any')
        assert not isopy.isdict(a, key_flavour='element')
        assert isopy.isdict(c, key_flavour='element')

        a = isopy.asrefval({})
        b = isopy.asrefval({}, default_value=1)
        c = isopy.asrefval({}, key_flavour='element')

        assert not isopy.isdict('dict')
        assert isopy.isdict(a)

        assert not isopy.isdict(a, default_value=1)
        assert isopy.isdict(b, default_value=1)

        assert isopy.isdict(a, key_flavour='any')
        assert not isopy.isdict(a, key_flavour='element')
        assert isopy.isdict(c, key_flavour='element')

    def test_isrefval(self):
        array = isopy.ones(2, 'ru pd cd'.split())
        a = isopy.asdict(array)
        assert not isopy.isrefval(a)

        a = isopy.asrefval(array)
        b = isopy.asrefval(array, default_value=1)
        c = isopy.asrefval(array, key_flavour='element', ratio_function='divide')
        d = isopy.asrefval(array, default_value=[1,2], molecule_functions='abundance')

        assert isopy.isrefval(a)

        assert isopy.isrefval(a, key_flavour='any')
        assert isopy.isrefval(c, key_flavour='element')
        assert not isopy.isrefval(a, key_flavour='element')

        assert isopy.isrefval(a, default_value=np.nan)
        assert isopy.isrefval(b, default_value=1)
        assert isopy.isrefval(b, default_value=[1])
        assert isopy.isrefval(b, default_value=[1, 1])
        assert isopy.isrefval(d, default_value=[1, 2])
        assert not isopy.isrefval(b, default_value=np.nan)
        assert not isopy.isrefval(b, default_value=[1, 2])
        assert not isopy.isrefval(b, default_value='a')

        assert isopy.isrefval(a, ratio_function=None)
        assert not isopy.isrefval(c, ratio_function=None)
        assert isopy.isrefval(c, ratio_function='divide')
        assert isopy.isrefval(c, ratio_function=np.divide)

        assert isopy.isrefval(a, molecule_functions=None)
        assert not isopy.isrefval(d, molecule_functions=None)
        assert isopy.isrefval(d, molecule_functions='abundance')
        assert isopy.isrefval(d, molecule_functions=(np.add, np.multiply, None))

    def test_copy(self):
        d = isopy.asdict(dict(ru=[1, 2, 3, 4], pd=[11, 12, 13, 14], cd=[21, 22, 23, 24]),
                         default_value = 1)
        copy = d.copy()

        assert copy is not d
        np.testing.assert_array_equal(copy.default_value, d.default_value)
        assert copy.keys == d.keys
        for key in d.keys:
            assert copy[key] is d[key]

        d = isopy.asrefval(dict(ru=[1, 2, 3, 4], pd=[11, 12, 13, 14], cd=[21, 22, 23, 24]),
                           default_value = 1, ratio_function='divide', molecule_functions='mass')
        copy = d.copy()

        assert copy is not d
        np.testing.assert_array_equal(copy.default_value, d.default_value)
        assert copy.ratio_function == d.ratio_function
        assert copy.molecule_functions == d.molecule_functions
        assert copy.keys == d.keys
        for key in d.keys:
            assert copy[key] is not d[key]
            np.testing.assert_array_equal(copy[key], d[key])



#TODO test changing dtype
class Test_Array:
    def test_create_0dim_array(self):
        mass_keys = ('104', '105', '106', '107')
        element_keys = ('mo', 'ru', 'pd', 'rh')
        isotope_keys = ('104ru', '105Pd', '106Pd', 'cd111')
        general_keys = ('harry', 'ron', 'hermione', 'neville')
        ratio_keys = ('harry/104ru', 'ron/105pd', 'hermione/106Pd', 'neville/cd111')
        mixed_keys = ('104', 'mo', '104ru', 'neville/cd111')
        molecule_keys = ('H2O', '(OH2)', 'HCl', 'HNO3')

        all_keys = (mass_keys, element_keys, isotope_keys, general_keys, ratio_keys, mixed_keys,
                    general_keys, molecule_keys)

        # 0-dim input
        data_list = [1, 2, 3, 4]
        data_tuple = (1, 2, 3, 4)
        data_array = np.array(data_list, dtype=np.float_)
        keys2 = isotope_keys
        keylist2 = isopy.keylist(isotope_keys)
        data_correct2 = np.array(data_tuple, dtype=[(str(keystring), np.float_) for keystring in keylist2])

        for keys in all_keys:
            keylist = isopy.askeylist(keys)

            with pytest.raises(ValueError):
                isopy.array(data_list[1:], keys)

            with pytest.raises(ValueError):
                isopy.array(data_list, keys[1:])

            data_dict = dict(zip(keys, data_list))
            data_structured = np.array(data_tuple, dtype=[(str(key), np.float_) for key in keys])
            data_correct = np.array(data_tuple, dtype=[(str(keystring), np.float_) for keystring in keylist])

            self.create_array(data_correct, keys, 0, data_list, keys)
            self.create_array(data_correct, keys, 0, data_list, data_structured.dtype)
            self.create_array(data_correct, keys, 0, data_list, dtype=data_structured.dtype)
            self.create_array(data_correct, keys, 0, data_tuple, keys)
            self.create_array(data_correct, keys, 0, data_array, keys)
            self.create_array(data_correct, keys, 0, data_dict)
            self.create_array(data_correct, keys, 0, data_structured)

            self.create_array(data_correct, keys, 0, data_list, keys, ndim=-1)
            self.create_array(data_correct, keys, 0, data_list, data_structured.dtype, ndim=-1)
            self.create_array(data_correct, keys, 0, data_list, dtype=data_structured.dtype, ndim=-1)
            self.create_array(data_correct, keys, 0, data_tuple, keys, ndim=-1)
            self.create_array(data_correct, keys, 0, data_array, keys, ndim=-1)
            self.create_array(data_correct, keys, 0, data_dict, ndim=-1)
            self.create_array(data_correct, keys, 0, data_structured, ndim=-1)

            self.create_array(data_correct, keys, 0, data_list, keys, ndim=0)
            self.create_array(data_correct, keys, 0, data_list, data_structured.dtype, ndim=0)
            self.create_array(data_correct, keys, 0, data_list, dtype=data_structured.dtype, ndim=0)
            self.create_array(data_correct, keys, 0, data_tuple, keys, ndim=0)
            self.create_array(data_correct, keys, 0, data_array, keys, ndim=0)
            self.create_array(data_correct, keys, 0, data_dict, ndim=0)
            self.create_array(data_correct, keys, 0, data_structured, ndim=0)

            #Overwrite keys in data
            self.create_array(data_correct2, keys2, 0, data_dict, isotope_keys)
            self.create_array(data_correct2, keys2, 0, data_list, isotope_keys, dtype=data_structured.dtype, ndim=0)
            self.create_array(data_correct2, keys2, 0, data_structured, isotope_keys)
            self.create_array(data_correct2, keys2, 0, data_dict, isotope_keys, ndim=-1)
            self.create_array(data_correct2, keys2, 0, data_structured, isotope_keys, ndim=-1)
            self.create_array(data_correct2, keys2, 0, data_dict, isotope_keys, ndim=0)
            self.create_array(data_correct2, keys2, 0, data_structured, isotope_keys, ndim=0)

        #1-dim input 1
        data_list = [[1, 2, 3, 4]]
        data_tuple = [(1, 2, 3, 4)]
        data_array = np.array(data_list, dtype=np.float_)
        for keys in all_keys:
            keylist = isopy.keylist(keys)

            with pytest.raises(ValueError):
                isopy.array(data_list, keys[1:])

            data_dict = dict(zip(keys, [[1], [2], [3], [4]]))
            data_structured = np.array(data_tuple, dtype=[(str(key), np.float_) for key in keys])
            data_correct = np.array(data_tuple[0], dtype=[(str(keystring), np.float_) for keystring in keylist])

            self.create_array(data_correct, keys, 0, data_list, keys, ndim=-1)
            self.create_array(data_correct, keys, 0, data_list, data_structured.dtype, ndim=-1)
            self.create_array(data_correct, keys, 0, data_list, dtype=data_structured.dtype, ndim=-1)
            self.create_array(data_correct, keys, 0, data_tuple, keys, ndim=-1)
            self.create_array(data_correct, keys, 0, data_array, keys, ndim=-1)
            self.create_array(data_correct, keys, 0, data_dict, ndim=-1)
            self.create_array(data_correct, keys, 0, data_structured, ndim=-1)

            self.create_array(data_correct, keys, 0, data_list, keys, ndim=0)
            self.create_array(data_correct, keys, 0, data_list, data_structured.dtype, ndim=0)
            self.create_array(data_correct, keys, 0, data_list, dtype=data_structured.dtype, ndim=0)
            self.create_array(data_correct, keys, 0, data_tuple, keys, ndim=0)
            self.create_array(data_correct, keys, 0, data_array, keys, ndim=0)
            self.create_array(data_correct, keys, 0, data_dict, ndim=0)
            self.create_array(data_correct, keys, 0, data_structured, ndim=0)

            # Overwrite keys in data
            self.create_array(data_correct2, keys2, 0, data_dict, isotope_keys, ndim=-1)
            self.create_array(data_correct2, keys2, 0, data_list, isotope_keys, dtype=data_structured.dtype, ndim=-1)
            self.create_array(data_correct2, keys2, 0, data_structured, isotope_keys, ndim=-1)
            self.create_array(data_correct2, keys2, 0, data_dict, isotope_keys, ndim=0)
            self.create_array(data_correct2, keys2, 0, data_structured, isotope_keys, ndim=0)

            with pytest.raises(ValueError):
                self.create_array(data_correct, keys, 0, data_list, np.dtype(float), ndim=-1)

        with pytest.raises(ValueError):
            isopy.array()

        with pytest.raises(ValueError):
            isopy.array([1,2], 'ru pd'.split(), cd = 3)

    def test_create_1dim_array(self):
        mass_keys = ('104', '105', '106', '107')
        element_keys = ('mo', 'ru', 'pd', 'rh')
        isotope_keys = ('104ru', '105Pd', '106Pd', 'cd111')
        general_keys = ('harry', 'ron', 'hermione', 'neville')
        ratio_keys = ('harry/104ru', 'ron/105pd', 'hermione/106Pd', 'neville/cd111')
        mixed_keys = ('104', 'mo', '104ru', 'neville/cd111')
        molecule_keys = ('H2O', '(OH2)', 'HCl', 'HNO3')

        all_keys = (mass_keys, element_keys, isotope_keys, general_keys, ratio_keys, mixed_keys,
                    general_keys, molecule_keys)

        # 0-dim input
        data_list = [1, 2, 3, 4]
        data_tuple = (1, 2, 3, 4)
        data_array = np.array(data_list, dtype=np.float_)
        keys2 = isotope_keys
        keylist2 = isopy.keylist(isotope_keys)
        data_correct2 = np.array([data_tuple], dtype=[(str(keystring), np.float_) for keystring in keylist2])

        for keys in all_keys:
            keylist = isopy.askeylist(keys)

            with pytest.raises(ValueError):
                isopy.array(data_list[1:], keys)

            with pytest.raises(ValueError):
                isopy.array(data_list, keys[1:])

            data_dict = dict(zip(keys, data_list))
            data_structured = np.array(data_tuple, dtype=[(str(key), np.float_) for key in keys])
            data_correct = np.array([data_tuple], dtype=[(str(keystring), np.float_) for keystring in keylist])

            self.create_array(data_correct, keys, 1, data_list, keys, ndim=1)
            self.create_array(data_correct, keys, 1, data_list, data_structured.dtype, ndim=1)
            self.create_array(data_correct, keys, 1, data_list, dtype=data_structured.dtype, ndim=1)
            self.create_array(data_correct, keys, 1, data_tuple, keys, ndim=1)
            self.create_array(data_correct, keys, 1, data_array, keys, ndim=1)
            self.create_array(data_correct, keys, 1, data_dict, ndim=1)
            self.create_array(data_correct, keys, 1, data_structured, ndim=1)

            # Overwrite keys in data
            self.create_array(data_correct2, keys2, 1, data_dict, isotope_keys, ndim=1)
            self.create_array(data_correct2, keys2, 1, data_structured, isotope_keys, ndim=1)

        # 1-dim input, n == 0
        data_list = []
        data_tuple = ()
        data_array = np.array(data_list, dtype=np.float_)
        data_correct2 = np.array([], dtype=[(str(keystring), np.float_) for keystring in keylist2])
        for keys in all_keys:
            keylist = isopy.keylist(keys)

            data_dict = dict(zip(keys, [[], [], [], []]))
            data_structured = np.array(data_list, dtype=[(str(key), np.float_) for key in keys])
            data_correct = np.array(data_list, dtype=[(str(keystring), np.float_) for keystring in keylist])

            self.create_array(data_correct, keys, 1, data_list, keys)
            self.create_array(data_correct, keys, 1, data_list, data_structured.dtype)
            self.create_array(data_correct, keys, 1, data_list, dtype=data_structured.dtype)
            self.create_array(data_correct, keys, 1, data_tuple, keys)
            self.create_array(data_correct, keys, 1, data_array, keys)
            self.create_array(data_correct, keys, 1, data_dict)
            self.create_array(data_correct, keys, 1, data_structured)

            self.create_array(data_correct, keys, 1, data_list, keys, ndim=1)
            self.create_array(data_correct, keys, 1, data_list, data_structured.dtype, ndim=1)
            self.create_array(data_correct, keys, 1, data_list, dtype=data_structured.dtype, ndim=1)
            self.create_array(data_correct, keys, 1, data_tuple, keys, ndim=1)
            self.create_array(data_correct, keys, 1, data_array, keys, ndim=1)
            self.create_array(data_correct, keys, 1, data_dict, ndim=1)
            self.create_array(data_correct, keys, 1, data_structured, ndim=1)

            # Overwrite keys in data
            self.create_array(data_correct2, keys2, 1, data_dict, isotope_keys)
            self.create_array(data_correct2, keys2, 1, data_structured, isotope_keys)
            self.create_array(data_correct2, keys2, 1, data_dict, isotope_keys, ndim=1)
            self.create_array(data_correct2, keys2, 1, data_structured, isotope_keys, ndim=1)

        # 1-dim input, n == 1
        data_list = [[1, 2, 3, 4]]
        data_tuple = [(1, 2, 3, 4)]
        data_array = np.array(data_list, dtype=np.float_)
        data_correct2 = np.array(data_tuple, dtype=[(str(keystring), np.float_) for keystring in keylist2])

        for keys in all_keys:
            keylist = isopy.keylist(keys)

            data_dict = dict(zip(keys, [[1], [2], [3], [4]]))
            data_structured = np.array(data_tuple, dtype=[(str(key), np.float_) for key in keys])
            data_correct = np.array(data_tuple, dtype=[(str(keystring), np.float_) for keystring in keylist])

            self.create_array(data_correct, keys, 1, data_list, keys)
            self.create_array(data_correct, keys, 1, data_list, data_structured.dtype)
            self.create_array(data_correct, keys, 1, data_list, dtype=data_structured.dtype)
            self.create_array(data_correct, keys, 1, data_tuple, keys)
            self.create_array(data_correct, keys, 1, data_array, keys)
            self.create_array(data_correct, keys, 1, data_dict)
            self.create_array(data_correct, keys, 1, data_structured)

            self.create_array(data_correct, keys, 1, data_list, keys, ndim=1)
            self.create_array(data_correct, keys, 1, data_list, data_structured.dtype, ndim=1)
            self.create_array(data_correct, keys, 1, data_list, dtype=data_structured.dtype, ndim=1)
            self.create_array(data_correct, keys, 1, data_tuple, keys, ndim=1)
            self.create_array(data_correct, keys, 1, data_array, keys, ndim=1)
            self.create_array(data_correct, keys, 1, data_dict, ndim=1)
            self.create_array(data_correct, keys, 1, data_structured, ndim=1)

            # Overwrite keys in data
            self.create_array(data_correct2, keys2, 1, data_dict, isotope_keys)
            self.create_array(data_correct2, keys2, 1, data_structured, isotope_keys)
            self.create_array(data_correct2, keys2, 1, data_dict, isotope_keys, ndim=1)
            self.create_array(data_correct2, keys2, 1, data_structured, isotope_keys, ndim=1)

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

            self.create_array(data_correct, keys, 1, data_list, keys)
            self.create_array(data_correct, keys, 1, data_list, data_structured.dtype)
            self.create_array(data_correct, keys, 1, data_list, dtype=data_structured.dtype)
            self.create_array(data_correct, keys, 1, data_tuple, keys)
            self.create_array(data_correct, keys, 1, data_array, keys)
            self.create_array(data_correct, keys, 1, data_dict)
            self.create_array(data_correct, keys, 1, data_structured)

            self.create_array(data_correct, keys, 1, data_list, keys, ndim=1)
            self.create_array(data_correct, keys, 1, data_list, data_structured.dtype, ndim=1)
            self.create_array(data_correct, keys, 1, data_list, dtype=data_structured.dtype, ndim=1)
            self.create_array(data_correct, keys, 1, data_tuple, keys, ndim=1)
            self.create_array(data_correct, keys, 1, data_array, keys, ndim=1)
            self.create_array(data_correct, keys, 1, data_dict, ndim=1)
            self.create_array(data_correct, keys, 1, data_structured, ndim=1)

            self.create_array(data_correct, keys, 1, data_list, keys, ndim=-1)
            self.create_array(data_correct, keys, 1, data_list, data_structured.dtype, ndim=-1)
            self.create_array(data_correct, keys, 1, data_list, dtype=data_structured.dtype, ndim=-1)
            self.create_array(data_correct, keys, 1, data_tuple, keys, ndim=-1)
            self.create_array(data_correct, keys, 1, data_array, keys, ndim=-1)
            self.create_array(data_correct, keys, 1, data_dict, ndim=-1)
            self.create_array(data_correct, keys, 1, data_structured, ndim=-1)

            with pytest.raises(ValueError):
                self.create_array(data_correct, keys, 0, data_list, keys, ndim=0)
            with pytest.raises(ValueError):
                self.create_array(data_correct, keys, 0, data_list, data_structured.dtype, ndim=0)
            with pytest.raises(ValueError):
                self.create_array(data_correct, keys, 0, data_list, dtype=data_structured.dtype, ndim=0)
            with pytest.raises(ValueError):
                self.create_array(data_correct, keys, 0, data_tuple, keys, ndim=0)
            with pytest.raises(ValueError):
                self.create_array(data_correct, keys, 0, data_array, keys, ndim=0)
            with pytest.raises(ValueError):
                self.create_array(data_correct, keys, 0, data_dict, ndim=0)
            with pytest.raises(ValueError):
                self.create_array(data_correct, keys, 0, data_structured, ndim=0)

            # Overwrite keys in data
            self.create_array(data_correct2, keys2, 1, data_dict, isotope_keys)
            self.create_array(data_correct2, keys2, 1, data_structured, isotope_keys)
            self.create_array(data_correct2, keys2, 1, data_dict, isotope_keys, ndim=1)
            self.create_array(data_correct2, keys2, 1, data_structured, isotope_keys, ndim=1)
            self.create_array(data_correct2, keys2, 1, data_dict, isotope_keys, ndim=-1)
            self.create_array(data_correct2, keys2, 1, data_structured, isotope_keys, ndim=-1)

    def test_create_dataframe(self):
        d = dict(ru=[1,2,3], pd=[11., 12., 13.], cd=[21,22,23])
        df = pandas.DataFrame(d)

        a = isopy.array(df)
        assert a.keys == ['ru', 'pd', 'cd']
        for key in d.keys():
            assert np.array_equal(a[key], d[key])

        # The data type should be preserved
        assert [a.dtype[i] for i in range(len(a.dtype))] == [df.dtypes[i] for i in range(len(df.dtypes))]

        a = isopy.array(df, ['rh', 'ag', 'te'], dtype=np.float64)
        assert a.keys == ['rh', 'ag', 'te']
        for key1, key2 in zip(['rh', 'ag', 'te'], d.keys()):
            assert np.array_equal(a[key1], d[key2])

        assert [a.dtype[i] for i in range(len(a.dtype))] == [np.float64 for i in range(3)]

    def test_create_exceptions(self):
        keys = 'ru pd cd'.split()
        values = [(1,2,3,4)]

        with pytest.raises(ValueError):
            isopy.array(values, keys)

        with pytest.raises(ValueError):
            isopy.array(values)

        values = [(1,2,3,4), (21, 22, 23)]

        with pytest.raises(ValueError):
            isopy.array(values, keys)

    def test_create_dtype(self):
        keys = 'ru pd cd'.split()
        values = [(1, '2', 'three')]

        array = isopy.array(values, keys)
        assert array['ru'].dtype == np.float64
        assert array['pd'].dtype == np.float64
        assert array['cd'].dtype == np.dtype('U5')

        with pytest.raises(ValueError):
            isopy.array(values, keys, dtype=np.float64)

        array = isopy.array(values, keys, dtype=[np.int64, np.float64, np.unicode_])
        assert array['ru'].dtype == np.int64
        assert array['pd'].dtype == np.float64
        assert array['cd'].dtype == np.dtype('U5')

        with pytest.raises(ValueError):
            isopy.array(values, keys, dtype=[np.int64, np.float64, np.unicode_, np.int64])

        array = isopy.array(values, keys, dtype=(np.int64, np.unicode_))
        assert array['ru'].dtype == np.int64
        assert array['pd'].dtype == np.int64
        assert array['cd'].dtype == np.dtype('U5')

        with pytest.raises(ValueError):
            isopy.array(values, keys, dtype=(np.int64, np.float64))

    def create_array(self, correct, keys, ndim, /, *args, **kwargs):
        result = isopy.array(*args, **kwargs)
        keylist = isopy.askeylist(keys)
        assert result.flavour == keylist.flavour
        assert isinstance(result, core.IsopyNdarray)
        assert result.keys == keylist
        assert result.ncols == len(keylist)
        assert result.ndim == ndim
        if ndim == 0:
            assert result.nrows == -1
            with pytest.raises(TypeError):
                len(result)
        else:
            assert result.nrows == result.size
            assert len(result) == result.size

        assert_array_equal_array(result, correct)

        result2 = isopy.array(result)
        assert result2 is not result
        assert_array_equal_array(result2, correct)

        result2 = isopy.asarray(result)
        assert result2 is result

        result2 = isopy.asanyarray(result)
        assert result2 is result

        for i, (key, value) in enumerate(result.items()):
            assert key == keylist[i]
            np.testing.assert_allclose(value, result[key])

        for i, value in enumerate(result.values()):
            key = keylist[i]
            np.testing.assert_allclose(value, result[key])

    def test_eq(self):
        data1 = np.array([[i for i in range(1, 7)], [i ** 2 for i in range(1, 7)]])
        data2 = np.array([[i for i in range(1, 7)], [i ** 2 for i in range(1, 7)], [i ** 3 for i in range(1, 7)]])
        data3 = np.array([[i for i in range(1, 4)] + [2 * i for i in range(4,7)],
                          [i ** 2 for i in range(1, 4)] + [i ** 2 for i in range(4, 7)]])

        array1 = isopy.array(data1, '102pd 104pd 105pd 106pd 108pd 110pd'.split())
        array2 = isopy.array(data2, '102pd 104pd 105pd 106pd 108pd 110pd'.split())
        array3 = isopy.array(data3, '102pd 104pd 105pd 106pd 108pd 110pd'.split())

        assert array1 != array2
        assert array1 != array3
        assert array2 != array3
        assert array1[0] != array3[0]
        assert array1 != array1[0]
        assert array1['102pd 104pd 105pd'.split()] != array2['102pd 104pd 105pd'.split()]
        assert array3 != array3.filter(mass_number_lt=106)

        assert array1 == array1
        assert array1[0] == array2[0]
        assert array1 == array2[:2]
        assert array1['102pd 104pd 105pd'.split()] == array3['102pd 104pd 105pd'.split()]
        assert array1['102pd 104pd 105pd'.split()] == array3['105pd 104pd 102pd'.split()]
        assert array1['102pd 104pd 105pd'.split()] == array3.filter(mass_number_lt = 106)
        assert array3 == array3.filter()

    def test_normalise(self):
        data = np.array([[i for i in range(1, 7)], [i**2 for i in range(1, 7)]])
        array = isopy.array(data, '102pd 104pd 105pd 106pd 108pd 110pd'.split())

        with pytest.raises(TypeError):
            array.normalise(key=1)

        # all
        array1 = array.normalise()
        np.testing.assert_allclose(np.sum(array1, axis=1), [1, 1])
        np.testing.assert_allclose(array1.tolist(), data / np.sum(data, axis=1, keepdims=True))

        array1 = array.normalise(10)
        np.testing.assert_allclose(np.sum(array1, axis=1), [10, 10])
        np.testing.assert_allclose(array1.tolist(), data / np.sum(data, axis=1, keepdims=True) * 10)

        array1 = array.normalise([1, 10])
        np.testing.assert_allclose(np.sum(array1, axis=1), [1, 10])
        np.testing.assert_allclose(array1.tolist(), data / np.sum(data, axis=1, keepdims=True) * [[1], [10]])

        # With key

        array1 = array.normalise(1, '110pd')
        np.testing.assert_allclose(np.sum(array1, axis=1), np.sum(data, axis=1) / [6, 36])
        np.testing.assert_allclose(array1.tolist(), data / [[6], [36]])

        array1 = array.normalise(10, '110pd')
        np.testing.assert_allclose(np.sum(array1, axis=1), np.sum(data, axis=1) / [0.6, 3.6])
        np.testing.assert_allclose(array1.tolist(), data / [[0.6], [3.6]])

        array1 = array.normalise([1, 10], '110pd')
        np.testing.assert_allclose(np.sum(array1, axis=1), np.sum(data, axis=1) / [6, 3.6])
        np.testing.assert_allclose(array1.tolist(), data / [[6], [3.6]])

        # Multiple keys

        array1 = array.normalise(1, ('104pd','110pd'))
        np.testing.assert_allclose(np.sum(array1, axis=1), np.sum(data, axis=1) / [8, 40])
        np.testing.assert_allclose(array1.tolist(), data / [[8], [40]])

        array1 = array.normalise(10, ('104pd', '110pd'))
        np.testing.assert_allclose(np.sum(array1, axis=1), np.sum(data, axis=1) / [0.8, 4.0])
        np.testing.assert_allclose(array1.tolist(), data / [[0.8], [4.0]])

        array1 = array.normalise([1, 10], ('104pd', '110pd'))
        np.testing.assert_allclose(np.sum(array1, axis=1), np.sum(data, axis=1) / [8, 4.0])
        np.testing.assert_allclose(array1.tolist(), data / [[8], [4.0]])

        # function

        array1 = array.normalise(1, isopy.keymax)
        np.testing.assert_allclose(np.sum(array1, axis=1), np.sum(data, axis=1) / [6, 36])
        np.testing.assert_allclose(array1.tolist(), data / [[6], [36]])

        array1 = array.normalise(10, isopy.keymax)
        np.testing.assert_allclose(np.sum(array1, axis=1), np.sum(data, axis=1) / [0.6, 3.6])
        np.testing.assert_allclose(array1.tolist(), data / [[0.6], [3.6]])

        array1 = array.normalise([1, 10], isopy.keymax)
        np.testing.assert_allclose(np.sum(array1, axis=1), np.sum(data, axis=1) / [6, 3.6])
        np.testing.assert_allclose(array1.tolist(), data / [[6], [3.6]])

    def test_ratio(self):
        array = isopy.random(20, [(1, 0.1), (10, 1), (0, 1)], 'ru pd cd'.split())

        with pytest.raises(ValueError):
            array.ratio('ag')

        # No denominator
        ratio = array.ratio()
        assert ratio.flavour == 'ratio'
        assert ratio.keys == 'ru/pd cd/pd'.split()
        for key in ratio.keys:
            np.testing.assert_allclose(ratio[key], array[key.numerator] / array[key.denominator])

        array2 = ratio.deratio()
        assert array2.flavour == 'element'
        assert array2.keys == 'ru pd cd'.split()
        for key in array2.keys:
            np.testing.assert_allclose(array2[key], array[key] / array['pd'])

        array2 = ratio.deratio(100, sort_keys=True)
        assert array2.flavour == 'element'
        assert array2.keys == 'ru pd cd'.split()
        for key in array2.keys:
            np.testing.assert_allclose(array2[key], array[key] / array['pd'] * 100)

        array2 = ratio.deratio(array['pd'], sort_keys = False)
        assert array2.flavour == 'element'
        assert array2.keys == 'ru cd pd'.split()
        for key in array2.keys:
            np.testing.assert_allclose(array2[key], array[key])

        # remove denominator
        ratio = array.ratio('pd')
        assert ratio.flavour == 'ratio'
        assert ratio.keys == 'ru/pd cd/pd'.split()
        for key in ratio.keys:
            np.testing.assert_allclose(ratio[key], array[key.numerator] / array[key.denominator])

        array2 = ratio.deratio()
        assert array2.flavour == 'element'
        assert array2.keys == 'ru pd cd'.split()
        for key in array2.keys:
            np.testing.assert_allclose(array2[key], array[key] / array['pd'])

        array2 = ratio.deratio(100, sort_keys=False)
        assert array2.flavour == 'element'
        assert array2.keys == 'ru cd pd'.split()
        for key in array2.keys:
            np.testing.assert_allclose(array2[key], array[key] / array['pd'] * 100)

        array2 = ratio.deratio(array['pd'])
        assert array2.flavour == 'element'
        assert array2.keys == 'ru pd cd'.split()
        for key in array2.keys:
            np.testing.assert_allclose(array2[key], array[key])

        # keep denominator
        ratio = array.ratio('pd', False)
        assert ratio.flavour == 'ratio'
        assert ratio.keys == 'ru/pd pd/pd cd/pd'.split()
        for key in ratio.keys:
            np.testing.assert_allclose(ratio[key], array[key.numerator] / array[key.denominator])

        array2 = ratio.deratio()
        assert array2.flavour == 'element'
        assert array2.keys == 'ru pd cd'.split()
        for key in array2.keys:
            np.testing.assert_allclose(array2[key], array[key] / array['pd'])

        array2 = ratio.deratio(100, sort_keys=True)
        assert array2.flavour == 'element'
        assert array2.keys == 'ru pd cd'.split()
        for key in array2.keys:
            np.testing.assert_allclose(array2[key], array[key] / array['pd'] * 100)

        array2 = ratio.deratio(array['pd'], sort_keys=False)
        assert array2.flavour == 'element'
        assert array2.keys == 'ru pd cd'.split()
        for key in array2.keys:
            np.testing.assert_allclose(array2[key], array[key])

        #test deratio with no common denominator
        array = isopy.random(20, [(1, 0.1), (10, 1), (0, 1)], 'ru/ag pd/ag cd/rh'.split())
        with pytest.raises(ValueError):
            array.deratio()

        array = isopy.random(20, [(1, 0.1), (10, 1), (0, 1)], 'ru pd cd'.split())
        with pytest.raises(TypeError):
            array.deratio()

    def test_getitem_1dim(self):
        # ndarray
        a = isopy.array(dict(ru=[1,2,3], pd=[11,12,13], cd=[21,22,23]))
        assert type(a) == core.IsopyNdarray

        # Key
        b = a['pd']
        assert type(b) is np.ndarray
        assert np.array_equal(a['pd'], [11,12,13])

        b = a['ru', 'cd']
        assert type(b) == core.IsopyNdarray
        assert b.keys == ['ru', 'cd']
        for key in b.keys:
            np.array_equal(b[key], a[key])

        b = a['cd', 'ru']
        assert type(b) == core.IsopyNdarray
        assert b.keys == ['cd', 'ru']
        for key in b.keys:
            np.array_equal(b[key], a[key])

        b = a[('ru', 'cd')]
        assert type(b) == core.IsopyNdarray
        assert b.keys == ['ru', 'cd']
        for key in b.keys:
            np.array_equal(b[key], a[key])

        b = a[['ru', 'cd']]
        assert type(b) == core.IsopyNdarray
        assert b.keys == ['ru', 'cd']
        for key in b.keys:
            np.array_equal(b[key], a[key])

        # Row
        b = a[0]
        assert type(b) == core.IsopyVoid
        assert b.keys == a.keys
        for key in b.keys:
            assert np.array_equal(b[key], a[key][0])

        b = a[[1]]
        assert type(b) == core.IsopyNdarray
        assert b.keys == a.keys
        for key in b.keys:
            np.array_equal(b[key], a[key][[1]])

        with pytest.raises(IndexError):
            a[0,2]

        with pytest.raises(IndexError):
            a[(0,2)]

        b = a[[0,2]]
        assert type(b) == core.IsopyNdarray
        assert b.keys == a.keys
        for key in b.keys:
            np.array_equal(b[key], a[key][[0, 2]])

        b = a[()]
        assert type(b) == core.IsopyNdarray
        assert b.keys == a.keys
        for key in b.keys:
            np.array_equal(b[key], a[key])
        
        # Row slice
        b = a[:]
        assert type(b) == core.IsopyNdarray
        assert b.keys == a.keys
        for key in b.keys:
            np.array_equal(b[key], a[key])

        b = a[:2]
        assert type(b) == core.IsopyNdarray
        assert b.keys == a.keys
        for key in b.keys:
            np.array_equal(b[key], a[key][[0,1]])

        b = a[::2]
        assert type(b) == core.IsopyNdarray
        assert b.keys == a.keys
        for key in b.keys:
            np.array_equal(b[key], a[key][[0,2]])

        b = a[3:]
        assert type(b) == core.IsopyNdarray
        assert b.keys == a.keys
        assert b.size == 0
        assert b.ndim == 1
            
        # Row and column
        b = a['pd', 1]
        assert type(b) is np.float64
        assert np.array_equal(b, a['pd'][1])

        b = a['pd', [1]]
        assert type(b) is np.ndarray
        assert np.array_equal(b, a['pd'][[1]])

        b = a['pd', [0,2]]
        assert type(b) is np.ndarray
        assert np.array_equal(b, a['pd'][[0,2]])

        b = a['pd', ::2]
        assert type(b) is np.ndarray
        assert np.array_equal(b, a['pd'][[0, 2]])

        b = a['pd', :]
        assert type(b) is np.ndarray
        assert np.array_equal(b, a['pd'])

        b = a[['pd'], 1]
        assert type(b) is core.IsopyVoid
        assert b.keys == ['pd']
        assert np.array_equal(b['pd'], 12.0)

        b = a[['pd'], [0, 2]]
        assert type(b) is core.IsopyNdarray
        assert b.keys == ['pd']
        assert np.array_equal(b['pd'], a['pd'][[0,2]])

        b = a[['pd'], ::2]
        assert type(b) is core.IsopyNdarray
        assert b.keys == ['pd']
        assert np.array_equal(b['pd'], a['pd'][[0,2]])

        b = a[['pd', 'ru'], 1]
        assert type(b) is core.IsopyVoid
        assert b.keys == ['pd', 'ru']
        for key in b.keys:
            assert np.array_equal(b[key], a[key][1])

        b = a[['pd', 'ru'], [0,2]]
        assert type(b) is core.IsopyNdarray
        assert b.keys == ['pd', 'ru']
        for key in b.keys:
            assert np.array_equal(b[key], a[key][[0,2]])

        b = a[['pd', 'ru'], ::2]
        assert type(b) is core.IsopyNdarray
        assert b.keys == ['pd', 'ru']
        for key in b.keys:
            assert np.array_equal(b[key], a[key][[0,2]])

        b = a[:, 1]
        assert type(b) is core.IsopyVoid
        assert b.keys == a.keys
        for key in b.keys:
            assert np.array_equal(b[key], a[key][1])

        b = a[:, [0, 2]]
        assert type(b) is core.IsopyNdarray
        assert b.keys == a.keys
        for key in b.keys:
            assert np.array_equal(b[key], a[key][[0,2]])

        b = a[:, ::2]
        assert type(b) is core.IsopyNdarray
        assert b.keys == a.keys
        for key in b.keys:
            assert np.array_equal(b[key], a[key][[0,2]])

        with pytest.raises(IndexError):
            a[:1, 1]

        with pytest.raises(IndexError):
            a[a > 1]

    def test_getitem_0dim(self):
        a1 = isopy.array(dict(ru=[1], pd=[11], cd=[21]))
        v = a1[0]
        a = isopy.array(dict(ru=1, pd=11, cd=21))

        # Key array
        b = a['pd']
        assert type(b) is np.ndarray
        assert np.array_equal(a['pd'], 11)

        b = a['ru', 'cd']
        assert type(b) == core.IsopyNdarray
        assert b.keys == ['ru', 'cd']
        for key in b.keys:
            np.array_equal(b[key], a[key])

        b = a['cd', 'ru']
        assert type(b) == core.IsopyNdarray
        assert b.keys == ['cd', 'ru']
        for key in b.keys:
            np.array_equal(b[key], a[key])

        b = a[('ru', 'cd')]
        assert type(b) == core.IsopyNdarray
        assert b.keys == ['ru', 'cd']
        for key in b.keys:
            np.array_equal(b[key], a[key])

        b = a[['ru', 'cd']]
        assert type(b) == core.IsopyNdarray
        assert b.keys == ['ru', 'cd']
        for key in b.keys:
            np.array_equal(b[key], a[key])

        # Key void
        b = v['pd']
        assert type(b) is np.float64
        assert np.array_equal(a['pd'], 11)

        b = v['ru', 'cd']
        assert type(b) == core.IsopyVoid
        assert b.keys == ['ru', 'cd']
        for key in b.keys:
            np.array_equal(b[key], a[key])

        b = v['cd', 'ru']
        assert type(b) == core.IsopyVoid
        assert b.keys == ['cd', 'ru']
        for key in b.keys:
            np.array_equal(b[key], a[key])

        b = v[('ru', 'cd')]
        assert type(b) == core.IsopyVoid
        assert b.keys == ['ru', 'cd']
        for key in b.keys:
            np.array_equal(b[key], a[key])

        b = v[['ru', 'cd']]
        assert type(b) == core.IsopyVoid
        assert b.keys == ['ru', 'cd']
        for key in b.keys:
            np.array_equal(b[key], a[key])

        # Row array
        b = a[:]
        assert type(b) is core.IsopyNdarray
        assert b.keys == a.keys
        for key in b.keys:
            assert np.array_equal(b[key], a[key])

        with pytest.raises(IndexError):
            a[1]

        with pytest.raises(IndexError):
            a[[]]

        with pytest.raises(IndexError):
            a[::2]

        # Row void
        b = v[:]
        assert type(b) is core.IsopyVoid
        assert b.keys == a.keys
        for key in b.keys:
            assert np.array_equal(b[key], a[key])

        with pytest.raises(IndexError):
            v[1]

        with pytest.raises(IndexError):
            v[[]]

        with pytest.raises(IndexError):
            v[::2]

        # Row and column array
        b = a['pd', :]
        assert type(b) is np.ndarray
        assert np.array_equal(a['pd'], 11)

        b = a[('ru', 'cd'), :]
        assert type(b) == core.IsopyNdarray
        assert b.keys == ['ru', 'cd']
        for key in b.keys:
            np.array_equal(b[key], a[key])

        b = a[:, :]
        assert type(b) == core.IsopyNdarray
        assert b.keys == a.keys
        for key in b.keys:
            np.array_equal(b[key], a[key])

        with pytest.raises(IndexError):
            a['pd', 1]

        with pytest.raises(IndexError):
            a['pd', []]

        with pytest.raises(IndexError):
            a['pd', ::2]

        # Row and column array
        b = v['pd', :]
        assert type(b) is np.float64
        assert np.array_equal(a['pd'], 11)

        b = v[('ru', 'cd'), :]
        assert type(b) == core.IsopyVoid
        assert b.keys == ['ru', 'cd']
        for key in b.keys:
            np.array_equal(b[key], a[key])

        b = v[:, :]
        assert type(b) == core.IsopyVoid
        assert b.keys == a.keys
        for key in b.keys:
            np.array_equal(b[key], a[key])

        with pytest.raises(IndexError):
            v['pd', 1]

        with pytest.raises(IndexError):
            v['pd', []]

        with pytest.raises(IndexError):
            v['pd', ::2]

    def test_setitem_1dim_keyless(self):
        # ndarray
        a = isopy.array(dict(ru=[1, 2, 3], pd=[11, 12, 13], cd=[21, 22, 23]))

        # Key
        a['pd'] = 1
        assert np.array_equal(a['ru'], [1, 2, 3])
        assert np.array_equal(a['pd'], [1, 1, 1])
        assert np.array_equal(a['cd'], [21, 22, 23])

        a['ru', 'cd'] = [10,20,30]
        assert np.array_equal(a['ru'], [10, 20, 30])
        assert np.array_equal(a['pd'], [1, 1, 1])
        assert np.array_equal(a['cd'], [10, 20, 30])

        a[('ru', 'cd')] = 2
        assert np.array_equal(a['ru'], [2, 2, 2])
        assert np.array_equal(a['pd'], [1, 1, 1])
        assert np.array_equal(a['cd'], [2, 2, 2])

        a[['ru', 'cd']] = [1, 2, 3]
        assert np.array_equal(a['ru'], [1, 2, 3])
        assert np.array_equal(a['pd'], [1, 1, 1])
        assert np.array_equal(a['cd'], [1, 2, 3])

        with pytest.raises(ValueError):
            a['pd'] = [1, 2, 3, 4]

        with pytest.raises(ValueError):
            a[['ru', 'cd']] = [1, 2, 3, 4]

        assert np.array_equal(a['ru'], [1, 2, 3])
        assert np.array_equal(a['pd'], [1, 1, 1])
        assert np.array_equal(a['cd'], [1, 2, 3])

        # Row
        a = isopy.array(dict(ru=[1, 2, 3], pd=[11, 12, 13], cd=[21, 22, 23]))

        a[0] = 10
        assert np.array_equal(a['ru'], [10, 2, 3])
        assert np.array_equal(a['pd'], [10, 12, 13])
        assert np.array_equal(a['cd'], [10, 22, 23])

        a[0] = [100]
        assert np.array_equal(a['ru'], [100, 2, 3])
        assert np.array_equal(a['pd'], [100, 12, 13])
        assert np.array_equal(a['cd'], [100, 22, 23])

        a[[1]] = 200
        assert np.array_equal(a['ru'], [100, 200, 3])
        assert np.array_equal(a['pd'], [100, 200, 13])
        assert np.array_equal(a['cd'], [100, 200, 23])

        with pytest.raises(IndexError):
            a[0, 2] = [10, 30]

        with pytest.raises(IndexError):
            a[(0, 2)] = [10, 30]

        assert np.array_equal(a['ru'], [100, 200, 3])
        assert np.array_equal(a['pd'], [100, 200, 13])
        assert np.array_equal(a['cd'], [100, 200, 23])

        a[[0, 2]] = [10, 30]
        assert np.array_equal(a['ru'], [10, 200, 30])
        assert np.array_equal(a['pd'], [10, 200, 30])
        assert np.array_equal(a['cd'], [10, 200, 30])

        # Row slice
        a[:] = [1, 2, 3]
        assert np.array_equal(a['ru'], [1, 2, 3])
        assert np.array_equal(a['pd'], [1, 2, 3])
        assert np.array_equal(a['cd'], [1, 2, 3])

        a[:2] = 10
        assert np.array_equal(a['ru'], [10, 10, 3])
        assert np.array_equal(a['pd'], [10, 10, 3])
        assert np.array_equal(a['cd'], [10, 10, 3])

        a[::2] = 1
        assert np.array_equal(a['ru'], [1, 10, 1])
        assert np.array_equal(a['pd'], [1, 10, 1])
        assert np.array_equal(a['cd'], [1, 10, 1])

        a[3:] = 1
        assert np.array_equal(a['ru'], [1, 10, 1])
        assert np.array_equal(a['pd'], [1, 10, 1])
        assert np.array_equal(a['cd'], [1, 10, 1])

        # Row and column
        a = isopy.array(dict(ru=[1, 2, 3], pd=[11, 12, 13], cd=[21, 22, 23]))

        a['pd', 1] = 200
        assert np.array_equal(a['ru'], [1, 2, 3])
        assert np.array_equal(a['pd'], [11, 200, 13])
        assert np.array_equal(a['cd'], [21, 22, 23])

        a['pd', [1]] = 12
        assert np.array_equal(a['ru'], [1, 2, 3])
        assert np.array_equal(a['pd'], [11, 12, 13])
        assert np.array_equal(a['cd'], [21, 22, 23])

        a['pd', [0, 2]] = 10
        assert np.array_equal(a['ru'], [1, 2, 3])
        assert np.array_equal(a['pd'], [10, 12, 10])
        assert np.array_equal(a['cd'], [21, 22, 23])

        a['pd', ::2] = [11, 13]
        assert np.array_equal(a['ru'], [1, 2, 3])
        assert np.array_equal(a['pd'], [11, 12, 13])
        assert np.array_equal(a['cd'], [21, 22, 23])

        a['pd', :] = [100, 200, 300]
        assert np.array_equal(a['ru'], [1, 2, 3])
        assert np.array_equal(a['pd'], [100, 200, 300])
        assert np.array_equal(a['cd'], [21, 22, 23])

        a[['pd'], 1] = 1
        assert np.array_equal(a['ru'], [1, 2, 3])
        assert np.array_equal(a['pd'], [100, 1, 300])
        assert np.array_equal(a['cd'], [21, 22, 23])

        a[['pd'], [0, 2]] = [11, 13]
        assert np.array_equal(a['ru'], [1, 2, 3])
        assert np.array_equal(a['pd'], [11, 1, 13])
        assert np.array_equal(a['cd'], [21, 22, 23])

        a[['pd'], ::2] = 100
        assert np.array_equal(a['ru'], [1, 2, 3])
        assert np.array_equal(a['pd'], [100, 1, 100])
        assert np.array_equal(a['cd'], [21, 22, 23])

        a[['pd', 'ru'], 1] = 5
        assert np.array_equal(a['ru'], [1, 5, 3])
        assert np.array_equal(a['pd'], [100, 5, 100])
        assert np.array_equal(a['cd'], [21, 22, 23])

        a[['pd', 'cd'], [0, 2]] = [6, 7]
        assert np.array_equal(a['ru'], [1, 5, 3])
        assert np.array_equal(a['pd'], [6, 5, 7])
        assert np.array_equal(a['cd'], [6, 22, 7])

        a[['pd', 'cd'], ::2] = 10
        assert np.array_equal(a['ru'], [1, 5, 3])
        assert np.array_equal(a['pd'], [10, 5, 10])
        assert np.array_equal(a['cd'], [10, 22, 10])

        a[:, 1] = 6
        assert np.array_equal(a['ru'], [1, 6, 3])
        assert np.array_equal(a['pd'], [10, 6, 10])
        assert np.array_equal(a['cd'], [10, 6, 10])

        a[:, [0, 2]] = [8, 9]
        assert np.array_equal(a['ru'], [8, 6, 9])
        assert np.array_equal(a['pd'], [8, 6, 9])
        assert np.array_equal(a['cd'], [8, 6, 9])

        a[:, ::2] = 7
        assert np.array_equal(a['ru'], [7, 6, 7])
        assert np.array_equal(a['pd'], [7, 6, 7])
        assert np.array_equal(a['cd'], [7, 6, 7])

        with pytest.raises(IndexError):
            a[:1, 1] = 1

        a[a >= 7] = 10
        assert np.array_equal(a['ru'], [10, 6, 10])
        assert np.array_equal(a['pd'], [10, 6, 10])
        assert np.array_equal(a['cd'], [10, 6, 10])

        a[a[['pd']] < 8] = 2
        assert np.array_equal(a['ru'], [10, 6, 10])
        assert np.array_equal(a['pd'], [10, 2, 10])
        assert np.array_equal(a['cd'], [10, 6, 10])

    def test_setitem_1dim_keyed(self):
        # ndarray
        a = isopy.array(dict(ru=[1, 2, 3], pd=[11, 12, 13], cd=[21, 22, 23]))
        b = isopy.array(dict(ru=[4, 6], cd=[24, 26])).default(100)
        c = dict(ru=7, pd=17, cd=27)
        d = isopy.asrefval(dict(ru=8, pd=18), default_value=200)

        # Key
        a['pd'] = c
        assert np.array_equal(a['ru'], [1, 2, 3])
        assert np.array_equal(a['pd'], [17, 17, 17])
        assert np.array_equal(a['cd'], [21, 22, 23])

        a[['ru', 'cd']] = d
        assert np.array_equal(a['ru'], [8, 8, 8])
        assert np.array_equal(a['pd'], [17, 17, 17])
        assert np.array_equal(a['cd'], [200, 200, 200])

        with pytest.raises(ValueError):
            a[['ru', 'cd']] = b

        assert np.array_equal(a['ru'], [8, 8, 8])
        assert np.array_equal(a['pd'], [17, 17, 17])
        assert np.array_equal(a['cd'], [200, 200, 200])

        # Row
        a = isopy.array(dict(ru=[1, 2, 3], pd=[11, 12, 13], cd=[21, 22, 23]))

        a[0] = c
        assert np.array_equal(a['ru'], [7, 2, 3])
        assert np.array_equal(a['pd'], [17, 12, 13])
        assert np.array_equal(a['cd'], [27, 22, 23])

        a[[1]] = d
        assert np.array_equal(a['ru'], [7, 8, 3])
        assert np.array_equal(a['pd'], [17, 18, 13])
        assert np.array_equal(a['cd'], [27, 200, 23])

        a[[0, 2]] = b
        assert np.array_equal(a['ru'], [4, 8, 6])
        assert np.array_equal(a['pd'], [100, 18, 100])
        assert np.array_equal(a['cd'], [24, 200, 26])

        # Row slice
        a = isopy.array(dict(ru=[1, 2, 3], pd=[11, 12, 13], cd=[21, 22, 23]))
        a[::2] = b
        assert np.array_equal(a['ru'], [4, 2, 6])
        assert np.array_equal(a['pd'], [100, 12, 100])
        assert np.array_equal(a['cd'], [24, 22, 26])

        # Row and column
        a = isopy.array(dict(ru=[1, 2, 3], pd=[11, 12, 13], cd=[21, 22, 23]))

        a['pd', 1] = c
        assert np.array_equal(a['ru'], [1, 2, 3])
        assert np.array_equal(a['pd'], [11, 17, 13])
        assert np.array_equal(a['cd'], [21, 22, 23])

        a['pd', [1]] = d
        assert np.array_equal(a['ru'], [1, 2, 3])
        assert np.array_equal(a['pd'], [11, 18, 13])
        assert np.array_equal(a['cd'], [21, 22, 23])

        a['pd', [0, 2]] = b
        assert np.array_equal(a['ru'], [1, 2, 3])
        assert np.array_equal(a['pd'], [100, 18, 100])
        assert np.array_equal(a['cd'], [21, 22, 23])

        a['pd', ::2] = d
        assert np.array_equal(a['ru'], [1, 2, 3])
        assert np.array_equal(a['pd'], [18, 18, 18])
        assert np.array_equal(a['cd'], [21, 22, 23])

        a['pd', :] = c
        assert np.array_equal(a['ru'], [1, 2, 3])
        assert np.array_equal(a['pd'], [17, 17, 17])
        assert np.array_equal(a['cd'], [21, 22, 23])

        a[['pd'], 1] = d
        assert np.array_equal(a['ru'], [1, 2, 3])
        assert np.array_equal(a['pd'], [17, 18, 17])
        assert np.array_equal(a['cd'], [21, 22, 23])

        a[['pd'], [0, 2]] = b
        assert np.array_equal(a['ru'], [1, 2, 3])
        assert np.array_equal(a['pd'], [100, 18, 100])
        assert np.array_equal(a['cd'], [21, 22, 23])

        a = isopy.array(dict(ru=[1, 2, 3], pd=[11, 12, 13], cd=[21, 22, 23]))
        a[['pd', 'ru'], 1] = c
        assert np.array_equal(a['ru'], [1, 7, 3])
        assert np.array_equal(a['pd'], [11, 17, 13])
        assert np.array_equal(a['cd'], [21, 22, 23])

        a[['pd', 'cd'], [0, 2]] = b
        assert np.array_equal(a['ru'], [1, 7, 3])
        assert np.array_equal(a['pd'], [100, 17, 100])
        assert np.array_equal(a['cd'], [24, 22, 26])

        a[['pd', 'cd'], ::2] = d
        assert np.array_equal(a['ru'], [1, 7, 3])
        assert np.array_equal(a['pd'], [18, 17, 18])
        assert np.array_equal(a['cd'], [200, 22, 200])

        a = isopy.array(dict(ru=[1, 2, 3], pd=[11, 12, 13], cd=[21, 22, 23]))
        a[:, 1] = c
        assert np.array_equal(a['ru'], [1, 7, 3])
        assert np.array_equal(a['pd'], [11, 17, 13])
        assert np.array_equal(a['cd'], [21, 27, 23])

        a[:, [0, 2]] = b
        assert np.array_equal(a['ru'], [4, 7, 6])
        assert np.array_equal(a['pd'], [100, 17, 100])
        assert np.array_equal(a['cd'], [24, 27, 26])

        a[:, ::2] = d
        assert np.array_equal(a['ru'], [8, 7, 8])
        assert np.array_equal(a['pd'], [18, 17, 18])
        assert np.array_equal(a['cd'], [200, 27, 200])

        a = isopy.array(dict(ru=[1, 2, 3], pd=[11, 12, 13], cd=[21, 22, 23]))
        a[a >= 12] = c
        assert np.array_equal(a['ru'], [1, 2, 3])
        assert np.array_equal(a['pd'], [11, 17, 17])
        assert np.array_equal(a['cd'], [27, 27, 27])

        a = isopy.array(dict(ru=[1, 2, 3], pd=[11, 12, 13], cd=[21, 22, 23]))
        a[a[['pd']] < 12] = d
        assert np.array_equal(a['ru'], [1, 2, 3])
        assert np.array_equal(a['pd'], [18, 12, 13])
        assert np.array_equal(a['cd'], [21, 22, 23])

    def test_setitem_0dim_keyless(self):
        # ndarray
        a1 = isopy.array(dict(ru=[1], pd=[11], cd=[21]))
        v = a1[0]
        a = isopy.array(dict(ru=1, pd=11, cd=21))

        # Key
        a['pd'] = 1
        assert np.array_equal(a['ru'], 1)
        assert np.array_equal(a['pd'], 1)
        assert np.array_equal(a['cd'], 21)

        v['pd'] = 1
        assert np.array_equal(v['ru'], 1)
        assert np.array_equal(v['pd'], 1)
        assert np.array_equal(v['cd'], 21)

        a['ru', 'cd'] = [10]
        assert np.array_equal(a['ru'], 10)
        assert np.array_equal(a['pd'], 1)
        assert np.array_equal(a['cd'], 10)

        v['ru', 'cd'] = [10]
        assert np.array_equal(v['ru'], 10)
        assert np.array_equal(v['pd'], 1)
        assert np.array_equal(v['cd'], 10)

        a[('ru', 'cd')] = 2
        assert np.array_equal(a['ru'], 2)
        assert np.array_equal(a['pd'], 1)
        assert np.array_equal(a['cd'], 2)

        v[('ru', 'cd')] = 2
        assert np.array_equal(v['ru'], 2)
        assert np.array_equal(v['pd'], 1)
        assert np.array_equal(v['cd'], 2)

        a[['ru', 'cd']] = [1]
        assert np.array_equal(a['ru'], 1)
        assert np.array_equal(a['pd'], 1)
        assert np.array_equal(a['cd'], 1)

        v[['ru', 'cd']] = [1]
        assert np.array_equal(v['ru'], 1)
        assert np.array_equal(v['pd'], 1)
        assert np.array_equal(v['cd'], 1)

        with pytest.raises(ValueError):
            a['pd'] = [1, 2]

        with pytest.raises(ValueError):
            v['pd'] = [1, 2]

        with pytest.raises(ValueError):
            a[['ru', 'cd']] = [1, 2]

        with pytest.raises(ValueError):
            v[['ru', 'cd']] = [1, 2]

        assert np.array_equal(a['ru'], 1)
        assert np.array_equal(a['pd'], 1)
        assert np.array_equal(a['cd'], 1)

        assert np.array_equal(v['ru'], 1)
        assert np.array_equal(v['pd'], 1)
        assert np.array_equal(v['cd'], 1)

        # Row
        a1 = isopy.array(dict(ru=[1], pd=[11], cd=[21]))
        v = a1[0]
        a = isopy.array(dict(ru=1, pd=11, cd=21))

        a[0] = 10
        assert np.array_equal(a['ru'], 10)
        assert np.array_equal(a['pd'], 10)
        assert np.array_equal(a['cd'], 10)

        v[0] = 1
        assert np.array_equal(v['ru'], 1)
        assert np.array_equal(v['pd'], 1)
        assert np.array_equal(v['cd'], 1)

        with pytest.raises(IndexError):
            a[0, 2] = 10

        with pytest.raises(IndexError):
            v[0, 2] = 10

        with pytest.raises(IndexError):
            a[(0, 2)] = 10

        with pytest.raises(IndexError):
            v[(0, 2)] = 10

        with pytest.raises(IndexError):
            a[[0, 2]] = 10

        with pytest.raises(IndexError):
            v[[0, 2]] = 10

        a1 = isopy.array(dict(ru=[1], pd=[11], cd=[21]))
        v = a1[0]
        a = isopy.array(dict(ru=1, pd=11, cd=21))

        assert np.array_equal(a['ru'], 1)
        assert np.array_equal(a['pd'], 11)
        assert np.array_equal(a['cd'], 21)

        assert np.array_equal(v['ru'], 1)
        assert np.array_equal(v['pd'], 11)
        assert np.array_equal(v['cd'], 21)

        # Row slice
        a[:] = 10
        assert np.array_equal(a['ru'], 10)
        assert np.array_equal(a['pd'], 10)
        assert np.array_equal(a['cd'], 10)

        v[:] = 10
        assert np.array_equal(v['ru'], 10)
        assert np.array_equal(v['pd'], 10)
        assert np.array_equal(v['cd'], 10)

        with pytest.raises(IndexError):
            a[:2] = 10

        with pytest.raises(IndexError):
            v[:2] = 10

        with pytest.raises(IndexError):
            a[::2] = 1

        with pytest.raises(IndexError):
            v[::2] = 1

        with pytest.raises(IndexError):
            a[3:] = 1

        with pytest.raises(IndexError):
            v[3:] = 1

        assert np.array_equal(a['ru'], 10)
        assert np.array_equal(a['pd'], 10)
        assert np.array_equal(a['cd'], 10)

        assert np.array_equal(v['ru'], 10)
        assert np.array_equal(v['pd'], 10)
        assert np.array_equal(v['cd'], 10)

        # Row and column
        a1 = isopy.array(dict(ru=[1], pd=[11], cd=[21]))
        v = a1[0]
        a = isopy.array(dict(ru=1, pd=11, cd=21))

        with pytest.raises(IndexError):
            a['pd', 1] = 200

        with pytest.raises(IndexError):
            v['pd', 1] = 200

        with pytest.raises(IndexError):
            a['pd', [1]] = 12

        with pytest.raises(IndexError):
            v['pd', [1]] = 12

        with pytest.raises(IndexError):
            a['pd', ::2] = 12

        with pytest.raises(IndexError):
            v['pd', ::2] = 12

        a['pd', :] = 100
        assert np.array_equal(a['ru'], 1)
        assert np.array_equal(a['pd'], 100)
        assert np.array_equal(a['cd'], 21)

        v['pd', :] = 100
        assert np.array_equal(v['ru'], 1)
        assert np.array_equal(v['pd'], 100)
        assert np.array_equal(v['cd'], 21)

        with pytest.raises(IndexError):
            a[['pd'], 1] = 12

        with pytest.raises(IndexError):
            v[['pd'], 1] = 12

        with pytest.raises(IndexError):
            a[['pd'], ::2] = 12

        with pytest.raises(IndexError):
            v[['pd'], ::2] = 12

        a[['pd'], :] = 12
        assert np.array_equal(a['ru'], 1)
        assert np.array_equal(a['pd'], 12)
        assert np.array_equal(a['cd'], 21)

        v[['pd'], :] = 12
        assert np.array_equal(v['ru'], 1)
        assert np.array_equal(v['pd'], 12)
        assert np.array_equal(v['cd'], 21)

        with pytest.raises(IndexError):
            a[:, 1]  = 12

        with pytest.raises(IndexError):
            v[:, 1]  = 12

        with pytest.raises(IndexError):
            a[:1, 1] = 1

        with pytest.raises(IndexError):
            v[:1, 1] = 1

        a1 = isopy.array(dict(ru=[1], pd=[11], cd=[21]))
        v = a1[0]
        a = isopy.array(dict(ru=1, pd=11, cd=21))

        a[a >= 7] = 10
        assert np.array_equal(a['ru'], 1)
        assert np.array_equal(a['pd'], 10)
        assert np.array_equal(a['cd'], 10)

        with pytest.raises(TypeError):
            v[a >= 7] = 10

        a[a[['pd']] < 12] = 2
        assert np.array_equal(a['ru'], 1)
        assert np.array_equal(a['pd'], 2)
        assert np.array_equal(a['cd'], 10)

        with pytest.raises(TypeError):
            v[v[['pd']] < 12] = 2

    def test_setitem_0dim_keyed(self):
        b = isopy.array(dict(ru=4, cd=24)).default(100)
        c = dict(ru=7, pd=17, cd=27)
        d = isopy.asrefval(dict(ru=8, pd=18), default_value=200)

        # ndarray
        a1 = isopy.array(dict(ru=[1], pd=[11], cd=[21]))
        v = a1[0]
        a = isopy.array(dict(ru=1, pd=11, cd=21))

        # Key
        a['pd'] = c
        assert np.array_equal(a['ru'], 1)
        assert np.array_equal(a['pd'], 17)
        assert np.array_equal(a['cd'], 21)

        v['pd'] = c
        assert np.array_equal(v['ru'], 1)
        assert np.array_equal(v['pd'], 17)
        assert np.array_equal(v['cd'], 21)

        a['ru', 'cd'] = d
        assert np.array_equal(a['ru'], 8)
        assert np.array_equal(a['pd'], 17)
        assert np.array_equal(a['cd'], 200)

        v['ru', 'cd'] = d
        assert np.array_equal(v['ru'], 8)
        assert np.array_equal(v['pd'], 17)
        assert np.array_equal(v['cd'], 200)

        a[('ru', 'pd')] = b
        assert np.array_equal(a['ru'], 4)
        assert np.array_equal(a['pd'], 100)
        assert np.array_equal(a['cd'], 200)

        v[('ru', 'pd')] = b
        assert np.array_equal(v['ru'], 4)
        assert np.array_equal(v['pd'], 100)
        assert np.array_equal(v['cd'], 200)

        # Row slice
        a1 = isopy.array(dict(ru=[1], pd=[11], cd=[21]))
        v = a1[0]
        a = isopy.array(dict(ru=1, pd=11, cd=21))

        a[:] = b
        assert np.array_equal(a['ru'], 4)
        assert np.array_equal(a['pd'], 100)
        assert np.array_equal(a['cd'], 24)

        v[:] = b
        assert np.array_equal(v['ru'], 4)
        assert np.array_equal(v['pd'], 100)
        assert np.array_equal(v['cd'], 24)

        # Row and column
        a1 = isopy.array(dict(ru=[1], pd=[11], cd=[21]))
        v = a1[0]
        a = isopy.array(dict(ru=1, pd=11, cd=21))

        a['pd', :] = c
        assert np.array_equal(a['ru'], 1)
        assert np.array_equal(a['pd'], 17)
        assert np.array_equal(a['cd'], 21)

        v['pd', :] = c
        assert np.array_equal(v['ru'], 1)
        assert np.array_equal(v['pd'], 17)
        assert np.array_equal(v['cd'], 21)

        a[['pd'], 0] = d
        assert np.array_equal(a['ru'], 1)
        assert np.array_equal(a['pd'], 18)
        assert np.array_equal(a['cd'], 21)

        v[['pd'], 0] = d
        assert np.array_equal(v['ru'], 1)
        assert np.array_equal(v['pd'], 18)
        assert np.array_equal(v['cd'], 21)

        a1 = isopy.array(dict(ru=[1], pd=[11], cd=[21]))
        v = a1[0]
        a = isopy.array(dict(ru=1, pd=11, cd=21))

        a[a >= 7] = b
        assert np.array_equal(a['ru'], 1)
        assert np.array_equal(a['pd'], 100)
        assert np.array_equal(a['cd'], 24)

        with pytest.raises(TypeError):
            v[a >= 7] = b

        a[a[['pd']] < 120] = c
        assert np.array_equal(a['ru'], 1)
        assert np.array_equal(a['pd'], 17)
        assert np.array_equal(a['cd'], 24)

        with pytest.raises(TypeError):
            v[v[['pd']] < 12] = c

    def test_parse_index(self):
        # Tests the few eventualities that is not tested in the get set tests

        a = isopy.array(dict(ru=[1, 2, 3], pd=[11, 12, 13], cd=[21, 22, 23]))

        with pytest.raises(IndexError):
            b = a[(1,)]

        b= a[(['pd'],)]
        assert b.keys == ['pd']

        with pytest.raises(IndexError):
            a[['Pd'],['Pd']]

        with pytest.raises(IndexError):
            a[[1],[1]]

        with pytest.raises(IndexError):
            a['pd', 1, 1]

    def test_copy(self):
        data = np.array([[i for i in range(1, 7)], [i ** 2 for i in range(1, 7)]])

        array = isopy.array(data, '102pd 104pd 105pd 106pd 108pd 110pd'.split())
        assert_array_equal_array(array.copy(), array)
        assert array.copy() is not array

    def test_asarray(self):
        # size 1, 0-dim
        array = isopy.random(-1, (1, 0.1), 'ru pd cd'.split(), seed=46)

        with pytest.raises(ValueError):
            isopy.asarray(array, ndim=2)

        array1 = isopy.asarray(array)
        assert array1 is array

        array1 = isopy.asarray(array, ndim=0)
        assert array1.ndim == 0
        assert array1 is array

        array1 = isopy.asarray(array, ndim=-1)
        assert array1.ndim == 0
        assert array1 is array

        array1 = isopy.asarray(array, ndim=1)
        assert array1.ndim == 1
        assert array1.base is array

        # size 1, 1-dim
        array = isopy.random(1, (1, 0.1), 'ru pd cd'.split(), seed=46)

        array1 = isopy.asarray(array)
        assert array1 is array

        array1 = isopy.asarray(array, ndim=0)
        assert array1.ndim == 0
        assert array1.base is array

        array1 = isopy.asarray(array, ndim=-1)
        assert array1.ndim == 0
        assert array1.base is array

        array1 = isopy.asarray(array, ndim=1)
        assert array1.ndim == 1
        assert array1 is array

        # size > 1, 1-dim
        array = isopy.random(2, (1, 0.1), 'ru pd cd'.split(), seed=46)

        array1 = isopy.asarray(array)
        assert array1 is array

        with pytest.raises(ValueError):
            array1 = isopy.asarray(array, ndim=0)

        array1 = isopy.asarray(array, ndim=-1)
        assert array1.ndim == 1
        assert array1 is array

        array1 = isopy.asarray(array, ndim=1)
        assert array1.ndim == 1
        assert array1 is array

    def test_asanyarray1(self):
        # size 1, 0-dim
        array = isopy.random(-1, (1, 0.1), 'ru pd cd'.split(), seed=46)

        with pytest.raises(ValueError):
            isopy.asanyarray(array, ndim=2)

        array1 = isopy.asanyarray(array)
        assert array1 is array

        array1 = isopy.asanyarray(array, ndim=0)
        assert array1.ndim == 0
        assert array1 is array

        array1 = isopy.asanyarray(array, ndim=-1)
        assert array1.ndim == 0
        assert array1 is array

        array1 = isopy.asanyarray(array, ndim=1)
        assert array1.ndim == 1
        assert array1.base is array

        # size 1, 1-dim
        array = isopy.random(1, (1, 0.1), 'ru pd cd'.split(), seed=46)

        array1 = isopy.asanyarray(array)
        assert array1 is array

        array1 = isopy.asanyarray(array, ndim=0)
        assert array1.ndim == 0
        assert array1.base is array

        array1 = isopy.asanyarray(array, ndim=-1)
        assert array1.ndim == 0
        assert array1.base is array

        array1 = isopy.asanyarray(array, ndim=1)
        assert array1.ndim == 1
        assert array1 is array

        # size > 1, 1-dim
        array = isopy.random(2, (1, 0.1), 'ru pd cd'.split(), seed=46)

        array1 = isopy.asanyarray(array)
        assert array1 is array

        with pytest.raises(ValueError):
            array1 = isopy.asanyarray(array, ndim=0)

        array1 = isopy.asanyarray(array, ndim=-1)
        assert array1.ndim == 1
        assert array1 is array

        array1 = isopy.asanyarray(array, ndim=1)
        assert array1.ndim == 1
        assert array1 is array

    def test_asanyarray2(self):
        # size 1, 0-dim
        array = isopy.random(-1, (1, 0.1), 'ru pd cd'.split(), seed=46)
        dictionary = array.to_dict()

        with pytest.raises(ValueError):
            isopy.asanyarray(dictionary, ndim=2)

        array1 = isopy.asanyarray(dictionary)
        assert_array_equal_array(array1, array)

        array1 = isopy.asanyarray(dictionary, ndim=0)
        assert array1.ndim == 0
        assert_array_equal_array(array1, array)

        array1 = isopy.asanyarray(dictionary, ndim=-1)
        assert array1.ndim == 0
        assert_array_equal_array(array1, array)

        array1 = isopy.asanyarray(dictionary, ndim=1)
        assert array1.ndim == 1
        assert_array_equal_array(array1[0], array)

        # size 1, 1-dim
        array = isopy.random(1, (1, 0.1), 'ru pd cd'.split(), seed=46)
        dictionary = array.to_dict()

        array1 = isopy.asanyarray(dictionary)
        assert_array_equal_array(array1, array)

        array1 = isopy.asanyarray(dictionary, ndim=0)
        assert array1.ndim == 0
        assert_array_equal_array(array1, array[0])

        array1 = isopy.asanyarray(dictionary, ndim=-1)
        assert array1.ndim == 0
        assert_array_equal_array(array1, array[0])

        array1 = isopy.asanyarray(dictionary, ndim=1)
        assert array1.ndim == 1
        assert_array_equal_array(array1, array)

        # size > 1, 1-dim
        array = isopy.random(2, (1, 0.1), 'ru pd cd'.split(), seed=46)
        dictionary = array.to_dict()

        array1 = isopy.asanyarray(dictionary)
        assert_array_equal_array(array1, array)

        with pytest.raises(ValueError):
            array1 = isopy.asanyarray(dictionary, ndim=0)

        array1 = isopy.asanyarray(dictionary, ndim=-1)
        assert array1.ndim == 1
        assert_array_equal_array(array1, array)

        array1 = isopy.asanyarray(dictionary, ndim=1)
        assert array1.ndim == 1
        assert_array_equal_array(array1, array)

    def test_asanyarray3(self):
        # size 1, 0-dim
        array1 = 1.0
        array2 = np.asarray(array1)

        with pytest.raises(ValueError):
            isopy.asanyarray(array1, ndim = -2)
        with pytest.raises(ValueError):
            isopy.asanyarray(array1, ndim = 3)

        result1 = isopy.asanyarray(array1)
        result2 = isopy.asanyarray(array2)
        assert result1.ndim == 0
        assert result2.ndim == 0
        np.testing.assert_allclose(result1, array2)
        assert result2 is array2

        result1 = isopy.asanyarray(array1, ndim=0)
        result2 = isopy.asanyarray(array2, ndim=0)
        assert result1.ndim == 0
        assert result2.ndim == 0
        np.testing.assert_allclose(result1, array2)
        assert result2 is array2

        result1 = isopy.asanyarray(array1, ndim = -1)
        result2 = isopy.asanyarray(array2, ndim = -1)
        assert result1.ndim == 0
        assert result2.ndim == 0
        np.testing.assert_allclose(result1, array2)
        assert result2 is array2

        result1 = isopy.asanyarray(array1, ndim=1)
        result2 = isopy.asanyarray(array2, ndim=1)
        assert result1.ndim == 1
        assert result2.ndim == 1
        np.testing.assert_allclose(result1, array2)
        assert result2.base is array2

        result1 = isopy.asanyarray(array1, ndim=2)
        result2 = isopy.asanyarray(array2, ndim=2)
        assert result1.ndim == 2
        assert result2.ndim == 2
        np.testing.assert_allclose(result1, array2)
        assert result2.base is array2

        # size 1, 1-dim
        array1 = [1.0]
        array2 = np.asarray(array1)

        result1 = isopy.asanyarray(array1)
        result2 = isopy.asanyarray(array2)
        assert result1.ndim == 1
        assert result2.ndim == 1
        np.testing.assert_allclose(result1, array2)
        assert result2 is array2

        result1 = isopy.asanyarray(array1, ndim=0)
        result2 = isopy.asanyarray(array2, ndim=0)
        assert result1.ndim == 0
        assert result2.ndim == 0
        np.testing.assert_allclose(result1, array2)
        assert result2.base is array2

        result1 = isopy.asanyarray(array1, ndim=-1)
        result2 = isopy.asanyarray(array2, ndim=-1)
        assert result1.ndim == 0
        assert result2.ndim == 0
        np.testing.assert_allclose(result1, array2)
        assert result2.base is array2

        result1 = isopy.asanyarray(array1, ndim=1)
        result2 = isopy.asanyarray(array2, ndim=1)
        assert result1.ndim == 1
        assert result2.ndim == 1
        np.testing.assert_allclose(result1, array2)
        assert result2 is array2

        result1 = isopy.asanyarray(array1, ndim=2)
        result2 = isopy.asanyarray(array2, ndim=2)
        assert result1.ndim == 2
        assert result2.ndim == 2
        np.testing.assert_allclose(result1, [array1])
        assert result2.base is array2

        # size >1, 1-dim
        array1 = [1, 2, 3]
        array2 = np.asarray(array1)

        result1 = isopy.asanyarray(array1)
        result2 = isopy.asanyarray(array2)
        assert result1.ndim == 1
        assert result2.ndim == 1
        np.testing.assert_allclose(result1, array2)
        assert result2 is array2

        with pytest.raises(ValueError):
            isopy.asanyarray(array1, ndim=0)
        with pytest.raises(ValueError):
            isopy.asanyarray(array2, ndim=0)

        result1 = isopy.asanyarray(array1, ndim=-1)
        result2 = isopy.asanyarray(array2, ndim=-1)
        assert result1.ndim == 1
        assert result2.ndim == 1
        np.testing.assert_allclose(result1, array2)
        assert result2 is array2

        result1 = isopy.asanyarray(array1, ndim=1)
        result2 = isopy.asanyarray(array2, ndim=1)
        assert result1.ndim == 1
        assert result2.ndim == 1
        np.testing.assert_allclose(result1, array2)
        assert result2 is array2

        result1 = isopy.asanyarray(array1, ndim=2)
        result2 = isopy.asanyarray(array2, ndim=2)
        assert result1.ndim == 2
        assert result2.ndim == 2
        np.testing.assert_allclose(result1, [array1])
        assert result2.base is array2

    def test_asanyarray4(self):
        array = [1, 2, 3, 4]

        result = isopy.asanyarray(array, dtype=np.int8)
        assert result.dtype == np.int8

        result = isopy.asanyarray(array, dtype=np.float64)
        assert result.dtype == np.float64

        result = isopy.asanyarray(array, dtype=str)
        assert result.dtype == '<U1'

        result = isopy.asanyarray(array, dtype=(np.float64, str))
        assert result.dtype == np.float64

        array = ['a', 'b', 'c', 'd']
        with pytest.raises(TypeError):
            isopy.asanyarray(array, dtype=np.int8)

        with pytest.raises(TypeError):
            isopy.asanyarray(array, dtype=np.float64)

        result = isopy.asanyarray(array, dtype=str)
        assert result.dtype == '<U1'

        result = isopy.asanyarray(array, dtype=(np.float64, str))
        assert result.dtype == '<U1'

    def test_get(self):
        a = isopy.array(dict(ru=[1, 2, 3], pd=[11, 12, 13], cd=[21, 22, 23]))
        np.testing.assert_array_equal(a.get('ru'), a['ru'])
        np.testing.assert_array_equal(a.get('ag'), [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(a.get('ag', 10), [10, 10, 10])
        np.testing.assert_array_equal(a.get('ag', [20]), [20, 20, 20])
        np.testing.assert_array_equal(a.get('ag', [10, 20, 30]), [10, 20, 30])
        np.testing.assert_array_equal(a.get(isopy.keystring('ru', flavour='general')), [np.nan, np.nan, np.nan])

        a = isopy.array(dict(ru=[4], pd=[14], cd=[24]))
        np.testing.assert_array_equal(a.get('ru'), a['ru'])
        np.testing.assert_array_equal(a.get('ag'), [np.nan])
        np.testing.assert_array_equal(a.get('ag', 10), [10])
        np.testing.assert_array_equal(a.get('ag', [20]), [20])
        np.testing.assert_array_equal(a.get(isopy.keystring('ru', flavour='general')), [np.nan])

        a = a[0]
        np.testing.assert_array_equal(a.get('ru'), a['ru'])
        np.testing.assert_array_equal(a.get('ag'), [np.nan])
        np.testing.assert_array_equal(a.get('ag', 10), 10)
        np.testing.assert_array_equal(a.get('ag', [20]), 20)
        np.testing.assert_array_equal(a.get(isopy.keystring('ru', flavour='general')), np.nan)

        a = isopy.array(dict(ru=5, pd=15, cd=25), ndim=0)
        np.testing.assert_array_equal(a.get('ru'), a['ru'])
        np.testing.assert_array_equal(a.get('ag'), np.nan)
        np.testing.assert_array_equal(a.get('ag', 10), 10)
        np.testing.assert_array_equal(a.get('ag', [20]), 20)
        np.testing.assert_array_equal(a.get(isopy.keystring('ru', flavour='general')), np.nan)

        get = a.get('rh', 10)
        assert get.dtype == np.float64

        get = a.get('rh', np.array(10, dtype=np.int8))
        assert get.dtype == np.int8

        get = a.get('rh', 'a')
        assert get.dtype == np.array('a').dtype

        b = isopy.array(dict(rh=10, pd=20, cd=30))
        np.testing.assert_array_equal(a.get('rh', b), b['rh'])
        np.testing.assert_array_equal(a.get('pd', b), a['pd'])

        b = b.to_refval(ratio_function='divide')

        np.testing.assert_array_equal(a.get('rh', b), b['rh'])
        np.testing.assert_array_equal(a.get('pd', b), a['pd'])
        np.testing.assert_array_equal(a.get('rh/pd', b), b['rh']/b['pd'])


# TODO to_array & to_refval args
class Test_ToMixin:
    def test_100_1_arv(self):
        a = isopy.random(20, (1, 0.1), 'ru pd cd'.split(), seed=46)

        tovalue = a.to_array()
        assert isopy.isarray(tovalue)
        assert_array_equal_array(a, isopy.array(tovalue))
        assert tovalue is not self

        tovalue = a.to_list()
        assert type(tovalue) is list
        assert False not in [type(v) is list for v in tovalue]
        assert_array_equal_array(a, isopy.array(tovalue, a.keys))

        tovalue = a.to_dict()
        assert type(tovalue) is dict
        assert False not in [type(v) is list for v in tovalue.values()]
        assert False not in [type(k) is str for k in tovalue.keys()]
        assert_array_equal_array(a, isopy.array(tovalue))

        tovalue = a.to_ndarray()
        assert type(tovalue) is np.ndarray
        assert a.keys == tovalue.dtype.names
        for key in a.keys:
            np.testing.assert_allclose(a[key], tovalue[str(key)])

        tovalue = a[0].to_ndarray()
        assert type(tovalue) is np.ndarray
        assert a[0].keys == tovalue.dtype.names
        for key in a.keys:
            np.testing.assert_allclose(a[key][0], tovalue[str(key)])

        tovalue = a.to_dataframe()
        assert type(tovalue) is pd.DataFrame
        assert a.keys == tovalue.columns
        for key in a.keys:
            np.testing.assert_allclose(a[key], tovalue[str(key)])

        core.pandas = None
        with pytest.raises(TypeError):
            a.to_dataframe()

        core.pandas = pd

        d = a.to_refval()

        assert type(d) is core.RefValDict
        assert d.keys == a.keys
        assert d.size == a.size
        assert d.ndim == a.ndim
        assert_array_equal_array(a, isopy.array(d))

        tovalue = d.to_refval()
        assert type(d) is core.RefValDict
        assert d.keys == a.keys
        assert d.size == a.size
        assert d.ndim == a.ndim
        assert_array_equal_array(a, isopy.array(d))
        assert tovalue is not d

        tovalue = d.to_array()
        assert isinstance(tovalue, core.IsopyArray)
        assert_array_equal_array(a, tovalue)

        tovalue = d.to_list()
        assert type(tovalue) is list
        assert False not in [type(v) is list for v in tovalue]
        assert_array_equal_array(a, isopy.array(tovalue, d.keys))

        tovalue = d.to_dict()
        assert type(tovalue) is dict
        assert False not in [type(v) is list for v in tovalue.values()]
        assert False not in [type(k) is str for k in tovalue.keys()]
        assert_array_equal_array(a, isopy.array(tovalue))

        tovalue = d.to_ndarray()
        assert type(tovalue) is np.ndarray
        assert d.keys == tovalue.dtype.names
        for key in d.keys:
            np.testing.assert_allclose(d[key], tovalue[str(key)])

        tovalue = d.to_dataframe()
        assert type(tovalue) is pd.DataFrame
        assert d.keys == tovalue.columns
        for key in d.keys:
            np.testing.assert_allclose(a[key], tovalue[str(key)])

    def test_1_1_arv(self):
        a = isopy.random(1, (1, 0.1), 'ru pd cd'.split(), seed=46)

        tovalue = a.to_array()
        assert isopy.isarray(tovalue)
        assert_array_equal_array(a, isopy.array(tovalue))
        assert tovalue is not self

        tovalue = a.to_list()
        assert type(tovalue) is list
        assert False not in [type(v) is list for v in tovalue]
        assert_array_equal_array(a, isopy.array(tovalue, a.keys))

        tovalue = a.to_dict()
        assert type(tovalue) is dict
        assert False not in [type(v) is list for v in tovalue.values()]
        assert False not in [type(k) is str for k in tovalue.keys()]
        assert_array_equal_array(a, isopy.array(tovalue))

        tovalue = a.to_ndarray()
        assert type(tovalue) is np.ndarray
        assert a.keys == tovalue.dtype.names
        for key in a.keys:
            np.testing.assert_allclose(a[key], tovalue[str(key)])

        tovalue = a.to_dataframe()
        assert type(tovalue) is pd.DataFrame
        assert a.keys == tovalue.columns
        for key in a.keys:
            np.testing.assert_allclose(a[key], tovalue[str(key)])

        d = a.to_refval()

        assert type(d) is core.RefValDict
        assert d.keys == a.keys
        assert d.size == a.size
        assert d.ndim != a.ndim
        assert d.ndim == 0
        assert_array_equal_array(a[0], isopy.array(d))

        tovalue = d.to_refval()
        assert type(d) is core.RefValDict
        assert d.keys == a.keys
        assert d.size == a.size
        assert d.ndim == 0
        assert_array_equal_array(a[0], isopy.array(d))
        assert tovalue is not d

        tovalue = d.to_array()
        assert isinstance(tovalue, core.IsopyArray)
        assert_array_equal_array(a[0], tovalue)

        tovalue = d.to_list()
        assert type(tovalue) is list
        assert_array_equal_array(a[0], isopy.array(tovalue, d.keys))

        tovalue = d.to_dict()
        assert type(tovalue) is dict
        assert False not in [type(k) is str for k in tovalue.keys()]
        assert_array_equal_array(a[0], isopy.array(tovalue))

        tovalue = d.to_ndarray()
        assert type(tovalue) is np.ndarray
        assert d.keys == tovalue.dtype.names
        for key in d.keys:
            np.testing.assert_allclose(d[key], tovalue[str(key)])

        tovalue = d.to_dataframe()
        assert type(tovalue) is pd.DataFrame
        assert d.keys == tovalue.columns
        for key in d.keys:
            np.testing.assert_allclose(a[key], tovalue[str(key)])

    def test_1_0_arv(self):
        a = isopy.random(None, (1, 0.1), 'ru pd cd'.split(), seed=46)

        tovalue = a.to_array()
        assert isopy.isarray(tovalue)
        assert_array_equal_array(a, isopy.array(tovalue))
        assert tovalue is not self

        tovalue = a.to_list()
        assert type(tovalue) is list
        assert_array_equal_array(a, isopy.array(tovalue, a.keys))

        tovalue = a.to_dict()
        assert type(tovalue) is dict
        assert False not in [type(k) is str for k in tovalue.keys()]
        assert_array_equal_array(a, isopy.array(tovalue))

        tovalue = a.to_ndarray()
        assert type(tovalue) is np.ndarray
        assert a.keys == tovalue.dtype.names
        for key in a.keys:
            np.testing.assert_allclose(a[key], tovalue[str(key)])

        tovalue = a.to_dataframe()
        assert type(tovalue) is pd.DataFrame
        assert a.keys == tovalue.columns
        for key in a.keys:
            np.testing.assert_allclose(a[key], tovalue[str(key)])

        d = a.to_refval()

        assert type(d) is core.RefValDict
        assert d.keys == a.keys
        assert d.size == a.size
        assert d.ndim == a.ndim
        assert_array_equal_array(a, isopy.array(d))

        tovalue = d.to_refval()
        assert type(d) is core.RefValDict
        assert d.keys == a.keys
        assert d.size == a.size
        assert d.ndim == a.ndim
        assert_array_equal_array(a, isopy.array(d))
        assert tovalue is not d

        tovalue = d.to_array()
        assert isinstance(tovalue, core.IsopyArray)
        assert_array_equal_array(a, tovalue)

        tovalue = d.to_list()
        assert type(tovalue) is list
        assert_array_equal_array(a, isopy.array(tovalue, d.keys))

        tovalue = d.to_dict()
        assert type(tovalue) is dict
        assert False not in [type(k) is str for k in tovalue.keys()]
        assert_array_equal_array(a, isopy.array(tovalue))

        tovalue = d.to_ndarray()
        assert type(tovalue) is np.ndarray
        assert d.keys == tovalue.dtype.names
        for key in d.keys:
            np.testing.assert_allclose(d[key], tovalue[str(key)])

        tovalue = d.to_dataframe()
        assert type(tovalue) is pd.DataFrame
        assert d.keys == tovalue.columns
        for key in d.keys:
            np.testing.assert_allclose(a[key], tovalue[str(key)])

    def test_1_0_arv_kf(self):
        a = isopy.array([1,2,3], 'ru pd cd')
        b = isopy.array([np.nan, 2, np.nan], 'rh pd ag')
        keys = 'rh pd ag'.split()

        tovalue = a.to_array(keys)
        assert isopy.isarray(tovalue)
        assert_array_equal_array(b, isopy.array(tovalue))
        assert tovalue is not self

        tovalue = a.to_list(keys)
        assert type(tovalue) is list
        assert_array_equal_array(b, isopy.array(tovalue, b.keys))

        tovalue = a.to_dict(keys)
        assert type(tovalue) is dict
        assert False not in [type(k) is str for k in tovalue.keys()]
        assert_array_equal_array(b, isopy.array(tovalue))

        tovalue = a.to_ndarray(keys)
        assert type(tovalue) is np.ndarray
        assert b.keys == tovalue.dtype.names
        for key in b.keys:
            np.testing.assert_allclose(b[key], tovalue[str(key)])

        tovalue = a.to_dataframe(keys)
        assert type(tovalue) is pd.DataFrame
        assert b.keys == tovalue.columns
        for key in b.keys:
            np.testing.assert_allclose(b[key], tovalue[str(key)])

        # Default
        b = isopy.array([10, 2, 10], 'rh pd ag')

        tovalue = a.to_array(keys, 10)
        assert isopy.isarray(tovalue)
        assert_array_equal_array(b, isopy.array(tovalue))
        assert tovalue is not self

        tovalue = a.to_list(keys, 10)
        assert type(tovalue) is list
        assert_array_equal_array(b, isopy.array(tovalue, b.keys))

        tovalue = a.to_dict(keys, 10)
        assert type(tovalue) is dict
        assert False not in [type(k) is str for k in tovalue.keys()]
        assert_array_equal_array(b, isopy.array(tovalue))

        tovalue = a.to_ndarray(keys, 10)
        assert type(tovalue) is np.ndarray
        assert b.keys == tovalue.dtype.names
        for key in b.keys:
            np.testing.assert_allclose(b[key], tovalue[str(key)])

        tovalue = a.to_dataframe(keys, 10)
        assert type(tovalue) is pd.DataFrame
        assert b.keys == tovalue.columns
        for key in b.keys:
            np.testing.assert_allclose(b[key], tovalue[str(key)])

        # Default a
        a = isopy.array([1, 2, 3], 'ru pd cd').default(10)

        tovalue = a.to_array(keys)
        assert isopy.isarray(tovalue)
        assert_array_equal_array(b, isopy.array(tovalue))
        assert tovalue is not self

        tovalue = a.to_list(keys)
        assert type(tovalue) is list
        assert_array_equal_array(b, isopy.array(tovalue, b.keys))

        tovalue = a.to_dict(keys)
        assert type(tovalue) is dict
        assert False not in [type(k) is str for k in tovalue.keys()]
        assert_array_equal_array(b, isopy.array(tovalue))

        tovalue = a.to_ndarray(keys)
        assert type(tovalue) is np.ndarray
        assert b.keys == tovalue.dtype.names
        for key in b.keys:
            np.testing.assert_allclose(b[key], tovalue[str(key)])

        tovalue = a.to_dataframe(keys)
        assert type(tovalue) is pd.DataFrame
        assert b.keys == tovalue.columns
        for key in b.keys:
            np.testing.assert_allclose(b[key], tovalue[str(key)])

        # Key filter
        a = isopy.array([1, 2, 3], 'ru pd cd')
        b = isopy.array([np.nan, 2], 'rh pd')
        keys = 'rh pd ag'.split()
        key_eq = 'ru rh pd'.split()

        tovalue = a.to_array(keys, key_eq=key_eq)
        assert isopy.isarray(tovalue)
        assert_array_equal_array(b, isopy.array(tovalue))
        assert tovalue is not self

        tovalue = a.to_list(keys, key_eq=key_eq)
        assert type(tovalue) is list
        assert_array_equal_array(b, isopy.array(tovalue, b.keys))

        tovalue = a.to_dict(keys, key_eq=key_eq)
        assert type(tovalue) is dict
        assert False not in [type(k) is str for k in tovalue.keys()]
        assert_array_equal_array(b, isopy.array(tovalue))

        tovalue = a.to_ndarray(keys, key_eq=key_eq)
        assert type(tovalue) is np.ndarray
        assert b.keys == tovalue.dtype.names
        for key in b.keys:
            np.testing.assert_allclose(b[key], tovalue[str(key)])

        tovalue = a.to_dataframe(keys, key_eq=key_eq)
        assert type(tovalue) is pd.DataFrame
        assert b.keys == tovalue.columns
        for key in b.keys:
            np.testing.assert_allclose(b[key], tovalue[str(key)])


    def test_100_1_text(self):
        # size 100, 1-dim
        a = isopy.random(20, (1, 0.1), 'ru pd cd'.split(), seed = 46, dtype = [np.float64, np.float64, np.int8])

        text = repr(a)
        assert type(text) is core.TableStr
        assert core.hashstr(text) == 'fc675acb96bc689d1858e474a3aa9482'
        assert core.hashstr(text._repr_html_()) == 'c3c446a2f8c29d6199009ea31338414d'

        text = str(a)
        assert type(text) is core.TableStr
        assert core.hashstr(text) == '515bb1377d76b86258c7117042f91ce7'
        assert core.hashstr(text._repr_html_()) == '51e4811855a932a400c9183b9dbf8697'

        text = a.tabulate()
        assert type(text) is core.TableStr
        assert core.hashstr(text) == '515bb1377d76b86258c7117042f91ce7'
        assert core.hashstr(text._repr_html_()) == '51e4811855a932a400c9183b9dbf8697'

        text = a.tabulate(include_row = True, include_dtype=True, nrows=12)
        assert type(text) is core.TableStr
        assert core.hashstr(text) == '673c2ede15379813eef450deb6c5349d'
        assert core.hashstr(text._repr_html_()) == '51988f1ae44a86b0b6db3289555e6b60'

        text = a.tabulate(row_names = [f'row {i+1}' for i in range(20)], floatfmt='.2f', intfmt='.1f')
        assert type(text) is core.TableStr
        assert core.hashstr(text) == '116d4ba627131b1c14a4e6ef9f930334'
        assert core.hashstr(text._repr_html_()) == '1b0f1f343650001cc61a8b54cccc5d45'

        text = a.tabulate(row_names=[f'row {i + 1}' for i in range(20)], nrows = 12,
                          floatfmt='f{:.2f}', intfmt='i{:.1f}')
        assert type(text) is core.TableStr
        assert core.hashstr(text) == '0144869d470752578acd85316a292d39'
        assert core.hashstr(text._repr_html_()) == '5ce0d4860254e5d7d542b0599474fefe'

        text = a.tabulate('simple')
        assert type(text) is core.TableStr
        assert core.hashstr(text) == '515bb1377d76b86258c7117042f91ce7'

        text = a.tabulate('markdown')
        assert type(text) is core.TableStr
        assert core.hashstr(text) == '6d23aa941a8d3ca6261472ed6b7ed77e'
        assert core.hashstr(text._repr_markdown_()) == '6d23aa941a8d3ca6261472ed6b7ed77e'

        text = a.tabulate('markdown', include_objinfo = True)
        assert type(text) is core.TableStr
        assert core.hashstr(text) == '8b9df21a9c83437c743b8b7feaf74a34'
        assert core.hashstr(text._repr_markdown_()) == '8b9df21a9c83437c743b8b7feaf74a34'

        text = a.tabulate('latex')
        assert type(text) is core.TableStr
        assert core.hashstr(text) == '8b8273158a3e6bbe08c4521543c9d02f'
        assert core.hashstr(text._repr_latex_()) == '8b8273158a3e6bbe08c4521543c9d02f'

        text = a.tabulate('latex', include_objinfo = True)
        assert type(text) is core.TableStr
        assert core.hashstr(text) == '0725d5994f115314c09daf16f3c07b69'
        assert core.hashstr(text._repr_latex_()) == '0725d5994f115314c09daf16f3c07b69'

        text = a.tabulate('html')
        assert type(text) is core.TableStr
        assert core.hashstr(text) == '150b21727d750ecbfb7aca491c51ec1c'
        assert core.hashstr(text._repr_html_()) == '150b21727d750ecbfb7aca491c51ec1c'

        with pytest.raises(ValueError):
            a.tabulate(row_names=[f'row {i + 1}' for i in range(100)])

        with pytest.raises(ValueError):
            a.tabulate(row_names='row 0')

        # As it used the same function we do not need to redo all the tests
        d = a.to_refval()

        text = repr(d)
        assert type(text) is core.TableStr
        assert core.hashstr(text) == '93380e8bd348b98d29a9a9d5fa28a5e2'
        assert core.hashstr(text._repr_html_()) == 'd6bd042e8230f8e9ecbd55fff7da31ea'

        text = str(d)
        assert type(text) is core.TableStr
        assert core.hashstr(text) == 'ca6ca962a113a357cafed7b01cef5f4f'
        assert core.hashstr(text._repr_html_()) == '7959300b42c5ba87937ea1e340a6e407'

    def test_1_1_text(self):
        a = isopy.random(1, (1, 0.1), 'ru pd cd'.split(), seed=46, dtype = [np.float64, np.float64, np.int8])

        text = repr(a)
        assert type(text) is core.TableStr
        assert core.hashstr(text) == '296739ff607d12ee6eb1a68cd11bc98d'
        assert core.hashstr(text._repr_html_()) == 'a3dc3e70930e0ddd2d1540cad4e5fb0a'

        text = str(a)
        assert type(text) is core.TableStr
        assert core.hashstr(text) == 'c113fc01aba641e1a1485e94fc09861b'
        assert core.hashstr(text._repr_html_()) == '16c2aa034c77fe9b2007fcd05c8bded5'

        text = a.tabulate(row_names=[f'row 1'])
        assert type(text) is core.TableStr
        assert core.hashstr(text) == 'cfa7f16e7e4509733ed67a239ade6ed8'
        assert core.hashstr(text._repr_html_()) == '53d16bb09f898508d89b61cf82450774'

        text = a.tabulate(row_names=f'row 1')
        assert type(text) is core.TableStr
        assert core.hashstr(text) == 'cfa7f16e7e4509733ed67a239ade6ed8'
        assert core.hashstr(text._repr_html_()) == '53d16bb09f898508d89b61cf82450774'

    def test_1_0_text(self):
        a = isopy.random(-1, (1, 0.1), 'ru pd cd'.split(), seed=46, dtype=[np.float64, np.float64, np.int8])
        assert a.ndim == 0

        text = repr(a)
        assert type(text) is core.TableStr
        assert core.hashstr(text) == '6fbd77821ad90fd29d3879636bc9ee99'
        assert core.hashstr(text._repr_html_()) == '4ec7e9c2bd1b2dcc4914037531306e1a'

        text = str(a)
        assert type(text) is core.TableStr
        assert core.hashstr(text) == 'c113fc01aba641e1a1485e94fc09861b'
        assert core.hashstr(text._repr_html_()) == '16c2aa034c77fe9b2007fcd05c8bded5'

        text = a.tabulate(row_names=[f'row 1'])
        assert type(text) is core.TableStr
        assert core.hashstr(text) == 'cfa7f16e7e4509733ed67a239ade6ed8'
        assert core.hashstr(text._repr_html_()) == '53d16bb09f898508d89b61cf82450774'

        text = a.tabulate(row_names=f'row 1')
        assert type(text) is core.TableStr
        assert core.hashstr(text) == 'cfa7f16e7e4509733ed67a239ade6ed8'
        assert core.hashstr(text._repr_html_()) == '53d16bb09f898508d89b61cf82450774'

        d = a.to_refval()

        text = repr(d)
        assert type(text) is core.TableStr
        assert core.hashstr(text) == '93a285fd104c52d5e32239e70d04c7e4'
        assert core.hashstr(text._repr_html_()) == 'dcdcbe279bf51aa89362abdecf04ec7c'

        text = str(d)
        assert type(text) is core.TableStr
        assert core.hashstr(text) == 'd1b7f665f88324a955271fd97468797b'
        assert core.hashstr(text._repr_html_()) == '60580bb64dd21619c2f4cea1bb19b5aa'

        a = isopy.random(1, (1, 0.1), 'ru pd cd'.split(), seed=46, dtype=[np.float64, np.float64, np.int8])
        a0 = a[0]

        text = repr(a0)
        assert type(text) is core.TableStr
        assert core.hashstr(text) == '83965745cb5e706177f4626716926fea'
        assert core.hashstr(text._repr_html_()) == '773b2663db153948bb8b567cdaa7f454'

        text = str(a0)
        assert type(text) is core.TableStr
        assert core.hashstr(text) == 'c113fc01aba641e1a1485e94fc09861b'
        assert core.hashstr(text._repr_html_()) == '16c2aa034c77fe9b2007fcd05c8bded5'

    def test_tablestr(self):
        a = isopy.random(20, (1, 0.1), 'ru pd cd'.split(), seed=46, dtype=[np.float64, np.float64, np.int8])

        text = str(a)
        assert type(text) is core.TableStr
        assert isinstance(text, str)
        assert str(text) == text
        assert repr(text) == text
        assert hash(text) == hash(str(text))

        copy = text.copy()
        assert copy is text
        assert copy == pyperclip.paste()

    def test_description(self):
        a = isopy.random(20, (1, 0.1), 'ru pd cd'.split(), seed=46)
        assert a._description_() == "IsopyNdarray(20, flavour='element', default_value=nan)"

        a = isopy.random(1, (1, 0.1), 'ru pd cd'.split(), seed=46)
        assert a._description_() == "IsopyNdarray(1, flavour='element', default_value=nan)"

        a = isopy.random(None, (1, 0.1), 'ru pd cd'.split(), seed=46).default(1)
        assert a._description_() == "IsopyNdarray(-1, flavour='element', default_value=1)"

        a = isopy.random(20, (1, 0.1), 'ru pd cd'.split(), seed=46)
        d = isopy.asrefval(a)
        assert d._description_() == ("RefValDict(20, readonly=False, key_flavour='any', "
                                     "ratio_function=None, molecule_functions=None, "
                                     f"default_value=nan)")

        a = isopy.random(2, (1, 0.1), 'ru pd cd'.split(), seed=46)
        d = isopy.core.RefValDict(a, default_value=[1,2], readonly = True, key_flavour='element|isotope')
        assert d._description_() == ("RefValDict(2, readonly=True, key_flavour='element|isotope', "
                                     "ratio_function=None, molecule_functions=None, "
                                     f"default_value=[1. 2.])")

        d = isopy.asrefval(a, ratio_function=np.divide, molecule_functions='abundance')
        assert d._description_() == ("RefValDict(2, readonly=False, key_flavour='any', "
                                     "ratio_function='divide', molecule_functions='abundance', "
                                     f"default_value=nan)")

        d = isopy.asrefval(a, ratio_function=np.add, molecule_functions='mass')
        assert d._description_() == ("RefValDict(2, readonly=False, key_flavour='any', "
                                     "ratio_function=<ufunc 'add'>, molecule_functions='mass', "
                                     f"default_value=nan)")

    def test_repr_html(self):
        a = isopy.random(20, (1, 0.1), 'ru pd cd'.split(), seed=46)
        d = a.to_refval()
        
        assert isopy.core.IPYTHON_REPR is True
        
        text = repr(a)
        assert text._repr_html_() == text._html
        assert a._repr_html_() == text._html

        text = repr(d)
        assert text._repr_html_() == text._html
        assert d._repr_html_() == text._html
        
        isopy.core.IPYTHON_REPR = False

        text = repr(a)
        assert text._repr_html_() is None
        assert a._repr_html_() is None

        text = repr(d)
        assert d._repr_html_() is None
        assert text._repr_html_() is None

        isopy.core.IPYTHON_REPR = True

class Test_EmptyArrays:
    def test_ones_zero_empty1(self):
        keys = 'ru pd cd'.split()
        keys2 = '101ru 105pd 111cd'.split()
        existing_array = isopy.array(ru = [1], pd = [2], cd = [3], dtype = [np.int8, np.float32, np.uint64])

        #creating array with size 1
        array = isopy.empty(None, keys)
        assert isinstance(array, core.IsopyArray)
        assert array.keys == keys
        assert array.ndim == 0
        assert array.size == 1
        assert array.nrows == -1

        array = isopy.empty(None, keys, ndim = 0)
        assert isinstance(array, core.IsopyArray)
        assert array.keys == keys
        assert array.ndim == 0
        assert array.size == 1
        assert array.nrows == -1

        array = isopy.empty(None, keys, ndim = -1)
        assert isinstance(array, core.IsopyArray)
        assert array.keys == keys
        assert array.ndim == 0
        assert array.size == 1
        assert array.nrows == -1

        array = isopy.empty(None, keys, ndim = 1)
        assert isinstance(array, core.IsopyArray)
        assert array.keys == keys
        assert array.ndim == 1
        assert array.size == 1
        assert array.nrows == 1

        array = isopy.empty(-1, keys)
        assert isinstance(array, core.IsopyArray)
        assert array.keys == keys
        assert array.ndim == 0
        assert array.size == 1
        assert array.nrows == -1

        array = isopy.empty(-1, keys, ndim = 0)
        assert isinstance(array, core.IsopyArray)
        assert array.keys == keys
        assert array.ndim == 0
        assert array.size == 1
        assert array.nrows == -1

        array = isopy.empty(-1, keys, ndim = -1)
        assert isinstance(array, core.IsopyArray)
        assert array.keys == keys
        assert array.ndim == 0
        assert array.size == 1
        assert array.nrows == -1

        array = isopy.empty(-1, keys, ndim = 1)
        assert isinstance(array, core.IsopyArray)
        assert array.keys == keys
        assert array.ndim == 1
        assert array.size == 1
        assert array.nrows == 1

        # creating array with size > 1
        array = isopy.empty(4, keys)
        assert isinstance(array, core.IsopyArray)
        assert array.keys == keys
        assert array.ndim == 1
        assert array.size == 4
        assert array.nrows == 4

        array = isopy.empty(4, keys, ndim =-1)
        assert isinstance(array, core.IsopyArray)
        assert array.keys == keys
        assert array.ndim == 1
        assert array.size == 4
        assert array.nrows == 4

        array = isopy.empty(4, keys, ndim = 1)
        assert isinstance(array, core.IsopyArray)
        assert array.keys == keys
        assert array.ndim == 1
        assert array.size == 4
        assert array.nrows == 4

        with pytest.raises(TypeError):
            array = isopy.empty(4, keys, ndim=-0)

        # dtype
        array = isopy.empty(4, keys, dtype=np.float64)
        assert isinstance(array, core.IsopyArray)
        assert array.ndim == 1
        assert array.size == 4
        for i in range(len(array.dtype)): assert array.dtype[i].type is np.float64

        array = isopy.empty(None, keys, dtype=np.int8)
        assert isinstance(array, core.IsopyArray)
        assert array.ndim == 0
        assert array.size == 1
        for i in range(len(array.dtype)): assert array.dtype[i].type is np.int8

        array = isopy.empty(-1, keys, dtype=np.int8)
        assert isinstance(array, core.IsopyArray)
        assert array.ndim == 0
        assert array.size == 1
        for i in range(len(array.dtype)): assert array.dtype[i].type is np.int8

        array = isopy.empty(tuple(), keys, dtype=np.int8)
        assert isinstance(array, core.IsopyArray)
        assert array.ndim == 0
        assert array.size == 1
        for i in range(len(array.dtype)): assert array.dtype[i].type is np.int8

        array = isopy.empty(1, keys, dtype=np.int8)
        assert isinstance(array, core.IsopyArray)
        assert array.ndim == 1
        assert array.size == 1
        for i in range(len(array.dtype)): assert array.dtype[i].type is np.int8

        with pytest.raises(ValueError):
            isopy.empty(-2, keys, dtype=np.int8)

        array = isopy.empty(1, keys, ndim = -1, dtype=np.int8)
        assert isinstance(array, core.IsopyArray)
        assert array.ndim == 0
        assert array.size == 1
        for i in range(len(array.dtype)): assert array.dtype[i].type is np.int8

        array = isopy.empty(-1, keys, ndim=1, dtype=np.int8)
        assert isinstance(array, core.IsopyArray)
        assert array.ndim == 1
        assert array.size == 1
        for i in range(len(array.dtype)): assert array.dtype[i].type is np.int8

        with pytest.raises(ValueError):
            isopy.empty(1, keys, ndim=-2, dtype=np.int8)

        with pytest.raises(ValueError):
            isopy.empty(1, keys, ndim=2, dtype=np.int8)

        array = isopy.empty(None, keys, ndim = 1, dtype=[np.int8, np.float32, np.uint64])
        assert isinstance(array, core.IsopyArray)
        assert array.ndim == 1
        assert array.size == 1
        assert array.dtype[0].type is np.int8
        assert array.dtype[1].type is np.float32
        assert array.dtype[2].type is np.uint64

        with pytest.raises(ValueError):
            isopy.empty(None, ndim=1, dtype=[np.int8, np.float32, np.uint64])

        with pytest.raises(ValueError):
            isopy.empty(None, keys, ndim=1, dtype=[np.int8, np.float32])

        # Test using an existing array for dtype
        array = isopy.empty(None, keys2, dtype=existing_array)
        assert isinstance(array, core.IsopyArray)
        assert array.ndim == 0
        assert array.size == 1
        assert array.keys == keys2
        assert array.dtype[0].type is np.int8
        assert array.dtype[1].type is np.float32
        assert array.dtype[2].type is np.uint64

        array = isopy.empty(None, keys2, ndim = 1, dtype=existing_array)
        assert isinstance(array, core.IsopyArray)
        assert array.ndim == 1
        assert array.size == 1
        assert array.keys == keys2
        assert array.dtype[0].type is np.int8
        assert array.dtype[1].type is np.float32
        assert array.dtype[2].type is np.uint64

        array = isopy.empty(None, dtype=existing_array)
        assert isinstance(array, core.IsopyArray)
        assert array.ndim == 0
        assert array.size == 1
        assert array.keys == keys
        assert array.dtype[0].type is np.int8
        assert array.dtype[1].type is np.float32
        assert array.dtype[2].type is np.uint64

        with pytest.raises(ValueError):
            isopy.empty(None, 'pd ru'.split(), dtype=existing_array)

        # Test using an existing array for keys
        array = isopy.empty(None, existing_array)
        assert isinstance(array, core.IsopyArray)
        assert array.ndim == 0
        assert array.size == 1
        assert array.keys == keys
        assert array.dtype[0].type is np.int8
        assert array.dtype[1].type is np.float32
        assert array.dtype[2].type is np.uint64

        array = isopy.empty(None, existing_array, ndim = 1)
        assert isinstance(array, core.IsopyArray)
        assert array.ndim == 1
        assert array.size == 1
        assert array.keys == keys
        assert array.dtype[0].type is np.int8
        assert array.dtype[1].type is np.float32
        assert array.dtype[2].type is np.uint64

        # Test using an existing array for rows
        array = isopy.empty(existing_array)
        assert isinstance(array, core.IsopyArray)
        assert array.ndim == 1
        assert array.size == 1
        assert array.keys == keys
        assert array.dtype[0].type is np.int8
        assert array.dtype[1].type is np.float32
        assert array.dtype[2].type is np.uint64

        array = isopy.empty(existing_array, ndim = 0)
        assert isinstance(array, core.IsopyArray)
        assert array.ndim == 0
        assert array.size == 1
        assert array.keys == keys
        assert array.dtype[0].type is np.int8
        assert array.dtype[1].type is np.float32
        assert array.dtype[2].type is np.uint64

        array = isopy.empty(existing_array, keys2)
        assert isinstance(array, core.IsopyArray)
        assert array.ndim == 1
        assert array.size == 1
        assert array.keys == keys2
        assert array.dtype[0].type is np.int8
        assert array.dtype[1].type is np.float32
        assert array.dtype[2].type is np.uint64

        # Check that ones only contains ones
        array1 = isopy.ones(4, ['Ru', 'Pd', 'Cd'])
        array2 = np.ones(4, [('Ru', np.float64), ('Pd', np.float64), ('Cd', np.float64)])
        assert isinstance(array1, core.IsopyArray)
        np.testing.assert_equal(array1, array2)

        # Check that zeros only contain zeros
        array1 = isopy.zeros(None, ['Ru', 'Pd', 'Cd'])
        array2 = np.zeros(None, [('Ru', np.float64), ('Pd', np.float64), ('Cd', np.float64)])
        assert isinstance(array1, core.IsopyArray)
        np.testing.assert_equal(array1, array2)

    def test_ones_zero_empty2(self):
        array = isopy.empty(None)
        assert not isinstance(array, core.IsopyArray)
        assert isinstance(array, np.ndarray)
        assert array.ndim == 0
        assert array.size == 1

        array = isopy.empty(1)
        assert not isinstance(array, core.IsopyArray)
        assert isinstance(array, np.ndarray)
        assert array.ndim == 1
        assert array.size == 1

        array = isopy.empty(10)
        assert isinstance(array, np.ndarray)
        assert not isinstance(array, core.IsopyArray)
        assert array.ndim == 1
        assert array.size == 10

        array = isopy.zeros(None)
        assert not isinstance(array, core.IsopyArray)
        assert isinstance(array, np.ndarray)
        assert array.ndim == 0
        assert array.size == 1
        np.testing.assert_allclose(array, 0)

        array = isopy.ones(10)
        assert not isinstance(array, core.IsopyArray)
        assert isinstance(array, np.ndarray)
        assert array.ndim == 1
        assert array.size == 10
        np.testing.assert_allclose(array, np.ones(10))

    def test_full1(self):
        keys = 'ru pd cd'.split()
        existing_array = isopy.array(ru=[1], pd=[2], cd=[3])

        array1 = isopy.full(None, 1.5, keys)
        array2 = np.full(None, 1.5, [('Ru', np.float64), ('Pd', np.float64), ('Cd', np.float64)])
        assert isinstance(array1, core.IsopyArray)
        np.testing.assert_equal(array1, array2)

        array1 = isopy.full(4, 1.5, keys)
        array2 = np.full(4, 1.5, [('Ru', np.float64), ('Pd', np.float64), ('Cd', np.float64)])
        assert isinstance(array1, core.IsopyArray)
        np.testing.assert_equal(array1, array2)

        array1 = isopy.full(existing_array, existing_array)
        assert isinstance(array1, core.IsopyArray)
        np.testing.assert_equal(array1, existing_array)

    def test_full2(self):
        array = isopy.full(None, 6)
        assert not isinstance(array, core.IsopyArray)
        assert isinstance(array, np.ndarray)
        assert array.ndim == 0
        assert array.size == 1
        np.testing.assert_allclose(array, 6)

        array = isopy.full(1, 6)
        assert not isinstance(array, core.IsopyArray)
        assert isinstance(array, np.ndarray)
        assert array.ndim == 1
        assert array.size == 1
        np.testing.assert_allclose(array, [6])

        array = isopy.full(10, [i for i in range(10)])
        assert isinstance(array, np.ndarray)
        assert not isinstance(array, core.IsopyArray)
        assert array.ndim == 1
        assert array.size == 10
        np.testing.assert_allclose(array, [i for i in range(10)])

    def test_random1(self):
        keys = 'ru pd cd'.split()

        rng = np.random.default_rng(seed=46)
        array = isopy.random(None, None, keys, seed=46)
        assert array.size == 1
        for i, value in enumerate(array.values()):
            np.testing.assert_allclose(value, rng.normal(size=1))

        rng = np.random.default_rng(seed=46)
        array = isopy.random(1000, None, keys, seed = 46)
        assert array.size == 1000
        for i, value in enumerate(array.values()):
            np.testing.assert_allclose(value, rng.normal(size=1000))

        rng = np.random.default_rng(seed=46)
        array = isopy.random(1000, 10, keys, seed=46)
        assert array.size == 1000
        for i, value in enumerate(array.values()):
            np.testing.assert_allclose(value, rng.normal(10, size=1000))

        rng = np.random.default_rng(seed=46)
        array = isopy.random(1000, (10, 100), keys, seed=46)
        assert array.size == 1000
        for i, value in enumerate(array.values()):
            np.testing.assert_allclose(value, rng.normal(10,100, size=1000))

        rng = np.random.default_rng(seed=46)
        input = [(1, 0.1), (10, 1), (0, 0.5)]
        array = isopy.random(1000, input, keys, seed=46)
        assert array.size == 1000
        for i, value in enumerate(array.values()):
            np.testing.assert_allclose(value, rng.normal(*input[i], size=1000))

        with pytest.raises(ValueError):
            isopy.random(1000, [(1, 0.1), (10, 1)], keys, seed=46)

        rng = np.random.default_rng(seed=46)
        input = [(1, 0.1), (10, 1), (0, 0.5)]
        array = isopy.random(1000, input, keys, seed=46)
        assert array.size == 1000
        for i, value in enumerate(array.values()):
            np.testing.assert_allclose(value, rng.normal(*input[i], size=1000))

        rng = np.random.default_rng(seed=46)
        array = isopy.random(1000, None, keys, distribution='poisson', seed=46)
        assert array.size == 1000
        for i, value in enumerate(array.values()):
            np.testing.assert_allclose(value, rng.poisson(size=1000))

        rng = np.random.default_rng(seed=46)
        array = isopy.random(1000, None, keys, distribution='rayleigh', seed=46)
        assert array.size == 1000
        for i, value in enumerate(array.values()):
            np.testing.assert_allclose(value, rng.rayleigh(size=1000))

    def test_random2(self):
        rng = np.random.default_rng(seed=46)
        array = isopy.random(None, None, seed=46)
        assert not isinstance(array, core.IsopyArray)
        assert isinstance(array, np.ndarray)
        assert array.ndim == 0
        assert array.size == 1
        np.testing.assert_allclose(array, rng.normal(size=1))

        rng = np.random.default_rng(seed=46)
        array = isopy.random(1, None, seed=46)
        assert not isinstance(array, core.IsopyArray)
        assert isinstance(array, np.ndarray)
        assert array.ndim == 1
        assert array.size == 1
        np.testing.assert_allclose(array, rng.normal(size=1))

        rng = np.random.default_rng(seed=46)
        array = isopy.random(100, None, seed=46)
        assert not isinstance(array, core.IsopyArray)
        assert isinstance(array, np.ndarray)
        assert array.ndim == 1
        assert array.size == 100
        np.testing.assert_allclose(array, rng.normal(size=100))


        rng = np.random.default_rng(seed=46)
        array = isopy.random(100, 33, seed=46)
        assert not isinstance(array, core.IsopyArray)
        assert isinstance(array, np.ndarray)
        assert array.ndim == 1
        assert array.size == 100
        np.testing.assert_allclose(array, rng.normal(33, size=100))

        rng = np.random.default_rng(seed=46)
        array = isopy.random(100, (33, 10), seed=46)
        assert not isinstance(array, core.IsopyArray)
        assert isinstance(array, np.ndarray)
        assert array.ndim == 1
        assert array.size == 100
        np.testing.assert_allclose(array, rng.normal(33, 10, size=100))

class Test_Misc:
    def test_is(self):
        key = isopy.keystring('pd')
        keylist =  isopy.keylist('ru pd cd'.split())
        array = isopy.ones(1, keylist)
        d = isopy.asdict({})
        refval = isopy.asrefval({})

        assert core.iskeystring(key.str()) is False
        assert core.iskeystring(key) is True

        assert core.iskeylist(keylist.strlist()) is False
        assert core.iskeylist(keylist) is True

        assert core.isarray(array.to_list()) is False
        assert core.isarray(array) is True

        assert core.isdict(d.to_dict()) is False
        assert core.isdict(d) is True
        assert core.isdict(refval) is True

        assert core.isrefval(refval.to_dict()) is False
        assert core.isrefval(refval) is True
        assert core.isrefval(d) is False

    def test_classname(self):
        keylist = isopy.keylist('101 105 111'.split())
        assert core.get_classname(keylist) == isopy.core.IsopyKeyList.__name__
        assert core.get_classname(isopy.core.IsopyKeyList) == isopy.core.IsopyKeyList.__name__

    def test_extract_kwargs(self):
        kwargs = dict(Test_a = 'testa', b = 'b', Test_b = 'testb', a = 'a')
        new_kwargs = core.extract_kwargs(kwargs, 'Test')
        assert len(kwargs) == 2
        assert len(new_kwargs) == 2
        assert kwargs['a'] == 'a'
        assert kwargs['b'] == 'b'
        assert new_kwargs['a'] == 'testa'
        assert new_kwargs['b'] == 'testb'

        kwargs = dict(Test_a='testa', b='b', Test_b='testb', a='a')
        new_kwargs = core.extract_kwargs(kwargs, 'Test', keep_prefix=True)
        assert len(kwargs) == 2
        assert len(new_kwargs) == 2
        assert kwargs['a'] == 'a'
        assert kwargs['b'] == 'b'
        assert new_kwargs['Test_a'] == 'testa'
        assert new_kwargs['Test_b'] == 'testb'

    def test_notgiven(self):
        assert not isopy.core.NotGiven
        assert str(isopy.core.NotGiven) == 'Optional'
        assert repr(isopy.core.NotGiven) == 'Optional'

    def test_renamed_kwarg(self):
        """
        Test that the function renames kwargs and forwards all arguments
        """
        def test(a=0, **_):
            return a == 1

        test2 = core.renamed_kwarg(b='a')(test)

        assert not test()
        assert test(1)
        assert test(a=1)
        assert not test(b=1)

        assert not test2()
        assert test2(1)
        assert test2(a=1)
        assert test2(b=1)

        assert test2(a=1, b=0)

        def func(*args, **kwargs):
            return args, kwargs

        func2 = core.renamed_kwarg(b='c')(func)
        assert func(1, 2, a=3, c=4) == func2(1, 2, a=3, b=4)

    def test_deprecated_function(self):
        def func():
            return 'success'

        defunc = core.deprecrated_function('deprecated')(func)

        assert func() == 'success'
        assert defunc() == 'success'

        def func(*args, **kwargs):
            return args, kwargs

        defunc = core.deprecrated_function('deprecated')(func)

        assert func(1,2, b=3, c=4) == defunc(1,2, b=3, c=4)

    def test_lru_cache(self):
        """
        Test the lru_cache function that is used to cache the output
        from the as* functions.
        """
        constant = 1
        def func(*args, **kwargs):
            return constant, args, kwargs

        cached = core.lru_cache(2)(func) # Only cache 2 things

        assert func(2, a=(3, 4)) == (1, (2,), {'a': (3,4)})
        assert func(2, a=[5,6]) == (1, (2,), {'a': [5,6]})
        assert cached(2, a=(3, 4)) == (1, (2,), {'a': (3, 4)})
        assert cached(2, a=[5, 6]) == (1, (2,), {'a': [5, 6]})

        constant = 2
        assert func(2, a=(3, 4)) == (2, (2,), {'a': (3, 4)})
        assert func(2, a=[5, 6]) == (2, (2,), {'a': [5, 6]})
        assert cached(2, a=(3, 4)) == (1, (2,), {'a': (3, 4)}) # Cached
        assert cached(2, a=[5, 6]) == (2, (2,), {'a': [5, 6]}) # Cannot be cached

        assert func(2, b=(3, 4)) == (2, (2,), {'b': (3, 4)})
        assert func(2, c=(3, 4)) == (2, (2,), {'c': (3, 4)})
        assert func(2, a=(3, 4)) == (2, (2,), {'a': (3, 4)}) # Not stored in cache any more

        # Make sure that TypeError are raised
        def func(*args, **kwargs):
            raise TypeError('isopy')

        cached = core.lru_cache(10)(func)

        with pytest.raises(TypeError) as err:
            cached()
        assert str(err.value) == 'isopy'

    def test_check_type(self):
        assert checks.check_type('test', '1', str) == '1'
        assert checks.check_type('test', '1', int, float, str) == '1'

        with pytest.raises(TypeError):
            checks.check_type('test', '1', int, float)
        
        coerced = checks.check_type('test', '1', int, float, coerce=True)
        assert type(coerced) == int
        assert coerced == 1

        coerced = checks.check_type('test', '1', int, float, coerce=True, coerce_into=float)
        assert type(coerced) == float
        assert coerced == 1

        with pytest.raises(TypeError):
            checks.check_type('test', 'a', int, float, coerce=True)

        coerced = checks.check_type('test', 'a', int, float, coerce=True, coerce_into=[float, str])
        assert type(coerced) == str
        assert coerced == 'a'

        with pytest.raises(TypeError):
            checks.check_type('test', None, int, float)

        assert checks.check_type('test', None, int, float, allow_none=True) is None