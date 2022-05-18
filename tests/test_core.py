import isopy
from isopy import core
import numpy as np
import pytest
import itertools
import pyperclip
import warnings


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
    def  test_comparison(self):
        for name, item in core.FLAVOURS.items():
            assert name in core.DEFAULT_TRY_FLAVOURS
            assert item() in core.DEFAULT_TRY_FLAVOURS
            assert item() in core.FLAVOURS.values()


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


# TODO test for MZ
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
                             same=['(H2O)', '(H)2(O)', 'h2o', '(((H2O)))'],
                             fails='1(H2O) ++H2O 1H)O H(16O (OH)+- (OH)+- +(OH) 2 + OH!'.split()  + [12],
                             different='HHO (2H)2O H2(16O) HNO3 HCl (OH)- 137Ba++ H+1HO H((OH)) ((OH)2)- H((OH)-)-'.split())

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
                             fails = [''],
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

            basekey = key.basekey
            key2 = keytype(string, ignore_charge = True)
            assert key2 == basekey

            key2 = keytype(key, ignore_charge = True)
            assert key2 == basekey

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
            gkey = isopy.GeneralKeyString(key)
            ikey = isopy.keystring(gkey)
            assert type(gkey) != type(ikey)
            assert gkey != ikey
            assert gkey == key
            assert ikey == key

            gkey2 = isopy.askeystring(gkey)
            assert type(gkey) == type(gkey2)
            assert gkey == gkey2

        key = core.keystring('ba++')
        assert core.keystring(key) != 'ba'
        assert core.keystring(key, ignore_charge=True) == 'ba'

        assert core.askeystring(key) != 'ba'
        assert core.askeystring(key, ignore_charge=True) == 'ba'

    def general_creation(self, keytype, correct = []):
        #Test creation with isopy.keystring() and isopy.askeystring
        for string in correct:
            key1 = isopy.keystring(string)
            assert type(key1) == keytype
            assert isinstance(key1, str)
            assert key1 == string

            key2 = isopy.askeystring(key1)
            assert key2 is key1

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
        keys = ['105', 'Pd', '105Pd', 'test', '104Ru/106Cd', 'H2O']
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
        for other in [isopy.IsotopeKeyString('74Ge'), isopy.RatioKeyString('88Sr/87Sr')]:
            for string in [isopy.MassKeyString('105'), isopy.ElementKeyString('Pd'),
                      isopy.IsotopeKeyString('105Pd'), isopy.GeneralKeyString('test'),
                    isopy.RatioKeyString('Cd/Ru'), isopy.MoleculeKeyString('H2O')]:
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

    def test_str(self):
        # Test the *str* method that turns key string into python strings

        key = isopy.MassKeyString('101')
        assert repr(key) == "MassKeyString('101')"
        assert key.str() == '101'
        str_options = dict(key='101', m='101')
        for k, v in str_options.items():
            assert key.str(k) == v
            assert key.str(f'key {{{k}}}') == f'key {v}'

        key = isopy.ElementKeyString('pd')
        assert repr(key) == "ElementKeyString('Pd')"
        assert key.str() == 'Pd'
        str_options = dict(key='Pd', es = 'pd', Es = 'Pd', ES = 'PD',
                           name = 'palladium', Name = 'Palladium', NAME = 'PALLADIUM')
        for k, v in str_options.items():
            assert key.str(k) == v
            assert key.str(f'key {{{k}}}') == f'key {v}'

        key = isopy.IsotopeKeyString('101pd')
        assert repr(key) == "IsotopeKeyString('101Pd')"
        assert key.str() == '101Pd'
        str_options = dict(key = '101Pd', m = '101', es = 'pd', Es = 'Pd', ES = 'PD',
                           name = 'palladium', Name = 'Palladium', NAME = 'PALLADIUM',
                           mEs = '101Pd', ESm = 'PD101', namem = 'palladium101', mNAME = '101PALLADIUM')
        str_options.update({'NAME-m': 'PALLADIUM-101', 'm-es': '101-pd'})
        for k, v in str_options.items():
            assert key.str(k) == v
            assert key.str(f'key {{{k}}}') == f'key {v}'

        key = isopy.MoleculeKeyString('H2O')
        assert repr(key) == "MoleculeKeyString('H2O')"
        assert key.str() == 'H2O'
        str_options = dict(key='H2O')
        for k, v in str_options.items():
            assert key.str(k) == v
            assert key.str(f'key {{{k}}}') == f'key {v}'

        key = isopy.MoleculeKeyString('((16O)(2H)2)-')
        assert repr(key) == "MoleculeKeyString('((16O)(2H)2)-')"
        assert key.str() == '((16O)(2H)2)-'
        str_options = dict(key='((16O)(2H)2)-')
        for k, v in str_options.items():
            assert key.str(k) == v
            assert key.str(f'key {{{k}}}') == f'key {v}'

        key = isopy.RatioKeyString('pd/101pd')
        assert repr(key) == "RatioKeyString('Pd/101Pd')"
        assert key.str() == 'Pd/101Pd'
        str_options = {'key': 'Pd/101Pd', 'n/d': 'Pd/101Pd', 'n': 'Pd', 'd': '101Pd'}
        for k, v in str_options.items():
            assert key.str(k) == v
            assert key.str(f'key {{{k}}}') == f'key {v}'
        assert key.str('n/d', 'es', 'key {Name-m}') == 'pd/key Palladium-101'
        assert key.str(('n/d', 'es', 'Name-m')) == 'pd/Palladium-101'
        assert key.str({'format': 'n/d', 'nformat': 'es', 'dformat': 'Name-m'}) == 'pd/Palladium-101'

        key = isopy.RatioKeyString('pd/ru//ag/cd')
        assert repr(key) == "RatioKeyString('Pd/Ru//Ag/Cd')"
        assert key.str() == 'Pd/Ru//Ag/Cd'
        str_options = {'key': 'Pd/Ru//Ag/Cd', 'n/d': 'Pd/Ru/Ag/Cd', 'n': 'Pd/Ru', 'd': 'Ag/Cd'}
        for k, v in str_options.items():
            assert key.str(k) == v
            assert key.str(f'key {{{k}}}') == f'key {v}'

        key = isopy.GeneralKeyString('hermione')
        assert repr(key) == "GeneralKeyString('hermione')"
        assert key.str() == 'hermione'
        str_options = dict(key='hermione')
        for k, v in str_options.items():
            assert key.str(k) == v
            assert key.str(f'key {{{k}}}') == f'key {v}'

    def test_charge(self):
        assert core.GeneralKeyString('Hermione++') == 'Hermione++'
        assert core.GeneralKeyString('Hermione++', ignore_charge=True) == 'Hermione++'
        assert core.GeneralKeyString(core.GeneralKeyString('Hermione++'), ignore_charge=True) == 'Hermione++'

        assert core.RatioKeyString('105Pd++/108Pd-', ignore_charge=True) == '105pd/108pd'

        key = core.RatioKeyString('105Pd++/108Pd-')
        assert str(key) == '105Pd++/108Pd-'
        assert key.numerator.charge == 2
        assert key.denominator.charge == -1
        assert core.RatioKeyString(key, ignore_charge=True) == '105pd/108pd'

        # Test charge attributes/methods for element and isotope key strings
        #Element
        for charge, strings in {-2: ['ba--'], -1: ['ba-'],
                                1: ['ba+'], 2: ['ba++']}.items():
            for string in strings:
                key = isopy.ElementKeyString(string)
                assert key.charge == charge
                assert str(key) == 'Ba' + ('+' * charge or '-' * abs(charge))
                assert key.basekey == 'Ba'
                key = isopy.ElementKeyString(key, ignore_charge=True)
                assert key.charge is None
                assert str(key) == 'Ba'
                key = isopy.ElementKeyString(string, ignore_charge=True)
                assert key.charge is None
                assert str(key) == 'Ba'
        key = isopy.ElementKeyString('Ba')
        for charge in [-2, -1, 1, 2]:
            key2 = key.set_charge(charge)
            assert key is not key2
            assert key.charge is None
            assert key2.charge is charge
        key = isopy.ElementKeyString('Ba+').set_charge(0)
        assert key.charge == None
        key = isopy.ElementKeyString('Ba+').set_charge(None)
        assert key.charge == None

        #isotope
        mass = isopy.refval.isotope.mass.get('ba138')
        assert isopy.IsotopeKeyString('138-ba').charge is None
        assert isopy.IsotopeKeyString('ba-138').charge is None
        for charge, strings in {-2: ['Ba--138', '138ba--'], -1: ['138ba-'],
                                1: ['138ba+', 'ba+138'], 2: ['138ba++', 'ba++138']}.items():
            for string in strings:
                key = isopy.IsotopeKeyString(string)
                assert key.charge == charge
                assert str(key) == '138Ba' + ('+' * charge or '-' * abs(charge))
                assert key.mz() == 138 / abs(charge)
                assert key.mz(True) == mass / abs(charge)
                assert key.basekey == '138Ba'
                key = isopy.IsotopeKeyString(key, ignore_charge=True)
                assert key.charge is None
                assert str(key) == '138Ba'
                key = isopy.IsotopeKeyString(string, ignore_charge=True)
                assert key.charge is None
                assert str(key) == '138Ba'
        key = isopy.IsotopeKeyString('138Ba')
        for charge in [-2, -1, 1, 2]:
            key2 = key.set_charge(charge)
            assert key is not key2
            assert key.charge is None
            assert key2.charge is charge
        key = isopy.IsotopeKeyString('138Ba+').set_charge(0)
        assert key.charge == None
        key = isopy.IsotopeKeyString('138Ba+').set_charge(None)
        assert key.charge == None

        #molecule
        # isotope
        mass = isopy.refval.isotope.mass.get('1H') * 2
        mass += isopy.refval.isotope.mass.get('16O')
        assert isopy.MoleculeKeyString('((1H)2(16O))').charge is None
        for charge, strings in {-2: ['((1H)2(16O))--'], -1: ['((1H)2(16O))-'],
                                1: ['((1H)2(16O))+'], 2: ['((1H)2(16O))++']}.items():
            for string in strings:
                key = isopy.MoleculeKeyString(string)
                assert key.charge == charge
                assert str(key) == '((1H)2(16O))' + ('+' * charge or '-' * abs(charge))
                assert key.mz() == 18 / abs(charge)
                assert key.mz(True) == mass / abs(charge)
                assert key.basekey == isopy.MoleculeKeyString(key, ignore_charge=True)
                key = isopy.MoleculeKeyString(key, ignore_charge=True)
                assert key.charge is None
                assert str(key) == '(1H)2(16O)'
                key = isopy.MoleculeKeyString(string, ignore_charge=True)
                assert key.charge is None
                assert str(key) ==  '(1H)2(16O)'
        key = isopy.MoleculeKeyString('H2O')
        for charge in [-2, -1, 1, 2]:
            key2 = key.set_charge(charge)
            assert key is not key2
            assert key.charge is None
            assert key2.charge is charge
        key = isopy.MoleculeKeyString('(H2O)+').set_charge(0)
        assert key.charge is None
        key = isopy.MoleculeKeyString('(H2O)+').set_charge(None)
        assert key.charge is None

    def test_fraction(self):
        fractions1 = isopy.refval.isotope.fraction
        fractions2 = dict(h1 = 0.5, h2 = 0.5, o16=0.6, o17=0.2, o18=0.1, ba137 = 1, pd105=0.1, pd108=0.5)
        fractions3 = isopy.ScalarDict(fractions2)

        isotopes = isopy.IsotopeKeyList('105pd', '137Ba', '1H', '16O', '5H')
        for key in isotopes:
            np.testing.assert_almost_equal(key.fraction(), fractions1.get(key))
            np.testing.assert_almost_equal(key.fraction(5), fractions1.get(key, 5))
            np.testing.assert_almost_equal(key.fraction(isotope_fractions = fractions2), fractions3.get(key))

        molecule = isopy.MoleculeKeyString('(1H)2(16O)')
        molfrac1 = fractions1.get('1h') ** 2 * fractions1.get('16o')
        molfrac2 = fractions3.get('1h') ** 2 * fractions3.get('16o')
        np.testing.assert_almost_equal(molecule.fraction(), molfrac1)
        np.testing.assert_almost_equal(molecule.fraction(isotope_fractions = fractions2), molfrac2)

        molecule = isopy.MoleculeKeyString('(H)2(16O)')
        molfrac1 = fractions1.get('16o')
        molfrac2 = fractions3.get('16o')
        np.testing.assert_almost_equal(molecule.fraction(), molfrac1)
        np.testing.assert_almost_equal(molecule.fraction(isotope_fractions=fractions2), molfrac2)

        molecule = isopy.MoleculeKeyString('(1H)2(O)')
        molfrac1 = fractions1.get('1h') ** 2
        molfrac2 = fractions3.get('1h') ** 2
        np.testing.assert_almost_equal(molecule.fraction(), molfrac1)
        np.testing.assert_almost_equal(molecule.fraction(isotope_fractions=fractions2), molfrac2)

        molecule = isopy.MoleculeKeyString('(5H)2(16O)')
        molfrac1 = fractions1.get('5h') ** 2 * fractions1.get('16o')
        np.testing.assert_almost_equal(molecule.fraction(), molfrac1)
        molfrac1 = fractions1.get('5h', 5) ** 2 * fractions1.get('16o', 5)
        np.testing.assert_almost_equal(molecule.fraction(5), molfrac1)

        molecule = isopy.MoleculeKeyString('(1H)(2H)(16O)')
        molfrac1 = fractions1.get('1h') * fractions1.get('2h') * fractions1.get('16o')
        molfrac2 = fractions3.get('1h') * fractions3.get('2h') * fractions3.get('16o')
        np.testing.assert_almost_equal(molecule.fraction(), molfrac1)
        np.testing.assert_almost_equal(molecule.fraction(isotope_fractions = fractions2), molfrac2)

        molecule = isopy.MoleculeKeyString('((16O)(1H))2')
        molfrac1 = (fractions1.get('16o') * fractions1.get('1h')) ** 2
        molfrac2 = (fractions3.get('16o') * fractions3.get('1h')) ** 2
        np.testing.assert_almost_equal(molecule.fraction(), molfrac1)
        np.testing.assert_almost_equal(molecule.fraction(isotope_fractions = fractions2), molfrac2)

        molecule = isopy.MoleculeKeyString('((2H)(18O))3((16O)(1H))2')
        molfrac1 = ((fractions1.get('16o') * fractions1.get('1h')) ** 2) * ((fractions1.get('18o') * fractions1.get('2h')) ** 3)
        molfrac2 = ((fractions3.get('16o') * fractions3.get('1h')) ** 2) * ((fractions3.get('18o') * fractions3.get('2h')) ** 3)
        np.testing.assert_almost_equal(molecule.fraction(), molfrac1)
        np.testing.assert_almost_equal(molecule.fraction(isotope_fractions = fractions2), molfrac2)

        ratio = isopy.RatioKeyString('108pd/105pd')
        isofrac1 = fractions1.get('pd108') / fractions1.get('pd105')
        isofrac2 = fractions2.get('pd108') / fractions2.get('pd105')
        np.testing.assert_almost_equal(ratio.isotope_fraction(), isofrac1)
        np.testing.assert_almost_equal(ratio.isotope_fraction(isotope_fractions=fractions2), isofrac2)

        ratio = isopy.RatioKeyString('108pd/pd')
        with pytest.raises(AttributeError):
            ratio.isotope_fraction()

        ratio = isopy.RatioKeyString('pd/105pd')
        with pytest.raises(AttributeError):
            ratio.isotope_fraction()

        ratio = isopy.RatioKeyString('108pd/105')
        with pytest.raises(AttributeError):
            ratio.isotope_fraction()

        ratio = isopy.RatioKeyString('108pd/hermione')
        with pytest.raises(AttributeError):
            ratio.isotope_fraction()

    def test_contains_key(self):
        for string in '(2H)2(16O) (((16O)(1H))2)(18F)'.split():
            key = core.MoleculeKeyString(string)
            assert not key.contains_element_key()
            assert key.contains_isotope_key()

        for string in '(H)2(16O) ((OH)2)(18F)'.split():
            key = core.MoleculeKeyString(string)
            assert key.contains_element_key()
            assert key.contains_isotope_key()

        for string in '(H)2(O) ((OH)2)F'.split():
            key = core.MoleculeKeyString(string)
            assert key.contains_element_key()
            assert not key.contains_isotope_key()

    def test_element(self):
        molecule = isopy.MoleculeKeyString('H2O')
        assert molecule.element() == 'H2O'

        molecule = isopy.MoleculeKeyString('(1H)2O')
        assert molecule.element() == 'H2O'

        molecule = isopy.MoleculeKeyString('(1H)2(16O)')
        assert molecule.element() == 'H2O'

        molecule = isopy.MoleculeKeyString('(1H)(2H)(16O)')
        assert molecule.element() == 'HHO'

        molecule = isopy.MoleculeKeyString('(OH)2')
        assert molecule.element() == '(OH)2'

        molecule = isopy.MoleculeKeyString('((16O)H)2')
        assert molecule.element() == '(OH)2'

        molecule = isopy.MoleculeKeyString('((16O)(2H))2')
        assert molecule.element() == '(OH)2'

        molecule = isopy.MoleculeKeyString('((16O)(2H))((18O)(1H))')
        assert molecule.element() == 'OHOH'

        element = isopy.ElementKeyString('pd')
        assert element.element() == 'pd'

        isotope = isopy.IsotopeKeyString('105pd')
        assert isotope.element() == 'pd'

    def test_isotopes(self):
        molecule = isopy.MoleculeKeyString('O')
        n = len(isopy.refval.element.isotopes['o'])
        isotopes = molecule.isotopes()
        assert len(isotopes) == n
        assert isotopes == [isopy.MoleculeKeyString(key) for key in isopy.refval.element.isotopes['o']]
        np.testing.assert_almost_equal(np.sum(isotopes.fractions()), 1, decimal=5)

        molecule = isopy.MoleculeKeyString('O2')
        n = len(isopy.refval.element.isotopes['o']) ** 2
        isotopes = molecule.isotopes()
        assert len(isotopes) == n
        assert isotopes == [isopy.MoleculeKeyString(key) for key in
                            '(16O)(16O) (17O)(16O) (18O)(16O) ' \
                            '(16O)(17O) (17O)(17O) (18O)(17O) ' \
                            '(16O)(18O) (17O)(18O) (18O)(18O)'.split()]
        np.testing.assert_almost_equal(np.sum(isotopes.fractions()), 1, decimal=5)

        molecule = isopy.MoleculeKeyString('OH')
        n = len(isopy.refval.element.isotopes['o']) * len(isopy.refval.element.isotopes['H'])
        isotopes = molecule.isotopes()
        assert len(isotopes) == n
        assert isotopes == [isopy.MoleculeKeyString(key) for key in
                            '(16O)(1H) (17O)(1H) (18O)(1H) ' \
                            '(16O)(2H) (17O)(2H) (18O)(2H)'.split()]
        np.testing.assert_almost_equal(np.sum(isotopes.fractions()), 1, decimal=5)

        molecule = isopy.MoleculeKeyString('H2(16O)')
        n =  len(isopy.refval.element.isotopes['H']) ** 2
        isotopes = molecule.isotopes()
        assert len(isotopes) == n
        assert isotopes == [isopy.MoleculeKeyString(key) for key in
                            '(1H)(1H)(16O) (2H)(1H)(16O) ' \
                            '(1H)(2H)(16O) (2H)(2H)(16O)'.split()]
        np.testing.assert_almost_equal(np.sum(isotopes.fractions()), isopy.refval.isotope.fraction['16o'], decimal=5)

        molecule = isopy.MoleculeKeyString('H2O')
        n = len(isopy.refval.element.isotopes['H']) ** 2 * len(isopy.refval.element.isotopes['O'])
        isotopes = molecule.isotopes()
        assert len(isotopes) == n
        np.testing.assert_almost_equal(np.sum(isotopes.fractions()), 1, decimal=5)

        element = isopy.ElementKeyString('Pd')
        assert element.isotopes() == '102pd 104pd 105pd 106pd 108pd 110pd'.split()

        isotope = isopy.IsotopeKeyString('105pd')
        assert isotope.isotopes() == ['105pd']


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

        molecule = self.creation(isopy.MoleculeKeyList,
                                correct=['H2O', 'HNO3', 'HCl'],
                                same=[('h20', '(hNO3))', '(h)(cl)')],
                                fail=['2(H2O)', 'hno3', 'hcl'])

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

        mixed = self.creation(isopy.MixedKeyList,
                              correct = ['ru', 'pd', '105pd', '106pd', '110cd/105pd', '111cd/105pd'],
                              same = [('ru', 'pd', '105pd', '106pd', '110cd/105pd', '111cd/105pd')],
                              fail = None)
        assert type(mixed[:2]) == isopy.ElementKeyList
        assert type(mixed[2:4]) == isopy.IsotopeKeyList
        assert type(mixed[4:]) == isopy.RatioKeyList

        ratio = core.RatioKeyList()
        assert type(ratio.numerators) is tuple
        assert type(ratio.denominators) is tuple
        assert len(ratio.numerators) == 0
        assert len(ratio.denominators) == 0
        for numerator, denominator in itertools.permutations((mass, element, isotope, molecule, general), 2):
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
            assert isopy.RatioKeyList(keys).common_denominator is None

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

        assert type(isopy.keylist(['ru', 'pd', '105pd', '106pd', '110cd/105pd', '111cd/105pd'])) is core.MixedKeyList
        assert type(isopy.askeylist(['ru', 'pd', '105pd', '106pd', '110cd/105pd', '111cd/105pd'])) is core.MixedKeyList

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

        array = isopy.array([i for i in range(len(correct))], correct)
        same_list = listtype(array.dtype)
        assert len(same_list) == len(correct)
        assert same_list == correct
        assert same_list == correct_list
        assert type(same_list) is listtype

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

        dictionary = {k: i for i, k in enumerate(correct)}
        assert isopy.keylist(dictionary) == correct_list
        assert isopy.askeylist(dictionary) == correct_list

        array = isopy.array(dictionary)
        assert isopy.keylist(array) == correct_list
        assert isopy.askeylist(array) == correct_list

        if fail:
            with pytest.raises(core.KeyParseError):
                listtype(fail)

            mixed_list = isopy.keylist(fail)
            assert type(mixed_list) is not listtype
            assert mixed_list != correct

            mixed_list = isopy.askeylist(fail)
            assert type(mixed_list) is not listtype
            assert mixed_list != correct

        return correct_list

    def test_compare(self):
        mass = self.compare(isopy.MassKeyList,
                             keys=['104', '105', '106'],
                             extra_keys=['99', '108', '111'],
                             notin=['70', '76', '80'],
                            other=['Ni', 'Ge', 'Se'])

        element = self.compare(isopy.ElementKeyList,
                                keys=['Ru', 'Pd', 'Cd'],
                                extra_keys=['Mo', 'Ag', 'Te'],
                                notin=['Ni', 'Ge', 'Se'],
                               other=['70Ge', '76Ge', '80Se'])

        isotope = self.compare(isopy.IsotopeKeyList,
                                keys=['104Ru', '105Pd', '106Cd'],
                                extra_keys=['99Ru', '106Pd', '111Cd'],
                                notin=['70Ge', '76Ge', '80Se'],
                               other=['Ni', 'Ge', 'Se'])

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

    def compare(self, listtype, keys, extra_keys=[], notin=[], other=None):
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

        if other is not None:
            assert keylist != other

        keylist_ = listtype(keys, ignore_duplicates = True)
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

        keylist3 = listtype(keys2, ignore_duplicates = True)
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

        with pytest.raises(ValueError):
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

            self.bitwise(isopy.RatioKeyList, key1, key2, rand, ror, rxor)

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

        molecule = isopy.MoleculeKeyList(['H2O', 'HNO3', 'HBr', 'HCl', 'HF', 'HI'])
        self.filter_key(isopy.MoleculeKeyList, 'key', molecule, molecule, ['OH', 'H2O2'])

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

        if prefix != 'key':
            result = keylist.filter(**{f'{prefix}': eq[0]})
            assert type(result) is listtype
            assert result == [keylist[i] for i, key in enumerate(eq) if key == eq[0]]
            assert len(result) != 0

            result = keylist.filter(**{f'{prefix}': eq[2:]})
            assert type(result) == listtype
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
        assert keylist.filter(mz_lt=66, mz_true_mass=True) == '64zn  132ba-- 66zn'.split()

        assert keylist.filter(mz_le=66) == '64zn  132ba-- 66zn'.split()
        assert keylist.filter(mz_le=66, mz_true_mass=True) == '64zn  132ba-- 66zn'.split()

        assert keylist.filter(mz_gt=66) == '67zn 68zn 137ba++ 70zn 136ba'.split()
        assert keylist.filter(mz_gt=66, mz_true_mass=True) == '67zn 68zn 137ba++ 70zn 136ba'.split()

        assert keylist.filter(mz_ge=66) == '132ba-- 66zn 67zn 68zn 137ba++ 70zn 136ba'.split()
        assert keylist.filter(mz_ge=66, mz_true_mass=True) == '67zn 68zn 137ba++ 70zn 136ba'.split()

        keylist = isopy.MoleculeKeyList('64zn  132ba-- 66zn 67zn 68zn 137ba++ 70zn 136ba'.split())

        assert keylist.filter(mz_eq=66) == '132ba-- 66zn'.split()
        assert keylist.filter(mz_eq=[66, 68]) == '132ba-- 66zn 68zn'.split()

        assert keylist.filter(mz_neq=66) == '64zn  67zn 68zn 137ba++ 70zn 136ba'.split()
        assert keylist.filter(mz_neq=[66, 68]) == '64zn 67zn 137ba++ 70zn 136ba'.split()

        assert keylist.filter(mz_lt=66) == '64zn'.split()
        assert keylist.filter(mz_lt=66, mz_true_mass=True) == '64zn  132ba-- 66zn'.split()

        assert keylist.filter(mz_le=66) == '64zn  132ba-- 66zn'.split()
        assert keylist.filter(mz_le=66, mz_true_mass=True) == '64zn  132ba-- 66zn'.split()

        assert keylist.filter(mz_gt=66) == '67zn 68zn 137ba++ 70zn 136ba'.split()
        assert keylist.filter(mz_gt=66, mz_true_mass=True) == '67zn 68zn 137ba++ 70zn 136ba'.split()

        assert keylist.filter(mz_ge=66) == '132ba-- 66zn 67zn 68zn 137ba++ 70zn 136ba'.split()
        assert keylist.filter(mz_ge=66, mz_true_mass=True) == '67zn 68zn 137ba++ 70zn 136ba'.split()

        keylist = isopy.keylist('64zn  132ba-- zn 67zn'.split())
        with pytest.raises(TypeError):
            keylist.filter(mz_eq = 66)

    def test_filter_flavour(self):
        keylist = isopy.keylist(['ru', 'pd', '105pd', '106pd', '110cd/105pd', '111cd/105pd'])

        sublist1 = keylist.filter(flavour_eq ='element')
        sublist2 = keylist.filter(flavour_neq = ['isotope', 'ratio'])
        assert type(sublist1) == core.ElementKeyList
        assert type(sublist2) == core.ElementKeyList
        assert sublist1 == ['ru', 'pd']
        assert sublist2 == ['ru', 'pd']

        sublist1 = keylist.filter(flavour_eq=['isotope', 'ratio'])
        sublist2 = keylist.filter(flavour_neq='element')
        assert type(sublist1) == core.MixedKeyList
        assert type(sublist2) == core.MixedKeyList
        assert sublist1 == ['105pd', '106pd', '110cd/105pd', '111cd/105pd']
        assert sublist2 == ['105pd', '106pd', '110cd/105pd', '111cd/105pd']

        keylist = isopy.keylist(['105', 'pd', '105pd', '110cd/105pd', 'H2O',
                                 '96', 'ru', '96ru', '96ru/101ru', 'HNO3'])
        list_types = [core.MassKeyList, core.ElementKeyList, core.IsotopeKeyList,
                      core.RatioKeyList, core.MoleculeKeyList]
        type_names = ['mass', 'element', 'isotope', 'ratio', 'molecule']
        for i in range(5):
            sublist = keylist.filter(flavour_eq=type_names[i])
            assert type(sublist) == list_types[i]

            sublist = keylist.filter(flavour_eq=list_types[i].flavour)
            assert type(sublist) == list_types[i]

            sublist = keylist.filter(flavour_neq=type_names[i])
            assert type(sublist) == core.MixedKeyList

        keylist = isopy.keylist('105 pd 99ru 108pd/105pd H2O hermione'.split())
        flavours = 'mass element isotope ratio molecule general'.split()
        for i, flavour in enumerate(flavours):
            eq = keylist[i:i+1]
            neq = keylist[:i] + keylist[i+1:]

            keq = keylist.filter(flavour_eq=flavour)
            assert keq == eq
            assert keq.flavour == flavour

            kneq = keylist.filter(flavour_neq=flavour)
            assert kneq == neq
            assert kneq.flavour == 'mixed'

    def test_invalid_filter(self):
        keylist = isopy.keylist(['105', 'pd', '105pd', '110cd/105pd', 'H2O', 'Hermione'])

        for key in keylist:
            assert key._filter(invalid = None) is False

    def test_charges(self):
        keylist = isopy.ElementKeyList('ba++ la+ ce'.split())
        assert keylist.charges == (2, 1, None)

        keylist = keylist.set_charges([0, 1, 2])
        assert keylist.charges == (None, 1, 2)
        keylist = keylist.set_charges(None)
        assert keylist.charges == (None, None, None)
        keylist = keylist.set_charges(2)
        assert keylist.charges == (2, 2, 2)

        with pytest.raises(ValueError):
            keylist.set_charges([1, 1])
        with pytest.raises(TypeError):
            keylist.set_charges('++')
        with pytest.raises(TypeError):
            keylist.set_charges(['++', '++', '++'])

        masses = tuple(isopy.refval.isotope.mass.get(k) / v for k, v in zip(('136ba', '137ba', '138ba'), (2, 1, 1)))

        keylist = isopy.IsotopeKeyList('136Ba++ 137ba+ 138ba'.split())
        assert keylist.charges == (2, 1, None)
        assert keylist.mz() == (136/2, 137/1, 138/1)
        assert keylist.mz(true_mass=True) == masses

        keylist = keylist.set_charges([0, 1, 2])
        assert keylist.charges == (None, 1, 2)
        keylist = keylist.set_charges(None)
        assert keylist.charges == (None, None, None)
        keylist = keylist.set_charges(2)
        assert keylist.charges == (2, 2, 2)

        keylist = isopy.IsotopeKeyList('136Ba-- 137ba- 138ba'.split())
        assert keylist.charges == (-2, -1, None)
        assert keylist.mz() == (136 / 2, 137 / 1, 138 / 1)
        np.testing.assert_allclose(keylist.mz(true_mass=True), masses)

        keylist = keylist.set_charges([0, -1, -2])
        assert keylist.charges == (None, -1, -2)
        keylist = keylist.set_charges(None)
        assert keylist.charges == (None, None, None)
        keylist = keylist.set_charges(-2)
        assert keylist.charges == (-2, -2, -2)

        with pytest.raises(ValueError):
            keylist.set_charges([1, 1])
        with pytest.raises(TypeError):
            keylist.set_charges('++')
        with pytest.raises(TypeError):
            keylist.set_charges(['++', '++', '++'])

        keylist = isopy.MoleculeKeyList('((1H)2(16O))-- ((1H)2(16O))- ((1H)2(16O))'.split())
        masses = [(isopy.refval.isotope.mass.get('1h') * 2 + isopy.refval.isotope.mass.get('16o')) / c for c in [2, 1, 1]]
        assert keylist.charges == (-2, -1, None)
        assert keylist.mz() == (18 / 2, 18 / 1, 18 / 1)
        np.testing.assert_allclose(keylist.mz(true_mass=True), masses)

        keylist = keylist.set_charges([0, -1, -2])
        assert keylist.charges == (None, -1, -2)
        keylist = keylist.set_charges(None)
        assert keylist.charges == (None, None, None)
        keylist = keylist.set_charges(-2)
        assert keylist.charges == (-2, -2, -2)

        with pytest.raises(ValueError):
            keylist.set_charges([1, 1])
        with pytest.raises(TypeError):
            keylist.set_charges('++')
        with pytest.raises(TypeError):
            keylist.set_charges(['++', '++', '++'])

    # TODO tests for Mixed, Molecule
    def test_add_subtract(self):
        # Add
        keylist = isopy.keylist('pd', 'cd')

        new = keylist + 'ru rh pd'.split()
        assert new == 'pd cd ru rh pd'.split()
        assert type(new) is isopy.ElementKeyList

        new = 'ru rh pd'.split() + keylist
        assert new == 'ru rh pd pd cd '.split()
        assert type(new) is isopy.ElementKeyList

        new = keylist + ['105pd', '111cd']
        assert new == 'pd cd 105pd 111cd'.split()
        assert type(new) is isopy.MixedKeyList

        new = ['105pd', '111cd'] + keylist
        assert new == '105pd 111cd pd cd '.split()
        assert type(new) is isopy.MixedKeyList

        #Sub
        keylist = isopy.keylist('pd', 'cd', '105pd', '111cd')

        new = keylist - 'cd 105pd'.split()
        assert new == 'pd 111cd'.split()
        assert type(new) is isopy.MixedKeyList

        new = keylist - '111cd 105pd'.split()
        assert new == 'pd cd'.split()
        assert type(new) is isopy.ElementKeyList

        new = 'rh pd 107ag'.split() - keylist
        assert new == 'rh 107ag'.split()
        assert type(new) is isopy.MixedKeyList

        new = 'rh pd ag'.split() - keylist
        assert new == 'rh ag'.split()
        assert type(new) is isopy.ElementKeyList

    def test_sorted(self):
        mass = isopy.MassKeyList('102 104 105 106 108 110'.split())
        mass2 = isopy.MassKeyList('104 108 102 110 106 105'.split())
        assert mass != mass2
        assert mass == mass2.sorted()

        element = isopy.ElementKeyList('ru rh pd ag cd te'.split())
        element2 = isopy.ElementKeyList('pd cd te ag rh ru'.split())
        assert element != element2
        assert element == element2.sorted()

        isotope = isopy.IsotopeKeyList('102ru 102pd 104ru 104pd 106pd 106cd'.split())
        isotope2 = isopy.IsotopeKeyList('106cd 104ru 102ru 104pd 102pd 106pd'.split())
        assert isotope != isotope2
        assert isotope == isotope2.sorted()

        molecule =  isopy.MoleculeKeyList('H2O', 'HCl', '(OH)2', 'HNO3')
        molecule2 = isopy.MoleculeKeyList('H2O HNO3 HCl (OH)2'.split())
        assert molecule != molecule2
        assert molecule == molecule2.sorted()

        molecule = isopy.MoleculeKeyList('H2O (2H)2(16O)++ (1H)2(16O) (2H)2(16O)'.split())
        molecule2 = isopy.MoleculeKeyList('(2H)2(16O)++ (2H)2(16O) (1H)2(16O) H2O'.split())
        assert molecule != molecule2
        assert molecule == molecule2.sorted()

        general = isopy.GeneralKeyList('ginny harry hermione luna neville ron'.split())
        general2 = isopy.GeneralKeyList('hermione ginny luna ron neville harry'.split())
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

        mixed = isopy.MixedKeyList('105 pd  99ru H2O 108pd/105pd hermione'.split())
        mixed2 = isopy.MixedKeyList('H2O 108pd/105pd 99ru 105 hermione pd'.split())
        assert mixed != mixed2
        assert mixed == mixed2.sorted()

    def test_reversed(self):
        mass = isopy.MassKeyList('104 108 102 110 106 105'.split()).reversed()
        assert mass != '104 108 102 110 106 105'.split()
        assert mass == list(reversed('104 108 102 110 106 105'.split()))

        element = isopy.ElementKeyList('pd cd te ag rh ru'.split()).reversed()
        assert element != 'pd cd te ag rh ru'.split()
        assert element == list(reversed('pd cd te ag rh ru'.split()))

        isotope = isopy.IsotopeKeyList('106cd 104ru 102ru 104pd 102pd 106pd'.split()).reversed()
        assert isotope != '106cd 104ru 102ru 104pd 102pd 106pd'.split()
        assert isotope == list(reversed('106cd 104ru 102ru 104pd 102pd 106pd'.split()))

        molecule = isopy.MoleculeKeyList('H2O HNO3 HCl (OH)2'.split()).reversed()
        assert molecule != 'H2O HNO3 HCl (OH)2'.split()
        assert molecule == list(reversed('H2O HNO3 HCl (OH)2'.split()))

        general = isopy.GeneralKeyList('hermione ginny luna ron neville harry'.split()).reversed()
        assert general != 'hermione ginny luna ron neville harry'.split()
        assert general == list(reversed('hermione ginny luna ron neville harry'.split()))

        ratio = (isopy.ElementKeyList('pd cd te ag rh ru'.split()) / 'pd').reversed()
        assert ratio.numerators != 'pd cd te ag rh ru'.split()
        assert ratio.numerators == list(reversed('pd cd te ag rh ru'.split()))

        ratio = ('pd' / isopy.ElementKeyList('pd cd te ag rh ru'.split())).reversed()
        assert ratio.denominators != 'pd cd te ag rh ru'.split()
        assert ratio.denominators == list(reversed('pd cd te ag rh ru'.split()))

        mixed = isopy.MixedKeyList('H2O 108pd/105pd 99ru 105 hermione pd'.split()).reversed()
        assert mixed != 'H2O 108pd/105pd 99ru 105 hermione pd'.split()
        assert mixed == list(reversed('H2O 108pd/105pd 99ru 105 hermione pd'.split()))

    def test_str(self):
        # Test the *strlist* method that turns key string into python strings

        key = isopy.MassKeyList('101')
        assert repr(key) == "MassKeyList('101')"
        assert key.strlist() == ['101']
        str_options = dict(key='101', m='101')
        for k, v in str_options.items():
            assert key.strlist(k) == [v]
            assert key.strlist(f'key {{{k}}}') == [f'key {v}']

        key = isopy.ElementKeyList('pd')
        assert repr(key) == "ElementKeyList('Pd')"
        assert key.strlist() == ['Pd']
        str_options = dict(key='Pd', es = 'pd', Es = 'Pd', ES = 'PD',
                           name = 'palladium', Name = 'Palladium', NAME = 'PALLADIUM')
        for k, v in str_options.items():
            assert key.strlist(k) == [v]
            assert key.strlist(f'key {{{k}}}') == [f'key {v}']

        key = isopy.IsotopeKeyList('101pd')
        assert repr(key) == "IsotopeKeyList('101Pd')"
        assert key.strlist() == ['101Pd']
        str_options = dict(key = '101Pd', m = '101', es = 'pd', Es = 'Pd', ES = 'PD',
                           name = 'palladium', Name = 'Palladium', NAME = 'PALLADIUM',
                           mEs = '101Pd', ESm = 'PD101', namem = 'palladium101', mNAME = '101PALLADIUM')
        str_options.update({'NAME-m': 'PALLADIUM-101', 'm-es': '101-pd'})
        for k, v in str_options.items():
            assert key.strlist(k) == [v]
            assert key.strlist(f'key {{{k}}}') == [f'key {v}']

        key = isopy.MoleculeKeyList('H2O')
        assert repr(key) == "MoleculeKeyList('H2O')"
        assert key.strlist() == ['H2O']
        str_options = dict(key='H2O')
        for k, v in str_options.items():
            assert key.strlist(k) == [v]
            assert key.strlist(f'key {{{k}}}') == [f'key {v}']

        key = isopy.RatioKeyList('pd/101pd')
        assert repr(key) == "RatioKeyList('Pd/101Pd')"
        assert key.strlist() == ['Pd/101Pd']
        str_options = {'key': 'Pd/101Pd', 'n/d': 'Pd/101Pd', 'n': 'Pd', 'd': '101Pd'}
        for k, v in str_options.items():
            assert key.strlist(k) == [v]
            assert key.strlist(f'key {{{k}}}') == [f'key {v}']
        assert key.strlist(('n/d', 'es', 'Name-m')) == ['pd/Palladium-101']
        assert key.strlist({'format': 'n/d', 'nformat': 'es', 'dformat': 'Name-m'}) == ['pd/Palladium-101']

        key = isopy.GeneralKeyList('hermione')
        assert repr(key) == "GeneralKeyList('hermione')"
        assert key.strlist() == ['hermione']
        str_options = dict(key='hermione')
        for k, v in str_options.items():
            assert key.strlist(k) == [v]
            assert key.strlist(f'key {{{k}}}') == [f'key {v}']

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

    def test_fractions(self):
        fractions1 = isopy.refval.isotope.fraction
        fractions2 = dict(h1=0.5, h2=0.5, o16=0.6, o17=0.2, o18=0.1, ba137=1, pd105=0.1)
        fractions3 = isopy.ScalarDict(fractions2)

        isotopes = isopy.IsotopeKeyList('105pd', '137Ba', '1H', '16O', '5H')
        assert type(isotopes.fractions()) is tuple
        assert len(isotopes.fractions()) == len(isotopes)
        np.testing.assert_array_almost_equal(isotopes.fractions(),
                                             [fractions1.get(key) for key in isotopes])
        np.testing.assert_array_almost_equal(isotopes.fractions(5),
                                             [fractions1.get(key, 5) for key in isotopes])
        np.testing.assert_array_almost_equal(isotopes.fractions(isotope_fractions=fractions2),
                                             [fractions3.get(key) for key in isotopes])

        molecules = isopy.MoleculeKeyList(['(1H)2(16O)', '(5H)2(16O)', '(1H)(2H)(16O)',
                                           '((16O)(1H))2', '((2H)(18O))3((16O)(1H))2'])
        molfrac1 = []
        molfrac2 = []
        molfrac1.append(fractions1.get('1h') ** 2 * fractions1.get('16o'))
        molfrac2.append(fractions3.get('1h') ** 2 * fractions3.get('16o'))
        molfrac1.append(fractions1.get('5h') ** 2 * fractions1.get('16o'))
        molfrac2.append(fractions3.get('5h') ** 2 * fractions3.get('16o'))
        molfrac1.append(fractions1.get('1h') * fractions1.get('2h') * fractions1.get('16o'))
        molfrac2.append(fractions3.get('1h') * fractions3.get('2h') * fractions3.get('16o'))
        molfrac1.append((fractions1.get('16o') * fractions1.get('1h')) ** 2)
        molfrac2.append((fractions3.get('16o') * fractions3.get('1h')) ** 2)
        molfrac1.append(((fractions1.get('16o') * fractions1.get('1h')) ** 2) * (
                    (fractions1.get('18o') * fractions1.get('2h')) ** 3))
        molfrac2.append(((fractions3.get('16o') * fractions3.get('1h')) ** 2) * (
                    (fractions3.get('18o') * fractions3.get('2h')) ** 3))

        np.testing.assert_array_almost_equal(molecules.fractions(), molfrac1)
        np.testing.assert_array_almost_equal(molecules.fractions(isotope_fractions=fractions2), molfrac2)

        molfrac1 = []
        molfrac1.append(fractions1.get('1h', 5) ** 2 * fractions1.get('16o', 5))
        molfrac1.append(fractions1.get('5h', 5) ** 2 * fractions1.get('16o', 5))
        molfrac1.append(fractions1.get('1h', 5) * fractions1.get('2h', 5) * fractions1.get('16o', 5))
        molfrac1.append((fractions1.get('16o', 5) * fractions1.get('1h', 5)) ** 2)
        molfrac1.append(((fractions1.get('16o', 5) * fractions1.get('1h', 5)) ** 2) * (
                (fractions1.get('18o', 5) * fractions1.get('2h', 5)) ** 3))

        assert type(molecules.fractions()) is tuple
        assert len(molecules.fractions()) == len(molecules)
        np.testing.assert_array_almost_equal(molecules.fractions(5), molfrac1)

    def test_getitem(self):
        keys = 'ru pd ag cd'.split()
        keylist = isopy.ElementKeyList(keys)

        for i in range(len(keys)):
            assert keylist[i] == keys[i]

        assert keylist[:] == keys
        assert keylist[:2] == keys[:2]
        assert keylist[1:3] == keys[1:3]

        assert keylist[(1,3)] == 'pd cd'.split()

        assert keylist[(2,)] == 'ag'
        assert isinstance(keylist[(2,)], core.ElementKeyList)

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



# 100 % coverage
class Test_Dict:
    def test_creation(self):
        # IsopyDict
        for v in [1, 1.4, {1,2,3}, [1,2,3], (1,2,3)]:
            with pytest.raises(TypeError):
                isopydict = isopy.IsopyDict(v)

        for v in ['str', [[1,2,3], [4,5,6]]]:
            with pytest.raises(ValueError):
                isopydict = isopy.IsopyDict(v)

        keys = '101 pd ru cd 105pd hermione'.split()
        values = [1, '2', 'a', [11, 12, 13], 5, 6]

        dictionary1 = dict(zip(keys, values))
        dictionary2 = dict(zip(keys[:3], values[:3]))
        dictionary3 = dict(zip(keys[3:], values[3:]))

        isopydict = isopy.IsopyDict(dictionary1)
        assert type(isopydict) is isopy.IsopyDict
        self.check_creation(isopydict, keys, values)
        assert isopydict.keylist() == keys

        isopydict = isopy.IsopyDict(**dictionary1)
        assert type(isopydict) is isopy.IsopyDict
        self.check_creation(isopydict, keys, values)
        assert isopydict.keylist() == keys

        isopydict = isopy.IsopyDict(dictionary2, dictionary3)
        assert type(isopydict) is isopy.IsopyDict
        self.check_creation(isopydict, keys, values)
        assert isopydict.keylist() == keys

        isopydict = isopy.IsopyDict(dictionary2)
        isopydict.update(dictionary3)
        assert type(isopydict) is isopy.IsopyDict
        self.check_creation(isopydict, keys, values)
        assert isopydict.keylist() == keys

        isopydict = isopy.IsopyDict(dictionary2)
        with pytest.raises(TypeError):
            isopydict.update({1,2,3})

        isopydict = isopy.IsopyDict(dictionary2)
        assert type(isopydict) is isopy.IsopyDict
        isopydict.update(dictionary3)
        self.check_creation(isopydict, keys, values)

        isopydict = isopy.IsopyDict(dictionary2, **dictionary3)
        assert type(isopydict) is isopy.IsopyDict
        self.check_creation(isopydict, keys, values)
        assert isopydict.keylist() == keys

        isopydict = isopy.IsopyDict(**dictionary2, **dictionary3)
        assert type(isopydict) is isopy.IsopyDict
        self.check_creation(isopydict, keys, values)
        assert isopydict.keylist() == keys

        isopydict = isopy.IsopyDict(zip(keys, values))
        assert type(isopydict) is isopy.IsopyDict
        self.check_creation(isopydict, keys, values)
        assert isopydict.keylist() == keys

        with pytest.raises(TypeError):
            # This doesnt work for isopy dicts at present
            isopydict = isopy.IsopyDict(values, keys)

        keys = '101 pd ru cd 105pd hermione'.split()
        values = [1, 2, 3, 4, 5, 6]
        array = isopy.array(values, keys)

        isopydict = isopy.IsopyDict(array)
        assert type(isopydict) is isopy.IsopyDict
        self.check_creation(isopydict, keys, values)

    def test_creation_scalar(self):
        # Scalar dict

        keys = '101 pd ru cd 105pd hermione'.split()
        values = [1, 2, 3, 4, 5, 6]

        dictionary1 = dict(zip(keys, values))
        dictionary2 = dict(zip(keys[:3], values[:3]))
        dictionary3 = dict(zip(keys[3:], values[3:]))

        isopydict = isopy.ScalarDict(dictionary1)
        assert type(isopydict) is isopy.ScalarDict
        self.check_creation_scalar(isopydict, keys, values, 0, 1)

        isopydict = isopy.ScalarDict(dictionary1, default_value=1)
        assert type(isopydict) is isopy.ScalarDict
        self.check_creation_scalar(isopydict, keys, values, 0, 1, 1)

        isopydict = isopy.ScalarDict(dictionary1, default_value=[1, 2])
        assert type(isopydict) is isopy.ScalarDict
        self.check_creation_scalar(isopydict, keys, values, 1, 2, [1, 2])

        isopydict = isopy.ScalarDict(**dictionary1)
        assert type(isopydict) is isopy.ScalarDict
        self.check_creation_scalar(isopydict, keys, values, 0, 1)

        isopydict = isopy.ScalarDict(dictionary2, dictionary3)
        assert type(isopydict) is isopy.ScalarDict
        self.check_creation_scalar(isopydict, keys, values, 0, 1)

        isopydict = isopy.ScalarDict(dictionary2)
        assert type(isopydict) is isopy.ScalarDict
        isopydict.update(dictionary3)
        self.check_creation_scalar(isopydict, keys, values, 0, 1)

        isopydict = isopy.ScalarDict(dictionary2, **dictionary3)
        assert type(isopydict) is isopy.ScalarDict
        self.check_creation_scalar(isopydict, keys, values, 0, 1)

        isopydict = isopy.ScalarDict(**dictionary2, **dictionary3)
        assert type(isopydict) is isopy.ScalarDict
        self.check_creation_scalar(isopydict, keys, values, 0, 1)

        isopydict = isopy.ScalarDict(zip(keys, values))
        assert type(isopydict) is isopy.ScalarDict
        self.check_creation_scalar(isopydict, keys, values, 0, 1)

        with pytest.raises(TypeError):
            # This doesnt work for isopy dicts at present
            isopydict = isopy.ScalarDict(values, keys)

        keys = '101 pd ru cd 105pd hermione'.split()
        values = [[1], [2], [3], [4], [5], [6]]

        dictionary1 = dict(zip(keys, values))
        dictionary2 = dict(zip(keys[:3], values[:3]))
        dictionary3 = dict(zip(keys[3:], values[3:]))

        isopydict = isopy.ScalarDict(dictionary1)
        assert type(isopydict) is isopy.ScalarDict
        self.check_creation_scalar(isopydict, keys, values, 0, 1)

        isopydict = isopy.ScalarDict(dictionary1, default_value=1)
        assert type(isopydict) is isopy.ScalarDict
        self.check_creation_scalar(isopydict, keys, values, 0, 1, 1)

        isopydict = isopy.ScalarDict(dictionary1, default_value=[1, 2])
        assert type(isopydict) is isopy.ScalarDict
        self.check_creation_scalar(isopydict, keys, values, 1, 2, [1, 2])

        isopydict = isopy.ScalarDict(**dictionary1)
        assert type(isopydict) is isopy.ScalarDict
        self.check_creation_scalar(isopydict, keys, values, 0, 1)

        isopydict = isopy.ScalarDict(dictionary2, dictionary3)
        assert type(isopydict) is isopy.ScalarDict
        self.check_creation_scalar(isopydict, keys, values, 0, 1)

        isopydict = isopy.ScalarDict(dictionary2)
        assert type(isopydict) is isopy.ScalarDict
        isopydict.update(dictionary3)
        self.check_creation_scalar(isopydict, keys, values, 0, 1)

        isopydict = isopy.ScalarDict(dictionary2, **dictionary3)
        assert type(isopydict) is isopy.ScalarDict
        self.check_creation_scalar(isopydict, keys, values, 0, 1)

        isopydict = isopy.ScalarDict(**dictionary2, **dictionary3)
        assert type(isopydict) is isopy.ScalarDict
        self.check_creation_scalar(isopydict, keys, values, 0, 1)

        isopydict = isopy.ScalarDict(zip(keys, values))
        assert type(isopydict) is isopy.ScalarDict
        self.check_creation_scalar(isopydict, keys, values, 0, 1)

        keys = '101 pd ru cd 105pd hermione'.split()
        values = [[1], 2, [3], [4,5], [5,7], 6]

        dictionary1 = dict(zip(keys, values))
        dictionary2 = dict(zip(keys[:3], values[:3]))
        dictionary3 = dict(zip(keys[3:], values[3:]))

        isopydict = isopy.ScalarDict(dictionary1)
        assert type(isopydict) is isopy.ScalarDict
        self.check_creation_scalar(isopydict, keys, values, 1, 2)

        isopydict = isopy.ScalarDict(dictionary1, default_value=1)
        assert type(isopydict) is isopy.ScalarDict
        self.check_creation_scalar(isopydict, keys, values, 1, 2, 1)

        isopydict = isopy.ScalarDict(dictionary1, default_value=[1, 2])
        assert type(isopydict) is isopy.ScalarDict
        self.check_creation_scalar(isopydict, keys, values, 1, 2, [1, 2])

        isopydict = isopy.ScalarDict(**dictionary1)
        assert type(isopydict) is isopy.ScalarDict
        self.check_creation_scalar(isopydict, keys, values, 1, 2)

        isopydict = isopy.ScalarDict(dictionary2, dictionary3)
        assert type(isopydict) is isopy.ScalarDict
        self.check_creation_scalar(isopydict, keys, values, 1, 2)

        isopydict = isopy.ScalarDict(dictionary2)
        assert type(isopydict) is isopy.ScalarDict
        isopydict.update(dictionary3)
        self.check_creation_scalar(isopydict, keys, values, 1, 2)

        isopydict = isopy.ScalarDict(dictionary2, **dictionary3)
        assert type(isopydict) is isopy.ScalarDict
        self.check_creation_scalar(isopydict, keys, values, 1, 2)

        isopydict = isopy.ScalarDict(**dictionary2, **dictionary3)
        assert type(isopydict) is isopy.ScalarDict
        self.check_creation_scalar(isopydict, keys, values, 1, 2)

        isopydict = isopy.ScalarDict(zip(keys, values))
        assert type(isopydict) is isopy.ScalarDict
        self.check_creation_scalar(isopydict, keys, values, 1, 2)

        keys = '101 pd ru cd 105pd hermione'.split()
        values = [[1, 2], 2, [3], [4, 5, 7], [5], 6]

        dictionary1 = dict(zip(keys, values))
        dictionary2 = dict(zip(keys[:3], values[:3]))
        dictionary3 = dict(zip(keys[3:], values[3:]))

        with pytest.raises(ValueError):
            isopy.ScalarDict(dictionary1)

        with pytest.raises(ValueError):
            isopy.ScalarDict(**dictionary1)

        with pytest.raises(ValueError):
            isopy.ScalarDict(dictionary2, dictionary3)

        isopydict = isopy.ScalarDict(dictionary2)
        with pytest.raises(ValueError):
            isopydict.update(dictionary3)

        with pytest.raises(ValueError):
            isopy.ScalarDict(dictionary2, **dictionary3)

        with pytest.raises(ValueError):
            isopy.ScalarDict(**dictionary2, **dictionary3)

        with pytest.raises(ValueError):
            isopy.ScalarDict(zip(keys, values))

        keys = '101 pd ru cd 105pd hermione'.split()
        values = [[1, 2], 2, [3], [], [5], 6]

        dictionary1 = dict(zip(keys, values))
        dictionary2 = dict(zip(keys[:3], values[:3]))
        dictionary3 = dict(zip(keys[3:], values[3:]))

        with pytest.raises(ValueError):
            isopy.ScalarDict(dictionary1)

        with pytest.raises(ValueError):
            isopy.ScalarDict(**dictionary1)

        with pytest.raises(ValueError):
            isopy.ScalarDict(dictionary2, dictionary3)

        isopydict = isopy.ScalarDict(dictionary2)
        with pytest.raises(ValueError):
            isopydict.update(dictionary3)

        with pytest.raises(ValueError):
            isopy.ScalarDict(dictionary2, **dictionary3)

        with pytest.raises(ValueError):
            isopy.ScalarDict(**dictionary2, **dictionary3)

        with pytest.raises(ValueError):
            isopy.ScalarDict(zip(keys, values))

        keys = '101 pd ru cd 105pd hermione'.split()
        values = [[1, 2], 2, [3], [[1], [2]], [5], 6]

        dictionary1 = dict(zip(keys, values))
        dictionary2 = dict(zip(keys[:3], values[:3]))
        dictionary3 = dict(zip(keys[3:], values[3:]))

        with pytest.raises(ValueError):
            isopy.ScalarDict(dictionary1)

        with pytest.raises(ValueError):
            isopy.ScalarDict(**dictionary1)

        with pytest.raises(ValueError):
            isopy.ScalarDict(dictionary2, dictionary3)

        isopydict = isopy.ScalarDict(dictionary2)
        with pytest.raises(ValueError):
            isopydict.update(dictionary3)

        with pytest.raises(ValueError):
            isopy.ScalarDict(dictionary2, **dictionary3)

        with pytest.raises(ValueError):
            isopy.ScalarDict(**dictionary2, **dictionary3)

        with pytest.raises(ValueError):
            isopy.ScalarDict(zip(keys, values))

    def check_creation(self, isopydict, keys, values, scalar = False):
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
            if scalar:
                assert type(value) == np.float64

        for i, key in enumerate(keys):
            assert key in isopydict
            assert isopydict[key] == values[i]

    def check_creation_scalar(self, isopydict, keys, values, ndim, size, default_value=np.nan):
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
            key = isopydict.keylist[i]
            assert value.dtype == np.float64
            assert value.ndim == ndim
            assert value.size == size

            np.testing.assert_allclose(value, np.full(size, values[keylist.index(key)]))

        for i, key in enumerate(isopydict.keys()):
            assert key in keylist

        for i, key in enumerate(keys):
            assert key in isopydict
            assert key in isopydict,keylist()

    def test_repr(self):
        keys = '101 pd ru cd 105pd hermione'.split()
        values = [1, '2', 'a', [11, 12, 13], 5, 6]

        isopydict = isopy.IsopyDict(dict(zip(keys, values)))
        repr(isopydict)
        str(isopydict)

        keys = '101 pd ru cd 105pd hermione'.split()
        values = [1, 2, 3, 4, 5, 6]

        isopydict = isopy.ScalarDict(dict(zip(keys, values)))
        repr(isopydict)
        str(isopydict)

    # Tests most methods
    def test_methods(self):
        keys = '101 pd ru cd 105pd hermione'.split()
        values = [1, '2', 'a', [11, 12, 13], 5, 6]

        dictionary1 = dict(zip(keys, values))

        # readonly = False

        isopydict = isopy.IsopyDict(dictionary1)
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
        isopydict = isopy.IsopyDict(dictionary1, default_value='default')
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

        with pytest.raises(AttributeError):
            isopydict.default_value = 'fail'

        # readonly = True

        isopydict = isopy.IsopyDict(dictionary1, readonly=True)
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
        value = isopydict.setdefault('ru')
        assert value == 'a'
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

        isopydict = isopy.ScalarDict(dictionary1, default_value=666)
        assert isopydict.default_value == 666
        assert isopydict.default_value.dtype == np.float64

        with pytest.raises(ValueError):
            isopydict['107ag'] = 'a'

        copy = isopydict.copy()
        assert copy is not isopydict
        assert copy.default_value == 666

        with pytest.raises(AttributeError):
            isopydict.default_value = 'fail'

        with pytest.raises(ValueError):
            isopydict = isopy.ScalarDict(dictionary1, default_value='a')

    def test_get_isopydict(self):
        keys = '101 pd ru cd 105pd hermione'.split()
        values = [1, '2', 'a', [11, 12, 13], 5, 6]
        subkeys1 = ['101', 'ru', '105pd', 'hermione']
        subkeys2 = ('105pd', '107ag', '103rh')

        dictionary = dict(zip(keys, values))
        isopydict1 = isopy.IsopyDict(dictionary)
        isopydict2 = isopy.IsopyDict(dictionary, default_value = 'default')

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

        # multiple values
        tup1 = isopydict1.get(subkeys1)
        tup2 = isopydict2.get(subkeys1)
        assert type(tup1) is tuple
        assert type(tup2) is tuple
        assert len(tup1) == len(subkeys1)
        assert len(tup2) == len(subkeys1)
        for i, key in enumerate(subkeys1):
            assert tup1[i] == dictionary.get(key)
            assert tup2[i] == dictionary.get(key)

        tup1 = isopydict1.get(subkeys2, None)
        tup2 = isopydict2.get(subkeys2, None)
        assert type(tup1) is tuple
        assert type(tup2) is tuple
        assert len(tup1) == len(subkeys2)
        assert len(tup2) == len(subkeys2)
        for i, key in enumerate(subkeys2):
            assert tup1[i] == dictionary.get(key, None)
            assert tup2[i] == dictionary.get(key, None)

        with pytest.raises(ValueError):
            tup1 = isopydict1.get(subkeys2)
        tup2 = isopydict2.get(subkeys2)
        assert len(tup2) == len(subkeys2)
        for i, key in enumerate(subkeys2):
            assert tup2[i] == dictionary.get(key, 'default')

        # Test with charge
        keys = 'ru++ cd- 101ru+ 105pd++ 108pd- 111cd--'.split()
        basekeys = 'ru cd 101ru 105pd 108pd 111cd'.split()
        values = [1, 2, 3, 4, 5, 6]
        dictionary1 = dict(zip(keys, values))
        dictionary2 = dict(zip(basekeys, values))

        isopydict1 = isopy.IsopyDict(dictionary1)
        isopydict2 = isopy.IsopyDict(dictionary2)
        for key in keys:
            assert isopy.keystring(key).charge is not None

            assert key in isopydict1
            assert key not in isopydict2

            assert isopydict1.get(key, 666) == dictionary1[key]
            assert isopydict2.get(key, 666) == dictionary1[key]
        assert isopydict1.get('137Ba++', 666) == 666
        assert isopydict2.get('137Ba++', 666) == 666

        for key in basekeys:
            assert isopy.keystring(key).charge is None

            assert key not in isopydict1
            assert key in isopydict2

            assert isopydict1.get(key, 666) == 666
            assert isopydict2.get(key, 666) == dictionary2[key]
        assert isopydict1.get('137Ba', 666) == 666
        assert isopydict2.get('137Ba', 666) == 666

    def test_get_scalardict(self):
        keys = '101 pd ru cd 105pd hermione'.split()
        values = [1, 2, 3, 4, 5, 6]
        subkeys1 = ['101', 'ru', '105pd', 'hermione']
        subkeys2 = ['105pd', '107ag', '103rh']
        subkeys3 =  ['105pd/ru', '107ag/ru']

        dictionary = dict(zip(keys, values))
        isopydict1 = isopy.ScalarDict(dictionary)
        isopydict2 = isopy.ScalarDict(dictionary, default_value=10)

        for key in keys:
            assert isopydict1[key] == dictionary.get(key)
            assert isopydict2[key] == dictionary.get(key)
            assert isopydict1[key] is not isopydict1[key]
            assert isopydict2[key] is not isopydict2[key]

            assert isopydict1.get(key) == dictionary.get(key)
            assert isopydict2.get(key) == dictionary.get(key)
            assert isopydict1.get(key) is not isopydict1.get(key)
            assert isopydict2.get(key) is not isopydict2.get(key)

            with pytest.raises(ValueError):
                isopydict1.get(key, [1, 2])
        assert isopydict1.get(101) == dictionary.get('101')
        assert isopydict2.get(101) == dictionary.get('101')
        for key in '103rh 107ag'.split():
            assert np.isnan(isopydict1.get(key))
            assert isopydict2.get(key) == 10

            assert isopydict1.get(key) is not isopydict1.default_value
            assert isopydict2.get(key) is not isopydict2.default_value

            value1 = isopydict1.get(key, 666)
            value2 = isopydict2.get(key, 666)

            assert value1 == 666
            assert value2 == 666
            assert value1.dtype == np.float64
            assert value2.dtype == np.float64

            with pytest.raises(ValueError):
                isopydict1.get(key, [1, 2])

        with pytest.raises(ValueError):
            isopydict1.get('ru', 'a')

        with pytest.raises(TypeError):
            isopydict1.get(3.14)


        # Test ratio
        assert isopydict1.get('105pd/ru') == 5/3
        assert isopydict2.get('105pd/ru') == 5/3
        assert isopydict1.get('105pd/ru', 666) == 5/3
        assert isopydict2.get('105pd/ru', 666) == 5/3
        assert isopydict1.get('105pd/ru') is not isopydict1.get('105pd/ru')
        assert isopydict2.get('105pd/ru') is not isopydict2.get('105pd/ru')
        assert isopydict1.get('105pd/ru', 666) is not isopydict1.get('105pd/ru', 666)
        assert isopydict2.get('105pd/ru', 666) is not isopydict2.get('105pd/ru', 666)

        # Nested list do not work
        assert np.isnan(isopydict1.get('105pd/ru//hermione')) # == (5 / 3) / 6
        assert isopydict2.get('105pd/ru//hermione') == 10 #(5 / 3) / 6
        assert isopydict1.get('105pd/ru//hermione', 666) == 666 #(5 / 3) / 6
        assert isopydict2.get('105pd/ru//hermione', 666) == 666 #(5 / 3) / 6

        assert np.isnan(isopydict1.get('105pd/ag'))
        assert isopydict2.get('105pd/ag') == 10
        assert isopydict1.get('105pd/ag', 666) == 666
        assert isopydict2.get('105pd/ag', 666) == 666

        assert np.isnan(isopydict1.get('107ag/ru'))
        assert isopydict2.get('107ag/ru') == 10
        assert isopydict1.get('107ag/ru', 666) == 666
        assert isopydict2.get('107ag/ru', 666) == 666

        # multiple values
        tup1 = isopydict1.get(subkeys1)
        tup2 = isopydict2.get(subkeys1)
        assert isinstance(tup1, core.IsopyArray)
        assert isinstance(tup2, core.IsopyArray)
        assert tup1.keys == subkeys1
        assert tup2.keys == subkeys1
        for key in subkeys1:
            np.testing.assert_allclose(tup1[key], dictionary.get(key))
            np.testing.assert_allclose(tup2[key], dictionary.get(key))

        tup1 = isopydict1.get(subkeys2, 666)
        tup2 = isopydict2.get(subkeys2, 666)
        assert isinstance(tup1, core.IsopyArray)
        assert isinstance(tup2, core.IsopyArray)
        assert tup1.keys == subkeys2
        assert tup2.keys == subkeys2
        for key in subkeys2:
            np.testing.assert_allclose(tup1[key], dictionary.get(key, 666))
            np.testing.assert_allclose(tup2[key], dictionary.get(key, 666))


        tup1 = isopydict1.get(subkeys2)
        tup2 = isopydict2.get(subkeys2)
        assert isinstance(tup1, core.IsopyArray)
        assert isinstance(tup2, core.IsopyArray)
        assert tup1.keys == subkeys2
        assert tup2.keys == subkeys2
        for key in subkeys2:
            np.testing.assert_allclose(tup1[key], dictionary.get(key, np.nan))
            np.testing.assert_allclose(tup2[key], dictionary.get(key, 10))

        tup1 = isopydict1.get(subkeys3)
        tup2 = isopydict2.get(subkeys3)
        assert isinstance(tup1, core.IsopyArray)
        assert isinstance(tup2, core.IsopyArray)
        assert tup1.keys == subkeys3
        assert tup2.keys == subkeys3
        for key in subkeys3:
            np.testing.assert_allclose(tup1[key], isopydict1.get(key, np.nan))
            np.testing.assert_allclose(tup2[key], isopydict2.get(key, 10))

        # Test with charge
        keys = 'ru++ cd- 101ru+ 105pd++ 108pd- 111cd--'.split()
        basekeys = 'ru cd 101ru 105pd 108pd 111cd'.split()
        values = [1, 2, 3, 4, 5, 6]
        dictionary1 = dict(zip(keys, values))
        dictionary2 = dict(zip(basekeys, values))

        isopydict1 = isopy.ScalarDict(dictionary1)
        isopydict2 = isopy.ScalarDict(dictionary2)
        for key in keys:
            assert isopy.keystring(key).charge is not None

            assert key in isopydict1
            assert key not in isopydict2

            assert isopydict1.get(key, 666) == dictionary1[key]
            assert isopydict1.get(key, 666) is not isopydict1.get(key, 666)
            assert isopydict2.get(key, 666) == dictionary1[key]
            assert isopydict2.get(key, 666) is not isopydict2.get(key, 666)
        assert isopydict1.get('137Ba++', 666) == 666
        assert isopydict2.get('137Ba++', 666) == 666

        for key in basekeys:
            assert isopy.keystring(key).charge is None

            assert key not in isopydict1
            assert key in isopydict2

            assert isopydict1.get(key, 666) == 666
            assert isopydict2.get(key, 666) == dictionary2[key]
            assert isopydict2.get(key, 666) is not isopydict2.get(key, 666)
        assert isopydict1.get('137Ba', 666) == 666
        assert isopydict2.get('137Ba', 666) == 666

    def test_copy(self):
        # IsopyDict
        keys = 'ru cd 101ru 105pd 108pd 111cd'.split()
        values = [1, '2', 'a', [11, 12, 13], 5, 6, 7]

        subkeys1 = 'ru cd'.split()
        subkeys2 = '101ru 105pd 108pd 111cd'.split()
        subkeys3 = '108pd 111cd'.split()
        subkeys4 = '101ru 111cd'.split()

        isopydict = isopy.IsopyDict(dict(zip(keys, values)), default_value='default')
        filtered = isopydict.copy(flavour_eq ='element')
        assert type(filtered) is type(isopydict)
        assert filtered.default_value == 'default'
        assert filtered is not isopydict
        assert len(filtered) == len(subkeys1)
        for key in subkeys1:
            assert filtered[key] == isopydict[key]

        filtered = isopydict.copy(flavour_eq='isotope')
        assert type(filtered) is type(isopydict)
        assert filtered.default_value == 'default'
        assert filtered is not isopydict
        assert len(filtered) == len(subkeys2)
        for key in subkeys2:
            assert filtered[key] == isopydict[key]

        filtered = isopydict.copy(mass_number_gt = 105)
        assert type(filtered) is type(isopydict)
        assert filtered.default_value == 'default'
        assert filtered is not isopydict
        assert len(filtered) == len(subkeys3)
        for key in subkeys3:
            assert filtered[key] == isopydict[key]

        filtered = isopydict.copy(element_symbol_eq = ['ru', 'rh', 'ag', 'cd'])
        assert type(filtered) is type(isopydict)
        assert filtered.default_value == 'default'
        assert filtered is not isopydict
        assert len(filtered) == len(subkeys4)
        for key in subkeys4:
            assert filtered[key] == isopydict[key]

        filtered = isopydict.copy(numerator_eq=['ru', 'rh', 'ag', 'cd'])
        assert type(filtered) is type(isopydict)
        assert filtered.default_value == 'default'
        assert filtered is not isopydict
        assert len(filtered) == 0

        # ScalarDict
        keys = 'ru cd 101ru 105pd 108pd 111cd'.split()
        values = [1, 2, 3, 4, 5, 6]

        subkeys1 = 'ru cd'.split()
        subkeys2 = '101ru 105pd 108pd 111cd'.split()
        subkeys3 = '108pd 111cd'.split()
        subkeys4 = '101ru 111cd'.split()

        isopydict = isopy.ScalarDict(dict(zip(keys, values)), default_value=666)
        filtered = isopydict.copy(flavour_eq='element')
        assert type(filtered) is type(isopydict)
        assert filtered.default_value == 666
        assert filtered is not isopydict
        assert len(filtered) == len(subkeys1)
        for key in subkeys1:
            assert filtered[key] == isopydict[key]

        filtered = isopydict.copy(flavour_eq='isotope')
        assert type(filtered) is type(isopydict)
        assert filtered.default_value == 666
        assert filtered is not isopydict
        assert len(filtered) == len(subkeys2)
        for key in subkeys2:
            assert filtered[key] == isopydict[key]

        filtered = isopydict.copy(mass_number_gt=105)
        assert type(filtered) is type(isopydict)
        assert filtered.default_value == 666
        assert filtered is not isopydict
        assert len(filtered) == len(subkeys3)
        for key in subkeys3:
            assert filtered[key] == isopydict[key]

        filtered = isopydict.copy(element_symbol_eq=['ru', 'rh', 'ag', 'cd'])
        assert type(filtered) is type(isopydict)
        assert filtered.default_value == 666
        assert filtered is not isopydict
        assert len(filtered) == len(subkeys4)
        for key in subkeys4:
            assert filtered[key] == isopydict[key]

        filtered = isopydict.copy(numerator_eq=['ru', 'rh', 'ag', 'cd'])
        assert type(filtered) is type(isopydict)
        assert filtered.default_value == 666
        assert filtered is not isopydict
        assert len(filtered) == 0

        filtered = isopydict.copy()
        assert type(filtered) is type(isopydict)
        assert filtered.default_value == 666
        assert filtered is not isopydict
        assert len(filtered) == len(isopydict)
        for key in isopydict:
            assert filtered[key] == isopydict[key]

    def test_to_array(self):
        keys = 'ru cd 101ru 105pd 108pd 111cd'.split()
        values = [1, 2, 3, 4, 5, 6]

        subkeys1 = 'ru pd cd'.split()
        subkeys2 = 'ru pd 108pd 111cd'.split()
        subkeys3 = '101ru 105pd 108pd 111cd'.split()
        subkeys4 = 'ru 108pd 111cd'.split()

        dictionary = dict(zip(keys, values))
        isodict1 = isopy.ScalarDict(dictionary)
        isodict2 = isopy.ScalarDict(dictionary, default_value=0)

        array1 = isodict1.to_array(subkeys1)
        array2 = isodict2.to_array(subkeys1)
        assert isinstance(array1, core.IsopyArray)
        assert isinstance(array2, core.IsopyArray)
        assert array1.flavour == 'element'
        assert array2.flavour == 'element'
        assert array1.ndim == 0
        assert array2.ndim == 0
        assert array1.keys == subkeys1
        assert array2.keys == subkeys1
        for key in subkeys1:
            np.testing.assert_allclose(array1[key], isodict1.get(key))
            np.testing.assert_allclose(array2[key], isodict2.get(key))
            np.testing.assert_allclose(array2[key], isodict2.get(key, default=0))

        assert array1 == isodict1.asarray(subkeys1)
        assert array2 == isodict2.asarray(subkeys1)

        array1 = isodict1.to_array(subkeys2)
        array2 = isodict2.to_array(subkeys2)
        assert isinstance(array1, core.IsopyArray)
        assert isinstance(array2, core.IsopyArray)
        assert array1.flavour == 'mixed'
        assert array2.flavour == 'mixed'
        assert array1.ndim == 0
        assert array2.ndim == 0
        assert array1.keys == subkeys2
        assert array2.keys == subkeys2
        for key in subkeys2:
            np.testing.assert_allclose(array1[key], isodict1.get(key))
            np.testing.assert_allclose(array2[key], isodict2.get(key))
            np.testing.assert_allclose(array2[key], isodict2.get(key, default=0))

        assert array1 == isodict1.asarray(subkeys2)
        assert array2 == isodict2.asarray(subkeys2)

        array1 = isodict1.to_array(flavour_eq ='isotope')
        array2 = isodict2.to_array(flavour_eq ='isotope')
        assert isinstance(array1, core.IsopyArray)
        assert isinstance(array2, core.IsopyArray)
        assert array1.flavour == 'isotope'
        assert array2.flavour == 'isotope'
        assert array1.ndim == 0
        assert array2.ndim == 0
        assert array1.keys == subkeys3
        assert array2.keys == subkeys3
        for key in subkeys3:
            np.testing.assert_allclose(array1[key], isodict1.get(key))
            np.testing.assert_allclose(array2[key], isodict2.get(key))
            np.testing.assert_allclose(array2[key], isodict2.get(key, default=0))

        assert array1 == isodict1.asarray(flavour_eq ='isotope')
        assert array2 == isodict2.asarray(flavour_eq ='isotope')

        array1 = isodict1.to_array(key_eq=subkeys2)
        array2 = isodict2.to_array(key_eq=subkeys2)
        assert isinstance(array1, core.IsopyArray)
        assert isinstance(array2, core.IsopyArray)
        assert array1.flavour == 'mixed'
        assert array2.flavour == 'mixed'
        assert array1.ndim == 0
        assert array2.ndim == 0
        assert array1.keys == subkeys4
        assert array2.keys == subkeys4
        for key in subkeys4:
            np.testing.assert_allclose(array1[key], isodict1.get(key))
            np.testing.assert_allclose(array2[key], isodict2.get(key))
            np.testing.assert_allclose(array2[key], isodict2.get(key, default=0))

        assert array1 == isodict1.asarray(key_eq=subkeys2)
        assert array2 == isodict2.asarray(key_eq=subkeys2)

#TODO test table
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
                isopy.array(data_list[1:], keys)

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

        # 1-dim input, n == 1
        data_list = [[1, 2, 3, 4]]
        data_tuple = [(1, 2, 3, 4)]
        data_array = np.array(data_list, dtype=np.float_)
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

    def test_create_array_dataframe(self):
        pass

    def test_create_array_table(self):
        pass

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
        else:
            assert result.nrows == result.size
            assert len(result) == result.size

            assert isinstance(result[0], core.IsopyVoid)

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
        assert array2.keys == 'ru cd pd'.split()
        for key in array2.keys:
            np.testing.assert_allclose(array2[key], array[key] / array['pd'])

        array2 = ratio.deratio(100)
        assert array2.flavour == 'element'
        assert array2.keys == 'ru cd pd'.split()
        for key in array2.keys:
            np.testing.assert_allclose(array2[key], array[key] / array['pd'] * 100)

        array2 = ratio.deratio(array['pd'])
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
        assert array2.keys == 'ru cd pd'.split()
        for key in array2.keys:
            np.testing.assert_allclose(array2[key], array[key] / array['pd'])

        array2 = ratio.deratio(100)
        assert array2.flavour == 'element'
        assert array2.keys == 'ru cd pd'.split()
        for key in array2.keys:
            np.testing.assert_allclose(array2[key], array[key] / array['pd'] * 100)

        array2 = ratio.deratio(array['pd'])
        assert array2.flavour == 'element'
        assert array2.keys == 'ru cd pd'.split()
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

        array2 = ratio.deratio(100)
        assert array2.flavour == 'element'
        assert array2.keys == 'ru pd cd'.split()
        for key in array2.keys:
            np.testing.assert_allclose(array2[key], array[key] / array['pd'] * 100)

        array2 = ratio.deratio(array['pd'])
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

    def test_get_set_column(self):
        # Sorry about this. Through though
        # Ndarray

        array = isopy.ones(2, 'ru 105pd 107ag cd'.split())
        assert array.flavour ==  'mixed'

        with pytest.raises(ValueError):
            array['ag']

        with pytest.raises(KeyError):
            array['ru', 'ag']

        row = array[[]]
        assert type(row) is np.ndarray
        assert row.size == 0

        row = array[tuple()]
        assert type(row) is np.ndarray
        assert row.size == 0

        row = array[1:1]
        assert row.size == 0

        # set to scalar

        array = isopy.ones(2, 'ru 105pd 107ag cd'.split())
        subarray = array['105pd']
        assert type(subarray) is np.ndarray
        assert not isinstance(subarray, core.IsopyArray)
        np.testing.assert_allclose(subarray, [1, 1])

        subarray[:] = 3
        np.testing.assert_allclose(subarray, [3, 3])
        np.testing.assert_allclose(array['105pd'], [3, 3])

        array['105pd'] = 4
        np.testing.assert_allclose(subarray, [4, 4])
        np.testing.assert_allclose(array['105pd'], [4, 4])

        # tuple 1
        array = isopy.ones(2, 'ru 105pd 107ag cd'.split())
        subarray = array['cd', '107ag']
        assert isinstance(subarray, core.IsopyArray)
        assert subarray.flavour == 'mixed'
        assert subarray.keys == ['cd', '107ag']

        subarray[:] = 3
        for key in subarray.keys:
            np.testing.assert_allclose(subarray[key], [3, 3])
            np.testing.assert_allclose(array[key], [3, 3])

        array = isopy.ones(2, 'ru 105pd 107ag cd'.split())
        subarray = array['cd', '107ag']

        subarray['cd', '107ag'] = 3
        for key in subarray.keys:
            np.testing.assert_allclose(subarray[key], [3, 3])
            np.testing.assert_allclose(array[key], [3, 3])

        # tuple 2
        array = isopy.ones(2, 'ru 105pd 107ag cd'.split())
        subarray = array[('cd', 'ru')]
        assert isinstance(subarray, core.IsopyArray)
        assert subarray.flavour == 'element'
        assert subarray.keys == ['cd', 'ru']

        subarray[:] = 3
        for key in subarray.keys:
            np.testing.assert_allclose(subarray[key], [3, 3])
            np.testing.assert_allclose(array[key], [3, 3])

        array = isopy.ones(2, 'ru 105pd 107ag cd'.split())
        subarray = array[('cd', 'ru')]

        subarray[('cd', 'ru')] = 3
        for key in subarray.keys:
            np.testing.assert_allclose(subarray[key], [3,3])
            np.testing.assert_allclose(array[key], [3, 3])

        # list
        array = isopy.ones(2, 'ru 105pd 107ag cd'.split())
        subarray = array[['105pd', '107ag']]
        assert isinstance(subarray, core.IsopyArray)
        assert subarray.flavour == 'isotope'
        assert subarray.keys == ['105pd', '107ag']

        array = isopy.ones(2, 'ru 105pd 107ag cd'.split())
        subarray = array[['105pd', '107ag']]

        subarray[['105pd', '107ag']] = 3
        for key in subarray.keys:
            np.testing.assert_allclose(subarray[key], [3, 3])
            np.testing.assert_allclose(array[key], [3, 3])

        # set to sequence of scalars

        array = isopy.ones(2, 'ru 105pd 107ag cd'.split())
        subarray = array['105pd']

        subarray[:] = [3,5]
        np.testing.assert_allclose(subarray, [3,5])
        np.testing.assert_allclose(array['105pd'], [3, 5])

        array['105pd'] = [4, 6]
        np.testing.assert_allclose(subarray, [4, 6])
        np.testing.assert_allclose(array['105pd'], [4, 6])

        #tuple 1
        array = isopy.ones(2, 'ru 105pd 107ag cd'.split())
        subarray = array['cd', '107ag']
        assert isinstance(subarray, core.IsopyArray)
        assert subarray.flavour == 'mixed'
        assert subarray.keys == ['cd', '107ag']

        subarray[:] = [3, 5]
        for key in subarray.keys:
            np.testing.assert_allclose(subarray[key], [3, 5])
            np.testing.assert_allclose(array[key], [3, 5])

        array = isopy.ones(2, 'ru 105pd 107ag cd'.split())
        subarray = array['cd', '107ag']

        subarray['cd', '107ag'] = [3, 5]
        for key in subarray.keys:
            np.testing.assert_allclose(subarray[key], [3, 5])
            np.testing.assert_allclose(array[key], [3, 5])

        # tuple 2
        array = isopy.ones(2, 'ru 105pd 107ag cd'.split())
        subarray = array[('cd', 'ru')]
        assert isinstance(subarray, core.IsopyArray)
        assert subarray.flavour == 'element'
        assert subarray.keys == ['cd', 'ru']

        subarray[:] = [3, 5]
        for key in subarray.keys:
            np.testing.assert_allclose(subarray[key], [3, 5])
            np.testing.assert_allclose(array[key], [3, 5])

        array = isopy.ones(2, 'ru 105pd 107ag cd'.split())
        subarray = array[('cd', 'ru')]

        subarray[('cd', 'ru')] = [3, 5]
        for key in subarray.keys:
            np.testing.assert_allclose(subarray[key], [3, 5])
            np.testing.assert_allclose(array[key], [3, 5])

        # list
        array = isopy.ones(2, 'ru 105pd 107ag cd'.split())
        subarray = array[['105pd', '107ag']]
        assert isinstance(subarray, core.IsopyArray)
        assert subarray.flavour == 'isotope'
        assert subarray.keys == ['105pd', '107ag']

        array = isopy.ones(2, 'ru 105pd 107ag cd'.split())
        subarray = array[['105pd', '107ag']]

        subarray[['105pd', '107ag']] = [3, 5]
        for key in subarray.keys:
            np.testing.assert_allclose(subarray[key], [3, 5])
            np.testing.assert_allclose(array[key], [3, 5])

        # set to isopy array

        array = isopy.ones(2, 'ru 105pd 107ag cd'.split())
        other = isopy.array(dict(ru=[3, 5], ag107=[3, 6], cd=[4, 6]))
        subarray = array['cd']
        array['cd'] = other
        np.testing.assert_allclose(subarray, [4, 6])
        np.testing.assert_allclose(array['cd'], [4, 6])

        # tuple1
        array = isopy.ones(2, 'ru 105pd 107ag cd'.split())
        subarray = array['cd', '107ag']
        assert isinstance(subarray, core.IsopyArray)
        assert subarray.flavour == 'mixed'
        assert subarray.keys == ['cd', '107ag']

        other = isopy.array(dict(ru = [3 ,5], ag107 = [3, 6], cd=[4, 6]))
        subarray[:] = other
        for key in subarray.keys:
            np.testing.assert_allclose(subarray[key], other.get(key))
            np.testing.assert_allclose(array[key], other.get(key))

        array = isopy.ones(2, 'ru 105pd 107ag cd'.split())
        subarray = array['cd', '107ag']

        other = isopy.array(dict(ru=[3, 5], ag107=[3, 6], cd=[4, 6]))
        subarray['cd', '107ag'] = other
        for key in subarray.keys:
            np.testing.assert_allclose(subarray[key], other.get(key))
            np.testing.assert_allclose(array[key], other.get(key))

        # tuple2
        array = isopy.ones(2, 'ru 105pd 107ag cd'.split())
        subarray = array[('cd', 'ru')]
        assert isinstance(subarray, core.IsopyArray)
        assert subarray.flavour == 'element'
        assert subarray.keys == ['cd', 'ru']

        other = isopy.array(dict(ru=3, ag107=3, cd=4))
        subarray[:] = other
        for key in subarray.keys:
            np.testing.assert_allclose(subarray[key], [other.get(key), other.get(key)])
            np.testing.assert_allclose(array[key], [other.get(key), other.get(key)])

        array = isopy.ones(2, 'ru 105pd 107ag cd'.split())
        subarray = array[('cd', 'ru')]

        other = isopy.array(dict(ru=3, ag107=3, cd=4))
        subarray[('cd', 'ru')] = other
        for key in subarray.keys:
            np.testing.assert_allclose(subarray[key], [other.get(key), other.get(key)])
            np.testing.assert_allclose(array[key], [other.get(key), other.get(key)])

        # list
        array = isopy.ones(2, 'ru 105pd 107ag cd'.split())
        subarray = array[['105pd', '107ag']]
        assert isinstance(subarray, core.IsopyArray)
        assert subarray.flavour == 'isotope'
        assert subarray.keys == ['105pd', '107ag']

        other = dict(ru=[3], ag107=[3], cd=[4])
        subarray[:] = other
        for key in ['105pd', 'ag107']:
            np.testing.assert_allclose(subarray[key], [other.get(key, [np.nan])[0], other.get(key, [np.nan])[0]])
            np.testing.assert_allclose(array[key], [other.get(key, [np.nan])[0], other.get(key, [np.nan])[0]])

        array = isopy.ones(2, 'ru 105pd 107ag cd'.split())
        subarray = array[['105pd', '107ag']]

        other = dict(ru=[3], ag107=[3], cd=[4])
        subarray[['105pd', 'ag107']] = other
        for key in ['105pd', 'ag107']:
            np.testing.assert_allclose(subarray[key], [other.get(key, [np.nan])[0], other.get(key, [np.nan])[0]])
            np.testing.assert_allclose(array[key], [other.get(key, [np.nan])[0], other.get(key, [np.nan])[0]])

        # Void

        array = isopy.ones(2, 'ru 105pd 107ag cd'.split())[0]
        assert array.flavour == 'mixed'

        with pytest.raises(ValueError):
            array['ag']

        with pytest.raises(KeyError):
            array['ru', 'ag']

        # set to scalar

        array = isopy.ones(2, 'ru 105pd 107ag cd'.split())[0]
        subarray = array['105pd']
        assert type(subarray) is np.float64
        assert not isinstance(subarray, core.IsopyArray)
        np.testing.assert_allclose(subarray, 1)

        array['105pd'] = 4
        assert subarray != 4
        np.testing.assert_allclose(array['105pd'], 4)

        # tuple 1
        array = isopy.ones(2, 'ru 105pd 107ag cd'.split())[0]
        subarray = array['cd', '107ag']
        assert isinstance(subarray, core.IsopyArray)
        assert subarray.flavour == 'mixed'
        assert subarray.keys == ['cd', '107ag']

        subarray[:] = 3
        for key in subarray.keys:
            np.testing.assert_allclose(subarray[key], 3)
            np.testing.assert_allclose(array[key], 3)

        array = isopy.ones(2, 'ru 105pd 107ag cd'.split())[0]
        subarray = array['cd', '107ag']

        subarray['cd', '107ag'] = 3
        for key in subarray.keys:
            np.testing.assert_allclose(subarray[key], 3)
            np.testing.assert_allclose(array[key], 3)

        # tuple 2
        array = isopy.ones(2, 'ru 105pd 107ag cd'.split())[0]
        subarray = array[('cd', 'ru')]
        assert isinstance(subarray, core.IsopyArray)
        assert subarray.flavour == 'element'
        assert subarray.keys == ['cd', 'ru']

        subarray[:] = 3
        for key in subarray.keys:
            np.testing.assert_allclose(subarray[key], 3)
            np.testing.assert_allclose(array[key], 3)

        array = isopy.ones(2, 'ru 105pd 107ag cd'.split())[0]
        subarray = array[('cd', 'ru')]

        subarray[('cd', 'ru')] = 3
        for key in subarray.keys:
            np.testing.assert_allclose(subarray[key], 3)
            np.testing.assert_allclose(array[key], 3)

        # list
        array = isopy.ones(2, 'ru 105pd 107ag cd'.split())[0]
        subarray = array[['105pd', '107ag']]
        assert isinstance(subarray, core.IsopyArray)
        assert subarray.flavour == 'isotope'
        assert subarray.keys == ['105pd', '107ag']

        array = isopy.ones(2, 'ru 105pd 107ag cd'.split())[0]
        subarray = array[['105pd', '107ag']]

        subarray[['105pd', '107ag']] = 3
        for key in subarray.keys:
            np.testing.assert_allclose(subarray[key], 3)
            np.testing.assert_allclose(array[key], 3)

        # set to sequence of scalars

        array = isopy.ones(2, 'ru 105pd 107ag cd'.split())[0]
        subarray = array['105pd']

        array['105pd'] = [4]
        assert subarray != 4
        np.testing.assert_allclose(array['105pd'], 4)

        # tuple 1
        array = isopy.ones(2, 'ru 105pd 107ag cd'.split())[0]
        subarray = array['cd', '107ag']
        assert isinstance(subarray, core.IsopyArray)
        assert subarray.flavour == 'mixed'
        assert subarray.keys == ['cd', '107ag']

        subarray[:] = [3]
        for key in subarray.keys:
            np.testing.assert_allclose(subarray[key], 3)
            np.testing.assert_allclose(array[key], 3)

        array = isopy.ones(2, 'ru 105pd 107ag cd'.split())[0]
        subarray = array['cd', '107ag']

        subarray['cd', '107ag'] = [3]
        for key in subarray.keys:
            np.testing.assert_allclose(subarray[key], 3)
            np.testing.assert_allclose(array[key], 3)

        # tuple 2
        array = isopy.ones(2, 'ru 105pd 107ag cd'.split())[0]
        subarray = array[('cd', 'ru')]
        assert isinstance(subarray, core.IsopyArray)
        assert subarray.flavour == 'element'
        assert subarray.keys == ['cd', 'ru']

        subarray[:] = [3]
        for key in subarray.keys:
            np.testing.assert_allclose(subarray[key], 3)
            np.testing.assert_allclose(array[key], 3)

        array = isopy.ones(2, 'ru 105pd 107ag cd'.split())[0]
        subarray = array[('cd', 'ru')]

        subarray[('cd', 'ru')] = [3]
        for key in subarray.keys:
            np.testing.assert_allclose(subarray[key], 3)
            np.testing.assert_allclose(array[key], 3)

        # list
        array = isopy.ones(2, 'ru 105pd 107ag cd'.split())[0]
        subarray = array[['105pd', '107ag']]
        assert isinstance(subarray, core.IsopyArray)
        assert subarray.flavour == 'isotope'
        assert subarray.keys == ['105pd', '107ag']

        array = isopy.ones(2, 'ru 105pd 107ag cd'.split())[0]
        subarray = array[['105pd', '107ag']]

        subarray[['105pd', '107ag']] = [3]
        for key in subarray.keys:
            np.testing.assert_allclose(subarray[key], 3)
            np.testing.assert_allclose(array[key], 3)

        # set to isopy array

        array = isopy.ones(2, 'ru 105pd 107ag cd'.split())[0]
        other = isopy.array(dict(ru=[3], ag107=[5], cd=[4]))
        subarray = array['cd']
        array['cd'] = other
        assert subarray != 4
        np.testing.assert_allclose(array['cd'], 4)

        # tuple1
        array = isopy.ones(2, 'ru 105pd 107ag cd'.split())[0]
        subarray = array['cd', '107ag']
        assert isinstance(subarray, core.IsopyArray)
        assert subarray.flavour == 'mixed'
        assert subarray.keys == ['cd', '107ag']

        other = isopy.array(dict(ru=[3], ag107=[5], cd=[4]))
        subarray[:] = other
        for key in subarray.keys:
            np.testing.assert_allclose(subarray[key], other.get(key)[0])
            np.testing.assert_allclose(array[key], other.get(key)[0])

        array = isopy.ones(2, 'ru 105pd 107ag cd'.split())[0]
        subarray = array['cd', '107ag']

        other = isopy.array(dict(ru=[3], ag107=[5], cd=[4]))
        subarray['cd', '107ag'] = other
        for key in subarray.keys:
            np.testing.assert_allclose(subarray[key], other.get(key)[0])
            np.testing.assert_allclose(array[key], other.get(key)[0])

        # tuple2
        array = isopy.ones(2, 'ru 105pd 107ag cd'.split())[0]
        subarray = array[('cd', 'ru')]
        assert isinstance(subarray, core.IsopyArray)
        assert subarray.flavour == 'element'
        assert subarray.keys == ['cd', 'ru']

        other = isopy.array(dict(ru=3, ag107=5, cd=4))
        subarray[:] = other
        for key in subarray.keys:
            np.testing.assert_allclose(subarray[key], other.get(key))
            np.testing.assert_allclose(array[key], other.get(key))

        array = isopy.ones(2, 'ru 105pd 107ag cd'.split())[0]
        subarray = array[('cd', 'ru')]

        other = isopy.array(dict(ru=3, ag107=5, cd=4))
        subarray[('cd', 'ru')] = other
        for key in subarray.keys:
            np.testing.assert_allclose(subarray[key], other.get(key))
            np.testing.assert_allclose(array[key], other.get(key))

        # list
        array = isopy.ones(2, 'ru 105pd 107ag cd'.split())[0]
        subarray = array[['105pd', '107ag']]
        assert isinstance(subarray, core.IsopyArray)
        assert subarray.flavour == 'isotope'
        assert subarray.keys == ['105pd', '107ag']

        other = dict(ru=[3], ag107=[3], cd=[4])
        subarray[:] = other
        for key in ['105pd', 'ag107']:
            np.testing.assert_allclose(subarray[key], other.get(key, [np.nan])[0])
            np.testing.assert_allclose(array[key], other.get(key, [np.nan])[0])

        array = isopy.ones(2, 'ru 105pd 107ag cd'.split())[0]
        subarray = array[['105pd', '107ag']]

        other = dict(ru=[3], ag107=[3], cd=[4])
        subarray[['105pd', '107ag']] = other
        for key in ['105pd', 'ag107']:
            np.testing.assert_allclose(subarray[key], other.get(key, [np.nan])[0])
            np.testing.assert_allclose(array[key], other.get(key, [np.nan])[0])

    def test_get_set_row(self):
        # scalar
        array = isopy.ones(4, 'ru 105pd 107ag cd'.split())
        row = array[1]
        row[:] = 2
        for key in array.keys():
            np.testing.assert_allclose(row[key], 2)
            np.testing.assert_allclose(array[key], [1, 2, 1, 1])

        array = isopy.ones(4, 'ru 105pd 107ag cd'.split())
        row = array[1]
        array[1] = 2
        for key in array.keys():
            np.testing.assert_allclose(row[key], 2)
            np.testing.assert_allclose(array[key], [1, 2, 1, 1])

        array = isopy.ones(4, 'ru 105pd 107ag cd'.split())
        row = array[1]
        row[:] = [2]
        for key in array.keys():
            np.testing.assert_allclose(row[key], 2)
            np.testing.assert_allclose(array[key], [1, 2, 1, 1])

        array = isopy.ones(4, 'ru 105pd 107ag cd'.split())
        row = array[1]
        array[1] = [2]
        for key in array.keys():
            np.testing.assert_allclose(row[key], 2)
            np.testing.assert_allclose(array[key], [1, 2, 1, 1])

        array = isopy.ones(4, 'ru 105pd 107ag cd'.split())
        row = array[1]
        with pytest.raises(IndexError):
            row[0]
        with pytest.raises(IndexError):
            row[0] = 1
        with pytest.raises(TypeError):
            len(row)

        array = isopy.ones(None, 'ru 105pd 107ag cd'.split())
        with pytest.raises(IndexError):
            array[0]
        with pytest.raises(IndexError):
            array[0] = 1
        with pytest.raises(TypeError):
            len(array)

        # sequence of scalars
        array = isopy.ones(4, 'ru 105pd 107ag cd'.split())
        row = array[1:3]
        row[:] = 2
        for key in array.keys():
            np.testing.assert_allclose(row[key], [2, 2])
            np.testing.assert_allclose(array[key], [1, 2, 2, 1])

        array = isopy.ones(4, 'ru 105pd 107ag cd'.split())
        row = array[1:3]
        array[1:3] = 2
        for key in array.keys():
            np.testing.assert_allclose(row[key], [2, 2])
            np.testing.assert_allclose(array[key], [1, 2, 2, 1])

        array = isopy.ones(4, 'ru 105pd 107ag cd'.split())
        row = array[1:3]
        row[:] = [2, 3]
        for key in array.keys():
            np.testing.assert_allclose(row[key], [2, 3])
            np.testing.assert_allclose(array[key], [1, 2, 3, 1])

        array = isopy.ones(4, 'ru 105pd 107ag cd'.split())
        row = array[1:3]
        array[1:3] = [2, 3]
        for key in array.keys():
            np.testing.assert_allclose(row[key], [2, 3])
            np.testing.assert_allclose(array[key], [1, 2, 3, 1])

        # This does not return a view
        array = isopy.ones(4, 'ru 105pd 107ag cd'.split())
        row = array[1, 3]
        row[:] = 2
        for key in array.keys():
            np.testing.assert_allclose(row[key], [2, 2])
            np.testing.assert_allclose(array[key], [1, 1, 1, 1])

        array = isopy.ones(4, 'ru 105pd 107ag cd'.split())
        row = array[1, 3]
        array[1, 3] = 2
        for key in array.keys():
            np.testing.assert_allclose(row[key], [1, 1])
            np.testing.assert_allclose(array[key], [1, 2, 1, 2])

        array = isopy.ones(4, 'ru 105pd 107ag cd'.split())
        row = array[(1, 3)]
        row[:] = [2, 3]
        for key in array.keys():
            np.testing.assert_allclose(row[key], [2, 3])
            np.testing.assert_allclose(array[key], [1, 1, 1, 1])

        array = isopy.ones(4, 'ru 105pd 107ag cd'.split())
        row = array[(1, 3)]
        array[(1, 3)] = [2, 3]
        for key in array.keys():
            np.testing.assert_allclose(row[key], [1, 1])
            np.testing.assert_allclose(array[key], [1, 2, 1, 3])

        array = isopy.ones(4, 'ru 105pd 107ag cd'.split())
        row = array[[1, 3]]
        row[:] = [2, 3]
        for key in array.keys():
            np.testing.assert_allclose(row[key], [2, 3])
            np.testing.assert_allclose(array[key], [1, 1, 1, 1])

        array = isopy.ones(4, 'ru 105pd 107ag cd'.split())
        row = array[[1, 3]]
        array[[1, 3]] = [2, 3]
        for key in array.keys():
            np.testing.assert_allclose(row[key], [1, 1])
            np.testing.assert_allclose(array[key], [1, 2, 1, 3])

        # IsopyArray

        # scalar
        array = isopy.ones(4, 'ru 105pd 107ag cd'.split())
        other = isopy.array(dict(ru=3, ag107=5, cd=4))
        row = array[1]
        array[1] = other
        for key in array.keys():
            np.testing.assert_allclose(row[key], other.get(key))
            np.testing.assert_allclose(array[key], [1, other.get(key), 1, 1])

        array = isopy.ones(4, 'ru 105pd 107ag cd'.split())
        other = isopy.array(dict(ru=[3], ag107=[5], cd=[4]))
        row = array[1]
        array[1] = other
        for key in array.keys():
            np.testing.assert_allclose(row[key], other.get(key)[0])
            np.testing.assert_allclose(array[key], [1, other.get(key)[0], 1, 1])

        # sequence of scalars
        array = isopy.ones(4, 'ru 105pd 107ag cd'.split())
        other = isopy.array(dict(ru=3, ag107=5, cd=4))
        row = array[1:3]
        row[:] = other
        for key in array.keys():
            np.testing.assert_allclose(row[key], [other.get(key), other.get(key)])
            np.testing.assert_allclose(array[key], [1, other.get(key), other.get(key), 1])

        array = isopy.ones(4, 'ru 105pd 107ag cd'.split())
        other = isopy.array(dict(ru=3, ag107=5, cd=4))
        row = array[1:3]
        array[1:3] = other
        for key in array.keys():
            np.testing.assert_allclose(row[key], [other.get(key), other.get(key)])
            np.testing.assert_allclose(array[key], [1, other.get(key), other.get(key), 1])

        array = isopy.ones(4, 'ru 105pd 107ag cd'.split())
        other = isopy.array(dict(ru=[3], ag107=[5], cd=[4]))
        row = array[1:3]
        row[:] = other
        for key in array.keys():
            np.testing.assert_allclose(row[key], [other.get(key)[0], other.get(key)[0]])
            np.testing.assert_allclose(array[key], [1, other.get(key)[0], other.get(key)[0], 1])

        array = isopy.ones(4, 'ru 105pd 107ag cd'.split())
        other = isopy.array(dict(ru=[3, 4], ag107=[5, 7], cd=[4, 6]))
        row = array[1:3]
        array[1:3] = other
        for key in array.keys():
            np.testing.assert_allclose(row[key], [other.get(key)[0], other.get(key)[1]])
            np.testing.assert_allclose(array[key], [1, other.get(key)[0], other.get(key)[1], 1])

        # This does not return a view
        array = isopy.ones(4, 'ru 105pd 107ag cd'.split())
        other = isopy.array(dict(ru=3, ag107=5, cd=4))
        row = array[1, 3]
        row[:] = other
        for key in array.keys():
            np.testing.assert_allclose(row[key], [other.get(key), other.get(key)])
            np.testing.assert_allclose(array[key], [1, 1, 1, 1])

        array = isopy.ones(4, 'ru 105pd 107ag cd'.split())
        other = isopy.array(dict(ru=3, ag107=5, cd=4))
        row = array[1, 3]
        array[1, 3] = other
        for key in array.keys():
            np.testing.assert_allclose(row[key], [1, 1])
            np.testing.assert_allclose(array[key], [1, other.get(key), 1, other.get(key)])

        array = isopy.ones(4, 'ru 105pd 107ag cd'.split())
        other = isopy.array(dict(ru=[3], ag107=[5], cd=[4]))
        row = array[(1, 3)]
        row[:] = other
        for key in array.keys():
            np.testing.assert_allclose(row[key], [other.get(key)[0], other.get(key)[0]])
            np.testing.assert_allclose(array[key], [1, 1, 1, 1])

        array = isopy.ones(4, 'ru 105pd 107ag cd'.split())
        other = isopy.array(dict(ru=[3], ag107=[5], cd=[4]))
        row = array[(1, 3)]
        array[(1, 3)] = other
        for key in array.keys():
            np.testing.assert_allclose(row[key], [1, 1])
            np.testing.assert_allclose(array[key], [1, other.get(key)[0], 1, other.get(key)[0]])

        array = isopy.ones(4, 'ru 105pd 107ag cd'.split())
        other = isopy.array(dict(ru=[3, 4], ag107=[5, 7], cd=[4, 6]))
        row = array[[1, 3]]
        row[:] = other
        for key in array.keys():
            np.testing.assert_allclose(row[key], [other.get(key)[0], other.get(key)[1]])
            np.testing.assert_allclose(array[key], [1, 1, 1, 1])

        array = isopy.ones(4, 'ru 105pd 107ag cd'.split())
        other = isopy.array(dict(ru=[3, 4], ag107=[5, 7], cd=[4, 6]))
        row = array[[1, 3]]
        array[[1, 3]] = other
        for key in array.keys():
            np.testing.assert_allclose(row[key], [1, 1])
            np.testing.assert_allclose(array[key], [1, other.get(key)[0], 1, other.get(key)[1]])

    def test_copy(self):
        data = np.array([[i for i in range(1, 7)], [i ** 2 for i in range(1, 7)]])

        array = isopy.array(data, '102pd 104pd 105pd 106pd 108pd 110pd'.split())
        subarray = array[('105pd', '108pd')]

        array['105pd'] = [10, 100]
        assert array[('105pd', '108pd')] == subarray

        subarray[0] = 5
        assert array[('105pd', '108pd')] == subarray

        array = isopy.array(data, '102pd 104pd 105pd 106pd 108pd 110pd'.split())
        subarray = array.copy(key_eq=('105pd', '108pd'))

        assert array is not array.copy()

        array['105pd'] = [10, 100]
        assert array[('105pd', '108pd')] != subarray

        subarray[0] = 5
        assert array[('105pd', '108pd')] != subarray

    def test_get_set2(self):
        #Exceptions for get and set methods

        data = np.array([[i for i in range(1, 7)], [i ** 2 for i in range(1, 7)]])

        array = isopy.array(data, '102pd 104pd 105pd 106pd 108pd 110pd'.split())
        assert array[[]].shape == (0,)
        assert array[[]].size == 0

        with pytest.raises(ValueError):
            array['hermione']

        with pytest.raises(ValueError):
            array[('hermione',)]

        with pytest.raises(ValueError):
            array['hermione'] = 1

        with pytest.raises(ValueError):
            array[('hermione',)] = 1

        with pytest.raises(ValueError):
            array['107ag'] = 1

        with pytest.raises(ValueError):
            array[('107ag')] = 1

        assert isinstance(array[0], core.IsopyVoid)
        assert isinstance(array[:1], core.IsopyNdarray)
        assert isinstance(array[(0,)], core.IsopyNdarray)

        array = array[0]
        assert array[[]].shape == (0,)
        assert array[[]].size == 0

        with pytest.raises(ValueError):
            array['hermione']

        with pytest.raises(ValueError):
            array[('hermione',)]

        with pytest.raises(IndexError):
            array[1]

        with pytest.raises(IndexError):
            array[(1,2)]

        with pytest.raises(ValueError):
            array['hermione'] = 1

        with pytest.raises(ValueError):
            array[('hermione',)] = 1

        with pytest.raises(ValueError):
            array['107ag'] = 1

        with pytest.raises(ValueError):
            array[('107ag',)] = 1

        with pytest.raises(IndexError):
            array[1] = 1

        with pytest.raises(IndexError):
            array[(1,2)] = 1

    # to file is tested in io tests
    def test_to_obj(self):
        # size 100, 1-dim
        array = isopy.random(20, (1, 0.1), 'ru pd cd'.split(), seed = 46)

        assert core.hashstr(repr(array)) == 'd89a1b8a6c7991d2a72c16734d347b0a'
        assert core.hashstr(str(array)) == '8019204da3be1770ee071e75afd1dd0f'
        assert core.hashstr(array.to_text()) == '8019204da3be1770ee071e75afd1dd0f'
        assert core.hashstr(array.to_text(delimiter='; ,')) == '963d15b6daaa48a20df1e7f7b9cddb2f'
        assert core.hashstr(array.to_text(include_row=True, include_dtype=True)) == '58c8b6aca285883da69dce6b8c417cc5'
        assert core.hashstr(array.to_text(f='{:.2f}')) == '0eace892417529a0669b972f5d76a57f'
        assert core.hashstr(array.to_text(nrows=5)) == '4d3e5af8172c01feed4a85a27e35152e'

        assert core.hashstr(array.to_table()) == 'e6b2c553bfb85c6975b8f4f87b012dca'
        assert core.hashstr(array.to_table(include_row=True)) == 'ea55dccb58e4b487bba26f104f03c7c9'
        assert core.hashstr(array.to_table(include_row=True, include_dtype=True)) == '26a4b6e262f5a2d8192c76fb1d602f23'
        assert core.hashstr(array.to_table(f='{:.2f}')) == '9359aa34f5c0291e29bb0e74ed116217'
        assert core.hashstr(array.to_table(nrows=5)) == 'ba4df1706736977a4f26221614708114'

        tovalue = array.to_list()
        assert type(tovalue) is list
        assert False not in [type(v) is list for v in tovalue]
        assert_array_equal_array(array, isopy.array(tovalue, array.keys))

        tovalue = array.to_dict()
        assert type(tovalue) is dict
        assert False not in [type(v) is list for v in tovalue.values()]
        assert False not in [type(k) is str for k in tovalue.keys()]
        assert_array_equal_array(array, isopy.array(tovalue))

        tovalue = array.to_ndarray()
        assert type(tovalue) is np.ndarray
        assert array.keys == tovalue.dtype.names
        for key in array.keys:
            np.testing.assert_allclose(array[key], tovalue[str(key)])

        tovalue = array[0].to_ndarray()
        assert type(tovalue) is np.void
        assert array[0].keys == tovalue.dtype.names
        for key in array.keys:
            np.testing.assert_allclose(array[key][0], tovalue[str(key)])

        try:
            array.to_clipboard()
        except pyperclip.PyperclipException:
            pass # Wont run on travis
        else:
            assert core.hashstr(pyperclip.paste()) == '8019204da3be1770ee071e75afd1dd0f'
            new_array1 = isopy.array_from_clipboard()
            assert isinstance(new_array1, core.IsopyArray)
            assert new_array1.flavour == array.flavour
            assert_array_equal_array(new_array1, array)

        # Size 1, 1-dim
        array = isopy.random(1, (1, 0.1), 'ru pd cd'.split(), seed=46)

        assert core.hashstr(repr(array)) == '309361dd9169fe6d0ffd8ec8be91eddd'
        assert core.hashstr(str(array)) == 'd3de30a8e60e6b9511b065c6bf795fa8'
        assert core.hashstr(array.to_text()) == 'd3de30a8e60e6b9511b065c6bf795fa8'
        assert core.hashstr(array.to_text(delimiter='; ,')) == '4cce754c47732294bd9e07e7acef7c1f'
        assert core.hashstr(array.to_text(include_row=True, include_dtype=True)) == '46832d224c9c15c09168ed7e3ee77faa'
        assert core.hashstr(array.to_text(f='{:.2f}')) == 'da66d5829d65ef08f36b257435f3e9e1'
        assert core.hashstr(array.to_text(nrows=5)) == 'd3de30a8e60e6b9511b065c6bf795fa8'

        tovalue = array.to_list()
        assert type(tovalue) is list
        assert False not in [type(v) is list for v in tovalue]
        assert_array_equal_array(array, isopy.array(tovalue, array.keys))

        tovalue = array.to_dict()
        assert type(tovalue) is dict
        assert False not in [type(v) is list for v in tovalue.values()]
        assert False not in [type(k) is str for k in tovalue.keys()]
        assert_array_equal_array(array, isopy.array(tovalue))

        tovalue = array.to_ndarray()
        assert type(tovalue) is np.ndarray
        assert array.keys == tovalue.dtype.names
        for key in array.keys:
            np.testing.assert_allclose(array[key], tovalue[str(key)])

        try:
            array.to_clipboard()
        except pyperclip.PyperclipException:
            pass # Wont run on travis
        else:
            assert core.hashstr(pyperclip.paste()) == 'd3de30a8e60e6b9511b065c6bf795fa8'

        # Size 1, 0-dim
        array = isopy.random(None, (1, 0.1), 'ru pd cd'.split(), seed=46)

        assert core.hashstr(repr(array)) == '395ea36d5e4c0bce9679de1146eae27c'
        assert core.hashstr(str(array)) == 'd3de30a8e60e6b9511b065c6bf795fa8'
        assert core.hashstr(array.to_text()) == 'd3de30a8e60e6b9511b065c6bf795fa8'
        assert core.hashstr(array.to_text(delimiter='; ,')) == '4cce754c47732294bd9e07e7acef7c1f'
        assert core.hashstr(array.to_text(include_row=True, include_dtype=True)) == '44829ca7e7c20790950bc22b3cebb272'
        assert core.hashstr(array.to_text(f='{:.2f}')) == 'da66d5829d65ef08f36b257435f3e9e1'
        assert core.hashstr(array.to_text(nrows=5)) == 'd3de30a8e60e6b9511b065c6bf795fa8'

        tovalue = array.to_list()
        assert type(tovalue) is list
        assert_array_equal_array(array, isopy.array(tovalue, array.keys))

        tovalue = array.to_dict()
        assert type(tovalue) is dict
        assert False not in [type(k) is str for k in tovalue.keys()]
        assert_array_equal_array(array, isopy.array(tovalue))

        tovalue = array.to_ndarray()
        assert type(tovalue) is np.ndarray
        assert array.keys == tovalue.dtype.names
        for key in array.keys:
            np.testing.assert_allclose(array[key], tovalue[str(key)])

        try:
            array.to_clipboard()
        except pyperclip.PyperclipException:
            pass # Wont run on travis
        else:
            assert core.hashstr(pyperclip.paste()) == 'd3de30a8e60e6b9511b065c6bf795fa8'

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
        array1 = 1
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
        array1 = [1]
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


class Test_ArrayFunctions:
    def test_singleinput(self):
        keys = 'ru pd cd'.split()
        array1 = isopy.random(None, [(0, 1), (1, 0.1), (0.5, 0.5)], keys, seed = 46)
        array2 = isopy.random(100, [(0, 1), (1, 0.1), (0.5, 0.5)], keys, seed = 46)
        array3 = array2.copy()
        array3['pd'][5] = np.nan

        # Elementwise functions
        for func in core.afnp_elementwise:
            self.singleinput(func, array1)
            self.singleinput(func, array2)
            self.singleinput(func, array3)

        # This means a warning should be raised for ufunc
        core.ALLOWED_NUMPY_FUNCTIONS.pop(np.abs)

        self.singleinput(np.abs, array1)
        self.singleinput(np.abs, array2)
        self.singleinput(np.abs, array3)

        core.ALLOWED_NUMPY_FUNCTIONS[np.abs] = lambda *args, **kwargs: 'hello'
        assert np.abs(array1) == 'hello'
        core.ALLOWED_NUMPY_FUNCTIONS[np.abs] = True

        # Cumulative functions
        for func in core.afnp_cumulative:
            self.singleinput(func, array1)
            self.singleinput(func, array2)
            self.singleinput(func, array3)

            self.singleinput(func, array1, axis = None)
            self.singleinput(func, array2, axis = None)
            self.singleinput(func, array3, axis = None)

            self.singleinput(func, array1, axis=0)
            self.singleinput(func, array2, axis=0)
            self.singleinput(func, array3, axis=0)

            self.singleinput(func, array2, axis=1)
            self.singleinput(func, array3, axis=1)

        # reducing numpy functions
        for func in core.afnp_reducing:
            self.singleinput(func, array1)
            self.singleinput(func, array2)
            self.singleinput(func, array3)

            self.singleinput(func, array1, axis=None)
            self.singleinput(func, array2, axis=None)
            self.singleinput(func, array3, axis=None)

            self.singleinput(func, array1, axis=0)
            self.singleinput(func, array2, axis=0)
            self.singleinput(func, array3, axis=0)

            self.singleinput(func, array2, axis=1)
            self.singleinput(func, array3, axis=1)

        #This means a warning should be raised for array_function
        core.ALLOWED_NUMPY_FUNCTIONS.pop(np.mean)

        self.singleinput(np.mean, array1)
        self.singleinput(np.mean, array2)
        self.singleinput(np.mean, array3)

        self.singleinput(np.mean, array1, axis=None)
        self.singleinput(np.mean, array2, axis=None)
        self.singleinput(np.mean, array3, axis=None)

        self.singleinput(np.mean, array1, axis=0)
        self.singleinput(np.mean, array2, axis=0)
        self.singleinput(np.mean, array3, axis=0)

        self.singleinput(np.mean, array2, axis=1)
        self.singleinput(np.mean, array3, axis=1)

        core.ALLOWED_NUMPY_FUNCTIONS[np.mean] = lambda *args, **kwargs: 'hello'
        assert np.mean(array1) == 'hello'
        core.ALLOWED_NUMPY_FUNCTIONS[np.mean] = True

        # reducing isopy functions
        for func in [isopy.mad, isopy.nanmad, isopy.se, isopy.nanse, isopy.sd, isopy.nansd,
                     isopy.sd2, isopy.sd95, isopy.se2, isopy.se95, isopy.mad2, isopy.mad95,
                     isopy.nansd2, isopy.nansd95, isopy.nanse2, isopy.nanse95, isopy.nanmad2, isopy.nanmad95]:
            self.singleinput(func, array1)
            self.singleinput(func, array2)
            self.singleinput(func, array3)

            self.singleinput(func, array1, axis=None)
            self.singleinput(func, array2, axis=None)
            self.singleinput(func, array3, axis=None)

            self.singleinput(func, array1, axis=0)
            self.singleinput(func, array2, axis=0)
            self.singleinput(func, array3, axis=0)

            self.singleinput(func, array2, axis=1)
            self.singleinput(func, array3, axis=1)



        # Test the reduce method
        self.singleinput(np.add.reduce, array1)
        self.singleinput(np.add.reduce, array2)
        self.singleinput(np.add.reduce, array3)

        self.singleinput(np.add.reduce, array1, axis=None)
        self.singleinput(np.add.reduce, array2, axis=None)
        self.singleinput(np.add.reduce, array3, axis=None)

        self.singleinput(np.add.reduce, array1, axis=0)
        self.singleinput(np.add.reduce, array2, axis=0)
        self.singleinput(np.add.reduce, array3, axis=0)

        self.singleinput(np.add.reduce, array2, axis=1)
        self.singleinput(np.add.reduce, array3, axis=1)

        # Test the accumulate method
        #self.singleinput(np.add.accumulate, array1) # Dosnt work for 0-dim
        self.singleinput(np.add.accumulate, array2)
        self.singleinput(np.add.accumulate, array3)

        self.singleinput(np.add.accumulate, array1, axis=None)
        #self.singleinput(np.add.accumulate, array2, axis=None) # Doesnt work for > 1-dim
        #self.singleinput(np.add.accumulate, array3, axis=None) # Doesnt work for > 1-dim

        # self.singleinput(np.add.accumulate, array1, axis=0) # Dosnt work for 0-dim
        self.singleinput(np.add.accumulate, array2, axis=0)
        self.singleinput(np.add.accumulate, array3, axis=0)

        self.singleinput(np.add.accumulate, array2, axis=1)
        self.singleinput(np.add.accumulate, array3, axis=1)

    def singleinput(self, func, array, axis = core.NotGiven, out = core.NotGiven, where = core.NotGiven):
        kwargs = {}
        if axis is not core.NotGiven: kwargs['axis'] = axis
        if out is not core.NotGiven: kwargs['out'] = out
        if where is not core.NotGiven: kwargs['where'] = where

        if axis is core.NotGiven or axis == 0:
            result = func(array, **kwargs)
            result2 = isopy.arrayfunc(func, array, **kwargs)

            assert type(result) is type(array)
            assert result.keys == array.keys
            for key in result.keys:
                rvalue = result[key]
                tvalue = func(array[key])
                assert type(rvalue) is not float
                if not isinstance(tvalue, (np.generic, np.ndarray)):
                    # isopy mad returns float for tvalue but i cant replicate it elsewhere
                    warnings.warn(f'function {func.__name__} returned {type(tvalue)} with a value of {tvalue} for an array with shape {array.shape}')
                else:
                    assert rvalue.size == tvalue.size
                    assert rvalue.ndim == tvalue.ndim
                np.testing.assert_allclose(rvalue, tvalue)

            assert_array_equal_array(result, result2)

            # For builtin functions
            funcname = {'amin': 'min', 'amax': 'max'}.get(func.__name__, func.__name__)
            if hasattr(array, func.__name__):
                assert_array_equal_array(result, getattr(array, funcname)(**kwargs))

        else:
            result = func(array, **kwargs)
            true = func(array.to_list(), **kwargs)
            if not isinstance(true, (np.generic, np.ndarray)):
                # isopy mad returns float for tvalue but i cant replicate it elsewhere
                warnings.warn(
                    f'function {func.__name__} returned {type(true)} with a value of {true} for an array with shape {array.shape}')
            else:
                assert result.size == true.size
                assert result.ndim == true.ndim
            np.testing.assert_allclose(result, true)

            result2 = isopy.arrayfunc(func, array, **kwargs)
            np.testing.assert_allclose(result2, true)

            # For builtin functions
            funcname = {'amin': 'min', 'amax': 'max'}.get(func.__name__, func.__name__)
            if hasattr(array, funcname):
                np.testing.assert_allclose(true, getattr(array, funcname)(**kwargs))

        # There isnt enough arguments to an error should be thrown
        with pytest.raises((ValueError, TypeError)):
            func()

        # In case its not caught by the array dispatcher.
        with pytest.raises((ValueError, TypeError)):
            isopy.arrayfunc(func)

    # TODO mixed flavours
    def test_dualinput(self):
        keys1 = 'ru pd cd'.split()
        keys2 = 'pd ag cd'.split()
        keys12 = 'ru pd cd ag'.split()
        keys21 = 'pd ag cd ru'.split()

        array1 = isopy.random(None, [(0, 1), (1, 0.1), (0.5, 0.5)], keys1, seed=1)
        array2 = isopy.random(None, [(0, 1), (1, 0.1), (0.5, 0.5)], keys2, seed=2)
        array3 = isopy.random(1, [(0, 1), (1, 0.1), (0.5, 0.5)], keys1, seed=3)
        array4 = isopy.random(1, [(0, 1), (1, 0.1), (0.5, 0.5)], keys2, seed=4)
        array5 = isopy.random(10, [(0, 1), (1, 0.1), (0.5, 0.5)], keys1, seed=5)
        array6 = isopy.random(10, [(0, 1), (1, 0.1), (0.5, 0.5)], keys2, seed=6)
        array7 = isopy.random(3, [(0, 1), (1, 0.1), (0.5, 0.5)], keys1, seed=7)
        value1 = 1.5
        value2 = [3.2]
        value3 = [i for i in range(10)]
        value4 = [i for i in range(3)]
        dict1 = dict(rh=array2['ag'], pd=array2['pd'], ag=array2['pd'], cd=array2['cd'], te=array2['cd'])
        dict2 = dict(rh=array4['ag'], pd=array4['pd'], ag=array4['pd'], cd=array4['cd'], te=array4['cd'])
        dict3 = dict(rh=array6['ag'], pd=array6['pd'], ag=array6['pd'], cd=array6['cd'], te=array6['cd'])
        dict4 = dict(rh=array7['ru'], pd=array7['pd'], ag=array7['pd'], cd=array7['cd'], te=array7['cd'])

        isopy_dual_functions = [isopy.add, isopy.subtract, isopy.multiply, isopy.divide, isopy.power]
        for func in (core.afnp_dual + isopy_dual_functions):
            # all different array combinations
            self.dualinput_aa(func, array1, array2, keys12, 0)
            self.dualinput_aa(func, array1, array4, keys12, 1)
            self.dualinput_aa(func, array1, array6, keys12, 1)

            self.dualinput_aa(func, array3, array2, keys12, 1)
            self.dualinput_aa(func, array3, array4, keys12, 1)
            self.dualinput_aa(func, array3, array6, keys12, 1)

            self.dualinput_aa(func, array5, array2, keys12, 1)
            self.dualinput_aa(func, array5, array4, keys12, 1)
            self.dualinput_aa(func, array5, array6, keys12, 1)

            self.dualinput_aa(func, array2, array1, keys21, 0)
            self.dualinput_aa(func, array2, array3, keys21, 1)
            self.dualinput_aa(func, array2, array5, keys21, 1)

            self.dualinput_aa(func, array4, array1, keys21, 1)
            self.dualinput_aa(func, array4, array3, keys21, 1)
            self.dualinput_aa(func, array4, array5, keys21, 1)

            self.dualinput_aa(func, array6, array1, keys21, 1)
            self.dualinput_aa(func, array6, array3, keys21, 1)
            self.dualinput_aa(func, array6, array5, keys21, 1)

            with pytest.raises(ValueError):
                self.dualinput_aa(func, array5, array7, keys12, 1)
            with pytest.raises(ValueError):
                self.dualinput_aa(func, array7, array6, keys21, 1)

            # array dict combinations
            self.dualinput_aa(func, array1, dict1, keys1, 0)
            self.dualinput_aa(func, array1, dict2, keys1, 0)
            self.dualinput_aa(func, array1, dict3, keys1, 1)

            self.dualinput_aa(func, array3, dict1, keys1, 1)
            self.dualinput_aa(func, array3, dict2, keys1, 1)
            self.dualinput_aa(func, array3, dict3, keys1, 1)

            self.dualinput_aa(func, array5, dict1, keys1, 1)
            self.dualinput_aa(func, array5, dict2, keys1, 1)
            self.dualinput_aa(func, array5, dict3, keys1, 1)

            with pytest.raises(ValueError):
                self.dualinput_aa(func, array5, dict4, keys1, 1)
            with pytest.raises(ValueError):
                self.dualinput_aa(func, array7, dict3, keys1, 1)

            # dict array combinations
            self.dualinput_aa(func, dict1, array1, keys1, 0)
            self.dualinput_aa(func, dict2, array1, keys1, 0)
            self.dualinput_aa(func, dict3, array1, keys1, 1)

            self.dualinput_aa(func, dict1, array3, keys1, 1)
            self.dualinput_aa(func, dict2, array3, keys1, 1)
            self.dualinput_aa(func, dict3, array3, keys1, 1)

            self.dualinput_aa(func, dict1, array5, keys1, 1)
            self.dualinput_aa(func, dict2, array5, keys1, 1)
            self.dualinput_aa(func, dict3, array5, keys1, 1)

            with pytest.raises(ValueError):
                self.dualinput_aa(func, dict3, array7, keys1, 1)
            with pytest.raises(ValueError):
                self.dualinput_aa(func, dict4, array5, keys1, 1)

            # all array value combinations
            self.dualinput_av(func, array1, value1, keys1)
            self.dualinput_av(func, array1, value2, keys1)
            self.dualinput_av(func, array1, value3, keys1)

            self.dualinput_av(func, array3, value1, keys1)
            self.dualinput_av(func, array3, value2, keys1)
            self.dualinput_av(func, array3, value3, keys1)

            self.dualinput_av(func, array5, value1, keys1)
            self.dualinput_av(func, array5, value2, keys1)
            self.dualinput_av(func, array5, value3, keys1)

            with pytest.raises(ValueError):
                self.dualinput_av(func, array5, value4, keys1)
            with pytest.raises(ValueError):
                self.dualinput_av(func, array7, value3, keys1)

            # all value array combinations
            self.dualinput_va(func, value1, array1, keys1)
            self.dualinput_va(func, value1, array3, keys1)
            self.dualinput_va(func, value1, array5, keys1)

            self.dualinput_va(func, value2, array1, keys1)
            self.dualinput_va(func, value2, array3, keys1)
            self.dualinput_va(func, value2, array5, keys1)

            self.dualinput_va(func, value3, array1, keys1)
            self.dualinput_va(func, value3, array3, keys1)
            self.dualinput_va(func, value3, array5, keys1)

            with pytest.raises(ValueError):
                self.dualinput_va(func, value3, array7, keys1)
            with pytest.raises(ValueError):
                self.dualinput_va(func, value4, array5, keys1)

            # There isnt enough arguments to an error should be thrown
            with pytest.raises((ValueError, TypeError)):
                func()

            # In case its not caught by the array dispatcher.
            with pytest.raises((ValueError, TypeError)):
                isopy.arrayfunc(func)

        np.testing.assert_allclose(isopy.add(value1, value2), np.add(value1, value2))
        np.testing.assert_allclose(isopy.subtract(value1, value3), np.subtract(value1, value3))
        np.testing.assert_allclose(isopy.divide(value3, value2), np.divide(value3, value2))
        np.testing.assert_allclose(isopy.multiply(value1, value2), np.multiply(value1, value2))
        np.testing.assert_allclose(isopy.power(value1, value2), np.power(value1, value2))

    def dualinput_aa(self, func, a1, a2, keys, ndim):
        if type(a1) is dict: b1 = isopy.ScalarDict(a1)
        else: b1 = a1
        if type(a2) is dict: b2 = isopy.ScalarDict(a2)
        else: b2 = a2

        result = func(a1, a2)
        assert isinstance(result, core.IsopyArray)
        assert result.keys == keys
        assert result.ndim == ndim
        for key in keys:
            true = func(b1.get(key, np.nan), b2.get(key, np.nan))
            assert result.size == true.size
            np.testing.assert_allclose(result[key], true)

        result2 = isopy.arrayfunc(func, a1, a2)
        assert result2.ndim == ndim
        assert_array_equal_array(result, result2)

        result2 = isopy.arrayfunc(func, a1, a2, default_value=1)
        for key in keys:
            true = func(b1.get(key, 1), b2.get(key, 1))
            assert result2.size == true.size
            np.testing.assert_allclose(result2[key], true)
        try:
            assert_array_equal_array(result2, result)
        except AssertionError:
            pass
        else:
            raise AssertionError()

        result2 = isopy.arrayfunc(func, a1, a2, default_value=(1, 2))
        assert result2.ndim == ndim
        for key in keys:
            true = func(b1.get(key, 1), b2.get(key, 2))
            assert result2.size == true.size
            np.testing.assert_allclose(result2[key], true)
        try:
            assert_array_equal_array(result, result2)
        except AssertionError:
            pass
        else:
            raise AssertionError()

        with pytest.raises(ValueError):
            isopy.arrayfunc(func, a1, a2, default_value=(1,))

        with pytest.raises(ValueError):
            isopy.arrayfunc(func, a1, a2, default_value=(1,2,3))

    def dualinput_av(self, func, a1, a2, keys):
        result = func(a1, a2)
        assert isinstance(result, core.IsopyArray)
        assert result.keys == keys
        for key in keys:
            true = func(a1.get(key), a2)
            assert result.size == true.size
            assert result.ndim == true.ndim
            np.testing.assert_allclose(result[key], true)

        result2 = isopy.arrayfunc(func, a1, a2)
        assert_array_equal_array(result, result2)

    def dualinput_va(self, func, a1, a2, keys):
        result = func(a1, a2)
        assert isinstance(result, core.IsopyArray)
        assert result.keys == keys
        for key in keys:
            true = func(a1, a2.get(key))
            assert result.size == true.size
            assert result.ndim == true.ndim
            np.testing.assert_allclose(result[key], true)

        result2 = isopy.arrayfunc(func, a1, a2)
        assert_array_equal_array(result, result2)

    def test_count_finite(self):
        array = isopy.array([[1,2,3], [4, np.nan, 6]], 'ru pd cd'.split())
        answer = isopy.array((2,1,2), 'ru pd cd'.split())

        assert_array_equal_array(isopy.nancount(array), answer)
        np.testing.assert_allclose(isopy.nancount(array, axis=1), [3, 2])
        assert isopy.nancount(array, axis=None) == 5

        data = [[1,2,3], [4, np.nan, 6]]
        answer = (2,1,2)

        assert isopy.nancount(data) == 5
        np.testing.assert_allclose(isopy.nancount(data, axis=0), answer)
        np.testing.assert_allclose(isopy.nancount(data, axis=1), [3, 2])

    def test_keyminmax(self):
        array = isopy.random(10, [3, 1, 5], 'ru pd cd'.split(), seed=46)
        array['ru'][1] = 100
        array['ru'][2] = -2

        assert isopy.keymax(array) == 'cd'
        assert isopy.keymax(array, np.mean) == 'ru'

        assert isopy.keymin(array) == 'pd'
        assert isopy.keymin(array, np.min) == 'ru'

    def test_where(self):
        data = [[1, 2, 3], [3,np.nan, 5], [6, 7, 8], [9, 10, np.nan]]
        array = isopy.array(data, 'ru pd cd'.split())

        where1 = [True, False, True, True]
        where1t = [[True, True, True], [False, False, False], [True, True, True], [True, True, True]]

        where2 = [[False, True, True], [False, False, True], [True, True, True], [True, True, False]]
        arrayw2 = isopy.array(where2, 'ru pd cd'.split(), dtype=bool)

        for func in [np.mean, np.std]:
            try:
                result = func(array, where=where1, axis=0)
                true = func(data, where=where1t, axis=0)
                np.testing.assert_allclose(result.tolist(), true)

                result = func(array, where=arrayw2, axis=0)
                true = func(data, where=where2, axis=0)
                np.testing.assert_allclose(result.tolist(), true)
            except Exception as err:
                raise AssertionError(f'func {func} test failed') from err

    def test_out(self):
        data = [[1, 2, 3], [3,np.nan, 5], [6, 7, 8], [9, 10, np.nan]]
        array = isopy.array(data, 'ru pd cd'.split())

        for func in [np.std, np.nanstd]:
            out = isopy.ones(None, 'ru pd cd'.split())
            result1 = func(array)
            result2 = func(array, out=out)
            assert_array_equal_array(result1, result2)
            assert result2 is out

            out = np.ones(4)
            result1 = func(array, axis=1)
            result2 = func(array, axis=1, out=out)
            np.testing.assert_allclose(result1, result2)
            assert result2 is out

            out = isopy.ones(None, 'ag ru cd'.split())
            result1 = func(array)
            assert result1.keys == 'ru pd cd'.split()
            result2 = func(array, out=out)
            assert result2 is out
            assert result2.keys == 'ag ru cd'.split()

            np.testing.assert_allclose(result1['ru'], result2['ru'])
            np.testing.assert_allclose(result1['cd'], result2['cd'])
            np.testing.assert_allclose(np.ones(4), result2['ag'])

    def test_special(self):
        # np.average, np.copyto
        tested = []

        #Tests for copyto
        tested.append(np.copyto)

        array1 = isopy.array([[1, 2, 3], [11, 12, 13]], 'ru pd cd'.split())
        array2 = isopy.empty(2, 'ru pd cd'.split())
        np.copyto(array2, array1)
        assert array2 is not array1
        assert_array_equal_array(array2, array1)

        dictionary = dict(ru = 1, rh = 1.5, pd = 2, ag = 2.5, cd = 3)
        array1 = isopy.array([1, 2, 3], 'ru pd cd'.split())
        array2 = isopy.empty(-1, 'ru pd cd'.split())
        np.copyto(array2, dictionary)
        assert_array_equal_array(array2, array1)

        #Tests for average
        tested.append(np.average)

        #axis = 0
        array = isopy.array([[1, 2, 3], [11, 12, 13]], 'ru pd cd'.split())
        weights = [2, 1]
        correct = isopy.array((13/3, 16/3, 19/3), 'ru pd cd'.split())
        result = np.average(array, weights=weights)
        assert_array_equal_array(result, correct)

        result = np.average(array, 0, weights=weights)
        assert_array_equal_array(result, correct)

        weights = isopy.array((2, 0.5, 1), 'ru pd cd'.split())
        with pytest.raises(TypeError):
            np.average(array, weights=weights)

        #axis = 1
        weights = [2, 0.5, 1]
        correct = np.array([(1 * 2 + 2 * 0.5 + 3 * 1) / 3.5, (11 * 2 + 12 * 0.5 + 13 * 1) / 3.5])
        result = np.average(array, axis=1, weights=weights)
        np.testing.assert_allclose(result, correct)

        result = np.average(array, 1, weights)
        np.testing.assert_allclose(result, correct)

        #axis = None
        # shape mismatch
        weights = [2, 1]
        with pytest.raises(TypeError):
            np.average(array, axis=None, weights=weights)

        weights = isopy.array((2, 0.5, 1), 'ru pd cd'.split())
        with pytest.raises(TypeError):
            np.average(array, axis=None, weights=weights)

        # axis = 0
        array = isopy.array([[1, 2, 3], [11, 12, 13]], 'ru pd cd'.split())
        weights = [[2, 0.5, 1], [0.5, 0.5, 0.5]]
        with pytest.raises(TypeError):
            #Shape mismatch
            np.average(array, weights=weights)

        weights = isopy.array([[2, 0.5, 1],[0.5, 0.5, 0.5]], 'ru pd cd'.split())
        correct = isopy.array(((1*2+11*0.5)/2.5,(2*0.5+12*0.5)/1, (3*1+13*0.5)/1.5), 'ru pd cd'.split())
        result = np.average(array, weights=weights)
        assert_array_equal_array(result, correct)

        result = np.average(array, 0, weights)
        assert_array_equal_array(result, correct)

        weights2 = isopy.IsopyDict(weights)
        result = np.average(array, weights=weights2)
        assert_array_equal_array(result, correct)

        weights2 = isopy.ScalarDict(weights)
        result = np.average(array, weights=weights2)
        assert_array_equal_array(result, correct)

        weights2 = weights.to_dict()
        with pytest.raises(Exception):
            #Not sure what will throw an error but soemthign will
            np.average(array, weights=weights2)

        # axis = 1
        weights = isopy.array([[2, 0.5, 1],[0.5, 0.5, 0.5]], 'ru pd cd'.split())
        correct = np.array([(1 * 2 + 2 * 0.5 + 3 * 1) / 3.5, (11 * 0.5 + 12 * 0.5 + 13 * 0.5) / 1.5])
        result = np.average(array, axis=1, weights=weights)
        np.testing.assert_allclose(result, correct)

        result = np.average(array, 1, weights)
        np.testing.assert_allclose(result, correct)

        weights2 = isopy.IsopyDict(weights)
        result = np.average(array, axis=1, weights=weights2)
        np.testing.assert_allclose(result, correct)

        weights2 = isopy.ScalarDict(weights)
        result = np.average(array, axis=1, weights=weights2)
        np.testing.assert_allclose(result, correct)

        weights2 = weights.to_dict()
        with pytest.raises(Exception):
            # Not sure what will throw an error but soemthign will
            np.average(array, axis=1, weights=weights2)

        weights = [[2, 0.5, 1],[0.5, 1, 2]]
        correct = np.array([(1*2+2*0.5+3*1)/3.5, (11*0.5+12*1+13*2)/3.5])
        result = np.average(array, axis=1, weights=weights)
        np.testing.assert_allclose(result, correct)

        result = np.average(array, 1, weights)
        np.testing.assert_allclose(result, correct)

        # axis = None
        weights = isopy.array([[2, 0.5, 1],[0.5, 0.5, 0.5]], 'ru pd cd'.split())
        correct = (1 * 2 + 2 * 0.5 + 3 * 1 + 11 * 0.5 + 12 * 0.5 + 13 * 0.5) / 5
        result = np.average(array, axis=None, weights=weights)
        np.testing.assert_allclose(result, correct)

        result = np.average(array, None, weights)
        np.testing.assert_allclose(result, correct)

        weights2 = isopy.IsopyDict(weights)
        result = np.average(array, axis=None, weights=weights2)
        np.testing.assert_allclose(result, correct)

        weights2 = isopy.ScalarDict(weights)
        result = np.average(array, axis=None, weights=weights2)
        np.testing.assert_allclose(result, correct)

        weights2 = weights.to_dict()
        with pytest.raises(Exception):
            # Not sure what will throw an error but soemthign will
            np.average(array, axis=None, weights=weights2)

        weights = [[2, 0.5, 1], [0.5, 0.5, 0.5]]
        correct = (1 * 2 + 2 * 0.5 + 3 * 1 + 11 * 0.5 + 12 * 0.5 + 13 * 0.5) / 5
        result = np.average(array, axis=None, weights=weights)
        np.testing.assert_allclose(result, correct)

        result = np.average(array, None, weights)
        np.testing.assert_allclose(result, correct)

        # Make sure functions are tested.
        for func in core.afnp_special:
            if func not in tested:
                raise ValueError(f'special function {func.__name__} not tested')

    def test_concatenate(self):
        # Axis = 0

        array1 = isopy.ones(2, 'ru pd cd'.split())
        array2 = isopy.ones(-1, 'pd ag107 cd'.split()) * 2
        array3 = isopy.ones(2, '101ru ag107'.split()) * 3

        array4 = isopy.ones(2) * 4
        array5 = isopy.ones(2) * 5

        np.testing.assert_allclose(isopy.concatenate(array4, array5), np.concatenate([array4, array5]))

        with pytest.raises(ValueError):
            isopy.concatenate(array1, array4)

        with pytest.raises(ValueError):
            isopy.concatenate(array1, array5)

        result = isopy.concatenate(array1)
        assert result is not array1
        assert_array_equal_array(result, array1)

        result = isopy.concatenate(array1, array2, array3)

        keys = array1.keys | array2.keys | array3.keys
        assert result.keys == keys
        for key in keys:
            true = np.concatenate( (array1.get(key), [array2.get(key)], array3.get(key)) )
            np.testing.assert_allclose(result[key], true)

        result = isopy.concatenate( (array1, array2, array3) )
        keys = array1.keys | array2.keys | array3.keys
        assert result.keys == keys
        for key in keys:
            true = np.concatenate((array1.get(key), [array2.get(key)], array3.get(key)))
            np.testing.assert_allclose(result[key], true)

        result = isopy.concatenate( [array1, array2, array3] )
        keys = array1.keys | array2.keys | array3.keys
        assert result.keys == keys
        for key in keys:
            true = np.concatenate((array1.get(key), [array2.get(key)], array3.get(key)))
            np.testing.assert_allclose(result[key], true)

        result = isopy.concatenate(array1, None, array3, axis=0)
        keys = array1.keys + array3.keys
        assert result.keys == keys
        for key in keys:
            true = np.concatenate((array1.get(key), [np.nan, np.nan], array3.get(key)))
            np.testing.assert_allclose(result[key], true)

        # Raises exception because it cannot guess the size of the mising array
        with pytest.raises(ValueError):
            result = isopy.concatenate(array1, array2, None, axis=0)

        # np.concencate
        result = np.concatenate( (array1) )
        assert result is not array1
        assert_array_equal_array(result, array1)

        result = np.concatenate((array1, array2, array3))
        keys = array1.keys | array2.keys | array3.keys
        assert result.keys == keys
        for key in keys:
            true = np.concatenate((array1.get(key), [array2.get(key)], array3.get(key)))
            np.testing.assert_allclose(result[key], true)

        result = np.concatenate([array1, array2, array3])
        keys = array1.keys | array2.keys | array3.keys
        assert result.keys == keys
        for key in keys:
            true = np.concatenate((array1.get(key), [array2.get(key)], array3.get(key)))
            np.testing.assert_allclose(result[key], true)


        # Axis = 0 with default value

        array1 = isopy.ones(2, 'ru pd cd'.split())
        array2 = isopy.ones(1, 'pd ag107 cd'.split()) * 2
        array3 = isopy.ones(2, '101ru ag107'.split()) * 3

        result = isopy.concatenate(array1, array2, array3, default_value=100)
        keys = array1.keys | array2.keys | array3.keys
        assert result.keys == keys
        for key in keys:
            true = np.concatenate((array1.get(key, 100), array2.get(key, 100), array3.get(key, 100)))
            np.testing.assert_allclose(result[key], true)

        result = isopy.concatenate(array1, None, array3, axis=0, default_value=66)
        keys = array1.keys | array3.keys
        assert result.keys == keys
        for key in keys:
            true = np.concatenate((array1.get(key, 66), [66, 66], array3.get(key, 66)))
            np.testing.assert_allclose(result[key], true)

        # Axis = 1
        array1 = isopy.ones(2, 'ru pd cd'.split())
        array2 = isopy.ones(1, 'rh ag'.split()) * 2
        array3 = isopy.ones(2, '103rh 107ag'.split()) * 3

        result = isopy.concatenate(array1, axis=1)
        assert result is not array1
        assert_array_equal_array(result, array1)

        result = isopy.concatenate(array1, array2, array3, axis=1)
        assert result.keys == array1.keys + array2.keys + array3.keys
        for key, v in array1.items():
            np.testing.assert_allclose(result[key], v)
        for key, v in array2.items():
            np.testing.assert_allclose(result[key], v.repeat(2))
        for key, v in array3.items():
            np.testing.assert_allclose(result[key], v)

        result = isopy.concatenate(array1, None, array3, axis=1)
        assert result.keys == array1.keys + array3.keys
        for key, v in array1.items():
            np.testing.assert_allclose(result[key], v)
        for key, v in array3.items():
            np.testing.assert_allclose(result[key], v)

        # np.concatenate

        result = np.concatenate( (array1), axis=1)
        assert result is not array1
        assert_array_equal_array(result, array1)

        result = np.concatenate((array1, array2, array3), axis=1)
        assert result.keys == array1.keys + array2.keys + array3.keys
        for key, v in array1.items():
            np.testing.assert_allclose(result[key], v)
        for key, v in array2.items():
            np.testing.assert_allclose(result[key], v.repeat(2))
        for key, v in array3.items():
            np.testing.assert_allclose(result[key], v)

        # invalid axis

        with pytest.raises(np.AxisError):
            result = isopy.concatenate(array1, array2, array3, axis=2)

    def test_arrayfunc(self):
        # default value

        array1 = isopy.ones(4, 'ru pd cd'.split()) * [1, 2 ,3, 4]
        array2 = isopy.ones(4, 'pd ag cd'.split()) * [0.1, 0.2, 0.3, 0.4]

        result1 = isopy.arrayfunc(np.add, array1, array2)
        result2 = isopy.arrayfunc(np.add, array1, array2, default_value=666)
        result3 = isopy.arrayfunc(np.add, array1, array2, default_value=(666, 111))
        result4 = isopy.arrayfunc(np.add, array1, array2, default_value=[111, 222, 333, 444])
        result5 = result1 = isopy.arrayfunc(isopy.add, x2 = array2, x1 = array1)
        assert result1.keys == 'ru pd cd ag'.split()
        assert result2.keys == 'ru pd cd ag'.split()
        assert result3.keys == 'ru pd cd ag'.split()
        for key in result1.keys():
            true1 = np.add(array1.get(key), array2.get(key))
            true2 = np.add(array1.get(key, 666), array2.get(key, 666))
            true3 = np.add(array1.get(key, 666), array2.get(key, 111))
            true4 = np.add(array1.get(key, [111, 222, 333, 444]), array2.get(key, [111, 222, 333, 444]))
            np.testing.assert_allclose(result1[key], true1)
            np.testing.assert_allclose(result2[key], true2)
            np.testing.assert_allclose(result3[key], true3)
            np.testing.assert_allclose(result4[key], true4)
            np.testing.assert_allclose(result5[key], true1)

        # keys
        array1 = isopy.ones(4, 'ru pd cd'.split()) * [1, 2, 3, 4]
        array2 = isopy.ones(4, 'pd ag cd'.split()) * [0.1, 0.2, 0.3, 0.4]

        result = isopy.arrayfunc(np.add, array1, array2, keys='rh pd ag'.split())
        assert result.keys == 'rh pd ag'.split()
        for key in result.keys:
            true = np.add(array1.get(key), array2.get(key))
            np.testing.assert_allclose(result[key], true)


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

        assert core.iskeystring(key.str()) is False
        assert core.iskeystring(key) is True

        assert core.iskeylist(keylist.strlist()) is False
        assert core.iskeylist(keylist) is True

        assert core.isarray(array.to_list()) is False
        assert core.isarray(array) is True

    def test_classname(self):
        keylist = isopy.keylist('101 105 111'.split())
        assert core.get_classname(keylist) == isopy.MassKeyList.__name__
        assert core.get_classname(isopy.MassKeyList) == isopy.MassKeyList.__name__

    def test_flavour(self):
        mass = isopy.keylist('101 105 111'.split())
        element = isopy.keylist('ru pd cd'.split())
        isotope = isopy.keylist('101ru 105pd 111cd'.split())
        ratio = isopy.keylist('ru/pd pd/pd cd/pd'.split())
        general = isopy.keylist('harry ron hermione'.split())
        mixed = isopy.keylist('ru 105pd cd/pd'.split())

        keys = (mass, element, isotope, ratio, general, mixed)
        flavours = (core.MassFlavour, core.ElementFlavour, core.IsotopeFlavour,
                    core.RatioFlavour, core.GeneralFlavour, core.MixedFlavour)
        names = 'mass Element ISOTOPE RatiO GENeral mixed'.split()

        for i in range(6):
            assert not flavours[i] == [i]
            assert not keys[i].flavour == i
            assert not flavours[i].flavour == float(i)

            for j in range(6):
                if i == j:
                    assert keys[i].flavour == flavours[j]
                    assert flavours[i].flavour == flavours[j]

                    assert keys[i].flavour == flavours[j]
                    assert keys[i].flavour == keys[j].flavour
                    assert keys[i].flavour == names[j]

                    assert flavours[i].flavour == flavours[j]
                    assert flavours[i].flavour == keys[j].flavour
                    assert flavours[i].flavour == names[j]

                    assert names[i] == keys[j].flavour
                    assert str(keys[i].flavour) == flavours[i].__name__
                else:
                    assert keys[i].flavour != flavours[j]
                    assert flavours[i].flavour != flavours[j]

                    assert keys[i].flavour != flavours[j]
                    assert keys[i].flavour != keys[j].flavour
                    assert keys[i].flavour != names[j]

                    assert flavours[i].flavour != flavours[j]
                    assert flavours[i].flavour != keys[j].flavour
                    assert flavours[i].flavour != names[j]

                    assert names[i] != keys[j].flavour
                    assert str(keys[i].flavour) != flavours[i].__class__.__name__

    def test_allowed_numpy_functions(self):
        # These are non-vital so just make sure they return a string
        result = isopy.allowed_numpy_functions()
        assert isinstance(result, str)

        result = isopy.allowed_numpy_functions('name')
        assert isinstance(result, str)

        result = isopy.allowed_numpy_functions('link')
        assert isinstance(result, str)

        result = isopy.allowed_numpy_functions('rst')
        assert isinstance(result, str)

        result = isopy.allowed_numpy_functions('markdown')
        assert isinstance(result, str)

        # this returns a list
        result = isopy.allowed_numpy_functions(delimiter=None)
        assert isinstance(result, list)
        assert False not in [isinstance(string, str) for string in result]

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
        assert str(isopy.core.NotGiven) == 'N/A'
        assert repr(isopy.core.NotGiven) == 'N/A'













