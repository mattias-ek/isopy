import isopy
from isopy import core
import pytest
import hashlib
import os
import numpy as np

# Some hashes seem to be different on Travis compared to my mac
class Test_Reference_Value:
    def hash_file(self, filename):
        filepath = os.path.join(os.path.dirname(isopy.__file__), 'referencedata', f'{filename}')
        with open(filepath, 'rb') as file:
            filehash = hashlib.md5(file.read()).hexdigest().lower()
        return filehash

    # Mass
    def test_mass_isotopes(self):
        assert type(isopy.refval.mass.isotopes) is isopy.IsopyDict

        assert isopy.refval.mass.isotopes[40] == ('40Ar', '40K', '40Ca')
        assert isopy.refval.mass.isotopes.get(96) == ('96Zr', '96Mo', '96Ru')

    # Element
    def test_element_all_symbols(self):
        assert type(isopy.refval.element.all_symbols) is tuple
        assert 'pd' in isopy.refval.element.all_symbols

    def test_element_symbol_name(self):
        filename = 'element_symbol_name.csv'
        valid_hash = ['b67bd960a682ea7ce6563d1a18c7f276', '1b7ea0b32389d4d601cdfecd327eb862']
        assert self.hash_file(filename) in valid_hash
        assert type(isopy.refval.element.symbol_name) is isopy.IsopyDict

        assert isopy.refval.element.symbol_name['pd'] == 'Palladium'
        assert isopy.refval.element.symbol_name.get('Ge') == 'Germanium'

    def test_element_name_symbol(self):
        assert type(isopy.refval.element.name_symbol) is dict

        assert isopy.refval.element.name_symbol['Palladium'] == 'Pd'
        assert isopy.refval.element.name_symbol.get('Germanium') == 'Ge'

    def test_element_atomic_weight(self):
        assert type(isopy.refval.element.atomic_weight) is isopy.ScalarDict

        np.testing.assert_allclose(isopy.refval.element.atomic_weight['pd'], 106.41532788648)
        np.testing.assert_allclose(isopy.refval.element.atomic_weight.get('ge'), 72.6295890304831)

    def test_element_atomic_number(self):
        filename = 'element_atomic_number.csv'
        valid_hash = ['dd448ec495daa9c09ab0507507808b24','71ddcf788ab22ad8e130c282775e41d8']
        assert self.hash_file(filename) in valid_hash
        assert type(isopy.refval.element.atomic_number) is isopy.IsopyDict

        assert isopy.refval.element.atomic_number['pd'] == 46
        assert isopy.refval.element.atomic_number.get('ge') == 32

    def test_element_initial_solar_system_abundance_L09(self):
        filename = 'element_initial_solar_system_abundance_L09.csv'
        valid_hash = 'e7b750542204ca5fb6515273f87f8d0b'
        assert self.hash_file(filename) == valid_hash
        assert type(isopy.refval.element.initial_solar_system_abundance_L09) is isopy.ScalarDict

        np.testing.assert_allclose(isopy.refval.element.initial_solar_system_abundance_L09['pd'], 1.36)
        np.testing.assert_allclose(isopy.refval.element.initial_solar_system_abundance_L09.get('ge'), 115.0)

    # Isotope
    def test_isotope_mass_W17(self):
        filename = 'isotope_mass_W17.csv'
        valid_hash = ['2e754cd8f5edec16e7afe2eedc5dbcb5', '457605e1208eb35b043d16b4514a53b3']
        assert self.hash_file(filename) in valid_hash
        assert type(isopy.refval.isotope.mass_W17) is isopy.ScalarDict

        np.testing.assert_allclose(isopy.refval.isotope.mass_W17['105pd'], 104.9050795)
        np.testing.assert_allclose(isopy.refval.isotope.mass_W17.get('76Ge'), 75.92140273)
        np.testing.assert_allclose(isopy.refval.isotope.mass_W17.get('108pd/105pd'), 1.0285859589859039)

    def test_isotope_mass_number(self):
        assert type(isopy.refval.isotope.mass_number) is isopy.ScalarDict

        np.testing.assert_allclose(isopy.refval.isotope.mass_number['105pd'], 105)
        np.testing.assert_allclose(isopy.refval.isotope.mass_number.get('76Ge'), 76)
        np.testing.assert_allclose(isopy.refval.isotope.mass_number.get('108pd/105pd'),1.0285714285714285)

    def test_isotope_best_measurement_fraction_M16(self):
        filename = 'isotope_best_measurement_fraction_M16.csv'
        valid_hash = '8a44595631a4ed269691f17b3d689fc0'
        assert self.hash_file(filename) == valid_hash
        assert type(isopy.refval.isotope.best_measurement_fraction_M16) is isopy.ScalarDict

        np.testing.assert_allclose(isopy.refval.isotope.best_measurement_fraction_M16['105pd'], 0.2233)
        np.testing.assert_allclose(isopy.refval.isotope.best_measurement_fraction_M16.get('76Ge'), 0.07745)
        np.testing.assert_allclose(isopy.refval.isotope.best_measurement_fraction_M16.get('108pd/105pd'), 1.1849529780564263)

    def test_isotope_initial_solar_system_fraction_L09(self):
        filename = 'isotope_initial_solar_system_fraction_L09.csv'
        valid_hash = '5c36d889eca012eeb04949e74ad49158'
        assert self.hash_file(filename) == valid_hash
        assert type(isopy.refval.isotope.initial_solar_system_fraction_L09) is isopy.ScalarDict

        np.testing.assert_allclose(isopy.refval.isotope.initial_solar_system_fraction_L09['105pd'], 0.2233)
        np.testing.assert_allclose(isopy.refval.isotope.initial_solar_system_fraction_L09.get('76Ge'), 0.07444)
        np.testing.assert_allclose(isopy.refval.isotope.initial_solar_system_fraction_L09.get('108pd/105pd'), 1.1849529780564263)

    def test_isotope_initial_solar_system_abundance_L09(self):
        filename = 'isotope_initial_solar_system_abundance_L09.csv'
        valid_hash = 'c77eccd80ccebe83a486ca282092c8a7'
        assert self.hash_file(filename) == valid_hash
        assert type(isopy.refval.isotope.initial_solar_system_abundance_L09) is isopy.ScalarDict

        np.testing.assert_allclose(isopy.refval.isotope.initial_solar_system_abundance_L09['105pd'], 0.3032)
        np.testing.assert_allclose(isopy.refval.isotope.initial_solar_system_abundance_L09.get('76Ge'), 8.5)
        np.testing.assert_allclose(isopy.refval.isotope.initial_solar_system_abundance_L09.get('108pd/105pd'), 1.184036939313984)

    def test_isotope_initial_solar_system_abundance_L09b(self):
        assert type(isopy.refval.isotope.initial_solar_system_abundance_L09) is isopy.ScalarDict

        np.testing.assert_allclose(isopy.refval.isotope.initial_solar_system_abundance_L09b['105pd'], 0.303688)
        np.testing.assert_allclose(isopy.refval.isotope.initial_solar_system_abundance_L09b.get('76Ge'), 8.5606)
        np.testing.assert_allclose(isopy.refval.isotope.initial_solar_system_abundance_L09b.get('108pd/105pd'), 1.1849529780564263)

    def test_isotope_sprocess_fraction_B11(self):
        filename = 'isotope_sprocess_fraction_B11.csv'
        valid_hash = '3b926ab45f85e6ff815da2907368dff8'
        assert self.hash_file(filename) == valid_hash
        assert type(isopy.refval.isotope.sprocess_fraction_B11) is isopy.ScalarDict

        np.testing.assert_allclose(isopy.refval.isotope.sprocess_fraction_B11['105pd'], 0.157)
        np.testing.assert_allclose(isopy.refval.isotope.sprocess_fraction_B11.get('mo95'), 0.696)
        np.testing.assert_allclose(isopy.refval.isotope.sprocess_fraction_B11.get('108pd/105pd'), 4.751592356687898)


class Test_Default_value:
    def test_isotope_mass(self):
        try:
            assert isopy.refval.isotope.mass is isopy.refval.isotope.mass_W17

            isopy.refval.isotope.mass = isopy.refval.isotope.mass_number
            assert isopy.refval.isotope.mass is isopy.refval.isotope.mass_number

            isopy.refval.reset()
            assert isopy.refval.isotope.mass is isopy.refval.isotope.mass_W17

            alt = {str(key): key.mass_number for key in isopy.refval.element.isotopes['pd']}
            with pytest.raises(TypeError):
                isopy.refval.isotope.mass = alt

            alt = isopy.IsopyDict(alt)
            with pytest.raises(TypeError):
                isopy.refval.isotope.mass = alt

            alt = isopy.ScalarDict(alt)
            isopy.refval.isotope.mass = alt
            assert isopy.refval.isotope.mass is alt
        finally:
            # So that other tests dont use the wrong reference value
            isopy.refval.reset()

    def test_isotope_fraction(self):
        try:
            assert isopy.refval.isotope.fraction is isopy.refval.isotope.best_measurement_fraction_M16

            isopy.refval.isotope.fraction = isopy.refval.isotope.initial_solar_system_fraction_L09
            assert isopy.refval.isotope.fraction is isopy.refval.isotope.initial_solar_system_fraction_L09

            isopy.refval.reset()
            assert isopy.refval.isotope.fraction is isopy.refval.isotope.best_measurement_fraction_M16

            alt = {str(key): key.mass_number for key in isopy.refval.element.isotopes['pd']}
            with pytest.raises(TypeError):
                isopy.refval.isotope.fraction = alt

            alt = isopy.IsopyDict(alt)
            with pytest.raises(TypeError):
                isopy.refval.isotope.fraction = alt

            alt = isopy.ScalarDict(alt)
            isopy.refval.isotope.fraction = alt
            assert isopy.refval.isotope.fraction is alt
        finally:
            # So that other tests dont use the wrong reference value
            isopy.refval.reset()


class Test_misc:
    def test_repr(self):
        # Just see that the function work

        repr(isopy.refval)
        repr(isopy.refval.mass)
        repr(isopy.refval.element)
        repr(isopy.refval.isotope)

    def test_ls(self):
        isopy.refval.ls()

        isopy.refval.mass.ls()
        isopy.refval.element.ls()
        isopy.refval.isotope.ls()

    def test_call_string(self):
        assert isopy.refval('mass.isotopes') is isopy.refval.mass.isotopes
        assert isopy.refval('element.atomic_weight') is isopy.refval.element.atomic_weight
        assert isopy.refval('isotope.mass') is isopy.refval.isotope.mass

        with pytest.raises(ValueError):
            # Unable to split into two
            isopy.refval('mass_isotopes')

        with pytest.raises(ValueError):
            # ratio does not exist
            isopy.refval('ratio.isotopes')

        with pytest.raises(ValueError):
            # reference value does not exist
            isopy.refval('isotopes.doesnt_exist')

    def test_call_dict(self):
        data = dict(Pd=[1.1, 2.2, 3.3], Cd=[11.1, np.nan, 13.3], Ru=[21.1, 22.2, 23.3])
        refval = isopy.refval(data)
        assert type(refval) is dict

        refval = isopy.refval(data, dict)
        assert type(refval) is dict

        refval = isopy.refval(data, isopy.IsopyDict)
        assert type(refval) is isopy.IsopyDict

        refval = isopy.refval(data, isopy.ScalarDict)
        assert type(refval) is isopy.ScalarDict

        data = isopy.IsopyDict(data)
        refval = isopy.refval(data)
        assert type(refval) is isopy.IsopyDict

        refval = isopy.refval(data, dict)
        assert type(refval) is isopy.IsopyDict

        refval = isopy.refval(data, isopy.IsopyDict)
        assert type(refval) is isopy.IsopyDict

        refval = isopy.refval(data, isopy.ScalarDict)
        assert type(refval) is isopy.ScalarDict

        data = isopy.ScalarDict(data)
        refval = isopy.refval(data)
        assert type(refval) is isopy.ScalarDict

        refval = isopy.refval(data, dict)
        assert type(refval) is isopy.ScalarDict

        refval = isopy.refval(data, isopy.IsopyDict)
        assert type(refval) is isopy.ScalarDict

        refval = isopy.refval(data, isopy.ScalarDict)
        assert type(refval) is isopy.ScalarDict