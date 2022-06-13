import isopy
from isopy import core
import pytest
import hashlib
import os
import numpy as np

#The hashes are different on Travis so these tests fail
class _Test_Integrity:
    def hash_file(self, filename):
        filepath = os.path.join(os.path.dirname(isopy.__file__), 'referencedata', f'{filename}')
        with open(filepath, 'rb') as file:
            filehash = hashlib.md5(file.read()).hexdigest().lower()
        return filehash

    def test_element_atomic_number(self):
        filename = 'element_atomic_number.csv'
        valid_hash = 'dd448ec495daa9c09ab0507507808b24'
        assert self.hash_file(filename) == valid_hash

    def test_element_symbol_name(self):
        filename = 'element_symbol_name.csv'
        valid_hash = 'b67bd960a682ea7ce6563d1a18c7f276'
        assert self.hash_file(filename) == valid_hash

    def test_element_initial_solar_system_abundance_L09(self):
        filename = 'element_initial_solar_system_abundance_L09.csv'
        valid_hash = 'e7b750542204ca5fb6515273f87f8d0b'
        assert self.hash_file(filename) == valid_hash

    def test_isotope_initial_solar_system_abundance_L09(self):
        filename = 'isotope_initial_solar_system_abundance_L09.csv'
        valid_hash = 'c77eccd80ccebe83a486ca282092c8a7'
        assert self.hash_file(filename) == valid_hash

    def test_isotope_initial_solar_system_fraction_L09(self):
        filename = 'isotope_initial_solar_system_fraction_L09.csv'
        valid_hash = '5c36d889eca012eeb04949e74ad49158'
        assert self.hash_file(filename) == valid_hash

    def test_isotope_best_measurement_fraction_M16(self):
        filename = 'isotope_best_measurement_fraction_M16.csv'
        valid_hash = '8a44595631a4ed269691f17b3d689fc0'
        assert self.hash_file(filename) == valid_hash

    def test_isotope_mass_W17(self):
        filename = 'isotope_mass_W17.csv'
        valid_hash = '2e754cd8f5edec16e7afe2eedc5dbcb5'
        assert self.hash_file(filename) == valid_hash

    def test_isotope_sprocess_fraction_B11(self):
        filename = 'isotope_sprocess_fraction_B11.csv'
        valid_hash = '3b926ab45f85e6ff815da2907368dff8'
        assert self.hash_file(filename) == valid_hash

class Test:
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

    def test_call(self):
        assert isopy.refval('mass.isotopes') is isopy.refval.mass.isotopes
        assert isopy.refval('element.atomic_weight') is isopy.refval.element.atomic_weight
        assert isopy.refval('isotope.mass') is isopy.refval.isotope.mass

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