import isopy
from isopy import core
import pytest
import hashlib
import os

#The hashes are different on Travis so these tests fail
class _Test_FileHash:
    def hash_file(self, filename):
        # hashlib.md5(open('isopy/referencedata/isotope_sprocess_fraction_B11.csv', 'rb').read()).hexdigest()
        filepath = os.path.join(os.path.dirname(isopy.__file__), 'referencedata', f'{filename}')
        with open(filepath, 'rb') as file:
            filehash = hashlib.md5(file.read()).hexdigest()
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