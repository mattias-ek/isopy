import isopy
from isopy import core
from isopy import io
import numpy as np
import functools
import os

__all__ = ['refval']

########################
### Reference Values ###
########################

def _load_RV_values(filename, datatype=None):
    filepath = os.path.join(os.path.dirname(__file__), 'referencedata', f'{filename}.csv')
    data = io.read_csv(filepath)

    if datatype is not None:
        data =  {key: datatype(value[0]) for key, value in data.items()}
    else:
        data = {key: value[0] for key, value in data.items()}
    return data

# Assumes all default and reference values are properties
def is_default_value(func):
    setattr(func.fget, '_defval', True)
    setattr(func.fget, '_descr', func.fget.__doc__.lstrip('\n').split('\n', 1)[0].strip())
    return func
    
def is_reference_value(func):
    setattr(func.fget, '_refval', True)
    setattr(func.fget, '_descr', func.fget.__doc__.lstrip('\n').split('\n', 1)[0].strip())
    return func

class RefValGroup:
    def __repr__(self):
        defval = []
        refval = []
        for name, item in self.__class__.__dict__.items():
            if type(item) is property:
                if getattr(item.fget, '_defval', False):
                    defval.append(f'{name} - {item.fget._descr}')
                if getattr(item.fget, '_refval', False):
                    refval.append(f'{name} - {item.fget._descr}')
        
        string = ''
        if defval:
            string += 'This group contains the following default values:\n\t'
            string += '\n\t'.join(defval)
        if string: string += '\n\n'
        string += 'This group contains the following reference values:\n\t'
        string += '\n\t'.join(refval)

        return string

    def ls(self):
        """
        Returns a list will all reference values in this group.
        """
        defval = []
        refval = []
        for name, item in self.__class__.__dict__.items():
            if getattr(item.fget, '_defval', False):
                defval.append(name)
            if getattr(item.fget, '_refval', False):
                refval.append(name)
            
        return defval + refval


class mass(RefValGroup):
    @is_reference_value        
    @core.cached_property
    def isotopes(self):
        """
        Dictionary containing all naturally isotopes with a given mass number.

        The dictionary is constructed from the isotopes in
        :attr:`isotope.best_abundance_measurement_M16`.

        The ``get()`` method of this dictionary will
        return an empty :class:`IsotopeKeyList` for absent keys.

        Examples
        --------
        >>> isopy.refval.mass.isotopes[40]
        IsotopeKeyList('40Ar', '40K', '40Ca')

        >>> isopy.refval.mass.isotopes.get(96)
        IsotopeKeyList('96Zr', '96Mo', '96Ru')
        """

        fraction = isopy.refval.isotope.best_measurement_fraction_M16
        isotopes = {f'{mass}':
                    isopy.IsotopeKeyList([k for k in fraction if k._filter(mass_number = {'key_eq': (mass,)})])
                    for mass in range(1, 239)}
        return core.IsopyDict(**{key: value for key, value in isotopes.items() if len(value) > 0},
                               default_value=isopy.IsotopeKeyList(), readonly=True)


#TODO use best isotope and isotope weight to calculate atomic weight
class element(RefValGroup):
    @is_reference_value
    @core.cached_property
    def isotopes(self):
        """
        Dictionary containing all naturally occurring isotopes for each element symbol.

        The dictionary is constructed from the isotopes in
        :attr:`isotope.best_abundance_measurement_M16`.

        The ``get()`` method of this dictionary will
        return an empty :class:`IsotopeKeyList` for absent keys.

        Examples
        --------
        >>> isopy.refval.element.isotopes['pd']
        IsotopeKeyList('102Pd', '104Pd', '105Pd', '106Pd', '108Pd', '110Pd')

        >>> isopy.refval.element.isotopes.get('ge')
        IsotopeKeyList('70Ge', '72Ge', '73Ge', '74Ge', '76Ge')
        """

        fraction = isopy.refval.isotope.best_measurement_fraction_M16
        return core.IsopyDict(**{symbol: isopy.IsotopeKeyList([k for k in fraction if k._filter(element_symbol = {'key_eq': (symbol,)})])
                               for symbol in self.all_symbols}, default_value=isopy.IsotopeKeyList(), readonly=True)
    
    @is_reference_value
    @core.cached_property
    def all_symbols(self):
        """
        A tuple of all the element symbols.

        Examples
        --------
        >>> isopy.refval.element.all_symbols[:5] # first 5 element symbols
        (ElementKeyString('H'), ElementKeyString('He'), ElementKeyString('Li'),
         ElementKeyString('Be'), ElementKeyString('B'))
        """
        return tuple(self.symbol_name.keys())
    
    @is_reference_value
    @is_default_value  
    @core.cached_property
    def symbol_name(self):
        """
        Dictionary containing the full element name mapped to the element symbol.

        The first letter of the element name is capitalised.

        The ``get()`` method of this dictionary will
        return ``None`` for absent keys.

        Examples
        --------
        >>> isopy.refval.element.symbol_name['pd']
        'Palladium'

        >>> isopy.refval.element.symbol_name.get('ge')
        'Germanium'
        """
        data = {isopy.ElementKeyString(key, allow_reformatting=False): value for key, value in _load_RV_values('element_symbol_name').items()}
        return core.IsopyDict(data, readonly = True)
        
    @is_reference_value
    @core.cached_property
    def name_symbol(self):
        """
        Dictionary containing the element symbol mapped to the full element name.

        The first letter of the element name must be capitalised.

        The ``get()`` method of this dictionary will
        return ``None`` for absent keys.

        Examples
        --------
        >>> isopy.refval.element.symbol_name['Palladium']
        ElementKeyString('Pd')

        >>> isopy.refval.element.symbol_name.get('Germanium')
        ElementKeyString('Ge')
        """
        return dict({name: symbol for symbol, name in self.symbol_name.items()}, readonly = True)

    @is_reference_value
    @core.cached_property
    def atomic_weight(self):
        """
        Dictionary containing the atomic weight for each element symbol.

        The atomic weights are calculated using the isotopic abundances from
        :attr:`isotope.best_abundance_measurement_M16` and the isotopic masses from
        :attr:`isotope.isotope.mass_W17`.

        The ``get()`` method of this dictionary will
        return ``None`` for absent keys.

        Examples
        --------
        >>> isopy.refval.element.atomic_weight['pd']
        106.41532788648

        >>> isopy.refval.element.atomic_weight.get('ge')
        72.6295890304831
        """
        weights = {}
        # Isotopes is also taken from best_measurement_fraction_M16
        for element, isotopes in self.isotopes.items():
            weights[element] = np.sum([refval.isotope.mass_W17.get(isotope) *
                                       refval.isotope.best_measurement_fraction_M16.get(isotope)
                                       for isotope in isotopes])
        return core.IsopyDict(**weights, readonly=True)
        
    @is_reference_value
    @core.cached_property
    def atomic_number(self):
        """
        Dictionary containing the atomic number for each element symbol.

        The ``get()`` method of this dictionary will
        return ``None`` for absent keys.

        Examples
        --------
        >>> isopy.refval.element.atomic_number['pd']
        46

        >>> isopy.refval.element.atomic_number.get('ge')
        32
        """
        return core.IsopyDict(_load_RV_values('element_atomic_number', int), readonly = True)
        
    @is_reference_value
    @core.cached_property
    def initial_solar_system_abundance_L09(self):
        """
        Dictionary containing the element abundance of the initial solar system composition (normalized to N(Si) = 10^6 atoms) from Lodders et al. (2019).

        Reference: `Lodders et al. (2019) <http://materials.springer.com/lb/docs/sm_lbs_978-3-540-88055-4_34>`_.

        The ``get()`` method of this dictionary will automatically calculate the ratio of two
        isotopes if both are present the dictionary. The ``get()`` method will return ``np.nan``
        for absent keys.
        """
        return core.ScalarDict(_load_RV_values('element_initial_solar_system_abundance_L09', np.float64), default_value=np.nan, readonly = True)

class isotope(RefValGroup):
    def __init__(self):
        self.__mass = None
        self.__fraction = None
    
    @is_default_value
    @property
    def mass(self):
        """
        Dictionary containing the default mass of each isotope.

        This is the default dictionary used in functions where the isotope mass is required.

        This dictionary is initially a copy of :attr:`isotope.mass_W17` but can be changed by the
        user. **Note** the default values used can change in future versions of isopy as newer
        values become avaliable.

        Examples
        --------
        >>> isopy.refval.isotope.mass['pd105']
        104.9050795

        >>> isopy.refval.isotope.mass.get('ge76')
        75.92140273

        >>> isopy.refval.isotope.mass.get('pd108/pd105')
        1.0285859589859039
        """
        if not self.__mass: self.__mass = self.mass_W17
        return self.__mass

    @mass.setter
    def mass(self, value):
        if not isinstance(value, core.IsopyDict):
            raise TypeError('attribute must be a dictionary')
        self.__mass = value
    
    @is_default_value
    @property
    def fraction(self):
        """
        Dictionary containing the default fraction of each isotope.

        This is the default dictionary used in functions where the isotope abundance is required.

        This dictionary is initially a copy of :attr:`isotope.best_measurement_fracton_M16`
        but can be changed by the user. **Note** the default values used can change in future
        versions of isopy as newer values become avaliable.

        Examples
        --------
        >>> isopy.refval.isotope.best_measurement_fraction_M16['pd105']
        0.2233

        >>> isopy.refval.isotope.best_measurement_fraction_M16.get('ge76')
        0.07745

        >>> isopy.refval.isotope.best_measurement_fraction_M16.get('pd108/pd105')
        1.1849529780564263
        """
        if not self.__fraction: self.__fraction = self.best_measurement_fraction_M16
        return self.__fraction

    @fraction.setter
    def fraction(self, value):
        if not isinstance(value, core.IsopyDict):
            raise TypeError('attribute must be a dictionary')
        self.__fraction = value
    
    @is_reference_value
    @core.cached_property
    def mass_W17(self):
        """
        Dictionary containing isotope mass of each isotope from Wang et al. (2016).

        Reference: `Wang et al. (2016) <https://doi.org/10.1088/1674-1137/41/3/030003>`_.

        The ``get()`` method of this dictionary will automatically calculate the ratio of two
        isotopes if both are present the dictionary. The ``get()`` method will return ``np.nan``
        for absent keys.

        Examples
        --------
        >>> isopy.refval.isotope.mass_W17['pd105']
        104.9050795

        >>> isopy.refval.isotope.mass_W17.get('ge76')
        75.92140273

        >>> isopy.refval.isotope.mass_W17.get('pd108/pd105')
        1.0285859589859039
        """
        return core.ScalarDict(_load_RV_values('isotope_mass_W17', np.float64), default_value=np.nan, readonly = True)

    @is_reference_value
    @core.cached_property
    def mass_number(self):
        """
        Dictionary containing mass number of each isotope.

        The ``get()`` method of this dictionary will automatically calculate the ratio of two
        isotopes if both are present the dictionary. The ``get()`` method will return ``np.nan``
        for absent keys.

        Examples
        --------
        >>> isopy.refval.isotope.mass_number['pd105']
        105

        >>> isopy.refval.isotope.mass_number.get('ge76')
        76

        >>> isopy.refval.isotope.mass_number.get('pd108/pd105')
        1.0285714285714285
        """
        return core.ScalarDict({key: int(key.mass_number) for key in self.mass_W17},
                               default_value=np.nan, readonly=True)
    
    @is_reference_value
    @core.cached_property
    def best_measurement_fraction_M16(self):
        """
        Dictionary containing the isotope fraction from the best avaliable measurement from Meija et al. (2016).

        Reference: `Meija et al. (2016) <https://doi.org/10.1515/pac-2015-0503>`_.

        The ``get()`` method of this dictionary will automatically calculate the ratio of two
        isotopes if both are present the dictionary. The ``get()`` method will return ``np.nan``
        for absent keys.

        Examples
        --------
        >>> isopy.refval.isotope.best_measurement_fraction_M16['pd105']
        0.2233

        >>> isopy.refval.isotope.best_measurement_fraction_M16.get('ge76')
        0.07745

        >>> isopy.refval.isotope.best_measurement_fraction_M16.get('pd108/pd105')
        1.1849529780564263
        """
        return core.ScalarDict(_load_RV_values('isotope_best_measurement_fraction_M16', np.float64), default_value=np.nan, readonly = True)
        
    @is_reference_value
    @core.cached_property
    def initial_solar_system_fraction_L09(self):
        """
        Dictionary containing the isotope fraction of the inital solar system composition from Lodders et al. (2019).

        Reference: `Lodders et al. (2019) <http://materials.springer.com/lb/docs/sm_lbs_978-3-540-88055-4_34>`_.

        The ``get()`` method of this dictionary will automatically calculate the ratio of two
        isotopes if both are present the dictionary. The ``get()`` method will return ``np.nan``
        for absent keys.

        Examples
        --------
        >>> isopy.refval.isotope.initial_solar_system_fraction_L09['pd105']
        0.2233

        >>> isopy.refval.isotope.initial_solar_system_fraction_L09.get('ge76')
        0.07444

        >>> isopy.refval.isotope.initial_solar_system_fraction_L09.get('pd108/pd105')
        1.1849529780564263
        """
        return core.ScalarDict(_load_RV_values('isotope_initial_solar_system_fraction_L09', np.float64), default_value = np.nan, readonly = True)
        
    @is_reference_value
    @core.cached_property
    def initial_solar_system_abundance_L09(self):
        """
        Dictionary containing the isotope abundance of the inital solar system composition (normalized to N(Si) = 10^6 atoms) from Lodders et al. (2019).

        **Note** These values are not directly taken from the table but calculated from the elemental abundances and
        the isotope fractions given the table. This ensures consistency between all three reference values. Discrepancies
        between the calculated isotope abundances and those listen in the table are due to rounding errors.

        Reference: `Lodders et al. (2019) <http://materials.springer.com/lb/docs/sm_lbs_978-3-540-88055-4_34>`_.

        The ``get()`` method of this dictionary will automatically calculate the ratio of two
        isotopes if both are present the dictionary. The ``get()`` method will return ``np.nan``
        for absent keys.

        Examples
        --------
        >>> isopy.refval.isotope.initial_solar_system_abundance_L09['pd105']
        0.303688

        >>> isopy.refval.isotope.initial_solar_system_abundance_L09.get('ge76')
        8.5606

        >>> isopy.refval.isotope.initial_solar_system_abundance_L09.get('pd108/pd105')
        1.1849529780564263
        """
        element_abundance = isopy.refval.element.initial_solar_system_abundance_L09
        isotope_fraction = isopy.refval.isotope.initial_solar_system_fraction_L09
        return core.ScalarDict({key: value * element_abundance.get(key.element_symbol) for key, value in isotope_fraction.items()},
                               default_value=np.nan, readonly = True)
        
    @is_reference_value
    @core.cached_property
    def sprocess_fraction_B11(self):
        """
        Dictionary containing the estimated s-process fraction of each isotope from Bisterzo et al. (2011).

        Reference: `Bisterzo et al. (2011) <https://doi.org/10.1111/j.1365-2966.2011.19484.x>`_.

        The ``get()`` method of this dictionary will automatically calculate the ratio of two
        isotopes if both are present the dictionary. The ``get()`` method will return ``np.nan``
        for absent keys.

        Examples
        --------
        >>> isopy.refval.isotope.initial_solar_system_abundance_L09['pd105']
        0.157

        >>> isopy.refval.isotope.initial_solar_system_abundance_L09.get('mo95')
        0.696

        >>> isopy.refval.isotope.initial_solar_system_abundance_L09.get('pd108/pd105')
        4.751592356687898
        """
        return core.ScalarDict(_load_RV_values('isotope_sprocess_fraction_B11', np.float64), default_value=np.nan, readonly = True)


class ReferenceValues:
    """ Reference values useful for working with geochemical data"""
    def __init__(self):
        self.reset()

    def __repr__(self):
        string = 'The following groups of reference values are avaliable:\n\n'
        string += f'mass - {self.__class__.mass.fget.__doc__}\n'
        string += f'element - {self.__class__.element.fget.__doc__}\n'
        string += f'isotope - {self.__class__.isotope.fget.__doc__}'
        return string


    def reset(self):
        """Resets the reference values to the default values"""
        self._element = element()
        self._isotope = isotope()
        self._mass = mass()

    @property
    def isotope(self):
        """Reference values for different isotopes"""
        return self._isotope

    @property
    def element(self):
        """Reference values for different elements"""
        return self._element

    @property
    def mass(self):
        """Reference values for different atomic masses"""
        return self._mass
