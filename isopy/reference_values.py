import isopy
from isopy import core
from isopy import io
import numpy as np
import functools
import os

########################
### Reference Values ###
########################

def load_refval_values(filename, keys_in_first ='r'):
    filepath = os.path.join(os.path.dirname(__file__), 'referencedata', f'{filename}.csv')
    data = io.read_csv(filepath, keys_in_first=keys_in_first)
    data = {key: value[0] for key, value in data.items()}

    return data

# Assumes all default and reference values are properties
def is_default_value(func):
    setattr(func.fget, '_defval', True)
    setattr(func.fget, '_descr', func.fget.__doc__.lstrip('\n').split('\n', 1)[0].strip())
    return func


class RefValGroup:
    def __init__(self, parent):
        self._parent = parent

    def __repr__(self):
        names = "', '".join(self.ls())
        return f"isopy.refval.{self.__class__.__name__}(['{names}'])"

    def _repr_markdown_(self):
        descriptions = []
        for name in self.ls():
            doc = self.__class__.__dict__[name].__doc__
            doc = doc.split('\n\n', 1)[0].replace('\n', '').strip()
            descriptions.append(f'* **{name}** - {doc}')

        descriptions = '\n'.join(descriptions)
        return f'**isopy.refval.{self.__class__.__name__}** contains the following reference values\n{descriptions}'

    def ls(self):
        """
        Returns a list will all reference values in this group.
        """
        defval = []
        refval = []
        for name, item in self.__class__.__dict__.items():
            if type(item) is property:
                if getattr(item.fget, '_defval', False):
                    defval.append(name)
                else:
                    refval.append(name)
            
        return defval + refval

class mass(RefValGroup):
    @core.cached_property
    def isotopes(self):
        """
        Dictionary containing all naturally occuring isotopes with a given mass number.

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
                    isopy.askeylist([k for k in fraction if k._filter_(mass_number = {'key_eq': (mass,)})])
                    for mass in range(1, 239)}
        return core.IsopyDict(**{key: value for key, value in isotopes.items() if len(value) > 0},
                               default_value=isopy.askeylist(), readonly=True)

class element(RefValGroup):
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
        return core.IsopyDict(**{symbol: isopy.askeylist([k for k in fraction if k._filter_(element_symbol = {'key_eq': (symbol,)})])
                               for symbol in self.all_symbols}, default_value=isopy.askeylist(), readonly=True)

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
        return core.IsopyDict(load_refval_values('element_symbol_name').items(), readonly = True)

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
            weights[element] = np.sum([self._parent.isotope.mass_W17.get(isotope) *
                                       self._parent.isotope.best_measurement_fraction_M16.get(isotope)
                                       for isotope in isotopes])
        return core.RefValDict(**weights, readonly=True, ratio_function=np.divide,
                               molecule_functions='abundance')

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
        return core.RefValDict(load_refval_values('element_atomic_number'), readonly = True)

    @core.cached_property
    def initial_solar_system_abundance_L09(self):
        """
        Dictionary containing the element abundance of the initial solar system composition (normalized to N(Si) = 10^6 atoms) from Lodders et al. (2019).

        Reference: `Lodders et al. (2019) <http://materials.springer.com/lb/docs/sm_lbs_978-3-540-88055-4_34>`_.

        The ``get()`` method of this dictionary will automatically calculate the ratio of two
        isotopes if both are present the dictionary. The ``get()`` method will return ``np.nan``
        for absent keys.
        """
        return core.RefValDict(load_refval_values('element_initial_solar_system_abundance_L09'),
                               default_value=np.nan, readonly = True, ratio_function=np.divide,
                               molecule_functions='abundance')

class isotope(RefValGroup):
    def __init__(self, parent):
        self._parent = parent
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
        if not self.__mass: self.__mass = self.mass_AME20
        return self.__mass

    @mass.setter
    def mass(self, value):
        if not isinstance(value, core.RefValDict):
            raise TypeError('attribute must be a scalar dictionary')
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
        if not isinstance(value, core.RefValDict):
            raise TypeError('attribute must be a dictionary')
        self.__fraction = value

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
        return core.RefValDict(load_refval_values('isotope_mass_W17'),
                               default_value=np.nan, readonly = True, ratio_function=np.divide,
                               molecule_functions='mass')

    @core.cached_property
    def mass_AME20(self):
        """
        Dictionary containing isotope mass of each isotope from the 2020 Atomic Mass Evaluation.

        Reference: `Atomic Mass Evaluation 2020 <https://www-nds.iaea.org/amdc/>`_.

        The ``get()`` method of this dictionary will automatically calculate the ratio of two
        isotopes if both are present the dictionary. The ``get()`` method will return ``np.nan``
        for absent keys.

        """
        return core.RefValDict(load_refval_values('isotope_mass_AME20', keys_in_first='c'),
                               default_value=np.nan, readonly=True, ratio_function=np.divide,
                               molecule_functions='mass')

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
        return core.RefValDict({key: int(key.mass_number) for key in self.mass_W17},
                               default_value=np.nan, readonly=True, ratio_function=np.divide,
                               molecule_functions='mass')

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
        return core.RefValDict(load_refval_values('isotope_best_measurement_fraction_M16'),
                               default_value=np.nan, readonly = True, ratio_function=np.divide,
                               molecule_functions='fraction')

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
        0.0.07444

        >>> isopy.refval.isotope.initial_solar_system_fraction_L09.get('pd108/pd105')
        1.1849529780564263
        """
        return core.RefValDict(load_refval_values('isotope_initial_solar_system_fraction_L09'),
                               default_value = np.nan, readonly = True, ratio_function=np.divide,
                               molecule_functions='fraction')

    @core.cached_property
    def initial_solar_system_abundance_L09(self):
        """
        Dictionary containing the isotope abundance of the inital solar system composition (normalized to N(Si) = 10^6 atoms) from Lodders et al. (2019).

        Reference: `Lodders et al. (2019) <http://materials.springer.com/lb/docs/sm_lbs_978-3-540-88055-4_34>`_.

        The ``get()`` method of this dictionary will automatically calculate the ratio of two
        isotopes if both are present the dictionary. The ``get()`` method will return ``np.nan``
        for absent keys.

        Examples
        --------
        >>> isopy.refval.isotope.initial_solar_system_abundance_L09['pd105']
        0.3032

        >>> isopy.refval.isotope.initial_solar_system_abundance_L09.get('ge76')
        8.5

        >>> isopy.refval.isotope.initial_solar_system_abundance_L09.get('pd108/pd105')
        1.184036939313984
        """
        return core.RefValDict(load_refval_values('isotope_initial_solar_system_abundance_L09'),
                               default_value = np.nan, readonly = True, ratio_function=np.divide,
                               molecule_functions='abundance')

    @core.cached_property
    def initial_solar_system_abundance_L09b(self):
        """
        Dictionary containing the isotope abundance calcualted from the elemental abundance and the isotope fraction.

        **Note** These values are not directly taken from the table but calculated from
        *element.initial_solar_system_abundance_L09* and *isotope.initial_solar_system_fraction_L09*. This ensures
        consistency between all three reference values. Discrepancies between the calculated these isotope abundances
        and *isotope.initial_solar_system_abundance_L09* are due to rounding errors.

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
        result = {k: v * element_abundance.get(k.element_symbol) for k, v in isotope_fraction.items()}

        return core.RefValDict(result, default_value=np.nan, readonly=True, ratio_function=np.divide,
                               molecule_functions='abundance')

    @core.cached_property
    def present_solar_system_fraction_AG89(self):
        """
        Dictionary containing the isotope fraction of the present solar system composition from Anders & Grevesse 1989.

        Reference: `Anders & Grevesse (1989) <https://doi.org/10.1016/0016-7037(89)90286-X>`_.

        The ``get()`` method of this dictionary will automatically calculate the ratio of two
        isotopes if both are present the dictionary. The ``get()`` method will return ``np.nan``
        for absent keys.
        """
        return core.RefValDict(load_refval_values('isotope_present_solar_system_fraction_AG89'),
                               default_value=np.nan, readonly=True, ratio_function=np.divide,
                               molecule_functions='fraction')

    @core.cached_property
    def initial_solar_system_abundance_AG89(self):
        """
        Dictionary containing the isotope abundance of the initial solar system abundance from Anders & Grevesse 1989.

        Data normalised such that normalized to N(Si) = 10^6 atoms.

        Reference: `Anders & Grevesse (1989) <https://doi.org/10.1016/0016-7037(89)90286-X>`_.

        The ``get()`` method of this dictionary will automatically calculate the ratio of two
        isotopes if both are present the dictionary. The ``get()`` method will return ``np.nan``
        for absent keys.
        """
        return core.RefValDict(load_refval_values('isotope_initial_solar_system_abundance_AG89'),
                               default_value=np.nan, readonly=True, ratio_function=np.divide,
                               molecule_functions='abundance')

    @core.cached_property
    def present_solar_system_abundance_AG89(self):
        """
        Dictionary containing the isotope abundance of the present solar system abundance from Anders & Grevesse 1989.

        Data normalised such that normalized to N(Si) = 10^6 atoms.

        Reference: `Anders & Grevesse (1989) <https://doi.org/10.1016/0016-7037(89)90286-X>`_.

        The ``get()`` method of this dictionary will automatically calculate the ratio of two
        isotopes if both are present the dictionary. The ``get()`` method will return ``np.nan``
        for absent keys.
        """
        return core.RefValDict(load_refval_values('isotope_present_solar_system_abundance_AG89'),
                               default_value=np.nan, readonly=True, ratio_function=np.divide,
                               molecule_functions='abundance')

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
        return core.RefValDict(load_refval_values('isotope_sprocess_fraction_B11'),
                               ratio_function=np.divide, default_value=np.nan, readonly = True)

class ReferenceValues:
    """ Reference values useful for working with geochemical data"""
    def __init__(self):
        self.reset()

    def __repr__(self):
        out = f"{self.__class__.__name__}("
        space = ' ' * len(out)
        mass = f"""mass=['{"', '".join(self.mass.ls())}']"""
        element = f"""element=['{"', '".join(self.element.ls())}']"""
        isotope = f"""isotope=['{"', '".join(self.isotope.ls())}']"""

        return f'{out}{mass},\n{space}{element},\n{space}{isotope})'

    def _repr_markdown_(self):
        return f'{self.mass._repr_markdown_()}\n\n{self.element._repr_markdown_()}\n\n{self.isotope._repr_markdown_()}'

    def ls(self):
        names = []
        names += [f'mass.{n}' for n in self.mass.ls()]
        names += [f'element.{n}' for n in self.element.ls()]
        names += [f'isotope.{n}' for n in self.isotope.ls()]

        return names

    def __call__(self, path_or_dict, datatype=dict):
        if type(path_or_dict) is str:
            try:
                group, value = path_or_dict.split('.', 1)
            except ValueError:
                raise ValueError(f'Unable to parse "{path_or_dict}"')

            group = getattr(self, group, None)
            if group is None:
                raise ValueError(f'Reference value group "{group}" not found')

            path_or_dict = getattr(group, value, None)
            if path_or_dict is None:
                raise ValueError(f'Reference value "{group.__class__.__name__}.{value}" not found')

        if isinstance(path_or_dict, datatype):
            return path_or_dict
        else:
            return datatype(path_or_dict)

    def reset(self):
        """Resets the reference values to the default values"""
        self._element = element(self)
        self._isotope = isotope(self)
        self._mass = mass(self)

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
