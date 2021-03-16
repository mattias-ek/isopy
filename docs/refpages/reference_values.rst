Reference values
================
A number of reference values are avaliable through ``isopy.refval`` that
are useful for geochemists. All the reference values are readonly to avoid
corrupting the values but dictionaries can be edited by creating a copy.

Most of the reference values are
:class:`IsopyDict` or :class:`ScalarDict` dictionaries that use
isopy key string as *keys* and therfore will automatically format
*key* strings.

Below is a list of the different reference values avaliable.

.. currentmodule:: isopy.reference_values

refval.mass
-----------
A collection of reference values relating to the mass number.

.. autosummary::
    :toctree: refval_mass
    :template: ref_attr.rst

    mass.isotopes

refval.element
--------------
A collection of reference values relating to the element symbol.

.. autosummary::
    :toctree: refval_element
    :template: ref_attr.rst

    element.isotopes
    element.all_symbols
    element.symbol_name
    element.name_symbol
    element.atomic_number
    element.atomic_weight

refval.isotope
--------------
A collection of reference values relating to individual isotopes.

.. autosummary::
    :toctree: refval_isotope
    :template: ref_attr.rst

    isotope.mass
    isotope.abundance
    isotope.mass_W17
    isotope.best_abundance_measurement_M16
    isotope.initial_solar_system_abundance_L09
    isotope.initial_solar_system_absolute_abundance_L09
    isotope.sprocess_abundance_B11





