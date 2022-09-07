Reference values
================
A number of reference values are avaliable through ``isopy.refval`` that
are useful for geochemists. All the reference values are readonly to avoid
corrupting the values but dictionaries can be edited by creating a copy.

Below is a list of the different reference values available.

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
    element.initial_solar_system_abundance_L09

refval.isotope
--------------
A collection of reference values relating to individual isotopes.

.. autosummary::
    :toctree: refval_isotope
    :template: ref_attr.rst

    isotope.mass
    isotope.fraction
    isotope.mass_number
    isotope.mass_W17
    isotope.mass_AME20
    isotope.best_measurement_fraction_M16
    isotope.initial_solar_system_fraction_L09
    isotope.initial_solar_system_abundance_L09
    isotope.initial_solar_system_abundance_L09b
    isotope.present_solar_system_fraction_AG89
    isotope.initial_solar_system_abundance_AG89
    isotope.present_solar_system_abundance_AG89
    isotope.sprocess_fraction_B11





