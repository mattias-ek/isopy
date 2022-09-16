.. _refvaldoc:

Reference Values
================

.. currentmodule:: isopy.reference_values

A number of reference values are avaliable through ``isopy.refval`` that are useful for geochemists.

All the reference values are readonly but dictionaries can be edited by creating a copy.

It is also possible to fetch reference values using :py:func:`asrefval` by passing a string with the
name of the reference value as the first argument e.g. ``isopy.asrefval('isotope.mass')``.

refval.mass
-----------
isotopes
++++++++
.. autoattribute:: mass.isotopes

refval.element
--------------

isotopes
++++++++
.. autoattribute:: element.isotopes

all_symbols
+++++++++++
.. autoattribute:: element.all_symbols

symbol_name
+++++++++++
.. autoattribute:: element.symbol_name

name_symbol
+++++++++++
.. autoattribute:: element.name_symbol

atomic_number
+++++++++++++
.. autoattribute:: element.atomic_number

atomic_weight
+++++++++++++
.. autoattribute:: element.atomic_weight

initial_solar_system_abundance_L09
++++++++++++++++++++++++++++++++++
.. autoattribute:: element.initial_solar_system_abundance_L09

refval.element
--------------

isotopes
++++++++
.. autoattribute:: isotope.mass

fraction
++++++++
.. autoattribute:: isotope.fraction

mass_number
++++++++
.. autoattribute:: isotope.mass_number

mass_W17
++++++++
.. autoattribute:: isotope.mass_W17

mass_AME20
++++++++++
.. autoattribute:: isotope.mass_AME20

best_measurement_fraction_M16
+++++++++++++++++++++++++++++
.. autoattribute:: isotope.best_measurement_fraction_M16

initial_solar_system_fraction_L09
+++++++++++++++++++++++++++++++++
.. autoattribute:: isotope.initial_solar_system_fraction_L09

initial_solar_system_abundance_L09
++++++++++++++++++++++++++++++++++
.. autoattribute:: isotope.initial_solar_system_abundance_L09

initial_solar_system_abundance_L09b
+++++++++++++++++++++++++++++++++++
.. autoattribute:: isotope.initial_solar_system_abundance_L09b

present_solar_system_fraction_AG89
++++++++++++++++++++++++++++++++++
.. autoattribute:: isotope.present_solar_system_fraction_AG89

initial_solar_system_abundance_AG89
+++++++++++++++++++++++++++++++++++
.. autoattribute:: isotope.initial_solar_system_abundance_AG89

present_solar_system_abundance_AG89
+++++++++++++++++++++++++++++++++++
.. autoattribute:: isotope.present_solar_system_abundance_AG89

sprocess_fraction_B11
+++++++++++++++++++++
.. autoattribute:: isotope.sprocess_fraction_B11