import isopy
import pytest

# TODO test creating new unit and

# rDelta_Epsilon, rDelta_Mu
# %, ppt, ppm, ppb etc are not tied to anything

def reset_units(func):
    def test_func():
        default_units = isopy.core.ALL_UNITS.copy()
        try:
            func()
        finally:
            isopy.core.ALL_UNITS = default_units
    return test_func

@reset_units
def test_UNIT():
    eps = isopy.new_unit('Epsilon', 'ε ')

    # Only one instance with a given name can exist
    with pytest.raises(ValueError):
        isopy.new_unit('Epsilon', 'eps')

    # should be stored as lower case
    assert 'Epsilon' not in isopy.core.ALL_UNITS
    assert 'epsilon' in isopy.core.ALL_UNITS

    assert isopy.core.ALL_UNITS['epsilon'] is eps
    assert isopy.asunit('epsilon') is eps
    assert isopy.asunit('EPSILON') is eps

    # __methods__
    assert str(eps) == 'Epsilon'
    assert repr(eps) == f'Unit("Epsilon", prefix="ε ")'
    assert hash(eps) != hash(isopy.asunit('isoEPSILON'))
    assert True if eps else False

    key1 = 'ε 102Pd'
    key2 = 'e 102Pd'

    # hasunit
    assert eps.hasunit(key1) is True
    assert eps.hasunit(key2) is False

    # removeunit
    assert eps.removeunit(key1) == '102Pd'
    assert eps.removeunit(key2) == key2

    #parse
    assert eps.parse(key1) == ('102Pd', eps)
    assert eps.parse(key2) == (key2, eps)

    #parser
    assert eps.parser(key1) == ('102Pd', eps)
    with pytest.raises(ValueError):
        eps.parser(key2)

@reset_units
def test_NoneUnit():
    none_unit = isopy.core.NONE_UNIT
    assert 'none' in isopy.core.ALL_UNITS
    assert type(isopy.core.ALL_UNITS['none']) is isopy.core.NoneUnit
    assert isopy.core.ALL_UNITS['none'] is none_unit

    assert isopy.asunit(None) is none_unit
    assert isopy.asunit(isopy.core.NONE_UNIT) is none_unit
    assert isopy.asunit('none') is none_unit
    assert isopy.asunit('None') is none_unit
    assert isopy.asunit('NONE') is none_unit

    assert str(none_unit) == 'None'
    assert repr(none_unit) == 'NoneUnit("None")'
    assert True if none_unit else False is False

    key1 = 'ε 102Pd'
    key2 = '102Pd ppm'

    assert none_unit.hasunit(key1) is True
    assert none_unit.hasunit(key2) is True

    assert none_unit.removeunit(key1) == key1
    assert none_unit.removeunit(key2) == key2

    assert none_unit.parse(key1) == (key1, none_unit)
    assert none_unit.parse(key2) == (key2, none_unit)

    assert none_unit.parser(key1) == (key1, none_unit)
    assert none_unit.parser(key2) == (key2, none_unit)

@reset_units
def test_SDUnit1():
    sd_unit = isopy.core.SD_UNIT

    assert 'sd' in isopy.core.ALL_UNITS
    assert type(isopy.core.ALL_UNITS['sd']) is isopy.core.SDUnit
    assert isopy.core.ALL_UNITS['sd'] is sd_unit

    assert isopy.asunit(sd_unit) is sd_unit
    assert isopy.asunit('sd') is sd_unit
    assert isopy.asunit('Sd') is sd_unit
    assert isopy.asunit('SD') is sd_unit

    assert str(sd_unit) == 'SD'
    assert repr(sd_unit) == 'SDUnit("SD", suffix=" SD")'
    assert True if sd_unit else False

    key1 = '102Pd SD'
    key2 = 'ε 102Pd'
    key3 = 'ε 102Pd SD'

    assert sd_unit.hasunit(key1) is True
    assert sd_unit.hasunit(key2) is False
    assert sd_unit.hasunit(key3) is True

    assert sd_unit.removeunit(key1) == '102Pd'
    assert sd_unit.removeunit(key2) == key2
    assert sd_unit.removeunit(key3) == key2

    assert sd_unit.parse(key1) == ('102Pd', sd_unit)
    assert sd_unit.parse(key2) == (key2, sd_unit)
    assert sd_unit.parse(key3) == (key2, sd_unit)

    assert sd_unit.parser(key1) == ('102Pd', sd_unit)
    with pytest.raises(ValueError):
        sd_unit.parser(key2)
    assert sd_unit.parser(key3) == (key2, sd_unit)

    eps_unit = isopy.toolbox.isotope.ISOEPSILON
    epssd_unit = eps_unit & sd_unit

    with pytest.raises(TypeError):
        sd_unit & eps_unit
    with pytest.raises(TypeError):
        eps_unit & epssd_unit

    assert isopy.asunit('isoEPSILON&SD')

    assert str(epssd_unit) == 'isoEPSILON&SD'
    assert repr(epssd_unit) == 'SDUnit("SD", primary_unit="isoEPSILON", suffix=" SD")'
    assert True if epssd_unit else False

    key1 = '102Pd SD'
    key2 = 'ε 102Pd'
    key3 = 'ε 102Pd SD'

    assert epssd_unit.hasunit(key1) is False
    assert epssd_unit.hasunit(key2) is False
    assert epssd_unit.hasunit(key3) is True

    assert epssd_unit.removeunit(key1) == key1
    assert epssd_unit.removeunit(key2) == key2
    assert epssd_unit.removeunit(key3) == '102Pd'

    assert epssd_unit.parse(key1) == (key1, epssd_unit)
    assert epssd_unit.parse(key2) == (key2, epssd_unit)
    assert epssd_unit.parse(key3) == ('102Pd', epssd_unit)

    with pytest.raises(ValueError):
        epssd_unit.parser(key1)
    with pytest.raises(ValueError):
        epssd_unit.parser(key2)
    assert epssd_unit.parser(key3) == ('102Pd', epssd_unit)

@reset_units
def test_SDUnit2():
    eps_unit = isopy.toolbox.isotope.ISOEPSILON
    sd_unit = isopy.core.SD_UNIT

    # These do not exist yet
    with pytest.raises(ValueError):
        isopy.asunit(f'2SD')
    with pytest.raises(ValueError):
        isopy.asunit(f'1.96SD')
    with pytest.raises(ValueError):
        isopy.asunit(f'95CISD')
    with pytest.raises(ValueError):
        isopy.asunit(f'95.5CISD')

    for zscore, ci in [('2', '95'), ('1.96', '95.5')]:
        sd2 = sd_unit.new(zscore=float(zscore))
        sd95 = sd_unit.new(ci=float(ci)/100)
        epssd2 = sd_unit.new(primary_unit=eps_unit, zscore=float(zscore))
        epssd95 = sd_unit.new(primary_unit=eps_unit, ci=float(ci)/100)

        assert str(sd2) == f'{zscore}SD'
        assert str(sd95) == f'{ci}CISD'
        assert str(epssd2) == f'isoEPSILON&{zscore}SD'
        assert str(epssd95) == f'isoEPSILON&{ci}CISD'

        assert repr(sd2) == f'SDUnit("SD", zscore={zscore}, suffix=" {zscore}SD")'
        assert repr(sd95) == f'SDUnit("SD", ci={float(ci)/100}, suffix=" {ci}% CI SD")'
        assert repr(epssd2) == f'SDUnit("SD", primary_unit="isoEPSILON", zscore={zscore}, suffix=" {zscore}SD")'
        assert repr(epssd95) == f'SDUnit("SD", primary_unit="isoEPSILON", ci={float(ci)/100}, suffix=" {ci}% CI SD")'

        assert sd2 is isopy.asunit(f'{zscore}SD')
        assert sd95 is isopy.asunit(f'{ci}CISD')
        assert epssd2 is eps_unit & sd2
        assert epssd95 is eps_unit & sd95
        assert epssd2 is isopy.asunit(f'isoEPSILON&{zscore}SD')
        assert epssd95 is isopy.asunit(f'isoEPSILON&{ci}CISD')

        assert hash(sd2) != hash(sd_unit)
        assert hash(sd2) != hash(epssd2)
        assert hash(epssd2) != hash(sd_unit)

        assert hash(sd95) != hash(sd_unit)
        assert hash(sd95) != hash(epssd95)
        assert hash(epssd95) != hash(sd_unit)

        assert hash(sd2) != hash(sd95)
        assert hash(epssd2) != hash(epssd95)

        key1 = f'102Pd {zscore}SD'
        key2 = f'ε 102Pd {zscore}SD'

        assert sd_unit.hasunit(key1) is True
        assert sd_unit.hasunit(key2) is True
        assert sd2.hasunit(key1) is True
        assert sd2.hasunit(key2) is True
        assert sd95.hasunit(key1) is False
        assert sd95.hasunit(key2) is False
        assert epssd2.hasunit(key1) is False
        assert epssd2.hasunit(key2) is True
        assert epssd95.hasunit(key1) is False
        assert epssd95.hasunit(key2) is False

        assert sd_unit.removeunit(key1) == '102Pd'
        assert sd_unit.removeunit(key2) == 'ε 102Pd'
        assert sd2.removeunit(key1) == '102Pd'
        assert sd2.removeunit(key2) == 'ε 102Pd'
        assert sd95.removeunit(key1) == key1
        assert sd95.removeunit(key2) == key2
        assert epssd2.removeunit(key1) == key1
        assert epssd2.removeunit(key2) == '102Pd'
        assert epssd95.removeunit(key1) == key1
        assert epssd95.removeunit(key2) == key2

        assert sd_unit.parse(key1) == ('102Pd', sd2)
        assert sd_unit.parse(key2) == ('ε 102Pd', sd2)
        assert sd2.parse(key1) == ('102Pd', sd2)
        assert sd2.parse(key2) == ('ε 102Pd', sd2)
        assert sd95.parse(key1) == (key1, sd95)
        assert sd95.parse(key2) == (key2, sd95)
        assert epssd2.parse(key1) == (key1, epssd2)
        assert epssd2.parse(key2) == ('102Pd', epssd2)
        assert epssd95.parse(key1) == (key1, epssd95)
        assert epssd95.parse(key2) == (key2, epssd95)

        assert sd_unit.parser(key1) == ('102Pd', sd2)
        assert sd_unit.parser(key2) == ('ε 102Pd', sd2)
        assert sd2.parser(key1) ==('102Pd', sd2)
        assert sd2.parser(key2) == ('ε 102Pd', sd2)
        with pytest.raises(ValueError):
            sd95.parser(key1)
        with pytest.raises(ValueError):
            sd95.parser(key2)
        with pytest.raises(ValueError):
            epssd2.parser(key1)
        assert epssd2.parser(key2) == ('102Pd', epssd2)
        with pytest.raises(ValueError):
            epssd95.parser(key1)
        with pytest.raises(ValueError):
            epssd95.parser(key2)

        key1 = f'102Pd {ci}% CI SD'
        key2 = f'ε 102Pd {ci}% CI SD'

        assert sd_unit.hasunit(key1) is True
        assert sd_unit.hasunit(key2) is True
        assert sd2.hasunit(key1) is False
        assert sd2.hasunit(key2) is False
        assert sd95.hasunit(key1) is True
        assert sd95.hasunit(key2) is True
        assert epssd2.hasunit(key1) is False
        assert epssd2.hasunit(key2) is False
        assert epssd95.hasunit(key1) is False
        assert epssd95.hasunit(key2) is True

        assert sd_unit.removeunit(key1) == '102Pd'
        assert sd_unit.removeunit(key2) == 'ε 102Pd'
        assert sd2.removeunit(key1) == key1
        assert sd2.removeunit(key2) == key2
        assert sd95.removeunit(key1) == '102Pd'
        assert sd95.removeunit(key2) == 'ε 102Pd'
        assert epssd2.removeunit(key1) == key1
        assert epssd2.removeunit(key2) == key2
        assert epssd95.removeunit(key1) == key1
        assert epssd95.removeunit(key2) == '102Pd'

        assert sd_unit.parse(key1) == ('102Pd', sd95)
        assert sd_unit.parse(key2) == ('ε 102Pd', sd95)
        assert sd2.parse(key1) == (key1, sd2)
        assert sd2.parse(key2) == (key2, sd2)
        assert sd95.parse(key1) == ('102Pd', sd95)
        assert sd95.parse(key2) == ('ε 102Pd', sd95)
        assert epssd2.parse(key1) == (key1, epssd2)
        assert epssd2.parse(key2) == (key2, epssd2)
        assert epssd95.parse(key1) == (key1, epssd95)
        assert epssd95.parse(key2) == ('102Pd', epssd95)

        assert sd_unit.parser(key1) == ('102Pd', sd95)
        assert sd_unit.parser(key2) == ('ε 102Pd', sd95)
        with pytest.raises(ValueError):
            sd2.parser(key1)
        with pytest.raises(ValueError):
            sd2.parser(key2)
        assert sd95.parser(key1) == ('102Pd', sd95)
        assert sd95.parser(key2) == ('ε 102Pd', sd95)
        with pytest.raises(ValueError):
            epssd2.parser(key1)
        with pytest.raises(ValueError):
            epssd2.parser(key2)
        with pytest.raises(ValueError):
            epssd95.parser(key1)
        assert epssd95.parse(key2) == ('102Pd', epssd95)

@reset_units
def test_PartPerUnit():
    pp_unit = isopy.core.PPM_UNIT

    assert 'ppm' in isopy.core.ALL_UNITS
    assert type(isopy.core.ALL_UNITS['ppm']) is isopy.core.PartsPerUnit
    assert isopy.core.ALL_UNITS['ppm'] is pp_unit

    assert isopy.asunit(pp_unit) is pp_unit
    assert isopy.asunit('ppm') is pp_unit
    assert isopy.asunit('ppm') is pp_unit
    assert isopy.asunit('ppm') is pp_unit

    assert str(pp_unit) == 'PPM'
    assert repr(pp_unit) == 'PartsPerUnit("PPM", parts_per=1E+06, suffix=" ppm")'
    assert True if pp_unit else False

    key1 = '102Pd ppm'
    key2 = 'ε 102Pd'
    key3 = 'ε 102Pd ppm'

    assert pp_unit.hasunit(key1) is True
    assert pp_unit.hasunit(key2) is False
    assert pp_unit.hasunit(key3) is True

    assert pp_unit.removeunit(key1) == '102Pd'
    assert pp_unit.removeunit(key2) == key2
    assert pp_unit.removeunit(key3) == key2

    assert pp_unit.parse(key1) == ('102Pd', pp_unit)
    assert pp_unit.parse(key2) == (key2, pp_unit)
    assert pp_unit.parse(key3) == (key2, pp_unit)

    assert pp_unit.parser(key1) == ('102Pd', pp_unit)
    with pytest.raises(ValueError):
        pp_unit.parser(key2)
    assert pp_unit.parser(key3) == (key2, pp_unit)

    eps_unit = isopy.toolbox.isotope.ISOEPSILON
    epspp_unit = eps_unit & pp_unit

    with pytest.raises(TypeError):
        pp_unit & eps_unit
    with pytest.raises(TypeError):
        eps_unit & epspp_unit

    assert isopy.asunit('isoEPSILON&PPM')

    assert str(epspp_unit) == 'isoEPSILON&PPM'
    assert repr(epspp_unit) == 'PartsPerUnit("PPM", parts_per=1E+06, primary_unit="isoEPSILON", suffix=" ppm")'
    assert True if epspp_unit else False

    key1 = '102Pd ppm'
    key2 = 'ε 102Pd'
    key3 = 'ε 102Pd ppm'

    assert epspp_unit.hasunit(key1) is False
    assert epspp_unit.hasunit(key2) is False
    assert epspp_unit.hasunit(key3) is True

    assert epspp_unit.removeunit(key1) == key1
    assert epspp_unit.removeunit(key2) == key2
    assert epspp_unit.removeunit(key3) == '102Pd'

    assert epspp_unit.parse(key1) == (key1, epspp_unit)
    assert epspp_unit.parse(key2) == (key2, epspp_unit)
    assert epspp_unit.parse(key3) == ('102Pd', epspp_unit)

    with pytest.raises(ValueError):
        epspp_unit.parser(key1)
    with pytest.raises(ValueError):
        epspp_unit.parser(key2)
    assert epspp_unit.parser(key3) == ('102Pd', epspp_unit)

@reset_units
def test_UnitGroup():
    eps_unit = isopy.asunit('isoEPSILON')
    mu_unit = isopy.asunit('isoMU')
    delta_unit = isopy.asunit('isoDELTA')
    sd_unit = isopy.asunit('sd')

    units1 = eps_unit | mu_unit | delta_unit
    units2 = isopy.asunit('isodelta|isomu!|isoepsilon')
    assert units1 != units2

    assert units1.name == 'isoEPSILON!|isoMU|isoDELTA'
    assert units2.name == 'isoDELTA|isoMU!|isoEPSILON'
    assert str(units1) == 'isoEPSILON!|isoMU|isoDELTA'
    assert str(units2) == 'isoDELTA|isoMU!|isoEPSILON'
    assert repr(units1) == 'UnitGroup("isoEPSILON", "isoMU", "isoDELTA", default_unit="isoEPSILON")'
    assert repr(units2) == 'UnitGroup("isoDELTA", "isoMU", "isoEPSILON", default_unit="isoMU")'

    assert len(units1) == 3
    assert len(units2) == 3
    assert units1.units == (eps_unit, mu_unit, delta_unit)
    assert units2.units == (delta_unit, mu_unit, eps_unit)

    for unit in (eps_unit, mu_unit, delta_unit):
        assert unit in units1
        assert unit in units2
    assert sd_unit not in units1
    assert sd_unit not in units2

    assert units1.default_unit is eps_unit
    assert units2.default_unit is mu_unit

@reset_units
def test_keystring():
    pre = isopy.asunit('isoEPSILON')
    suf = isopy.asunit('PPM')
    both = isopy.asunit('isoEPSILON&PPM')

    # unit, flavour, basekey, alt_prefix, alt_suffix
    tests = [(pre, 'isotope', '102Pd', 'ε ', ''),
             (suf, 'isotope', '102Pd', '', ' ppm'),
             (both, 'isotope', '102Pd', 'ε ', ' ppm')]
    for keystringfunc in [isopy.keystring, isopy.askeystring]:
        for unit, flavour, basekey, alt_prefix, alt_suffix in tests:
            unit_parser = unit.parser
            prefix = unit.prefix
            suffix = unit.suffix
            unitkey = f"{prefix}{basekey}{suffix}"
            altkey = f"{alt_prefix}{basekey}{alt_suffix}"

            # Key without unit
            keystring = keystringfunc(basekey)
            assert keystring.flavour == flavour
            assert keystring.unit is isopy.core.NONE_UNIT
            assert str(keystring) == basekey
            assert str(keystring.base) == basekey
            assert keystring.base is keystring

            keystring = keystringfunc(basekey, keyparser=unit)
            assert keystring.flavour == flavour
            assert keystring.unit is isopy.core.NONE_UNIT
            assert str(keystring) == basekey
            assert str(keystring.base) == basekey
            assert keystring.base is keystring

            keystring = keystringfunc(basekey, unit=unit)
            assert keystring.flavour == flavour
            assert keystring.unit is unit
            assert str(keystring) == unitkey
            assert str(keystring.base) == basekey
            assert keystring.base is not keystring

            # Key with unit
            keystring = keystringfunc(unitkey)
            assert not keystring.unit
            assert keystring.flavour == 'general'
            assert str(keystring) == unitkey
            assert str(keystring.base) == unitkey
            assert keystring.base is keystring

            keystring = keystringfunc(unitkey, keyparser=unit)
            assert keystring.unit is isopy.core.NONE_UNIT
            assert keystring.flavour == flavour
            assert str(keystring) == basekey
            assert str(keystring.base) == basekey
            assert keystring.base is keystring

            keystring = keystringfunc(unitkey, unit=unit)
            assert keystring.unit is unit
            assert keystring.flavour == flavour
            assert str(keystring) == unitkey
            assert str(keystring.base) == basekey
            assert keystring.base is not keystring

            # Key with alt unit
            keystring = keystringfunc(altkey, unit=unit)
            assert keystring.unit is unit
            assert keystring.flavour == 'general'
            assert str(keystring) == f'{prefix}{altkey}{suffix}'
            assert str(keystring.base) == altkey
            assert keystring.base is not keystring


