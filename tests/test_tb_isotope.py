import isopy
import numpy as np
import pytest

# calculate_mass_fractionation_factor, remove_mass_fractionation, add_mass_fractionation
def test_mass_fractionation1():
    # Testing with input as isotope array
    # Using default reference values

    mass_ref = isopy.refval.isotope.mass_W17
    fraction_ref = isopy.refval.isotope.best_measurement_fraction_M16

    unfractionated = isopy.random(100, (1, 0.001), keys=isopy.refval.element.isotopes['pd'], seed = 46)
    unfractionated = unfractionated * fraction_ref
    unfractionated['108pd'] = fraction_ref.get('108pd/105pd') * unfractionated['105pd']

    mf_factor = isopy.random(100, (0, 2), seed=47)

    c_fractionated1 = isopy.tb.add_mass_fractionation(unfractionated, mf_factor, '105pd')
    c_fractionated2 = isopy.tb.add_mass_fractionation(unfractionated, mf_factor)
    assert c_fractionated1.keys == unfractionated.keys
    assert c_fractionated1.size == unfractionated.size
    assert c_fractionated2.keys == unfractionated.keys
    assert c_fractionated2.size == unfractionated.size

    c_unfractionated1 = isopy.tb.remove_mass_fractionation(c_fractionated1, mf_factor, '105pd')
    c_unfractionated2 = isopy.tb.remove_mass_fractionation(c_fractionated2, mf_factor)
    assert c_unfractionated1.keys == unfractionated.keys
    assert c_unfractionated1.size == unfractionated.size
    assert c_unfractionated2.keys == unfractionated.keys
    assert c_unfractionated2.size == unfractionated.size

    c_mf_factor2 = isopy.tb.calculate_mass_fractionation_factor(c_fractionated1, '108pd/105pd')
    np.testing.assert_allclose(c_mf_factor2, mf_factor)

    for key in unfractionated.keys:
        mass_diff = mass_ref.get(key/'105pd')
        fractionated = unfractionated[key] * (mass_diff ** mf_factor)
        np.testing.assert_allclose(c_fractionated1[key], fractionated)
        np.testing.assert_allclose(c_unfractionated1[key], unfractionated[key])
        np.testing.assert_allclose(c_unfractionated2[key], unfractionated[key])

    #Changing reference values
    mass_ref = isopy.refval.isotope.mass_number
    fraction_ref = isopy.refval.isotope.initial_solar_system_fraction_L09

    unfractionated = isopy.random(100, (1, 0.001), keys=isopy.refval.element.isotopes['pd'],
                                   seed=46)
    unfractionated = unfractionated * fraction_ref
    unfractionated['108pd'] = fraction_ref.get('108pd/105pd') * unfractionated['105pd']
    unfractionated2 = unfractionated.ratio('105pd')

    mf_factor = isopy.random(100, (0, 2), seed=47)

    c_fractionated1 = isopy.tb.add_mass_fractionation(unfractionated, mf_factor, '105pd', isotope_masses=mass_ref)
    c_fractionated2 = isopy.tb.add_mass_fractionation(unfractionated, mf_factor, isotope_masses=mass_ref)
    assert c_fractionated1.keys == unfractionated.keys
    assert c_fractionated1.size == unfractionated.size
    assert c_fractionated2.keys == unfractionated.keys
    assert c_fractionated2.size == unfractionated.size

    c_unfractionated1 = isopy.tb.remove_mass_fractionation(c_fractionated1, mf_factor, '105pd', isotope_masses=mass_ref)
    c_unfractionated2 = isopy.tb.remove_mass_fractionation(c_fractionated2, mf_factor, isotope_masses=mass_ref)
    assert c_unfractionated1.keys == unfractionated.keys
    assert c_unfractionated1.size == unfractionated.size
    assert c_unfractionated2.keys == unfractionated.keys
    assert c_unfractionated2.size == unfractionated.size

    c_mf_factor2 = isopy.tb.calculate_mass_fractionation_factor(c_fractionated1, '108pd/105pd',
                                                                isotope_masses=mass_ref, isotope_fractions=fraction_ref)
    np.testing.assert_allclose(c_mf_factor2, mf_factor)

    for key in unfractionated.keys:
        mass_diff = mass_ref.get(key / '105pd')
        fractionated = unfractionated[key] * (mass_diff ** mf_factor)
        np.testing.assert_allclose(c_fractionated1[key], fractionated)
        np.testing.assert_allclose(c_unfractionated1[key], unfractionated[key])
        np.testing.assert_allclose(c_unfractionated2[key], unfractionated[key])

# calculate_mass_fractionation_factor, remove_mass_fractionation, add_mass_fractionation
def test_mass_fractionation2():
    # Testing with input as ratio array
    # Using default reference values

    mass_ref = isopy.refval.isotope.mass_W17
    fraction_ref = isopy.refval.isotope.best_measurement_fraction_M16

    unfractionated = isopy.random(100, (1, 0.001), keys=isopy.refval.element.isotopes['pd'],
                                   seed=46)
    unfractionated = unfractionated * fraction_ref
    unfractionated['108pd'] = fraction_ref.get('108pd/105pd') * unfractionated['105pd']
    unfractionated = unfractionated.ratio('105pd')

    mf_factor = isopy.random(100, (0, 2), seed=47)

    c_fractionated2 = isopy.tb.add_mass_fractionation(unfractionated, mf_factor)
    assert c_fractionated2.keys == unfractionated.keys
    assert c_fractionated2.size == unfractionated.size

    c_unfractionated2 = isopy.tb.remove_mass_fractionation(c_fractionated2, mf_factor)
    assert c_unfractionated2.keys == unfractionated.keys
    assert c_unfractionated2.size == unfractionated.size

    c_mf_factor2 = isopy.tb.calculate_mass_fractionation_factor(c_fractionated2, '108pd/105pd')
    np.testing.assert_allclose(c_mf_factor2, mf_factor)

    for key in unfractionated.keys:
        mass_diff = mass_ref.get(key)
        fractionated = unfractionated[key] * (mass_diff ** mf_factor)
        np.testing.assert_allclose(c_fractionated2[key], fractionated)
        np.testing.assert_allclose(c_unfractionated2[key], unfractionated[key])

    # Changing reference values
    mass_ref = isopy.refval.isotope.mass_number
    fraction_ref = isopy.refval.isotope.initial_solar_system_fraction_L09

    unfractionated = isopy.random(100, (1, 0.001), keys=isopy.refval.element.isotopes['pd'],
                                   seed=46)
    unfractionated = unfractionated * fraction_ref
    unfractionated['108pd'] = fraction_ref.get('108pd/105pd') * unfractionated['105pd']
    unfractionated = unfractionated.ratio('105pd')

    mf_factor = isopy.random(100, (0, 2), seed=47)

    c_fractionated2 = isopy.tb.add_mass_fractionation(unfractionated, mf_factor, isotope_masses=mass_ref)
    assert c_fractionated2.keys == unfractionated.keys
    assert c_fractionated2.size == unfractionated.size

    c_unfractionated2 = isopy.tb.remove_mass_fractionation(c_fractionated2, mf_factor, isotope_masses=mass_ref)
    assert c_unfractionated2.keys == unfractionated.keys
    assert c_unfractionated2.size == unfractionated.size

    c_mf_factor2 = isopy.tb.calculate_mass_fractionation_factor(c_fractionated2, '108pd/105pd',
                                                                isotope_masses=mass_ref, isotope_fractions=fraction_ref)
    np.testing.assert_allclose(c_mf_factor2, mf_factor)

    for key in unfractionated.keys:
        mass_diff = mass_ref.get(key)
        fractionated = unfractionated[key] * (mass_diff ** mf_factor)
        np.testing.assert_allclose(c_fractionated2[key], fractionated)
        np.testing.assert_allclose(c_unfractionated2[key], unfractionated[key])


class Test_MassIndependentCorrection:
    def test_one(self):
        # Default reference values
        mass_ref = isopy.refval.isotope.mass_W17
        fraction_ref = isopy.refval.isotope.best_measurement_fraction_M16

        unfractionated1 = isopy.random(100, (1, 0.001), keys=isopy.refval.element.isotopes['pd'],
                                       seed=46)
        unfractionated1 = unfractionated1 * fraction_ref
        unfractionated1['108pd'] = fraction_ref.get('108pd/105pd') * unfractionated1['105pd']
        unfractionated2 = unfractionated1.ratio('105pd')

        n_unfractionated2 = (unfractionated2 / fraction_ref - 1) * 10000

        mf_factor = isopy.random(100, (0, 2), seed=47)
        fractionated1 = isopy.tb.add_mass_fractionation(unfractionated2, mf_factor)
        fractionated2 = fractionated1.deratio(unfractionated1['105pd'])

        self.run(fractionated1, unfractionated2, '108pd/105pd')
        self.run(fractionated2, unfractionated2, '108pd/105pd')

        self.run(fractionated1, n_unfractionated2, '108pd/105pd', factor=10_000)
        self.run(fractionated2, n_unfractionated2, '108pd/105pd', factor=10_000)

        self.run(fractionated1, n_unfractionated2, '108pd/105pd', factor='epsilon')
        self.run(fractionated2, n_unfractionated2, '108pd/105pd', factor='epsilon')


        # Different reference values

        mass_ref = isopy.refval.isotope.mass_number
        fraction_ref = isopy.refval.isotope.initial_solar_system_fraction_L09

        unfractionated1 = isopy.random(100, (1, 0.001), keys=isopy.refval.element.isotopes['pd'],
                                       seed=46)
        unfractionated1 = unfractionated1 * fraction_ref
        unfractionated1['108pd'] = fraction_ref.get('108pd/105pd') * unfractionated1['105pd']
        unfractionated2 = unfractionated1.ratio('105pd')
        n_unfractionated2 = (unfractionated2 / fraction_ref - 1) * 10000

        mf_factor = isopy.random(100, (0, 2), seed=47)
        fractionated1 = isopy.tb.add_mass_fractionation(unfractionated2, mf_factor,
                                                        isotope_masses=mass_ref)
        fractionated2 = fractionated1.deratio(unfractionated1['105pd'])

        self.run(fractionated1, unfractionated2, '108pd/105pd', mass_ref=mass_ref, fraction_ref=fraction_ref)
        self.run(fractionated2, unfractionated2, '108pd/105pd', mass_ref=mass_ref, fraction_ref=fraction_ref)

        self.run(fractionated1, n_unfractionated2, '108pd/105pd', factor=10_000, mass_ref=mass_ref, fraction_ref=fraction_ref)
        self.run(fractionated2, n_unfractionated2, '108pd/105pd', factor=10_000, mass_ref=mass_ref, fraction_ref=fraction_ref)

        self.run(fractionated1, n_unfractionated2, '108pd/105pd', factor='epsilon', mass_ref=mass_ref, fraction_ref=fraction_ref)
        self.run(fractionated2, n_unfractionated2, '108pd/105pd', factor='epsilon', mass_ref=mass_ref, fraction_ref=fraction_ref)

    def test_two(self):
        # With interference correctionn
        # We wont get an exact match here so we have to lower the tolerance.

        # Default reference values
        mass_ref = isopy.refval.isotope.mass_W17
        fraction_ref = isopy.refval.isotope.best_measurement_fraction_M16

        mf_factor = isopy.random(100, (0, 2), seed=47)
        data = isopy.random(100, (1, 0.1), keys='101ru 102pd 104pd 105pd 106pd 108pd 110pd 111cd'.split(), seed=46)
        data = data * fraction_ref
        data['108pd'] = fraction_ref.get('108pd/105pd') * data['105pd']

        fractionated = data.copy()
        fractionated = isopy.tb.add_mass_fractionation(fractionated, mf_factor)

        for key in fractionated.keys.filter(element_symbol='pd'):
            if (ru:=fraction_ref.get(f'ru{key.mass_number}/ru101', 0)) > 0:
                ru *= fractionated['101ru'] * (mass_ref.get(f'ru{key.mass_number}/ru101', 0) ** mf_factor)
                fractionated[key] += ru

            if (cd:=fraction_ref.get(f'cd{key.mass_number}/cd111', 0)) > 0:
                cd *= fractionated['111cd'] * (mass_ref.get(f'cd{key.mass_number}/cd111', 0) ** mf_factor)
                fractionated[key] += cd

        correct1 = data.copy(element_symbol = 'pd').ratio('105pd')
        correct2 = (correct1 / fraction_ref - 1)
        correct3 = (correct1 / fraction_ref - 1) * 10_000

        self.run(fractionated, correct1, '108pd/105pd')
        self.run(fractionated, correct2, '108pd/105pd', factor=1)
        self.run(fractionated, correct3, '108pd/105pd', factor=10_000)
        self.run(fractionated, correct3, '108pd/105pd', factor='epsilon')

        # Different reference values
        mass_ref = isopy.refval.isotope.mass_number
        fraction_ref = isopy.refval.isotope.initial_solar_system_fraction_L09

        mf_factor = isopy.random(100, (0, 2), seed=47)
        data = isopy.random(100, (1, 0.1), keys='101ru 102pd 104pd 105pd 106pd 108pd 110pd 111cd'.split(), seed=46)
        data = data * fraction_ref
        data['108pd'] = fraction_ref.get('108pd/105pd') * data['105pd']

        fractionated = data.copy()
        fractionated = isopy.tb.add_mass_fractionation(fractionated, mf_factor, isotope_masses=mass_ref)

        for key in fractionated.keys.filter(element_symbol='pd'):
            if (ru := fraction_ref.get(f'ru{key.mass_number}/ru101', 0)) > 0:
                ru *= fractionated['101ru'] * (
                            mass_ref.get(f'ru{key.mass_number}/ru101', 0) ** mf_factor)
                fractionated[key] += ru

            if (cd := fraction_ref.get(f'cd{key.mass_number}/cd111', 0)) > 0:
                cd *= fractionated['111cd'] * (
                            mass_ref.get(f'cd{key.mass_number}/cd111', 0) ** mf_factor)
                fractionated[key] += cd

        correct1 = data.copy(element_symbol='pd').ratio('105pd')
        correct2 = (correct1 / fraction_ref - 1)
        correct3 = (correct1 / fraction_ref - 1) * 10_000

        self.run(fractionated, correct1, '108pd/105pd', mass_ref=mass_ref, fraction_ref=fraction_ref)
        self.run(fractionated, correct2, '108pd/105pd', factor=1, mass_ref=mass_ref, fraction_ref=fraction_ref)
        self.run(fractionated, correct3, '108pd/105pd', factor=10_000, mass_ref=mass_ref, fraction_ref=fraction_ref)
        self.run(fractionated, correct3, '108pd/105pd', factor='epsilon', mass_ref=mass_ref, fraction_ref=fraction_ref)

    def test_three(self):
        # Normalisations

        # Default reference values
        mass_ref = isopy.refval.isotope.mass_W17
        fraction_ref = isopy.refval.isotope.best_measurement_fraction_M16

        mf_factor = isopy.random(100, (0, 2), seed=47)
        data = isopy.random(100, (1, 0.1), keys='102pd 104pd 105pd 106pd 108pd 110pd'.split(), seed=46)
        data = data * fraction_ref
        data['108pd'] = fraction_ref.get('108pd/105pd') * data['105pd']

        fractionated = data.copy()
        fractionated = isopy.tb.add_mass_fractionation(fractionated, mf_factor)

        correct1 = data.copy(element_symbol='pd').ratio('105pd')
        correct2 = (correct1 / fraction_ref - 1)
        correct3 = correct2 * 1000
        correct4 = correct2 * 10_000
        correct5 = correct2 * 1_000_000

        self.run(fractionated, correct1, '108pd/105pd')
        self.run(fractionated, correct2, '108pd/105pd', factor=1)
        self.run(fractionated, correct3, '108pd/105pd', factor=1000)
        self.run(fractionated, correct3, '108pd/105pd', factor='ppt')
        self.run(fractionated, correct3, '108pd/105pd', factor='permil')
        self.run(fractionated, correct4, '108pd/105pd', factor=10_000)
        self.run(fractionated, correct4, '108pd/105pd', factor='epsilon')
        self.run(fractionated, correct5, '108pd/105pd', factor=1_000_000)
        self.run(fractionated, correct5, '108pd/105pd', factor='mu')
        self.run(fractionated, correct5, '108pd/105pd', factor='ppm')

        # Single value
        std1 = isopy.random(100, (1, 0.1), keys='102pd 104pd 105pd 106pd 108pd 110pd'.split(), seed=48)
        std1 = std1 * fraction_ref
        rstd1 = std1.ratio('pd105')

        correct1 = data.copy(element_symbol='pd').ratio('105pd')
        correct2 = (correct1 / np.mean(rstd1) - 1)
        correct3 = correct2 * 1000
        correct4 = correct2 * 10_000
        correct5 = correct2 * 1_000_000

        self.run(fractionated, correct2, '108pd/105pd', norm_val=rstd1)
        self.run(fractionated, correct2, '108pd/105pd', factor=1, norm_val=rstd1)
        self.run(fractionated, correct3, '108pd/105pd', factor=1000, norm_val=rstd1)
        self.run(fractionated, correct3, '108pd/105pd', factor='ppt', norm_val=rstd1)
        self.run(fractionated, correct3, '108pd/105pd', factor='permil', norm_val=rstd1)
        self.run(fractionated, correct4, '108pd/105pd', factor=10_000, norm_val=rstd1)
        self.run(fractionated, correct4, '108pd/105pd', factor='epsilon', norm_val=rstd1)
        self.run(fractionated, correct5, '108pd/105pd', factor=1_000_000, norm_val=rstd1)
        self.run(fractionated, correct5, '108pd/105pd', factor='mu', norm_val=rstd1)
        self.run(fractionated, correct5, '108pd/105pd', factor='ppm', norm_val=rstd1)

        std1 = np.mean(std1)
        rstd1 = np.mean(rstd1)

        self.run(fractionated, correct2, '108pd/105pd', norm_val=rstd1)
        self.run(fractionated, correct2, '108pd/105pd', factor=1, norm_val=rstd1)
        self.run(fractionated, correct3, '108pd/105pd', factor=1000, norm_val=rstd1)
        self.run(fractionated, correct3, '108pd/105pd', factor='ppt', norm_val=rstd1)
        self.run(fractionated, correct3, '108pd/105pd', factor='permil', norm_val=rstd1)
        self.run(fractionated, correct4, '108pd/105pd', factor=10_000, norm_val=rstd1)
        self.run(fractionated, correct4, '108pd/105pd', factor='epsilon', norm_val=rstd1)
        self.run(fractionated, correct5, '108pd/105pd', factor=1_000_000, norm_val=rstd1)
        self.run(fractionated, correct5, '108pd/105pd', factor='mu', norm_val=rstd1)
        self.run(fractionated, correct5, '108pd/105pd', factor='ppm', norm_val=rstd1)

        # Multiple
        std1 = isopy.random(100, (1, 0.1), keys='102pd 104pd 105pd 106pd 108pd 110pd'.split(),
                            seed=48)
        std1 = std1 * fraction_ref
        rstd1 = std1.ratio('pd105')

        std2 = isopy.random(50, (1, 0.1), keys='102pd 104pd 105pd 106pd 108pd 110pd'.split(),
                            seed=49)
        std2 = std2 * fraction_ref
        rstd2 = std2.ratio('pd105')

        correct1 = data.copy(element_symbol='pd').ratio('105pd')
        correct2 = (correct1 / (np.mean(rstd1)/2 + np.mean(rstd2)/2) - 1)
        correct3 = correct2 * 1000
        correct4 = correct2 * 10_000
        correct5 = correct2 * 1_000_000

        self.run(fractionated, correct2, '108pd/105pd', norm_val=(rstd1, rstd2))
        self.run(fractionated, correct2, '108pd/105pd', factor=1, norm_val=(rstd1, rstd2))
        self.run(fractionated, correct3, '108pd/105pd', factor=1000, norm_val=(rstd1, rstd2))
        self.run(fractionated, correct3, '108pd/105pd', factor='ppt', norm_val=(rstd1, rstd2))
        self.run(fractionated, correct3, '108pd/105pd', factor='permil', norm_val=(rstd1, rstd2))
        self.run(fractionated, correct4, '108pd/105pd', factor=10_000, norm_val=(rstd1, rstd2))
        self.run(fractionated, correct4, '108pd/105pd', factor='epsilon', norm_val=(rstd1, rstd2))
        self.run(fractionated, correct5, '108pd/105pd', factor=1_000_000, norm_val=(rstd1, rstd2))
        self.run(fractionated, correct5, '108pd/105pd', factor='mu', norm_val=(rstd1, rstd2))
        self.run(fractionated, correct5, '108pd/105pd', factor='ppm', norm_val=(rstd1, rstd2))

        std1 = np.mean(std1)
        rstd1 = np.mean(rstd1)

        self.run(fractionated, correct2, '108pd/105pd', norm_val=(rstd1, rstd2))
        self.run(fractionated, correct2, '108pd/105pd', factor=1, norm_val=(rstd1, rstd2))
        self.run(fractionated, correct3, '108pd/105pd', factor=1000, norm_val=(rstd1, rstd2))
        self.run(fractionated, correct3, '108pd/105pd', factor='ppt', norm_val=(rstd1, rstd2))
        self.run(fractionated, correct3, '108pd/105pd', factor='permil', norm_val=(rstd1, rstd2))
        self.run(fractionated, correct4, '108pd/105pd', factor=10_000, norm_val=(rstd1, rstd2))
        self.run(fractionated, correct4, '108pd/105pd', factor='epsilon', norm_val=(rstd1, rstd2))
        self.run(fractionated, correct5, '108pd/105pd', factor=1_000_000, norm_val=(rstd1, rstd2))
        self.run(fractionated, correct5, '108pd/105pd', factor='mu', norm_val=(rstd1, rstd2))
        self.run(fractionated, correct5, '108pd/105pd', factor='ppm', norm_val=(rstd1, rstd2))

        std2 = np.mean(std2)
        rstd2 = np.mean(rstd2)

        self.run(fractionated, correct2, '108pd/105pd', norm_val=(rstd1, rstd2))
        self.run(fractionated, correct2, '108pd/105pd', factor=1, norm_val=(rstd1, rstd2))
        self.run(fractionated, correct3, '108pd/105pd', factor=1000, norm_val=(rstd1, rstd2))
        self.run(fractionated, correct3, '108pd/105pd', factor='ppt', norm_val=(rstd1, rstd2))
        self.run(fractionated, correct3, '108pd/105pd', factor='permil', norm_val=(rstd1, rstd2))
        self.run(fractionated, correct4, '108pd/105pd', factor=10_000, norm_val=(rstd1, rstd2))
        self.run(fractionated, correct4, '108pd/105pd', factor='epsilon', norm_val=(rstd1, rstd2))
        self.run(fractionated, correct5, '108pd/105pd', factor=1_000_000, norm_val=(rstd1, rstd2))
        self.run(fractionated, correct5, '108pd/105pd', factor='mu', norm_val=(rstd1, rstd2))
        self.run(fractionated, correct5, '108pd/105pd', factor='ppm', norm_val=(rstd1, rstd2))

        # Different reference values
        mass_ref = isopy.refval.isotope.mass_number
        fraction_ref = isopy.refval.isotope.initial_solar_system_fraction_L09

        mf_factor = isopy.random(100, (0, 2), seed=47)
        data = isopy.random(100, (1, 0.1), keys='102pd 104pd 105pd 106pd 108pd 110pd'.split(),
                            seed=46)
        data = data * fraction_ref
        data['108pd'] = fraction_ref.get('108pd/105pd') * data['105pd']

        fractionated = data.copy()
        fractionated = isopy.tb.add_mass_fractionation(fractionated, mf_factor, isotope_masses=mass_ref)

        correct1 = data.copy(element_symbol='pd').ratio('105pd')
        correct2 = (correct1 / fraction_ref - 1)
        correct3 = correct2 * 1000
        correct4 = correct2 * 10_000
        correct5 = correct2 * 1_000_000

        self.run(fractionated, correct1, '108pd/105pd', mass_ref=mass_ref, fraction_ref=fraction_ref)
        self.run(fractionated, correct2, '108pd/105pd', factor=1, mass_ref=mass_ref, fraction_ref=fraction_ref)
        self.run(fractionated, correct3, '108pd/105pd', factor=1000, mass_ref=mass_ref, fraction_ref=fraction_ref)
        self.run(fractionated, correct3, '108pd/105pd', factor='ppt', mass_ref=mass_ref, fraction_ref=fraction_ref)
        self.run(fractionated, correct3, '108pd/105pd', factor='permil', mass_ref=mass_ref, fraction_ref=fraction_ref)
        self.run(fractionated, correct4, '108pd/105pd', factor=10_000, mass_ref=mass_ref, fraction_ref=fraction_ref)
        self.run(fractionated, correct4, '108pd/105pd', factor='epsilon', mass_ref=mass_ref, fraction_ref=fraction_ref)
        self.run(fractionated, correct5, '108pd/105pd', factor=1_000_000, mass_ref=mass_ref, fraction_ref=fraction_ref)
        self.run(fractionated, correct5, '108pd/105pd', factor='mu', mass_ref=mass_ref, fraction_ref=fraction_ref)
        self.run(fractionated, correct5, '108pd/105pd', factor='ppm', mass_ref=mass_ref, fraction_ref=fraction_ref)

    def run(self, data, correct, mb_ratio, factor = None, mass_ref = None, fraction_ref=None, norm_val = None):
        if type(factor) is str:
            func = getattr(isopy.tb.mass_independent_correction, factor)
            factor = None
        else:
            func = isopy.tb.mass_independent_correction

        kwargs = {}
        if factor is not None: kwargs['normalisation_factor'] = factor
        if mass_ref is not None: kwargs['isotope_masses'] = mass_ref
        if fraction_ref is not None: kwargs['isotope_fractions'] = fraction_ref
        if norm_val is not None: kwargs['normalisation_value'] = norm_val

        corrected = func(data, mb_ratio, **kwargs)
        assert corrected.keys == correct.keys - mb_ratio
        assert corrected.size == correct.size
        assert corrected.ndim == correct.ndim
        for key in corrected.keys:
            np.testing.assert_allclose(corrected[key], correct[key])


class Test_IsobaricInterferences:
    def test_one(self):
        # No mass fractionation factor
        # Single interference isotope

        # Default reference values
        fraction_ref = isopy.refval.isotope.best_measurement_fraction_M16

        base_data = isopy.random(100, (1, 0.01), keys='101ru 102pd 104pd 105pd 106pd 108pd 110pd 111cd'.split())

        base_data = base_data * fraction_ref
        data = base_data.copy()
        for key in data.keys.filter(element_symbol='pd'):
            data[key] += fraction_ref.get(f'ru{key.mass_number}/ru101', 0) * data['101ru']
            data[key] += fraction_ref.get(f'cd{key.mass_number}/cd111', 0) * data['111cd']

        interferences1 = {'ru': ('102pd', '104pd'), 'cd': ('106pd', '108pd', '110pd')}
        correct1 = base_data.copy()
        correct1['101ru', '111cd'] = 0

        interferences2 = {'ru': ('104pd',), 'cd': ('106pd', '108pd')}
        correct2 = base_data.copy()
        correct2['101ru', '111cd'] = 0
        correct2['102pd'] = data['102pd']
        correct2['110pd'] = data['110pd']

        self.run(data, data, correct1, correct2, interferences1, interferences2, '105pd')

        # Different reference values

        fraction_ref = isopy.refval.isotope.initial_solar_system_fraction_L09

        base_data = isopy.random(100, (1, 0.01), keys='101ru 102pd 104pd 105pd 106pd 108pd 110pd 111cd'.split())

        base_data = base_data * fraction_ref
        data = base_data.copy()
        for key in data.keys.filter(element_symbol='pd'):
            data[key] += fraction_ref.get(f'ru{key.mass_number}/ru101', 0) * data['101ru']
            data[key] += fraction_ref.get(f'cd{key.mass_number}/cd111', 0) * data['111cd']

        interferences1 = {'ru': ('102pd', '104pd'), 'cd': ('106pd', '108pd', '110pd')}
        correct1 = base_data.copy()
        correct1['101ru', '111cd'] = 0

        interferences2 = {'ru': ('104pd',), 'cd': ('106pd', '108pd')}
        correct2 = base_data.copy()
        correct2['101ru', '111cd'] = 0
        correct2['102pd'] = data['102pd']
        correct2['110pd'] = data['110pd']

        self.run(data, data, correct1, correct2, interferences1, interferences2, '105pd',
                 fraction_ref=fraction_ref)

    def test_two(self):
        # No mass fractionation factor
        # Multiple interference isotopes

        # Default reference values
        fraction_ref = isopy.refval.isotope.best_measurement_fraction_M16

        base_data = isopy.random(100, (1, 0.01), keys='99ru 101ru 102pd 104pd 105pd 106pd 108pd 110pd 111cd 112cd'.split())
        # 112cd > 111cd, 101ru > 99ru
        base_data = base_data * fraction_ref
        data1 = base_data.copy()
        data1['99ru', '111cd'] = -1 # so that we dont accidentally make this the largest isotope
        for key in data1.keys.filter(key_neq = 'ru101 cd112'.split()):
            data1[key] += fraction_ref.get(f'ru{key.mass_number}/ru101', 0) * data1['101ru']
            data1[key] += fraction_ref.get(f'cd{key.mass_number}/cd112', 0) * data1['112cd']

        interferences1 = {'ru': ('102pd', '104pd'), 'cd': ('106pd', '108pd', '110pd')}
        correct1 = base_data.copy()
        correct1['101ru', '112cd'] = 0
        correct1['99ru', '111cd'] = -1

        interferences2 = {'ru99': ('104pd',), 'cd111': ('106pd', '108pd')}
        data2 = base_data.copy()
        data2['ru101', 'cd112'] = -1 # so that we dont accidentally make this the largest isotope
        for key in data2.keys.filter(key_neq='ru99 cd111 102pd 110pd'.split()):
            data2[key] += fraction_ref.get(f'ru{key.mass_number}/ru99', 0) * data2['99ru']
            data2[key] += fraction_ref.get(f'cd{key.mass_number}/cd111', 0) * data2['111cd']
        correct2 = base_data.copy()
        correct2['99ru', '111cd'] = 0
        correct2['101ru', '112cd'] = -1

        self.run(data1, data2, correct1, correct2, interferences1, interferences2, '105pd')

        # Different reference values
        fraction_ref = isopy.refval.isotope.initial_solar_system_fraction_L09

        base_data = isopy.random(100, (1, 0.01),
                                 keys='99ru 101ru 102pd 104pd 105pd 106pd 108pd 110pd 111cd 112cd'.split())
        # 112cd > 111cd, 101ru > 99ru
        base_data = base_data * fraction_ref
        data1 = base_data.copy()
        data1['99ru', '111cd'] = -1  # so that we dont accidentally make this the largest isotope
        for key in data1.keys.filter(key_neq='ru101 cd112'.split()):
            data1[key] += fraction_ref.get(f'ru{key.mass_number}/ru101', 0) * data1['101ru']
            data1[key] += fraction_ref.get(f'cd{key.mass_number}/cd112', 0) * data1['112cd']

        interferences1 = {'ru': ('102pd', '104pd'), 'cd': ('106pd', '108pd', '110pd')}
        correct1 = base_data.copy()
        correct1['101ru', '112cd'] = 0
        correct1['99ru', '111cd'] = -1

        interferences2 = {'ru99': ('104pd',), 'cd111': ('106pd', '108pd')}
        data2 = base_data.copy()
        data2['ru101', 'cd112'] = -1  # so that we dont accidentally make this the largest isotope
        for key in data2.keys.filter(key_neq='ru99 cd111 102pd 110pd'.split()):
            data2[key] += fraction_ref.get(f'ru{key.mass_number}/ru99', 0) * data2['99ru']
            data2[key] += fraction_ref.get(f'cd{key.mass_number}/cd111', 0) * data2['111cd']
        correct2 = base_data.copy()
        correct2['99ru', '111cd'] = 0
        correct2['101ru', '112cd'] = -1

        self.run(data1, data2, correct1, correct2, interferences1, interferences2, '105pd',
                 fraction_ref=fraction_ref)

    def test_three(self):
        #Mass fractionation
        #Single interference isotope
        mass_ref = isopy.refval.isotope.mass_W17
        fraction_ref = isopy.refval.isotope.best_measurement_fraction_M16

        base_data = isopy.random(100, (1, 0.01),
                                 keys='101ru 102pd 104pd 105pd 106pd 108pd 110pd 111cd'.split())
        mf_factor = isopy.random(100, (0,2))

        base_data = base_data * fraction_ref
        data = base_data.copy()
        for key in data.keys.filter(element_symbol='pd'):
            if (ru:=fraction_ref.get(f'ru{key.mass_number}/ru101', 0)) > 0:
                ru *= data['101ru'] * (mass_ref.get(f'ru{key.mass_number}/ru101', 0) ** mf_factor)
                data[key] += ru

            if (cd:=fraction_ref.get(f'cd{key.mass_number}/cd111', 0)) > 0:
                cd *= data['111cd'] * (mass_ref.get(f'cd{key.mass_number}/cd111', 0) ** mf_factor)
                data[key] += cd

        interferences1 = {'ru': ('102pd', '104pd'), 'cd': ('106pd', '108pd', '110pd')}
        correct1 = base_data.copy()
        correct1['101ru', '111cd'] = 0

        interferences2 = {'ru': ('104pd',), 'cd': ('106pd', '108pd')}
        correct2 = base_data.copy()
        correct2['101ru', '111cd'] = 0
        correct2['102pd'] = data['102pd']
        correct2['110pd'] = data['110pd']

        self.run(data, data, correct1, correct2, interferences1, interferences2, '105pd',
                 mf_factor=mf_factor)

        #M Multiple interference isotopes
        # Different reference values
        mass_ref = isopy.refval.isotope.mass_number
        fraction_ref = isopy.refval.isotope.initial_solar_system_fraction_L09

        base_data = isopy.random(100, (1, 0.01),
                                 keys='99ru 101ru 102pd 104pd 105pd 106pd 108pd 110pd 111cd 112cd'.split())
        # 112cd > 111cd, 101ru > 99ru
        base_data = base_data * fraction_ref
        data1 = base_data.copy()
        data1['99ru', '111cd'] = -1  # so that we dont accidentally make this the largest isotope
        for key in data1.keys.filter(key_neq='ru101 cd112'.split()):
            if (ru:=fraction_ref.get(f'ru{key.mass_number}/ru101', 0)) > 0:
                ru *= data1['101ru'] * (mass_ref.get(f'ru{key.mass_number}/ru101', 0) ** mf_factor)
                data1[key] += ru

            if (cd:=fraction_ref.get(f'cd{key.mass_number}/cd112', 0)) > 0:
                cd *= data1['cd112'] * (mass_ref.get(f'cd{key.mass_number}/cd112', 0) ** mf_factor)
                data1[key] += cd

        interferences1 = {'ru': ('102pd', '104pd'), 'cd': ('106pd', '108pd', '110pd')}
        correct1 = base_data.copy()
        correct1['101ru', '112cd'] = 0
        correct1['99ru', '111cd'] = -1

        interferences2 = {'ru99': ('104pd',), 'cd111': ('106pd', '108pd')}
        data2 = base_data.copy()
        data2['ru101', 'cd112'] = -1  # so that we dont accidentally make this the largest isotope
        for key in data2.keys.filter(key_neq='ru99 cd111 102pd 110pd'.split()):
            if (ru:=fraction_ref.get(f'ru{key.mass_number}/ru99', 0)) > 0:
                ru *= data2['ru99'] * (mass_ref.get(f'ru{key.mass_number}/ru99', 0) ** mf_factor)
                data2[key] += ru

            if (cd:=fraction_ref.get(f'cd{key.mass_number}/cd111', 0)) > 0:
                cd *= data2['111cd'] * (mass_ref.get(f'cd{key.mass_number}/cd111', 0) ** mf_factor)
                data2[key] += cd

        correct2 = base_data.copy()
        correct2['99ru', '111cd'] = 0
        correct2['101ru', '112cd'] = -1

        self.run(data1, data2, correct1, correct2, interferences1, interferences2, '105pd',
                 fraction_ref=fraction_ref, mass_ref=mass_ref, mf_factor=mf_factor)

    def run(self, data1, data2, correct1, correct2, interferences1, interferences2, denom=None,
            mf_factor=None, fraction_ref=None, mass_ref=None):

        interferences = isopy.tb.find_isobaric_interferences('pd', data1)
        assert len(interferences) == len(interferences)
        for key in interferences1:
            assert key in interferences
            assert interferences[key] == interferences1[key]

        corrected1 = isopy.tb.remove_isobaric_interferences(data1, interferences,
                                                            mf_factor=mf_factor,
                                                            isotope_fractions=fraction_ref,
                                                            isotope_masses=mass_ref)
        assert corrected1.keys == correct1.keys
        assert corrected1.size == correct1.size
        for key in corrected1.keys:
            np.testing.assert_allclose(corrected1[key], correct1[key])


        corrected2 = isopy.tb.remove_isobaric_interferences(data2, interferences2,
                                                            mf_factor=mf_factor,
                                                            isotope_fractions=fraction_ref,
                                                            isotope_masses=mass_ref)
        assert corrected2.keys == correct2.keys
        assert corrected2.size == correct2.size
        for key in corrected2.keys:
            np.testing.assert_allclose(corrected2[key], correct2[key])


        #Ratio test data
        if denom is not None:
            data1 = data1.ratio(denom)
            data2 = data2.ratio(denom)
            correct1 = correct1.ratio(denom)
            correct2 = correct2.ratio(denom)

            interferences = isopy.tb.find_isobaric_interferences('pd', data1)
            assert len(interferences) == len(interferences)
            for key in interferences1:
                assert key in interferences
                assert interferences[key] == interferences1[key]

            corrected1 = isopy.tb.remove_isobaric_interferences(data1, interferences,
                                                            mf_factor=mf_factor,
                                                            isotope_fractions=fraction_ref,
                                                            isotope_masses=mass_ref)
            assert corrected1.keys == correct1.keys
            assert corrected1.size == correct1.size
            for key in corrected1.keys:
                np.testing.assert_allclose(corrected1[key], correct1[key])

            corrected2 = isopy.tb.remove_isobaric_interferences(data2, interferences2,
                                                            mf_factor=mf_factor,
                                                            isotope_fractions=fraction_ref,
                                                            isotope_masses=mass_ref)
            assert corrected2.keys == correct2.keys
            assert corrected2.size == correct2.size
            for key in corrected2.keys:
                np.testing.assert_allclose(corrected2[key], correct2[key])

    def test_find(self):
        interferences = isopy.tb.find_isobaric_interferences('pd', ('ru', 'cd'))
        assert len(interferences) == 2
        assert 'ru' in interferences
        assert interferences['ru'] == ('102Pd', '104Pd')
        assert 'cd' in interferences
        assert interferences['cd'] == ('106Pd', '108Pd', '110Pd')

        interferences = isopy.tb.find_isobaric_interferences('pd', ('ru', 'rh', 'ag', 'cd'))
        assert len(interferences) == 2
        assert 'ru' in interferences
        assert interferences['ru'] == ('102Pd', '104Pd')
        assert 'cd' in interferences
        assert interferences['cd'] == ('106Pd', '108Pd', '110Pd')

        interferences = isopy.tb.find_isobaric_interferences('ce')
        assert len(interferences) == 4
        assert 'xe' in interferences
        assert interferences['xe'] == ('136Ce',)
        assert 'ba' in interferences
        assert interferences['ba'] == ('136Ce', '138Ce')
        assert 'la' in interferences
        assert interferences['la'] == ('138Ce', )
        assert 'nd' in interferences
        assert interferences['nd'] == ('142Ce',)

        interferences = isopy.tb.find_isobaric_interferences('138ce')
        assert len(interferences) == 2
        assert 'ba' in interferences
        assert interferences['ba'] == ('138Ce',)
        assert 'la' in interferences
        assert interferences['la'] == ('138Ce',)

        interferences = isopy.tb.find_isobaric_interferences('zn', ('ni', 'ge', 'ba++'))
        assert len(interferences) == 3
        assert 'ni' in interferences
        assert interferences['ni'] == ('64Zn',)
        assert 'ge' in interferences
        assert interferences['ge'] == ('70Zn',)
        assert 'ba++' in interferences
        assert interferences['ba++'] == ('66Zn', '67Zn', '68Zn')


class Test_rDelta():

    def test_rDelta1(self):
        # Data is a single value
        data = isopy.refval.isotope.fraction.asarray(element_symbol='pd')

        # Dict
        reference =  isopy.refval.isotope.fraction
        correct1 = isopy.zeros(None, data.keys)
        correct2 = isopy.ones(None, data.keys)

        self.run(data, data, reference, correct1, correct2)

        # Single array
        reference = isopy.random(100, keys=data.keys)
        correct1 = data / np.mean(reference) - 1
        correct2 = data / np.mean(reference)
        self.run(data, data, reference, correct1, correct2)
        self.run(data, data, np.mean(reference), correct1, correct2)

        correct1 = correct1 * 10_000
        correct2 = correct2 * 10_000
        self.run(data, data, reference, correct1, correct2, 10_000)
        self.run(data, data, np.mean(reference), correct1, correct2, 10_000)

        # Multiple values
        reference1 = isopy.random(100, keys=data.keys)
        reference2 = isopy.random(100, keys=data.keys)
        meanmean = np.mean(reference1)/2 + np.mean(reference2)/2

        correct1 = data / meanmean - 1
        correct2 = data / meanmean
        self.run(data, data, (reference1, reference2), correct1, correct2)
        self.run(data, data, (np.mean(reference1), reference2), correct1, correct2)
        self.run(data, data, (np.mean(reference1), np.mean(reference2)), correct1, correct2)

        correct1 = correct1 * 10_000
        correct2 = correct2 * 10_000
        self.run(data, data, (reference1, reference2), correct1, correct2, 10_000)
        self.run(data, data, (np.mean(reference1), reference2), correct1, correct2, 10_000)
        self.run(data, data, (np.mean(reference1), np.mean(reference2)), correct1, correct2, 10_000)

        # Keys that do not match
        data2 = data.copy()
        data2['105pd', '106pd'] = np.nan
        reference1 = isopy.random(100, keys='101ru 102pd 104pd 105pd 108pd 110pd 111cd'.split())
        reference2 = isopy.random(100, keys='101ru 102pd 104pd 106pd 108pd 110pd 111cd'.split())
        meanmean = np.mean(reference1) / 2 + np.mean(reference2) / 2

        correct1 = data / meanmean - 1
        correct2 = data / meanmean
        self.run(data, data2, (reference1, reference2), correct1, correct2)
        self.run(data, data2, (np.mean(reference1), reference2), correct1, correct2)
        self.run(data, data2, (np.mean(reference1), np.mean(reference2)), correct1, correct2)

        correct1 = correct1 * 10_000
        correct2 = correct2 * 10_000
        self.run(data, data2, (reference1, reference2), correct1, correct2, 10_000)
        self.run(data, data2, (np.mean(reference1), reference2), correct1, correct2, 10_000)
        self.run(data, data2, (np.mean(reference1), np.mean(reference2)), correct1, correct2, 10_000)

    def test_rDelta2(self):
        data = isopy.random(100, keys=isopy.refval.element.isotopes['pd'])
        data = data * isopy.refval.isotope.fraction

        # Dict
        reference =  isopy.refval.isotope.fraction
        correct1 = data / reference - 1
        correct2 = data / reference

        self.run(data, data, reference, correct1, correct2)

        # Single array
        reference = isopy.random(100, keys=data.keys)
        correct1 = data / np.mean(reference) - 1
        correct2 = data / np.mean(reference)
        self.run(data, data, reference, correct1, correct2)
        self.run(data, data, np.mean(reference), correct1, correct2)

        correct1 = correct1 * 10_000
        correct2 = correct2 * 10_000
        self.run(data, data, reference, correct1, correct2, 10_000)
        self.run(data, data, np.mean(reference), correct1, correct2, 10_000)

        # Multiple values
        reference1 = isopy.random(100, keys=data.keys)
        reference2 = isopy.random(100, keys=data.keys)
        meanmean = np.mean(reference1)/2 + np.mean(reference2)/2

        correct1 = data / meanmean - 1
        correct2 = data / meanmean
        self.run(data, data, (reference1, reference2), correct1, correct2)
        self.run(data, data, (np.mean(reference1), reference2), correct1, correct2)
        self.run(data, data, (np.mean(reference1), np.mean(reference2)), correct1, correct2)

        correct1 = correct1 * 10_000
        correct2 = correct2 * 10_000
        self.run(data, data, (reference1, reference2), correct1, correct2, 10_000)
        self.run(data, data, (np.mean(reference1), reference2), correct1, correct2, 10_000)
        self.run(data, data, (np.mean(reference1), np.mean(reference2)), correct1, correct2, 10_000)

        # Keys that do not match
        data2 = data.copy()
        data2['105pd', '106pd'] = np.nan
        reference1 = isopy.random(100, keys='101ru 102pd 104pd 105pd 108pd 110pd 111cd'.split())
        reference2 = isopy.random(100, keys='101ru 102pd 104pd 106pd 108pd 110pd 111cd'.split())
        meanmean = np.mean(reference1) / 2 + np.mean(reference2) / 2

        correct1 = data / meanmean - 1
        correct2 = data / meanmean
        self.run(data, data2, (reference1, reference2), correct1, correct2)
        self.run(data, data2, (np.mean(reference1), reference2), correct1, correct2)
        self.run(data, data2, (np.mean(reference1), np.mean(reference2)), correct1, correct2)

        correct1 = correct1 * 10_000
        correct2 = correct2 * 10_000
        self.run(data, data2, (reference1, reference2), correct1, correct2, 10_000)
        self.run(data, data2, (np.mean(reference1), reference2), correct1, correct2, 10_000)
        self.run(data, data2, (np.mean(reference1), np.mean(reference2)), correct1, correct2, 10_000)

    def test_presets(self):
        data = isopy.random(100, keys=isopy.refval.element.isotopes['pd'])
        data = data * isopy.refval.isotope.fraction
        reference = isopy.refval.isotope.fraction

        correct = (data / reference - 1) * 1000
        normalised = isopy.tb.rDelta.ppt(data, reference)
        denormalised = isopy.tb.inverse_rDelta.ppt(normalised, reference)
        self.compare(correct, normalised)
        self.compare(data, denormalised)

        correct = (data / reference - 1) * 1000
        normalised = isopy.tb.rDelta.permil(data, reference)
        denormalised = isopy.tb.inverse_rDelta.permil(normalised, reference)
        self.compare(correct, normalised)
        self.compare(data, denormalised)

        correct = (data / reference - 1) * 10_000
        normalised = isopy.tb.rDelta.epsilon(data, reference)
        denormalised = isopy.tb.inverse_rDelta.epsilon(normalised, reference)
        self.compare(correct, normalised)
        self.compare(data, denormalised)

        correct = (data / reference - 1) * 1_000_000
        normalised = isopy.tb.rDelta.mu(data, reference)
        denormalised = isopy.tb.inverse_rDelta.mu(normalised, reference)
        self.compare(correct, normalised)
        self.compare(data, denormalised)

        correct = (data / reference - 1) * 1_000_000
        normalised = isopy.tb.rDelta.ppm(data, reference)
        denormalised = isopy.tb.inverse_rDelta.ppm(normalised, reference)
        self.compare(correct, normalised)
        self.compare(data, denormalised)

    def run(self, data1, data2, reference_value, correct1, correct2, factor=1):

        normalised = isopy.tb.rDelta(data1, reference_value, factor=factor)
        assert normalised.keys == data1.keys
        assert normalised.size == data1.size
        assert normalised.ndim == data1.ndim
        for key in normalised.keys:
            np.testing.assert_allclose(normalised[key], correct1[key])

        denormalised = isopy.tb.inverse_rDelta(normalised, reference_value, factor=factor)
        assert denormalised.keys == data1.keys
        assert denormalised.size == data1.size
        assert denormalised.ndim == data1.ndim
        for key in denormalised.keys:
            np.testing.assert_allclose(denormalised[key], data2[key])

        normalised = isopy.tb.rDelta(data1, reference_value, factor=factor, deviations=0)
        assert normalised.keys == data1.keys
        assert normalised.size == data1.size
        assert normalised.ndim == data1.ndim
        for key in normalised.keys:
            np.testing.assert_allclose(normalised[key], correct2[key])

        denormalised = isopy.tb.inverse_rDelta(normalised, reference_value, factor=factor, deviations=0)
        assert denormalised.keys == data1.keys
        assert denormalised.size == data1.size
        assert denormalised.ndim == data1.ndim
        for key in denormalised.keys:
            np.testing.assert_allclose(denormalised[key], data2[key])

    def compare(self, correct, calculated):
        assert calculated.keys == correct.keys
        assert calculated.size == correct.size
        assert calculated.ndim == correct.ndim
        for key in calculated.keys:
            np.testing.assert_allclose(calculated[key], correct[key])


class Test_OutliersLimits:
    def test_limits(self):
        data = isopy.random(100, (1,1), keys=isopy.refval.element.isotopes['pd'])

        median = np.median(data)
        mean = np.mean(data)
        mad3 = isopy.mad3(data)
        sd2 = isopy.sd2(data)

        upper = isopy.tb.upper_limit(data)
        assert upper == median + mad3

        upper = isopy.tb.upper_limit(data, np.mean, isopy.sd2)
        assert upper == mean + sd2

        upper = isopy.tb.upper_limit.sd2(data)
        assert upper == mean + sd2

        upper = isopy.tb.upper_limit(data, 1, isopy.sd2)
        assert upper == 1 + sd2

        upper = isopy.tb.upper_limit(data, np.mean, 1)
        assert upper == mean + 1

        upper = isopy.tb.upper_limit(data, 1, 1)
        assert upper == 2

        lower = isopy.tb.lower_limit(data)
        assert lower == median - mad3

        lower = isopy.tb.lower_limit.sd2(data)
        assert lower == mean - sd2

        lower = isopy.tb.lower_limit(data, np.mean, isopy.sd2)
        assert lower == mean - sd2

        lower = isopy.tb.lower_limit(data, 1, isopy.sd2)
        assert lower == 1 - sd2

        lower = isopy.tb.lower_limit(data, np.mean, 1)
        assert lower == mean - 1

        lower = isopy.tb.lower_limit(data, 1, 1)
        assert lower == 0

    def test_find_outliers1(self):
        #axis = 0
        data = isopy.random(100, (1, 1), keys=isopy.refval.element.isotopes['pd'])

        median = np.median(data)
        mean = np.mean(data)
        mad3 = isopy.mad3(data)
        sd = isopy.sd(data)

        median_outliers = (data > (median + mad3)) + (data < (median - mad3))
        mean_outliers = (data > (mean + sd)) + (data < (mean - sd))
        mean_outliers1 = (data > (1 + sd)) + (data < (1 - sd))
        mean_outliers2 = (data > (mean + 1)) + (data < (mean - 1))
        mean_outliers3 = (data > (1 + 1)) + (data < (1 - 1))

        outliers = isopy.tb.find_outliers(data)
        assert outliers.keys == data.keys
        assert outliers.size == data.size
        for key in outliers.keys:
            np.testing.assert_allclose(outliers[key], median_outliers[key])

        outliers = isopy.tb.find_outliers(data, np.mean, isopy.sd)
        assert outliers.keys == data.keys
        assert outliers.size == data.size
        for key in outliers.keys:
            np.testing.assert_allclose(outliers[key], mean_outliers[key])

        outliers = isopy.tb.find_outliers.sd(data)
        assert outliers.keys == data.keys
        assert outliers.size == data.size
        for key in outliers.keys:
            np.testing.assert_allclose(outliers[key], mean_outliers[key])

        outliers = isopy.tb.find_outliers(data, 1, isopy.sd)
        assert outliers.keys == data.keys
        assert outliers.size == data.size
        for key in outliers.keys:
            np.testing.assert_allclose(outliers[key], mean_outliers1[key])

        outliers = isopy.tb.find_outliers(data, np.mean, 1)
        assert outliers.keys == data.keys
        assert outliers.size == data.size
        for key in outliers.keys:
            np.testing.assert_allclose(outliers[key], mean_outliers2[key])

        outliers = isopy.tb.find_outliers(data, 1, 1)
        assert outliers.keys == data.keys
        assert outliers.size == data.size
        for key in outliers.keys:
            np.testing.assert_allclose(outliers[key], mean_outliers3[key])

        # invert
        median_outliers = np.invert(median_outliers)
        mean_outliers = np.invert(mean_outliers)
        mean_outliers1 = np.invert(mean_outliers1)
        mean_outliers2 = np.invert(mean_outliers2)
        mean_outliers3 = np.invert(mean_outliers3)

        outliers = isopy.tb.find_outliers(data, invert=True)
        assert outliers.keys == data.keys
        assert outliers.size == data.size
        for key in outliers.keys:
            np.testing.assert_allclose(outliers[key], median_outliers[key])

        outliers = isopy.tb.find_outliers(data, np.mean, isopy.sd, invert=True)
        assert outliers.keys == data.keys
        assert outliers.size == data.size
        for key in outliers.keys:
            np.testing.assert_allclose(outliers[key], mean_outliers[key])

        outliers = isopy.tb.find_outliers.sd(data, invert=True)
        assert outliers.keys == data.keys
        assert outliers.size == data.size
        for key in outliers.keys:
            np.testing.assert_allclose(outliers[key], mean_outliers[key])

        outliers = isopy.tb.find_outliers(data, 1, isopy.sd, invert=True)
        assert outliers.keys == data.keys
        assert outliers.size == data.size
        for key in outliers.keys:
            np.testing.assert_allclose(outliers[key], mean_outliers1[key])

        outliers = isopy.tb.find_outliers(data, np.mean, 1, invert=True)
        assert outliers.keys == data.keys
        assert outliers.size == data.size
        for key in outliers.keys:
            np.testing.assert_allclose(outliers[key], mean_outliers2[key])

        outliers = isopy.tb.find_outliers(data, 1, 1, invert=True)
        assert outliers.keys == data.keys
        assert outliers.size == data.size
        for key in outliers.keys:
            np.testing.assert_allclose(outliers[key], mean_outliers3[key])

    def test_find_outliers2(self):
        # axis = 0
        data = isopy.random(100, (1, 1), keys=isopy.refval.element.isotopes['pd'])

        median = np.median(data)
        mean = np.mean(data)
        mad3 = isopy.mad3(data)
        sd = isopy.sd2(data)

        median_outliers = np.any((data > (median + mad3)) + (data < (median - mad3)), axis=1)
        mean_outliers = np.any((data > (mean + sd)) + (data < (mean - sd)), axis=1)
        mean_outliers1 = np.any((data > (1 + sd)) + (data < (1 - sd)), axis=1)
        mean_outliers2 = np.any((data > (mean + 1)) + (data < (mean - 1)), axis=1)
        mean_outliers3 = np.any((data > (1 + 1)) + (data < (1 - 1)), axis=1)

        outliers = isopy.tb.find_outliers(data, axis=1)
        assert len(outliers) == data.size
        np.testing.assert_allclose(outliers, median_outliers)

        outliers = isopy.tb.find_outliers(data, np.mean, isopy.sd2, axis=1)
        assert len(outliers) == data.size
        np.testing.assert_allclose(outliers, mean_outliers)

        outliers = isopy.tb.find_outliers.sd2(data, axis=1)
        assert len(outliers) == data.size
        np.testing.assert_allclose(outliers, mean_outliers)

        outliers = isopy.tb.find_outliers(data, 1, isopy.sd2, axis=1)
        assert len(outliers) == data.size
        np.testing.assert_allclose(outliers, mean_outliers1)

        outliers = isopy.tb.find_outliers(data, np.mean, 1, axis=1)
        assert len(outliers) == data.size
        np.testing.assert_allclose(outliers, mean_outliers2)

        outliers = isopy.tb.find_outliers(data, 1, 1, axis=1)
        assert len(outliers) == data.size
        np.testing.assert_allclose(outliers, mean_outliers3)

        # invert

        median_outliers = np.invert(median_outliers)
        mean_outliers = np.invert(mean_outliers)
        mean_outliers1 = np.invert(mean_outliers1)
        mean_outliers2 = np.invert(mean_outliers2)
        mean_outliers3 = np.invert(mean_outliers3)

        outliers = isopy.tb.find_outliers(data, axis=1, invert=True)
        assert len(outliers) == data.size
        np.testing.assert_allclose(outliers, median_outliers)

        outliers = isopy.tb.find_outliers(data, np.mean, isopy.sd2, axis=1, invert=True)
        assert len(outliers) == data.size
        np.testing.assert_allclose(outliers, mean_outliers)

        outliers = isopy.tb.find_outliers.sd2(data, axis=1, invert=True)
        assert len(outliers) == data.size
        np.testing.assert_allclose(outliers, mean_outliers)

        outliers = isopy.tb.find_outliers(data, 1, isopy.sd2, axis=1, invert=True)
        assert len(outliers) == data.size
        np.testing.assert_allclose(outliers, mean_outliers1)

        outliers = isopy.tb.find_outliers(data, np.mean, 1, axis=1, invert=True)
        assert len(outliers) == data.size
        np.testing.assert_allclose(outliers, mean_outliers2)

        outliers = isopy.tb.find_outliers(data, 1, 1, axis=1, invert=True)
        assert len(outliers) == data.size
        np.testing.assert_allclose(outliers, mean_outliers3)


class Test_Make:
    def test_make_array1(self):
        # No mass fractionation

        mass_ref = isopy.refval.isotope.mass_W17
        fraction_ref = isopy.refval.isotope.best_measurement_fraction_M16

        correct = isopy.ones(None, keys='102pd 104pd 105pd 106pd 108pd 110pd'.split())
        correct = correct * fraction_ref
        correct10 = correct.normalise(10, '106pd')

        self.compare(correct, isopy.tb.make_ms_array('pd'))
        self.compare(correct10, isopy.tb.make_ms_beams('pd', integrations=None))
        self.compare(correct10, isopy.tb.make_ms_sample('pd', integrations=None))

        correct = isopy.ones(None, keys='101ru 102pd 104pd 105pd 106pd 108pd 110pd 111cd'.split())
        correct = correct * fraction_ref
        for key in correct.keys.filter(key_neq = 'ru101 cd111'.split()):
            correct[key] += fraction_ref.get(f'ru{key.mass_number}/ru101', 0) * correct['101ru']
            correct[key] += fraction_ref.get(f'cd{key.mass_number}/cd111', 0) * correct['111cd']
        correct10 = correct.normalise(10, isopy.keymax)

        self.compare(correct, isopy.tb.make_ms_array('pd', '101ru', '111cd'))
        self.compare(correct10, isopy.tb.make_ms_beams('pd', '101ru', '111cd', integrations=None))

        correct = isopy.ones(None, keys='101ru 102pd 104pd 105pd 106pd 108pd 110pd 111cd'.split())
        correct['101ru'] *= 0.1
        correct['111cd'] *= 0.01
        correct = correct * fraction_ref
        for key in correct.keys.filter(key_neq='ru101 cd111'.split()):
            correct[key] += fraction_ref.get(f'ru{key.mass_number}/ru101', 0) * correct['101ru']
            correct[key] += fraction_ref.get(f'cd{key.mass_number}/cd111', 0) * correct['111cd']
        correct10 = correct.normalise(10, '106pd')

        self.compare(correct, isopy.tb.make_ms_array('pd', **{'101ru': 0.1, '111cd':0.01}))
        self.compare(correct, isopy.tb.make_ms_array('pd', ru101 = 0.1, cd111=0.01))
        self.compare(correct10, isopy.tb.make_ms_beams('pd', ru101 = 0.1, cd111=0.01, integrations=None))
        self.compare(correct10, isopy.tb.make_ms_sample('pd', ru101 = 0.1, cd111=0.01, integrations=None))

        correct = isopy.ones(None, keys='99ru 101ru 102pd 104pd 105pd 106pd 108pd 110pd 111cd 112cd'.split())
        correct['101ru'] *= 0.1
        correct['99ru'] *= 0.1
        correct['111cd'] *= 0.01
        correct['112cd'] *= 0
        correct2 = correct * fraction_ref
        correct = correct2.copy()
        for key in correct.keys.filter(key_neq='ru99 ru101 cd111'.split()):
            correct[key] += fraction_ref.get(f'ru{key.mass_number}/ru101', 0) * correct['101ru']
            correct[key] += fraction_ref.get(f'ru{key.mass_number}/ru99', 0) * correct['99ru']
            correct[key] += fraction_ref.get(f'cd{key.mass_number}/cd111', 0) * correct['111cd']
        correct['ru99'] += fraction_ref.get(f'ru99/ru101', 0) * correct2['101ru']
        correct['101ru'] += fraction_ref.get(f'ru101/ru99', 0) * correct2['ru99']

        correct10 = correct.normalise(10, '106pd')

        self.compare(correct, isopy.tb.make_ms_array('pd', **{'101ru': 0.1, '111cd': 0.01, '99ru': 0.1, '112cd': 0}))
        self.compare(correct, isopy.tb.make_ms_array('pd', ru101=0.1, cd111=0.01, ru99=0.1, cd112=0))
        self.compare(correct10,
                     isopy.tb.make_ms_beams('pd', ru101=0.1, cd111=0.01, ru99=0.1, cd112=0, integrations=None))
        self.compare(correct10,
                     isopy.tb.make_ms_sample('pd', ru101=0.1, cd111=0.01, ru99=0.1, cd112=0, integrations=None))

        # Different reference values

        mass_ref = isopy.refval.isotope.mass_number
        fraction_ref = isopy.refval.isotope.initial_solar_system_fraction_L09

        correct = isopy.ones(None, keys='102pd 104pd 105pd 106pd 108pd 110pd'.split())
        correct = correct * fraction_ref
        correct10 = correct.normalise(10, '106pd')

        self.compare(correct, isopy.tb.make_ms_array('pd', isotope_fractions=fraction_ref, isotope_masses=mass_ref))
        self.compare(correct10, isopy.tb.make_ms_beams('pd', integrations=None, isotope_fractions=fraction_ref, isotope_masses=mass_ref))
        self.compare(correct10, isopy.tb.make_ms_sample('pd', integrations=None, isotope_fractions=fraction_ref, isotope_masses=mass_ref))


        correct = isopy.ones(None, keys='101ru 102pd 104pd 105pd 106pd 108pd 110pd 111cd'.split())
        correct = correct * fraction_ref
        for key in correct.keys.filter(key_neq='ru101 cd111'.split()):
            correct[key] += fraction_ref.get(f'ru{key.mass_number}/ru101', 0) * correct['101ru']
            correct[key] += fraction_ref.get(f'cd{key.mass_number}/cd111', 0) * correct['111cd']
        correct10 = correct.normalise(10, isopy.keymax)

        self.compare(correct, isopy.tb.make_ms_array('pd', '101ru', '111cd', isotope_fractions=fraction_ref, isotope_masses=mass_ref))
        self.compare(correct10, isopy.tb.make_ms_beams('pd', '101ru', '111cd', integrations=None, isotope_fractions=fraction_ref, isotope_masses=mass_ref))

        correct = isopy.ones(None, keys='101ru 102pd 104pd 105pd 106pd 108pd 110pd 111cd'.split())
        correct['101ru'] *= 0.1
        correct['111cd'] *= 0.01
        correct = correct * fraction_ref
        for key in correct.keys.filter(key_neq='ru101 cd111'.split()):
            correct[key] += fraction_ref.get(f'ru{key.mass_number}/ru101', 0) * correct['101ru']
            correct[key] += fraction_ref.get(f'cd{key.mass_number}/cd111', 0) * correct['111cd']
        correct10 = correct.normalise(10, '106pd')

        self.compare(correct, isopy.tb.make_ms_array('pd', **{'101ru': 0.1, '111cd': 0.01}, isotope_fractions=fraction_ref, isotope_masses=mass_ref))
        self.compare(correct, isopy.tb.make_ms_array('pd', ru101=0.1, cd111=0.01, isotope_fractions=fraction_ref, isotope_masses=mass_ref))
        self.compare(correct10, isopy.tb.make_ms_beams('pd', ru101=0.1, cd111=0.01, integrations=None, isotope_fractions=fraction_ref, isotope_masses=mass_ref))
        self.compare(correct10, isopy.tb.make_ms_sample('pd', ru101=0.1, cd111=0.01, integrations=None, isotope_fractions=fraction_ref, isotope_masses=mass_ref))

    def test_make_array2(self):
        # At this stage we know that the functions correctly create the arrays.
        # So we only need to make sure that what we create can be reversed using the
        # mass independent correction.

        # Default reference values

        mass_ref = isopy.refval.isotope.mass_W17
        fraction_ref = isopy.refval.isotope.best_measurement_fraction_M16

        correct = isopy.ones(None, keys='102pd 104pd 105pd 106pd 110pd'.split())
        correct = correct * fraction_ref
        correct = correct.ratio('105pd')

        result = isopy.tb.make_ms_array('pd', 'ru', 'cd')
        corrected = isopy.tb.mass_independent_correction(result, '108pd/105pd')
        self.compare(correct, corrected)

        result = isopy.tb.make_ms_beams('pd', 'ru', 'cd', integrations=None)
        corrected = isopy.tb.mass_independent_correction(result, '108pd/105pd')
        self.compare(correct, corrected)

        result = isopy.tb.make_ms_sample('pd', ru=1, cd=1, integrations=None)
        corrected = isopy.tb.mass_independent_correction(result, '108pd/105pd')
        self.compare(correct, corrected)

        result = isopy.tb.make_ms_array('pd', ru101=0.1, cd111 = 0.01)
        corrected = isopy.tb.mass_independent_correction(result, '108pd/105pd')
        self.compare(correct, corrected)

        result = isopy.tb.make_ms_beams('pd', ru101=0.1, cd111 = 0.01, integrations=None)
        corrected = isopy.tb.mass_independent_correction(result, '108pd/105pd')
        self.compare(correct, corrected)

        result = isopy.tb.make_ms_sample('pd', ru101=0.1, cd111 = 0.01, integrations=None)
        corrected = isopy.tb.mass_independent_correction(result, '108pd/105pd')
        self.compare(correct, corrected)

        result = isopy.tb.make_ms_array('pd', ru101=0.1, cd111=0.01, ru99=0.1)
        corrected = isopy.tb.mass_independent_correction(result, '108pd/105pd')
        self.compare(correct, corrected)

        result = isopy.tb.make_ms_beams('pd', ru101=0.1, cd111=0.01, ru99=0.1, cd112=0, integrations=None)
        corrected = isopy.tb.mass_independent_correction(result, '108pd/105pd')
        self.compare(correct, corrected)

        result = isopy.tb.make_ms_sample('pd', ru101=0.1, cd111=0.01, ru99=0.1, cd112=0, integrations=None)
        corrected = isopy.tb.mass_independent_correction(result, '108pd/105pd')
        self.compare(correct, corrected)

        # Different default values
        mass_ref = isopy.refval.isotope.mass_number
        fraction_ref = isopy.refval.isotope.initial_solar_system_fraction_L09

        correct = isopy.ones(None, keys='102pd 104pd 105pd 106pd 110pd'.split())
        correct = correct * fraction_ref
        correct = correct.ratio('105pd')

        result = isopy.tb.make_ms_array('pd', 'ru', 'cd', isotope_masses=mass_ref, isotope_fractions=fraction_ref)
        corrected = isopy.tb.mass_independent_correction(result, '108pd/105pd', isotope_masses=mass_ref, isotope_fractions=fraction_ref)
        self.compare(correct, corrected)

        result = isopy.tb.make_ms_beams('pd', 'ru', 'cd', integrations=None, isotope_masses=mass_ref, isotope_fractions=fraction_ref)
        corrected = isopy.tb.mass_independent_correction(result, '108pd/105pd', isotope_masses=mass_ref, isotope_fractions=fraction_ref)
        self.compare(correct, corrected)

        result = isopy.tb.make_ms_sample('pd', ru=1, cd=1, integrations=None, isotope_masses=mass_ref, isotope_fractions=fraction_ref)
        corrected = isopy.tb.mass_independent_correction(result, '108pd/105pd', isotope_masses=mass_ref, isotope_fractions=fraction_ref)
        self.compare(correct, corrected)

        result = isopy.tb.make_ms_array('pd', ru101=0.1, cd111=0.01, isotope_masses=mass_ref, isotope_fractions=fraction_ref)
        corrected = isopy.tb.mass_independent_correction(result, '108pd/105pd', isotope_masses=mass_ref, isotope_fractions=fraction_ref)
        self.compare(correct, corrected)

        result = isopy.tb.make_ms_beams('pd', ru101=0.1, cd111=0.01, integrations=None, isotope_masses=mass_ref, isotope_fractions=fraction_ref)
        corrected = isopy.tb.mass_independent_correction(result, '108pd/105pd', isotope_masses=mass_ref, isotope_fractions=fraction_ref)
        self.compare(correct, corrected)

        result = isopy.tb.make_ms_sample('pd', ru101=0.1, cd111=0.01, integrations=None, isotope_masses=mass_ref, isotope_fractions=fraction_ref)
        corrected = isopy.tb.mass_independent_correction(result, '108pd/105pd', isotope_masses=mass_ref, isotope_fractions=fraction_ref)
        self.compare(correct, corrected)

    def test_make_array3(self):
        correct = isopy.tb.make_ms_array('pd', 'ru', 'cd').normalise(10, isopy.keymax)
        result = isopy.tb.make_ms_beams('pd', 'ru', 'cd', random_seed=46)
        self.compare_sd(correct, 100, result)
        result = isopy.tb.make_ms_sample('pd', ru=1, cd=1, random_seed=46)
        self.compare_sd(correct, 100, result)

        correct = isopy.tb.make_ms_array('pd', ru101=0.1, cd111=0.01).normalise(10, isopy.keymax)
        result = isopy.tb.make_ms_beams('pd', ru101=0.1, cd111=0.01, random_seed=46)
        self.compare_sd(correct, 100, result)
        result = isopy.tb.make_ms_sample('pd', ru101=0.1, cd111=0.01, random_seed=46)
        self.compare_sd(correct, 100, result)

        correct = isopy.tb.make_ms_array('pd', ru101=0.1, cd111=0.01, ru99=0.1, cd112=0).normalise(10, isopy.keymax)
        result = isopy.tb.make_ms_beams('pd', ru101=0.1, cd111=0.01, ru99=0.1, cd112=0, random_seed=46)
        self.compare_sd(correct, 100, result)
        result = isopy.tb.make_ms_sample('pd', ru101=0.1, cd111=0.01, ru99=0.1, cd112=0, random_seed=46)
        self.compare_sd(correct, 100, result)

        # Integrations
        correct = isopy.tb.make_ms_array('pd', ru101=0.1, cd111=0.01).normalise(10, isopy.keymax)
        result = isopy.tb.make_ms_beams('pd', ru101=0.1, cd111=0.01, random_seed=46, integrations=200)
        self.compare_sd(correct, 200, result)
        result = isopy.tb.make_ms_sample('pd', ru101=0.1, cd111=0.01, random_seed=46, integrations=200)
        self.compare_sd(correct, 200, result)

        # Fixed Key & Value
        correct = isopy.tb.make_ms_array('pd', ru101=0.1, cd111=0.01).normalise(10)
        result = isopy.tb.make_ms_beams('pd', ru101=0.1, cd111=0.01, random_seed=46,
                                        integrations=200, fixed_key=None)
        self.compare_sd(correct, 200, result)
        result = isopy.tb.make_ms_sample('pd', ru101=0.1, cd111=0.01, random_seed=46,
                                         integrations=200, fixed_key=None)
        self.compare_sd(correct, 200, result)

        correct = isopy.tb.make_ms_array('pd', ru101=0.1, cd111=0.01).normalise(1)
        result = isopy.tb.make_ms_beams('pd', ru101=0.1, cd111=0.01, random_seed=46,
                                        integrations=200, fixed_key=None, fixed_voltage=1)
        self.compare_sd(correct, 200, result)
        result = isopy.tb.make_ms_sample('pd', ru101=0.1, cd111=0.01, random_seed=46,
                                         integrations=200, fixed_key=None, fixed_voltage=1)
        self.compare_sd(correct, 200, result)

        correct = isopy.tb.make_ms_array('pd', ru101=0.1, cd111=0.01).normalise(10, '102pd')
        result = isopy.tb.make_ms_beams('pd', ru101=0.1, cd111=0.01, random_seed=46,
                                        integrations=200, fixed_key='102pd')
        self.compare_sd(correct, 200, result)
        result = isopy.tb.make_ms_sample('pd', ru101=0.1, cd111=0.01, random_seed=46,
                                         integrations=200, fixed_key='102pd')
        self.compare_sd(correct, 200, result)

        correct = isopy.tb.make_ms_array('pd', ru101=0.1, cd111=0.01).normalise(5, '102pd')
        result = isopy.tb.make_ms_beams('pd', ru101=0.1, cd111=0.01, random_seed=46,
                                        integrations=200, fixed_key='102pd', fixed_voltage=5)
        self.compare_sd(correct, 200, result)
        result = isopy.tb.make_ms_sample('pd', ru101=0.1, cd111=0.01, random_seed=46,
                                         integrations=200, fixed_key='102pd', fixed_voltage=5)
        self.compare_sd(correct, 200, result)

        correct = isopy.tb.make_ms_array('pd', ru101=0.1, cd111=0.01).normalise(10, ('102pd', '104pd'))
        result = isopy.tb.make_ms_beams('pd', ru101=0.1, cd111=0.01, random_seed=46,
                                        integrations=200, fixed_key=('102pd', '104pd'))
        self.compare_sd(correct, 200, result)
        result = isopy.tb.make_ms_sample('pd', ru101=0.1, cd111=0.01, random_seed=46,
                                         integrations=200, fixed_key=('102pd', '104pd'))
        self.compare_sd(correct, 200, result)

        correct = isopy.tb.make_ms_array('pd', ru101=0.1, cd111=0.01).normalise(100,
                                                                                ('102pd', '104pd'))
        result = isopy.tb.make_ms_beams('pd', ru101=0.1, cd111=0.01, random_seed=46,
                                        integrations=200, fixed_key=('102pd', '104pd'), fixed_voltage=100)
        self.compare_sd(correct, 200, result)
        result = isopy.tb.make_ms_sample('pd', ru101=0.1, cd111=0.01, random_seed=46,
                                         integrations=200, fixed_key=('102pd', '104pd'), fixed_voltage=100)
        self.compare_sd(correct, 200, result)

    def test_spike(self):
        spike = isopy.array(pd104 = 1, pd106=0, pd108=1, pd110=0)
        spike = spike.normalise(1)
        sample = isopy.refval.isotope.fraction.asarray(element_symbol='pd')
        sample = sample.normalise(1, spike.keys)

        correct = isopy.add(sample * 0.5, spike * 0.5, 0)
        correct = correct.normalise(10, isopy.keymax)
        result = isopy.tb.make_ms_sample('pd', spike=spike, integrations=None)
        self.compare(correct, result)
        result = isopy.tb.make_ms_sample('pd', spike=spike)
        self.compare_sd(correct, 100, result)

        correct = isopy.add(sample * 0.1, spike * 0.9, 0)
        correct = correct.normalise(10, isopy.keymax)
        result = isopy.tb.make_ms_sample('pd', spike=spike, integrations=None, spike_fraction=0.9)
        self.compare(correct, result)
        result = isopy.tb.make_ms_sample('pd', spike=spike, spike_fraction=0.9)
        self.compare_sd(correct, 100, result)

    def test_blank(self):
        sample = isopy.refval.isotope.fraction.asarray(element_symbol='pd')
        blank = isopy.ones(None, sample.keys)
        blank = blank + isopy.refval.isotope.fraction
        blank = blank.normalise(1)

        blank2 = blank.normalise(0.01, '106pd')
        correct = sample.normalise(10-0.01, '106pd')
        correct = correct + blank2
        result = isopy.tb.make_ms_sample('pd', blank=blank, integrations=None)
        self.compare(correct, result)
        result = isopy.tb.make_ms_sample('pd', blank=blank)
        self.compare_sd(correct, 100, result)

        blank2 = blank.normalise(0.1, '106pd')
        correct = sample.normalise(10 - 0.1, '106pd')
        correct = correct + blank2
        result = isopy.tb.make_ms_sample('pd', blank=blank, blank_fixed_voltage=0.1, integrations=None)
        self.compare(correct, result)
        result = isopy.tb.make_ms_sample('pd', blank=blank, blank_fixed_voltage=0.1)
        self.compare_sd(correct, 100, result)

        blank2 = blank.normalise(0.1, '102pd')
        correct = sample.normalise(10 - blank2['106pd'], '106pd')
        correct = correct + blank2
        result = isopy.tb.make_ms_sample('pd', blank=blank, blank_fixed_voltage=0.1,
                                         blank_fixed_key='102pd', integrations=None)
        self.compare(correct, result)
        result = isopy.tb.make_ms_sample('pd', blank=blank, blank_fixed_voltage=0.1,
                                         blank_fixed_key='102pd')
        self.compare_sd(correct, 100, result)

        blank2 = blank.normalise(0.1, ('102pd', '104pd'))
        correct = sample.normalise(10 - blank2['106pd'], '106pd')
        correct = correct + blank2
        result = isopy.tb.make_ms_sample('pd', blank=blank, blank_fixed_voltage=0.1,
                                         blank_fixed_key=('102pd', '104pd'), integrations=None)
        self.compare(correct, result)
        result = isopy.tb.make_ms_sample('pd', blank=blank, blank_fixed_voltage=0.1,
                                         blank_fixed_key=('102pd', '104pd'))
        self.compare_sd(correct, 100, result)

        blank2 = blank.normalise(0.01, '106pd')
        correct = sample.normalise(10 - blank2['102pd'], '102pd')
        correct = correct + blank2
        result = isopy.tb.make_ms_sample('pd', blank=blank, integrations=None, fixed_key='102pd')
        self.compare(correct, result)
        result = isopy.tb.make_ms_sample('pd', blank=blank, fixed_key='102pd')
        self.compare_sd(correct, 100, result)

        blank2 = blank.normalise(0.01, '106pd')
        correct = sample.normalise(10 - blank2[('102pd', '104pd')].sum(axis=None), ('102pd', '104pd'))
        correct = correct + blank2
        result = isopy.tb.make_ms_sample('pd', blank=blank, integrations=None, fixed_key=('102pd', '104pd'))
        self.compare(correct, result)
        result = isopy.tb.make_ms_sample('pd', blank=blank, fixed_key=('102pd', '104pd'))
        self.compare_sd(correct, 100, result)

        blank2 = blank.normalise(0.01, '106pd')
        correct = sample.normalise(10 - blank2.sum(axis=None), None)
        correct = correct + blank2
        result = isopy.tb.make_ms_sample('pd', blank=blank, integrations=None, fixed_key=None)
        self.compare(correct, result)
        result = isopy.tb.make_ms_sample('pd', blank=blank, fixed_key=None)
        self.compare_sd(correct, 100, result)

    def compare(self, correct, result):
        assert result.keys == correct.keys
        assert result.size == correct.size
        assert result.ndim == correct.ndim
        for key in result.keys:
            np.testing.assert_allclose(result[key], correct[key])

    def compare_sd(self, correct, integrations, result):
        assert result.keys == correct.keys
        assert result.size == integrations
        assert result.ndim == 1
        for key in result.keys:
            np.testing.assert_allclose(np.mean(result[key]), correct[key], rtol=0, atol=isopy.sd(result[key]))


class Test_JohnsonNyquistNoise:
    def test_one(self):
        self.run(10)
        self.run(10, 1E12)
        self.run(10, time=4.1)
        self.run(10, T=400)
        self.run(10, 1E10)
        self.run(10, cpv=1E7)

        voltages = isopy.refval.isotope.fraction.asarray(element_symbol='pd').normalise(10, isopy.keymax)
        resistors = isopy.full(None, 1E11, voltages.keys)
        resistors['102pd'] = 1E13
        resistors['106pd'] = 1E10
        self.run(voltages)
        self.run(voltages, 1E12)
        self.run(voltages, resistors)
        self.run(voltages, time=4.1)
        self.run(voltages, T=400)
        self.run(voltages, 1E10)
        self.run(voltages, cpv=1E7)

    def test2(self):
        Os187 = [0.000052, 0.000522, 0.001044, 0.002088, 0.003132, 0.004176, 0.005220, 0.007830, 0.010439, 0.026099, 0.052197]
        Os188 = [0.000324, 0.003244, 0.006487, 0.012974, 0.019462, 0.025949, 0.032436, 0.048654, 0.064872, 0.162180, 0.324359]
        jk_correct = [0.0140985, 0.0014082, 0.0007042, 0.0003521, 0.0002347, 0.0001760, 0.0001408, 0.0000939, 0.0000704, 0.0000282, 0.0000141]
        combined_correct = [0.0141373, 0.0014467, 0.0007422, 0.0003892, 0.0002709, 0.0002115, 0.0001756, 0.0001270, 0.0001022, 0.0000547, 0.0000360]

        Os_data = isopy.array(os187=Os187, os188=Os188)
        jk_result = isopy.tb.johnson_nyquist_noise(Os_data, 1E12, include_counting_statistics=False)

        np.testing.assert_allclose(jk_result, 0.00004510199454/10)

        ratio = Os_data['187Os'] / Os_data['188Os']
        jk_result2 = np.square(jk_result) * np.square(1/Os_data['188Os'])
        jk_result2 += np.square(jk_result) * np.square(Os_data['187Os'] / np.square(Os_data['188Os']))
        jk_result2 = np.sqrt(jk_result2)
        jk_result2 = np.round(jk_result2,7)

        np.testing.assert_allclose(jk_result2, jk_correct)

        combined_result = isopy.tb.johnson_nyquist_noise(Os_data, 1E12)
        combined_result2 = np.square(combined_result['187Os']) * np.square(1/Os_data['188Os'])
        combined_result2 += np.square(combined_result['188Os']) * np.square(Os_data['187Os'] / np.square(Os_data['188Os']))
        combined_result2 = np.sqrt(combined_result2)
        combined_result2 = np.round(combined_result2, 7)
        np.testing.assert_allclose(combined_result2, combined_correct)

        #Different resistors
        Os187 = [0.000052, 0.000522, 0.001044, 0.002088, 0.003132, 0.004176, 0.005220, 0.007830,
                 0.010439, 0.026099, 0.052197]

        Os188 = [0.000324, 0.003244, 0.006487, 0.012974, 0.019462, 0.025949, 0.032436, 0.048654,
                 0.064872, 0.162180, 0.324359]
        jk_correct = [0.0083241, 0.0008329, 0.0004166, 0.0002083, 0.0001388, 0.0001041, 0.0000833, 0.0000555, 0.0000417, 0.0000167, 0.0000083]
        combined_correct = [0.0083897, 0.0008965, 0.0004780, 0.0002662, 0.0001939, 0.0001568, 0.0001339, 0.0001020, 0.0000850, 0.0000497, 0.0000342]

        Os_data = isopy.array(os187=Os187, os188=Os188)
        jk_result = isopy.tb.johnson_nyquist_noise(Os_data, {'187Os': 1E13}, include_counting_statistics=False)

        jk_result2 = np.square(jk_result['187Os']) * np.square(1 / Os_data['188Os'])
        jk_result2 += np.square(jk_result['188Os']) * np.square(
            Os_data['187Os'] / np.square(Os_data['188Os']))
        jk_result2 = np.sqrt(jk_result2)
        jk_result2 = np.round(jk_result2, 7)

        np.testing.assert_allclose(jk_result2, jk_correct)

        combined_result = isopy.tb.johnson_nyquist_noise(Os_data, {'187Os': 1E13})
        combined_result2 = np.square(combined_result['187Os']) * np.square(1 / Os_data['188Os'])
        combined_result2 += np.square(combined_result['188Os']) * np.square(
            Os_data['187Os'] / np.square(Os_data['188Os']))
        combined_result2 = np.sqrt(combined_result2)
        combined_result2 = np.round(combined_result2, 7)
        np.testing.assert_allclose(combined_result2, combined_correct)



    # This just verifies the equation written in the documentation
    # Which I got from dissecting the spreadsheet of the paper
    def run(self, voltage, resistor=None, time=None, T = None, R = None, cpv = None):
        kwargs = {}

        if resistor is None: resistor = 1E11
        else: kwargs['resistor'] = resistor

        if time is None: time = 8.389
        else: kwargs['integration_time'] = time
        if T is None: T = 309
        else: kwargs['T'] = T
        if R is None: R = 1E11
        else: kwargs['R'] = R
        if cpv is None: cpv = 6.25E7
        else: kwargs['cpv'] = cpv
        kb = np.float64(1.3806488E-23)

        jk_correct = (4 * kb * T * resistor) / time
        jk_correct = np.sqrt(jk_correct)
        jk_correct = jk_correct * (R / resistor)

        cs = 1 / (voltage * cpv * time)
        cs = np.sqrt(cs) * voltage

        combined_correct = np.sqrt(np.square(jk_correct) + np.square(cs))

        jk_result = isopy.tb.johnson_nyquist_noise(voltage, **kwargs, include_counting_statistics=False)
        combined_result = isopy.tb.johnson_nyquist_noise(voltage, **kwargs)

        if isinstance(resistor, isopy.core.IsopyArray):
            assert jk_result.keys == jk_correct.keys
            assert jk_result.size == jk_correct.size
            assert jk_result.ndim == jk_correct.ndim
            for key in jk_result.keys:
                np.testing.assert_allclose(jk_result[key], jk_correct[key])
        else:
            assert not isinstance(jk_result, isopy.core.IsopyArray)
            np.testing.assert_allclose(jk_result, jk_correct)

        if isinstance(voltage, isopy.core.IsopyArray):
            assert combined_result.keys == combined_correct.keys
            assert combined_result.size == combined_correct.size
            assert combined_result.ndim == combined_correct.ndim
            for key in combined_result.keys:
                np.testing.assert_allclose(combined_result[key], combined_correct[key])
        else:
            assert not isinstance(combined_result, isopy.core.IsopyArray)
            np.testing.assert_allclose(combined_result, combined_correct)




















