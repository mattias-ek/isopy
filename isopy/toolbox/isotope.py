import numpy as np
import isopy as isopy
from isopy import core

__all__ = ['remove_mass_fractionation', 'add_mass_fractionation',
           'calculate_mass_fractionation_factor', 'mass_independent_correction',
           'remove_isobaric_interferences',
           'make_ms_array', 'make_ms_beams', 'make_ms_sample', 'johnson_nyquist_noise',
           'normalise_data', 'denormalise_data', 'find_outliers']

import isopy.checks
import isopy.core

"""
Functions for isotope data reduction
"""

def johnson_nyquist_noise(voltage, resistor = 1E11, integration_time = 8.389, include_counting_statistics = True,
                          T = 309, R = 1E11, cpv = 6.25E7):
    """
    Calculate the Johnson-Nyquist noise and counting statistics for a given voltage.

    The Johnson-Nyquist noise (:math:`n_{jn}` is calculated as:

    .. math::
        n_{jn} = \\sqrt{ \\frac{4*k_{b}*T*r} {t} } * \\frac{R} {r}

    The counting statistics, or shot noise, (:math:`n_{cs}` is calculated as:

    .. math::
        n_{cs} = \\sqrt{ \\frac{1} {v * c_{V} * t}} * v

    The two are combined as:

    .. math::
        n_{all} = \\sqrt{ (n_{jn})^2 + (n_{cs})^2 }

    The noise for a specific isotope ratio can be calculated as:

    .. math::
        n_{n,d} = \\sqrt{ (n_{n})^2 + (n_{d})^2 }

    where :math:`n` is the numerator isotope and :math:`d` is the denominator isotope.

    Adapted from the equations in `Liu & Pearson (2014) Chemical Geology, 10, 301-311
    <https://doi.org/10.1016/j.chemgeo.2013.11.008>`_.

    Parameters
    ----------
    voltage : IsotopeArray, float, np.ndarray
        The measured voltages. :math:`v` in the equations above.
    resistor : IsotopeArray, float, np.ndarray, dict
        The resistor for the measurement. Default value is ``1E11``. :math:`r` in the equations above.
        If *resistor* is a dictionary ``1E11`` will be used for values not in the dictionary.
    integration_time : float, optional
        The integration time in seconds for a single measurement. Default value is ``8.389``.
        :math:`t` in the equations above.
    include_counting_statistics: bool, optional
        If ``True`` then the counting statistics are included in the returned value. Default value is ``True``.
    T : float, optional
        Amplifier housing temperature in kelvin. Default value is ``309``. :math:`T` in the equations above.
    R : float, optional
        *voltage* values are reported as volts for this resistor value. Default value is
        ``1E11``. :math:`R` in the equations above.
    cpv : float, optional
        Counts per volt per second. Default value is ``6.25E7``. :math:`C_{V}` in the equations above.

    Returns
    -------
    noise: np.float or np.ndarray or IsotopeArray
        The noise in V for the given voltage/set of voltages.

    Examples
    --------
    >>> isopy.tb.johnson_nyquist_noise(10) * 1000 #in millivolts
    0.13883808575503015
    >>> isopy.tb.johnson_nyquist_noise(10, 1E10) * 1000
    0.14528174343845432

    >>> array = isopy.tb.make_ms_array('pd')
    >>> array = array * (10 / array['106pd']) #10v on the largest isotope
    >>> isopy.tb.johnson_nyquist_noise(array) * 1000 #in millivolts
    (row) , 102Pd   , 104Pd   , 105Pd   , 106Pd   , 108Pd   , 110Pd
    None  , 0.03025 , 0.08932 , 0.12565 , 0.13884 , 0.13663 , 0.09156
    >>> resistors = dict(pd102=1E13, pd106=1E10) #1E11 is used for missing keys
    >>> isopy.tb.johnson_nyquist_noise(array, resistors) * 1000
    (row) , 102Pd   , 104Pd   , 105Pd   , 106Pd   , 108Pd   , 110Pd
    None  , 0.02672 , 0.08932 , 0.12565 , 0.14528 , 0.13663 , 0.09156
    """

    voltage = isopy.checks.check_type('voltage', voltage, isopy.core.IsotopeArray, np.ndarray, np.float, coerce=True,
                                      coerce_into=[isopy.core.IsotopeArray, np.float, np.array])
    resistor = isopy.checks.check_type('resistor', resistor, isopy.core.IsotopeArray, np.ndarray, dict, coerce=True,
                                       coerce_into=[isopy.core.IsotopeArray, np.float, np.array])
    integration_time = isopy.checks.check_type('integration_time', integration_time, np.float, coerce=True)
    include_counting_statistics = isopy.checks.check_type('include_counting_statistics', include_counting_statistics,
                                                          bool)
    if isinstance(resistor, dict):
        resistor = core.IsopyDict(resistor, default_value=1E11)
        resistor = isopy.IsotopeArray({key: resistor.get(key) for key in voltage.keys})

    T = isopy.checks.check_type('T', T, np.float, coerce=True)
    R = isopy.checks.check_type('R', R, np.float, coerce=True)
    cpv = isopy.checks.check_type('cpv', cpv, np.float, coerce=True)

    kB = np.float(1.3806488E-023) # Boltsman constant

    t_noise = np.sqrt((4 * kB * T * resistor) / integration_time) * (R / resistor)

    if include_counting_statistics:
        c_stat = np.sqrt(1 / (voltage * cpv * integration_time)) * voltage
        return np.sqrt(c_stat ** 2 + t_noise ** 2)
    else:
        return t_noise

def make_ms_array(*args, mf_factor = None, isotope_abundances = None, isotope_masses=None, **kwargs):
    """
    Constructs an isotope array where the first created isotope contains the sum of all given
    isotopes with the same mass number. The construction order is first *args* and then *kwargs*.

    Parameters
    ----------
    args
        Each arg can be either a isotope array, a element key or an isotope key string.
        If arg is an isotope array an exception will be raised if two or more isotopes have the
        same mass number. If arg is an element key string then all isotopes of that element are
        added to the array. If arg is an isotope key string then only that isotope and isotopes
        with the same mass number as any other isotope in the array is added.
    mf_factor
        If given this mass fractionation factor is applied to all isotopes of an element
        prior to being added to the result.
    isotope_abundances
        Reference value for the isotope fractions of different elements.
        Defaults to ``isopy.refv.isotope.fraction``.
    isotope_masses
        Reference value for the isotope masses of different elements.
        Defaults to ``isopy.refv.isotope.mass``.
    kwargs
        Each kwarg key must be either an element key string or an isotope key string. The isotope
        fractions added to the array based on the kwarg key are multiplied by the kwarg value.
        Otherwise behaves the same as for *args*.

    Returns
    -------
    result : IsotopeArray
        The constructed isotope array.

    Examples
    --------
    >>> isopy.tb.make_ms_array('pd')
    (row) , 102Pd   , 104Pd   , 105Pd   , 106Pd   , 108Pd   , 110Pd
    None  , 0.01020 , 0.11140 , 0.22330 , 0.27330 , 0.26460 , 0.11720

    >>> isopy.tb.make_ms_array('pd', ru101=0.1)
    (row) , 101Ru   , 102Pd   , 104Pd   , 105Pd   , 106Pd   , 108Pd   , 110Pd
    None  , 0.01706 , 0.04175 , 0.13002 , 0.22330 , 0.27330 , 0.26460 , 0.11720

    >>> isopy.tb.make_ms_array('pd', ru101=0.1, ru99=0) #99Ru doesnt contribute anything to the array but gets a contribution from 101Ru.
    (row) , 99Ru    , 101Ru   , 102Pd   , 104Pd   , 105Pd   , 106Pd   , 108Pd   , 110Pd
    None  , 0.01276 , 0.01706 , 0.04175 , 0.13002 , 0.22330 , 0.27330 , 0.26460 , 0.11720

    >>> isopy.tb.make_ms_array('pd', 'cd')
    102Pd  , 104Pd  , 105Pd  , 106Pd   , 108Pd  , 110Pd   , 111Cd   , 112Cd   , 113Cd   , 114Cd   , 116Cd
    0.0102 , 0.1114 , 0.2233 , 0.28579 , 0.2735 , 0.24205 , 0.12804 , 0.24117 , 0.12225 , 0.28729 , 0.07501

    >>> isopy.tb.make_ms_array('pd', cd=1) #Same as example above
    (row) , 102Pd   , 104Pd   , 105Pd   , 106Pd   , 108Pd   , 110Pd   , 111Cd   , 112Cd   , 113Cd   , 114Cd   , 116Cd
    None  , 0.01020 , 0.11140 , 0.22330 , 0.28579 , 0.27350 , 0.24205 , 0.12804 , 0.24117 , 0.12225 , 0.28729 , 0.07501

    See Also
    --------
    make_ms_beams, make_ms_sample
    """

    isotope_abundances = isopy.checks.check_reference_value('isotope_abundances', isotope_abundances,
                                                           isopy.refval.isotope.abundance)
    isotope_masses = isopy.checks.check_reference_value('isotope_masses', isotope_masses,
                                                        isopy.refval.isotope.mass)

    input = []
    for key in args:
        input.append((key, 1))
    for item in kwargs.items():
        input.append(item)


    isotope_keys = isopy.IsotopeKeyList()
    for key, val in input:
        if isinstance(key, core.IsopyArray):
            keys = key.keys()
        else:
            key = isopy.askeystring(key)
            if isinstance(key, isopy.ElementKeyString):
                keys = isopy.refval.element.isotopes[key]
            elif isinstance(key, isopy.IsotopeKeyString):
                keys = isopy.IsotopeKeyList(key)
            else:
                raise ValueError('args must be an element or isotope key string')

        isotope_keys += keys.filter(mass_number_neq = isotope_keys.mass_numbers)

    isotope_keys = isotope_keys.sorted()
    mass_keys = isotope_keys.mass_numbers
    mass_array = isopy.zeros(None, mass_keys)
    for key, val in input:
        if isinstance(key, core.IsopyArray):
            array = key
            keys = array.keys()
        else:
            key = isopy.askeystring(key)
            if isinstance(key, isopy.ElementKeyString):
                element = key
            else:
                element = key.element_symbol

            keys = isopy.refval.element.isotopes[element].filter(mass_number_eq=mass_keys)
            array = isopy.ones(None, keys) * isotope_abundances * val

        if mf_factor is not None:
            array = add_mass_fractionation(array, mf_factor, isotope_masses=isotope_masses)

        mass_array = isopy.add(mass_array, isopy.array(array, keys.mass_numbers), default_value = 0)

    return isopy.IsotopeArray(mass_array, isotope_keys)


def make_ms_beams(*args, mf_factor=None, maxv = 10, integrations = 100, integration_time=8.389, resistor=1E11,
                  random_seed = None, isotope_abundances=None, isotope_masses=None, **kwargs):
    """
    Simulates a series of measurements with a standard deviation equal to the johnson-nyquist noise
    and counting statistics.

    *args* and *kwargs* are passed to ``isopy.tb.make_ms_array()`` to create the ms array.

    Parameters
    ----------
    args
        Each arg can be either a isotope array, a element key or an isotope key string.
        If arg is an isotope array an exception will be raised if two or more isotopes have the
        same mass number. If arg is an element key string then all isotopes of that element are
        added to the array. If arg is an isotope key string then only that isotope and isotopes
        with the same mass number as any other isotope in the array is added.
    mf_factor
        If given this mass fractionation factor is applied to all isotopes of an element
        prior to being added to the result.
    maxv
        The voltage of the most abundant isotope in the array. The value for all other isotopes in
        the array are adjusted accordingly.
    integrations
        The number of simulated measurements. If ``None`` no measurements are simulated and the
        retured array contains the true values.
    integration_time
        The integration time for each simulated measurement.
    resistor
        The resistor used for each measurement. A isotope array or a dictionary can be passed to
        to give different resistor values for different isotopes.
    random_seed
        Must be an integer. Seed given to the random generator.
    isotope_abundances
        Reference value for the isotope fractions of different elements.
        Defaults to ``isopy.refv.isotope.fraction``.
    isotope_masses
        Reference value for the isotope masses of different elements.
        Defaults to ``isopy.refv.isotope.mass``.
    kwargs
        Each kwarg key must be either an element key string or an isotope key string. The isotope
        fractions added to the array based on the kwarg key are multiplied by the kwarg value.
        Otherwise behaves the same as for *args*.

    Returns
    -------
    IsotopeArray
        The simulated measurements.

    Examples
    --------
    >>> isopy.tb.make_ms_beams('pd', ru101=0.01).pprint(nrows=10)
    (row) , 101Ru   , 102Pd   , 104Pd   , 105Pd   , 106Pd    , 108Pd   , 110Pd
    0     , 0.06243 , 0.48869 , 4.14431 , 8.17035 , 9.99984  , 9.68171 , 4.28824
    1     , 0.06241 , 0.48865 , 4.14423 , 8.17033 , 9.99990  , 9.68189 , 4.28823
    2     , 0.06245 , 0.48872 , 4.14423 , 8.17058 , 9.99992  , 9.68178 , 4.28837
    3     , 0.06241 , 0.48868 , 4.14430 , 8.17042 , 10.00005 , 9.68161 , 4.28839
    4     , 0.06240 , 0.48870 , 4.14417 , 8.17053 , 10.00018 , 9.68180 , 4.28834
    ...   , ...     , ...     , ...     , ...     , ...      , ...     , ...
    95    , 0.06244 , 0.48866 , 4.14432 , 8.17056 , 10.00010 , 9.68187 , 4.28827
    96    , 0.06243 , 0.48870 , 4.14410 , 8.17044 , 10.00014 , 9.68184 , 4.28848
    97    , 0.06238 , 0.48867 , 4.14404 , 8.17048 , 9.99992  , 9.68159 , 4.28840
    98    , 0.06245 , 0.48871 , 4.14434 , 8.17031 , 9.99998  , 9.68169 , 4.28829
    99    , 0.06243 , 0.48863 , 4.14438 , 8.17047 , 10.00003 , 9.68145 , 4.28832

    See Also
    --------
    make_ms_array, make_ms_sample
    """

    beams = make_ms_array(*args, **kwargs, mf_factor=mf_factor, isotope_masses=isotope_masses, isotope_abundances=isotope_abundances)
    if beams.size != 1:
        raise ValueError('The constructed ms array has a size larger than one')

    if maxv is not None:
        beams = beams * (maxv / np.max(beams, axis=None))

    noise = johnson_nyquist_noise(beams, integration_time=integration_time, resistor=resistor)
    rng = np.random.default_rng(random_seed)

    if integrations is None:
        return beams
    else:
        return isopy.array({key: rng.normal(beams[key], noise[key], integrations) for key in beams.keys()})


def make_ms_sample(ms_array, *, fnat = None, fins = None,  maxv = 10,
                   spike_mixture = None, spike_fraction = 0.5,
                   integrations = 100, integration_time = 8.389, resistor = 1E11,
                   random_seed = None,
                   isotope_abundances=None, isotope_masses=None, **interferences):
    """
    Creates a simulated the measurement of a sample with natural and instrumental mass
    fractionation added to the array. The standard deviation of measurements for each isotope
    is equal to the johnson-nyquist noise and counting statistics.

    Parameters
    ----------
    ms_array
        Any object that can be singularly passed to ``make_ms_array`` which returns valid array.
    fnat
        If given, the natural fractionation fractionation factor is applied to the ms_array
        before *interferences* are added to the ms_array.
    fins
        If given, the instrumental mass fractionation factor is applied to the ms_array
        at the same time the *interferences* are added to the ms_array.
    spike_mixture
        If given this spike mixture will be added ms_array after *fnat* but before *fins* and
        *interferences* are added to the array.
    spike_fraction
        The fraction of spike in the final mixture based on the isotopes in *spike_mixture*.
    maxv
        The voltage of the most abundant isotope in the array. The value for all other isotopes in
        the array are adjusted accordingly.
    integrations
        The number of simulated measurements. If ``None`` no measurements are simulated and the
        retured array contains the true values.
    integration_time
        The integration time for each simulated measurement.
    resistor
        The resistor used for each measurement. A isotope array or a dictionary can be passed to
        to give different resistor values for different isotopes.
    random_seed
        Must be an integer. Seed given to the random generator.
    isotope_abundances
        Reference value for the isotope fractions of different elements.
        Defaults to ``isopy.refv.isotope.fraction``.
    isotope_masses
        Reference value for the isotope masses of different elements.
        Defaults to ``isopy.refv.isotope.mass``.
    interferences
        Each kwarg key must be either an element key string or an isotope key string. The isotope
        fractions added to the array based on the kwarg key are multiplied by the kwarg value.
        Otherwise behaves the same as for *args*.

    Returns
    -------
    IsotopeArray
        The simulated measurements.

    Examples
    --------
    >>> isopy.tb.make_ms_sample('pd', ru101=0.01, cd111=0.001, fnat = 0.1, fins=-1.6).pprint(nrows=10)
    (row) , 101Ru   , 102Pd   , 104Pd   , 105Pd   , 106Pd    , 108Pd   , 110Pd   , 111Cd
    0     , 0.06343 , 0.51076 , 4.26008 , 8.28716 , 9.99964  , 9.41352 , 4.06078 , 0.00468
    1     , 0.06340 , 0.51082 , 4.26016 , 8.28714 , 10.00006 , 9.41351 , 4.06064 , 0.00466
    2     , 0.06340 , 0.51083 , 4.26006 , 8.28692 , 9.99981  , 9.41356 , 4.06047 , 0.00470
    3     , 0.06339 , 0.51079 , 4.26018 , 8.28711 , 10.00002 , 9.41349 , 4.06073 , 0.00471
    4     , 0.06341 , 0.51087 , 4.26022 , 8.28713 , 9.99986  , 9.41357 , 4.06062 , 0.00465
    ...   , ...     , ...     , ...     , ...     , ...      , ...     , ...     , ...
    95    , 0.06341 , 0.51080 , 4.26010 , 8.28729 , 9.99992  , 9.41332 , 4.06070 , 0.00465
    96    , 0.06339 , 0.51082 , 4.25994 , 8.28712 , 10.00012 , 9.41364 , 4.06071 , 0.00467
    97    , 0.06339 , 0.51083 , 4.26012 , 8.28728 , 9.99976  , 9.41363 , 4.06069 , 0.00468
    98    , 0.06341 , 0.51078 , 4.26010 , 8.28700 , 9.99994  , 9.41348 , 4.06065 , 0.00469
    99    , 0.06339 , 0.51082 , 4.26010 , 8.28705 , 9.99997  , 9.41386 , 4.06072 , 0.00468


    See Also
    --------
    make_ms_array, make_ms_beams
    """
    ms_array = make_ms_array(ms_array, mf_factor = fnat,
                          isotope_abundances=isotope_abundances, isotope_masses=isotope_masses)

    if spike_mixture is not None:
        spike_mixture = isopy.asarray(spike_mixture)
        ms_array = make_ms_array(ms_array, isotope_abundances=isotope_abundances, isotope_masses=isotope_masses)

        spsum = np.sum(spike_mixture, axis=1)
        smpsum = np.sum(ms_array.copy(key_eq=spike_mixture.keys()), axis = 1)
        spike_mixture = spike_mixture / spsum * smpsum
        spike_mixture = spike_mixture * (spike_fraction / (1-spike_fraction))
        ms_array = isopy.add(ms_array, spike_mixture, 0)

    ms_array = make_ms_array(ms_array, mf_factor = fins,
                          isotope_abundances=isotope_abundances, isotope_masses=isotope_masses,
                           **interferences)

    ms_array = make_ms_beams(ms_array, maxv=maxv, integrations=integrations,
                          integration_time=integration_time, random_seed=random_seed,
                          resistor=resistor, isotope_abundances=isotope_abundances)

    return ms_array


def mass_independent_correction(data, mf_ratio, normalisation_value = None, normalisation_factor=None,
                                isotope_abundances=None, isotope_masses=None):
    """
    A quick function for mass-independent data correction.

    The data is corrected for mass fractionation, isobaric interferences and finally, if *normalisation_factor* is given,
    normalised to the *isotope_abundances*. If *normalisation_factor* is not given the unnormalised corrected data is
    returned.

    An interference correction will be applied for all isotopes that are different from the *mf_ratio* numerator
    element. This will be done together with the mass fractionation correction to account for isobaric interferences
    on the *mf_ratio*.

    Parameters
    ----------
    data
        The data to be corrected. Can be either an isotope array or a ratio array.
    mf_ratio
        The data will be internally normalised to this ratio. Must be present in data.
    normalisation_value
        If given the result in normalised to this value with *normalisation_factor*. If
        *normalisation_factor* is not given a normalisation factor of 1 is used.
    normalisation_factor
        If given and *normalisation_value* is not the result is normalised against
        *isotope_abundances*.
    isotope_abundances
        Reference value for the isotope fractions of different elements.
        Defaults to ``isopy.refv.isotope.fraction``
    isotope_masses
        Reference value for the isotope masses of different elements.
        Defaults to ``isopy.refv.isotope.mass``

    Returns
    -------
    IsotopeArray or RatioArray
        The corrected data, will be the same flavour *data*. Interference isotope keys will be
        removed from the returned array.
    """
    data = isopy.checks.check_type('data', data, isopy.core.IsotopeArray, isopy.core.RatioArray, coerce=True)
    mf_ratio = isopy.checks.check_type('mf_ratio', mf_ratio, isopy.core.RatioKeyString, coerce=True)
    normalisation_factor = isopy.checks.check_type('normalisation_factor', normalisation_factor, np.float, str, coerce=True, allow_none=True)
    isotope_abundances = isopy.checks.check_reference_value('isotope_abundances', isotope_abundances, isopy.refval.isotope.abundance)
    isotope_masses = isopy.checks.check_reference_value('isotope_masses', isotope_masses, isopy.refval.isotope.mass)


    if isinstance(data, isopy.RatioArray):
        if not data.has_common_denominator:
            raise ValueError('data must hav a common denominator')
        elif data.common_denominator != mf_ratio.denominator:
            raise ValueError('data and mf_ratio do not have the same denominator')
        rat = data
    else:
        rat = data.ratio(mf_ratio.denominator)

    interference_isotopes = _find_isobaric_interferences(data, mf_ratio.numerator.element_symbol)

    #Convert the data into a ratio array
    beta = calculate_mass_fractionation_factor(rat, mf_ratio, isotope_abundances, isotope_masses)

    #Do a combined mass fractionation and isobaric interference correction.
    #This can account for isobaric interferences on isotopes in *mf_ratio*
    for i in range(100):
        rat2 = rat
        prev_beta = beta

        for infiso in interference_isotopes:
            rat2 = remove_isobaric_interferences(rat2, infiso, beta, isotope_abundances, isotope_masses)

        # Calculate the mass fractionation.
        beta = calculate_mass_fractionation_factor(rat2, mf_ratio, isotope_abundances, isotope_masses)

        if np.all(np.abs(beta - prev_beta) < 0.000001):
            break #Beta value has converged so no need for more iterations.
    else:
        raise ValueError('values did not converge after 100 iterations of the interference correction')

    #Remove the isotopes on interfering elements
    rat = rat2.copy(numerator_element_symbol_eq=mf_ratio.numerator.element_symbol)

    #Correct for mass fractionation
    rat = remove_mass_fractionation(rat, beta, isotope_masses)

    if normalisation_value is not None:
        if normalisation_factor is None: normalisation_factor = 1
        rat = normalise_data(rat, normalisation_value, normalisation_factor)

    elif normalisation_factor is not None:
        rat = normalise_data(data, isotope_abundances, normalisation_factor)

    # Return the corrected data
    return rat

def _find_isobaric_interferences(measured, element):
    if isinstance(measured, isopy.RatioArray):
        # Find the isotopes that can cause isobaric interferences.
        interference_isotopes = measured.keys.numerators.filter(numerator_element_symbol_neq=element)

        # Only do the correction for the isotope on interfering element with the biggest beam
        interference_elements = isopy.ElementKeyList()
        for iso in interference_isotopes:  # This preserves the order of the elements
            if iso.numerator.element_symbol not in interference_elements:
                interference_elements += iso.element_symbol


        result = isopy.RatioKeyList()
        for interference_element in interference_elements:
            biggest = measured.copy(numerator_element_symbol_eq=interference_element)
            result += isopy.argmaxkey(biggest)

        return result

    if isinstance(measured, isopy.IsotopeArray):
        interference_isotopes = measured.keys().filter(element_symbol_neq=element)

        interference_elements = isopy.ElementKeyList()
        for iso in interference_isotopes:
            if iso.element_symbol not in interference_elements:
                interference_elements += iso.element_symbol

        result = isopy.IsotopeKeyList()

        for interference_element in interference_elements:
            biggest = measured.copy(element_symbol_eq=interference_element)
            result += isopy.argmaxkey(biggest)

        return result



def calculate_mass_fractionation_factor(data, mf_ratio, isotope_abundances=None, isotope_masses=None):
    """
    Calculate the mass fractionation factor for a given ratio in *data*.

    .. math::
        \\alpha = \\ln{( \\frac{r_{n,d}}{R_{n,d}} )} * \\frac{1}{ \\ln{( m_{n,d} } )}

    where :math:`n` is the numerator isotope and :math:`d` is the denominator isotope.

    Parameters
    ----------
    data
        Fractionated data. :math:`r` in the equation above.
    mf_ratio
        The isotope ratio from which the fractionation factor should be calculated. :math:`n,d` in the equation above.
    isotope_abundances
        Reference value for the isotope fractions of different elements.
        Defaults to ``isopy.refv.isotope.fraction``.
    isotope_masses
        Reference value for the isotope masses of different elements.
        Defaults to ``isopy.refv.isotope.mass``.

    Returns
    -------
    float
        The fractionation factor for *ratio* in *data*. :math:`\\alpha` in the equation above.

    Examples
    --------
    >>> array = isopy.tb.make_ms_array('pd').ratio('105pd')
    >>> array = isopy.tb.add_mass_fractionation(array, 0.1)
    >>> isopy.tb.calculate_mass_fractionation_factor(array, '108pd/105pd')
    0.09999999999999679

    >>> array = isopy.tb.make_ms_array('pd').ratio('105pd')
    >>> array = isopy.tb.remove_mass_fractionation(array, 0.15)
    >>> isopy.tb.calculate_mass_fractionation_factor(array, '108pd/105pd')
    -0.1500000000000046

    See Also
    --------
    remove_mass_fractionation
    """
    data = isopy.checks.check_type('data', data, isopy.core.RatioArray, isopy.core.IsotopeArray, coerce=True)
    mf_ratio = isopy.checks.check_type('ratio', mf_ratio, isopy.core.RatioKeyString, coerce=True)
    isotope_abundances = isopy.checks.check_reference_value('isotope_abundances', isotope_abundances, isopy.refval.isotope.abundance)
    isotope_masses = isopy.checks.check_reference_value('isotope_masses', isotope_masses, isopy.refval.isotope.mass)

    if isinstance(data, core.IsotopeArray):
        data = data.get(mf_ratio.numerator) / data.get(mf_ratio.denominator)
    else:
        data = data.get(mf_ratio)

    return (np.log(data / isotope_abundances.get(mf_ratio)) / np.log(isotope_masses.get(mf_ratio)))


def remove_mass_fractionation(data, fractionation_factor, isotope_masses=None):
    """
    Remove exponential mass fractionation from *data*.

    Calculated using:

    .. math::
        R_{n,d} = \\frac{r_{n,d}}{(m_{n,d})^{ \\alpha }}

    where :math:`n` is the numerator isotope and :math:`d` is the denominator isotope.

    Parameters
    ----------
    data : RatioArray
        Array containing data to be changed. :math:`r` in the equation above.
    fractionation_factor : float
        Fractionation factor to be applied. :math:`\\alpha` in the equation above
    isotope_masses
        Reference value for the isotope masses of different elements.
        Defaults to ``isopy.refv.isotope.mass``.

    Returns
    -------
    RatioArray
        Will have the same flavour as *data*. :math:`R` in the equation above.

    Examples
    --------
    >>> array = isopy.tb.make_ms_array('pd').ratio('105pd')
    >>> array
    (row) , 102Pd/105Pd , 104Pd/105Pd , 106Pd/105Pd , 108Pd/105Pd , 110Pd/105Pd
    None  , 0.04568     , 0.49888     , 1.22391     , 1.18495     , 0.52485
    >>> array = isopy.tb.remove_mass_fractionation(array, 0.15)
    >>> array
    (row) , 102Pd/105Pd , 104Pd/105Pd , 106Pd/105Pd , 108Pd/105Pd , 110Pd/105Pd
    None  , 0.04588     , 0.49960     , 1.22218     , 1.17995     , 0.52120
    >>> isopy.tb.calculate_mass_fractionation_factor(array, '108pd/105pd')
    -0.1500000000000046

    See Also
    --------
    isopy.tb.add_mass_fractionation
    """
    data = isopy.checks.check_type('data', data, isopy.core.RatioArray, isopy.core.IsotopeArray, coerce=True)
    fractionation_factor = isopy.checks.check_type('fractionation_factor', fractionation_factor, np.float, np.ndarray,
                                                   coerce=True, coerce_into=[np.float, np.array], allow_none=True)
    isotope_masses = isopy.checks.check_reference_value('isotope_masses', isotope_masses, isopy.refval.isotope.mass)


    if fractionation_factor is None:
        return data
    else:
        if isinstance(data, core.IsotopeArray):
            denom = isopy.argmaxkey(data)
            mass_array = isopy.array(isotope_masses.get(data.keys() / denom), data.keys())
        else:
            mass_array = isopy.array(isotope_masses.get(data.keys()))

        return data / (mass_array ** fractionation_factor)


def add_mass_fractionation(data, fractionation_factor, isotope_masses=None):
    """
    Add exponential mass fractionation to *data*.

    Calculated using:

    .. math::
        R_{n,d} = r_{n,d} * (m_{n,d})^{ \\alpha }

    where :math:`n` is the numerator isotope and :math:`d` is the denominator isotope.

    Parameters
    ----------
    data
        Array containing data to which the mass fractionation will be applied. :math:`r` in the equation above.
    fractionation_factor
        Fractionation factor to be applied. :math:`\\alpha` in the equation above
    isotope_masses
        Reference value for the isotope masses of different elements.
        Defaults to ``isopy.refv.isotope.mass``.

    Returns
    -------
    RatioArray, IsotopeArray
        Will have the same flavour as *data*. :math:`R` in the equation above.

    Examples
    --------
    >>> array = isopy.tb.make_ms_array('pd').ratio('105pd')
    >>> array
    (row) , 102Pd/105Pd , 104Pd/105Pd , 106Pd/105Pd , 108Pd/105Pd , 110Pd/105Pd
    None  , 0.04568     , 0.49888     , 1.22391     , 1.18495     , 0.52485

    >>> array = isopy.tb.add_mass_fractionation(array, 0.1)
    >>> array
    102Pd/105Pd         , 104Pd/105Pd         , 106Pd/105Pd        , 108Pd/105Pd       , 110Pd/105Pd
    0.04554614408342225 , 0.49840232023288333 , 1.2250738795847462 , 1.188297479483829 , 0.5273039811481615
    >>> isopy.tb.calculate_mass_fractionation_factor(array, '108pd/105pd')
    0.09999999999999679

    See Also
    --------
    isopy.tb.add_mass_fractionation
    """
    data = isopy.checks.check_type('data', data, isopy.core.RatioArray, isopy.core.IsotopeArray, coerce=True)
    fractionation_factor = isopy.checks.check_type('fractionation_factor', fractionation_factor, np.float, np.ndarray,
                                                   coerce=True, coerce_into=[np.float, np.array],allow_none=True)
    isotope_masses = isopy.checks.check_reference_value('isotope_masses', isotope_masses, isopy.refval.isotope.mass)

    if fractionation_factor is None:
        return data
    else:
        if isinstance(data, core.IsotopeArray):
            denom = isopy.argmaxkey(data)
            mass_array = isopy.array(isotope_masses.get(data.keys() / denom), data.keys())
        else:
            mass_array = isopy.array(isotope_masses.get(data.keys()))

        return  data * (mass_array ** fractionation_factor)


def remove_isobaric_interferences(data, interference_isotope, mf_factor = None, isotope_abundances = None, isotope_masses=None):
    """
    Remove all isobaric interferences for a given element.

    Parameters
    ----------
    data
        The data that isobaric interferences should be removed from.
    interference_isotope
        The isotope of the interfering element that should be used for the correction. Must be
        present in *data*.
    mf_factor
        If given, this mass fractionation factor is applied to the values for for the correction.
    isotope_abundances
        Reference value for the isotope fractions of different elements.
        Defaults to ``isopy.refv.isotope.fraction``.
    isotope_masses
        Reference value for the isotope masses of different elements.
        Defaults to ``isopy.refv.isotope.mass``.

    Returns
    -------
    IsotopeArray or RatioArray
        The interference corrected array

    Examples
    --------
    >>> isopy.tb.make_ms_array('pd', ru101=0, mf_factor=-1.6)
    (row) , 101Ru   , 102Pd   , 104Pd   , 105Pd   , 106Pd   , 108Pd   , 110Pd
    None  , 0.00000 , 0.01085 , 0.11485 , 0.22671 , 0.27330 , 0.25680 , 0.11045
    >>> array = isopy.tb.make_ms_array('pd', ru101=0.01, mf_factor=-1.6)
    >>> array
    (row) , 101Ru   , 102Pd   , 104Pd   , 105Pd   , 106Pd   , 108Pd   , 110Pd
    None  , 0.00173 , 0.01400 , 0.11665 , 0.22671 , 0.27330 , 0.25680 , 0.11045
    >>> beta = isopy.tb.calculate_mass_fractionation_factor(array, '108pd/105pd')
    >>> isopy.tb.remove_isobaric_interferences(array, 'ru101', beta).pprint()
    (row) , 101Ru   , 102Pd   , 104Pd   , 105Pd   , 106Pd   , 108Pd   , 110Pd
    None  , 0.00000 , 0.01085 , 0.11485 , 0.22671 , 0.27330 , 0.25680 , 0.11045
    """

    data = isopy.checks.check_type('data', data, isopy.core.RatioArray, isopy.IsotopeArray, coerce=True)
    interference_isotope = isopy.checks.check_type('interference_isotope', interference_isotope, isopy.core.IsotopeKeyString, coerce=True)
    mf_factor = isopy.checks.check_type('mf_factor', mf_factor, np.float, np.ndarray, coerce=True,
                                        coerce_into=[np.float, np.array], allow_none=True)
    isotope_abundances = isopy.checks.check_reference_value('isotope_abundances', isotope_abundances,
                                                           isopy.refval.isotope.abundance)
    isotope_masses = isopy.checks.check_reference_value('isotope_masses', isotope_masses, isopy.refval.isotope.mass)

    #Get the information we need to make the correction that works
    # for both isotope arrays and ratio arrays
    if isinstance(data, isopy.core.RatioArray):
        keys = data.keys()
        numer = keys.numerators
        denom = keys.common_denominator
    elif isinstance(data, isopy.core.IsotopeArray):
        keys = data.keys()
        numer = keys
        denom = None
    else:
        raise TypeError('variable "data" must be a IsotopeArray or a RatioArray not {}'.format(data.__class__.__name__))

    #Get the abundances of all isotopes of the interfering element
    inf_data = isopy.IsotopeArray(isotope_abundances.get(element_symbol_eq=interference_isotope.element_symbol))

    #Turn into a ratio relative to *isotope*
    inf_data = inf_data.ratio(interference_isotope, remove_denominator=False)

    #Account for mass fractionation of *mf_factor* is given
    if mf_factor is not None:
        inf_data = add_mass_fractionation(inf_data, mf_factor, isotope_masses=isotope_masses)


    #Scale relative to the measured value of *isotope*
    inf_data = inf_data * data[keys[numer.index(interference_isotope)]]

    #Convert to a mass array for easy lookup later
    inf_data = isopy.MassArray(inf_data, keys=inf_data.keys().numerators.mass_numbers)

    #Create the output array
    out = isopy.empty(data.nrows, keys)

    #Loop through each key in *out* and remove interference
    for i, key in enumerate(keys):
        out[key] = data[key] - inf_data.get(numer[i].mass_number, 0)

    #If *data* is a ratio array then correct for any interference on the denominator
    if denom is not None:
        out = out / (1 - inf_data.get(denom.mass_number, 0))

    return out


def normalise_data(data, reference_data, factor=1, is_deviation=False):
    """
    Normalise data to the given reference values.

    .. math::
        n = (\\frac{m} {r} - d ) * f

    where *n* is the normalised data, *m* is *data*, *f* is *factor*, *r* is *reference_data*,
    and *d* is 1 if *is_deviation* is ``False`` and 0 if *is_deviation* is ``True``.

    Parameters
    ----------
    data
        Data to be normalised
    reference_data
        The reference values used to normalise the data. Multiple values can be passed in a list.
        If multiple values are passed or *reference_values* has a size larger than 1 the mean of
        the values are used.
    factor
        The multiplication factor to be applied to *data* during the normalisation.
    is_deviation
        Set to ``True`` if *data* represents a deviation from the
        reference values.

    Examples
    --------
    >>> ref = isopy.tb.make_ms_array('pd').ratio('105pd')
    >>> array = isopy.tb.make_ms_sample('pd', fins=1, maxv=0.1).ratio('105pd')
    >>> norm = isopy.tb.normalise_data(array, ref)
    >>> norm
    (row) , 102Pd/105Pd , 104Pd/105Pd , 106Pd/105Pd , 108Pd/105Pd , 110Pd/105Pd
    0     , -28.60967   , -9.54006    , 9.50745     , 28.57674    , 47.70528
    1     , -28.76605   , -9.52060    , 9.53867     , 28.61274    , 47.69063
    2     , -28.73309   , -9.56115    , 9.51017     , 28.57170    , 47.63513
    3     , -28.53972   , -9.60582    , 9.48726     , 28.57055    , 47.60977
    4     , -28.69282   , -9.59705    , 9.50470     , 28.55289    , 47.65280
    ...   , ...         , ...         , ...         , ...         , ...
    95    , -28.50008   , -9.60504    , 9.52317     , 28.56964    , 47.66661
    96    , -28.64389   , -9.47620    , 9.52398     , 28.59006    , 47.70599
    97    , -28.64383   , -9.55584    , 9.48523     , 28.58221    , 47.67802
    98    , -28.64546   , -9.53501    , 9.52313     , 28.59109    , 47.66902
    99    , -28.67808   , -9.52044    , 9.51687     , 28.57950    , 47.62100

    >>> isopy.tb.normalise_data(isopy.sd2(array), ref, 1000, is_deviation=True)
    (row) , 102Pd/105Pd , 104Pd/105Pd , 106Pd/105Pd , 108Pd/105Pd , 110Pd/105Pd
    None  , 0.17452     , 0.05670     , 0.03890     , 0.03899     , 0.04946
    >>> isopy.sd2(norm)
    (row) , 102Pd/105Pd , 104Pd/105Pd , 106Pd/105Pd , 108Pd/105Pd , 110Pd/105Pd
    None  , 0.17452     , 0.05670     , 0.03890     , 0.03899     , 0.04946
    """
    data = isopy.checks.check_type('data', data, isopy.core.IsopyArray, coerce=True, coerce_into=isopy.core.asarray)
    factor = isopy.checks.check_type('factor', factor, np.float, str, coerce=True)
    if isinstance(factor, str):
        if factor in ['delta', 'permil', 'ppt']:
            factor = 1000
        elif factor in ['epsilon']:
            factor = 10000
        elif factor in ['mu', 'ppm']:
            factor = 1000000
        else:
            raise ValueError('parameter "factor": "{}" not an avaliable option.'.format(factor))
    is_deviation = isopy.checks.check_type('is_deviation', is_deviation, bool)

    reference_data = _combine(reference_data)

    new = data / reference_data
    if not is_deviation: new = new - 1
    new = new * factor

    return new


def denormalise_data(data, reference_data, factor=1, is_deviation=False):
    """
    Denormalise data to the given reference values.

    .. math::
        m = (\\frac{n} {f} + d ) * r

    *m* is the denormalised data, *n* is the normalised *data*,  *f* is *factor*,
    *r* is *reference_values*,  and *d* is 1 if *is_deviation* is ``False`` and
    0 if *is_deviation* is ``True``.


    Parameters
    ----------
    data
        Normalised data to be denormalised
    reference_data
        The reference values used to denormalise the data.
    factor
        The multiplication factor applied to *data* during the normalisation.
    is_deviation
        Set to ``True`` if *data* represents a deviation from the
        reference values.


    Examples
    --------
    >>> ref = isopy.tb.make_ms_array('pd').ratio('105pd')
    >>> array = isopy.tb.make_ms_sample('pd', fins=1).ratio('105pd')
    >>> norm = isopy.toolbox.isotope.normalise_data(array, ref, 1000)
    >>> denorm = isopy.tb.denormalise_data(norm, ref, 1000)
    >>> denorm
    (row) , 102Pd/105Pd , 104Pd/105Pd , 106Pd/105Pd , 108Pd/105Pd , 110Pd/105Pd
    0     , 0.04437     , 0.49412     , 1.23554     , 1.21881     , 0.54985
    1     , 0.04437     , 0.49413     , 1.23556     , 1.21882     , 0.54986
    2     , 0.04437     , 0.49412     , 1.23558     , 1.21887     , 0.54988
    3     , 0.04438     , 0.49413     , 1.23560     , 1.21887     , 0.54988
    4     , 0.04437     , 0.49413     , 1.23553     , 1.21879     , 0.54985
    ...   , ...         , ...         , ...         , ...         , ...
    95    , 0.04436     , 0.49415     , 1.23562     , 1.21883     , 0.54989
    96    , 0.04438     , 0.49412     , 1.23558     , 1.21884     , 0.54988
    97    , 0.04437     , 0.49412     , 1.23555     , 1.21888     , 0.54986
    98    , 0.04437     , 0.49411     , 1.23557     , 1.21880     , 0.54986
    99    , 0.04438     , 0.49410     , 1.23558     , 1.21884     , 0.54987
    >>> denorm/array
    (row) , 102Pd/105Pd , 104Pd/105Pd , 106Pd/105Pd , 108Pd/105Pd , 110Pd/105Pd
    0     , 1.00000     , 1.00000     , 1.00000     , 1.00000     , 1.00000
    1     , 1.00000     , 1.00000     , 1.00000     , 1.00000     , 1.00000
    2     , 1.00000     , 1.00000     , 1.00000     , 1.00000     , 1.00000
    3     , 1.00000     , 1.00000     , 1.00000     , 1.00000     , 1.00000
    4     , 1.00000     , 1.00000     , 1.00000     , 1.00000     , 1.00000
    ...   , ...         , ...         , ...         , ...         , ...
    95    , 1.00000     , 1.00000     , 1.00000     , 1.00000     , 1.00000
    96    , 1.00000     , 1.00000     , 1.00000     , 1.00000     , 1.00000
    97    , 1.00000     , 1.00000     , 1.00000     , 1.00000     , 1.00000
    98    , 1.00000     , 1.00000     , 1.00000     , 1.00000     , 1.00000
    99    , 1.00000     , 1.00000     , 1.00000     , 1.00000     , 1.00000
    """
    data = isopy.checks.check_type('data', data, isopy.core.IsopyArray, coerce=True, coerce_into=isopy.core.asarray)
    factor = isopy.checks.check_type('factor', factor, np.float, str, coerce=True)
    if isinstance(factor, str):
        if factor in ['delta', 'permil', 'ppt']:
            factor = 1000
        elif factor in ['epsilon']:
            factor = 10000
        elif factor in ['mu', 'ppm']:
            factor = 1000000
        else:
            raise ValueError('parameter "factor": "{}" not an avaliable option.'.format(factor))
    is_deviation = isopy.checks.check_type('is_deviation', is_deviation, bool)

    reference_data = _combine(reference_data)

    new = data / factor
    if not is_deviation: new = new + 1
    new = new * reference_data

    return new


def _combine(values):
    if isinstance(values, dict):
        return values

    if type(values) is list:
        values = [isopy.asarray(v) for v in values]
        values = isopy.concatenate(values)
    else:
        values = isopy.asarray(values)

    if values.size > 1:
        values = np.nanmean(values)

    return values


def find_outliers(data, cval = np.median, pmval=isopy.mad3, axis = None):
    """
    Find all outliers in data.

    Returns an array where outliers are marked with ``True`` and everything else ``False``.

    Parameters
    ----------
    data : isopy_array_like
        The outliers will be calculated for each column in *data*.
    cval : scalar, Callable
        Either the center value or a function that returns the center
        value when called with *data*.
    pmval : scalar, Callable
        Either the uncertainty value or a function that returns the
        uncertainty when called with *data*.
    axis : {0, 1}, Optional
        If not given then an array with each individual outlier marked is returned. Otherwise
        ``np.any(outliers, axis)`` is returned.

    Examples
    --------
    >>> array = isopy.tb.make_ms_sample('pd', integrations=10)
    >>> array['pd102'][:2] *= 2
    >>> array['pd110'][-1] *= 2
    >>> isopy.tb.find_outliers(array)
    (row) , 102Pd , 104Pd , 105Pd , 106Pd , 108Pd , 110Pd
    0     , True  , False , True  , False , False , False
    1     , True  , False , False , False , False , False
    2     , False , False , False , False , False , False
    3     , False , False , False , False , False , False
    4     , False , False , False , False , False , False
    5     , False , False , False , False , False , False
    6     , False , False , False , False , False , False
    7     , False , False , False , False , False , False
    8     , False , False , False , False , False , False
    9     , False , False , False , False , False , True

    >>> isopy.tb.find_outliers(array, axis=1)
    array([ True,  True, False, False, False, False, False, False, False, True])

    >>> isopy.tb.find_outliers(array, np.mean, isopy.sd2)
    (row) , 102Pd , 104Pd , 105Pd , 106Pd , 108Pd , 110Pd
    0     , False , False , True  , False , False , False
    1     , False , False , False , False , False , False
    2     , False , False , False , False , False , False
    3     , False , False , False , False , False , False
    4     , False , False , False , False , False , False
    5     , False , False , False , False , False , False
    6     , False , False , False , False , False , False
    7     , False , False , False , False , False , False
    8     , False , False , False , False , False , False
    9     , False , False , False , False , False , True

    >>> isopy.tb.find_outliers(array, np.mean(array), isopy.sd2(array))
    (row) , 102Pd , 104Pd , 105Pd , 106Pd , 108Pd , 110Pd
    0     , False , False , True  , False , False , False
    1     , False , False , False , False , False , False
    2     , False , False , False , False , False , False
    3     , False , False , False , False , False , False
    4     , False , False , False , False , False , False
    5     , False , False , False , False , False , False
    6     , False , False , False , False , False , False
    7     , False , False , False , False , False , False
    8     , False , False , False , False , False , False
    9     , False , False , False , False , False , True
    """
    data = isopy.checks.check_type('data', data, isopy.core.IsopyArray, coerce=True)
    axis = isopy.checks.check_type('axis', axis, int, allow_none=True)

    if callable(cval):
        cval = cval(data)
    if callable(pmval):
        pmval = pmval(data)
    pmval = np.abs(pmval)

    outliers = (data > (cval + pmval)) + (data < (cval - pmval))

    if axis is None:
        return outliers
    else:
        return np.any(outliers, axis=axis)