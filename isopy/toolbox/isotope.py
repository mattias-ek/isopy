import numpy as np
import isopy as isopy
from isopy import core

__all__ = ['remove_mass_fractionation', 'add_mass_fractionation',
           'mass_fractionation_factor', 'internal_normalisation',
           'remove_isobaric_interferences', 'find_isobaric_interferences',
           'make_ms_array', 'make_ms_beams', 'make_ms_sample', 'johnson_nyquist_noise',
           'rDelta', 'inverse_rDelta']

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


    where :math:`n` is the numerator isotope and :math:`d` is the denominator isotope.

    Adapted from the equations in `Liu & Pearson (2014) Chemical Geology, 10, 301-311
    <https://doi.org/10.1016/j.chemgeo.2013.11.008>`_.

    Parameters
    ----------
    voltage : isotope_rarray, float, np.ndarray
        The measured voltages. :math:`v` in the equations above.
    resistor : isotope_rarray, float, np.ndarray, dict
        The resistor for the measurement. Default value is ``1E11``. :math:`r` in the equations above.
        If *resistor* is a dictionary *R* will be used for values not in the dictionary.
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
    noise: np.float or np.ndarray or isotope_rarray
        The noise in V for the given voltage/set of voltages.

    Examples
    --------
    >>> isopy.tb.johnson_nyquist_noise(10) * 1000 #in millivolts
    0.13883808575503015
    >>> isopy.tb.johnson_nyquist_noise(10, 1E10) * 1000 # 1E10 resistor
    0.14528174343845432

    >>> array = isopy.tb.make_ms_array('pd')
    >>> array = array * (10 / array['106pd']) #10v on the largest isotope
    >>> isopy.tb.johnson_nyquist_noise(array) * 1000 #in millivolts
    (row)      102Pd (f8)    104Pd (f8)    105Pd (f8)    106Pd (f8)    108Pd (f8)    110Pd (f8)
    -------  ------------  ------------  ------------  ------------  ------------  ------------
    None          0.03025       0.08932       0.12565       0.13884       0.13663       0.09156
    IsopyNdarray(-1, flavour='isotope', default_value=nan)
    >>> resistors = dict(pd102=1E13, pd106=1E10) #1E11 is used for missing keys
    >>> isopy.tb.johnson_nyquist_noise(array, resistors) * 1000
    (row)      102Pd (f8)    104Pd (f8)    105Pd (f8)    106Pd (f8)    108Pd (f8)    110Pd (f8)
    -------  ------------  ------------  ------------  ------------  ------------  ------------
    None          0.02672       0.08932       0.12565       0.14528       0.13663       0.09156
    IsopyNdarray(-1, flavour='isotope', default_value=nan)
    """

    voltage = isopy.checks.check_type('voltage', voltage, isopy.core.IsopyArray, np.ndarray, np.float64, coerce=True,
                                      coerce_into=[isopy.core.IsopyArray, np.float64, np.array])
    resistor = isopy.checks.check_type('resistor', resistor, isopy.core.IsopyArray, np.ndarray, dict, coerce=True,
                                       coerce_into=[isopy.core.IsopyArray, np.float64, np.array])
    integration_time = isopy.checks.check_type('integration_time', integration_time, np.float64, coerce=True)
    include_counting_statistics = isopy.checks.check_type('include_counting_statistics', include_counting_statistics,
                                                          bool)
    if isinstance(resistor, (core.IsopyArray, dict)):
        resistor = core.RefValDict(resistor, default_value=R)
        resistor = isopy.array({key: resistor.get(key) for key in voltage.keys})

    T = isopy.checks.check_type('T', T, np.float64, coerce=True)
    R = isopy.checks.check_type('R', R, np.float64, coerce=True)
    cpv = isopy.checks.check_type('cpv', cpv, np.float64, coerce=True)

    kB = np.float64(1.3806488E-023) # Boltsman constant

    t_noise = np.sqrt((4 * kB * T * resistor) / integration_time) * (R / resistor)

    if include_counting_statistics:
        c_stat = np.sqrt(1 / (voltage * cpv * integration_time)) * voltage
        return np.sqrt(c_stat ** 2 + t_noise ** 2)
    else:
        return t_noise

def make_ms_array(*args, mf_factor = None, isotope_fractions = 'isotope.fraction', isotope_masses='isotope.mass', **kwargs):
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
    isotope_fractions
        Reference value for the isotope fractions of different elements.
        Defaults to ``isopy.refval.isotope.abundance``.
    isotope_masses
        Reference value for the isotope masses of different elements.
        Defaults to ``isopy.refval.isotope.mass``.
    kwargs
        Each kwarg key must be either an element key string or an isotope key string. The isotope
        fractions added to the array based on the kwarg key are multiplied by the kwarg value.
        Otherwise behaves the same as for *args*.

    Returns
    -------
    result : IsopyArray
        The constructed isotope array.

    Examples
    --------
    >>> isopy.tb.make_ms_array('pd')
    (row)      102Pd (f8)    104Pd (f8)    105Pd (f8)    106Pd (f8)    108Pd (f8)    110Pd (f8)
    -------  ------------  ------------  ------------  ------------  ------------  ------------
    None          0.01020       0.11140       0.22330       0.27330       0.26460       0.11720
    IsopyNdarray(-1, flavour='isotope', default_value=nan)

    >>> isopy.tb.make_ms_array('pd', ru101=0.1)
    (row)      101Ru (f8)    102Pd (f8)    104Pd (f8)    105Pd (f8)    106Pd (f8)    108Pd (f8)    110Pd (f8)
    -------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
    None          0.01706       0.04175       0.13002       0.22330       0.27330       0.26460       0.11720
    IsopyNdarray(-1, flavour='isotope', default_value=nan)

    >>> isopy.tb.make_ms_array('pd', ru101=0.1, ru99=0) #99Ru doesnt contribute anything to the array but gets a contribution from 101Ru.
    (row)      99Ru (f8)    101Ru (f8)    102Pd (f8)    104Pd (f8)    105Pd (f8)    106Pd (f8)    108Pd (f8)    110Pd (f8)
    -------  -----------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
    None         0.01276       0.01706       0.04175       0.13002       0.22330       0.27330       0.26460       0.11720
    IsopyNdarray(-1, flavour='isotope', default_value=nan)

    >>> isopy.tb.make_ms_array('pd', 'cd')
    (row)      102Pd (f8)    104Pd (f8)    105Pd (f8)    106Pd (f8)    108Pd (f8)    110Pd (f8)    111Cd (f8)    112Cd (f8)    113Cd (f8)    114Cd (f8)    116Cd (f8)
    -------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
    None          0.01020       0.11140       0.22330       0.28579       0.27350       0.24205       0.12804       0.24117       0.12225       0.28729       0.07501
    IsopyNdarray(-1, flavour='isotope', default_value=nan)

    >>> isopy.tb.make_ms_array('pd', cd=1) #Same as example above
    (row)      102Pd (f8)    104Pd (f8)    105Pd (f8)    106Pd (f8)    108Pd (f8)    110Pd (f8)    111Cd (f8)    112Cd (f8)    113Cd (f8)    114Cd (f8)    116Cd (f8)
    -------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
    None          0.01020       0.11140       0.22330       0.28579       0.27350       0.24205       0.12804       0.24117       0.12225       0.28729       0.07501
    IsopyNdarray(-1, flavour='isotope', default_value=nan)

    See Also
    --------
    make_ms_beams, make_ms_sample
    """

    isotope_fractions = isopy.refval(isotope_fractions, isopy.RefValDict)
    isotope_masses = isopy.refval(isotope_masses, isopy.RefValDict)

    input = []
    for key in args:
        if type(key) is tuple:
            input.append((k, 1) for k in key)
        elif isinstance(key, dict):
            kwargs.update(key)
        else:
            input.append((key, 1))

    for item in kwargs.items():
        input.append(item)

    out_keys = isopy.askeylist()
    for key, val in input:
        if isinstance(key, core.IsopyArray):
            keys = key.keys()
        else:
            key = isopy.askeystring(key)
            if key.flavour in isopy.asflavour('element|isotope|molecule'):
                keys = key.isotopes()
            else:
                raise ValueError(f'Unknown input: {key} input must be an element or isotope key string')

        keys = keys.filter(mz_neq = out_keys.mz)

        out_keys += tuple(dict(zip(keys.mz, keys)).values())

    out_keys = out_keys.sorted()
    mz_keys = [key.mz for key in out_keys]
    mzkey_map = dict(zip(mz_keys, out_keys))

    out = isopy.zeros(None, out_keys)
    for key, val in input:
        if isinstance(key, core.IsopyArray):
            array = key
        else:
            key = isopy.askeystring(key)
            keys = key.element_symbol.isotopes
            keys = keys.filter(mz_eq=mz_keys)

            array = isotope_fractions.to_array(keys)
            #array = isopy.array(keys.fractions(isotope_fractions=isotope_fractions), keys)

        if mf_factor is not None:
            array = add_mass_fractionation(array, mf_factor, isotope_masses=isotope_masses)
            array = array.normalise(np.sum(isotope_fractions.to_array(keys), axis=None))

        array = array * val

        for k, v in array.items():
            out[mzkey_map[k.mz]] += v

    return out


def make_ms_beams(*args, mf_factor=None, fixed_voltage = 10, fixed_key = isopy.keymax, integrations = 100, integration_time=8.389, resistor=1E11,
                  random_seed = None, isotope_fractions='isotope.fraction', isotope_masses='isotope.mass', **kwargs):
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
    fixed_voltage
        The voltage of *fixed_key* in the array. The value for all other isotopes in
        the array are adjusted accordingly.
    fixed_key
        If not given then the this defaults to the most abundant isotope. If ``None`` then the sum
        of all isotopes in the array will be set to *fixed_voltage*
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
    isotope_fractions
        Reference value for the isotope fractions of different elements.
        Defaults to ``isopy.refval.isotope.abundance``.
    isotope_masses
        Reference value for the isotope masses of different elements.
        Defaults to ``isopy.refval.isotope.mass``.
    kwargs
        Each kwarg key must be either an element key string or an isotope key string. The isotope
        fractions added to the array based on the kwarg key are multiplied by the kwarg value.
        Otherwise behaves the same as for *args*.

    Returns
    -------
    IsopyArray
        The simulated measurements.

    Examples
    --------
    >>> isopy.tb.make_ms_beams('pd', ru101=0.01)
    (row)      101Ru (f8)    102Pd (f8)    104Pd (f8)    105Pd (f8)    106Pd (f8)    108Pd (f8)    110Pd (f8)
    -------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
    0             0.06239       0.48870       4.14431       8.17054       9.99988       9.68156       4.28856
    1             0.06241       0.48866       4.14418       8.17052      10.00002       9.68148       4.28846
    2             0.06244       0.48865       4.14416       8.17049       9.99998       9.68170       4.28839
    3             0.06241       0.48866       4.14424       8.17060       9.99996       9.68172       4.28826
    4             0.06240       0.48867       4.14432       8.17043      10.00002       9.68168       4.28830
                   ...           ...           ...           ...           ...           ...           ...
    95            0.06244       0.48868       4.14397       8.17066      10.00006       9.68180       4.28820
    96            0.06243       0.48868       4.14439       8.17048      10.00007       9.68168       4.28838
    97            0.06243       0.48863       4.14421       8.17031      10.00014       9.68160       4.28837
    98            0.06237       0.48867       4.14419       8.17038       9.99999       9.68159       4.28828
    99            0.06241       0.48864       4.14425       8.17037      10.00001       9.68175       4.28839
    IsopyNdarray(100, flavour='isotope', default_value=nan)

    See Also
    --------
    make_ms_array, make_ms_sample
    """

    beams = make_ms_array(*args, **kwargs, mf_factor=mf_factor, isotope_masses=isotope_masses, isotope_fractions=isotope_fractions)
    if beams.size != 1:
        raise ValueError('The constructed ms array has a size larger than one')

    if fixed_voltage is not None:
        beams = beams.normalise(fixed_voltage, fixed_key)

    if integrations is None:
        return beams
    else:
        noise = johnson_nyquist_noise(beams, integration_time=integration_time, resistor=resistor)

        return isopy.random(integrations, list(zip(beams.values(), noise.values())), keys=beams.keys, seed=random_seed)



def make_ms_sample(ms_array, *, fnat = None, fins = None, fixed_voltage = 10, fixed_key = isopy.keymax,
                   blank = None, blank_fixed_voltage = 0.01, blank_fixed_key = isopy.keymax,
                   spike = None, spike_fraction = 0.5,
                   integrations = 100, integration_time = 8.389, resistors = 1E11,
                   random_seed = None,
                   isotope_fractions='isotope.fraction', isotope_masses='isotope.mass', **interferences):
    """
    Creates a simulated the measurement of a sample with natural and instrumental mass
    fractionation added to the array. The standard deviation of measurements for each isotope
    is equal to the johnson-nyquist noise and counting statistics.

    Parameters
    ----------
    ms_array
        Any object that can be passed to ``make_ms_array`` to returns valid array.
        Also accepts a tuple or a dict which will be unpacked appropriately.
    fnat
        If given, the natural fractionation fractionation factor is applied to the ms_array
        before *interferences* are added to the ms_array.
    fins
        If given, the instrumental mass fractionation factor is applied to the ms_array
        at the same time the *interferences* are added to the ms_array.
    fixed_voltage
        The voltage of *fixed_key* in the array. The value for all other isotopes in
        the array are adjusted accordingly.
    fixed_key
        If not given then the this defaults to the most abundant isotope. If ``None`` then the sum
        of all isotopes in the array will be set to *fixed_voltage*
    blank
        The blank sample to be added to the sample. Can be object that can be singularly passed
        to ``make_ms_array`` which returns valid array. Also accepts a tuple or a dict which will
        be unpacked appropriately.
    blank_fixed_voltage
        The voltage of the *blank_fixed_key* in returned sample that is *blank*.
    blank_fixed_key
        If not given then the this defaults to the most abundant isotope. If ``None`` then the sum
        of all isotopes in the array will be set to *blank_fixed_voltage*
    spike
        If given this spike mixture will be added ms_array after *fnat* but before *fins* and
        *interferences* are added to the array.
    spike_fraction
        The fraction of spike in the final mixture based on the isotopes in *spike*.
    integrations
        The number of simulated measurements. If ``None`` no measurements are simulated and the
        retured array contains the true values.
    integration_time
        The integration time for each simulated measurement.
    resistors
        The resistor used for each measurement. A isotope array or a dictionary can be passed to
        to give different resistor values for different isotopes.
    random_seed
        Must be an integer. Seed given to the random generator.
    isotope_fractions
        Reference value for the isotope fractions of different elements.
        Defaults to ``isopy.refval.isotope.abundance``.
    isotope_masses
        Reference value for the isotope masses of different elements.
        Defaults to ``isopy.refval.isotope.mass``.
    interferences
        Each kwarg key must be either an element key string or an isotope key string. The isotope
        fractions added to the array based on the kwarg key are multiplied by the kwarg value.
        Otherwise behaves the same as for *args*. A dictionary named interferences can be passed
        for keys that cannot be passed as kwargs, e.g. '137Ba++'.

    Returns
    -------
    IsopyArray
        The simulated measurements.

    Examples
    --------
    >>> isopy.tb.make_ms_sample('pd', ru101=0.01, cd111=0.001, fnat = 0.1, fins=-1.6)
    (row)      101Ru (f8)    102Pd (f8)    104Pd (f8)    105Pd (f8)    106Pd (f8)    108Pd (f8)    110Pd (f8)    111Cd (f8)
    -------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
    0             0.49399       1.29459       4.70717       8.28435      10.00013       9.41279       4.09031       0.03591
    1             0.49401       1.29462       4.70724       8.28425      10.00002       9.41281       4.09014       0.03593
    2             0.49401       1.29459       4.70734       8.28425       9.99998       9.41288       4.09025       0.03594
    3             0.49404       1.29462       4.70733       8.28416      10.00019       9.41271       4.09034       0.03592
    4             0.49396       1.29463       4.70736       8.28428      10.00011       9.41278       4.09022       0.03591
                   ...           ...           ...           ...           ...           ...           ...           ...
    95            0.49395       1.29455       4.70720       8.28435       9.99995       9.41284       4.09020       0.03592
    96            0.49404       1.29467       4.70718       8.28439      10.00013       9.41264       4.09039       0.03592
    97            0.49405       1.29465       4.70730       8.28448       9.99992       9.41271       4.09031       0.03591
    98            0.49394       1.29459       4.70733       8.28431      10.00006       9.41278       4.09040       0.03592
    99            0.49397       1.29457       4.70732       8.28445       9.99980       9.41286       4.09026       0.03592
    IsopyNdarray(100, flavour='isotope', default_value=nan)


    See Also
    --------
    make_ms_array, make_ms_beams
    """
    ms_array = make_ms_array(ms_array, mf_factor = fnat,
                             isotope_fractions=isotope_fractions, isotope_masses=isotope_masses)

    if spike is not None:
        spike = isopy.asarray(spike)

        spsum = np.sum(spike, axis=1)
        smpsum = np.sum(ms_array.to_array(key_eq=spike.keys()), axis = 1)

        spike = spike / spsum * smpsum
        spike = spike * spike_fraction
        ms_array = ms_array * (1 - spike_fraction)

        ms_array = ms_array + spike.default(0)

    ms_array = make_ms_array(ms_array, mf_factor = fins,
                             isotope_fractions=isotope_fractions, isotope_masses=isotope_masses,
                             **interferences.pop('interferences', {}), **interferences)

    ms_array = ms_array.normalise(fixed_voltage, fixed_key)

    if blank is not None:
        blank = make_ms_array(blank, mf_factor = fins, isotope_fractions=isotope_fractions,
                              isotope_masses=isotope_masses)

        blank = blank.normalise(blank_fixed_voltage, blank_fixed_key)
        if fixed_key is None:
            sum = isopy.subtract(ms_array, blank.default(0), keys=ms_array.keys).sum(axis=None)
            ms_array = ms_array.normalise(sum, None)
        elif isinstance(fixed_key, (str, list, tuple)):
            sum = isopy.subtract(ms_array.default(0), blank, keys=fixed_key).sum(axis=None)
            ms_array = ms_array.normalise(sum, fixed_key)
        else:
            sum = isopy.subtract(ms_array, blank.default(0), keys=fixed_key(ms_array)).sum(axis=None)
            ms_array = ms_array.normalise(sum, fixed_key(ms_array))

        ms_array = isopy.add(ms_array, blank.default(0), keys=ms_array.keys)

    ms_array = make_ms_beams(ms_array, fixed_voltage=None, integrations=integrations,
                             integration_time=integration_time, random_seed=random_seed,
                             resistor=resistors, isotope_fractions=isotope_fractions)

    return ms_array

@core.append_preset_docstring
@core.add_preset(('ppt', 'permil'), extnorm_factor=1000)
@core.add_preset('epsilon', extnorm_factor=1E4)
@core.add_preset(('mu', 'ppm'), extnorm_factor=1E6)
def internal_normalisation(data, mf_ratio, interference_correction=True,
                           extnorm_value = None, extnorm_factor=None,
                           isotope_fractions='isotope.fraction', isotope_masses='isotope.mass', mf_tol=1E-8):
    """
    A data reduction scheme for internaly normalised data.

    If *interference_correction* is True an interference correction will be applied for all isotopes that are different from the *mf_ratio* numerator
    element. This will be done together with the mass fractionation correction to account for isobaric interferences
    on the *mf_ratio*. If more than one isotope exists for an
    an element the largest isotope is used for the interference correction.

    If *extnorm_value* and/or *extnorm_factor* is given then the returned data is
    externally normalised to these values using the rDelta function. The default value for
    *extnorm_value* is 1 and the default value for *extnorm_factor* is the same as
    *isotope_fractions*, should only one of these values
    be given.

    Parameters
    ----------
    data
        The data to be corrected. Isotope arrays will automatically be converted to ratio arrays.
        Ratio arrays must have a common denominator that is the same as the that of *mf_ratio*.
    mf_ratio
        The data will be internally normalised to this ratio. Must be present in *data*.
    interference_correction
        If True then the data is corrected for isobaric interferences.
    extnorm_value
        If given the result in normalised to this value with *extnorm_factor*. If
        *extnorm_value* is not given a normalisation factor of 1 is used.
    extnorm_factor
        If given and *normalisation_value* is not the result is normalised against
        *isotope_fractions*.
    isotope_fractions
        Reference value for the isotope fractions of different elements.
        Defaults to ``isopy.refval.isotope.abundance``
    isotope_masses
        Reference value for the isotope masses of different elements.
        Defaults to ``isopy.refval.isotope.mass``
    mf_tol
        Only when the difference between the current and the previous mass_fractionation factor
        is below this value is the interference correction assumed to have converged.

    Returns
    -------
    IsopyArray
        Only contains the isotopes of the element with the same element symbol as *mf_ratio*

    Examples
    --------
    >>> array = isopy.tb.make_ms_array('pd', ru101 = 0.1, cd111 = 0.1, mf_factor=0.1).normalise(10, isopy.keymax)
    >>> array
    (row)      101Ru (f8)    102Pd (f8)    104Pd (f8)    105Pd (f8)    106Pd (f8)    108Pd (f8)    110Pd (f8)    111Cd (f8)
    -------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
    None          0.62089       1.51955       4.72959       8.12576      10.00000       9.68819       4.73963       0.46692
    IsopyNdarray(-1, flavour='isotope', default_value=nan)
    >>> isopy.tb.internal_normalisation(array, '108Pd/105Pd')
    (row)      102Pd/105Pd (f8)    104Pd/105Pd (f8)    106Pd/105Pd (f8)    110Pd/105Pd (f8)
    -------  ------------------  ------------------  ------------------  ------------------
    None                0.04568             0.49888             1.22391             0.52485
    IsopyNdarray(-1, flavour='ratio[isotope, isotope]', default_value=nan)
    >>> isopy.tb.internal_normalisation(array, '108Pd/105Pd') / isopy.refval.isotope.fraction
    (row)      102Pd/105Pd (f8)    104Pd/105Pd (f8)    106Pd/105Pd (f8)    110Pd/105Pd (f8)
    -------  ------------------  ------------------  ------------------  ------------------
    None                1.00000             1.00000             1.00000             1.00000
    IsopyNdarray(-1, flavour='ratio[isotope, isotope]', default_value=nan)
    >>> isopy.tb.internal_normalisation(array, '108Pd/105Pd', interference_correction=False, extnorm_factor=1)
    (row)      102Pd/105Pd (f8)    104Pd/105Pd (f8)    106Pd/105Pd (f8)    110Pd/105Pd (f8)
    -------  ------------------  ------------------  ------------------  ------------------
    None                3.11997             0.16916             0.00343             0.10006
    IsopyNdarray(-1, flavour='ratio[isotope, isotope]', default_value=nan)
    """
    #data = isopy.checks.check_array('data', data, 'isotope', 'ratio')
    mf_ratio = isopy.checks.check_type('mf_ratio', mf_ratio, isopy.core.RatioKeyString, coerce=True)
    extnorm_factor = isopy.checks.check_type('extnorm_factor', extnorm_factor, np.float64, str, coerce=True, allow_none=True)
    isotope_fractions = isopy.asrefval(isotope_fractions, ratio_function='divide')
    isotope_masses = isopy.asrefval(isotope_masses, ratio_function='divide')

    # Convert the data into a ratio array
    if data.flavour == 'ratio':
        if data.keys.common_denominator is None:
            raise ValueError('data must hav a common denominator')
        elif data.keys.common_denominator != mf_ratio.denominator:
            raise ValueError('data and mf_ratio do not have the same denominator')
        rat = data
    else:
        rat = data.ratio(mf_ratio.denominator)

    if interference_correction is True:
        isobaric_interferences = find_isobaric_interferences(mf_ratio.numerator.element_symbol, data)
    elif  isinstance(interference_correction, dict):
        isobaric_interferences = interference_correction
    else:
        isobaric_interferences = {}

    #Find the initial mass fractionation
    beta = mass_fractionation_factor(rat, mf_ratio, isotope_fractions=isotope_fractions, isotope_masses=isotope_masses)

    if isobaric_interferences:
        #Do a combined mass fractionation and isobaric interference correction.
        #This can account for isobaric interferences on isotopes in *mf_ratio*
        for i in range(100):
            rat2 = rat
            prev_beta = beta

            rat2 = remove_isobaric_interferences(rat2, isobaric_interferences,
                                                 beta, isotope_fractions=isotope_fractions, isotope_masses=isotope_masses)

            # Calculate the mass fractionation.
            beta = mass_fractionation_factor(rat2, mf_ratio, isotope_fractions=isotope_fractions, isotope_masses=isotope_masses)

            if np.all(np.abs(beta - prev_beta) < mf_tol):
                break #Beta value has converged so no need for more iterations.
        else:
            raise ValueError('values did not converge after 100 iterations of the interference correction')
    else:
        rat2 = rat

    #Remove the isotopes on interfering elements and the mass bias ratio
    rat = rat2.to_array(numerator_element_symbol_eq=mf_ratio.numerator.element_symbol, key_neq=mf_ratio)

    #Correct for mass fractionation
    rat = remove_mass_fractionation(rat, beta, isotope_masses=isotope_masses)

    if extnorm_value is not None:
        if extnorm_factor is None: extnorm_factor = 1
        rat = rDelta(rat, extnorm_value, factor = extnorm_factor)

    elif extnorm_factor is not None:
        rat = rDelta(rat, isotope_fractions, factor = extnorm_factor)

    # Return the corrected data
    return rat

def mass_fractionation_factor(data, mf_ratio,
                              isotope_fractions='isotope.fraction', isotope_masses='isotope.mass'):
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
    isotope_fractions
        Reference value for the isotope fractions of different elements.
        Defaults to ``isopy.refval.isotope.abundance``.
    isotope_masses
        Reference value for the isotope masses of different elements.
        Defaults to ``isopy.refval.isotope.mass``.

    Returns
    -------
    float
        The fractionation factor for *ratio* in *data*. :math:`\\alpha` in the equation above.

    Examples
    --------
    >>> array = isopy.tb.make_ms_array('pd').ratio('105pd')
    >>> array = isopy.tb.add_mass_fractionation(array, 0.1)
    >>> isopy.tb.mass_fractionation_factor(array, '108pd/105pd')
    0.09999999999999679

    >>> array = isopy.tb.make_ms_array('pd').ratio('105pd')
    >>> array = isopy.tb.remove_mass_fractionation(array, 0.15)
    >>> isopy.tb.mass_fractionation_factor(array, '108pd/105pd')
    -0.1500000000000046

    See Also
    --------
    remove_mass_fractionation
    """
    #data = isopy.checks.check_array('data', data, 'isotope', 'ratio')
    mf_ratio = isopy.checks.check_type('ratio', mf_ratio, isopy.core.RatioKeyString, coerce=True)
    isotope_fractions = isopy.refval(isotope_fractions, isopy.RefValDict)
    isotope_masses = isopy.refval(isotope_masses, isopy.RefValDict)

    if data.flavour == 'isotope':
        data = data.get(mf_ratio.numerator) / data.get(mf_ratio.denominator)
    else:
        data = data.get(mf_ratio)

    return (np.log(data / isotope_fractions.get(mf_ratio)) / np.log(isotope_masses.get(mf_ratio)))


def remove_mass_fractionation(data, fractionation_factor, denom = None, isotope_masses='isotope.mass'):
    """
    Remove exponential mass fractionation from *data*.

    Calculated using:

    .. math::
        R_{n,d} = \\frac{r_{n,d}}{(m_{n,d})^{ \\alpha }}

    where :math:`n` is the numerator isotope and :math:`d` is the denominator isotope.

    Parameters
    ----------
    data : IsopyArray
        Array containing data to be changed. :math:`r` in the equation above.
    fractionation_factor : float
        Fractionation factor to be applied. :math:`\\alpha` in the equation above
    denom
        The denominator isotope if *data* is an isotope array. If not given then the key with the
        largest median value is used.
    isotope_masses
        Reference value for the isotope masses of different elements.
        Defaults to ``isopy.refval.isotope.mass``.

    Returns
    -------
    IsopyArray
        Will have the same flavour as *data*. :math:`R` in the equation above.

    Examples
    --------
    >>> array = isopy.tb.make_ms_array('pd').ratio('105pd')
    >>> array
    (row)      102Pd/105Pd (f8)    104Pd/105Pd (f8)    106Pd/105Pd (f8)    108Pd/105Pd (f8)    110Pd/105Pd (f8)
    -------  ------------------  ------------------  ------------------  ------------------  ------------------
    None                0.04568             0.49888             1.22391             1.18495             0.52485
    IsopyNdarray(-1, flavour='ratio[isotope, isotope]', default_value=nan)
    >>> array = isopy.tb.remove_mass_fractionation(array, 0.15)
    >>> array
    (row)      102Pd/105Pd (f8)    104Pd/105Pd (f8)    106Pd/105Pd (f8)    108Pd/105Pd (f8)    110Pd/105Pd (f8)
    -------  ------------------  ------------------  ------------------  ------------------  ------------------
    None                0.04588             0.49960             1.22218             1.17995             0.52120
    IsopyNdarray(-1, flavour='ratio[isotope, isotope]', default_value=nan)
    >>> isopy.tb.mass_fractionation_factor(array, '108pd/105pd')
    -0.1500000000000046

    See Also
    --------
    isopy.tb.add_mass_fractionation
    """
    #data = isopy.checks.check_array('data', data, 'isotope', 'ratio')
    fractionation_factor = isopy.checks.check_type('fractionation_factor', fractionation_factor, np.float64, np.ndarray,
                                                   coerce=True, coerce_into=[np.float64, np.array], allow_none=True)
    isotope_masses = isopy.refval(isotope_masses, isopy.RefValDict)


    if fractionation_factor is None:
        return data
    else:
        if data.flavour == 'isotope':
            if denom is None:
                denom = isopy.keymax(data)
            mass_array = isopy.array(isotope_masses.to_array(data.keys / denom), data.keys)
        else:
            mass_array = isotope_masses.to_array(data.keys)

        return data / (mass_array ** fractionation_factor)


def add_mass_fractionation(data, fractionation_factor, denom=None, isotope_masses='isotope.mass'):
    """
    Add exponential mass fractionation to *data*.

    Calculated using:

    .. math::
        r_{n,d} = R_{n,d} * (m_{n,d})^{ \\alpha }

    where :math:`n` is the numerator isotope and :math:`d` is the denominator isotope.

    Parameters
    ----------
    data
        Array containing data to which the mass fractionation will be applied. :math:`R` in the equation above.
    fractionation_factor
        Fractionation factor to be applied. :math:`\\alpha` in the equation above
    denom
        The denominator isotope if *data* is an isotope array. If not given then the key with the
        largest median value is used.
    isotope_masses
        Reference value for the isotope masses of different elements.
        Defaults to ``isopy.refval.isotope.mass``.

    Returns
    -------
    IsopyArray
        Will have the same flavour as *data*. :math:`r` in the equation above.

    Examples
    --------
    >>> array = isopy.tb.make_ms_array('pd').ratio('105pd')
    >>> array
    (row)      102Pd/105Pd (f8)    104Pd/105Pd (f8)    106Pd/105Pd (f8)    108Pd/105Pd (f8)    110Pd/105Pd (f8)
    -------  ------------------  ------------------  ------------------  ------------------  ------------------
    None                0.04568             0.49888             1.22391             1.18495             0.52485
    IsopyNdarray(-1, flavour='ratio[isotope, isotope]', default_value=nan)
    >>> array = isopy.tb.add_mass_fractionation(array, 0.1)
    >>> array
    (row)      102Pd/105Pd (f8)    104Pd/105Pd (f8)    106Pd/105Pd (f8)    108Pd/105Pd (f8)    110Pd/105Pd (f8)
    -------  ------------------  ------------------  ------------------  ------------------  ------------------
    None                0.04555             0.49840             1.22507             1.18830             0.52730
    IsopyNdarray(-1, flavour='ratio[isotope, isotope]', default_value=nan)
    >>> isopy.tb.mass_fractionation_factor(array, '108pd/105pd')
    0.09999999999999679

    See Also
    --------
    isopy.tb.add_mass_fractionation
    """
    #data = isopy.checks.check_array('data', data, 'isotope', 'ratio')

    fractionation_factor = isopy.checks.check_type('fractionation_factor', fractionation_factor, np.float64, np.ndarray,
                                                   coerce=True, coerce_into=[np.float64, np.array],allow_none=True)

    isotope_masses = isopy.refval(isotope_masses, isopy.RefValDict)

    if fractionation_factor is None:
        return data
    else:
        if data.flavour == 'isotope':
            if denom is None:
                denom = isopy.keymax(data)
            mass_array = isopy.array(isotope_masses.to_array(data.keys / denom), keys=data.keys)
        else:
            mass_array = isotope_masses.to_array(data.keys)

        return  data * (mass_array ** fractionation_factor)


def find_isobaric_interferences(main, interferences=None):
    """
    Find isobaric interferences for all isotopes in *main* for each element in *interferences*.

    Parameters
    ----------
    main
        A string, sequence of string or any object compatible with ``askeystring`` that returns a
        element or isotope key list. For element keys it will find interferences for all isotopes
        in *interferences* with that element symbol. If there are no isotopes in *interferences*
        with that element symbol it will find the isobaric interferences for all naturally occurring
        isotopes of that element.
    interferences
        A string, sequence of string or any object compatible with ``askeystring`` that returns a
        element or isotope key list. It will find the isobaric interferences for all naturally
        occurring isotopes of elements in *interferences*. If not given then it will find the
        isobaric interferences for isotopes in main of all naturally occurring elements.

    Returns
    -------
    isobaric_interferences
        A dictionary containing the all isobaric interferences of isotopes in *main* for elements
        in *interferences*.

    Examples
    --------
    >>> isopy.tb.find_isobaric_interferences('pd', ('ru', 'cd'))
    IsopyDict(default_value = (), readonly = False,
    {"Ru": IsotopeKeyList('102Pd', '104Pd')
    "Cd": IsotopeKeyList('106Pd', '108Pd', '110Pd')})

    >>> array = isopy.tb.make_ms_array('pd', '101ru', '111cd')
    >>> isopy.tb.find_isobaric_interferences('pd', array)
    IsopyDict(default_value = (), readonly = False,
    {"Ru": IsotopeKeyList('102Pd', '104Pd')
    "Cd": IsotopeKeyList('106Pd', '108Pd', '110Pd')})

    >>> isopy.tb.find_isobaric_interferences('ce')
    IsopyDict(default_value = (), readonly = False,
    {"Xe": IsotopeKeyList('136Ce')
    "Ba": IsotopeKeyList('136Ce', '138Ce')
    "La": IsotopeKeyList('138Ce')
    "Nd": IsotopeKeyList('142Ce')})

    >>> isopy.tb.find_isobaric_interferences('ce138')
    IsopyDict(default_value = (), readonly = False,
    {"Ba": IsotopeKeyList('138Ce')
    "La": IsotopeKeyList('138Ce')})
    """
    main = isopy.askeylist(main).flatten(ignore_duplicates=True)

    if len(main.flavour) > 1:
        raise ValueError('All keys must be either element or isotope')

    if interferences is not None:
        interferences = isopy.askeylist(interferences).flatten(ignore_duplicates=True)
        # Create a list of all the possible isotopes
        interference_isotopes = interferences.isotopes
    else:
        interference_isotopes = isopy.askeylist()


    if  main.flavour in isopy.asflavour('element|molecule[element]'):
        # Either ElementKeyList or a MoleculeKeyList with no isotopes
        main_isotopes = isopy.keylist()
        for element in main:
            if element in interference_isotopes.element_symbols:
                main_isotopes += interference_isotopes.filter(element_symbol_eq=element)
            else:
                main_isotopes += isopy.refval.element.isotopes.get(element, tuple())
    else:
        main_isotopes = main

    if interferences is None:
        interference_isotopes = isopy.keylist()
        for mass in main_isotopes.mass_numbers:
            interference_isotopes += isopy.refval.mass.isotopes.get(mass)

    interference_elements = isopy.askeylist(interference_isotopes.element_symbols, ignore_duplicates=True,
                                            flavour=('element', 'isotope', 'molecule'))

    result = isopy.IsopyDict(default_value=tuple())
    for element in interference_elements:
        interferences = element.isotopes()
        main = main_isotopes.filter(element_symbol_neq=element,
                                    mz_eq=interferences.mz)
        if len(main) > 0:
            result[element] = main
    return result


def remove_isobaric_interferences(data, isobaric_interferences, mf_factor=None,
                                   isotope_fractions='isotope.fraction', isotope_masses='isotope.mass'):
    """
    Remove all isobaric interferences for a given element(s).

    Parameters
    ----------
    data
        The data that isobaric interferences should be removed from.
    isobaric_interferences
        A dictionary mapping either the element or the isotope to be the isotopes that
        this element has isobaric interferences on that you wish to correct for. If the key is
        an element key string then the isotope of that element in the array with the largest
        signal will be used for the correction.
    mf_factor
        If given, this mass fractionation factor is applied to the values for for the correction.
    isotope_fractions
        Reference value for the isotope fractions of different elements.
        Defaults to ``isopy.refval.isotope.abundance``.
    isotope_masses
        Reference value for the isotope masses of different elements.
        Defaults to ``isopy.refval.isotope.mass``.

    Returns
    -------
    IsopyArray
        The interference corrected array

    Examples
    --------
    >>> array = isopy.tb.make_ms_array('pd', ru101 = 0.1, cd111 = 0.1)
    >>> array
    (row)      101Ru (f8)    102Pd (f8)    104Pd (f8)    105Pd (f8)    106Pd (f8)    108Pd (f8)    110Pd (f8)    111Cd (f8)
    -------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
    None          0.01706       0.04175       0.13002       0.22330       0.27455       0.26549       0.12968       0.01280
    IsopyNdarray(-1, flavour='isotope', default_value=nan)
    >>> interferences = isopy.tb.find_isobaric_interferences('pd', array)
    >>> interferences
    IsopyDict(readonly=False, key_flavour='any', default_value=())
    {ElementKeyString('Ru'): IsopyKeyList('102Pd', '104Pd', flavour='isotope'), ElementKeyString('Cd'): IsopyKeyList('106Pd', '108Pd', '110Pd', flavour='isotope')}
    >>> isopy.tb.remove_isobaric_interferences(array, interferences)
    (row)      101Ru (f8)    102Pd (f8)    104Pd (f8)    105Pd (f8)    106Pd (f8)    108Pd (f8)    110Pd (f8)    111Cd (f8)
    -------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
    None          0.00000       0.01020       0.11140       0.22330       0.27330       0.26460       0.11720       0.00000
    IsopyNdarray(-1, flavour='isotope', default_value=nan)
    >>> isopy.tb.make_ms_array('pd', ru101 = 0, cd111 = 0)
    (row)      101Ru (f8)    102Pd (f8)    104Pd (f8)    105Pd (f8)    106Pd (f8)    108Pd (f8)    110Pd (f8)    111Cd (f8)
    -------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
    None          0.00000       0.01020       0.11140       0.22330       0.27330       0.26460       0.11720       0.00000
    IsopyNdarray(-1, flavour='isotope', default_value=nan)

    >>> array = isopy.tb.make_ms_array('pd', ru101 = 0.1, cd111 = 0.1).ratio('106pd')
    >>> array
    (row)      101Ru/106Pd (f8)    102Pd/106Pd (f8)    104Pd/106Pd (f8)    105Pd/106Pd (f8)    108Pd/106Pd (f8)    110Pd/106Pd (f8)    111Cd/106Pd (f8)
    -------  ------------------  ------------------  ------------------  ------------------  ------------------  ------------------  ------------------
    None                0.06214             0.15207             0.47358             0.81333             0.96700             0.47236             0.04664
    IsopyNdarray(-1, flavour='ratio[isotope, isotope]', default_value=nan)
    >>> interferences = isopy.tb.find_isobaric_interferences('pd', array)
    >>> interferences
    IsopyDict(readonly=False, key_flavour='any', default_value=())
    {ElementKeyString('Ru'): IsopyKeyList('102Pd', '104Pd', flavour='isotope'), ElementKeyString('Cd'): IsopyKeyList('106Pd', '108Pd', '110Pd', flavour='isotope')}
    >>> isopy.tb.remove_isobaric_interferences(array, interferences)
    (row)      101Ru/106Pd (f8)    102Pd/106Pd (f8)    104Pd/106Pd (f8)    105Pd/106Pd (f8)    108Pd/106Pd (f8)    110Pd/106Pd (f8)    111Cd/106Pd (f8)
    -------  ------------------  ------------------  ------------------  ------------------  ------------------  ------------------  ------------------
    None                0.00000             0.03732             0.40761             0.81705             0.96817             0.42883             0.00000
    IsopyNdarray(-1, flavour='ratio[isotope, isotope]', default_value=nan)
    >>> isopy.tb.make_ms_array('pd', ru101 = 0, cd111 = 0).ratio('106pd')
    (row)      101Ru/106Pd (f8)    102Pd/106Pd (f8)    104Pd/106Pd (f8)    105Pd/106Pd (f8)    108Pd/106Pd (f8)    110Pd/106Pd (f8)    111Cd/106Pd (f8)
    -------  ------------------  ------------------  ------------------  ------------------  ------------------  ------------------  ------------------
    None                0.00000             0.03732             0.40761             0.81705             0.96817             0.42883             0.00000
    IsopyNdarray(-1, flavour='ratio[isotope, isotope]', default_value=nan)
    """

    #data = isopy.checks.check_array('data', data, 'isotope', 'ratio')

    mf_factor = isopy.checks.check_type('mf_factor', mf_factor, np.float64, np.ndarray, coerce=True,
                                        coerce_into=[np.float64, np.array], allow_none=True)
    isotope_fractions = isopy.refval(isotope_fractions, isopy.RefValDict)
    isotope_masses = isopy.refval(isotope_masses, isopy.RefValDict)

    # Get the information we need to make the correction that works
    # for both isotope arrays and ratio arrays
    if data.flavour == 'ratio':
        keys = data.keys()
        numer = keys.numerators
        denom = keys.common_denominator
    elif data.flavour == 'isotope':
        keys = data.keys()
        numer = keys
        denom = None

    if isinstance(isobaric_interferences, dict) and not isinstance(isobaric_interferences, core.IsopyDict):
        isobaric_interferences = core.IsopyDict(isobaric_interferences)
    elif not isinstance(isobaric_interferences, isopy.IsopyDict):
        raise ValueError('isobaric_interferences must be a dict')

    out = data.copy()

    for interference_element, correct_isotopes in isobaric_interferences.items():
        #If the key of the dictionary is an isotope use this isotope to make the correction
        if isopy.iskeystring(interference_element, flavour='isotope'):
            if interference_element not in numer:
                raise KeyError(f'isotope {interference_element} not in data')
            interference_isotope = interference_element
            interference_element = interference_isotope.element_symbol

        #If the key is an element key string use the isotope of that element in the array that has the largest signal.
        else:
            if data.flavour == 'ratio':
                interference_isotope = isopy.keymax(data.filter(numerator_element_symbol_eq=interference_element))
                interference_isotope = interference_isotope.numerator
            else:
                interference_isotope = isopy.keymax(data.filter(element_symbol_eq=interference_element))

        # Get the abundances of all isotopes of the interfering element
        inf_data = isotope_fractions.to_array(interference_isotope.element_symbol.isotopes)

        # Turn into a ratio relative to interference isotope
        inf_data = inf_data.ratio(interference_isotope, remove_denominator=False)

        # Account for mass fractionation of *mf_factor* is given
        if mf_factor is not None:
            inf_data = add_mass_fractionation(inf_data, mf_factor, isotope_masses=isotope_masses)

        # Scale relative to the measured value of *isotope*
        inf_data = inf_data * data[keys[numer.index(interference_isotope)]]

        # Convert to a mass array for easy lookup later
        inf_data = {key.numerator.mz: value for key, value in inf_data.items()}

        # Loop through each key and remove interference
        for i, key in enumerate(keys):
            if (numer[i] in correct_isotopes or numer[i].element_symbol == interference_element):
                out[key] = out[key] - inf_data.get(numer[i].mz, 0)

        # If *data* is a ratio array then correct for any interference on the denominator
        if denom is not None and denom in correct_isotopes:
            out = out / (1 - inf_data.get(denom.mz, 0))

    return out

@core.append_preset_docstring
@core.add_preset(('ppt', 'permil'), factor=1000)
@core.add_preset('epsilon', factor=1E4)
@core.add_preset(('mu', 'ppm'), factor=1E6)
def rDelta(data, reference_data, factor=1, deviations=1):
    """
    Externally normalise data to the given reference values.

    .. math::
        \Delta^{r} \\textrm{normalised} = (\\frac{\\textrm{data}} {\\textrm{reference data}} - \\textrm{deviations} ) * \\textrm{factor}


    Parameters
    ----------
    data
        Data to be normalised
    reference_data
        The reference values used to normalise the data. If a reference values contains more than
        one value the mean is used. Multiple values can be passed as a tuple in which case the
        mean of those values is used.
    factor
        The multiplication factor to be applied to *data* during the normalisation.
    deviations
        ``1`` if *reference_data* should be subtracted from *data*.

    Examples
    --------
    >>> ref = isopy.tb.make_ms_array('pd').ratio('105pd')
    >>> array = isopy.tb.make_ms_sample('pd', fins=1, fixed_voltage=0.1).ratio('105pd')
    >>> isopy.tb.rDelta(array, ref)
    (row)      102Pd/105Pd (f8)    104Pd/105Pd (f8)    106Pd/105Pd (f8)    108Pd/105Pd (f8)    110Pd/105Pd (f8)
    -------  ------------------  ------------------  ------------------  ------------------  ------------------
    0                  -0.03424            -0.00985             0.00929             0.02838             0.04719
    1                  -0.03133            -0.00905             0.00956             0.02905             0.04813
    2                  -0.02188            -0.00911             0.00908             0.02869             0.04882
    3                  -0.02505            -0.00954             0.00946             0.02845             0.04779
    4                  -0.02477            -0.00964             0.00937             0.02881             0.04688
                         ...                 ...                 ...                 ...                 ...
    95                 -0.02539            -0.00981             0.00991             0.02870             0.04741
    96                 -0.03228            -0.00967             0.00965             0.02834             0.04753
    97                 -0.02951            -0.01016             0.00947             0.02935             0.04818
    98                 -0.02557            -0.00915             0.00982             0.02870             0.04775
    99                 -0.02639            -0.00938             0.00919             0.02882             0.04825
    IsopyNdarray(100, flavour='ratio[isotope, isotope]', default_value=nan)

    >>> isopy.tb.rDelta(array, ref, factor=1E4)
    (row)      102Pd/105Pd (f8)    104Pd/105Pd (f8)    106Pd/105Pd (f8)    108Pd/105Pd (f8)    110Pd/105Pd (f8)
    -------  ------------------  ------------------  ------------------  ------------------  ------------------
    0                -342.37270           -98.49275            92.90093           283.76418           471.91203
    1                -313.33018           -90.46715            95.64488           290.54236           481.25233
    2                -218.81828           -91.11607            90.79193           286.86373           488.23901
    3                -250.54618           -95.42314            94.63710           284.45763           477.88220
    4                -247.71363           -96.39945            93.69597           288.07581           468.76867
                         ...                 ...                 ...                 ...                 ...
    95               -253.86651           -98.10488            99.10170           287.04024           474.05514
    96               -322.77188           -96.65319            96.45140           283.40338           475.31330
    97               -295.11884          -101.55413            94.69753           293.47311           481.82985
    98               -255.68137           -91.53948            98.16303           286.97835           477.46565
    99               -263.94694           -93.75097            91.94048           288.19007           482.45473
    IsopyNdarray(100, flavour='ratio[isotope, isotope]', default_value=nan)

    >>> isopy.tb.rDelta.epsilon(array, ref) # preset where factor = 1E4
    (row)      102Pd/105Pd (f8)    104Pd/105Pd (f8)    106Pd/105Pd (f8)    108Pd/105Pd (f8)    110Pd/105Pd (f8)
    -------  ------------------  ------------------  ------------------  ------------------  ------------------
    0                -342.37270           -98.49275            92.90093           283.76418           471.91203
    1                -313.33018           -90.46715            95.64488           290.54236           481.25233
    2                -218.81828           -91.11607            90.79193           286.86373           488.23901
    3                -250.54618           -95.42314            94.63710           284.45763           477.88220
    4                -247.71363           -96.39945            93.69597           288.07581           468.76867
                         ...                 ...                 ...                 ...                 ...
    95               -253.86651           -98.10488            99.10170           287.04024           474.05514
    96               -322.77188           -96.65319            96.45140           283.40338           475.31330
    97               -295.11884          -101.55413            94.69753           293.47311           481.82985
    98               -255.68137           -91.53948            98.16303           286.97835           477.46565
    99               -263.94694           -93.75097            91.94048           288.19007           482.45473
    IsopyNdarray(100, flavour='ratio[isotope, isotope]', default_value=nan)

    """
    data = isopy.checks.check_type('data', data, isopy.core.IsopyArray, coerce=True, coerce_into=isopy.core.asarray)
    factor = isopy.checks.check_type('factor', factor, np.float64, coerce=True)
    deviations = isopy.checks.check_type('deviations', deviations, np.float64, coerce=True)

    reference_data = _combine_reference_values(reference_data)

    new = data / reference_data
    new = new - deviations
    new = new * factor

    return new

@core.append_preset_docstring
@core.add_preset(('ppt', 'permil'), factor=1000)
@core.add_preset('epsilon', factor=1E4)
@core.add_preset(('mu', 'ppm'), factor=1E6)
def inverse_rDelta(data, reference_data, factor=1, deviations=1):
    """
    Denormalise data to the given reference values.

    .. math::
        \\textrm{denormalised data} = (\\frac{\Delta^{r} \\textrm{data}} {\\textrm{factor}} + \\textrm{deviations}) * \\textrm{reference data}


    Parameters
    ----------
    data
        Normalised data to be denormalised
    reference_data
        The reference values used to denormalise the data. If multiple values are passed
        the mean of all the reference values are used. If a reference values contains more than
        one value the mean is used.
    factor
        The multiplication factor applied to *data* during the normalisation.
    deviations
        ``1`` if *reference_data* should be added to the denormalised data.


    Examples
    --------
    >>> ref = isopy.tb.make_ms_array('pd').ratio('105pd')
    >>> array = isopy.tb.make_ms_sample('pd', fins=1).ratio('105pd')
    >>> norm = isopy.toolbox.isotope.rDelta(array, ref, 1E4)
    >>> isopy.tb.inverse_rDelta(norm, ref, 1E4)
    (row)      102Pd/105Pd (f8)    104Pd/105Pd (f8)    106Pd/105Pd (f8)    108Pd/105Pd (f8)    110Pd/105Pd (f8)
    -------  ------------------  ------------------  ------------------  ------------------  ------------------
    0                   0.04438             0.49412             1.23557             1.21884             0.54989
    1                   0.04438             0.49413             1.23560             1.21887             0.54989
    2                   0.04437             0.49410             1.23555             1.21879             0.54988
    3                   0.04437             0.49412             1.23555             1.21880             0.54989
    4                   0.04437             0.49410             1.23551             1.21880             0.54987
                         ...                 ...                 ...                 ...                 ...
    95                  0.04437             0.49411             1.23557             1.21882             0.54988
    96                  0.04437             0.49411             1.23557             1.21883             0.54988
    97                  0.04437             0.49413             1.23557             1.21886             0.54986
    98                  0.04437             0.49413             1.23555             1.21883             0.54987
    99                  0.04438             0.49413             1.23559             1.21884             0.54988
    IsopyNdarray(100, flavour='ratio[isotope, isotope]', default_value=nan)
    >>> isopy.tb.inverse_rDelta.epsilon(norm, ref) / array
    (row)      102Pd/105Pd (f8)    104Pd/105Pd (f8)    106Pd/105Pd (f8)    108Pd/105Pd (f8)    110Pd/105Pd (f8)
    -------  ------------------  ------------------  ------------------  ------------------  ------------------
    0                   1.00000             1.00000             1.00000             1.00000             1.00000
    1                   1.00000             1.00000             1.00000             1.00000             1.00000
    2                   1.00000             1.00000             1.00000             1.00000             1.00000
    3                   1.00000             1.00000             1.00000             1.00000             1.00000
    4                   1.00000             1.00000             1.00000             1.00000             1.00000
                         ...                 ...                 ...                 ...                 ...
    95                  1.00000             1.00000             1.00000             1.00000             1.00000
    96                  1.00000             1.00000             1.00000             1.00000             1.00000
    97                  1.00000             1.00000             1.00000             1.00000             1.00000
    98                  1.00000             1.00000             1.00000             1.00000             1.00000
    99                  1.00000             1.00000             1.00000             1.00000             1.00000
    IsopyNdarray(100, flavour='ratio[isotope, isotope]', default_value=nan)
    """
    data = isopy.checks.check_type('data', data, isopy.core.IsopyArray, coerce=True, coerce_into=isopy.core.asarray)
    factor = isopy.checks.check_type('factor', factor, np.float64, coerce=True)
    deviations = isopy.checks.check_type('deviations', deviations, np.float64, coerce=True)

    reference_data = _combine_reference_values(reference_data)

    new = data / factor
    new = new + deviations
    new = new * reference_data

    return new


def _combine_reference_values(values):
    if isinstance(values, dict):
        return values

    if not isinstance(values, (list, tuple)):
        values = (values,)
    else:
        values = values

    out = []
    for v in values:
        v = isopy.asarray(v)
        if v.size > 1:
            v = np.nanmean(v)
        out.append(v)

    if len(out) == 1:
        return isopy.RefValDict(out[0], ratio_function='divide')
    else:
        return isopy.RefValDict(np.mean(isopy.rstack(*out)), ratio_function='divide')