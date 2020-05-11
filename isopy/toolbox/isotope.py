import numpy as _np
import isopy as _isopy

__all__ = ['make_sample', 'remove_mass_fractionation', 'add_mass_fractionation',
           'calculate_mass_fractionation_factor', 'mass_independent_correction',
           'add_isobaric_interferences', 'remove_isobaric_interferences']

"""
Functions for isotope data reduction
"""

def make_sample(element, interference_isotopes = None, interference_values= None, integrations = 100,
                integration_time=4, maxv = 10, fins=None, fnat=None, sprocess=None, std_abu=None, std_mass = None,
                sprocess_fractions = None):
    """
    Returns a simulated measurement for all isotopes of *element*.

    The measured values are calculated using a normal distribution of the Johnson-Nyquist noise and counting statistics.

    The sample is created in the following order:

    * s-process contributions added

    * natural fractionation is added

    * isobaric interferences are added

    * instrumental fractionation is added

    * Johnson-Nyquist noise is added

    Parameters
    ----------
    element : ElementString
        All isotopes of this element will be included in the returned array.
    interference_isotopes : IsotopeList, optional
        If given then each isotope in *interference_isotopes* is added to the returned array and the isobaric
        interferences of the element is added to all isotopes in the array.
    interference_values : list of floats, optional
        The value of each isotope in *interference_isotope* when isobaric interferences is calculated. Must be given
        if *interference_isotopes* is given.
    integrations : int, optional
        The number of integrations in the returned array. Default value is ``100``
    integration_time : float, optional
        The integration time for each integration in the returned array. Used to calculate the Johnson_Nyquist noise.
        Default value is ``4.0``
    maxv : float, optional
        The voltage of the *element* isotopes with the highest abundance. Default value is ``10.0``
    fins : float, optional
        If given then an instrumental fractionation factor of *fins* is added to the data.
    fnat : float, optional
        If given then an natural fractionation factor of *fnat* is added to the data.
    sprocess : float, optional
        If given then *sprocess* amounts of s-process isotopes are added to the data. Given in permil.
    std_abu : IsotopeArray, dict, optional
        An object with a ".get(item, default_value)" method containing the abundance of each isotope/ratio needed for
        this function. Default is the "best isotope fraction" reference values.
    std_mass : IsotopeArray, dict, optional
        An object with a ".get(item, default_value)" method containing the masses of each isotope/ratio needed for
        this function. Default is the "isotope mass" reference values.
    sprocess_fractions : IsotopeArray, dict, optional
        An object with a ".get(item, default_value)" method containing the s-process fraction of each isotope needed
        for this function. Default is the "sprocess isotope fraction" reference values.

    Returns
    -------
    IsotopeArray
        An array with size of *integrations*
    """
    #Check input
    element = _isopy.core.check_type('element', element, _isopy.core.ElementString, coerce=True)
    interference_isotopes = _e.check_type('interference_isotopes', interference_isotopes, _isopy.core.IsotopeList,
                                          coerce=True, allow_none=True)
    interference_values = _e.check_type_list('interference_values', interference_values, _np.float, coerce=True,
                                             make_list=True, allow_none=(interference_isotopes is None))
    if interference_isotopes is not None and len(interference_values) != len(interference_isotopes):
        raise ValueError('size of "interference_isotopes" and "interference_values" do not match')
    integrations = _e.check_type('integrations', integrations, int, coerce=True)
    integration_time = _e.check_type('integration_time', integration_time, _np.float, float, coerce=True)
    maxv = _e.check_type('maxv', maxv, _np.float, float, coerce=True)
    fins = _e.check_type('fins', fins, _np.float, float, coerce=True, allow_none=True)
    fnat = _e.check_type('fnat', fnat, _np.float, float, coerce=True, allow_none=True)
    sprocess = _e.check_type('sprocess', sprocess, _np.float, float, coerce=True, allow_none=True)
    std_abu = _e.check_reference_value('std_abu', std_abu, 'best isotope fraction')
    std_mass = _e.check_reference_value('std_mass', std_mass, 'isotope mass')
    sprocess_fractions = _e.check_reference_value('sprocess_fractions', sprocess_fractions,
                                                          'sprocess isotope fraction')

    data = std_abu.get(std_abu.isotope_keys().copy(element_symbol=element)).reshape(1)
    denom = _np.argmax(data, axis=None)

    #add s-process anomaly
    if sprocess is not None: data = data + (data * sprocess_fractions.get(data.keys(), 0) * (sprocess / 1000))

    # add natural fractionation
    rat = data.ratio(denom)
    if fnat is not None: rat = add_mass_fractionation(rat, fnat)

    #Add isobaric interferences
    if interference_isotopes is not None:
        for i in range(len(interference_isotopes)):
            isotope = interference_isotopes[i]
            value = interference_values[i] / maxv
            rat = add_isobaric_interferences(rat, isotope, value,std_abu=std_abu, std_mass=std_mass)


    #add intrumental fractionation
    if fins is not None: rat = add_mass_fractionation(rat, fins)

    #convert back into isotopes and set max voltage
    data = rat.deratio(maxv)

    #calculate jk noise
    noise = _isopy.toolbox.misc.johnson_nyquist_noise(data, integration_time=integration_time)
    #create measured data
    random_generator = _np.random.default_rng()
    measured_data = _isopy.core.IsotopeArray(integrations, keys=data.keys())
    for key in measured_data.keys():
        measured_data[key] = random_generator.normal(data[key][0], noise[key][0], integrations)

    return measured_data


def mass_independent_correction(data, mf_ratio, normalisation_factor=None, std_abu=None, std_mass=None):
    """
    A quick function for mass-independent data correction.

    The data is corrected for mass fractionation, isobaric interferences and finally, if *normalisation_factor* is given,
    normalised to the values in *std_abu*. If *normalisation_factor* is not given the unnormalised corrected data is
    returned.

    An interference correction will be applied for all isotopes that are different from the *mf_ratio* numerator
    element. This will be done together with the mass fractionation correction to account for isobaric interferences
    on the *mf_ratio*.

    Parameters
    ----------
    data : IsotopeArray
        The data to be corrected.
    mf_ratio : RatioString
        The data will be internally normalised to this ratio.
    normalisation_factor : float, optional
        If given the corrected data is passed to :func:`normalise_data` together with the *std_abu* and with
        *normalisation_factor*.
    std_abu : dict
        An object with a ".get(item, default_value)" method containing the abundance of each isotope/ratio needed for
        this function. Default is the "best isotope fraction" reference values.
    std_mass : dict
        An object with a ".get(item, default_value)" method containing the masses of each isotope/ratio needed for
        this function. Default is the "isotope mass" reference values.

    Returns
    -------
    RatioArray
        The data corrected for mass-independent variation. Only the isotopes of the element in *mf_ratio* is returned.

    """
    data = _isopy.core.check_type('data', data, _isopy.core.IsotopeArray, coerce=True)
    mf_ratio = _isopy.core.check_type('mf_ratio', mf_ratio, _isopy.core.RatioString, coerce=True)
    normalisation_factor = _isopy.core.check_type('normalisation_factor', normalisation_factor, _np.float, str, coerce=True, allow_none=True)
    std_abu = _isopy.core.check_reference_value('std_abu', std_abu, 'best isotope fraction')
    std_mass = _isopy.core.check_reference_value('std_mass', std_mass, 'isotope mass')

    #Find the isotopes that can cause isobaric interferences.
    interference_isotopes = data.keys().copy(element_symbol_not=mf_ratio.numerator.element_symbol)

    #Convert the data into a ratio array
    rat = data.ratio(mf_ratio.denominator)
    beta = calculate_mass_fractionation_factor(rat, mf_ratio, std_abu, std_mass)

    #Do a combined mass fractionation and isobaric interference correction.
    #This can account for isobaric interferences on isotopes in *mf_ratio*
    prev_beta = beta
    for i in range(10):
        rat2 = rat
        for infiso in interference_isotopes:
            rat2 = remove_isobaric_interferences(rat2, infiso, beta, std_abu, std_mass)

        # Calculate the mass fractionation.
        beta = calculate_mass_fractionation_factor(rat2, mf_ratio, std_abu, std_mass)

        if beta/prev_beta < 0.0001:
            break #Beta value has converged so no need for more iterations.

    #Remove the isotopes on interfering elements
    rat = rat2.copy(element_symbol= mf_ratio.numerator.element_symbol)

    #Correct for mass fractionation
    rat = remove_mass_fractionation(rat, beta, std_mass)

    if normalisation_factor is not None:
        #Normalise the corrected data relative to *std_abu* and return
        rat =  _isopy.toolbox.general.normalise_data(rat, std_abu, normalisation_factor)
        rat =  _isopy.toolbox.general.normalise_data(rat, std_abu, normalisation_factor)

    # Return the corrected data
    return rat


def calculate_mass_fractionation_factor(data, ratio, std_abu=None, std_mass=None):
    """
    Calculate the mass fractionation factor for a given ratio in *data*.

    .. math::
        \\alpha = \\ln{( \\frac{r_{n,d}}{R_{n,d}} )} * \\frac{1}{ \\ln{( m_{n,d} } )}

    where :math:`n` is the numerator isotope and :math:`d` is the denominator isotope.

    Parameters
    ----------
    data : RatioArray
        Fractionated data. :math:`r` in the equation above.
    ratio : RatioString
        The isotope ratio from which the fractionation factor should be calculated. :math:`n,d` in the equation above.
    std_abu : RatioArray, dict, optional
        An object with a ".get(item, default_value)" method containing the abundance of each isotope/ratio needed for
        this function. Default is the "best isotope fraction" reference values. :math:`R` in the equation above.
    std_mass : RatioArray, dict, optional
        An object with a ".get(item, default_value)" method containing the masses of each isotope/ratio needed for
        this function. Default is the "isotope mass" reference values. :math:`m` in the equation above.

    Returns
    -------
    float
        The fractionation factor for *ratio* in *data*. :math:`\\alpha` in the equation above.
    """
    data = _isopy.core.check_type('data', data, _isopy.core.RatioArray, coerce=True)
    ratio = _isopy.core.check_type('ratio', ratio, _isopy.core.RatioString, coerce=True)
    std_abu = _isopy.core.check_reference_value('std_abu', std_abu, 'best isotope fraction')
    std_mass = _isopy.core.check_reference_value('std_mass', std_mass, 'isotope mass')

    return (_np.log(data.get(ratio) / std_abu.get(ratio)) / _np.log(std_mass.get(ratio)))


def remove_mass_fractionation(data, fractionation_factor, std_mass=None):
    """
    Remove exponential mass fractionation from the data.

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
    std_mass : RatioArray, dict, optional
        An object with a ".get(item, default_value)" method containing the masses of each isotope/ratio needed for
        this function. Default is the "isotope mass" reference values. :math:`m` in the equation above

    Returns
    -------
    RatioArray
        The corrected data. :math:`R` in the equation above.
    """
    data = _isopy.core.check_type('data', data, _isopy.core.RatioArray, coerce=True)
    fractionation_factor = _isopy.core.check_type('fractionation_factor', fractionation_factor, _np.float, _np.ndarray,
                                         coerce=True, coerce_into=[_np.float, _np.array])
    std_mass = _isopy.core.check_reference_value('std_mass', std_mass, 'isotope mass')

    return data / (std_mass.get(data.keys()) ** fractionation_factor)


def add_mass_fractionation(data, fractionation_factor, std_mass=None):
    """
    Add exponential mass fractionation to the data.

    Calculated using:

    .. math::
        R_{n,d} = r_{n,d} * (m_{n,d})^{ \\alpha }

    where :math:`n` is the numerator isotope and :math:`d` is the denominator isotope.

    Parameters
    ----------
    data : RatioArray
        Array containing data to be changed. :math:`r` in the equation above.
    fractionation_factor : float
        Fractionation factor to be applied. :math:`\\alpha` in the equation above
    std_mass : RatioArray, dict, optional
        An object with a ".get(item, default_value)" method containing the masses of each isotope/ratio needed for
        this function. Default is the "isotope mass" reference values. :math:`m` in the equation above

    Returns
    -------
    RatioArray
        The corrected data. :math:`R` in the equation above.
    """
    data = _isopy.core.check_type('data', data, _isopy.core.RatioArray, coerce=True)
    fractionation_factor = _isopy.core.check_type('fractionation_factor', fractionation_factor, _np.float, _np.ndarray,
                                         coerce=True, coerce_into=[_np.float, _np.array])
    std_mass = _isopy.core.check_reference_value('std_mass', std_mass, 'isotope mass')

    return data * (std_mass.get(data.keys()) ** fractionation_factor)


def remove_isobaric_interferences(data, isotope, mf_factor = None, std_abu = None, std_mass=None):
    """
    Remove all isobaric interferences for a given element.

    Parameters
    ----------
    data : IsotopeArray, RatioArray
        The data that isobaric interferences should be removed from.
    isotope : IsotopeString
        The isotope of the interfering element that should be used for the correction. Must be present in *data*
    mf_factor : float, optional
        If given then this amount of exponential mass fractionation is added to the isobaric interferences values before
        they are removed from *data*.
    std_abu : IsotopeArray, dict
        An object with a ".get(item, default_value)" method containing the abundance of each isotope/ratio needed for
        this function.Default is the "best isotope fraction" reference values.
    std_mass : RatioArray, dict
        An object with a ".get(item, default_value)" method containing the masses of each isotope/ratio needed for
        this function. Default is the "isotope mass" reference values.

    Returns
    -------
    IsotopeArray or RatioArray
        Data minus the isobaric interferences.
    """

    data = _isopy.core.check_type('data', data, _isopy.core.RatioArray, coerce=True)
    isotope = _isopy.core.check_type('isotope', isotope, _isopy.core.IsotopeString, coerce=True)
    mf_factor = _isopy.core.check_type('mf_factor', mf_factor, _np.float, _np.ndarray, coerce=True,
                              coerce_into=[_np.float, _np.array], allow_none=True)
    std_abu = _isopy.core.check_reference_value('std_abu', std_abu, 'best isotope fraction')
    std_mass = _isopy.core.check_reference_value('std_mass', std_mass, 'isotope mass')

    #Get the information we need to make the correction that works
    # for both isotope arrays and ratio arrays
    if isinstance(data, _isopy.core.RatioArray):
        keys = data.keys()
        numer = keys.numerators()
        denom = keys.get_common_denominator()
    elif isinstance(data, _isopy.core.IsotopeArray):
        keys = data.keys()
        numer = keys
        denom = None
    else:
        raise TypeError('variable "data" must be a IsotopeArray or a RatioArray not {}'.format(data.__class__.__name__))

    #Get the abundances of all isotopes of the interfering element
    inf_data = std_abu.get(std_abu.isotope_keys().copy(element_symbol=isotope.element_symbol))

    #Turn into a ratio relative to *isotope*
    inf_data = inf_data.ratio(isotope)

    #Account for mass fractionation of *mf_factor* is given
    if mf_factor is not None: inf_data = add_mass_fractionation(inf_data, mf_factor, std_mass=std_mass)

    #Scale relative to the measured value of *isotope*
    inf_data = inf_data * data[keys[numer.index(isotope)]]

    #Convert to a mass array for easy lookup later
    inf_data = _dt.MassArray(inf_data, keys=inf_data.keys().numerators().mass_numbers())

    #Create the output array
    out = _dt.array(data.nrows, keys=keys)

    #Loop through each key in *out* and remove interference
    for i in range(len(keys)):
        out[keys[i]] = data[keys[i]] - inf_data.get(numer[i].mass_number, 0)

    #If *data* is a ratio array then correct for any interference on the denominator
    if denom is not None:
        out = out / (1 - inf_data.get(denom.mass_number, 0))

    return out


def add_isobaric_interferences(data, isotope, value, mf_factor = None, std_abu=None, std_mass=None):
    """
    Add isobaric interferences of a given element to the data.

    Parameters
    ----------
    data : IsotopeArray, RatioArray
        The data that isobaric interferences should be added to.
    isotope : IsotopeString
        The isotope of the interfering element that should be used to add isobaric interferences. If not already
        present, a columns with this key will be added to the data.
    value : float
        This amount of *isotope* will be added to the data. The contribution from the other isotopes of the element
        is scaled to this value.
    mf_factor : float, optional
        If given then this amount of exponential mass fractionation is added to the isobaric interferences values before
        they are added to *data*.
    std_abu : IsotopeArray, dict
        An object with a ".get(item, default_value)" method containing the abundance of each isotope/ratio needed for
        this function.Default is the "best isotope fraction" reference values.
    std_mass : RatioArray, dict
        An object with a ".get(item, default_value)" method containing the masses of each isotope/ratio needed for
        this function. Default is the "isotope mass" reference values.

    Returns
    -------
    IsotopeArray or RatioArray
        Data plus the isobaric interferences.
    """
    data = _isopy.core.check_type('data', data, _isopy.core.RatioArray, coerce=True)
    isotope = _isopy.core.check_type('isotope', isotope, _isopy.core.IsotopeString, coerce=True)
    mf_factor = _isopy.core.check_type('mf_factor', mf_factor, _np.float, _np.ndarray, coerce=True,
                              coerce_into=[_np.float, _np.array], allow_none=True)
    std_abu = _isopy.core.check_reference_value('std_abu', std_abu, 'best isotope fraction')
    std_mass = _isopy.core.check_reference_value('std_mass', std_mass, 'isotope mass')

    # Get the information we need to make the correction that works
    # for both isotope arrays and ratio arrays
    if isinstance(data, _isopy.core.RatioArray):
        keys = data.keys()
        numer = keys.numerators()
        denom = keys.get_common_denominator()
        out_keys = keys
        #Make sure that *isotope* is part of the returned array
        if isotope not in numer:
            numer.append(isotope)
            out_keys.append(isotope/denom)
    elif isinstance(data, _isopy.core.IsotopeArray):
        keys = data.keys()
        numer = keys
        denom = None
        out_keys = keys
        # Make sure that *isotope* is part of the returned array
        if isotope not in numer:
            numer.append(isotope)
            out_keys.append(isotope)
    else:
        raise TypeError('variable "data" must be a IsotopeArray or a RatioArray not {}'.format(data.__class__.__name__))

    # Get the abundances of all isotopes of the interfering element
    inf_data = std_abu.get(std_abu.isotope_keys().copy(element_symbol=isotope.element_symbol))

    # Turn into a ratio relative to *isotope*
    inf_data = inf_data.ratio(isotope)

    # Account for mass fractionation of *mf_factor* is given
    if mf_factor is not None: inf_data = add_mass_fractionation(inf_data, mf_factor, std_mass=std_mass)

    # Scale relative to the measured value of *isotope*
    inf_data = inf_data * value

    # Convert to a mass array for easy lookup later
    inf_data = _isopy.core.MassArray(inf_data, keys=inf_data.keys().numerators().mass_numbers())

    # Create the output array

    out = _isopy.core.asarray(data.nrows, keys=out_keys)

    # Loop through each key in *out* and add interference
    for i in range(len(out_keys)):
        out[out_keys[i]] = data.get(out_keys[i], 0) + inf_data.get(numer[i].mass_number, 0)

    # If *data* is a ratio array then correct for any interference on the denominator
    if denom is not None:
        out = out / (1 + inf_data.get(denom.mass_number, 0))

    return out



