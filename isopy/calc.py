import numpy as np
from isopy.dtypes import *
import datetime as dt

#Rules
#Input can be dict or numpy but result is always numpy

################
### Finished ###
################


######################
### todo Docstring ###
######################


def isotope_to_ratio(data, denominator):
    """
    Converts an IsotopeArray into a RatioArray with a specified denominator.

    Parameters
    ----------
    data : IsotopeArray
        Data that is to be converted to an RatioArray
    denominator : IsotopeString
        Denominator to be used in ratios. Must be present in data.

    Returns
    -------
    RatioArray
    """
    if not isinstance(data, IsotopeArray): data = IsotopeArray(data)
    data = IsotopeArray(data)
    isotopes = data.keys()
    denominator = IsotopeString(denominator)

    if denominator not in isotopes: raise ValueError('denominator "{}" not found in data'.format(denominator))

    ratios = RatioList(['{}/{}'.format(iso, denominator) for iso in isotopes])

    new = RatioArray(len(data), keys = ratios)

    for rat in ratios:
        new[rat] = data[rat.numerator] / data[denominator]

    return new


def ratio_to_isotope(data, denominator_value=1):
    """
    Converts a ratio data array into isotope data array.

    Parameters
    ----------
    data : RatioArray
        Ratio data to be converted. All ratios must share the same denominator
    denominator_value : float
        Set denominator to this value and multiply all other values by it. Default = 1

    Returns
    -------
    IsotopeArray
        
    """
    if not isinstance(data, RatioArray): data = RatioArray(data)
    ratios = data.keys()

    numerators = ratios.get_numerators()
    denominator = ratios.get_common_denominator()

    isotopes = numerators.copy()
    if denominator not in isotopes: isotopes.append(denominator)

    new = IsotopeArray(len(data), keys = isotopes)

    for rat in ratios:
        new[rat.numerator] = data[rat] * denominator_value

    if denominator not in numerators: new[denominator] = denominator_value

    return new


def normalise_ratio_to_reference(data, reference_abu, factor = 1, subtract_one = True):
    if not isinstance(data, RatioArray): data = RatioArray(data)
    if not isinstance(reference_abu, (IsopyDict, RatioArray)): reference_abu = IsopyDict(reference_abu)

    if isinstance(factor, str):
        if factor.lower() in ['delta', 'ppt', 'permil']: factor = 1000
        elif factor.lower() in ['eps', 'epsilon']: factor = 10000
        elif factor.lower() in ['ppm', 'mu']: factor = 1000000
        else: raise ValueError('factor string not recognized.')

    new =  data / reference_abu
    if subtract_one: new = new - 1
    new = new * factor

    return new


def denomralise_ratio_from_reference(data, reference_abu, factor = 1, add_one = True):
    if not isinstance(data, RatioArray): data = RatioArray(data)
    if not isinstance(reference_abu, (IsopyDict, RatioArray)): reference_abu = IsopyDict(reference_abu)

    if isinstance(factor, str):
        if factor.lower() in ['delta', 'ppt', 'permil']: factor = 1000
        elif factor.lower() in ['eps', 'epsilon']: factor = 10000
        elif factor.lower() in ['ppm', 'mu']: factor = 1000000
        else: raise ValueError('factor string not recognized.')

    new = data / factor
    if add_one: new = new + 1
    new = new * reference_abu

    return new


def internal_normalisation(data, ratio, std_abu, std_mass, law='exponential'):
    """
    Correct data for mass fractionation using a fixed ratio and specified law.

    Parameters
    ----------
    data : numpy.ndarray
        Data array containing values to be correctd
    ratio : RatioString
        Ratio that will be fixed. Must be present in data
    std_abu : dict
        A dict containing the reference composition of the normalisation ratio.
    std_mass : dict
        A dict containing all the mass of all ratios in data.
    law : str
        The mass fractionation law to be used. Only exponential fractionation is supported. Default = 'exponential'

    Returns
    -------
    numpy.ndarray
        Normalised data array.
    """

    # Calculate the mass
    beta = calculate_mass_fractionation_factor(data, ratio, std_abu, std_mass, law)

    # Return corrected data
    return correct_mass_fractionation(data, beta, std_mass, law)


def correct_mass_fractionation(data, mf_factor, std_mass, law='exponential'):
    """
    Correct data using specified mass fractionation factor.

    Parameters
    ----------
    data : RatioArray
        Data array containing values to be corrected
    mf_factor : float
        Mass fractionation factor to be applied to data
    std_mass : IsopyDict
        A dict containing all the mass of all ratios in data.
    law : str
        The mass fractionation law to be used. Only exponential fractionation is supported. Default = 'exponential'

    Returns
    -------
    RatioArray
        Normalised data array.
    """
    if not isinstance(data, RatioArray): data = RatioArray(data)
    if not isinstance(std_mass, IsopyDict): std_mass = IsopyDict(std_mass)

    new = data.copy()
    if law == 'exponential':
        for ratio in RatioList(data):
            new[ratio] = data[ratio] / (std_mass[ratio] ** mf_factor)

    return new


def calculate_mass_fractionation_factor(data, ratio, std_abu, std_mass, law='exponential'):
    """
    Calculate the mass fractionation factor for a specified ratio.

    Parameters
    ----------
    data : RatioArray
        Data array containing values to be corrected
    ratio : RatioString
        Ratio that will be fixed. Must be present in data
    std_abu : IsopyDict
        A dict containing the reference composition of the normalisation ratio.
    std_mass : IsopyDict
        A dict containing all the mass of all ratios in data.
    law : str
        The mass fractionation law to be used. Only exponential fractionation is supported. Default = 'exponential'

    Returns
    -------
    RatioArray
        Normalised data array.
    """
    if not isinstance(data, RatioArray): data = RatioArray(data)
    if not isinstance(std_abu, IsopyDict): std_abu = IsopyDict(std_abu)
    if not isinstance(std_mass, IsopyDict): std_abu = IsopyDict(std_mass)
    if not isinstance(ratio, RatioString): ratio = RatioString(ratio)

    if law == 'exponential':
        return calculate_exponential_mass_fractionation_factor(data[ratio], std_abu[ratio], std_mass[ratio])


def calculate_exponential_mass_fractionation_factor(measured_abu, std_abu, std_mass):
    """
    Return the exponential mass fractionation factor.

    Parameters
    ----------
    measured_abu : float
        The measured abundance ratio between two isotopes
    std_abu : float
        The reference abundance ratio between two isotopes
    std_mass : float
        The mass ratio between two isotopes

    Returns
    -------
    float
        The exponential mass fractionation factor
    """
    return (np.log(measured_abu / std_abu) / np.log(std_mass))


def mix_isotopes(proportions, *isotope_data):
    """

    Parameters
    ----------
    proportions : list
        A list of the proportion of each isotope abundance to be mixed.
    isotope_data : numpy.ndarray
        A structured numpy array with each field corresponting to an isotope. Only those isotopes present in all
        given data arrays given will be used.

    Returns
    -------
    numpy.ndarray

    """
    if not isinstance(proportions, (list, tuple)): raise TypeError('proportions must be a list or a tuple')

    # Make sure that the
    plen = None
    for p in proportions:
        try:
            if plen is None:
                plen = len(p)
            elif plen != len(p):
                raise ValueError('plen')
        except (ValueError, TypeError):
            # p has no length
            pass
    if plen is None: plen = 1

    if len(isotope_data) < 2: raise ValueError('at least two different isotope abundances must be supplied')
    if len(proportions) > len(isotope_data): raise ValueError()

    isotopes = IsotopeList(isotope_data[0])

    for i in range(1, len(isotope_data)):
        isotopes = isotopes.filter(isotope_data[i])

    result = IsotopeArray(plen, keys = isotopes)

    for iso in isotopes:
        for i in range(len(proportions)):
            print(iso, proportions[i], isotope_data[i][iso], proportions[i] * isotope_data[i][iso])
            result[iso] += proportions[i] * isotope_data[i][iso]

    return result


def york_regression(X, Y, Xerr, Yerr, r=0, tol=0.00001):
    # Based on the formulas given in York et al 2014
    # Check input
    if not isinstance(r, (float, int)): raise ValueError('r must be an integer or float')
    if r > 1 or r < 0: raise ValueError('r must be between 0 and 1')
    if not isinstance(tol, float): raise ValueError('tol must be a float')

    if not isinstance(X, np.ndarray): X = np.array(X)
    X = X.flatten()

    if not isinstance(Y, np.ndarray): Y = np.array(Y)
    Y = Y.flatten()

    if X.size != Y.size: raise ValueError('X and Y must have the same size')

    if isinstance(Xerr, (float, int)):
        Xerr = np.ones(X.size) * Xerr
    else:
        if not isinstance(Xerr, np.ndarray): Xerr = np.array(Xerr)
        Xerr = Xerr.flatten()
        if Xerr.size != X.size: raise ValueError('Xerr and X must have the same size')

    if isinstance(Yerr, (float, int)):
        Yerr = np.ones(Y.size) * Yerr
    else:
        if not isinstance(Yerr, np.ndarray): Yerr = np.array(Yerr)
        Yerr = Yerr.flatten()
        if Yerr.size != Y.size: raise ValueError('Yerr and Y must have the same size')

    # Do calculations
    WX = 1 / (Xerr ** 2)
    WY = 1 / (Yerr ** 2)

    k = np.sqrt(WX * WY)
    slope = (np.mean(X * Y) - np.mean(X) * np.mean(Y)) / (np.mean(X ** 2) - np.mean(X) ** 2)

    # Iterativley converge self.slope until tol is reached
    for i in range(1000):
        W = (WX * WY) / (WX + (slope ** 2) * WY - 2 * slope * r * k)
        Xbar = np.sum(W * X) / np.sum(W)
        Ybar = np.sum(W * Y) / np.sum(W)
        U = X - Xbar
        V = Y - Ybar
        beta = W * (U / WY + (slope * V) / WX - (slope * U + V) * (r / k))

        b2 = np.sum(W * beta * V) / np.sum(W * beta * U)
        dif = np.sqrt((b2 / slope - 1) ** 2)
        slope = b2
        if dif < tol: break

    # Calculate self.intercept
    intercept = Ybar - slope * Xbar

    # Calcualte adjusted points
    x = Xbar + beta
    xbar = np.sum(W * x) / np.sum(W)
    u = x - xbar

    y = Ybar + slope * beta
    ybar = np.sum(W * y) / np.sum(W)
    v = y - ybar

    slope_error = np.sqrt(1 / np.sum(W * (u ** 2)))
    intercept_error = np.sqrt(1 / np.sum(W) + ((0 - xbar) ** 2) * (slope_error ** 2))

    S = np.sum(W * (Y - slope * X - intercept) ** 2)

    # TODO add MSDW
    return YorkRegression(slope, slope_error, intercept, intercept_error, np.sum(W), xbar)

class YorkRegression:
    def __init__(self, slope, slope_error, intercept, intercept_error, sumW, xbar):
        self.slope = slope
        self.slope_error = slope_error
        self.intercept = intercept
        self.intercept_error = intercept_error
        self.sumW = sumW
        self.xbar = xbar

    def __repr__(self):
        return 'y=({} +- {})x + ({} +- {}); xbar = {}, sum(W) = {}'.format(self.slope, self.slope_error,
                                                                           self.intercept, self.intercept_error,
                                                                           self.xbar, self.sumW)

    def y_error(self, x):
        return np.sqrt(1 / self.sumW + ((x - self.xbar) ** 2) * (self.slope_error ** 2))




###################
### In progress ###
###################


def isobaric_interference_correction(data, ratio, abundance, mass = None, mf_factor = None, mf_law = 'exponential'):
    """
    Removed the isobaric contribution to all isotopes with the same mass as the specified ratios numerator. 
        
    Parameters
    ----------
    data : numpy.ndarray
        Must 
    ratio : RatioString
        A ratio where the numerator is the isotope whose contribution to other isotopes of the same mass will be removed.
        The denominator is the isotope that will we used to calculate contribution of the numerator. Denominator must be
        present in data.
    abundance : value, dict
        Can either be a value of the ratio specified or a dict of values where the ratio value can be found.
    mass : value, dict
        Only required if mass fractionation correction is required. Can either be a value of the ratio specified or a dict of values
        where the ratio value can be found.
    mf_factor : value
        If specified then the this value will be used to correct the mass fractionation. Will be applied to the
        abundance value of the ratio specified. This if used data should not have been internally normalised .  
    mf_law : str
        Mass fractionation law to be used. Defaults to the exponential law

    Returns
    -------

    """
    ratio = RatioString(ratio)

    names = any_list(data.dtype.names)
    if isinstance(names, IsotopeList):
        numer = names
        denom = None
    else:
        #Must be RatioList
        if not names.has_common_denominator(): raise ValueError('all ratios in data must share a common denominator')
        numer = names.get_numerators()
        denom = names.get_denominators()

    try:
        value = data[names[numer.index(ratio.denominator)]]
    except:
        raise ValueError('{} not found in data'.format(ratio.denominator))

    if isinstance(abundance, dict):
        value = value * get_isorat_from_dict(ratio, abundance)
    else:
        value = value * abundance

    if mass is None and mf_factor is not None:
        raise ValueError('mass must be given if mass fractionation is required')
    elif mf_law == 'exponential':
        if isinstance(mass, dict):
            value = value * (get_isorat_from_dict(ratio, mass) ^ mf_factor)
        else:
            value = value * (mass ^ mf_factor)
    else:
        raise ValueError('Invalid mass fractionation law')

    for i in range(len(data_fields)):
        # For isotopes or the numeratior if data is ratio
        if ratio.numerator.A == numer[i].A:
            data[data_fields[i]] = data[data_fields[i]] - value

            if denom is not None:
                if ratio.numerator.A == denom[i].A:
                    data[data_fields[i]] = data[data_fields[i]] / value
    return data


###################
### Not started ###
###################
def MAD(array, k = 1.4826):
    return np.median(np.abs(array - np.median(array))) * k

def outlier_rejection(data, rejection_threshold = 3, k = 1.4826):
    outliers = outlier_index(data, rejection_threshold, k)

    return data[outliers]


def outlier_index(data, rejection_threshold = 3, k = 1.4826):
    try:
        median = np.median(data)
        mad = MAD(data, k)
        return np.all(data < (median + mad * rejection_threshold), data > (median - mad* rejection_threshold))
    except TypeError:
        try:
            names = data.dtype.names
            index = []
            for name in names:
                median = np.median(data[name])
                mad = MAD(data[name], k)
                index.append(np.all([data[name] < (median + mad* rejection_threshold), data[name] > (median - mad* rejection_threshold)], 0))
            return np.all(index,0)
        except: raise


def DS_inversion_siebert():
    pass


def DS_inversion_rudge():
    pass






