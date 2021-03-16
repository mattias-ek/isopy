import numpy as np
import scipy.optimize as optimize
import isopy as isopy
import itertools as itertools
from collections import namedtuple as _namedtuple
from matplotlib import cm as mplcm
from matplotlib import colors as mplcolors
from isopy import core
from . import isotope
from datetime import datetime as dt

__all__ = ['ds_inversion', 'ds_correction']

import isopy.checks

"""
Functions for doublespike data reduction
"""
_DSResult = _namedtuple('DSResult', ('method', 'lambda_', 'alpha', 'beta', 'fnat', 'fins', 'spike_faction',
                                     'sample_fraction', 'Q'))

def _inversion_rudge_function( x, P, n, T, m):
    x = x.reshape(3,-1)
    lambda_ = x[0]
    alpha = x[1]/(1-lambda_)
    beta = x[2]

    return ((lambda_ * T) + ((1-lambda_) * n * np.exp(-alpha * P)) - m * np.exp(-beta * P)).flatten()

#Currently no used as it doesnt work when the input has more than one dimension.
#Might just be a matter of rearranging the output
def _inversion_rudge_jacobian( x, P, n, T, m):

    x = x.reshape(3,-1)
    lambda_ = x[0]
    alpha = x[1]/(1-lambda_)
    beta = x[2]
    output = np.transpose([T - (n * np.exp(-alpha * P) * (1 + alpha * P)), -n * P * np.exp(-alpha * P), m * P * np.exp(-beta * P)])
    return output

#split if more than x so that it doesnt break
def ds_inversion_rudge(m, n, T, P, xtol = 1e-10):
    """
    Double spike inversion using the method of Rudge et al. 2009, Chemical Geology 265 420-431.

    Parameters
    ----------
    self
    m : ndarray
        Measured isotope ratios. Must be an 1-dim array of size 3 or a 2-dim array with first dimension of 3.
    n : ndarray
        Reference isotope ratios. Must be a 1-dim array of size 3 or a 2-dim array with the same shape as `m`.
    T : ndarray
        Spike isotope ratios. Must be a 1-dim array of size 3 or a 2-dim array with the same shape as `m`.
    P : ndarray
        Log of mass isotope ratios. Must be a 1-dim array of size 3 or a 2-dim array with the same shape as `m`.
    xtol : float
        Required tolerance for inversion. Default value is 1E-10

    Returns
    -------
    DSResult
    """
    mndim = m.ndim
    if m.size == 3: m = m.reshape(3,1)
    elif m.ndim == 1: raise ValueError()
    elif m.ndim == 2:
        if m.shape[0] != 3: raise ValueError()
    else: raise ValueError('m has to many dimensions ({})'.format(m.ndim))

    if n.size == 3: n = np.ones(m.shape) * n.reshape(3, 1)
    elif n.shape != m.shape: raise ValueError('Shape of n {} does not match shape of m {}'.format(n.shape, m.shape))

    if T.size == 3: T = np.ones(m.shape) * T.reshape(3, 1)
    elif T.shape != m.shape: ValueError('Shape of T {} does not match shape of m {}'.format(T.shape, m.shape))

    if P.size == 3: P = np.ones(m.shape) * P.reshape(3, 1)
    elif P.shape != m.shape: ValueError('Shape of P {} does not match shape of m {}'.format(P.shape, m.shape))

    xtol = isopy.checks.check_type('xtol', xtol, np.float_, float, coerce=True)

    A = np.transpose([T - n, -n * P, m * P])
    b = np.transpose(m - n)
    x0 = np.transpose(np.linalg.solve(A, b))

    x, infodict, ier, mesg =  optimize.fsolve(_inversion_rudge_function, x0, (P, n, T, m), None, True, xtol = xtol)

    if ier != 1:
        x = np.ones(m.shape) * np.nan

    if mndim == 2: x = x.reshape(3,-1)
    lambda_ = x[0]
    alpha = x[1]/(1-lambda_)
    beta = x[2]

    Fnat = alpha * -1
    Fins = beta

    spike_fraction = (1 + ((1-lambda_)/lambda_) * ((1 + np.sum((n / np.exp(alpha * P)), axis = 0)) / (1 + np.sum(T, axis = 0)))) ** (-1)
    sample_fraction = (1-spike_fraction)
    Q = sample_fraction / spike_fraction

    return DSResult('rudge', alpha, beta, lambda_, Fnat, Fins, spike_fraction, sample_fraction, Q)

def ds_inversion_siebert(MS, ST, SP, Mass, Fins_guess = 2, Fnat_guess = -0.00001, outer_loop_iterations = 3, inner_loop_iterations=6):
    """
    Double spike inversion using the method of Siebert et al. 2001, Geochemistry, Geophysics, Geosystems 2,

    Parameters
    ----------
    self
    MS : ndarray
        Measured isotope ratios. Must be an 1-dim array of size 3 or a 2-dim array with first dimension of 3.
    ST : ndarray
        Reference isotope ratios. Must be a 1-dim array of size 3 or a 2-dim array with the same shape as `MS`.
    SP : ndarray
        Spike isotope ratios. Must be a 1-dim array of size 3 or a 2-dim array with the same shape as `MS`.
    Mass : ndarray
        Log of mass isotope ratios. Must be a 1-dim array of size 3 or a 2-dim array with the same shape as `MS`.
    Fins_guess : float
        Starting guess for for the instrumental fractionation factor. Default value is 2
    Fnat_guess : float
        Starting guess for the natural fractionation factor. Default value is -0.00001
    outer_loop_iterations : int
        Number of iterations of the outer loop. Default value is 3
    inner_loop_iterations
        Number of iterations of the inner loop. Default value is 6

    Returns
    -------
    DSResult
    """
    output_index = 0
    mndim = MS.ndim
    if MS.size == 3:
        MS = MS.reshape(3, 1)
    elif MS.ndim == 1: raise ValueError()
    elif MS.ndim == 2:
        if MS.shape[0] != 3: raise ValueError()
    else: raise ValueError('MS has to many dimensions ({})'.format(MS.ndim))

    if ST.size == 3: ST = ST.reshape(3)
    elif ST.shape != MS.shape: raise ValueError('Shape of ST {} does not match shape of MS {}'.format(ST.shape, MS.shape))

    if ST.size == 3: ST = ST.reshape(3)
    elif ST.shape != MS.shape: ValueError('Shape of ST {} does not match shape of MS {}'.format(ST.shape, MS.shape))

    if Mass.size == 3: Mass = Mass.reshape(3)
    elif Mass.shape != MS.shape: ValueError('Shape of Mass {} does not match shape of MS {}'.format(Mass.shape, MS.shape))

    Fins_guess = isopy.checks.check_type('Fins_guess', Fins_guess, float, np.float_,
                                         coerce=True, coerce_into=np.float_)
    Fnat_guess = isopy.checks.check_type('Fnat_guess', Fnat_guess, float, np.float_,
                                         coerce=True, coerce_into=np.float_)
    outer_loop_iterations = isopy.checks.check_type('outer_loop_iterations', outer_loop_iterations, int)
    inner_loop_iterations = isopy.checks.check_type('inner_loop_iterations', inner_loop_iterations, int)

    dlen = MS.shape[1]
    SA = np.zeros((outer_loop_iterations, 3, dlen), dtype = np.float64)
    Fins = np.zeros((outer_loop_iterations, 3, dlen), dtype = np.float64)
    Fnat = np.zeros((outer_loop_iterations, 3, dlen), dtype = np.float64)
    MT = np.zeros((3, dlen), dtype = np.float64)

    x = 0
    y = 1
    z = 2
    Ri = output_index

    Fnat_Ri = Fnat_guess
    Fins_Ri = Fins_guess

    for i1 in range(outer_loop_iterations):
        SA[i1, x,:] = ST[x] * Mass[x] ** Fnat_Ri
        SA[i1, y,:] = ST[y] * Mass[y] ** Fnat_Ri
        SA[i1, z,:] = ST[z] * Mass[z] ** Fnat_Ri

        a = (ST[y] * (SA[i1, z] - SP[z]) + SA[i1, y] * (SP[z] - ST[z]) + SP[y] * (ST[z] - SA[i1, z])) / (ST[y] * (SA[i1, x] - SP[x]) + SA[i1, y] * (SP[x] - ST[x]) + SP[y] * (ST[x] - SA[i1, x]))
        b = (ST[x] * (SA[i1, z] - SP[z]) + SA[i1, x] * (SP[z] - ST[z]) + SP[x] * (ST[z] - SA[i1, z])) / (ST[x] * (SA[i1, y] - SP[y]) + SA[i1, x] * (SP[y] - ST[y]) + SP[x] * (ST[y] - SA[i1, y]))
        c = ST[z] - a * ST[x] - b * ST[y]

        for i2 in range(inner_loop_iterations):
            MT[x] = MS[x] * Mass[x] ** -Fins_Ri
            MT[y] = MS[y] * Mass[y] ** -Fins_Ri
            MT[z] = MS[z] * Mass[z] ** -Fins_Ri

            d = (MS[z] - MT[z]) / (MS[x] - MT[x])
            e = MS[z] - d * MS[x]
            f = (MS[z] - MT[z]) / (MS[y] - MT[y])
            g = MS[z] - f * MS[y]

            MT[x] = (b * g - b * e + e * f - c * f) / (a * f + b * d - d * f)
            MT[y] = (a * e - a * g + d * g - c * d) / (a * f + b * d - d * f)
            MT[z] = a * MT[x] + b * MT[y] + c

            #gives positive Fins
            Fins[i1, 0] = np.log(MS[x] / MT[x]) / np.log(Mass[x])
            Fins[i1, 1] = np.log(MS[y] / MT[y]) / np.log(Mass[y])
            Fins[i1, 2] = np.log(MS[z] / MT[z]) / np.log(Mass[z])
            Fins_Ri = Fins[i1, Ri]

        a = (MS[y] * (MT[z] - SP[z]) + MT[y] * (SP[z] - MS[z]) + SP[y] * (MS[z] - MT[z])) / (MS[y] * (MT[x] - SP[x]) + MT[y] * (SP[x] - MS[x]) + SP[y] * (MS[x] - MT[x]))
        b = (MS[x] * (MT[z] - SP[z]) + MT[x] * (SP[z] - MS[z]) + SP[x] * (MS[z] - MT[z])) / (MS[x] * (MT[y] - SP[y]) + MT[x] * (SP[y] - MS[y]) + SP[x] * (MS[y] - MT[y]))
        c = MS[z] - a * MS[x] - b * MS[y]

        d = (ST[z] - SA[i1, z]) / (ST[x] - SA[i1, x])
        e = ST[z] - d * ST[x]
        f = (ST[z] - SA[i1, z]) / (ST[y] - SA[i1, y])
        g = ST[z] - f * ST[y]

        SA[i1, x] = (b * g - b * e + e * f - c * f) / (a * f + b * d - d * f)
        SA[i1, y] = (a * e - a * g + d * g - c * d) / (a * f + b * d - d * f)
        SA[i1, z] = a * SA[i1, x] + b * SA[i1, y] + c

        Fnat[i1, x] = np.log(SA[i1, x] / ST[x]) / np.log(Mass[x])
        Fnat[i1, y] = np.log(SA[i1, y] / ST[y]) / np.log(Mass[y])
        Fnat[i1, z] = np.log(SA[i1, z] / ST[z]) / np.log(Mass[z])
        Fnat_Ri = Fnat[i1, Ri]

    sample_fraction = ((1 / (MT[x] + MT[y] + MT[z] + 1)) - (1 / (SP[x] + SP[y] + SP[z] + 1))) / ((1 / (SA[outer_loop_iterations - 1, x] + SA[outer_loop_iterations - 1, y] + SA[outer_loop_iterations - 1, z] + 1)) - (1 / (SP[x] + SP[y] + SP[z] + 1)))
    Fnat_f = Fnat[-1, Ri]
    Fins_f = Fins[-1, Ri]

    lambda_ = ((1-sample_fraction) ** (-1) - 1) / ((1 + np.sum(SA[-1], axis=0)) / (1 + np.sum(SP, axis=0)))
    lambda_ = 1 - (lambda_ / (lambda_ + 1))

    if mndim == 1:
        sample_fraction =  sample_fraction[0]
        Fnat_f= Fnat_f[0]
        Fins_f = Fins_f[0]
        lambda_ = lambda_[0]

    spike_fraction = (1-sample_fraction)
    Q = sample_fraction/spike_fraction

    alpha = Fnat_f * -1
    beta = Fins_f

    return DSResult('siebert', alpha, beta, lambda_, Fnat_f, Fins_f, spike_fraction, sample_fraction, Q)

def ds_inversion(measured, spike, standard = None, isotope_masses = None, inversion_keys = None, method ='rudge', **method_kwargs):
    """
    Double spike inversion.

    Parameters
    ----------
    measured : RatioArray, IsotopeArray
        Measured isotope ratios
    standard : RatioArray, IsotopeArray, dict
        References isotope ratios or a dict or references values
    spike : RatioArray, IsotopeArray, dict
        Spike isotope ratios or a dict or references values
    isotope_masses : RatioArray, IsotopeArray, dict, Optional
        Mass isotope ratios or a dict or references values. If not given hte :attr:`isotope.mass` will be used.
    inversion_keys : RatioKeyList, Optional
        List of keys in measured that will be used for the inversion. Only necessary if measured contains more than 3 keys.
    method : str
        Inversion method to be used. Options are 'rudge' and 'siebert'.
    method_kwargs
        Keyword arguments for inversion method. See `inversion_rudge` and `inversion_siebert` for list of possible arguments.

    Returns
    -------
    DSResult
    """
    measured = isopy.checks.check_type('measured', measured, isopy.core.RatioArray, isopy.IsotopeArray,
                                       coerce=True)

    spike = isopy.checks.check_type('spike', spike, core.RatioArray, core.IsotopeArray, dict,
                                    coerce=True)

    inversion_keys = isopy.checks.check_type('inversion_keys', inversion_keys, isopy.IsotopeKeyList,
                                             isopy.RatioKeyList, coerce=True, allow_none=True)
    if standard is None:
        standard = isopy.refval.isotope.abundance
    else:
        standard = isopy.checks.check_type('standard', standard, core.RatioArray, core.IsotopeArray,
                                           dict,
                                           coerce=True)

    isotope_masses = isopy.checks.check_reference_value('isotope_masses', isotope_masses, isopy.refval.isotope.mass)

    numer, denom, inversion_keys = _deduce_inversion_keys(spike, inversion_keys)

    if isinstance(measured, isopy.IsotopeArray):
        m = np.array([measured.get(n) / measured.get(denom) for n in numer])
    else:
        m = np.array([measured.get(key) for key in inversion_keys])

    if isinstance(spike, isopy.IsotopeArray):
        T = np.array([spike.get(n) / spike.get(denom) for n in numer])
    else:
        T = np.array([spike.get(key) for key in inversion_keys])

    if isinstance(standard, isopy.IsotopeArray):
        n = np.array([standard.get(n) / standard.get(denom) for n in numer])
    else:
        n = np.array([standard.get(key) for key in inversion_keys])

    if isinstance(isotope_masses, isopy.IsotopeArray):
        P = np.array([isotope_masses.get(n) / isotope_masses.get(denom) for n in numer])
    else:
        P = np.array([isotope_masses.get(key) for key in inversion_keys])

    if method == 'rudge': return ds_inversion_rudge(m, n, T, np.log(P), **method_kwargs)
    elif method == 'siebert': return ds_inversion_siebert(m, n, T, P, **method_kwargs)
    else: raise ValueError(f'method "{method}" not recognized')

def ds_correction(measured, spike, standard=None, inversion_keys = None, *, isotope_masses = None, isotope_fractions=None,
                  method ='rudge', **method_kwargs):
    """
    Do an iterative double spike inversion to correct for isobaric interferences.

    An isobaric interference correction is applied for all isotopes in *measured* that have
    a different element symbol from that in *spike*.

    Parameters
    ----------
    measured : RatioArray, IsotopeArray
        Measured isotope ratios
    standard : RatioArray, IsotopeArray, dict
        References isotope ratios or a dict or references values
    spike : RatioArray, IsotopeArray, dict
        Spike isotope ratios or a dict or references values
    isotope_masses : RatioArray, IsotopeArray, dict, Optional
        Mass isotope ratios or a dict or references values. If not given hte :attr:`isotope.mass` will be used.
    inversion_keys : RatioKeyList, Optional
        List of keys in measured that will be used for the inversion. Only necessary if measured contains more than 3 keys.
    method : str
        Inversion method to be used. Options are 'rudge' and 'siebert'.
    method_kwargs
        Keyword arguments for inversion method. See `inversion_rudge` and `inversion_siebert` for list of possible arguments.

    Returns
    -------
    DSResult
    """
    measured = isopy.checks.check_type('measured', measured, isopy.core.RatioArray,
                                       isopy.IsotopeArray,
                                       coerce=True)

    spike = isopy.checks.check_type('spike', spike, isopy.core.RatioArray, isopy.core.IsotopeArray,
                                    dict,
                                    coerce=True, coerce_into=isopy.core.RatioArray)

    inversion_keys = isopy.checks.check_type('inversion_keys', inversion_keys, isopy.IsotopeKeyList,
                                             isopy.RatioKeyList, coerce=True, allow_none=True)

    isotope_fractions = isopy.checks.check_reference_value('isotope_fractions', isotope_fractions,
                                                           isopy.refval.isotope.abundance)
    isotope_masses = isopy.checks.check_reference_value('isotope_masses', isotope_masses,
                                                        isopy.refval.isotope.mass)
    if standard is None:
        standard = isopy.refval.isotope.abundance
    else:
        standard = isopy.checks.check_type('standard', standard, core.RatioArray, core.IsotopeArray,
                                           dict, coerce=True, coerce_into=isopy.core.RatioArray)


    numer, denom, inversion_keys = _deduce_inversion_keys(spike, inversion_keys)

    interference_isotopes = isotope._find_isobaric_interferences(measured, denom.element_symbol)

    inversion = ds_inversion(measured, spike, standard, isotope_masses, inversion_keys, method, **method_kwargs)
    fins = inversion.fins

    for i in range(10):
        measured2 = measured.copy()
        fins2 = fins

        for infiso in  interference_isotopes:
            measured2 = isotope.remove_isobaric_interferences(measured2, infiso, fins, isotope_fractions, isotope_masses)

        inversion = ds_inversion(measured2, spike, standard, isotope_masses, inversion_keys, method,
                                 **method_kwargs)
        fins = inversion.fins

        if np.all(np.abs(fins - fins2) < 0.000001):
            break  # Beta value has converged so no need for more iterations.
    else:
        raise ValueError(
            'inversion did not converge after 10 iterations of the interference correction')
    return inversion

#TODO cache for speedier grid
def _deduce_inversion_keys(spike, inversion_keys):
    if inversion_keys is None:
        if not isinstance(spike, core.IsopyArray):
            raise ValueError(f'Can not deduce inversion keys from spike since it is not an isopy array')
        elif isinstance(spike, isopy.IsotopeArray):
            if len(spike.keys) != 4:
                raise ValueError(f'inversion keys can not be deduced from *spike* as it'
                                 f'has {len(spike.keys)} isotope keys instead of the expected 4')
            else:
                denom = isopy.argmaxkey(spike)
                numer = spike.keys - denom
                inversion_keys = numer / denom
        elif isinstance(spike, isopy.RatioArray):
            if len(spike.keys) != 3:
                raise ValueError(f'inversion keys can not be deduced from *spike* as it'
                                 f'has {len(spike.keys)} ratio keys instead of the expected 3')
            else:
                denom = spike.keys.common_denominator
                numer = spike.keys.numerators
                inversion_keys = numer / denom
    else:
        if isinstance(inversion_keys, isopy.IsotopeKeyList):
            if len(inversion_keys) != 4:
                raise ValueError(f'got {len(inversion_keys)} inversion isotope keys instead of 4')
            elif isinstance(spike, isopy.IsotopeArray):
                spike = spike.copy(key_eq=inversion_keys)
                denom = isopy.argmaxkey(spike)
                numer = inversion_keys - denom
            elif isinstance(spike, isopy.RatioArray):
                denom = spike.keys.common_denominator
                numer = inversion_keys - denom
            inversion_keys = numer / denom
        if isinstance(inversion_keys, isopy.RatioKeyList):
            if len(inversion_keys) != 3:
                raise ValueError(f'got {len(inversion_keys)} inversion ratio keys instead of 3')
            elif not inversion_keys.has_common_denominator:
                raise ValueError(f'inversion key ratios do not have a common denominator')
            else:
                denom = inversion_keys.common_denominator
                numer = inversion_keys.numerators

    return numer, denom, inversion_keys


def ds_grid(standard, spike1, spike2=None, inversion_keys=None, n=99,  *,
            fnat = None, fins = 2,  maxv = 10,
            integrations = 100, integration_time = 8.389, resistor = 1E11,
            random_seed = None, method='rudge',
            isotope_masses=None, isotope_fractions = None,
            correction_method=ds_correction,
            **interferences):
    isotope_fractions = isopy.checks.check_reference_value('isotope_fractions', isotope_fractions,
                                                           isopy.refval.isotope.abundance)
    isotope_masses = isopy.checks.check_reference_value('isotope_masses', isotope_masses,
                                                        isopy.refval.isotope.mass)

    numer, denom, inversion_keys = _deduce_inversion_keys(spike1, inversion_keys)

    standard = isotope.make_ms_array(standard, mf_factor=fnat, isotope_fractions=isotope_fractions,
                                     isotope_masses=isotope_masses)

    spike_fractions = np.linspace(0, 1, n + 2)[1:-1]
    spike1 = spike1 / np.sum(spike1, axis=None)

    if spike2 is None:
        spike2 = 0
        spike1_fractions = np.array([1])
    else:
        spike1_fractions = np.linspace(0, 1, n+2)[1:-1]
        spike2 = spike2 / np.sum(spike2, axis=None)

    result_solutions = []
    for spike1_fraction in spike1_fractions:
        spike_mixture = (spike1 * spike1_fraction) + (spike2 * (1-spike1_fraction))
        result_solutions.append([])
        for spike_fraction in spike_fractions:
            measured = isotope.make_ms_sample(standard, spike_mixture=spike_mixture,
                                              spike_fraction=spike_fraction,
                                              fnat=None, fins=fins, maxv=maxv,
                                              integrations=integrations,
                                              integration_time=integration_time,
                                              resistor=resistor,
                                              random_seed=random_seed,
                                              isotope_fractions=isotope_fractions,
                                              isotope_masses=isotope_masses, **interferences)
            try:
                solution = correction_method(measured, spike_mixture, standard, inversion_keys,
                                         isotope_masses=isotope_masses,
                                         isotope_fractions=isotope_fractions,
                                         method=method)
                result_solutions[-1].append(solution)
            except:
                nan = np.full(measured.size, np.nan)
                solution = DSResult(nan.copy(), nan.copy(), nan.copy(),
                                    named_args = (nan.copy(), nan.copy(), nan.copy(), nan.copy(), nan.copy(), nan.copy(), nan.copy()),
                                    method='failed')
                result_solutions[-1].append(solution)

    return DSGridResult(spike_fractions, spike1_fractions, result_solutions)

class DSResult:
    def __init__(self, method, alpha, beta, lambda_, fnat, fins, spike_fraction, sample_fraction, Q):
        self.alpha = alpha
        self.beta = beta
        self.lambda_ = lambda_

        self.fnat = fnat
        self.fins = fins
        self.spike_fraction = spike_fraction
        self.sample_fraction = sample_fraction
        self.Q = Q

        self.p = self.spike_fraction
        self.method = method

        self.__named_args = (alpha, beta, lambda_, fnat, fins, spike_fraction, sample_fraction, Q)

    def Delta(self, mass_ratio, isotope_masses = None):
        if isotope_masses is None: isotope_masses.refv.isotope.mass
        if isinstance(mass_ratio, str):
            mass_ratio = self.__isotope_masses.get(mass_ratio)

        return np.power(mass_ratio, self.fnat) - 1

    def Delta_prime(self, mass_ratio, isotope_masses = None):
        if isotope_masses is None: isotope_masses.refv.isotope.mass
        if isinstance(mass_ratio, str):
            mass_ratio = self.__isotope_masses.get(mass_ratio)

        return np.log(mass_ratio) * self.fnat

    def delta(self, mass_ratio, isotope_masses = None):
        return self.Delta(mass_ratio, isotope_masses) * 1000

    def delta_prime(self, mass_ratio, isotope_masses = None):
        return self.Delta_prime(mass_ratio, isotope_masses) * 1000

    def ppm(self, mass_ratio, isotope_masses = None):
        return self.Delta(mass_ratio, isotope_masses) * 1E6

    def ppm_prime(self, mass_ratio, isotope_masses = None):
        return self.Delta_prime(mass_ratio, isotope_masses) * 1E6

    def __array_function__(self, func, types, args, kwargs):
        return self.__class__(self.method, *[func(value, **kwargs) for value in self.__named_args])

class DSGridResult:
    def __init__(self, x, y, solutions):
        self.x = x
        self.y = y
        self.__solutions = solutions

    def __getattr__(self, attr):
        return np.array([[getattr(solution, attr) for solution in row] for row in self.__solutions])

    def __array_function__(self, func, types, args, kwargs):
        return self.__class__(self.x, self.y, [[func(solution, **kwargs) for solution in row] for row in self.__solutions])


#TODO double_spike_correction method
#TODO allow ds_uncernianty grid to take correction funcion

def _ds_uncertainty_grid(standard, spike1, spike2, mass, *, isotopes = None, max_voltage = 10, resistors = None,
                        integration_time = 4, cycles = 100, resolution = 19, Fnat = 0.1, Fins = 1.6,
                        ds_method ='Siebert', fast_mode = True, output_attr = 'alpha'):

    if isinstance(standard, _dtypes.IsotopeArray):
        if isotopes is None: isotopes = standard.keys()
        std = np.array([standard[i] for i in isotopes]).flatten()
    elif isinstance(standard, _dtypes.IsopyDict):
        std = np.array([standard[i] for i in isotopes]).flatten()
    else:
        std = np.asarray(standard).flatten()

    if std.size != 4: raise ValueError('standard must contain exactly four values')

    if isinstance(spike1, str):
        if isotopes is None: raise ValueError('spike1: "isotopes" not set')
        sp1 = np.zeros(4, dtype='f8')
        sp1[isotopes.index(spike1)] = 1
    elif isinstance(spike1, _dtypes.IsotopeArray):
        if isotopes is None: raise ValueError('spike1: "isotopes" not set')
        std = np.array([spike1[i] for i in isotopes]).flatten()
    else:
        sp1 = np.asarray(spike1).flatten()
    if sp1.size != 4: raise ValueError('spike1 must contain exactly four values')

    if isinstance(spike2, str):
        if isotopes is None: raise ValueError('spike2: "isotopes" not set')
        sp2 = np.zeros(4, dtype='f8')
        sp2[isotopes.index(spike2)] = 1
    elif isinstance(spike2, _dtypes.IsotopeArray):
        if isotopes is None: raise ValueError('spike2: "isotopes" not set')
        std = np.array([spike2[i] for i in isotopes]).flatten()
    else:
        sp2 = np.asarray(spike2).flatten()
    if sp2.size != 4: raise ValueError('spike2 must contain exactly four values')

    if isinstance(mass, (_dtypes.IsotopeArray, _dtypes.IsopyDict)):
        if isotopes is None: raise ValueError('mass: "isotopes" not set')
        mas = np.array([mass[i] for i in isotopes]).flatten()
    else:
        mas = np.asarray(mass).flatten()
    if mas.size != 4: raise ValueError('mass must contain exactly four values')

    #At this point standard, spike1 and spike2 are arrays of 4.
    standard = std / np.sum(std)
    spike1 = sp1 / np.sum(sp1)
    spike2 = sp2 / np.sum(sp2)
    mass = mas

    prop = np.linspace(0.0, 1, resolution + 2)[1:-1]

    denom = np.argmax(spike2)
    rat_index = [True] * 4
    rat_index[denom] = False

    mass_ratio = mass[rat_index] / mass[denom]

    if ds_method == 'Rudge':
        method = Rudge_inversion
        mass_ratio2 = np.log(mass_ratio)

    elif ds_method == 'Siebert':
        method = Siebert_inversion
        mass_ratio2 = mass_ratio

    ref_ratio = (standard[rat_index] / standard[denom]) * mass_ratio ** Fnat

    if fast_mode is False:
        z = np.zeros((resolution, resolution, cycles), dtype=np.float64) * np.nan
        for y in range(resolution):
            spike = prop[y] * spike1 + (1-prop[y]) * spike2
            sp_rat = spike[rat_index] / spike[denom]
            for x in range(resolution):
                sample = prop[x] * spike + (1 - prop[x]) * standard
                sample = sample * max_voltage / np.max(sample)
                noise = johnson_nyquist_noise(sample, resistors, integration_time)
                noisy_sample = np.random.normal(sample, noise, (cycles, 4)).transpose()
                smp_rat = noisy_sample[rat_index] / noisy_sample[denom]
                smp_rat = np.transpose(smp_rat.transpose() * mass_ratio ** Fins)

                solution = method(smp_rat, ref_ratio, sp_rat, mass_ratio2)
                z[y,x, :] = solution.__getattribute__(output_attr)

    if fast_mode is True:
        big = np.ones((resolution, resolution, cycles, 4))
        small = np.ones((resolution, resolution, 4))

        spike = big * spike1 * prop.reshape(resolution, 1, 1, 1) + big * spike2 * (1 - prop).reshape(resolution, 1, 1,                                                                                          1)
        smp = spike[:, :, 1, :] * prop.reshape(1, resolution, 1) + small * standard * (1 - prop).reshape(1, resolution,
                                                                                                         1)
        smp = smp.reshape(-1, 4)
        smp = smp * (max_voltage / np.max(smp, axis=1).reshape(-1, 1))
        noise = johnson_nyquist_noise(smp, resistors, integration_time)

        noisy_sample = np.random.normal(smp, noise, (cycles, resolution ** 2, 4))
        noisy_sample = noisy_sample.transpose().reshape(4, -1)
        smp_ratio = noisy_sample[rat_index] / noisy_sample[denom] * (mass_ratio ** Fins).reshape(3, 1)

        spike = spike.flatten().reshape(-1, 4).transpose()
        sp_ratio = spike[rat_index] / spike[denom]

        #TODO check error that comes from rudge and do a specieal error
        solution = method(smp_ratio, ref_ratio, sp_ratio, mass_ratio2)
        z = solution.__getattribute__(output_attr).transpose().flatten().reshape(resolution, resolution, -1)

    std = np.nanstd(z, axis=2) * 2
    count = np.count_nonzero(~np.isnan(z), axis=2)

    std[count < cycles * 0.9] = np.nan

    if output_attr == 'alpha':
        mean = np.nanmean(z, axis=2)
        std[np.abs(mean - Fnat) > std] = np.nan

    argmin_z = np.nanargmin(std)
    min_x_pos = argmin_z % std.shape[0]
    min_y_pos = int(argmin_z / std.shape[0])

    #x = sp in sp+smp, y = sp1 in sp1+sp2
    return prop, prop, std / np.sqrt(count), (min_y_pos, min_x_pos)

def _plot_ds_uncertainty_grid(*args, **kwargs):
    x, y, z, best = ds_uncertainty_grid(*args, **kwargs)

    levels = [0] + [x * 0.0001 for x in range(1, 10)] + [x * 0.001 for x in range(1, 10)] + [x * 0.01 for x in
                                                                                             range(1, 10)] + [0.1]
    min_z = z[best]
    min_x = x[best[1]]
    min_y = y[best[0]]

    title = 'Smallest uncertianty: {:.5f} ($2\\sigma/\\sqrt{}n{}$) ' \
            'at x: {:.2f}, y: {:.2f}'.format(min_z, '{', '}', min_x, min_y)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.subplots_adjust(hspace=0.3)
    CS = ax.contourf(x, y, z, levels=levels, cmap=mplcm.get_cmap(mplcm.jet), norm=mplcolors.PowerNorm(0.23))
    fig.colorbar(CS, ax=ax, ticks=[0, 0.001, 0.01, 0.1, 1])
    plt.xlabel('Spike fraction in Spike/Sample mix')
    plt.ylabel('Spike1 fraction in Spike1/Spike2 mix')
    plt.title(title)
    plt.show()


def _ds_cocktail_list(standard, mass, spikes = None, isotopes = None, output_mass_ratio = None, output_multiple = 1000, **kwargs):
    if isinstance(standard, _dtypes.IsotopeArray):
        if isotopes is None: isotopes = standard.keys()
    elif isinstance(standard, _dtypes.IsopyDict):
        if isotopes is None: raise ValueError('"isotopes" not given')
    else:
        raise ValueError('"standard" must be either a IsotopeArrray or a IsopyDict')

    if isotopes is None: raise ValueError('"isotopes" not given')
    if len(isotopes) < 4: raise ValueError('At least 4 isotopes are needed')

    if spikes is None: spikes = {}
    if not isinstance(spikes, dict): raise ValueError('"spikes" must be a dict')
    for s in spikes:
        if not isinstance(s, _dtypes.IsotopeArray, _dtypes.IsopyDict):
            raise ValueError('Each entry in "spikes" must be either a IsotopeArrray or a IsopyDict')

    if not isinstance(mass, (_dtypes.IsotopeArray, _dtypes.IsopyDict)):
        raise ValueError('"mass" must be either a IsotopeArray or a IsopyDict')

    output = []
    for iso in itertools.combinations(isotopes, 4):
        for spike in itertools.combinations(iso, 2):
            x, y, z, best = uncertianty_grid(standard, spikes.get(spike[0], spike[0]), spikes.get(spike[1], spike[1]), mass, isotopes = iso, **kwargs)

            x_min = x[best[1]]
            y_min = y[best[0]]

            z_min = z[best]

            if output_mass_ratio is not None:
                z_min = (output_mass_ratio ** z[best] -1) * output_multiple

            output.append([tuple(iso), tuple(spike), (x_min, y_min, z_min), x, y, z, best])
    return output







