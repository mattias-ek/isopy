import numpy as np
import scipy.optimize as optimize
import isopy as isopy
import itertools as itertools
from collections import namedtuple as _namedtuple
from matplotlib import cm as mplcm
from matplotlib import colors as mplcolors
from isopy import core
from . import isotope
import warnings
from datetime import datetime as dt

__all__ = ['ds_correction', 'ds_Delta', 'ds_Delta_prime', 'ds_grid']

import isopy.checks

"""
Functions for doublespike data reduction
"""

def _inversion_rudge_function(x, P, n, T, m):
    x = x.reshape(3, -1)
    lambda_ = x[0]
    alpha = x[1] / (1 - lambda_)
    beta = x[2]

    return ((lambda_ * T) + ((1 - lambda_) * n * np.exp(-alpha * P)) - m * np.exp(
        -beta * P)).flatten()


# Currently no used as it doesnt work when the input has more than one dimension.
# Might just be a matter of rearranging the output
def _inversion_rudge_jacobian(x, P, n, T, m):
    x = x.reshape(3, -1)
    lambda_ = x[0]
    alpha = x[1] / (1 - lambda_)
    beta = x[2]
    output = np.transpose(
        [T - (n * np.exp(-alpha * P) * (1 + alpha * P)), -n * P * np.exp(-alpha * P),
         m * P * np.exp(-beta * P)])
    return output


# split if more than x so that it doesnt break
def ds_inversion_rudge(m, n, T, P, xtol=1e-10, **kwargs):
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
    if m.size == 3:
        m = m.reshape(3, 1)
    elif m.ndim == 1:
        raise ValueError()
    elif m.ndim == 2:
        if m.shape[0] != 3: raise ValueError(f'shape of m is incorrect ({m.shape})')
    else:
        raise ValueError('m has to many dimensions ({})'.format(m.ndim))

    if n.size == 3:
        n = np.ones(m.shape) * n.reshape(3, 1)
    elif n.shape != m.shape:
        raise ValueError('Shape of n {} does not match shape of m {}'.format(n.shape, m.shape))

    if T.size == 3:
        T = np.ones(m.shape) * T.reshape(3, 1)
    elif T.shape != m.shape:
        ValueError('Shape of T {} does not match shape of m {}'.format(T.shape, m.shape))

    if P.size == 3:
        P = np.ones(m.shape) * P.reshape(3, 1)
    elif P.shape != m.shape:
        ValueError('Shape of P {} does not match shape of m {}'.format(P.shape, m.shape))

    xtol = isopy.checks.check_type('xtol', xtol, np.float_, float, coerce=True)

    A = np.transpose([T - n, -n * P, m * P])
    b = np.transpose(m - n)
    x0 = np.transpose(np.linalg.solve(A, b))

    x, infodict, ier, mesg = optimize.fsolve(_inversion_rudge_function, x0, (P, n, T, m), None,
                                             True, xtol=xtol, **kwargs)
    if ier != 1:
        x = np.ones(m.shape) * np.nan

    if mndim == 2: x = x.reshape(3, -1)
    lambda_ = x[0]
    alpha = x[1] / (1 - lambda_)
    beta = x[2]

    Fnat = alpha * -1
    Fins = beta

    spike_fraction = (1 + ((1 - lambda_) / lambda_) * (
                (1 + np.sum((n / np.exp(alpha * P)), axis=0)) / (1 + np.sum(T, axis=0)))) ** (-1)
    sample_fraction = (1 - spike_fraction)
    Q = sample_fraction / spike_fraction

    static_items = dict(method='rudge')
    dynamic_items = dict(alpha=alpha,
                         beta=beta,
                         lambda_=lambda_,
                         fnat=Fnat,
                         fins=Fins,
                         spike_fraction=spike_fraction,
                         sample_fraction=sample_fraction,
                         Q=Q)

    return DSResult(static_items, dynamic_items)


def ds_inversion_siebert(MS, ST, SP, Mass, Fins_guess=2, Fnat_guess=-0.00001,
                         outer_loop_iterations=3, inner_loop_iterations=6):
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
    elif MS.ndim == 1:
        raise ValueError()
    elif MS.ndim == 2:
        if MS.shape[0] != 3: raise ValueError()
    else:
        raise ValueError('MS has to many dimensions ({})'.format(MS.ndim))

    if ST.size == 3:
        ST = ST.reshape(3)
    elif ST.shape != MS.shape:
        raise ValueError('Shape of ST {} does not match shape of MS {}'.format(ST.shape, MS.shape))

    if ST.size == 3:
        ST = ST.reshape(3)
    elif ST.shape != MS.shape:
        ValueError('Shape of ST {} does not match shape of MS {}'.format(ST.shape, MS.shape))

    if Mass.size == 3:
        Mass = Mass.reshape(3)
    elif Mass.shape != MS.shape:
        ValueError('Shape of Mass {} does not match shape of MS {}'.format(Mass.shape, MS.shape))

    Fins_guess = isopy.checks.check_type('Fins_guess', Fins_guess, float, np.float_,
                                         coerce=True, coerce_into=np.float_)
    Fnat_guess = isopy.checks.check_type('Fnat_guess', Fnat_guess, float, np.float_,
                                         coerce=True, coerce_into=np.float_)
    outer_loop_iterations = isopy.checks.check_type('outer_loop_iterations', outer_loop_iterations,
                                                    int)
    inner_loop_iterations = isopy.checks.check_type('inner_loop_iterations', inner_loop_iterations,
                                                    int)

    dlen = MS.shape[1]
    SA = np.zeros((outer_loop_iterations, 3, dlen), dtype=np.float64)
    Fins = np.zeros((outer_loop_iterations, 3, dlen), dtype=np.float64)
    Fnat = np.zeros((outer_loop_iterations, 3, dlen), dtype=np.float64)
    MT = np.zeros((3, dlen), dtype=np.float64)

    x = 0
    y = 1
    z = 2
    Ri = output_index

    Fnat_Ri = Fnat_guess
    Fins_Ri = Fins_guess

    with warnings.catch_warnings():
        #This supresses warning for unsolvable inversions
        #Mostly applicable for creating doublespike maps
        warnings.simplefilter("ignore")

        for i1 in range(outer_loop_iterations):
            SA[i1, x, :] = ST[x] * Mass[x] ** Fnat_Ri
            SA[i1, y, :] = ST[y] * Mass[y] ** Fnat_Ri
            SA[i1, z, :] = ST[z] * Mass[z] ** Fnat_Ri

            a = (ST[y] * (SA[i1, z] - SP[z]) + SA[i1, y] * (SP[z] - ST[z]) + SP[y] * (
                        ST[z] - SA[i1, z])) / (
                            ST[y] * (SA[i1, x] - SP[x]) + SA[i1, y] * (SP[x] - ST[x]) + SP[y] * (
                                ST[x] - SA[i1, x]))
            b = (ST[x] * (SA[i1, z] - SP[z]) + SA[i1, x] * (SP[z] - ST[z]) + SP[x] * (
                        ST[z] - SA[i1, z])) / (
                            ST[x] * (SA[i1, y] - SP[y]) + SA[i1, x] * (SP[y] - ST[y]) + SP[x] * (
                                ST[y] - SA[i1, y]))
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

                # gives positive Fins
                Fins[i1, 0] = np.log(MS[x] / MT[x]) / np.log(Mass[x])
                Fins[i1, 1] = np.log(MS[y] / MT[y]) / np.log(Mass[y])
                Fins[i1, 2] = np.log(MS[z] / MT[z]) / np.log(Mass[z])
                Fins_Ri = Fins[i1, Ri]

            a = (MS[y] * (MT[z] - SP[z]) + MT[y] * (SP[z] - MS[z]) + SP[y] * (MS[z] - MT[z])) / (
                        MS[y] * (MT[x] - SP[x]) + MT[y] * (SP[x] - MS[x]) + SP[y] * (MS[x] - MT[x]))
            b = (MS[x] * (MT[z] - SP[z]) + MT[x] * (SP[z] - MS[z]) + SP[x] * (MS[z] - MT[z])) / (
                        MS[x] * (MT[y] - SP[y]) + MT[x] * (SP[y] - MS[y]) + SP[x] * (MS[y] - MT[y]))
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

        sample_fraction = ((1 / (MT[x] + MT[y] + MT[z] + 1)) - (1 / (SP[x] + SP[y] + SP[z] + 1))) / ((
                             1 / (SA[outer_loop_iterations - 1, x] + SA[outer_loop_iterations - 1, y] +
                                 SA[outer_loop_iterations - 1, z] + 1)) - (1 / (SP[x] + SP[y] + SP[z] + 1)))
        Fnat_f = Fnat[-1, Ri]
        Fins_f = Fins[-1, Ri]

        lambda_ = ((1 - sample_fraction) ** (-1) - 1) / (
                    (1 + np.sum(SA[-1], axis=0)) / (1 + np.sum(SP, axis=0)))
        lambda_ = 1 - (lambda_ / (lambda_ + 1))

        if mndim == 1:
            sample_fraction = sample_fraction[0]
            Fnat_f = Fnat_f[0]
            Fins_f = Fins_f[0]
            lambda_ = lambda_[0]

        spike_fraction = (1 - sample_fraction)
        Q = sample_fraction / spike_fraction

        alpha = Fnat_f * -1
        beta = Fins_f

    static_items = dict(method='siebert')
    dynamic_items = dict(alpha=alpha,
                         beta=beta,
                         lambda_=lambda_,
                         fnat=Fnat_f,
                         fins=Fins_f,
                         spike_fraction=spike_fraction,
                         sample_fraction=sample_fraction,
                         Q=Q)

    return DSResult(static_items, dynamic_items)


def ds_inversion(measured, spike, standard=None, isotope_masses=None, inversion_keys=None,
                 method='rudge', **method_kwargs):
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
        Keys used for the inversion. Can either be 3 ratio key strings or 4 isotope key strings.
        Does not have to be given if the inversion keys can be inferred from *spike*.
    method : str
        Inversion method to be used. Options are 'rudge' and 'siebert'.
    method_kwargs
        Keyword arguments for inversion method. See `inversion_rudge` and `inversion_siebert` for list of possible arguments.

    Returns
    -------
    inversion_result : DSResult
        The returned *DSResult* object contains the the following attributes:

        * ``method`` - Name of the method used to do the inversion.

        * ``alpha`` - The natural fractionation factor as defined by Rudge (:math:`n = N * m^\\alpha`).
          **Note** this value has the opposite sign to *fnat*.

        * ``beta`` - The instrumental mass fractionation factor as defined by Rudge (:math:`m = M * m^\\beta`).
          Same as *fins*.

        * ``lambda_`` - The lambda value defined by Rudge.

        * ``fnat`` - The natural fractionation factor as defined by Siebert (:math:`\\textrm{SA} = \\textrm{ST} * m^\\alpha`).
          **Note** this value has the opposite sign to *alpha*.

        * ``fins`` - The instrumental fractionation factor as defined by Siebers (:math:`\\textrm{MT} = \\textrm{MS} * m^\\alpha`).
          Same as *beta*.

        * ``spike_fraction`` - The fraction of spike in the sample-spike mixture. Calculated on the
          basis of the four isotopes used in the inversion.

        * ``sample_fraction`` - The fraction of sample in the sample-spike mixture. Calculated on the
          basis of the four isotopes used in the inversion.

        * ``Q`` - The *sample_fraction* to *spike_fraction* ratio.

        Array functions, e.g. ``np.mean`` and ``isopy.sd`` can be used on this object. The
        function will be performed on each attribute and a new *DSResult* object returned.
    """
    measured = isopy.checks.check_type('measured', measured, isopy.core.RatioArray,
                                       isopy.IsotopeArray,
                                       coerce=True)

    spike = isopy.checks.check_type('spike', spike, core.RatioArray, core.IsotopeArray, dict,
                                    coerce=True)

    inversion_keys = isopy.checks.check_type('inversion_keys', inversion_keys, isopy.IsotopeKeyList,
                                             isopy.RatioKeyList, coerce=True, allow_none=True)
    if standard is None:
        standard = isopy.refval.isotope.fraction
    else:
        standard = isopy.checks.check_type('standard', standard, core.RatioArray, core.IsotopeArray,
                                           dict,
                                           coerce=True)

    isotope_masses = isopy.checks.check_reference_value('isotope_masses', isotope_masses,
                                                        isopy.refval.isotope.mass)

    numerators, denominator, inversion_keys = _deduce_inversion_keys(spike, inversion_keys)

    if isinstance(measured, isopy.IsotopeArray):
        m = np.array([measured.get(numer) / measured.get(denominator) for numer in numerators])
    else:
        m = np.array([measured.get(key) for key in inversion_keys])


    if isinstance(spike, isopy.IsotopeArray):
        T = np.array([spike.get(numer) / spike.get(denominator) for numer in numerators])
    else:
        T = np.array([spike.get(key) for key in inversion_keys])

    if isinstance(standard, isopy.IsotopeArray):
        n = np.array([standard.get(numer) / standard.get(denominator) for numer in numerators])
    else:
        n = np.array([standard.get(key) for key in inversion_keys])

    if isinstance(isotope_masses, isopy.IsotopeArray):
        P = np.array([isotope_masses.get(numer) / isotope_masses.get(denominator) for numer in numerators])
    else:
        P = np.array([isotope_masses.get(key) for key in inversion_keys])

    if method == 'rudge':
        return ds_inversion_rudge(m, n, T, np.log(P), **method_kwargs)
    elif method == 'siebert':
        return ds_inversion_siebert(m, n, T, P, **method_kwargs)
    else:
        raise ValueError(f'method "{method}" not recognized')


def ds_correction(measured, spike, standard=None, inversion_keys=None,
                  interference_correction = True,
                  isotope_fractions=None, isotope_masses=None,
                  method='rudge', fins_tol = 0.000001, **method_kwargs):
    """
    A double spike data reduction.

    If *interference_correction* is True a correction is applied for all isotopes in *measured* that
    have a different element symbol from the keys in *spike*. If more than one isotope exists for an
    an element the largest isotope is used for the interference correction.

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
        Keys used for the inversion. Can either be 3 ratio key strings or 4 isotope key strings.
        Does not have to be given if the inversion keys can be inferred from *spike*.
    method : str
        Inversion method to be used. Options are 'rudge' and 'siebert'.
    fins_tol : float
        The interference correction is considered a success once the difference between the current and the
        previous *fins* value is below this value.
    method_kwargs
        Keyword arguments for inversion method. For advanced users only.
        See `inversion_rudge` and `inversion_siebert` for list of possible arguments.

    Returns
    -------
    inversion_result : DSResult
        The returned *DSResult* object contains the the following attributes:

        * ``method`` - Name of the method used to do the inversion.

        * ``alpha`` - The natural fractionation factor as defined by Rudge (:math:`n = N * m^\\alpha`).
          **Note** this value has the opposite sign to *fnat*.

        * ``beta`` - The instrumental mass fractionation factor as defined by Rudge (:math:`m = M * m^\\beta`).
          Same as *fins*.

        * ``lambda_`` - The lambda value defined by rudge.

        * ``fnat`` - The natural fractionation factor as defined by Siebert (:math:`\\textrm{SA} = \\textrm{ST} * m^\\alpha`).
          **Note** this value has the opposite sign to *alpha*.

        * ``fins`` - The instrumental fractionation factor as defined by Siebers (:math:`\\textrm{MT} = \\textrm{MS} * m^\\alpha`).
          Same as *beta*.

        * ``spike_fraction`` - The fraction of spike in the sample-spike mixture. Calculated on the
          basis of the four isotopes used in the inversion.

        * ``sample_fraction`` - The fraction of sample in the sample-spike mixture. Calculated on the
          basis of the four isotopes used in the inversion.

        * ``Q`` - The *sample_fraction* to *spike_fraction* ratio.

        Array functions, e.g. ``np.mean`` and ``isopy.sd`` can be used on this object. The
        function will be performed on each attribute and a new *DSResult* object returned.
    """
    measured = isopy.checks.check_type('measured', measured, isopy.core.RatioArray,
                                       isopy.IsotopeArray,
                                       coerce=True)

    spike = isopy.checks.check_type('spike', spike, isopy.core.RatioArray, isopy.core.IsotopeArray,
                                    dict,
                                    coerce=True, coerce_into=isopy.core.RatioArray)

    inversion_keys = isopy.checks.check_type('inversion_keys', inversion_keys, isopy.IsotopeKeyList,
                                             isopy.RatioKeyList, coerce=True, allow_none=True)

    isotope_fractions = isopy.checks.check_reference_value('isotope_abundances', isotope_fractions,
                                                           isopy.refval.isotope.fraction)
    isotope_masses = isopy.checks.check_reference_value('isotope_masses', isotope_masses,
                                                        isopy.refval.isotope.mass)
    if standard is None:
        standard = isotope_fractions
    else:
        standard = isopy.checks.check_type('standard', standard, core.RatioArray, core.IsotopeArray,
                                           dict, coerce=True, coerce_into=isopy.core.RatioArray)

    # If inversion keys are not given they are inferred from the spike
    numer, denom, inversion_keys = _deduce_inversion_keys(spike, inversion_keys)

    # Find isotopes of potential isobaric interferences
    # If more than one isotope of an element exits the correction is made with the largest one
    # Thus only the largest isotope is returned from this function
    if interference_correction is True:
        isobaric_interferences = isotope.find_isobaric_interferences(denom.element_symbol, measured)
    elif isinstance(interference_correction, dict):
        isobaric_interferences = interference_correction
    else:
        isobaric_interferences = []

    # Calculate the starting value
    inversion = ds_inversion(measured, spike, standard, isotope_masses, inversion_keys, method,
                             **method_kwargs)
    fins = inversion.fins

    # Iteratively perform the interference correction
    # Stop once the result converges
    if isobaric_interferences:
        for i in range(10):
            measured2 = measured.copy()
            fins2 = fins

            measured2 = isotope.remove_isobaric_interferences(measured2, isobaric_interferences,
                                                              fins, isotope_fractions, isotope_masses)
            #Recalculate the instrumental fractionation
            inversion = ds_inversion(measured2, spike, standard, isotope_masses, inversion_keys, method,
                                     **method_kwargs)
            fins = inversion.fins

            if np.all(np.abs(fins - fins2) < fins_tol):
                break  # Beta value has converged so no need for more iterations.
        else:
            raise ValueError(
                'inversion did not converge after 10 iterations of the interference correction')


    return inversion


# TODO cache for speedier grid
def _deduce_inversion_keys(spike, inversion_keys):
    if inversion_keys is None:
        if not isinstance(spike, core.IsopyArray):
            raise ValueError(
                f'Can not deduce inversion keys from spike since it is not an isopy array')
        elif isinstance(spike, isopy.IsotopeArray):
            if len(spike.keys) != 4:
                raise ValueError(f'inversion keys can not be deduced from *spike* as it'
                                 f'has {len(spike.keys)} isotope keys instead of the expected 4')
            else:
                denom = isopy.keymax(spike)
                numer = spike.keys - denom
                inversion_keys = numer / denom
                return numer, denom, inversion_keys

        elif isinstance(spike, isopy.RatioArray):
            if len(spike.keys) != 3:
                raise ValueError(f'inversion keys can not be deduced from *spike* as it'
                                 f'has {len(spike.keys)} ratio keys instead of the expected 3')
            else:
                denom = spike.keys.common_denominator
                numer = spike.keys.numerators
                inversion_keys = numer / denom
                return numer, denom, inversion_keys
    else:
        inversion_keys = isopy.askeylist(inversion_keys)
        if isinstance(inversion_keys, isopy.IsotopeKeyList):
            if len(inversion_keys) != 4:
                raise ValueError(f'got {len(inversion_keys)} inversion isotope keys instead of 4')
            elif isinstance(spike, isopy.IsotopeArray):
                spike = spike.copy(key_eq=inversion_keys)
                denom = isopy.keymax(spike)
                numer = inversion_keys - denom
                inversion_keys = numer / denom
                return numer, denom, inversion_keys

            elif isinstance(spike, isopy.RatioArray):
                denom = spike.keys.common_denominator
                numer = inversion_keys - denom
                inversion_keys = numer / denom
                return numer, denom, inversion_keys

        elif isinstance(inversion_keys, isopy.RatioKeyList):
            if len(inversion_keys) != 3:
                raise ValueError(f'got {len(inversion_keys)} inversion ratio keys instead of 3')
            elif inversion_keys.common_denominator is None:
                raise ValueError(f'inversion key ratios do not have a common denominator')
            else:
                denom = inversion_keys.common_denominator
                numer = inversion_keys.numerators
                return numer, denom, inversion_keys

    raise ValueError('Unable to deduce the inversion keys')

def ds_grid(standard, spike1, spike2=None, inversion_keys=None, n=19, *,
            fnat=0, fins=2, fixed_voltage=10, fixed_key = None,
            blank = None, blank_fixed_voltage = None, blank_fixed_key = None,
            integrations=100, integration_time=8.389, resistor=1E11,
            random_seed=46, method='siebert',
            isotope_masses=None, isotope_abundances=None,
            correction_method=ds_correction,
            **kwargs):
    """
    Compute the inversion result for a simulated measurement with varied sample/spike
    and spike1/spike2 ratios.

    Parameters
    ----------
    standard
        Any object that can be passed to ``make_ms_array`` to returns valid array.
        Also accepts a tuple or a dict which will be unpacked appropriately.
    spike1
        The composition of spike 1. If spike 2 is not given then this must contain both spikes.
    spike2
        The composition of spike 2. Not necessary if *spike1* contains both spikes. In this case
        spike1/spike2 ratio will be fixed.
    inversion_keys
        Keys used for the inversion. Can either be 3 ratio key strings or 4 isotope key strings.
        Does not have to be given if the inversion keys can be inferred from *spike1*.
    n
        The number of intervals in the grid. The total number of data points will be :math:`n^2`.
    fnat
        If given, the natural fractionation fractionation factor is applied to the ms_array
        before *interferences* are added to the ms_array.
    fins
        If given, the instrumental mass fractionation factor is applied to the ms_array
        at the same time the *interferences* are added to the ms_array.
    fixed_voltage
        The voltage of the most abundant isotope in the array. The value for all other isotopes in
        the array are adjusted accordingly.
    fixed_key
        If not given then the sum of the inversion keys will be set to *fixed_voltage*.
    blank
        The blank sample to be added to the sample. Can be object that can be singularly passed
        to ``make_ms_array`` which returns valid array. Also accepts a tuple or a dict which will
        be unpacked appropriately.
    blank_fixed_voltage
        The voltage of the *blank_fixed_key* in returned sample that is *blank*.
    blank_fixed_key
        If not given then the sum of the inversion keys will be set to *blank_fixed_voltage*.
    integrations
        The number of simulated measurements.
    integration_time
        The integration time for each simulated measurement.
    resistor
        The resistor used for each measurement. A isotope array or a dictionary can be passed to
        to give different resistor values for different isotopes.
    random_seed
        Seed given for the random generator. The same seed will be used for all data points
        resulting in the same normal distribution for each set of integrations. If ``None`` then
        each point in the grid will have a different normal distribution.
    method
        Method used for the doublespike inversion. Default is the  ``"siebert"`` method as it is
        faster however it occasionally fails to find solutions to extreme edge cases. If these
        are important use the ``"rudge"`` method instead.
    correction_method
        The method used to perform the double spike inversion. Must have the same signature as
        ``ds_correction``. Defaults to ``ds_correction``.
    isotope_abundances
        Reference value for the isotope fractions of different elements.
        Defaults to ``isopy.refval.isotope.abundance``.
    isotope_masses
        Reference value for the isotope masses of different elements.
        Defaults to ``isopy.refval.isotope.mass``.
    kwargs
        Prefix interferences with ``interference_`` and method kwargs with ``method_``.


    Returns
    -------
    grid_result : DSGridResult
        The returned *DSGridResult* contains the following attributes:

        * ``solutions.method`` - Grid containing the method used to do the inversion for each datapoint.

        * ``solutions.alpha`` - Grid containing the natural fractionation factor as defined by Rudge (:math:`n = N * m^\\alpha`) for each datapoint.
          **Note** this value has the opposite sign to *fnat*.

        * ``solutions.beta`` - Grid containing the instrumental mass fractionation factor as defined by Rudge (:math:`m = M * m^\\beta`) for each datapoint.
          Same as *fins*.

        * ``solutions.lambda_`` - Grid containing the lambda value defined by Rudge for each datapoint.

        * ``solutions.fnat`` - Grid containing the natural fractionation factor as defined by Siebert (:math:`\\textrm{SA} = \\textrm{ST} * m^\\alpha`) for each datapoint.
          **Note** this value has the opposite sign to *alpha*.

        * ``solutions.fins`` - Grid containing the  instrumental fractionation factor as defined by Siebert (:math:`\\textrm{MT} = \\textrm{MS} * m^\\alpha`)  for each datapoint.
          Same as *beta*.

        * ``solutions.spike_fraction`` - Grid containing the  fraction of spike in the sample-spike mixture for each datapoint. Calculated on the
          basis of the four isotopes used in the inversion.

        * ``solutions.sample_fraction`` - Grid containing the  fraction of sample in the sample-spike mixture for each datapoint. Calculated on the
          basis of the four isotopes used in the inversion.

        * ``solutions.Q`` - Grid containing the  *sample_fraction* to *spike_fraction* ratio for each datapoint.

        * ``input.doublespike_fraction`` - List of the true double spike fraction (:math:`\\frac{\\textrm{double spike}} {\\textrm{double spike} + \\textrm{sample}}`) for each datapoint in the x-axis.

        * ``input.sample_fraction`` - List of the true sample fraction (:math:`\\frac{\\textrm{sample}} {\\textrm{double spike} + \\textrm{sample}}`) for each datapoint in the x-axis.

        * ``input.spike1_fraction`` - List of the true spike 1 fraction (:math:`\\frac{\\textrm{spike 1}} {\\textrm{spike 1} + \\textrm{spike 2}}`) for each datapoint in the y-axis.

        * ``input.spike2_fraction`` - List of the true spike 2 fraction (:math:`\\frac{\\textrm{spike 2}} {\\textrm{spike 1} + \\textrm{spike 2}}`) for each datapoint in the y-axis.

        * ``input.fnat`` - The true natural fractionation value for each datapoint.

        * ``input.fins`` - The true instrumental fractionation value for each datapoint.

        * ``input.measured`` Grid containing the *measured* values for each datapoint.

        The returned *DSGridResult* contains the following methods:

        * ``xyz(zeval=isopy.sd, zattr='solutions.fnat')`` - Returns ``input.doublespike_fraction``. ``input.spike1_fraction``, ``zeval(getattr(grid_result, zattr))``.

        * ``yz(xval=0.5, zeval=isopy.sd, zattr='solutions.fnat')`` - Returns ``input.spike1_fraction`` and a 1-dimensional array of ``zeval(getattr(grid_result, zattr))`` at *xval*.

        * ``xz(yval=0.5, zeval=isopy.sd, zattr='solutions.fnat')`` - Returns ``input.doublespike_fraction`` and a 1-dimensional array of ``zeval(getattr(grid_result, zattr))`` at *yval*.
    """
    isotope_abundances = isopy.checks.check_reference_value('isotope_abundances', isotope_abundances,
                                                            isopy.refval.isotope.fraction)
    isotope_masses = isopy.checks.check_reference_value('isotope_masses', isotope_masses,
                                                        isopy.refval.isotope.mass)

    numer, denom, inversion_keys = _deduce_inversion_keys(spike1, inversion_keys)

    standard = isotope.make_ms_array(standard, mf_factor=fnat, isotope_fractions=isotope_abundances,
                                     isotope_masses=isotope_masses)

    interference_kw = core.extract_kwargs(kwargs, 'interference')
    method_kw = core.extract_kwargs(kwargs, 'method')

    if fixed_key is None:
        fixed_key = inversion_keys.flatten(ignore_duplicates=True)
    elif blank_fixed_key is None:
        blank_fixed_key = inversion_keys.flatten(ignore_duplicates=True)

    doublespike_fractions = np.linspace(0, 1, n + 2)[1:-1]
    spike1 = spike1 / np.sum(spike1, axis=None)

    if spike2 is None:
        spike2 = 0
        spike1_fractions = np.array([1])
    else:
        spike1_fractions = np.linspace(0, 1, n + 2)[1:-1]
        spike2 = spike2 / np.sum(spike2, axis=None)

    result_solutions = IsopyResultList()
    all_measured = IsopyResultList()
    for spike1_fraction in spike1_fractions:
        spike_mixture = (spike1 * spike1_fraction) + (spike2 * (1 - spike1_fraction))
        result_solutions.append(IsopyResultList())
        all_measured.append(IsopyResultList())
        for doublespike_fraction in doublespike_fractions:
            measured = isotope.make_ms_sample(standard, spike=spike_mixture,
                                              spike_fraction=doublespike_fraction,
                                              fnat=None, fins=fins,
                                              fixed_voltage=fixed_voltage, fixed_key=fixed_key,
                                              blank=blank, blank_fixed_voltage=blank_fixed_voltage,
                                              blank_fixed_key=blank_fixed_key,
                                              integrations=integrations,
                                              integration_time=integration_time,
                                              resistors=resistor,
                                              random_seed=random_seed,
                                              isotope_fractions=isotope_abundances,
                                              isotope_masses=isotope_masses, **interference_kw)
            all_measured[-1].append(measured)
            try:
                solution = correction_method(measured, spike_mixture, standard, inversion_keys,
                                             isotope_masses=isotope_masses,
                                             isotope_abundances=isotope_abundances,
                                             method=method, **method_kw)
                result_solutions[-1].append(solution)
            except:
                nan = np.full(measured.size, np.nan)
                solution = DSResult(dict(method='rudge'), dict(alpha=nan, beta=nan, lambda_=nan,
                                     fnat=nan, fins=nan, spike_fraction=nan, sample_fraction=nan,
                                     Q=nan))
                result_solutions[-1].append(solution)

    input = IsopyNamedResult(static_items=dict(doublespike_fraction = doublespike_fractions,
                                               sample_fraction = 1-doublespike_fractions,
                                               spike1_fraction = spike1_fractions,
                                               spike2_fraction = 1-spike1_fractions,
                                               fnat = fnat,
                                               fins = fins),
                             dynamic_items=dict(measured=all_measured))
    return DSGridResult(dynamic_items=dict(input=input, solutions=result_solutions))

@core.append_preset_docstring
@core.add_preset(('delta', 'permil'), factor=1000)
@core.add_preset('mu', factor=1E6)
def ds_Delta(mass_ratio, fnat, reference_fnat=0, *, factor=1, isotope_masses=None):
    """
    Calculate the Δ value for *mass_ratio* of a sample using the *fnat* mass fractionation factor.

    .. math::

        \\Delta \\frac{smp_{i}} {smp_{j}} = \\left( \\left(\\frac{mass_i} {mass_j} \\right)^{fnat} - 1\\right) * \\textrm{factor}

    Where :math:`norm` is the normalisation factor. If *reference_fnat* values are given :math:`fnat` is the difference
    between the *fnat* and *reference_fnat*:

    .. math::

        fnat = fnat_{smp} - fnat_{ref}

    If multiple *reference_fnat* values are passed the mean of those values is used. If the each
    *reference_fnat* passed has more than one value the mean is used.
    """
    if type(fnat) is DSResult:
        fnat = fnat.fnat

    reference_fnat = _combine(reference_fnat)

    if isotope_masses is None:
        isotope_masses = isopy.refval.isotope.mass

    fnat = fnat - reference_fnat

    if isinstance(mass_ratio, str):
        mass_ratio = isotope_masses.get(mass_ratio)

    return (np.power(mass_ratio, fnat) - 1) * factor

@core.append_preset_docstring
@core.add_preset(('delta', 'permil'), factor=1000)
@core.add_preset('mu', factor=1E6)
def ds_Delta_prime(mass_ratio, fnat, reference_fnat=0, *, factor=1, isotope_masses=None):
    """
    Calculate the Δ' value for *mass_ratio* of a sample using the *fnat* mass fractionation factor.

    .. math::

         \\Delta^{\\prime}  \\frac{smp_{i}} {smp_{j}} = \\textrm{norm} * fnat * log\\left(\\frac{mass_i} {mass_j} \\right)

    Where *norm* is the normalisation factor and *fnat* is the difference
    between the sample and optional standard:

    .. math::

        fnat = fnat_{smp} - fnat_{std}

    """
    if type(fnat) is DSResult:
        fnat = fnat.fnat

    reference_fnat = _combine(reference_fnat)

    if isotope_masses is None:
        isotope_masses = isopy.refval.isotope.mass

    fnat = fnat - reference_fnat

    if isinstance(mass_ratio, str):
        mass_ratio = isotope_masses.get(mass_ratio)

    return np.log(mass_ratio) * fnat * factor

def _combine(value):
    if type(value) is not tuple:
        values = (value,)
    else:
        values = value

    out = []
    for value in values:
        if type(value) is DSResult:
            value = value.fnat
        value = np.asarray(value)
        if value.size > 1:
            value = np.nanmean(value)
        out.append(value)

    if len(out) == 1:
        return out[0]
    else:
        return np.mean(out)


class IsopyNamedResult:
    def __init__(self, static_items = None, dynamic_items=None):
        self.__static_item_names = []
        self.__dynamic_item_names = []

        if static_items:
            for name, value in static_items.items():
                setattr(self, name, value)
                self.__static_item_names.append(name)

        if dynamic_items:
            for name, value in dynamic_items.items():
                setattr(self, name, value)
                self.__dynamic_item_names.append(name)

    def __repr__(self):
        sitems = [f'{name}={getattr(self, name).__repr__()}' for name in self.__static_item_names]
        ditems = [f'{name}={getattr(self, name).__repr__()}' for name in self.__dynamic_item_names]
        items = "\n".join(sitems + ditems)
        return f'{type(self).__name__}({items})'

    def __array_function__(self, func, types, args, kwargs):
        static_items = {name: getattr(self, name) for name in self.__static_item_names}
        dynamic_items = {name: func(getattr(self, name), *args[1:], **kwargs) for name in self.__dynamic_item_names}
        return self.__class__(static_items, dynamic_items)

    def __getitem__(self, name):
        return getattr(self, name)

    def items(self):
        return ((name, getattr(self, name)) for name in self.__dynamic_item_names)

    def values(self):
        return (getattr(self, name) for name in self.__dynamic_item_names)

    def keys(self):
        return (name for name in self.__dynamic_item_names)


class IsopyResultList(list):
    def __array_function__(self, func, types, args, kwargs):
        return self.__class__(func(item, *args[1:], **kwargs) for item in self)

    def __getattr__(self, attr):
        if attr[:1] == '_': raise AttributeError() #Avoid issues with numpy special attributes
        return self.__class__(getattr(item, attr) for item in self)


class DSResult(IsopyNamedResult):
    method: str
    pass


class DSGridResult(IsopyNamedResult):
    def __getattr__(self, attr):
        attrs = attr.split('.')

        if len(attrs) == 1:
            lastattr = self.solutions
        else:
            lastattr = self

        for a in attrs:
            lastattr = getattr(lastattr, a)

        return lastattr

    def xyz(self, zeval=isopy.sd, zattr='solutions.fnat'):
        """
        Return a tuple of ``grid.input.doublespike_fraction``, ``grid.input.spike1_fraction`` and ``zeval(getattr(grid, zattr))``.

        This which can be used in conjunction with the :func:`plot_grid` function e.g. ``isopy.tb.plot_grid(plt, *grid.xyz())``.

        Parameters
        ----------
        zeval
            Function to evaluate *zattr*.
        zattr
            The attribute to be evaluated.
        """
        zval = getattr(self, zattr)
        zval = zeval(zval)
        return self.input.doublespike_fraction, self.input.spike1_fraction, zval

    def yz(self, xval=0.5, zeval=isopy.sd, zattr='solutions.fnat'):
        """
        Return ``grid.input.spike1_fraction`` and a 1-dimensional arrays of z at *xval*.

        z is evaluated as for :func:`xyz`.

        Parameters
        ----------
        xval
            If there is no exact match the *x* value closest to *xval* is used.
        zeval
            Function to evaluate *zattr*.
        zattr
            The attribute to be evaluated
        """
        x, y, z = self.xyz(zeval, zattr)
        ix = np.nanargmin(np.abs(x-xval))
        return y, [r[ix] for r in z]

    def xz(self, yval=0.5, zeval=isopy.sd, zattr='solutions.fnat'):
        """
        Return ``grid.input.doublespike_fraction`` and a 1-dimensional arrays of z at *yval*.

        z is evaluated as for :func:`xyz`.

        Parameters
        ----------
        yval
            If there is no exact match the *y* value closest to *yval* is used.
        zeval
            Function to evaluate *zattr*.
        zattr
            The attribute to be evaluated
        """
        x, y, z = self.xyz(zeval, zattr)
        iy = np.nanargmin(np.abs(y-yval))
        return x, z[iy]

